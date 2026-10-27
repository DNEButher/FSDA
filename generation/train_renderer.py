"""train_renderer.py — train the day->night conditional renderer (paper Sec. 3).

SR3-style conditional UNet trained on DarkDriving real day/night pairs
(direction flipped: condition = day, target = night). Loss = standard SR3
paired loss  max(MSE(x0_pred, GT), MSE(eps_pred, noise))  with continuous
noise-level sampling. The architecture / schedule below must stay identical
to the one in diffusion_parent.py so that the resulting EMA checkpoint loads
into diffusion_parent.load_renderer() with zero changes.

Reference run (paper): DarkDriving 9,530 pairs, 150k iters, batch 16,
random 128x128 crops, Adam 1e-4, EMA decay 0.9999 (starts at 5k).

Usage (single GPU):
    python generation/train_renderer.py --dd-dir /data/DarkDriving_720p \
        --iters 150000 --out outputs/renderer
Outputs (every --save-every iters and at the final step):
    out/Gen_I{step}.pth   raw weights
    out/Ema_I{step}.pth   EMA shadow   <- use this one for generation
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from _sr3 import networks
from d2n_dataset import DayToNightDataset

# identical to diffusion_parent.BASE_MODEL (keep the two in sync)
BASE_MODEL = {
    "which_model_G": "ddpm",
    "finetune_norm": False,
    "unet": {
        "in_channel": 6, "out_channel": 3, "inner_channel": 64,
        "channel_multiplier": [1, 1, 2, 2, 4],
        "attn_res": [16], "res_blocks": 2, "dropout": 0
    },
    "beta_schedule": {
        "train": {"schedule": "linear", "n_timestep": 8,
                  "linear_start": 0.0001, "linear_end": 0.02, "time_scale": 64},
        "val": {"schedule": "linear", "n_timestep": 8,
                "linear_start": 0.0001, "linear_end": 0.02, "time_scale": 64}
    },
    "diffusion": {
        "image_size": 128, "channels": 6, "conditional": True,
        "w_gt": 0.1, "w_snr": 0.5, "w_str": 0.1, "w_lpips": 0.0
    },
}


class EMA:
    def __init__(self, net, decay=0.9999, start=5000):
        self.decay, self.start, self.step = decay, start, 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        self.step += 1
        if self.step < self.start:
            return
        d = self.decay
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def sr3_paired_loss(gd, LQ, GT):
    """Standard SR3 paired loss: max(L_x0, L_eps), continuous noise level."""
    b = GT.shape[0]
    t = np.random.randint(1, gd.num_timesteps + 1)
    ts = gd.time_scale
    c = torch.FloatTensor(
        np.random.uniform(gd.sqrt_alphas_cumprod_prev[(t - 1) * ts],
                          gd.sqrt_alphas_cumprod_prev[t * ts], size=b)
    ).to(GT.device).view(b, -1)

    noise = torch.randn_like(GT)
    z_t = gd.q_sample(GT, c.view(-1, 1, 1, 1), noise)

    eps_pred, _ = gd.denoise_fn(torch.cat([LQ, z_t], dim=1), c)
    x0_pred = gd.predict_start(z_t, c.view(-1, 1, 1, 1), eps_pred)

    mse_x0 = F.mse_loss(x0_pred, GT, reduction='none')
    mse_eps = F.mse_loss(eps_pred, noise, reduction='none')
    l_x0 = mse_x0.reshape(b, -1).mean(-1)
    l_eps = mse_eps.reshape(b, -1).mean(-1)
    return torch.mean(torch.maximum(l_x0, l_eps)), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dd-dir', required=True, help='DarkDriving root (day/ night/)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--iters', type=int, default=150000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--ema-decay', type=float, default=0.9999)
    ap.add_argument('--ema-start', type=int, default=5000)
    ap.add_argument('--save-every', type=int, default=10000)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    opt = {"model": BASE_MODEL, "phase": "train",
           "gpu_ids": [0], "distributed": False}
    gd = networks.define_G(opt, student=False)
    gd = gd.netG if hasattr(gd, 'netG') else gd
    if hasattr(gd, 'module'):
        gd = gd.module
    gd = gd.cuda().train()
    print(f'[model] UNet+Diffusion n_timestep={gd.num_timesteps} '
          f'time_scale={gd.time_scale}', flush=True)

    ds = DayToNightDataset(args.dd_dir, crop=args.crop)
    print(f'[data] d2n pairs = {len(ds)}', flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)

    optG = torch.optim.Adam(gd.parameters(), lr=args.lr)
    ema = EMA(gd, decay=args.ema_decay, start=args.ema_start)

    print(f'[train] iters={args.iters} batch={args.batch} crop={args.crop} '
          f'lr={args.lr}', flush=True)
    step, t_clk, loss_acc = 0, time.time(), 0.0
    done = False
    while not done:
        for batch in dl:
            LQ = batch['LQ'].cuda(non_blocking=True)    # = day
            GT = batch['GT'].cuda(non_blocking=True)    # = night
            loss, t = sr3_paired_loss(gd, LQ, GT)
            optG.zero_grad(set_to_none=True)
            loss.backward()
            optG.step()
            ema.update(gd)

            step += 1
            loss_acc += loss.item()
            if step % args.log_every == 0:
                ips = step / (time.time() - t_clk)
                print(f'[{step}/{args.iters}] loss={loss_acc / args.log_every:.4f} '
                      f'(t={t}) {ips:.2f} it/s', flush=True)
                loss_acc = 0.0

            if step % args.save_every == 0 or step >= args.iters:
                gen_p = out_dir / f'Gen_I{step}.pth'
                torch.save(gd.state_dict(), gen_p)
                ema_p = out_dir / f'Ema_I{step}.pth'
                torch.save(ema.state_dict(), ema_p)
                print(f'[save] {gen_p.name} + {ema_p.name}', flush=True)

            if step >= args.iters:
                done = True
                break

    print(f'[done] {step} iters -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()