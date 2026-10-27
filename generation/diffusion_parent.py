"""diffusion_parent.py — Diffusion Parent generation (paper Eq. 1-3).

Identical photometric chain as direct_parent.py (same seed -> same gamma /
noise / flare per image), then the degraded image is noised to a low noise
level t0 = 4 (SDEdit warm start, Eq. 3) and denoised by 4 deterministic
DDIM steps conditioned on the original daytime image:

    x_t0 = sqrt(abar_t0) * I_init + sqrt(1-abar_t0) * eps
    I_diff = DDIM(x_t0 | I_day),  t = 4 -> 1, no random term

The renderer is an SR3-style conditional UNet trained on DarkDriving
day/night pairs (generation/train_renderer.py). Architecture definition is
vendored in generation/_sr3 (subset of the ReDDiT codebase, see
generation/_sr3/__init__.py for attribution).

Usage:
    python generation/diffusion_parent.py --config configs/fsda_sigma8.yaml \
        [--skip N] [--limit N] [--shard i --nshards j]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from direct_parent import (load_stats, resize16, list_images, make_rng,
                           photometric_chain, FlareBank)

# architecture / noise schedule actually used by the paper (identical to the
# 8-step official SR3 skeleton: n_timestep=8, time_scale=64 -> 512 levels)
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
        # loss weights (required keys for define_G; unused at inference).
        # values identical to generation/train_renderer.py training config
        "w_gt": 0.1, "w_snr": 0.5, "w_str": 0.1, "w_lpips": 0.0
    },
}


def load_renderer(ema_ckpt, device='cuda'):
    """Build the UNet + GaussianDiffusion and load EMA weights."""
    from _sr3 import networks
    # phase 'test' skips the orthogonal re-init (weights are loaded below)
    opt = {"model": BASE_MODEL, "phase": "test",
           "gpu_ids": [0], "distributed": False}
    gd = networks.define_G(opt, student=False)
    gd = gd.netG if hasattr(gd, 'netG') else gd
    if hasattr(gd, 'module'):
        gd = gd.module
    gd.load_state_dict(torch.load(ema_ckpt, map_location='cpu'))
    return gd.to(device).eval()


def to_t(x01):
    return torch.from_numpy(
        (x01.transpose(2, 0, 1) * 2 - 1).astype(np.float32)
    ).unsqueeze(0).cuda()


def sa_ddim(gd, x_lq, x_init, t0, stride=1):
    """SDEdit warm start + deterministic DDIM (core of Eq. 3)."""
    device = x_lq.device
    B = x_lq.shape[0]
    ts = gd.time_scale

    with torch.no_grad():
        c = gd.sqrt_alphas_cumprod[t0 * ts]
        s1 = gd.sqrt_one_minus_alphas_cumprod[t0 * ts]
        x_t = c * x_init + s1 * torch.randn(x_init.shape, device=device)

    for t_step in reversed(range(stride, t0 + 1, stride)):
        with torch.no_grad():
            t = x_t.new_ones([B], dtype=torch.long) * t_step
            s = x_t.new_ones([B], dtype=torch.long) * (t_step - stride)
            noise_level = torch.FloatTensor(
                [gd.sqrt_alphas_cumprod_prev[t_step * ts]]
            ).repeat(B, 1).to(device)
            eps = gd.denoise_fn(torch.cat([x_lq, x_t], dim=1), noise_level)[0]
            x_0 = gd.predict_start_from_noise(x_t, t * ts, eps)
            x_0 = torch.clamp(x_0, -1., 1.)
            x_t = gd.sqrt_alphas_cumprod[s * ts] * x_0 + \
                  gd.sqrt_one_minus_alphas_cumprod[s * ts] * eps
    return torch.clamp(x_t, -1, 1)


def ddim_render(gd, day01, init01, t0):
    out = sa_ddim(gd, to_t(day01), to_t(init01), t0, stride=1)
    return ((out.squeeze(0).clamp(-1, 1) + 1) / 2 * 255
            ).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fsda_sigma8.yaml')
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg['fs_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    # seed torch as well: the warm-start eps draw is the only torch RNG user
    torch.manual_seed(cfg['seed'] + args.shard)

    qs, gq, rg, bg, sa_, sb = load_stats(cfg['stats_json'])
    bank = FlareBank(cfg['flare_dirs']) if cfg.get('flare_prob', 1.0) > 0 else None
    frng = make_rng(cfg['seed'], args.shard)
    gd = load_renderer(cfg['renderer_ckpt'])

    mine = list_images(cfg['day_dir'], args.skip, args.limit,
                       args.shard, args.nshards)
    print(f'[diffusion] shard {args.shard}/{args.nshards}: {len(mine)} imgs',
          flush=True)

    import time
    clk = time.time()
    for i, p in enumerate(mine):
        dst = out_dir / f'{p.stem}.png'
        if dst.exists():
            continue
        pil = resize16(Image.open(p).convert('RGB'))
        day01 = np.asarray(pil, dtype=np.float32) / 255.0
        init01 = photometric_chain(
            day01, qs, gq, rg, bg, sa_, sb, bank, frng,
            gamma_mode=cfg.get('gamma_mode', 'stats'),
            gamma_fixed=cfg.get('gamma_fixed', 2.36),
            flare_prob=cfg.get('flare_prob', 1.0))
        arr = ddim_render(gd, day01, init01, cfg['t0'])
        Image.fromarray(arr, 'RGB').save(dst)
        if (i + 1) % 25 == 0:
            fps = (i + 1) / (time.time() - clk)
            print(f'  [{i + 1}/{len(mine)}] {fps:.2f} img/s', flush=True)
    n = len(list(out_dir.glob('*.png')))
    print(f'[diffusion] DONE {n} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()