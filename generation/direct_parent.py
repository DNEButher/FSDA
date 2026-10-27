"""direct_parent.py — Direct Parent generation (paper Eq. 1-2).

Photometric degradation + lens-flare injection, NO diffusion:
    I_phys = N_{sigma(L)}( clip(I_day)^gamma * g )          (Eq. 1)
    I_dir  = clip( I_phys + d' * T(F) ),  d' ~ U(0.1, 2.0)  (Eq. 2)

gamma is sampled from the quantile-calibrated distribution in
configs/degradation_stats.json; g and sigma(L) = a + b*L are taken from the
same file. The noise level L uses BT.601 luma (calibration convention).

Run with the same seed as diffusion_parent.py: both parents share the
identical gamma / noise / flare draws per image, so the only difference
between the two parents is the diffusion pass (paired construction).

Usage:
    python generation/direct_parent.py --config configs/fsda_sigma8.yaml \
        [--skip N] [--limit N] [--shard i --nshards j]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# BT.601 luma — the noise sigma(L) was calibrated on this convention
_LUM = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ------------------------------------------------------------------
# degradation constants (Eq. 1)
# ------------------------------------------------------------------
def load_stats(p):
    st = json.loads(Path(p).read_text())
    qs = np.array(st['gamma']['quantiles'])
    gq = np.array(st['gamma']['gamma_q'])
    rg, bg = st['color_gain']['r_over_g'], st['color_gain']['b_over_g']
    nz = st['noise_night']
    return qs, gq, rg, bg, nz['a'], nz['b']


def darken(gt01, gamma, rg, bg):
    """clip(I)^gamma * g, applied per channel (R and B relative to G)."""
    lq = np.power(np.clip(gt01, 0, 1), max(gamma, 0.5), dtype=np.float32)
    lq[..., 0] *= rg
    lq[..., 2] *= bg
    return lq


def noise_level_dependent(lq, sig_a, sig_b):
    """Additive i.i.d. Gaussian noise with sigma(L) = a + b*L, clipped to [0,1].

    np.random consumption order matters: this is the only np.random draw of
    each iteration, paired with diffusion_parent.py.
    """
    lum = lq @ _LUM
    sigma = sig_a + sig_b * lum
    return np.clip(
        lq + sigma[..., None] * np.random.randn(*lq.shape).astype(np.float32),
        0, 1)


def sample_gamma(qs, gq):
    """u ~ U(q0.05, q0.95), piecewise-linear inverse interpolation."""
    u = np.random.uniform(qs[0], qs[-1])
    return float(np.interp(u, qs, gq))


# ------------------------------------------------------------------
# flare bank (Eq. 2)
# ------------------------------------------------------------------
class FlareBank:
    """Flare7K++ library + random affine canvas (rotation / scale / bleed)."""

    def __init__(self, dirs):
        self.files = sorted(sum(
            [[str(p) for p in Path(d).glob('*.png')] for d in dirs], []))
        assert len(self.files) > 100, f'flare bank broken: {len(self.files)}'
        self.cache = {}

    def _load(self, fp):
        if fp not in self.cache:
            if len(self.cache) > 48:
                self.cache.clear()
            self.cache[fp] = Image.open(fp).convert('RGB')
        return self.cache[fp]

    def canvas(self, shape, frng):
        """T(F): per-channel black-base removal -> rotate U(0,360) ->
        scale U(0.5,1.0) rel. long side -> random position (bleed allowed) ->
        brightness d' ~ U(0.1, 2.0). Uses an isolated Generator `frng`."""
        H, W = shape
        fp = self.files[int(frng.integers(len(self.files)))]
        pil = self._load(fp)
        ang = float(frng.uniform(0, 360))
        pil = pil.rotate(ang, resample=Image.BILINEAR)
        s = float(frng.uniform(0.5, 1.0)) * max(H, W) / max(pil.size)
        nw = max(8, int(pil.width * s))
        nh = max(8, int(pil.height * s))
        pil = pil.resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (W, H), 0)
        cx = int(frng.integers(-nw // 3, max(W - 2 * nw // 3, 1)))
        cy = int(frng.integers(-nh // 3, max(H - 2 * nh // 3, 1)))
        canvas.paste(pil, (cx, cy))
        fl = np.asarray(canvas, dtype=np.float32) / 255.0
        mn = fl.reshape(-1, 3).min(0)
        mx = fl.reshape(-1, 3).max(0)
        fl = (fl - mn) / (mx - mn + 1e-6)
        dp = float(frng.uniform(0.1, 2.0))
        return fl * dp


# ------------------------------------------------------------------
# shared helpers
# ------------------------------------------------------------------
def resize16(pil):
    iw, ih = pil.size
    nw, nh = (iw // 16) * 16, (ih // 16) * 16
    if nw == 0 or nh == 0:
        nw, nh = iw, ih
    return pil.resize((nw, nh), Image.BICUBIC)


def list_images(day_dir, skip=0, limit=None, shard=0, nshards=1):
    paths = sorted(p for p in Path(day_dir).iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if skip:
        paths = paths[skip:]
    if limit:
        paths = paths[:limit]
    return paths[shard::nshards]


def make_rng(seed, shard):
    """RNG discipline (reproducibility): np.random seeded per shard and used
    only for gamma + noise; flare draws use an isolated Generator with an
    offset stream, so adding/removing flares never shifts gamma/noise."""
    np.random.seed(seed + shard)
    frng = np.random.default_rng(seed + 7919 * (shard + 1))
    return frng


def photometric_chain(day01, qs, gq, rg, bg, sa_, sb, bank, frng,
                      gamma_mode='stats', gamma_fixed=2.36, flare_prob=1.0):
    """Full Eq.(1)+Eq.(2) chain -> the warm-start / Direct-Parent image."""
    if gamma_mode == 'stats':
        gamma = sample_gamma(qs, gq)
    else:
        gamma = gamma_fixed
    dark = darken(day01, gamma, rg, bg)
    base = noise_level_dependent(dark, sa_, sb)

    flc = None
    if bank is not None and frng.uniform() < flare_prob:
        flc = bank.canvas(day01.shape[:2], frng)
    init01 = np.clip(base + flc, 0, 1) if flc is not None else base
    return init01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fsda_sigma8.yaml')
    ap.add_argument('--skip', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg['fp_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    qs, gq, rg, bg, sa_, sb = load_stats(cfg['stats_json'])
    bank = FlareBank(cfg['flare_dirs']) if cfg.get('flare_prob', 1.0) > 0 else None
    frng = make_rng(cfg['seed'], args.shard)

    mine = list_images(cfg['day_dir'], args.skip, args.limit,
                       args.shard, args.nshards)
    print(f'[direct] shard {args.shard}/{args.nshards}: {len(mine)} imgs',
          flush=True)

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
        Image.fromarray((init01 * 255).round().astype(np.uint8), 'RGB').save(dst)
        if (i + 1) % 200 == 0:
            print(f'  [{i + 1}/{len(mine)}]', flush=True)
    n = len(list(out_dir.glob('*.png')))
    print(f'[direct] DONE {n} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()