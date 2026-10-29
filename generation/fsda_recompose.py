"""fsda_recompose.py — FSDA frequency recomposition (paper Eq. 4-7).

Log-luminance band swap between the two parents, no network:

    Y_x  = 0.2126 R + 0.7152 G + 0.0722 B            (Rec.709 luma)
    L_x  = log(Y_x + eps),   H_x = L_x - blur_sigma(L_x)
    forward (FSDA):  RGB_mix = RGB_diff * exp( w(x) * (H_dir - H_diff) )

    w(x) = 1 everywhere            -> FSDA-uniform (ablation arm)
    w(x) = 1 - smoothstep(LP_diff; tau1, tau2)
                                   -> FSDA adaptive form (paper default):
                                     dark regions swap fully, glowing
                                     light sources keep the Diffusion
                                     Parent's low band
    reverse arm:     swap the roles of the two parents.

Low frequencies (illumination / bloom / tone) come from the Diffusion
Parent; high frequencies (edges / texture) come from the Direct Parent.
tau1/tau2 are frozen quantiles of the pooled LP_diff statistics (paper
Sec. 3) — do not tune them per run.

After mixing, the script assembles a YOLO training set:

    <out_dir>/images/train/    day .jpg originals + mixed .png (same stems)
    <out_dir>/labels/train/    one YOLO .txt per stem (labels are shared by
                               both views — geometry is identical by
                               construction)
    <out_dir>/fsda_train.yaml  ultralytics data file

Idempotent: existing files are skipped, never overwritten.

Usage:
    python generation/fsda_recompose.py --config configs/fsda_adaptive.yaml \
        [--workers 12] [--mix-only | --assemble-only]
"""
import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

EPS = 1e-3
REPO = Path(__file__).resolve().parents[1]

# set by _init() in every worker process (Windows uses spawn, not fork)
_FP = _FS = _OUT = None
_SIGMA = 8
_DIRECTION = 'forward'
_WEIGHT = 'uniform'
_TAU1 = _TAU2 = None


def blur(l, sigma):
    try:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(l, sigma, mode='reflect')
    except ImportError:
        r = int(3 * sigma) + 1
        x = np.arange(-r, r + 1, dtype='float64')
        k = np.exp(-x * x / (2 * sigma * sigma))
        k /= k.sum()
        p = np.pad(l, r, mode='reflect')
        rows = np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 0, p)
        return np.apply_along_axis(lambda v: np.convolve(v, k, 'valid'), 1, rows)


def luma(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def adaptive_w(lp_lo, tau1, tau2):
    """Swap weight from the low band of the Diffusion Parent (Eq. 7).

    w = 1 (full swap) in dark regions -> smoothstep -> w = 0 (keep the
    Diffusion Parent) around bright / glowing light sources.
    """
    t = np.clip((lp_lo - tau1) / (tau2 - tau1), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def freqmix(rgb_hi, rgb_lo, sigma, w_map=None):
    """Blend the high band of `rgb_hi` into `rgb_lo`, weighted by `w_map`.

    forward arm: hi = Direct Parent, lo = Diffusion Parent.
    w_map = None keeps the original uniform-swap arithmetic bit-exact
    (gate-verified against the frozen campaign renders); a 2-D w_map
    follows the adaptive form  RGB = RGB_lo * exp(w * (H_hi - H_lo)).
    """
    y_hi, y_lo = luma(rgb_hi), luma(rgb_lo)
    l_hi, l_lo = np.log(y_hi + EPS), np.log(y_lo + EPS)
    if w_map is None:
        l_mix = blur(l_lo, sigma) + l_hi - blur(l_hi, sigma)
        out = rgb_lo * (np.exp(l_mix) / (y_lo + EPS))[..., None]
    else:
        d = (l_hi - blur(l_hi, sigma)) - (l_lo - blur(l_lo, sigma))
        out = rgb_lo * np.exp(w_map * d)[..., None]
    return np.clip(out, 0.0, 1.0)


def _init(fp, fs, out, sigma, direction, weight, tau1, tau2):
    global _FP, _FS, _OUT, _SIGMA, _DIRECTION, _WEIGHT, _TAU1, _TAU2
    _FP, _FS, _OUT, _SIGMA, _DIRECTION = fp, fs, out, sigma, direction
    _WEIGHT, _TAU1, _TAU2 = weight, tau1, tau2


def work(stem):
    d = _OUT / f'{stem}.png'
    if d.exists():
        return 1
    fp = np.asarray(Image.open(_FP / f'{stem}.png').convert('RGB')
                    ).astype('float32') / 255.0
    fs = np.asarray(Image.open(_FS / f'{stem}.png').convert('RGB')
                    ).astype('float32') / 255.0
    if _DIRECTION == 'forward':
        rgb_hi, rgb_lo = fp, fs          # low from fs, high from fp
    else:
        rgb_hi, rgb_lo = fs, fp          # reverse arm
    if _WEIGHT == 'adaptive':
        assert _DIRECTION == 'forward', 'adaptive weight is forward-only'
        lp_lo = blur(np.log(luma(rgb_lo) + EPS), _SIGMA)
        w = adaptive_w(lp_lo, _TAU1, _TAU2)
        out = freqmix(rgb_hi, rgb_lo, _SIGMA, w)
    else:
        out = freqmix(rgb_hi, rgb_lo, _SIGMA)            # uniform w == 1
    Image.fromarray((out * 255).astype('uint8')).save(d)
    return 0


def _link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def assemble(cfg, out_dir, split_path):
    day_dir = Path(cfg['day_dir'])
    labels_dir = Path(cfg['labels_dir'])
    stems = split_path.read_text().split()

    img_tr = out_dir / 'images' / 'train'
    lbl_tr = out_dir / 'labels' / 'train'
    img_tr.mkdir(parents=True, exist_ok=True)
    lbl_tr.mkdir(parents=True, exist_ok=True)

    for stem in stems:
        src = next((day_dir / f'{stem}{e}' for e in ('.jpg', '.jpeg', '.png')
                    if (day_dir / f'{stem}{e}').exists()), None)
        assert src is not None, f'day image missing: {stem}'
        _link_or_copy(src, img_tr / src.name)

        lb = labels_dir / f'{stem}.txt'
        assert lb.exists(), f'label missing: {lb}'
        _link_or_copy(lb, lbl_tr / lb.name)

        # mixed renders are written straight into img_tr by the mix step
        assert (img_tr / f'{stem}.png').exists(), \
            f'frequency-mixed render missing: {stem} (run the mix step first)'

    for c in (out_dir / 'labels').rglob('*.cache'):
        c.unlink()

    yml = out_dir / 'fsda_train.yaml'
    yml.write_text(
        f'path: {out_dir.resolve()}\n'
        'train: images/train\n'
        'val: images/train\n'
        '\n'
        'nc: 5\n'
        'names: [person, bicycle, car, bus, truck]\n')

    n_img = len(list(img_tr.iterdir()))
    n_lbl = len(list(lbl_tr.iterdir()))
    assert n_img == 2 * len(stems) and n_lbl == len(stems), (n_img, n_lbl)
    print(f'[assemble] imgs={n_img} labels={n_lbl} -> {yml}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/fsda_sigma8.yaml')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--mix-only', action='store_true')
    ap.add_argument('--assemble-only', action='store_true')
    ap.add_argument('--split', default=str(REPO / 'splits' / 'bdd_day_train.txt'))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    fp_dir, fs_dir = Path(cfg['fp_dir']), Path(cfg['fs_dir'])
    out_dir = Path(cfg['out_dir'])
    sigma = int(cfg.get('sigma', 8))
    direction = cfg.get('direction', 'forward')
    weight = cfg.get('weight', 'uniform')
    assert direction in ('forward', 'reverse'), direction
    assert weight in ('uniform', 'adaptive'), weight
    tau1 = float(cfg.get('tau1', -1.40))
    tau2 = float(cfg.get('tau2', -0.60))
    if weight == 'adaptive':
        assert direction == 'forward', 'adaptive weight is forward-only'
    img_tr = out_dir / 'images' / 'train'
    img_tr.mkdir(parents=True, exist_ok=True)

    if not args.assemble_only:
        stems = sorted(p.stem for p in fs_dir.glob('*.png')
                       if (fp_dir / f'{p.stem}.png').exists())
        print(f'[fx-gen] direction={direction} weight={weight} '
              f'sigma={sigma} stems={len(stems)}', flush=True)
        with Pool(args.workers, initializer=_init,
                  initargs=(fp_dir, fs_dir, img_tr, sigma, direction,
                            weight, tau1, tau2)) as pool:
            for i, _ in enumerate(pool.imap_unordered(work, stems, chunksize=32)):
                if (i + 1) % 1000 == 0:
                    print(f'  [{i + 1}/{len(stems)}]', flush=True)
        n = len(list(img_tr.glob('*.png')))
        assert n >= len(stems), f'mixed count mismatch: {n}/{len(stems)}'

    if not args.mix_only:
        assemble(cfg, out_dir, Path(args.split))
    print('[fx] DONE', flush=True)


if __name__ == '__main__':
    main()