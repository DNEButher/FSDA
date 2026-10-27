"""d2n_dataset.py — paired day/night dataset for renderer training.

DarkDriving real pairs, direction flipped for the day->night task:
    LQ (condition) = day crop,  GT (target) = night crop (same coordinates,
    frame-aligned). The photometric degradation of direct_parent.py does NOT
participate in training: the target is real night appearance (light bloom,
street lamps, sensor noise) — degradations are only used on the sampling
side as the SDEdit warm start.

Output convention: {'LQ', 'GT'} float tensors in [-1, 1], same crop + same
flip for both views.

Usage (sanity check):
    python generation/d2n_dataset.py --dd-dir /data/DarkDriving_720p --limit 4
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


def _fit(a, s):
    """Small image -> center-crop to s x s (resize only in extreme cases)."""
    h, w = a.shape[:2]
    if h >= s and w >= s:
        y, x = (h - s) // 2, (w - s) // 2
        return a[y:y + s, x:x + s]
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    im = im.resize((s, s), Image.BICUBIC)
    return np.asarray(im, dtype=np.float32) / 255.0


class DayToNightDataset(data.Dataset):
    def __init__(self, dd_dir, crop=128, use_flip=True):
        """dd_dir: DarkDriving root containing day/ and night/ (same-stem jpg pairs)."""
        self.dd_dir = Path(dd_dir)
        self.stems = sorted(p.stem for p in (self.dd_dir / 'day').glob('*.jpg')
                            if (self.dd_dir / 'night' / f'{p.stem}.jpg').exists())
        self.crop = crop
        self.use_flip = use_flip

    def __len__(self):
        return len(self.stems)

    def _load_crop(self, idx):
        stem = self.stems[idx]
        day = np.asarray(
            Image.open(self.dd_dir / 'day' / f'{stem}.jpg').convert('RGB'),
            dtype=np.float32) / 255.0
        night = np.asarray(
            Image.open(self.dd_dir / 'night' / f'{stem}.jpg').convert('RGB'),
            dtype=np.float32) / 255.0

        h, w = day.shape[:2]
        s = self.crop
        if h > s and w > s:
            y = np.random.randint(0, h - s)
            x = np.random.randint(0, w - s)
            day, night = day[y:y + s, x:x + s], night[y:y + s, x:x + s]
        elif h != s or w != s:
            day, night = _fit(day, s), _fit(night, s)
        if self.use_flip and np.random.random() < 0.5:
            day, night = day[:, ::-1].copy(), night[:, ::-1].copy()
        return day, night

    def __getitem__(self, idx):
        day, night = self._load_crop(idx)
        to_neg = lambda a: torch.from_numpy(a.transpose(2, 0, 1).copy()) * 2 - 1
        return {'LQ': to_neg(day), 'GT': to_neg(night)}


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dd-dir', required=True)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--limit', type=int, default=4)
    a = ap.parse_args()

    ds = DayToNightDataset(a.dd_dir, crop=a.crop)
    print(f'[data] DarkDriving pairs: {len(ds)}')
    for it in [ds[i] for i in range(min(a.limit, len(ds)))]:
        l_day = float(((it['LQ'] + 1) / 2).mean())
        l_night = float(((it['GT'] + 1) / 2).mean())
        print(f"LQ(day) lum={l_day:.3f}  GT(night) lum={l_night:.3f}  "
              f"shape={tuple(it['LQ'].shape)}")


if __name__ == '__main__':
    _main()