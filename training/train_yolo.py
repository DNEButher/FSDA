"""train_yolo.py — detection fine-tuning (paper Sec. 4 fairness protocol).

The recipe in configs/yolo11n.yaml is IDENTICAL for every arm (fairness
protocol); the only experimental variable is --data (the training-set yaml
produced by generation/fsda_recompose.py, or a user-prepared one). Trained
weights land in <project>/ft_<name>/weights/best.pt.

Usage (from the repository root):
    python training/train_yolo.py --data outputs/fsda/fsda_train.yaml \
        --name fsda [--seed 42]
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/yolo11n.yaml')
    ap.add_argument('--data', required=True, help='ultralytics data yaml')
    ap.add_argument('--name', required=True, help='run name -> ft_<name>')
    ap.add_argument('--seed', type=int, default=None,
                    help='default: the seed in the recipe config')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    r = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else r['seed']

    model = YOLO(r['weights'])
    model.train(
        data=args.data,
        epochs=r['epochs'], batch=r['batch'], imgsz=r['imgsz'],
        patience=r['patience'], seed=seed, deterministic=r['deterministic'],
        project=str(r['project']), name=f'ft_{args.name}', exist_ok=True,
        cache=False, workers=r['workers'], val=True,
        optimizer=r['optimizer'], lr0=r['lr0'], cos_lr=r['cos_lr'],
        mosaic=r['mosaic'], close_mosaic=r['close_mosaic'],
        hsv_h=r['hsv_h'], hsv_s=r['hsv_s'], hsv_v=r['hsv_v'],
        scale=r['scale'], translate=r['translate'], fliplr=r['fliplr'],
        amp=r['amp'], rect=r['rect'], single_cls=False, pretrained=True,
        device=args.device,
    )
    print(f'[train_yolo] DONE -> {r["project"]}/ft_{args.name}', flush=True)


if __name__ == '__main__':
    main()