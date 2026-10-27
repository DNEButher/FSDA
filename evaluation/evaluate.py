"""evaluate.py — pooled mAP50 / mAP50-95 on the BDD100K night test set.

Frozen evaluation protocol (paper Sec. 4):

    images   : held-out BDD100K night test split (3,877 images; stems in
               splits/bdd_night_test.txt)
    detector : ultralytics YOLO, imgsz=1280, conf=0.001, NMS iou=0.6,
               max_det=300 (ultralytics default, stated for completeness)
    matching : per-image greedy argmax in score order; one ground-truth box
               is matched at most once per detector box
    AP       : all-point interpolation over the detections of ALL images
               pooled together (dataset-level PR curves), 5-class macro
               average
    mAP50-95 : the same matching repeated at IoU .50:.05:.95, then averaged

Usage:
    python evaluation/evaluate.py --weights runs/ft_fsda/weights/best.pt \
        --img-dir /data/bdd100k_night/images/test \
        --labels-dir /data/bdd100k_night/labels/test \
        [--limit 500] [--out results.json]
"""
import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
THRS = [round(float(t), 2) for t in np.arange(0.50, 0.951, 0.05)]
CLS = ['person', 'bicycle', 'car', 'bus', 'truck']


def load_gt(lbl_dir, stem, w, h):
    lbl = Path(lbl_dir) / f'{stem}.txt'
    if not lbl.exists():
        return np.zeros((0, 5), np.float32)
    bs = []
    for line in lbl.read_text().strip().splitlines():
        if not line:
            continue
        p = line.split()
        c = int(float(p[0]))
        cx, cy, ww, hh = map(float, p[1:5])
        bs.append([(cx-ww/2)*w, (cy-hh/2)*h, (cx+ww/2)*w, (cy+hh/2)*h, c])
    return np.array(bs, np.float32) if bs else np.zeros((0, 5), np.float32)


def iou_1m(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a1 = (box[2]-box[0])*(box[3]-box[1])
    a2 = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    return inter / np.maximum(a1+a2-inter, 1e-9)


def ap_curve(d, nn):
    if nn == 0 or not d:
        return 0.0
    d.sort(key=lambda x: -x[0])
    tps = np.array([x[1] for x in d], np.float64)
    ct, cf = np.cumsum(tps), np.cumsum(1-tps)
    rec, prec = ct/nn, ct/np.maximum(ct+cf, 1e-9)
    rec = np.concatenate([[0], rec, [1]])
    prec = np.concatenate([[1], prec, [0]])
    for i in range(len(prec)-2, -1, -1):
        prec[i] = max(prec[i], prec[i+1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx+1]-rec[idx])*prec[idx+1]))


def pooled_maps(weights, img_dir, stems, lbl_dir):
    model = YOLO(str(weights))
    tp_all = {t: [] for t in THRS}
    n_gt_tot = {}
    for k, stem in enumerate(stems):
        p = img_dir / f'{stem}.png'
        if not p.exists():
            p = img_dir / f'{stem}.jpg'
        with Image.open(p) as im:
            w, h = im.size
        r = model(str(p), imgsz=1280, verbose=False, conf=0.001, iou=0.6,
                  max_det=300)
        bx = r[0].boxes
        gt = load_gt(lbl_dir, stem, w, h)
        n = len(gt)
        gcls = gt[:, 4].astype(int) if n else np.array([], int)
        for c in range(5):
            kk = int(np.sum(gcls == c))
            if kk:
                n_gt_tot[c] = n_gt_tot.get(c, 0) + kk
        if bx is None or len(bx) == 0:
            continue
        db = bx.xyxy.cpu().numpy().astype(np.float32)
        ds = bx.conf.cpu().numpy().astype(np.float32)
        dc = bx.cls.cpu().numpy().astype(np.int32)
        order = np.argsort(-ds)
        for t in THRS:
            gm = np.zeros(n, bool)
            for i in order:
                c = int(dc[i])
                same = np.where((gcls == c) & (~gm))[0]
                if len(same):
                    ious = iou_1m(db[i], gt[same, :4])
                    b = int(np.argmax(ious))
                    if ious[b] >= t:
                        gm[same[b]] = True
                        tp_all[t].append((float(ds[i]), 1, c))
                    else:
                        tp_all[t].append((float(ds[i]), 0, c))
                else:
                    tp_all[t].append((float(ds[i]), 0, c))
        if (k+1) % 500 == 0:
            print(f'    [{k+1}/{len(stems)}]', flush=True)
    per50, per5095 = [], []
    for c in range(5):
        nn = n_gt_tot.get(c, 0)
        per = [ap_curve([(s, tp) for s, tp, cc in tp_all[t] if cc == c], nn)
               for t in THRS]
        per50.append(per[0])
        per5095.append(float(np.mean(per)))
    return (float(np.mean(per50)), float(np.mean(per5095)), per50, per5095)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--img-dir', required=True)
    ap.add_argument('--labels-dir', required=True)
    ap.add_argument('--stems', default=str(REPO / 'splits' / 'bdd_night_test.txt'))
    ap.add_argument('--limit', type=int, default=None,
                    help='smoke test: first N images only')
    ap.add_argument('--out', default=None, help='write results to this json')
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    stems = Path(args.stems).read_text().split()
    present = []
    for stem in stems:
        if (img_dir / f'{stem}.png').exists() or (img_dir / f'{stem}.jpg').exists():
            present.append(stem)
    print(f'[eval] {len(present)}/{len(stems)} test images found', flush=True)
    if args.limit:
        present = present[:args.limit]

    t0 = time.time()
    m50, m5095, per50, per5095 = pooled_maps(
        args.weights, img_dir, present, args.labels_dir)
    print(f'  mAP50={m50:.4f}  mAP50-95={m5095:.4f}  '
          + ' '.join(f'{c}={per50[i]:.3f}' for i, c in enumerate(CLS))
          + f'  ({len(present)} imgs, {time.time()-t0:.0f}s)', flush=True)

    if args.out:
        res = {
            'weights': str(args.weights),
            'n_images': len(present),
            'map50': round(m50, 6), 'map50_95': round(m5095, 6),
            'per_class_map50': {c: round(per50[i], 6) for i, c in enumerate(CLS)},
            'per_class_map50_95': {c: round(per5095[i], 6) for i, c in enumerate(CLS)},
            't_end': dt.datetime.now().isoformat(timespec='seconds'),
        }
        Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f'[eval] RESULT -> {args.out}', flush=True)


if __name__ == '__main__':
    main()
