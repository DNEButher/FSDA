#!/usr/bin/env python
"""Supervision-fidelity analysis for FSDA (v1 SPIC revision).

Motivation
----------
The revised narrative claims that diffusion rendering trades supervision-
critical high-frequency structure for nighttime appearance, and that
frequency-selective recombination restores it.  This script turns that
adjective into a measurement: for every annotated box we compute the
consistency between the high-frequency band of a rendered night image and
that of its daytime source, then (optionally) correlate per-class fidelity
with per-class AP50 gains across arms.

Metrics (per box, inside the box region only)
---------------------------------------------
ncc_hp  : NCC between HP(log-luma) of day and render, HP = L - G_sigma(L).
          sigma defaults to 8 px, the same split used by FSDA recombination.
edge_ncc: NCC between Sobel-gradient magnitudes of day and render.
dL      : median |log-luma| shift inside the box (appearance-change context,
          NOT a fidelity metric; large = night appearance actually moved).

Usage (server)
--------------
python server_fidelity_analysis.py \
    --day-dir  /path/k7_day_images \
    --labels-dir /path/k7_labels \
    --arms fs=/path/renders_fs fp=/path/renders_fp fx=/path/renders_fx \
    [--sigma 8] [--limit N] [--ap-json ap_gains.json] \
    [--out fidelity.json]

--ap-json format: {"k7": {"person": 0.51, ...}, "fs": {...}, ...}
(per-class AP50; per-class gain = arm - k7).  When given, the script also
emits Spearman/Pearson correlation between per-class ncc_hp and per-class
AP50 gain over all (arm, class) pairs, plus a scatter-friendly table.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter, sobel
    from scipy.stats import spearmanr
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover
    HAVE_SCIPY = False


CLASSES = {0: "person", 1: "bicycle", 2: "car", 3: "bus", 4: "truck"}


def log_luma(img):
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    return np.log(luma + 1e-3)


def highpass(L, sigma):
    if HAVE_SCIPY:
        return L - gaussian_filter(L, sigma)
    # pure-numpy fallback: separable box blur x3 ~ Gaussian
    k = int(np.ceil(3 * sigma))
    size = 2 * k + 1
    kernel = np.exp(-0.5 * (np.arange(-k, k + 1) / sigma) ** 2)
    kernel /= kernel.sum()

    def blur1d(a, axis):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (k, k)
        ap = np.pad(a, pad, mode="reflect")
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis, ap)

    return L - blur1d(blur1d(L, 0), 1)


def grad_mag(L):
    if HAVE_SCIPY:
        gx = sobel(L, axis=1)
        gy = sobel(L, axis=0)
    else:
        gx = np.gradient(L, axis=1)
        gy = np.gradient(L, axis=0)
    return np.hypot(gx, gy)


def ncc(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def load_boxes(lbl_path, W, H):
    boxes = []
    if not os.path.isfile(lbl_path):
        return boxes
    with open(lbl_path) as f:
        for line in f:
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            x0 = max(int((cx - w / 2) * W), 0)
            x1 = min(int((cx + w / 2) * W), W)
            y0 = max(int((cy - h / 2) * H), 0)
            y1 = min(int((cy + h / 2) * H), H)
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                boxes.append((c, x0, x1, y0, y1))
    return boxes


def size_bucket(h_px):
    if h_px <= 64:
        return "small<=64"
    if h_px <= 128:
        return "mid<=128"
    return "big>128"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True,
                    help="name=render_dir entries")
    ap.add_argument("--sigma", type=float, default=8.0)
    ap.add_argument("--limit", type=int, default=0,
                    help="only first N stems (smoke)")
    ap.add_argument("--ap-json", default="")
    ap.add_argument("--out", default="fidelity.json")
    args = ap.parse_args()

    stems = sorted(os.path.splitext(f)[0] for f in os.listdir(args.day_dir)
                   if f.lower().endswith((".jpg", ".png")))
    if args.limit:
        stems = stems[: args.limit]

    per_arm = {name: {} for name, _ in
               (a.split("=", 1) for a in args.arms)}
    # records[arm][class] -> list of (ncc_hp, edge_ncc, dL, bucket)
    records = {name: {c: [] for c in CLASSES.values()}
               for name in per_arm}

    n_done = 0
    for stem in stems:
        day_path = os.path.join(args.day_dir, stem + ".jpg")
        if not os.path.isfile(day_path):
            day_path = os.path.join(args.day_dir, stem + ".png")
            if not os.path.isfile(day_path):
                continue
        lbl_path = os.path.join(args.labels_dir, stem + ".txt")
        day_img = Image.open(day_path)
        W, H = day_img.size
        boxes = load_boxes(lbl_path, W, H)
        if not boxes:
            continue
        L_day = log_luma(day_img)
        HP_day = highpass(L_day, args.sigma)
        G_day = grad_mag(L_day)

        for arm_spec in args.arms:
            name, rdir = arm_spec.split("=", 1)
            rpath = None
            for ext in (".png", ".jpg"):
                cand = os.path.join(rdir, stem + ext)
                if os.path.isfile(cand):
                    rpath = cand
                    break
            if rpath is None:
                continue
            r_img = Image.open(rpath).convert("RGB")
            if r_img.size != (W, H):
                r_img = r_img.resize((W, H), Image.BILINEAR)
            L_r = log_luma(r_img)
            HP_r = highpass(L_r, args.sigma)
            G_r = grad_mag(L_r)

            for c, x0, x1, y0, y1 in boxes:
                sl = (slice(y0, y1), slice(x0, x1))
                ncc_hp = ncc(HP_day[sl], HP_r[sl])
                edge = ncc(G_day[sl], G_r[sl])
                dL = float(np.median(np.abs(L_r[sl] - L_day[sl])))
                rec = (ncc_hp, edge, dL, size_bucket(y1 - y0))
                records[name][CLASSES.get(c, str(c))].append(rec)
        n_done += 1
        if n_done % 500 == 0:
            print(f"processed {n_done}/{len(stems)} stems", flush=True)

    def summarize(vals):
        arr = np.array([v for v in vals if np.isfinite(v)])
        if len(arr) == 0:
            return {"n": 0}
        q25, q50, q75 = np.percentile(arr, [25, 50, 75])
        return {"n": int(len(arr)), "med": round(float(q50), 4),
                "iqr": round(float(q75 - q25), 4)}

    out = {"sigma": args.sigma, "n_stems": n_done, "arms": {}}
    for name, cls_rec in records.items():
        arm_out = {}
        all_ncc, all_edge = [], []
        for cls, recs in cls_rec.items():
            if not recs:
                continue
            nccs = [r[0] for r in recs]
            edges = [r[1] for r in recs]
            dLs = [r[2] for r in recs]
            arm_out[cls] = {"ncc_hp": summarize(nccs),
                            "edge_ncc": summarize(edges),
                            "dL": summarize(dLs),
                            "n": len(recs)}
            all_ncc.extend(nccs)
            all_edge.extend(edges)
        for bucket in ("small<=64", "mid<=128", "big>128"):
            bk = [r[0] for recs in cls_rec.values() for r in recs
                  if r[3] == bucket]
            arm_out.setdefault("by_size", {})[bucket] = summarize(bk)
        arm_out["overall_ncc_hp"] = summarize(all_ncc)
        arm_out["overall_edge_ncc"] = summarize(all_edge)
        out["arms"][name] = arm_out

    # optional correlation with per-class AP gains
    if args.ap_json:
        with open(args.ap_json) as f:
            apd = json.load(f)
        base = apd.get("k7") or apd.get("base")
        pairs = []
        for name in out["arms"]:
            if name == "k7" or name not in apd or base is None:
                continue
            for cls in CLASSES.values():
                if cls not in apd[name] or cls not in base:
                    continue
                rec = out["arms"][name].get(cls)
                if not rec or rec["ncc_hp"].get("n", 0) < 5:
                    continue
                pairs.append({"arm": name, "cls": cls,
                              "ncc_hp": rec["ncc_hp"]["med"],
                              "ap_gain": round(apd[name][cls] - base[cls], 4)})
        out["correlation_points"] = pairs
        if len(pairs) >= 4:
            x = [p["ncc_hp"] for p in pairs]
            y = [p["ap_gain"] for p in pairs]
            out["pearson"] = round(float(np.corrcoef(x, y)[0, 1]), 4)
            if HAVE_SCIPY:
                out["spearman"] = round(
                    float(spearmanr(x, y).statistic), 4)

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: (v if k != "arms" else
                          {a: {"overall_ncc_hp": d["overall_ncc_hp"],
                               "overall_edge_ncc": d["overall_edge_ncc"]}
                           for a, d in v.items()})
                      for k, v in out.items()}, indent=2))
    print(f"\nwritten: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()