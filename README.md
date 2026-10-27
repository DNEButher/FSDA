# FSDA — Frequency-Swapped Diffusion Augmentation for Nighttime Object Detection

Official code for the paper *"Frequency-Swapped Diffusion Augmentation for
Nighttime Object Detection"*.

FSDA turns **labeled daytime images** into night-time training views without
touching the labels. Two parents are generated from the same daytime image
with the same random draws, then recombined in the frequency domain:

```
day image ──┬─ photometric chain (gamma / color gain / noise / flare)
            │        └────────────► Direct Parent      (appearance-realistic)
            └─ same chain + 4-step conditional DDIM (SDEdit warm start)
                     └────────────► Diffusion Parent   (structure-reliable)

log-luma band swap:   FSDA = low bands of Diffusion Parent
                           + high bands of Direct Parent      (Eq. 4-6)
```

Because both parents share the identical geometry, the recomposed image keeps
every ground-truth box valid — the augmentation adds night appearance
diversity while preserving supervision. With a frozen YOLO11n recipe on
BDD100K, adding 7k FSDA views to a 7k daytime-only baseline improves night
mAP50 from **0.4216 to 0.4664 (+4.5)**.

| parents (day / Direct / Diffusion / FSDA) | frequency recomposition |
|---|---|
| ![](examples/fsda_fig2_parents.png) | ![](examples/example_freqmix_panel_1.png) |

## Repository layout

```
configs/
  fsda_sigma8.yaml        main recipe (degradation + recomposition settings)
  reverse_arm.yaml        reverse frequency-swap ablation arm
  yolo11n.yaml            detection training recipe (fixed for every arm)
  degradation_stats.json  calibrated degradation constants (gamma quantiles,
                          color gains, noise model sigma(L) = a + bL)
generation/
  direct_parent.py        photometric chain + flare injection   (Eq. 1-2)
  diffusion_parent.py     SDEdit warm start + 4-step DDIM       (Eq. 3)
  fsda_recompose.py       frequency recomposition + training-set assembly (Eq. 4-6)
  train_renderer.py       train the day->night renderer from scratch
  d2n_dataset.py          paired DarkDriving day/night dataset
  _sr3/                   vendored minimal SR3-style diffusion code (see its
                          __init__.py for attribution)
fidelity/
  analyze_fidelity.py     supervision-preservation analysis (NCC / edge /
                          log-luma stats vs the daytime original)
training/
  train_yolo.py           YOLO11n fine-tuning (fairness protocol)
evaluation/
  evaluate.py             pooled mAP50 / mAP50-95 on the BDD night test set
splits/
  bdd_day_train.txt       6,999 daytime training stems (paper's 7k pool)
  bdd_night_test.txt      3,877 night test stems (held out)
examples/                 qualitative panels used in the paper
```

## Installation

```bash
pip install -r requirements.txt   # torch >= 2.0, ultralytics, numpy, scipy, ...
```

A CUDA GPU is required for `diffusion_parent.py` and `train_renderer.py`;
everything else is CPU-only.

## Data preparation

| Item | Used by | Notes |
|---|---|---|
| BDD100K 100k images + detection labels | parents, training, test | daytime training images from the 100k split (stems in `splits/bdd_day_train.txt`); night test = 3,877 stems in `splits/bdd_night_test.txt`; convert labels to YOLO txt with any standard BDD converter |
| Flare7K++ | flare bank (Eq. 2) | point `flare_dirs` at the three compound/reflective flare directories |
| DarkDriving (paired day/night) | renderer training | only needed to retrain the renderer; 9,530 same-scene pairs |
| Renderer checkpoint | `diffusion_parent.py` | pre-trained EMA weights (150k iters) — download from [Releases](../../releases) and set `renderer_ckpt` |

Then edit the placeholder paths in `configs/fsda_sigma8.yaml`:

```yaml
flare_dirs:  [/data/Flare7Kpp/...]
day_dir:     /data/bdd100k_day_images
labels_dir:  /data/bdd_day_labels        # YOLO-format txt, same stems
renderer_ckpt: /data/Ema_I150000.pth
```

## Quickstart (full pipeline)

All commands run from the repository root.

```bash
# 1) Direct Parent — photometric degradation + flare (CPU, parallelizable)
python generation/direct_parent.py --config configs/fsda_sigma8.yaml

# 2) Diffusion Parent — same seed => identical gamma/noise/flare draws,
#    plus the 4-step conditional DDIM pass (GPU)
python generation/diffusion_parent.py --config configs/fsda_sigma8.yaml

# 3) FSDA recomposition + YOLO training-set assembly (CPU)
python generation/fsda_recompose.py --config configs/fsda_sigma8.yaml

# 4) Train the detector (frozen recipe; only the data yaml differs per arm)
python training/train_yolo.py --data outputs/fsda/fsda_train.yaml --name fsda

# 5) Evaluate on the held-out BDD night test set
python evaluation/evaluate.py --weights runs/ft_fsda/weights/best.pt \
    --img-dir /data/bdd100k_night/images/test \
    --labels-dir /data/bdd100k_night/labels/test
```

To reproduce the **reverse frequency-swap ablation**, reuse the same parents
and only change the config:

```bash
python generation/fsda_recompose.py --config configs/reverse_arm.yaml
python training/train_yolo.py --data outputs/fsda_reverse/fsda_train.yaml --name fsda_reverse
```

Large-scale generation can be sharded across machines / GPUs:

```bash
python generation/diffusion_parent.py --config configs/fsda_sigma8.yaml --shard 0 --nshards 4
python generation/diffusion_parent.py --config configs/fsda_sigma8.yaml --shard 1 --nshards 4
# ... (run direct_parent.py with the SAME shard topology so that the
#      gamma/noise/flare draws stay paired per image)
```

## Main results (BDD100K night test, 3,877 images, YOLO11n)

| Training set (7k day + 7k augmented) | mAP50 | mAP50-95 |
|---|---|---|
| Day-only baseline (no augmentation)   | 0.4216 | — |
| + ODGEN (strongest external baseline) | 0.4505 | — |
| + Direct Parent (ours)                | 0.4491 | 0.2584 |
| + Diffusion Parent (ours)             | 0.4510 | 0.2619 |
| **+ FSDA (ours)**                     | **0.4664** | **0.2670** |

Multi-seed (seeds 42/43/44, mean ± std): FSDA **0.4631 ± 0.0087** vs
day-only 0.4234 ± 0.0096 — the +4 pp margin is far outside seed noise.
Zero-shot transfer of the same checkpoints (no target-domain training):
ACDC-night +0.90, NightCity +0.89, Dark Zurich +0.29 AP50 over the day-only
baseline. See the paper for the full comparison table, the
budget-vs-value study with real night data, and the supervision-fidelity
analysis (`fidelity/analyze_fidelity.py`).

## Retraining the renderer

The released renderer checkpoint was produced with:

```bash
python generation/train_renderer.py --dd-dir /data/DarkDriving_720p \
    --iters 150000 --batch 16 --crop 128 --out outputs/renderer
# use outputs/renderer/Ema_I150000.pth as `renderer_ckpt`
```

## Reproducibility notes

- **RNG discipline** — `np.random` (seed + shard) is consumed only by the
  gamma and noise draws; flare draws use an isolated `default_rng` stream;
  the DDIM warm-start noise uses `torch.manual_seed(seed + shard)`. Both
  parent scripts therefore produce *identical* photometric draws per image,
  which is what makes the paired construction label-preserving.
- **Bit-exactness** — all generation math (degradation constants, flare
  affine canvas, DDIM loop, band swap) is the exact code used for the
  paper's runs.
- **Validation split** — the assembled data yaml sets `val` to the training
  images for convenience; the paper's runs selected `best.pt` on an internal
  validation split. Final metrics are always computed with
  `evaluation/evaluate.py` on the held-out night test set, so this only
  affects intermediate checkpoint selection.

## License and citation

Code is released under the MIT License (see `LICENSE`). Datasets and the
vendored `_sr3` subset keep their original licenses/attributions.

The paper is under review; the citation entry will be added upon publication.

## Acknowledgments

- [ReDDiT](https://arxiv.org/abs/2410.12346) (CVPR 2025) and SR3 — the
  diffusion architecture vendored in `generation/_sr3`.
- Flare7K++ — lens-flare source library.
- DarkDriving — paired day/night driving images for renderer training.
- BDD100K — daytime training images and the night test protocol.
- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics).