# Task 2-4: Architecture Comparison — ResNet50 vs VGG16 vs DenseNet121

A controlled, fair three-way comparison of three ImageNet-pretrained CNN architectures
on the BUSI breast ultrasound benign/malignant classification task. This is a Task 2/3/4
deliverable — separate from Task 1's ad-hoc tuning exploration in `BUSI_Classification.ipynb`
and `docs/RUN_INDEX.md`.

## The three models

- **ResNet50** (`task2_resnet_baseline/`) — 50-layer residual network; skip connections
  let it be trained much deeper than plain CNNs without vanishing gradients. This is a
  **fresh, simplified baseline built only for this comparison** — not the heavily-tuned
  Task 1 ResNet (`../BUSI_Classification.ipynb`), which uses partial layer freezing, a
  class-weighted loss and F-beta(2) checkpoint selection tuned specifically for that
  investigation. That tuned model remains a Task 1-only result.
- **VGG16** (`task2_vgg16/`) — 16-layer network of stacked 3x3 convolutions followed by
  three fully-connected layers; no skip connections, the oldest and most parameter-heavy
  of the three (134M parameters, mostly in the classifier's first `Linear(25088, 4096)`).
- **DenseNet121** (`task2_densenet121/`) — connects each layer to every other layer within
  a dense block (feature reuse instead of feature re-learning), giving it by far the fewest
  parameters of the three (7M) for a comparable depth.

All three used ImageNet-pretrained weights (`IMAGENET1K_V2` for ResNet50, `IMAGENET1K_V1`
for VGG16/DenseNet121 — the versions available in this torchvision install) as the
starting point.

## Experimental setup (identical across all three models)

The point of this comparison is to isolate the effect of **architecture alone**, so
every other part of the training recipe is held fixed:

| | Value |
|---|---|
| Dataset | BUSI ultrasound images, 647 total (437 benign / 210 malignant) |
| Split | `train.xlsx` (517) → `train_sub` (413) / `val` (104), stratified 80/20, `RANDOM_SEED=42`; `test.xlsx` (130) held out untouched until final evaluation |
| Fine-tuning | Full — every layer trainable, nothing frozen (the three architectures have no comparable "equivalent freezing depth", so full fine-tuning sidesteps that confound) |
| Loss | Plain `CrossEntropyLoss()` — no class weighting, no label smoothing |
| Optimizer | AdamW, lr=1e-4, all parameters |
| Scheduler | Cosine annealing over the epoch budget |
| Batch size | 16 |
| Epoch budget | 20, with early stopping (patience=8) |
| Checkpoint selection | Best validation **macro F1** (`f1_score(average="macro")`) |
| Augmentation | Horizontal flip + rotation(±15°) + color jitter on train; resize+normalize only on val/test |

Every metric below is **macro-averaged** (benign and malignant scored separately, then
averaged with equal weight) — the convention the assignment PDF specifies as "across all
categories" — not a positive-class-only or class-weighted number.

## Results (test set, n=130: 39 malignant / 91 benign)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | AUC (malignant = positive) | Best epoch |
|---|---|---|---|---|---|---|
| ResNet50 (baseline) | 0.9000 | 0.8792 | 0.8846 | 0.8818 | 0.9265 | 9 / 17 |
| VGG16 | 0.8615 | 0.8352 | 0.8352 | 0.8352 | 0.9112 | 13 / 20 |
| DenseNet121 | 0.9000 | 0.9021 | 0.8553 | 0.8741 | **0.9301** | 9 / 17 |

AUC uses malignant as the positive class (see `ROC_comparison.ipynb` for the full ROC
curves and per-model false-positive/true-positive tradeoff, including the underlying
fpr/tpr values these observations are drawn from).

## Findings and conclusions

- **All three reach similar, solidly-working performance** once the training method is
  held fixed: accuracy 0.86–0.90, macro F1 0.84–0.88, AUC 0.91–0.93 — none of them fail,
  and the spread between best and worst is modest (≤0.038 accuracy, ≤0.019 AUC).
- **DenseNet121 has the highest AUC (0.9301) and the highest macro precision (0.9021),
  but the lowest macro recall (0.8553)** of the three — its confident predictions are the
  most trustworthy (see the ROC curve's low-false-positive-rate region), but at the
  default 0.5 threshold it is more conservative about calling "malignant."
- **ResNet50 baseline is the most balanced model** (precision ≈ recall ≈ 0.88) and ties
  DenseNet121 for the highest raw accuracy (0.90). Its ROC curve is initially weaker than
  DenseNet121's at very low false-positive rates but overtakes it through the middle of
  the curve.
- **VGG16 is the weakest of the three on every metric here** (lowest accuracy, lowest
  macro F1, lowest AUC) — but still clearly functional (F1 0.8352, AUC 0.9112, far above
  the 0.5 random-guess baseline), not a failed run.
- **Architecture matters less than training recipe, at least on this dataset.** Task 1's
  exploration (`docs/RUN_INDEX.md`) saw much larger swings from changing the training
  method alone on a single architecture (e.g. run1's overfit baseline vs. later
  regularized/class-weighted versions) than the spread seen here from swapping
  architectures under one fixed, simple recipe. That suggests most of the earlier gains
  in Task 1 came from *how* the model was trained, not from architecture choice per se.
- **Caveat on sample size**: the test set has only 130 images (39 malignant). A few
  percentage points of difference between models — e.g. DenseNet121's AUC edge over
  ResNet50 (0.9301 vs 0.9265, a 0.0036 gap) — is within plausible single-split sampling
  noise. Unlike Task 1's ResNet (validated with 5-fold cross-validation in `run10`), these
  Task 2 numbers come from a single train/val/test split, so this ranking should be read
  as indicative, not a certified ordering.

## Where to look for more detail

- Per-model training curves, confusion matrices and prediction visualizations:
  `task2_resnet_baseline/outputs/`, `task2_vgg16/outputs/`, `task2_densenet121/outputs/`.
- ROC curves and AUC derivation: `ROC_comparison.ipynb` / `roc_comparison.png` /
  `roc_comparison_results.json`.
- Task 1's separate, much more heavily-tuned ResNet investigation (not comparable to the
  numbers above): `BUSI_Classification.ipynb`, `docs/RUN_INDEX.md`, `docs/IDEAL_RESULT.md`.
