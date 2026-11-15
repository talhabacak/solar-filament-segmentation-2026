# Final Report — Section 3: The Pipeline

*This part lists every design decision in processing order, with its parameter values and the
measurement that fixed it. Adopted values are shown in bold. Numbers are taken from committed
artefacts; the ablation ledger (Appendix A) lists the rejected alternatives.*

## 3.1 Preprocessing and caches

- Images are used as delivered (2048², 8-bit greyscale). There is no global normalisation beyond
  per-model input scaling: U-Net inputs are x/255 replicated to three channels, and YOLO uses the
  Ultralytics defaults. There is no CLAHE and no intensity harmonisation. A stack of physical
  channels (raw image, local contrast, radial map) measured −0.005 pixel-Dice, and re-reading it
  through the candidate gate confirmed that the loss falls exactly where the channels were meant to
  help (the faintest quartile reached 5 rows against 11 for the baseline).
- Training images and per-entry masks are pre-rendered once into uint8 memory-mapped caches at each
  training size (**512, 768, 1024, 1536**), with masks bit-packed per annotator entry. A live 2048²
  JPEG decode plus seven 2048² rasterisations cost ~100 ms per sample and starved the GPU.
- The mask cache thresholds the area-averaged downsample at 0.5, and that choice was measured rather
  than assumed: over 1925 ground-truth instances it retains 98.1 % of the area, reaches IoU 0.915
  against the native mask, **deletes none**, and splits 0.47 % into two components; the thinnest
  quartile (≤6 px) still loses nothing. Lowering the threshold to 0.25 enlarges masks (area ratio
  1.047 overall, 1.08 in that thinnest band) without improving their shape (IoU 0.864 against 0.863
  there).
- All candidate features are computed from the **same uint8 caches at training and application
  time by a single shared function** [`build_fused_pool.fuse_and_featurise`]. This is essential:
  an earlier float/uint8 asymmetry shifted selector probabilities by up to 0.549 and flipped 5.9 %
  of keep decisions.

## 3.2 Detector lane

- **yolo11m-seg, imgsz 1792, nc=1 (single_cls), medoid targets**: each detector is trained on one
  annotator copy per image, the medoid annotator's (`src/data/export_yolo.py`, which writes
  `data/yolo_v1`). An all-copies detector, in which every annotator entry is a training sample,
  later measured +0.0051 PQ on fold 0 (E021), but the five production detectors were not retrained.
  The full arguments are stored as `train_args` in each released checkpoint
  (`outputs/runs/e013_*/weights/best.pt`): epochs 40, patience 15, batch 2, optimizer `auto`,
  lrf 0.01, weight decay 5e-4, warmup 3, mask_ratio 4, overlap_mask false. For this schedule
  (about 360 iterations of 64 images, below Ultralytics' 10 000) `auto` resolves to AdamW with
  lr 0.002 and momentum 0.9, so the lr0 0.01 and momentum 0.937 stored with the same arguments are
  not used.
- Augmentation respects the imaging geometry: mosaic, mixup, copy-paste, erasing, shear and
  perspective are **off** (they destroy the full-disc radial context), rotations are limited to
  ±10°, flips are horizontal only, and photometric augmentation is brightness only. Two options
  that the semantic lane deliberately excludes, because the disc lies at a fixed centre with a
  measured radius spread of 0.3 %, are retained here: translate 0.05 and scale 0.10. Retraining without
  them was measured against a pool built on the same semantic field: the detector produces more
  candidates (2139 against 1991 on fold 0) and the pool's reach moves by +7 of 1510 ground-truth
  copies (23 gained, 16 lost), inside the ±8-row spread of every fusion setting (§3.4). The option
  is immaterial, and the trained detectors **are kept**.
- Backbone and input size were swept rather than assumed: yolo11l (+0.000), yolov8x@1792 (−0.0130),
  yolov8l@2048 (−0.0045), and a crack-pretrained yolov8x (+0.0055 over its own control, below
  yolo11m).
- Raw candidates are cached at **conf ≥ 0.05, NMS IoU 0.5, max_det 300** for each fold model on its
  held-out images (strictly out of fold). NMS was measured at 0.3, 0.4, 0.6 and 0.7 without the
  selector, against a control rebuilt at 0.5 through the same path: reach moves by 0, +1, 0 and +5
  of 1510 rows. Fusion clusters overlapping detections in any case, so the axis is flat and
  **0.5 is kept** out of fold.
- The **test-time** detector caches use **NMS IoU 0.05**, the optimum of the first-week sweep on one
  detector before fusion (`src/models/nms_sweep.py`: +0.0105 PQ, flat from 0.01 to 0.15), a gain
  the fused ensemble absorbed (sub005 at 0.05 read the same leaderboard 0.36 as sub004). The
  mismatch with the out-of-fold 0.5 was found when the system was reproduced (2026-11-15) and then
  measured end to end on all five folds: each fold's detector rerun once with NMS off, NMS applied
  offline, every pool scored through that fold's out-of-fold selector (which reproduces the
  shipped out-of-fold scores exactly) [`src/eda/nms_sweep_5fold.py`]:

  | detector NMS | candidates | matching | kept | PQ | Δ vs 0.5, 95 % CI | folds up |
  |---|---|---|---|---|---|---|
  | 0.05 (test-time setting) | 9130 | 4149 | 3984 | 0.4406 | −0.0016 [−0.0037, +0.0006] | 2/5 |
  | 0.1 | 9330 | 4150 | 3991 | 0.4421 | −0.0001 [−0.0021, +0.0020] | 3/5 |
  | 0.3 | 10011 | 4150 | 3987 | 0.4420 | −0.0002 [−0.0017, +0.0013] | 3/5 |
  | **0.5** (out-of-fold setting) | 10043 | 4152 | 4000 | **0.4421** | — | — |
  | 0.7 | 9886 | 4148 | 4053 | 0.4419 | −0.0003 [−0.0017, +0.0012] | 3/5 |

  The rebuilt 0.5 pool reads the shipped pool's PQ to −0.0001. The axis is flat through the
  selector, and the mismatch costs at most a sub-floor −0.0016, so the submission is unchanged. Out
  of fold each image has one detector's candidates and at test time five, so this prices NMS on the
  out-of-fold pool, not on the five-detector test pool. A fold-0 reading made first
  [`src/eda/nms_mismatch_f0.py`] gave +0.0007 for 0.05 and +0.0045 [+0.0000, +0.0088] for 0.1; the
  latter does not survive five folds.

## 3.3 Semantic lane

- **U-Net from segmentation_models_pytorch (Iakubovskii, 2019), efficientnet-b3 encoder with
  ImageNet initialisation (Deng et al., 2009), scSE decoder attention (Roy et al., 2018), imgsz
  1536, batch 2, AdamW lr 3e-4 with OneCycle (pct_start 0.1), weight decay 1e-4, 30 epochs**; loss
  = BCE + soft Dice + **clDice (weight 0.3, 5-epoch warmup)** on all-copies entries; five per-fold
  models.
- The heavy-augmentation variant uses the same loss, **60 epochs**, geometric flips, rot90 and ±20°
  rotations, and station-calibrated photometric augmentation. The 2×2 study (augmentation ×
  schedule) shows an interaction rather than main effects: light-60 degrades (0.6972), heavy-30 is
  flat (0.6975), and **heavy-60 gains (0.7028 against a 0.6995 matched control)**.
- The architecture axis is closed by measurement, and unusually completely: nine encoders
  (13 M–93 M), three decoder widths, UPerNet (Xiao et al., 2018), **UNet++** (Zhou et al., 2018),
  two transformer families (SegFormer/MixTransformer at 512 and 1536; Xie et al., 2021), two
  attention variants, a dilated-context module on the stage that carries faint objects,
  anti-aliased downsampling (Zhang, 2019), and bilinear decoder upsampling. Every one of them lands
  within ±0.001 of a same-recipe seed replicate when swapped into the deployment-faithful reference.
  Two are instructive rather than merely negative. UNet++ is a *better standalone model* (pool PQ
  0.4041 against 0.3999 for the production member) yet adds only +0.0004 as a member; the
  transformer is the only family measured to be genuinely decorrelated from the production member
  (error correlation 0.871 against 0.926–0.940 for every existing member, replicated on a second
  fold) yet loses −0.0052 against a seed control on the second fold. Strength and diversity are
  both necessary, and on 707 images every strong model is similar.
- The **instance-balanced (blob-local) variant** keeps the loss above and adds a term that weights
  each connected ground-truth instance equally within a 40 px radius (weight 1.0), so small
  filaments count as much as large ones; it is trained with the heavy augmentation for **60
  epochs**, one model per fold [`train_blobloss_local.py`; released weights `outputs/runs_aug/bloblocal_f*_s1536_w1.0_r40`].
  A single fold-0 model of it did not help, but all five folds blended at half of the field
  measured **+0.0061**, 95 % CI [+0.0051, +0.0071], the largest single realised gain of the project
  (`docs/15_ABLATION_LEDGER.md`, §7).
- **Test-time semantic ensemble `recipe_blob5_50`**, twelve members: five blob-local models at
  0.10 each (0.50 of the field), two heavily augmented members at 1/6, and five cross-fold clDice
  models at 1/30. Each member uses flip test-time augmentation and infers at its training size. The
  members are similar enough that the ensemble behaves like **nine** independent members, and
  averaging k similar members gains A(1 − 1/k) with A = +0.0092 and R² = 0.9954, so at most +0.0010
  remains to be gained from further members of the same kind; this is why the project stopped
  adding members.
- The blend weights were checked out of fold rather than inherited. Sweeping the blob share from
  0.35 to 1.00 raises *reach* monotonically on the folds used for the choice (3342 → 3395 → 3466
  rows, i.e. +71 from production to an all-blob pool) but moves the held-out folds by +10 and −1
  rows, and the leaderboard's unrounded ordering independently places the blob-heavier recipe below
  production. **0.50 is kept.**

## 3.4 Fusion into a candidate pool

- Per image, detector candidates (all fold models at test time; the fold's own model out of fold)
  and semantic connected components (**threshold 0.2, min_area 256**) are clustered by union-find
  at **IoU > 0.5**; each cluster is fused into one mask by confidence-weighted pixel voting
  (**vote 0.5**); and clusters are made disjoint greedily at build time.
- Threshold 0.2 replaced the original 0.3, and it is one of the two changes that distinguish the
  final submission from its predecessor. The adoption evidence is **+0.0050** over five folds, 95 %
  CI [+0.0011, +0.0087], positive on 4 of 5 folds, with the selector **refitted** on the new pool.
  With the old selector held fixed, the same change measures +0.0011, CI [−0.0025, +0.0048], which
  is a tie: the gain comes not from the threshold alone but from the threshold combined with a
  selector that has seen the resulting distribution. The pool's own fold-0 reading of +0.0071 is
  one fold of the tied arm and is not the adoption evidence.
- The three fusion thresholds were swept without the selector, because a change that rebuilds the
  candidate distribution cannot be read through a selector fitted on the old distribution. Reach
  relative to a control built through the same path at production settings, over 1510 fold-0
  copies: clustering at 0.4/0.6/0.7 → −4/+1/+3 rows (at −164/+153/+382 candidates); vote at
  0.40/0.60 → +6/−8; detector NMS at 0.3/0.4/0.6/0.7 → 0/+1/0/+5. **Every axis is flat**: no
  setting moves reach by more than 8 rows, while each variant exchanges up to 28 rows with
  production, which is why taking their union was the change that was measured end to end and
  lost (−0.0134). An earlier reading of these sweeps reported a loss of 18–32 rows for every
  variant. It compared the sweep pools, built on the blob field, against a reference built on a
  different semantic field, so a field difference of 24 rows was attributed to each threshold;
  rebuilding production settings through the sweep's own path removed it (Appendix A.8).
- The extraction family was explored to closure: a multi-threshold union raises raw reachability by
  +0.0318 while the oracle *drops* by 0.0177 (nested candidates carve each other), and whole masks
  with selection-time carving degrade the selector (−0.0134).

## 3.5 Selection

- **Two-stage HistGradientBoosting selector (Pedregosa et al., 2011) with 42 features**: a
  classifier multiplied by a value regressor fitted on positives only,
  `P(y=1) × E[iou_sum | y=1]`, normalised through an inner leave-one-fold-out quantile map and
  fitted out of fold on the pooled five-fold candidate set. The ten photometric and orientation
  features added to the original 32, together with the extraction threshold of 0.2, are what
  distinguishes the unbagged system (PQ 0.4421) from its predecessor.
- **Image-bootstrap bagging (Breiman, 1996).** In the submitted system the classifier–regressor
  pair is replaced by the mean of five pairs, each fitted on a bootstrap sample of *images* (not of
  candidates, which are correlated within an image), and the same averaging is used inside the
  quantile map so that the gate keeps its meaning. Locally this is +0.0010, CI [−0.0009, +0.0030],
  below the measurement floor; the leaderboard's unrounded ordering places it first among our
  submissions, and a replicate with a second bootstrap seed is also ahead of the unbagged system,
  so the ordering is not a single draw. Ten bags instead of five, a slower classifier, a more
  strongly regularised classifier and a marginal-band cascade were all measured; some read
  slightly higher locally, and none places above five bags in the leaderboard ordering
  [`src/models/fit_selector_bagged.py`, Appendix A.2]. With bagging the pooled cross-fitted PQ is
  **0.4431**.
- The loss formulation was tested against two alternatives under the identical protocol. Weighting
  every row by the PQ at stake, |iou_sum − ½·c·PQ|, measures −0.0009, and replacing the
  factorisation with a single regressor on all rows, which is the quantity the metric literally
  requires, measures −0.0016. The zero-inflated target explains this: the factorised form exploits
  the classifier's calibration at a 41 % positive rate. **The production loss is kept.**
- Three further feature families were measured and none added anything: image-level context (each
  candidate observing its image's score distribution, +0.0001), magnetic polarity-inversion
  summaries (tied), and threshold-stability descriptors (−0.0006). The last result holds despite a
  very large raw separation (0.876 against 0.309 on the 0.5/0.3 area ratio), because the six
  existing `sem_prob_*` features already carry that information. Perturbing the input masks by one
  pixel and recomputing the features as augmentation measures −0.0017.
- **Each mask loses its one-pixel boundary ring before featurisation**, and only the area features
  are recomputed, which replicates exactly the pool on which the selector was fitted. The
  implementation (`mask_shrink.shrink_frac` at a nominal 0.05) thresholds a distance field whose
  outer ring is tied, so on a thin structure any nominal fraction up to 0.19 removes the whole ring,
  a median 19 % of the area; the nominal 2.5 %, 5 % and 10 % settings once reported as equivalent
  were the same operation. A genuinely fractional shrink (`mask_shrink.shrink_true_frac`), swept once
  this was found, peaks at 10 % on the earlier pipeline (0.4331 against 0.4286 for the ring), but the
  leaderboard ranked the ring above it in both paired submissions, and above a true-fractional
  shrink of the same area. **The ring is kept.**
- **Keep policy: conf ≥ 0.40 AND p_sel ≥ 0.60.** This policy is derived rather than tuned. Because
  PQ = ΣIoU / (TP + ½FP + ½FN), a candidate is worth keeping when its expected copy-weighted IoU
  sum exceeds ½·c·PQ, which at the deployed operating point is 0.361. Measured calibration confirms
  that the gate lies exactly there: the positive rate rises monotonically with the score (0.026 at
  p ≤ 0.30 to 0.899 above 0.80), and the mean iou_sum crosses 0.361 between the 0.55–0.60 band
  (0.364) and the 0.60–0.65 band (0.485). A looser gate was submitted and scored lower on the
  leaderboard.
- Interior holes are left as the model draws them. The annotation protocol defines a filament mask
  as a single connected region without holes, and filling ours is worth +0.0001 locally (13 of 1117
  submitted masks change), but the leaderboard's unrounded ordering places the hole-filled
  submission *below* the original, so the conformity argument is aesthetic rather than scored.

## 3.6 Submission assembly

- Retained masks are written as COCO run-length encodings (Lin et al., 2014). Disjointness is
  guaranteed by construction and re-validated, and the validator also enforces the 1024-mask bound,
  round-trip identity and area sanity [`validate_submission.py`]. The final submission writes 1117
  instances over 177 of 180 images; three images legitimately yield nothing above threshold.
- The submission path refuses any checkpoint that reads external data at inference, because the
  organisers' clarification restricts final inference to the provided H-alpha frames.
- **Choice of the final submission.** The competition page allows one submission to be selected
  for the private leaderboard, and the selected one is the bagged system
  (`outputs/submissions/sub036_t020pox_bag5.csv`), which the leaderboard's unrounded ordering
  places first among our submissions. The analysis in this item and the next was made earlier, when
  two selections were expected and the private leaderboard was to score the better of the two; it
  is kept because it compares the selected system with the strongest prepared alternative. Under
  two selections the second slot is worth what it adds to the first, not its own expected score:

  $$\mathrm{gain}(B) = \mathrm{E}[\max(\mathrm{PQ}_A, \mathrm{PQ}_B)] - \mathrm{E}[\mathrm{PQ}_A].$$

  With the bagged system as $A$, six candidates were rebuilt out of
  fold with the recipe each was submitted with, and the expectation was taken over random subsets
  of 180 training images, the size of the test set [`src/eda/final_pair_selection.py`]. The bagged
  system with the marginal-band cascade at quantile 0.85 adds the most, +0.0011 (better than $A$
  on 66.5 % of the subsets; +0.0008 over $A$ on all 707 images), against +0.0003 to +0.0007 for
  the others, including a second bootstrap seed and ten bags, whose keep decisions are too close
  to $A$'s to add anything. It contains every mask of $A$ and admits 245 further candidates out of
  fold. This cascade variant is shipped as the prepared alternative
  (`outputs/submissions/sub042_t020pox_bag5_cascade085.csv`, built as described in
  `experiments/sub042.md`); on the leaderboard's unrounded ordering it places second, below the
  bagged system, and it was not selected.
- **Robustness of the alternative.** Because the hidden labels may differ from the training labels
  in ways no image property reveals (§2.3), the two were also compared on the subgroups where the
  system is weakest [`src/eda/e219_test_generalisation.py`]. The cascade variant is better on 9
  of 13 subgroups, and its gains sit in the hard regimes: +0.0067 on 2018–2022 frames, which the
  test set samples at twice the training rate, +0.0039 in the busiest quartile of disc
  high-frequency energy, and +0.0036 to +0.0038 for two of the six annotator groups. It loses
  0.0012 on the largest annotator group and 0.0032 in the second-calmest quartile. As a second
  selection it would have insured the first against a harder test set rather than duplicating it.
  With one selection, the bagged system was chosen: the cascade's local gain (+0.0008) is below
  the ±0.003 floor, and the leaderboard ordering places the bagged system first.

## 3.7 Deliberate exclusions from the production path

Each exclusion rests on a measurement rather than an opinion: physical input channels (−0.005);
GONG-distilled initialisation (+0.0015 at best, −0.0081 at 50 epochs) and in-domain masked-image
pretraining (indistinguishable from a seed replicate on both folds); transductive self-training;
the tiled native-resolution lane (0.4277 against 0.4286); the Mask2Former third source (+0.0020
across five folds, and +0.0003 more when the selector is told which lane proposed a candidate) and,
in place of the detector, Mask2Former as the only instance lane (−0.081 across five folds, 0.361
against 0.442 with the selector refitted on the new pool: its confident quarter-resolution masks
dominate the confidence-weighted fusion and lower SQ from 0.678 to 0.629;
[`outputs/eda/_m2f_instead_5fold.py`]); CRF
and intensity snapping (−0.070); magnetogram, AIA-304 and neighbouring-frame inputs (genuine
pixel-level information, no PQ); a candidate-conditioned mask refiner in three formulations (all
negative, the best of them −0.015); and larger backbones and inference sizes.

## 3.8 Reference systems

To show what each stage contributes, simpler systems built from the same components were scored
under the same protocol: all 707 training images, out of fold, the exact scorer, and every image
without a kept mask counted with all of its ground truth as false negatives
[`src/eda/e221_reference_systems.py`]. The thresholds of the reference systems were chosen in sample
as the best of a grid, which favours them; the full system's operating point is cross-fitted.

| system | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| one human annotator, scored against the others | 0.340 | 0.634 | 0.536 | | | |
| detector only (YOLO11m-seg, conf ≥ 0.4, best of 0.3–0.5) | 0.398 | 0.665 | 0.598 | 4492 | 2339 | 3707 |
| semantic lane only (field ≥ 0.5, components ≥ 256 px) | 0.413 | 0.677 | 0.610 | 5273 | 3810 | 2926 |
| fusion with a confidence gate only (conf ≥ 0.7, best of 0.4–0.8) | 0.434 | 0.675 | 0.643 | 5106 | 2569 | 3093 |
| **full system** (learned selector) | **0.443** | 0.678 | 0.653 | 4817 | 1729 | 3382 |
| majority-consensus mask (knows every annotator) | 0.641 | 0.790 | 0.812 | | | |

: Reference systems on the 707 training images, out of fold. The two human rows bound the task:
one annotator against the others, and an oracle that knows every annotator's drawing.

The semantic lane alone is stronger than the detector alone (+0.015), fusing the two with nothing
more than a confidence gate adds +0.021, and the learned selector adds +0.009 over the best such
gate. The selector's contribution is entirely in precision: it keeps 289 fewer true positives than
the gate and removes 840 false positives. Every automatic system in the table scores higher against the
annotators than one annotator scores against the others, and the consensus oracle shows how much of
the remaining distance is a property of the labels rather than of the image.
