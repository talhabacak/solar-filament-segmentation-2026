---
title: "Solar Filament Instance Segmentation with a Two-Lane Candidate Pool and a Metric-Derived Selector"
author: "Talha Bacak (Kaggle team TalhaBacak)"
subtitle: "Technical report, Solar Filament Segmentation Challenge 2026"
repo: "https://github.com/talhabacak/solar-filament-segmentation-2026"
lang: en
---

::: abstract
**Abstract.** We describe the system behind our two final submissions. Two independent lanes
propose filament instances on the full-disc GONG H-alpha frame: a YOLO11m-seg detector at 1792 px
and a twelve-member U-Net (EfficientNet-B3) semantic ensemble at 1536 px. Candidates are fused by
confidence-weighted voting, described by 42 features, and kept or dropped by a bagged two-stage
gradient-boosted selector whose keep threshold is derived from the Panoptic Quality formula and
not tuned. Out of fold on all 707 training images the system reaches PQ 0.443 (SQ 0.678,
RQ 0.653), where one annotator scored against the others reaches 0.340; the public leaderboard
displays 0.40. We report every parameter, the validation protocol, the Dice, IoU and relation
distributions, the morphology of the predictions including barbs, and the measured negative
results. The repository rebuilds both submitted files byte for byte.
:::

# 1. Task and metric

The competition scores Panoptic Quality [2],

$$\mathrm{PQ}=\frac{\sum_{(g,p)\in TP}\mathrm{IoU}(g,p)}{|TP|+\tfrac12|FP|+\tfrac12|FN|},$$

with a match requiring IoU > 0.5, pooled over annotator-image entries: an image annotated by
several people is scored once against each annotator's copy. Our implementation
(`src/filament/pq.py`) is tested for equality with the organisers' published scorer. Three
consequences shaped the design. (i) The task is instance reconstruction: a correct probability map
that merges two filaments loses two matches. (ii) Reward is discontinuous at IoU 0.5, so the last
stage must decide which candidates to keep. (iii) That decision has a price that the formula
states, which we use in Section 3.5.

Two human reference points bound the scores below (Table 2). One annotator scored against the
others reaches PQ 0.340, and two annotators drawing the same filament agree at mean IoU 0.634. A
majority-consensus mask built from every annotator's drawing reaches 0.641.

# 2. Data and validation protocol

**Data.** 707 training and 180 test frames (2048 × 2048, 8-bit greyscale JPEG) from the six GONG
stations [11], 2011 to 2022, with 8199 polygon annotations in 1154 annotator-image entries [1].
The solar disc is centred at (1023.6, 1024.6) px with radius 899.95 px (standard deviation 0.73 px
over all 887 frames), so no geometric registration is applied. Filament pixels cover 0.302 % of a
frame.

**Split.** Five folds, `GroupKFold` grouped by physical image, so that all annotator entries of an
image share a fold. The assignment was frozen once (`data/splits/folds_v1.parquet`) and used
unchanged by every experiment. Station counts differ by at most two images between folds.

**Out-of-fold protocol.** Every learned layer is cross-fitted. Network predictions for a training
image come only from the fold model that did not see it; the selector is fitted on four folds and
applied to the fifth, with its score normalised through an inner leave-one-fold-out quantile map.
All PQ values in this report are pooled over the five held-out folds with the exact scorer.
Differences are judged by a paired bootstrap over images [12] (4000 resamples) against a pre-registered
floor of ±0.003: a seed replicate of one ensemble member moves PQ by ±0.001, and the same selector
variant re-read on five different fold partitions moves with standard deviation 0.0015.

**Test labels.** The public MAGFiLO 1.0 release [1] contains annotations for all 180 test images.
We verified this at the identifier level in the first week, quarantined the release, and never
used it for training, model selection or prediction.

# 3. Pipeline

Table 1 lists every parameter. The stages are described in processing order.

**3.1 Preprocessing.** Images are used as delivered. There is no contrast equalisation, intensity
harmonisation or limb-darkening correction; a stack of physical channels (local contrast, radial
map) measured −0.005 pixel-Dice and is not used. For the semantic lane the frame is resized once
to 1536 × 1536, scaled to [0, 1] and replicated to three channels. Masks are downsampled by area
averaging and thresholded at 0.5, a choice we measured: it retains 98.1 % of the annotated area
and deletes no instance. The detector uses the Ultralytics input pipeline at 1792 px. Every
annotator entry is a separate training sample in both lanes, because the metric scores against
every copy (+0.0051 PQ over a consensus target).

**3.2 Detector lane.** YOLO11m-seg [3], single class, one model per fold. Augmentations that break
the full-disc geometry (mosaic, mixup, copy-paste, erasing, shear, perspective, vertical flip) are
off. Out-of-fold candidates are cached at confidence ≥ 0.05 with NMS IoU 0.5; at test time the
candidates of all five fold models are pooled. Input size was swept (1024, 1536, 1792, 2048; 1792
is the last that improves), as were larger backbones (YOLO11l +0.000, YOLOv8x −0.013).

**3.3 Semantic lane.** U-Net [4] with an ImageNet-initialised [13] EfficientNet-B3 encoder [5]
and scSE decoder attention [7], built with `segmentation_models_pytorch` [6]. The loss is
BCE + soft Dice + 0.3 · clDice [8]; clDice rewards skeleton overlap, because a one-pixel break
splits an instance and PQ charges it twice. Three member types are trained: (a) the base recipe,
one model per fold; (b) an instance-balanced ("blob") variant, one per fold, whose loss weights
each connected instance equally within a 40 px radius, so that small filaments count as much as
large ones; (c) two heavily augmented models trained for 60 epochs at 768 and 1536 px on the fold-0 split. The
test-time field is their weighted mean, 0.50 for (b), 1/6 for (a) and 1/6 for each model of (c),
each member inferred at its training size with horizontal-flip averaging.

**3.4 Fusion.** Per image, detector masks (confidence ≥ 0.05, centroid r/R ≤ 0.97) and connected
components of the semantic field (p ≥ 0.2, area ≥ 256 px) are clustered by union-find at
IoU > 0.5. Each cluster becomes one mask by confidence-weighted pixel voting at 0.5, and masks are
made pixel-disjoint greedily in confidence order, as the submission format requires. Each
candidate receives 42 features from one shared function used at training and at test time:
geometry (area, aspect, eccentricity, boundary complexity, skeleton length, width), position
(r/R, latitude), semantic support (mean, maximum and 90th-percentile probability, ring contrast),
detector support (per-model confidence, agreement, vote count), image context (candidate density),
photometry of the raw frame (contrast against a ring, dark core) and structure-tensor orientation
(coherence, alignment of texture with the mask axis).

**3.5 Selector and keep rule.** The selector is two HistGradientBoosting models [9]: a classifier
for "this candidate matches some annotator copy" multiplied by a regressor, fitted on positives
only, for the IoU summed over the copies it matches. The product is mapped to a quantile of its
out-of-fold distribution, giving a score p~sel~. The submitted selector averages five such pairs,
each fitted on a bootstrap sample of images [10]. The keep rule is derived from the metric: adding
a candidate raises PQ when its expected copy-weighted IoU sum exceeds ½ · c · PQ (c = mean copies
per image), which is 0.361 at our operating point. Measured calibration crosses that value between
the score bands 0.55–0.60 (0.364) and 0.60–0.65 (0.485), so the gate is confidence ≥ 0.40 and
p~sel~ ≥ 0.60.

**3.6 Post-processing and output.** Before featurisation each mask loses its one-pixel boundary
ring, which corrects a measured tendency of our contours to enclose the annotators' (+0.0128 PQ).
Masks under 64 px are dropped and the rest are written as run-length encodings. A validator checks
disjointness, round-trip identity and the 1024-mask bound. The first final submission holds 1117
instances on 177 of 180 images. The second adds a cascade: a classifier fitted on out-of-fold
candidates with p~sel~ in [0.40, 0.60) admits the test candidates of that band above its 0.85
quantile (65 more instances). It was chosen because it adds the most to the first as a second
slot, E[max(PQ₁, PQ₂)] − E[PQ₁] = +0.0011 over random 180-image subsets, with its gains in the
hardest regimes (+0.0067 on 2018–2022 frames).

| Component | Setting |
|---|---|
| Detector | YOLO11m-seg, 1792 px, single class, 40 epochs, patience 15, batch 2, lr0 0.01 → lrf 0.01, momentum 0.937, weight decay 5e-4, warm-up 3 epochs |
| Detector augmentation | rotation ±10°, translate 0.05, scale 0.10, horizontal flip 0.5, brightness 0.3; all others off |
| Semantic model | U-Net, EfficientNet-B3 (ImageNet), scSE, 1536 px, batch 2, AdamW lr 3e-4, weight decay 1e-4, OneCycle (pct_start 0.1), mixed precision, 30 epochs |
| Semantic loss | BCE + Dice + 0.3 · clDice (5-epoch warm-up); blob variant: radius 40 px, weight 1.0 |
| Semantic augmentation | horizontal flip 0.5, rotation ±10° about the disc centre, gamma 0.8–1.25; heavy members: flips, 90° rotations, ±20°, station-calibrated photometry, 60 epochs |
| Fusion | semantic threshold 0.2, minimum area 256 px, cluster IoU 0.5, vote 0.5, detector NMS 0.5, at most 300 detections |
| Selector | classifier: 300 iterations, lr 0.05, 31 leaves, minimum leaf 25, L2 1.0; regressor: 300 iterations, lr 0.06; 5 image-bootstrap copies; 10 042 training candidates |
| Keep rule | confidence ≥ 0.40 and p~sel~ ≥ 0.60 |

: All parameters of the submitted system.

# 4. Results

Table 2 scores simpler systems built from the same components under the same protocol. The
thresholds of the three reference systems were chosen in sample, which favours them. The semantic
lane alone beats the detector alone, fusing them adds +0.021, and the learned selector adds +0.009
more, all of it precision: it removes 840 false positives at a cost of 289 true positives.

| System | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| One annotator vs. the others | 0.340 | 0.634 | 0.536 | | | |
| Detector only | 0.398 | 0.665 | 0.598 | 4492 | 2339 | 3707 |
| Semantic lane only | 0.413 | 0.677 | 0.610 | 5273 | 3810 | 2926 |
| Fusion, confidence gate | 0.434 | 0.675 | 0.643 | 5106 | 2569 | 3093 |
| **Full system** | **0.443** | 0.678 | 0.653 | 4817 | 1729 | 3382 |
| Consensus of all annotators | 0.641 | 0.790 | 0.812 | | | |

: Out-of-fold results on the 707 training images (8199 annotator copies).

**IoU and Dice distributions.** Over the matched copies, IoU has mean 0.678 and median 0.683
(quartiles 0.615 and 0.742; 5th and 95th percentiles 0.533 and 0.810). The corresponding Dice has
mean 0.805 and median 0.812 (quartiles 0.762 and 0.852; 5th and 95th percentiles 0.695 and 0.895).
On the 2483 matched copies that two annotators drew, our mask is closer to an annotator than the
second annotator is in 76.2 % of cases. Mask quality is therefore at the reproducibility of the
label, and the remaining loss is in which objects are recognised: recall is 0.588 at precision
0.736. These distributions and Table 3 are read before selector bagging (PQ 0.4421); bagging
adds +0.0010.

**Relation distributions.** Table 3 uses the definition of the organisers' self-evaluation
notebook (one IoU matrix per annotator-image entry; a link is any overlap). Fragmentation and
merging are each near 2–3 %; the rest of the unmatched mass is pure misses and pure extras.

| Side | one | two or more | none |
|---|---|---|---|
| Ground truth → predictions | 69.9 % | 2.1 % | 28.0 % |
| Predictions → ground truth | 87.8 % | 2.6 % | 9.6 % |

: One-to-many (fragmentation) and many-to-one (merging) relations, out of fold.

**Where the loss is.** Of the on-disc annotator copies, 60.6 % are matched, 11.5 % had a candidate
above IoU 0.5 that the selector rejected, 3.4 % were never proposed, and 24.6 % have a candidate
with IoU between 0.05 and 0.5: the right object with the wrong shape. Of that band's copies, 66.5 % were
drawn by no second annotator; where a second annotator confirms the copy, our candidate is too
short, not too thin (median skeleton ratio 0.72, width ratio 1.04). PQ is flat across the disc (0.446 to 0.455 for
r/R < 0.8) and falls at the limb through recognition alone (RQ 0.556 at 0.8–0.9), while SQ stays
at 0.68.

**Leaderboard.** The public leaderboard displays 0.40. The offset of −0.04 to −0.05 from the
local estimate has been the same for every generation of the pipeline, including early ones
without the selector, and the test images are statistically indistinguishable from the training
images (classifier AUC 0.534). Scoring the same masks against different annotators moves PQ by
−0.036 to +0.038, so we attribute the offset to the test annotators and expect 0.40 to 0.42 on the
private half.

**Cost.** For the 180 test images, each of the twelve semantic members takes about 21 s on one
GPU; fusion and featurisation take 0.48 s per image on a CPU; the selector takes seconds.

# 5. Morphology of the predictions

**Barbs.** Using the annotated spines, we define a barb pixel as one that is thinner than the body
(removed by a morphological opening at 0.35 of the body width) and farther from the spine than
half a body width plus 2 px. Barbs make up 0.17 % of the annotated area, in 303 of 8199 copies.
Our matched masks cover 38.0 % of barb pixels where a second annotator covers 49.3 %, and 80.5 %
of the body where a second annotator covers 79.7 %. The body is thus recovered as completely as a
second person draws it; barbs are where the system falls short of human agreement, and also where
humans agree least. Figure 1 shows both outcomes: a short barb attached to a dark body is
followed, a long faint one is not. This is the same problem as the missing faint length of a
filament: among annotator-confirmed copies with the wrong shape, the ensemble's probability is
essentially zero over the missing part in 75 %. The limit is therefore in the networks' response
and not in the post-processing; moving the fusion vote towards the union costs −0.058 PQ.

![Barbs (shaded) of annotated filaments with the annotator's outline (solid green), our
out-of-fold mask (dashed red) and the annotated spine (yellow). Top row: the three matched copies
whose barbs are recovered best; bottom row: the three recovered
least.](../../docs/report/figures/fig4_fine_structure.png)

**Boundaries.** Deficits of matched masks lie at the rim: 74 % of surplus and 82 % of missing
pixels are within 3 px of the annotated boundary. Snapping masks to image intensity, the
physically motivated choice, costs −0.070 PQ, because the target is the annotation convention.

**Unannotated filaments.** The sixteen scored false positives with the highest selector score are
all visibly filaments that no annotator of that image drew (`outputs/gallery/fp_unannotated.png`).
Part of the false-positive count is therefore recall of real structure.

# 6. Measured and not adopted

Each row of Table 4 was implemented and measured under the protocol of Section 2. The full record,
more than 200 numbered experiments with their acceptance rules, is in the repository.

| Change | Measured effect on PQ |
|---|---|
| Nine encoders (13–93 M parameters), UNet++, UPerNet, SegFormer as a member | within ±0.001 of a seed replicate |
| Native-resolution (2048 px) tiled lane | 0.4277 against 0.4286 |
| Mask2Former as a third lane | +0.0020, CI [−0.0027, +0.0066] |
| Candidate-conditioned mask refiner, three forms | best −0.015 |
| Physical input channels | −0.005 pixel-Dice |
| Pretraining on unlabelled GONG frames, two routes | equal to a seed replicate |
| Co-temporal HMI magnetogram as teacher input | +0.0015 and +0.0003 on two folds |
| Centre-to-limb sphere-rotation augmentation | +0.0000 and −0.0036 |
| Semantic threshold 0.10 with refitted selector | +0.0010, CI [−0.0019, +0.0038] |
| Keep threshold set by image regime | +0.0019, CI [−0.0006, +0.0044] |
| Filling interior holes | +0.0001 |

: Changes that did not pass the ±0.003 floor.

The apparent headroom of the selection stage is also not recoverable: reordering the candidate
pool with knowledge of one annotator is worth +0.033, but the same choice scored on a different
annotator loses, so it reflects one person's judgement.

# 7. Reproducibility, data provenance and tools

**Code.** The repository (MIT licence) is modular: `src/filament` (metric, encodings), `src/data`
(caches), `src/models` (training, inference, submission) and `src/eda` (one script per
measurement). `requirements.txt` pins every package. The notebook
`notebooks/00_end_to_end_pipeline.ipynb` runs the pipeline in three modes: the final stage from
the shipped test candidate pool (CPU, seconds), full inference from the released weights, or a
complete retrain.

**Determinism.** From the candidate pool, both submitted files are rebuilt byte for byte, and so
is the first after refitting the selector from the shipped training pool. From the checkpoints,
the rebuild of the predecessor system was byte-identical on the original machine; on a different
GPU architecture 1091 of 1117 encodings were identical and one instance changed (±0.0004 PQ).

**External data.** No external annotation was used. ImageNet encoder weights and COCO-pretrained
[14] detector weights are the only external inputs to the submitted models. Magnetograms and
unlabelled GONG frames appear only in the experiments of Table 4, and inference reads only the
provided H-alpha images.

**AI assistance.** AI tools were used for coding and drafting, as the organisers permit. Every
design decision above traces to a recorded measurement in the repository.

# References

::: references
[1] A. Ahmadzadeh et al., "MAGFiLO: a dataset of manually annotated filaments from H-alpha observations," *Sci. Data* 11, 1031, 2024.

[2] A. Kirillov et al., "Panoptic segmentation," CVPR, 2019.

[3] G. Jocher and J. Qiu, *Ultralytics YOLO11*, software, 2024.

[4] O. Ronneberger, P. Fischer and T. Brox, "U-Net: convolutional networks for biomedical image segmentation," MICCAI, 2015.

[5] M. Tan and Q. Le, "EfficientNet: rethinking model scaling for convolutional neural networks," ICML, 2019.

[6] P. Iakubovskii, *Segmentation Models Pytorch*, software, 2019.

[7] A. G. Roy, N. Navab and C. Wachinger, "Concurrent spatial and channel squeeze and excitation in fully convolutional networks," MICCAI, 2018.

[8] S. Shit et al., "clDice: a novel topology-preserving loss function for tubular structure segmentation," CVPR, 2021.

[9] F. Pedregosa et al., "Scikit-learn: machine learning in Python," *JMLR* 12, 2011.

[10] L. Breiman, "Bagging predictors," *Mach. Learn.* 24, 1996.

[11] J. W. Harvey et al., "The Global Oscillation Network Group (GONG) project," *Science* 272, 1996.

[12] B. Efron and R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.

[13] J. Deng et al., "ImageNet: a large-scale hierarchical image database," CVPR, 2009.

[14] T.-Y. Lin et al., "Microsoft COCO: common objects in context," ECCV, 2014.
:::
