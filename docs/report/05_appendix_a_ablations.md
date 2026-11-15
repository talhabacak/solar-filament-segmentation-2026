# Appendix A — Ablation ledger

*This appendix lists every experiment run in the project, its result, and the mechanism it
established. Entries are grouped by outcome: adopted changes (A.1), changes within measurement
noise (A.2), changes that degraded the score (A.3), and changes for which an oracle or a
precondition improved while realised PQ did not (A.4). Local numbers are pooled five-fold
cross-fitted PQ unless stated otherwise; pixel-Dice figures are fold-0 validation results of the
semantic lane.*

## A.1 Adopted changes

| Change | Effect | Mechanism |
|---|---|---|
| clDice loss (w 0.3, 5-epoch warmup) | pixel-Dice 0.6919→0.7019; PQ 0.4011→0.4066 | centreline continuity keeps thin structures intact as single components |
| All-copies training targets | 0.4283→0.4334 | learning the annotator distribution calibrates what to draw; FP 498→414 |
| Mask shrink (one-pixel boundary ring) | 0.4334→0.4387 (fold 0); 0.3995→0.4123 (5-fold) | contours systematically enclosed the GT; SQ 0.6632→0.6710 |
| Semantic lane added to detector lane | leaderboard 0.36→0.37 | different error profile; the only addition the pipeline never absorbed |
| Selector v5 (32 features, cross-lane) | 0.3954→0.4251 | provenance features carry the largest block (+0.0117) |
| Shrink + selector jointly | 0.4286 | shrink raises the oracle (0.5024→0.5078), and the selector realises it |
| uint8/float featurisation parity | bug fix | pool and application disagreed enough to flip 5.9% of keep decisions |
| Heavy augmentation × 60 epochs | 0.6995→0.7028 (pixel-Dice) | an interaction, not a main effect: light-60 degrades, heavy-30 is flat |
| Recipe-diverse semantic ensemble | 0.6980→0.7034 at native 2048 | members differing by recipe and resolution, correlations 0.908–0.932 |
| Ten photometric and orientation features (selector 32 → **42**) | 0.4372 cross-fitted | the last feature family that paid; described in §3.5 |
| Semantic extraction threshold 0.3 → **0.2**, selector refitted | **+0.0050**, 5 folds, CI [+0.0011, +0.0087], 4/5 | more candidates *and* a selector that has seen their distribution; with the old selector held fixed the same change is tied (+0.0011) |
| Selector image-bootstrap bagging (five bags) | +0.0010, 5 folds, CI [−0.0009, +0.0030] | averaging over resampled images lowers the selector's variance; below the floor locally, first in the leaderboard ordering, and a second bootstrap seed also places above the unbagged system |

: Adopted changes.

The two rows before the last raise the cross-fitted score to 0.4421; bagging raises it to
**0.4431**, and the leaderboard's unrounded ordering places the bagged result first among our own
submissions. Bagging is the only change adopted on the strength of the leaderboard ordering rather
than of a local reading above the floor; a replicate with a second bootstrap seed, submitted the
following day, confirmed the direction.

## A.2 Changes within measurement noise

| Change | Result |
|---|---|
| Detector capacity (yolo11m vs 11l) | 0.3902 vs 0.3906 |
| 8-model diversity ensemble | 0.4880 vs 0.4879; members differed only by seed/size |
| Tighter NMS + TTA on top of the ensemble | leaderboard 0.36 = 0.36 (absorbed) |
| mask_ratio 1 vs 2 | no difference |
| Soft consensus targets | 0.4071 vs 0.4066 |
| Selector v6 (metric-derived value rule) | 0.4255 = 0.4255; cleaner formulation, same operating point |
| Smooth-IoU regression target | 0.4206 vs 0.4255 |
| Crop-CNN scorer | Spearman 0.6039 vs tabular 0.6776; blend ≈ 0 |
| Single-band input | 0.6973 vs 0.7019 |
| GONG distillation over ImageNet | 0.6996 vs 0.6981 |
| Per-source confidence gates | +0.0010 in-sample |
| 2048 detector lane through pool+selector | 0.4290 vs 0.4286 (single-model gain absorbed) |
| Transductive self-training | fold-0 pixel-Dice 0.7024 vs 0.7028; the paired leaderboard reading cannot resolve differences below 0.01 |
| **Spine auxiliary head** | pixel-Dice 0.6997 vs 0.7028 (an earlier run with a corrupted spine target is retracted) |
| Ten bags instead of five | +0.0002 over five bags, CI [−0.0015, +0.0020] |
| Slower classifier (learning rate 0.03, 500 iterations) | +0.0006 over five bags |
| Regularised classifier (15 leaves, minimum leaf 60, L2 5) | +0.0014, CI [−0.0004, +0.0032] |
| Marginal-band cascade (quantile 0.85) | +0.0003 |
| Thirteen neighbourhood-geometry features (endpoint gap, tangent compatibility, collinearity) | +0.0022 bagged on the production partition; mean +0.0005 over five partitions |
| Twenty frame-relative evidence features (§4.7) | +0.0009, CI [−0.0009, +0.0027] |
| Disc high-frequency energy as a selector feature (§4.7) | +0.0002 |
| Keep threshold set by disc high-frequency energy (§4.7) | +0.0019, CI [−0.0006, +0.0044]; a single lowered threshold +0.0018 |

: Changes whose effect lies within measurement noise.

In the leaderboard's ordering none of the submitted selector variants in the last eight rows places
above five bags, and the two with the largest local readings (the regularised classifier and the
neighbourhood-geometry block) place below the unbagged system. §2.4 gives the reason local readings
of this size cannot order them: the partition alone moves such a difference by a standard
deviation of 0.0015.

## A.3 Changes that degraded the score

| Change | Cost | Reason |
|---|---|---|
| Snapping to Otsu | 0.4334→0.3634 | the target is an annotation convention, not physical structure |
| Physical input channels (raw, local contrast, radius) | 0.6971 vs 0.7019 | reach on never-proposed copies also falls (92 → 87 of 440); the 2019–2022 reading is in A.6 |
| 100-epoch schedule under light augmentation | degrades | memorisation, not schedule length (see A.1 heavy×60) |
| Spine ribbon as output representation | IoU 0.6572 vs 0.6666 | width cannot be recovered from a centreline, even with oracle width |
| Crop context 1.5→3.0 | 0.3909 vs 0.3993 | coverage gained, resolution lost |
| Removing de-overlap entirely | 0.4382 vs 0.5321 oracle | duplicates selected → FP ×10 |
| Learned in-cluster representative | 0.4216 vs 0.4241 | Spearman 0.62 too weak to overturn the consensus |
| ConvNeXt-tiny (Liu et al., 2022a) / SwinV2-T / Swin-V2-B (Liu et al., 2022b) | −0.006 to −0.017 | capacity is not the constraint |
| Sentinel-1 and Satlas (Bastani et al., 2023) initialisations | −0.0096, −0.0074 | prior from the wrong modality |
| EfficientNetV2 (B3 / S; Tan and Le, 2021) | −0.0051, −0.0065 | V2 < V1 at matched capacity |
| GONG pretraining instead of ImageNet | 0.6778 vs 0.6981 | 9,701 frames insufficient |
| GONG pretraining extended to 50 epochs | 0.6915 vs 0.6996 | distils towards the teacher, away from the annotation convention |
| yolov8x backbone (COCO) | 0.4063 vs 0.4193 | 2.6× parameters, worse |
| Inference above training size | −0.0067 to −0.0072 (3 models) | filters fitted to an apparent filament width |
| channels_last memory format | 4.7× slower | depthwise-separable convolutions fall back to slow kernels |
| Multi-threshold extraction | 0.4125 vs 0.4286 | reachability +0.0318 destroyed by build-time carving |
| Whole-mask pool + multi-threshold + late carving | 0.4152 vs 0.4286 | faint nested candidates degrade the selector (AUC 0.8937→0.8681) |
| Training images sampled in balance across cycle phase (weight inversely proportional to the square root of the era's size) | pixel-Dice −0.019 on held-out 2019–2022 images, −0.004 on earlier ones | showing 31 late frames three times as often fits those frames, not the era (§4.7) |

: Changes that degraded the score.

## A.4 Oracle gains that did not transfer to the learner

This is the most frequently repeated pattern of the project, with eighteen structurally distinct
mechanisms in all. The eleven below are those motivated by an oracle; the remainder are distributed
by subject across A.5–A.9 (new modalities, the transformer member, the three formulations of the
refiner, lane identity) and are collected on one axis in Figure 3.

| Mechanism | Oracle / precondition | Realised |
|---|---|---|
| Metric-derived value rule (v6) | +0.0054 | 0.0000 |
| De-overlap moved to the end | +0.0124 | 0.0000 |
| Learned cluster representative | +0.0225 | −0.0025 |
| Crop refiner cascade | +0.0043, then overfitting | — |
| Crop-CNN scorer | correlation 0.798 | ≈0.0000 |
| Seed-refined third lane | +0.0030 | not captured |
| Selection-time de-overlap | +0.024 | +0.0022 in-sample, −0.0011 cross-fitted |
| Multi-threshold extraction | reachability +0.0318 | −0.0161 |
| Composition of the above two | both preconditions | −0.0134 |
| Query lane (Mask2Former) as third source | reaches 120 of the 414 truly unreachable GT rows (29.0 %), 44 beyond every other source | +0.0020 over five folds |
| Spine auxiliary supervision | annotator agreement +0.079 on centrelines | pixel-Dice −0.0031 |

: Mechanisms whose oracle or precondition improved while realised PQ did not.

A further selection oracle, measured after the freeze on the frames where recall is lowest, is
reported in §4.7 rather than here because it was never turned into a mechanism: its transfer across
annotators was measured first, at −0.365.

The candidate-level value signal saturates at a Spearman correlation of 0.62–0.68 with realised IoU
across every feature family (tabular, CNN, provenance) and every candidate family (proposals,
thresholded components, learned queries). This is the binding constraint of the problem under our
formulation.

*The subsections below cover the work carried out after the final submission was frozen. Every
entry is read with the deployment-faithful protocol: the member or change is swapped into the
reference at matched member count, the selector is fitted out of fold, and the difference is
assessed with a paired image bootstrap against the ±0.003 band. Where a change rebuilds the
candidate distribution, the reading is instead selector-free (reach), because a selector fitted on
the old distribution measures the mismatch as much as the change.*

## A.5 Additional observational modalities

| Source | What was established | Realised |
|---|---|---|
| SDO/HMI magnetogram, retrieved and registered for all 887 frames | registration verified physically (H-alpha darkness against log \|B\| peaks at 0° rotation); pixel-level AUC **+0.0146** over a capacity-matched control, **0** when the magnetograms are shuffled across images | teacher (field at input) +0.0015 / +0.0003 on two folds; privileged-information student −0.0000 / −0.0006 |
| Magnetic PIL summaries as selector features | 78 % coverage, nine features | tied |
| Magnetic prior on generation | never-proposed rows lie no closer to a PIL than found ones (53.0 vs 51.4 px) | unsupported |
| AIA 304 (co-temporal, different formation height) | pixel AUC +0.0109 over the matched control | below the magnetogram, not expanded |
| Neighbouring H-alpha frame (temporal teacher) | +0.0125 pixel AUC on pairs under 1 h | same magnitude as the magnetogram, which already failed to convert |

: Additional observational modalities.

The magnetogram result is the project's clearest demonstration that information and score are
different quantities: the field genuinely indicates where filaments are at the pixel level, yet
instance matching at IoU 0.5 does not change when edges become slightly cleaner (TP 911→913,
FP 302→298).

## A.6 Representation and architecture

| Change | Against the seed floor |
|---|---|
| UnetPlusPlus | +0.0004 (a better standalone model: pool PQ 0.4041 against 0.3999) |
| SegFormer/MixTransformer member @512 | +0.0026 fold 0, **−0.0052 fold 1** (weight-matched) |
| the same @1536 | +0.0016 fold 0 |
| dilated-context module on the 96 px skip | −0.0002 |
| decoder width at the 384 px stage (64→128) | +0.0011 |
| bilinear decoder upsampling | +0.0006 |
| anti-aliased downsampling (BlurPool on five strided convolutions) | −0.0000 |
| elastic shape jitter, calibrated to the annotators' 0.629 | +0.0003 |
| sphere-rotation centre-to-limb augmentation | +0.0000 / −0.0036 |
| in-domain masked-image pretraining, fold held out | +0.0005 / −0.0002 |
| detector without translate/scale jitter | reach −17 rows |
| filament-depth compression towards the chromospheric background, α ~ U(0.55, 1) | −0.0081 on the 2019–2022 hold-out (pixel-Dice) |
| high-pass input channel with a fixed gain, on the 2019–2022 hold-out | −0.0026 (pixel-Dice; the same-distribution reading is in A.3) |
| high-pass input channel with a per-frame gain, on the 2019–2022 hold-out | −0.0056 (pixel-Dice; carries 1.02× of the contrast regime against 2.42× for the raw image) |
| every encoder skip conditioned on disc high-frequency energy (FiLM), on the 2019–2022 hold-out | +0.0142, −0.0072, −0.0189 over three seed pairs, mean −0.0040 (pixel-Dice) |

: Representation and architecture changes, measured against a same-recipe seed replicate. The last
four rows are paired runs trained on 2011–2018 and validated on 2019–2022, the last of them over
three seeds; §2.4 gives the seed noise of such a pair (standard deviation about 0.008).

The transformer entry is the informative one. It is the only member family measured to be
genuinely **decorrelated** from the production member (error correlation 0.871 against 0.926–0.940
for every existing member including a seed replicate, replicated on a second fold at 0.867), and it
still does not convert. Decorrelation is necessary but not sufficient: the member must also be
strong (its standalone pool PQ is 0.326 against 0.400 for the production member), and on 707
images every strong model is similar. UNet++ illustrates the same point from the other side: it is
stronger alone and no more useful in the blend.

## A.7 Selection-stage variants

| Change | Result |
|---|---|
| image-level context features (score rank, margin, neighbour score, local sum) | +0.0001 |
| threshold-stability features (area ratios, 0.3–0.7 IoU, high-threshold components, edge gradient) | −0.0006 |
| input augmentation (1 px mask jitter, features recomputed) | −0.0017 |
| loss weighted by the PQ at stake, \|iou_sum − ½·c·PQ\| | −0.0009 |
| single regressor on all rows, direct E[iou_sum] | −0.0016 |
| near-gate region refitted on itself | AUC 0.541 against the global 0.589 |

: Selection-stage variants: three readings of one information limit.

The stability family deserves comment because its raw separation is very large (0.876 against 0.309
on the 0.5/0.3 area ratio; candidates whose high-threshold core splits in two have a positive rate
of 0.554 against 0.673, a detector of the merge failure that requires no ground truth), and it
still adds nothing: the six existing probability-field features already carry that information.

## A.8 Candidate-pool thresholds, measured without the selector

Reach over the 1510 ground-truth copies of fold 0, relative to a control built through the same
path (same semantic field, same detector cache) at production settings:

| configuration | candidates | reach | gained | lost | net |
|---|---|---|---|---|---|
| **control** (cluster IoU 0.5, vote 0.5, NMS 0.5) | 1927 | **1087** | — | — | — |
| control rebuilt from raw detections, NMS applied offline at 0.5 | 1927 | 1087 | 0 | 0 | 0 |
| cluster 0.4 / 0.6 / 0.7 | 1763 / 2080 / 2309 | 1083 / 1088 / 1090 | 3 / 1 / 4 | 7 / 0 / 1 | −4 / +1 / +3 |
| vote 0.40 / 0.60 | 1914 / 1931 | 1093 / 1079 | 17 / 16 | 11 / 24 | +6 / −8 |
| detector NMS 0.3 / 0.4 | 1912 / 1929 | 1087 / 1088 | 1 / 1 | 1 / 0 | 0 / +1 |
| detector NMS 0.6 / 0.7 | 1912 / 1898 | 1087 / 1092 | 1 / 6 | 1 / 1 | 0 / +5 |
| detector without translate/scale jitter | 1965 | 1094 | 23 | 16 | +7 |
| blob share 0.65 / 0.80 / 1.00 | — | +4 / +9 / +10 rows on fold 0, −2 / −4 / −1 on fold 1 | | | |

: Candidate-pool thresholds, read without the selector.

Every axis is flat: no setting moves reach by more than 8 rows. The second row is a check on the
instrument: the detector was rerun once with NMS disabled and the NMS applied offline, and at 0.5
it reproduces the control exactly. An earlier version of this table reported every variant as a
loss of 17–32 rows against a reference of 1111. That reference was built on a different semantic
field (the eight-arm fold-0 committee blended with the blob member) while the sweep pools used the
blob field, so the 24-row difference between the two fields was attributed to each threshold. That
difference belongs to the semantic field, a member choice measured separately as the committee
arm, not to any threshold. The variants still exchange rows with the control, so the complementarity is real,
but taking the union is the change that was measured end to end and lost (−0.0134), because the
selector must then filter the additional candidates.

**Extraction threshold below 0.2.** The metric's own arithmetic (§1.2) favours inclusive masks, the
shape band is a length deficit along the filament (§4.3), and the threshold had never been swept
below 0.2. The rows below use the production twelve-member field; they were read on a different
pool from the table above, so absolute reach is not comparable across the two tables.

| extraction threshold | candidates | reach | gained | lost |
|---|---|---|---|---|
| **0.20** (production) | 1927 | **1087** | — | — |
| 0.15 | 1943 | 1096 | 15 | 6 |
| 0.10 | 1936 | 1106 | 27 | 8 |
| 0.05 | 1936 | 1108 | 40 | 19 |

: Semantic extraction threshold below production, fold 0, read without the selector.

Reach rises at no candidate cost, because a lower threshold lengthens existing components rather
than creating new ones, and the loss count turns at 0.05. Over five folds at 0.10 the pool gains 58
positives for 55 candidates, and in the pool 151 shape-band copies cross IoU 0.5, 49 of them drawn
independently by two annotators, while 65 copies that were matched fall below it. That loss is not
overshoot: capping how far a mask may grow beyond its 0.2 core leaves the gain unchanged and
increases the loss (8 on fold 0 without a cap; 13, 16 and 17 with caps of 8, 4 and 2 px). With a selector refitted on the new pool the change reads +0.0010,
CI [−0.0019, +0.0038], 3/5 folds, and in the leaderboard ordering it falls below every submission
built at 0.2. The production threshold stays at 0.2
[`src/eda/threshold_pool_twin.py`, `outputs/eda/_capped_growth.py`].

## A.9 Mask refinement: three formulations

The refiner takes the image, the U-Net probability field, the detector's mask union, the candidate
mask and its signed distance field, and outputs the annotators' consensus mask. It is trained out of
fold, with ground truth used only as the training target.

| formulation | matched (n=780) | boundary band (n=341) | crossings of IoU 0.5 up / down |
|---|---|---|---|
| regenerate, single-copy target | 0.687 → 0.645 | 0.300 → 0.243 | 19 / 64 |
| regenerate, consensus target | 0.676 → 0.613 | 0.295 → 0.233 | 28 / 91 |
| **residual from the identity**, consensus target | 0.676 → 0.661 | 0.295 → 0.265 | **0 / 18** |

: Mask refinement in three formulations (values before → after refinement).

The residual form, in which the candidate's mask is carried into the logits with a zero-initialised
head so that the model reproduces the candidate exactly before any gradient step, reduces the damage
to a third without changing its sign. The best correction is no correction, as §4.2 predicts: SQ is
flat at the label's own reproducibility in every band and for every filament type, so a model
trained on that label cannot systematically improve the shape.

## A.10 Structural negative results

**Gains below the measurement floor do not accumulate.** Three changes from different mechanisms
(decoder width +0.0011, bilinear upsampling +0.0006, elastic jitter +0.0003) measure −0.0004
together against a member-count control. Six last-epoch checkpoints, each +0.0007 alone, measure
−0.0024 together. The first case rules out a strategy of accumulating many small gains even when
the changes are uncorrelated.

**Reach and realised PQ diverge repeatedly.** There are four independent instances: multi-threshold
extraction (reach +0.0318, realised −0.0161), relaxed candidate thresholds, de-overlap removal, and
the blend weight. (An earlier version also listed the clustering threshold; its reach, re-read
against a control built through the same path, is flat, A.8.) A configuration that covers more ground truth also
hands the selector more candidates that it must reject, and the exchange rate is unfavourable at
every operating point we measured.

## A.11 Mask extent after selection

All rows hold selection fixed at the submitted system's out-of-fold keep decisions, so the
selector is not refitted, and change only the pixels of the kept masks. Masks remain disjoint;
pixels added to one mask are taken only from background, in selector-score order.

| variant | scope | ΔPQ | note |
|---|---|---|---|
| restore the boundary ring on every kept mask | 5 folds | −0.0117 | the erosion is worth more than first measured (+0.0046) |
| restore it only on masks thinner than W | 5 folds | −0.0001 [−0.0004, +0.0001] | W chosen on four folds, read on the fifth |
| perfect fill inside each kept bounding box | shape band, 1110 copies | ceiling +0.034 | survives annotator transfer at 6 % |
| perfect fill inside the box enlarged 1.5× | shape band, 1110 copies | ceiling +0.054 | survives annotator transfer at 25 % |
| extend with the detector's probability ≥ 0.4, boundary ring removed | fold 0 | −0.031 [−0.045, −0.017] | best of six extension variants |
| extend with the detector's probability ≥ 0.2 | fold 0 | −0.137 | 31 % of the added pixels lie on the filament |
| axial extension from mask ends, straight or ridge-tracked | fold 0, 896 matched / 62 agreed shape-band copies | best: 7 copies gained, 41 lost | evidence AUC up to 0.91, but only 2–6 % of points beyond the ends are filament |

: Changes to the extent of kept masks, with selection fixed. Ceilings are copy-wise oracles and
upper bounds; the transfer column decides the fill on one annotator and scores it on another.

The detector's sub-threshold probability is the one source in the pipeline that carries part of
the missing length (§4.6): its binary mask covers 45 % of the pixels a fused mask misses on copies
drawn by two annotators, and probability ≥ 0.3 covers 61 %. It carries more excess width than
length, which is why every extension loses. The last row follows the filament instead of
widening it: from each skeleton end of a kept mask, either along the end tangent or turning at every
2 px towards the darkest ridge, since the filaments' waviness does not persist (a straight
continuation of an annotated spine stays inside the filament for 71 % of points at 20 px and 22 % at
80 px, and extrapolating its curvature does no better). The evidence separates continuation well,
but beyond the ends of our masks the filament rarely continues, so no stop rule rescues more copies
than it breaks [`src/eda/e223_axial_continuation.py`, `src/eda/e224_spine_curvature.py`].
