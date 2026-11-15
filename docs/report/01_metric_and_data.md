# Final Report — Draft Sections 1–2

## 1. The evaluation metric

### 1.1 Definition and verified semantics

The competition is scored by Panoptic Quality (Kirillov et al., 2019):

::: equation
$$\mathrm{PQ} = \frac{\sum_{(g,p)\in \mathit{TP}} \mathrm{IoU}(g,p)}{|\mathit{TP}| + \tfrac{1}{2}|\mathit{FP}| + \tfrac{1}{2}|\mathit{FN}|}$$

[(1)]{.eqno}
:::

We did not rely on the published description alone. The organisers released their exact scorer in
the Self-Evaluation notebook, and our local implementation was verified against it line by line;
every local score in this report is produced by that parity-verified implementation
[`outputs/research/official_pq_scorer_extracted.py`, `src/filament/pq.py`]. The details that shaped
our design decisions are the following.

1. **Matching is strict.** A prediction matches a ground-truth instance only at IoU **strictly
   greater** than 0.5. No Hungarian assignment is needed: because predictions must be
   pixel-disjoint, at most one prediction can exceed 0.5 against a given ground-truth mask, so
   thresholding alone induces a one-to-one matching.
2. **Aggregation is global, per annotator-image entry.** The same prediction set for an image is
   scored independently against *each* annotator's copy of that image, and the TP/FP/FN counts and
   the IoU sum accumulate over all entries of all images before a single division. There is no
   per-image averaging.
3. **Predictions must be pixel-disjoint**, and a practical constraint follows from the reference
   tooling: the RLE stack of pycocotools overflows 32-bit integers at h·w·n ≥ 2³², which at 2048²
   is exactly 1,024 masks per image. Our submission validator enforces both
   [`src/models/validate_submission.py`].

### 1.2 Consequences of multi-annotator scoring

Of the 707 training images, 411 carry one annotator entry, 145 two, and 151 three (1,154 entries;
8,199 ground-truth instances). Scoring one prediction set against several diverging copies has two
consequences that governed our work.

**A perfect score is impossible, and the ceiling is computable.** Two annotators drawing the same
filament agree at median IoU 0.629 (mean 0.634), measured over 2,144 cross-annotator matched pairs
[`src/eda/spine_vs_mask_iou.py`]. One prediction set cannot simultaneously match copies that
disagree with each other to this extent. With our matched-mask quality (SQ 0.678 at the submitted
operating point) and perfect recall inside the consensus tiers that the metric rewards predicting,
the pooled metric is bounded at PQ ≈ 0.63 [`src/eda/what_would_055_take.py --sq 0.6782`].

A corollary reframes all mask-quality work: our SQ (0.678) already **exceeds** inter-annotator
agreement (0.634 on the same mean-over-matched-pairs statistic). A model trained on all annotator
copies predicts close to the consensus, and the consensus lies closer to each individual than the
individuals lie to one another. The same measurement shows that annotators agree substantially
better on the *centreline* than on the boundary (ribbon IoU 0.708 against mask IoU 0.629), which
motivates the auxiliary supervision reported in §4.

**The metric prices every prediction, and the price is computable.** Predicting a filament drawn
by $k$ of an image's $N$ annotators yields $k$ true positives and $N-k$ false positives. Since
leaving it unpredicted still costs $k/2$ in false negatives, the *marginal* value of predicting it
satisfies

::: equation
$$\Delta\mathrm{PQ} \propto \frac{2\,\mathrm{SQ}\,k}{N},$$

[(2)]{.eqno}
:::

so a cluster is worth predicting whenever $2\,\mathrm{SQ}\,k/N$ exceeds the PQ already achieved. At
the submitted operating point (PQ 0.4421, SQ 0.6782) the threshold is $k/N > 0.326$, and the lowest
tier, an instance drawn by one annotator in three, lies at 0.333. Every consensus tier is therefore
still worth predicting, but by a margin of 0.007 rather than comfortably, and the margin closes as
PQ rises: the same fixed point solved at the perfect-recall ceiling drops the 1-of-3 tier.
Predicting minority-consensus filaments is correct *at our score*, not in general. We record this
derivation together with its correction history: an earlier version charged each cluster its
*average* rather than its *marginal* contribution and wrongly concluded that minority tiers should
be skipped [`experiments/MISTAKES.md`].

In the form used by the pipeline, the same inequality states that a candidate earns its place when
its expected copy-weighted IoU sum exceeds ½·c·PQ, which at the deployed operating point is 0.361.
That number is not a tuning constant, and the selector's measured calibration confirms that it lies
where the metric places it: the mean copy-weighted IoU crosses 0.361 between the 0.55–0.60 score
band (0.364) and the 0.60–0.65 band (0.485), which is exactly where the deployed gate is placed
[`src/eda/selector_information_limit.py`].

### 1.3 What the metric does *not* measure

Two measured results frame everything that follows. First, snapping predicted masks to intensity
structure (Otsu thresholding within the local window; Otsu, 1979) costs −0.070 PQ: the target is
the annotators' drawing convention, not a physically optimal boundary. Second, the deficit of masks
that already match is concentrated at the rim: 74% of our surplus pixels and 82% of our missing
pixels lie within 3 px of the ground-truth boundary [`src/eda/sq_anatomy.py`], so the residual mask
error is boundary noise of the kind annotators introduce. Both results lead to the same
conclusion: beyond IoU ≈ 0.7, the metric measures agreement with a convention whose irreducible
spread is the inter-annotator figure of 0.629 given above.

## 2. Data

### 2.1 Composition and geometry

The data comprise 707 training and 180 test H-alpha full-disk frames (2048², 8-bit greyscale,
JPEG q≈90) from the six ground-based GONG stations (B, C, L, M, T, U; Harvey et al., 1996),
2011–2022. Measured geometry: the solar disc is centred at (1023.6, 1024.6) px with radius
899.95 px, standard deviation 0.73 px over all 887 frames (898.0–902.3), identically in the training
and test sets [`src/eda/disk_refit.py`]. The GONG pipeline rescales every frame to this disc: the
apparent solar diameter changes by 3.3 % between perihelion and aphelion, about 30 px at this scale,
while the fitted radius spans 4.4 px and does not follow the season (correlation −0.063 with the
expected apparent size) [`outputs/eda/_sun_distance.py`]. No annotated filament extends beyond
903 px from the centre, and filament pixels (union over annotators) cover 0.302% of a frame
[`src/eda/tile_geometry.py`].

### 2.2 Station variation lies in contrast, not in level

Per-station statistics over the training set [`src/eda/aug_calibration.py`] show that the mean disc
intensity varies only between 108.9 and 113.0 (the exposure control loop normalises it), while
local contrast varies by 46% (31.0 at station T to 45.2 at M): atmospheric transparency and seeing
act directly on the contrast of an absorption feature. This measurement, not intuition, set our
augmentation ranges (brightness ±5 DN, contrast ±0.22, blur σ 0.2–0.7 at 768 px, scaled with input
size); our first hand-picked ranges were wrong on every axis and in both directions, which is
recorded as a methodological point in §6.

A related negative result: the competition JPEGs carry genuine q≈90 blocking at native resolution
(blockiness 1.139), but area downsampling to any training size removes it (1.003–1.020).
Compression augmentation at training resolution therefore synthesises an artefact that the model
never encounters, and changes the image 9× more than native-resolution encoding would; it was
removed [`src/eda/aug_onthefly.py`].

### 2.3 Relationship between training and test sets

The test split is i.i.d. with the training set rather than a temporal or spatial hold-out
[`src/eda/train_test_shift.py`]: all 12 years appear on both sides in counts close to proportional,
with one exception that matters, since 2021 and 2022 hold 7.8 % and 3.9 % of the test images
against 4.0 % and 1.7 % of the training images, and those are the years on which the pipeline is
weakest (§4.7; reweighting the local score onto the test set's year mix lowers it by 0.0022); no
station is held out (training 14.4–18.0% per station, test 14.4–18.9%); covariate shifts are
0.03 sd (sharpness), 0.15 sd (level) and −0.13 sd (contrast); 31 calendar days appear on both
sides, 18.3% of test images share a day with a training image, and 36% lie within 24 h of one.
This measurement predicts, correctly as §4 shows, that transductive adaptation has nothing to adapt
to, and it justifies validation on station-stratified folds.

**Inspection of the test images.** We examined all 180 test frames without labels, together with
the submitted masks and the candidates just below the keep threshold, and compared every per-frame
quantity we could measure with the training frames [`outputs/eda/_test_full_review.py`,
`outputs/eda/_test_review_checks.py`, `outputs/eda/_hf_mix_by_quartile.py`,
`outputs/eda/_sun_distance.py`]. The public and private leaderboards are undisclosed halves of these
180 images, so the inspection covers both. Reweighting the out-of-fold score onto the test set's mix
shifts it by −0.0002 for the disc high-frequency energy of §4.7, −0.0000 for station, +0.0006 for
disc brightness, +0.0003 for the Earth–Sun distance and −0.0009 for the solar B0 angle (Meeus,
1998), against −0.0022 for year (above) and about −0.0005 for frames with a bright sky halo. The
axes overlap, since year, halo and station are correlated; taken together the test set is about
0.003 harder than the local estimate, and on no axis we could measure does it leave the training
distribution. Two test frames lie above the highest high-frequency energy in training (2069 and
1943 against 1571), none below the lowest.

| hf quartile | hf range | training images | test images | test / training | out-of-fold PQ [95 % CI] | recall |
|---|---|---|---|---|---|---|
| Q1 | 179–509 | 177 | 33 | 0.73 | 0.4622 [0.4436, 0.4806] | 0.620 |
| Q2 | 509–587 | 177 | 48 | 1.07 | 0.4507 [0.4317, 0.4678] | 0.601 |
| Q3 | 587–725 | 176 | 57 | 1.27 | 0.4477 [0.4326, 0.4632] | 0.597 |
| Q4 | 725–1571 | 177 | 42 | 0.93 | 0.4151 [0.3974, 0.4325] | 0.539 |

: Training and test images by quartile of disc high-frequency energy, with training edges, and the
submitted system's out-of-fold PQ in each (image bootstrap). The test set is lighter than training
in the quartile where the system is strongest and slightly lighter in the one where it is weakest,
and the two effects cancel. §4.7 measures the same decomposition at the unbagged operating point.

The inspection found no failure type that local validation does not already show. No submitted
mask lies beyond ±70° heliographic longitude, where the annotation protocol does not mark
filaments, and the longitude profile of the kept masks matches the out-of-fold profile (91.3 %
against 91.6 % within 50°). The frames carry the defects of a ground-based network (overexposure,
doubled limb rings, cloud, brightness gradients, vignetting, a disc darker than the training
range), and none produces a burst of false positives; frames with a disc darker than the training
10th percentile are 12.8 % of the test set against 10.0 % of training, and score higher locally
(0.4634 against 0.4411). The one recurring loss is long, well-defined filaments left just below the
keep threshold, mostly near the limb or at high latitude. Out of fold, admitting the long
high-latitude members of that band is worth +0.0009, of which +0.0003 is not already admitted by
the marginal-band cascade of the prepared alternative submission (§3.6). The inspection produced no change
to the submission.

**Expected score on the hidden labels.** The public leaderboard has displayed a score about 0.04 to
0.05 below the local estimate for every version of the pipeline: −0.050 and −0.049 for two early
submissions that had neither the learned selector nor the five-model detector ensemble, and about
−0.043 for the final one. An offset that does not depend on the pipeline cannot come from a mismatch
between how the pipeline is validated and how it is deployed, so we asked what the unlabelled test
images can say about it [`src/eda/e219_test_generalisation.py`]. Three measurements bound the
answer. First, the test images are statistically indistinguishable from the training images: a
classifier given every per-frame quantity of this section (disc high-frequency energy, disc level
and spread, limb width, sky halo, year, month, station) separates the two sets at a cross-validated
AUC of 0.534, and reweighting the out-of-fold score by the resulting density ratio gives 0.4395
(95 % interval 0.425–0.454, effective sample 249 of 707 images). Image composition therefore accounts
for about 0.004 of the offset. Second, the annotator is a large source of variation that no image
property reveals. Scoring the same submitted masks on the same images against different annotators,
an annotator's deviation from the image mean ranges from −0.036 to +0.038 (standard deviation 0.028
over 14 annotators with at least 15 shared images), and two annotators of the same batch differ by
0.07 on identical images. A test set whose annotations lean towards the stricter annotators would
show the observed offset for every pipeline alike. Third, the public and private halves are random
halves of 180 images: on test-sized subsets of the out-of-fold images, the private score differs
from the public one with standard deviation 0.018. Combining the composition estimate with the
public reading under a Gaussian model of a shared shift gives an expected test-wide score of 0.403
to 0.418, depending on the assumed spread of that shift (0.03 to 0.01). Our best estimate of the
private score is therefore 0.40 to 0.42, and differences of up to 0.02 between systems on the
public half do not determine their order on the private half. The per-image annotator counts that
the public MAGFiLO release would reveal for the test images were not used in this estimate.

### 2.4 Validation protocol

We use GroupKFold by physical image (five folds): all annotator entries of an image share a fold,
so no multi-copy leakage crosses the split. The folds are nearly perfectly stratified by station
(fold 0 holds 24/25/25/25/22/21 of the six stations, and the other folds differ by at most two
images in any station) and by contrast (fold means 37.8–39.3), so fold-0 numbers generalise across
folds. We report pooled five-fold cross-fitted PQ for every adopted decision; the unbagged
operating point at which §4 is measured reads 0.4421, and the submitted, bagged selector 0.4431. Local validation ranked public leaderboard results correctly in
six paired observations (local 0.3551 → leaderboard 0.28 through local 0.4286 → leaderboard 0.38).
The offset is ≈−0.05 only between like-for-like pooled measurements (−0.0500 and −0.0486 on the
two such pairs); the early submissions based on fold 0 alone show −0.060 to −0.076, so the offset
is not a constant and was never treated as one [`docs/21_LEADERBOARD_CALIBRATION.md`]. One
suspected cause was not distributional: the out-of-fold pool uses one detector per fold, whereas the
submission uses an ensemble of five. We later measured the consequence directly. Because only the model that did not
see an image may propose candidates for it out of fold, 74 % of local candidates carry a single
detector vote against a median of four at test time, and the per-image mean of that count alone
separates local from test images at AUC 0.996. There is therefore no common support on which to
reweight local validation towards the test distribution (Shimodaira, 2000). The mismatch can,
however, be run backwards on labelled data: scoring the held-out folds with their detector features
transported onto the test marginal changes pooled PQ by +0.0000, because the selector barely uses
the features that shift most [`src/eda/transfer_instrument.py`, `src/eda/mismatch_cost.py`]. A
second difference was found only when the system was reproduced: the test-time detector
candidates are made with NMS 0.05, the out-of-fold ones with 0.5 (§3.2). Read through the
out-of-fold selectors on all five folds, 0.05 costs −0.0016 [−0.0037, +0.0006] against 0.5, inside
the ±0.003 floor described below [`src/eda/nms_sweep_5fold.py`]. We
stopped relying on the offset once our submissions crossed the leaderboard's
two-decimal display step; from then on the leaderboard was used only through the **ordering** of
our own submissions, which the API returns on the unrounded score and which therefore resolves
differences that the displayed value hides [`src/eda/board_ordering.py`].

Two properties of the folds were verified rather than assumed. First, differences on which a
decision rests are assessed with a paired image bootstrap against a pre-registered band of ±0.003
whose provenance is measured: a same-recipe seed replicate of one ensemble member moves the
reference by ±0.001, the bootstrap's own half-width on 707 images is ≈0.003, and gains below that
band have historically failed to transfer to the leaderboard. Second, the folds are unbiased by
temporal proximity despite being random over images: 31 % of training images (219 of 707) have a
cross-fold neighbour within 24 h, yet images with such a neighbour score 0.4420 against 0.4422 for
those without, so there is no near-duplicate inflation
[`src/eda/_near_duplicate_inflation.py`].

A third property was measured late, and it changes how the second must be read. The bootstrap
resamples images within one fixed partition, so it cannot see the variance that comes from the
partition itself. Re-reading one selector variant on five partitions of the same 707 images (the
production folds, three reshuffled assignments and one balanced on annotator count and contrast)
gives deltas of +0.0001, −0.0000, +0.0032, −0.0005 and −0.0004: mean +0.0005, standard deviation
0.0015, while the baseline's own PQ varies with standard deviation 0.0008 only
[`src/eda/fold_design_audit.py`]. The spread of one variant's delta across partitions (from −0.0005
to +0.0032) is as large as the bagged form of the same feature block's reading on the production
partition, +0.0022, whose bootstrap interval excluded zero. A single-partition interval is therefore a lower
bound on the uncertainty of a difference, and a difference below about 0.003 cannot be resolved
locally by any amount of resampling within one partition. Balancing the folds does not help; the
remedy is to average over partitions. This is also the most economical explanation of why the
local magnitude of our sub-floor selector variants did not predict their order on the
leaderboard.

A fourth property concerns the training-stage comparisons of §4.7, which are paired runs validated
on the 63 images from 2019–2022. Five identical control runs, three of them with the same seed,
reached best validation pixel-Dice 0.6437, 0.6531, 0.6537, 0.6555 and 0.6601: a standard deviation
of 0.0059, because nondeterministic GPU kernels move a run about as much as a change of seed. A
single control–treatment pair therefore differs by about 0.008 by chance alone, and the ±0.01 band
we had applied to one pair is only about 1.2 standard deviations wide. The conditioning experiment
of §4.7 shows the consequence: its first pair read +0.014, and its two replicates −0.007 and
−0.019. Training-stage changes are therefore read over at least three seed pairs, and consistency
across the epochs of one pair is not counted as independent evidence, since the epochs are
successive points of the same two trajectories.

### 2.5 Data provenance

All ground truth used anywhere in the pipeline is provided by the competition. The public MAGFiLO
1.0 release (Ahmadzadeh et al., 2024), which we verified to contain annotations for all 180 test
images, was quarantined in the first week and never used for training or prediction. Our only
interaction with it is an ID-level overlap count and a label-value analysis restricted to training
stems, with no test geometry decoded (§8 gives the full declaration, including the organisers'
ruling and our compliance). External *unlabelled* GONG archive frames (confirmed as permissible by
the organisers) were used in the pretraining experiments reported in §4; the provenance of the final
model is stated where it is frozen.

One clarification shaped this section after the fact. The organisers' answer in discussion 729396
permits external data for obtaining additional information but states that **for the final test,
inference takes place solely on the H-alpha images**. Co-temporal SDO/HMI magnetograms and AIA 304
frames were therefore retrieved, registered and measured only as *teachers*, i.e. training-time
inputs to models whose knowledge could be transferred to an H-alpha-only student, and the submission
path refuses any checkpoint that reads them [`src/models/submit_v2.py`]. The measurements are
reported in §4 together with their outcome: the field carries genuine pixel-level information, and
none of it converts into PQ.
