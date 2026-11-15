# Final Report — Section 4: A Measured Characterisation of the Problem

*This part is the report's contribution beyond the pipeline itself: a decomposition, obtained by
direct measurement, of where the score is gained and lost, in which every closed axis is reported
with the experiment that closed it. Unless stated otherwise, all figures are deployment-faithful
five-fold out-of-fold readings over 707 images.*

## 4.1 The loss budget

The decompositions in this part are measured at the operating point of the unbagged selector
(pooled cross-fitted PQ 0.4421). The submitted selector averages five bootstrap copies of the same
learner and reads 0.4431 (§3.5). It keeps ten more candidates out of about four thousand, which can
move the shares below by a few tenths of a percentage point at most. At the unbagged operating point
the exact metric admits a complete accounting over all 8199 annotator copies:

| outcome of a ground-truth copy | share | interpretation |
|---|---|---|
| matched | 58.6 % | an accepted prediction reaches IoU > 0.5 |
| **shape band** | **24.9 %** | a candidate exists at 0.05 < IoU ≤ 0.5: correct object, wrong shape |
| selector | 12.6 % | a candidate above IoU 0.5 existed and was not kept |
| never proposed | 4.0 % | no candidate exceeds IoU 0.05 |

: Outcome of every ground-truth annotator copy at the production operating point.

Reference points for the same metric: a single human annotator, scored against the others, reaches
0.3399, and a majority-consensus mask constructed from every annotator reaches 0.6414. The system
is already well above a single annotator, and the rest of this part decomposes the remaining
distance to the consensus oracle.

Two properties of the budget shape everything that follows. The loss is concentrated where most of
the ground truth lies: 2827 of the 3397 unmatched copies (83.2 %) are inside r < 0.8, so the limb,
where the loss *rate* is highest, holds a sixth of the loss *count*. And the unmatched copies are
predominantly ones that only one annotator drew. For each copy in an image with more than one
annotator we ask whether any other annotator drew a matching filament at IoU > 0.5:

| outcome | copies | drawn by no second annotator |
|---|---|---|
| matched | 3033 | 18.1 % |
| selector | 537 | 59.4 % |
| shape band | 1268 | 66.5 % |
| never proposed | 155 | 92.3 % |

: Share of copies that no second annotator of the same image drew, by outcome, in images with more
than one annotator [`src/eda/error_atlas.py`].

What we match is mostly what annotators agree on, and what we miss is mostly what they do not. This
does not make every miss irreducible (§4.3 isolates the ones that are not), but it sets the scale on
which any recovered miss has to be priced.

## 4.2 Recognition versus shape

![*Left:* SQ, RQ and PQ by radius band, over all 8199 annotator copies scored out of fold. Mask
quality is flat at 0.65–0.68 from disc centre to far limb; the entire radial structure of PQ is
recognition. The dotted line is the rate at which two annotators of the same image agree with each
other (0.629), which our matched masks lie *above*. *Right:* inside the disc (r 0.2–0.7), the
counterfactual value of a perfect stage, split by contrast. Faint filaments lose 0.39 PQ to
recognition and 0.14 to shape; for bright filaments the ordering is reversed.
[`src/eda/report_figures.py`, built from `outputs/eda/pq_decomposition.parquet`]](figures/fig1_rq_sq.png)

PQ factorises as SQ × RQ, which makes two counterfactuals exact: perfect shapes would move PQ to
RQ, and perfect matching would move it to SQ. Measured per radius band:

| band | n | SQ | RQ | PQ | loss from matching | loss from shape |
|---|---|---|---|---|---|---|
| centre r < 0.2 | 358 | 0.672 | 0.666 | 0.447 | 0.225 | 0.219 |
| disc 0.2–0.7 | 5350 | 0.678 | 0.672 | 0.455 | 0.223 | 0.216 |
| outer 0.7–0.8 | 1463 | 0.679 | 0.657 | 0.446 | 0.233 | 0.211 |
| limb 0.8–0.9 | 937 | 0.683 | 0.557 | 0.380 | 0.303 | 0.177 |
| far limb > 0.9 | 91 | 0.649 | 0.093 | 0.060 | 0.589 | 0.033 |

: SQ, RQ and PQ by radius band (in units of the disc radius), with the loss attributable to
matching (SQ − PQ) and to shape (RQ − PQ).

**SQ is flat at 0.65–0.68 everywhere, including the far limb.** Mask quality does not depend on
position; every difference in PQ across the disc is due to recognition. Inside the disc the two
losses are equal, and at the limb recognition accounts for 63 %.

The comparison with human agreement can be made on the same copies rather than on averages. On the
2483 matched copies that a second annotator also drew, our mask agrees with the annotator at median
IoU 0.683 (mean 0.678), and the second annotator agrees with the same annotator at median 0.635
(mean 0.638). The paired difference has median +0.050 (mean +0.048), and our mask is the closer of
the two in 76.2 % of copies. The shape of a matched mask therefore sits above the reproducibility of
the label it is scored against, which is why every attempt in this project to *correct* a mask has
failed (Appendix A.9): a model trained to improve on that label is trained on the disagreement
between annotators.

Stratified by filament type inside the disc band, the balance *inverts*:

| population | n | RQ | SQ | the loss is mostly |
|---|---|---|---|---|
| faint (contrast ≤ 0.08) | 1012 | **0.375** | 0.620 | recognition (73 %) |
| bright (> 0.16) | 834 | **0.808** | 0.718 | **shape** (62 %) |
| thin (≤ 6 px) | 1137 | 0.574 | 0.644 | recognition |
| short (≤ 80 px) | 1066 | 0.489 | 0.691 | recognition |
| long (> 220 px) | 1509 | 0.721 | 0.680 | shape |
| Unidentifiable chirality | 1901 | 0.571 | **0.688** | recognition |

: Recognition and segmentation quality by filament population inside the disc band.

The Unidentifiable chirality class states the pattern most clearly: its masks are the *best* we
draw (SQ 0.688, the highest of the three classes) while its recall is the lowest (0.571 against
0.718/0.724). Those filaments are not drawn badly; they are not found.

Of the 2090 unmatched copies in the disc band, 27.9 % had a pool candidate above IoU 0.5 that the
selector rejected, a further 25.3 % have a candidate at 0.40–0.50, and only 9.1 % were never
covered by any candidate. The misses are near-misses rather than invisible filaments.

## 4.3 The addressable error population

A model fitted to predict whether a copy will be missed from measurable properties reaches AUC
0.804 from physical properties alone (area, length, width, contrast, radius, latitude, crowding,
elongation), 0.658 from annotator agreement alone, and 0.841 from both. This contradicts an
explanation on which the project had relied: the misses are *not* predominantly the copies about
which annotators disagree.

Isolating the copies that admit no such explanation, namely those drawn by every annotator of the
image with pairwise IoU ≥ 0.6, gives 1563 copies, of which we miss 166. Recovering them at IoU 0.7
would be worth +0.0158 PQ. Their profile relative to matched copies is the main finding:

| property | unambiguous miss | matched | ratio |
|---|---|---|---|
| skeleton length | 121.0 | 161.4 | 0.75 |
| contrast | 0.104 | 0.126 | 0.83 |
| width | 7.9 | 8.6 | 0.92 |
| **crowding** (copies within 200 px) | **3.0** | **1.0** | **3.00** |

: Profile of the unambiguous missed copies against matched copies.

The missed copies are short, faint and, above all, **crowded**. Crowding had not appeared as a
variable in any earlier diagnostic. By mechanism, 63 of the 166 are selector rejections just below
the gate, 52 have a candidate at median IoU 0.44 that no component shapes correctly, 34 are merged
into a neighbour, and 17 are fragmented.

![The error atlas. *Left:* the outcome of each of the 8199 annotator copies. Only 4.0 % were never
proposed by any lane; a quarter lie in the shape band and an eighth were proposed and then rejected
by the selector. *Centre:* the 166 unambiguous missed copies, profiled against matched copies.
Crowding is the distinguishing property at 3.00×; every other property is within 25 % of the
matched value. *Right:* the same 166 copies split by mechanism, each valued in PQ at the deployed
denominator. The whole addressable population is worth +0.0158, and no single mechanism within it
is worth more than +0.0060. [`src/eda/report_figures.py`, built from
`outputs/eda/error_atlas.parquet` and `outputs/eda/crowding_mechanism.parquet`]](figures/fig2_atlas.png)

A similar restriction changes what the shape band is. Over the whole band the symmetric difference
is dominated by lateral structure and drawn width, as the summary reports. Among the 425 shape-band
copies that two annotators drew independently at
mutual IoU > 0.5, 87 % of them inside r < 0.8, our mask has the right *width* and half the
*length*:

| measure (median, deployed mask against the copy) | value |
|---|---|
| IoU | 0.409 |
| precision, \|P ∩ G\| / \|P\| | 0.766 |
| recall, \|P ∩ G\| / \|G\| | 0.506 |
| skeleton-length ratio | 0.723 |
| width ratio | 1.038 |

: Geometry of the 425 consensus-confirmed shape-band copies [`outputs/eda/_deficit_kind.py`].

The missing half is not held by another candidate (a median of 1.1 % of it is covered by any other
pool candidate), and in 75.4 % of these copies the ensemble's probability field is essentially zero
over it: mean probability 0.153 in the gap against 0.947 over the part we find. In the raw image the
gap is 0.29 times as dark as the found part, relative to the local background, and in 15.7 % of
copies it is as bright as the background itself. The mechanism is therefore that the annotator
draws a continuous ribbon over a filament that the model sees as a chain of dark cores. Growing the
mask does not repair this: a two-pixel dilation lowers the median IoU of these copies from 0.409 to
0.400 and their precision from 0.766 to 0.636, because it reaches the missing length only by
absorbing background. Lowering the extraction threshold, which lengthens masks along the field
instead, is reported in Appendix A.8.

## 4.4 The selection stage is at its information limit

Three independent measurements indicate that the selector cannot be improved by better selection.

**Calibration.** The positive rate rises monotonically with the selector score (0.026 below 0.30
to 0.899 above 0.80), and the mean copy-weighted IoU crosses the metric's break-even value (0.361)
exactly at the deployed gate of 0.60. In the region occupied by the 63 addressable rejections
(small, crowded candidates with scores of 0.45–0.60) the mean value is 0.264 against a break-even
of 0.361, so accepting them would be a net loss, and a selector refitted *only* inside that region
separates worse than the global one (AUC 0.541 against 0.589).

**Transfer.** The two oracles on which the project relied are reordering the candidate pool
(+0.033) and perfect selection from the pool (+0.069), both with the choice made and scored on the
same annotator. When the choice is made on annotator A and scored on annotator B, they transfer at
−25 % and −32 %: negative, not merely small. The apparent headroom is one annotator's judgement,
and acting on it reduces the score. A learned router motivated by that oracle measured −0.0065.
Two oracles do transfer, and both are small: the per-candidate contour level at 15 %, and choosing
the best ensemble member per image at +23 %, the only positive transfer we measured
[`outputs/eda/ranker_ceiling_null.json`, `outputs/eda/router_oracle.json`]. A third selection
oracle, measured later on the frames where recall is lowest, reproduces the pattern: +0.0498 within
an annotator, a transferred fraction of −0.365 across annotators (§4.7).

**Saturation of inputs.** Three feature families were added and measured: image-level context
(+0.0001), magnetic polarity-inversion summaries (tied), and threshold-stability descriptors
(−0.0006). Two loss reformulations, including the one the metric literally requires, measure
−0.0009 and −0.0016, and input augmentation measures −0.0017.

## 4.5 The capture ceiling

The central empirical result of this project is that *gains at the level of an oracle or of an
intermediate property do not convert into realised PQ.* Eighteen structurally distinct mechanisms
were built and measured. The five below are the most instructive, because in each the intermediate
property genuinely improved and was verified against a control before PQ was measured:

| mechanism | what improved | realised |
|---|---|---|
| query-lane third source (Mask2Former) | reaches 29.0 % of the ground truth that nothing else shaped (120 of the 414 truly unreachable rows, 44 of them beyond every other source) | +0.0020 over five folds |
| lane identity for the selector | pilot on one fold's inner CV +0.0030 | +0.0003 under the full protocol |
| magnetogram as a teacher | pixel AUC +0.0146 over a matched control, 0 when shuffled | +0.0015 / +0.0003 |
| transformer ensemble member | error correlation 0.871, the lowest measured in this project | −0.0052 against a seed control on fold 1 |
| two-lane mask refiner | learns the target; the loss of the regeneration run falls 1.29 → 0.37 | −0.015 at best, in three formulations |

: Five capture failures. In each, the intermediate quantity improved and was verified against a
control, but realised PQ did not follow.

<!-- registration note: outputs/eda/magnetogram_alignment.json is a 12-frame pilot that fits a
     -30 deg rotation and is SUPERSEDED by the full sweep (ledger 26c), which finds correlation
     +0.186 at 0 deg against ~+0.044 at every other angle. Both registration runs used 0 deg. -->

The common signature is a measurable improvement in the intermediate quantity with no change in
matched instances. The metric matches at IoU 0.5, and none of these mechanisms moves candidates
across that threshold.

![Ten of the eighteen capture failures, on a common axis. The blue bar is the intermediate quantity
that motivated each mechanism (reachability, an oracle's value, a pixel-level AUC, an error
correlation, a crop-level fit), and the red bar is the PQ it returned under the deployment-faithful
protocol. The grey band is the ±0.003 measurement floor, whose provenance is given in §2.4. None of
the ten clears it, five are negative, and the two largest intermediate gains (the transformer member
at 0.055 of error decorrelation, multi-threshold extraction at +0.032 of reach) are among the worst
realised. [`src/eda/report_figures.py`, values from `docs/15_ABLATION_LEDGER.md`]](figures/fig3_capture.png)

## 4.6 Axes closed by measurement

Each of the following was a candidate explanation for the remaining gap and was closed by a
controlled experiment (full table in Appendix A): **capacity** (nine encoders of 13 M–93 M, three
decoder widths, UNet++, UPerNet, two transformer families; the smallest model remains the best);
**resolution** (a native-2048 tiled lane with blended overlaps, 0.4277 against 0.4286);
**sampling** (anti-aliased downsampling −0.0000, bilinear upsampling +0.0006, label downsampling
measured to be lossless); **pretraining** by two routes (GONG distillation and in-domain
masked-image reconstruction, both indistinguishable from a seed replicate); **new modalities**
(co-temporal magnetograms, AIA 304 and neighbouring frames, which carry genuine pixel-level
information but yield no PQ); **the limb** in every direction, including an exact centre-to-limb
augmentation that rotates the solar sphere; **augmentation** (elastic shape jitter, calibrated to
the annotators' own agreement of 0.629, and a compression of filament depth towards the
chromospheric background, which targets the fainter filaments of 2019–2022); **input
normalisation** (a high-pass channel with a fixed gain and with a per-frame gain, the latter
removing the frame-to-frame contrast regime almost entirely); **thresholds** (clustering IoU,
fusion vote and detector NMS, each flat within 8 of 1510 reachable copies, the blend weight, and
the semantic extraction threshold below 0.2); **image quality** (atmospheric seeing and
solar-cycle phase, both measured from the frame itself, neither of which predicts PQ, §4.7); and
**transduction**, predicted in advance to be null by the measured absence of train/test shift, and
confirmed.

Three further tests addressed the extent of the kept masks after selection, with the keep decisions
held fixed so that no selector needed refitting. Removing each mask's one-pixel boundary ring is
worth +0.0117 on the current pipeline, and restoring the ring only on thin masks is flat
(−0.0001, 95 % interval −0.0004 to +0.0001, threshold chosen on four folds and read on the fifth)
[`src/eda/e215_conditional_unshrink.py`]. A perfect gap closer confined to each kept mask's bounding
box would lift 412 shape-band copies above IoU 0.5 against the annotator who drew them, but when the
fill is decided on one annotator and scored on another only 6 % of that gain survives; filling
beyond the box, along the filament, survives at 25 %, which is the length deficit of §4.3
[`src/eda/e216_inbox_closer_ceiling.py`, `src/eda/e216b_inbox_closer_transfer.py`]. Finally, the
detector's own sub-threshold mask probability, which the binarised output discards, does carry part
of that length: on copies drawn by two annotators, the detector's binary mask already covers 45 % of
the pixels our fused mask misses and probability ≥ 0.3 covers 61 %. But only 41 % of the added
pixels belong to the filament, and extending the kept masks with that field costs between 0.031 and
0.137 PQ on fold 0 [`src/eda/e217_yolo_soft_masks.py`, `src/eda/e217b_soft_extension.py`]. The
fusion vote and the boundary erosion are what separate that length from the excess width that
accompanies it.

A separate class of negative result deserves mention because it constrains strategy: gains below
the measurement floor do not accumulate. Three changes from different mechanisms, each between
+0.0003 and +0.0011 alone, measure −0.0004 together against a member-count control; six last-epoch
checkpoints, each +0.0007 alone, measure −0.0024 together.

## 4.7 Regime sensitivity: observed, but not a recoverable opportunity

The sections above treat the images as exchangeable. They are not: the pipeline's recall depends on
an image regime that can be measured from the frame itself, without metadata. We measured four
per-frame quantities [`outputs/eda/_image_quality.py`]: the 10–90 % transition width across the
solar limb, which is a step in truth and therefore a seeing proxy independent of the filaments; the
high-frequency energy of the disc (the variance of the Laplacian inside r < 0.85, written *hf*);
the disc mean; and the disc standard deviation, which follows the solar cycle (10.5 at the 2014–15
maximum of cycle 24, 8.2 in 2022; Hathaway, 2015). Seeing improves over the archive rather than
degrading (median limb width 84.5 px in 2011, 47.2 px in 2022).

Only one of the four predicts per-image PQ: *hf* (Spearman −0.140, p = 2·10⁻⁴). Limb width
(+0.050, p = 0.19), disc sd (+0.023, p = 0.54) and disc mean (+0.024, p = 0.52) do not, which closes
atmospheric blur and cycle phase as explanations and, with them, deblurring, blur augmentation and
a regression on cycle phase. *hf* is not crowding under another name: its partial correlation with
PQ is −0.135 given the candidate count, −0.139 given the ground-truth count, −0.098 (p = 0.010)
given the selector's own density features, and −0.150 (p = 7.6·10⁻⁵) given all of these and the
other quality axes together [`outputs/eda/_hf_anatomy.py`].

| hf quartile | median hf | SQ | RQ | FP / GT | recall | PQ |
|---|---|---|---|---|---|---|
| Q1 | 471 | 0.683 | 0.679 | 0.210 | 0.623 | 0.464 |
| Q2 | 544 | 0.682 | 0.661 | 0.221 | 0.602 | 0.450 |
| Q3 | 651 | 0.674 | 0.665 | 0.200 | 0.597 | 0.448 |
| Q4 | 890 | 0.676 | 0.616 | 0.212 | 0.540 | 0.416 |

: Decomposition of PQ by quartile of disc high-frequency energy. SQ and the false-positive rate are
flat; the whole degradation is recall.

The aggregate degradation is visible in PQ itself; what is not visible without decomposition is its
nature. Mask quality and precision do not change across the quartiles, and the loss is entirely
recall (0.623 to 0.540). By stage, the rise in unmatched copies from Q1 to Q4 is 0.030 at the
selector (candidates above IoU 0.5 that were rejected, 0.114 to 0.144), 0.021 in the shape band and
0.012 among copies never proposed. About half of the rise is therefore *observed* at the
selection stage. Whether it is *recoverable* there is a separate question, answered below.

The same signature appears along the archive's time axis [`outputs/eda/_late_years.py`]. The 60
images from 2019–2022, the minimum between cycles 24 and 25 and the rise of cycle 25, score 0.3875
against 0.4453 for the rest (paired image bootstrap: −0.0578, 95 % CI [−0.1081, −0.0121],
P(Δ < 0) = 0.992). The deficit survives each control we applied: at matched disc sd it is −0.108
(45 images), reweighting the earlier images to the late years' ground-truth density accounts for
0.0028 of it, and seeing is better in those years, not worse. Their filaments are shallower (ring
contrast 0.078 against 0.114), and on the copies that two annotators drew independently the
selector discards 16.3 % in the late years against 6.6 % before them. The test set gives 2021 and
2022 about twice their training weight (§2.3); reweighting the local score onto the test set's year
mix lowers it by 0.0022 through composition alone.

**Interventions.** Each was read with the deployment-faithful protocol and judged against a rule
fixed before the run:

| intervention | stage | reading | rule | outcome |
|---|---|---|---|---|
| *hf* as a selector feature | selection | +0.0002, CI [−0.0013, +0.0016], 3/5 folds | ≥ +0.003 | below |
| 20 frame-relative evidence features (percentile and z-score within the frame) | selection | +0.0009, CI [−0.0009, +0.0027], 4/5 folds; Q4 +0.0026, recall +0.0023 | ≥ +0.003 | below |
| detector features transported to the test marginal | selection | +0.0000 on labelled data | ≥ +0.003 | below |
| extraction threshold 0.20 → 0.10, selector refitted | generation | +0.0010, CI [−0.0019, +0.0038] | ≥ +0.003 | below; lower on the leaderboard |
| filament-depth compression (augmentation) | training | late-year pixel-Dice −0.0081 | beyond ±0.01 | within band |
| high-pass input channel, fixed gain | training | late-year pixel-Dice −0.0026 | beyond ±0.01 | within band |
| high-pass input channel, per-frame gain | training | late-year pixel-Dice −0.0056 | beyond ±0.01 | within band |
| keep threshold set by *hf* (0.55 above its training median, 0.575 below) | selection | +0.0019, CI [−0.0006, +0.0044], 4/5 folds; one lowered threshold +0.0018 | ≥ +0.003 | below; lower on the leaderboard |
| every encoder skip conditioned on *hf* (FiLM) | training | late-year pixel-Dice +0.0142, −0.0072, −0.0189 over three seed pairs (mean −0.0040) | ≥ +0.01 on the mean of three pairs | below |
| training images sampled in balance across cycle phase | training | pixel-Dice on held-out late years −0.019, on earlier years −0.004 | late ≥ +0.01, earlier ≥ −0.005 | below |

: Interventions against the regime dependence, each with its pre-registered promotion rule.
Selection rows are pooled five-fold PQ against the deployed selector; training rows are paired
runs validated on the 2019–2022 images, except the last, which is validated on half of them with
the other half in training.

Of the first seven rows, the frame-relative block is the only one whose gain concentrates where the
mechanism predicts (roughly 72 % of its pooled +0.0009 comes from Q4, through recall). We had also pre-registered a quartile
criterion for it, that the Q4 gain should be at least twice the best other quartile, and it failed
at 1.4×. We record that criterion as failed rather than as evidence against the mechanism, because a
ratio of two noisy quartiles is not a principled test of an interaction. The per-frame gain channel
is the clearest negative: it reduces the contrast regime carried by the input from 2.42× (raw
image, between the twelve lowest- and highest-contrast frames) to 1.02×, and the late-year Dice
does not improve.

**The ceiling, and why it is not headroom.** The decisive measurement is the size of the
opportunity at the selection stage [`src/eda/hf_oracle_ceiling.py`,
`src/eda/hf_oracle_transfer.py`]. Applying the metric's break-even rule (eq. 5, §4.8) with each candidate's
*realised* value, i.e. admitting every rejected candidate that would have paid, raises PQ by
+0.0314, +0.0357, +0.0372 and +0.0498 in Q1–Q4. The ceiling grows with *hf*, as does the number of
rejected candidates that would have paid (142 to 259), exactly as the mechanism predicts; taken at
face value, the frame-relative block captures 5 % of it. The oracle, however, decides with values
realised against the annotators who drew that image. On the 296 images with at least two annotator
entries we split the entries into halves A and B, decided break-even on A and scored on B. The
deployed gate scores 0.4542, the within-annotator oracle 0.5272, and the transferred oracle
**0.4276**: a transferred fraction
$(\mathrm{PQ}_{\text{transfer}} - \mathrm{PQ}_{\text{floor}}) / (\mathrm{PQ}_{\text{within}} - \mathrm{PQ}_{\text{floor}}) = -0.365$.
The rejected-but-paying candidates pay only against the annotator who drew them that way, and a
decision rule learned from one annotator lowers the score against another.

**Three interventions after the ceiling.** The last three rows of the table were designed after
this measurement, each on a different part of the pipeline.

*A keep threshold set by the regime.* The selector's score does not mean the same thing in every
regime. A candidate scored 0.55–0.60 has a mean realised value (eq. 5) of −0.009 in the quietest
quartile and +0.076 in the busiest, so the break-even score is lower in busy frames
[`src/eda/regime_gate.py`]. Thresholds chosen by nested cross-validation, split at the training
median of *hf*, read +0.0019, with recall +0.052 in the busy half. Every choice made, in each of
the five folds and on each half of the annotators, places the busy threshold lower, and this is the
only intervention whose decision transfers across annotators with a positive sign (0.291). A single
lowered threshold chosen the same way, however, reads +0.0018: the regime adds +0.0001 to what a
global move of the gate already captures, and the leaderboard ordering places the submission below
the final submission, as it placed the global move before it.

*Conditioning the segmentation network on the regime.* Instead of removing the regime from the
input, the network is given it: every encoder skip is scaled and shifted per channel by a small
network of log *hf* (feature-wise linear modulation; Perez et al., 2018), with *hf* computed from the
input itself and the modulation initialised as the identity, so that a pair differs only in the
conditioning [`src/models/hf_film.py`]. The first pair read +0.0142; two replicate pairs read
−0.0072 and −0.0189, a mean of −0.0040 over three seeds. Even in the first pair the gain did not grow
with *hf* (it improved 26 of the 63 validation frames), and §2.4 shows why one pair could not
decide the question.

*Balancing the training set across the solar cycle.* The training set is balanced by station and by
*hf* quartile, but not by cycle phase: 2019–2022 hold 8.9 % of the images and 3.9 % of the
filaments. Sampling images with weight inversely proportional to the square root of their era's
size draws the late era about three times as often (4.7 % to 13.9 %). On a split that places half
of the late-year images in training, this lowered pixel-Dice on the held-out late-year images by
0.019 and on earlier images by 0.004 [`data/splits/folds_late.parquet`]. It is a single pair, but
the difference is more than twice the pair noise of §2.4 and holds over epochs 15–29 (mean −0.014):
showing a few late frames more often fits those frames rather than the era.

We conclude that **regime sensitivity is real, and that observed regime sensitivity is not the same
thing as a recoverable performance opportunity.** None of the tested selection, augmentation,
input-normalisation, conditioning and sampling interventions passed its pre-registered promotion
threshold. The scope of that statement is the one tested: other architectures, self-supervised
encoders, temporal context and training objectives are untouched, and nothing here shows that they
could not close the recall gap at the generation stage. What the measurements do show is that the
part of the gap visible to a per-candidate selector is annotator-specific, that the image-level
part it can see is captured as well by a global move of the gate, and that the part at the
generation stage did not respond to the input, conditioning, sampling and augmentation changes
available to this architecture on 707 images.

## 4.8 Set-level optimisation and open hypotheses

An earlier draft of this subsection listed three hypotheses. Formalising the second one closed it,
so it is reported here as a result rather than as a plan.

**A set-level PQ objective cannot outperform the per-candidate rule, and this can be proved.** The
proposal was a differentiable PQ surrogate over candidate *sets* instead of per-candidate value
regression, motivated by the global denominator of PQ: keeping one candidate should change the value
of the others. It does not. Let each image have $c$ annotator copies, $k$ kept predictions and $G$
ground-truth rows, and let $K$ denote the kept set. The scorer's own accounting gives
$\mathit{fp} = kc - \mathit{tp}$ and $\mathit{fn} = G - \mathit{tp}$, so the denominator summed over
images is

::: equation
$$D = \mathit{tp} + \frac{kc - \mathit{tp}}{2} + \frac{G - \mathit{tp}}{2} = \frac{1}{2}\sum_{i \in K} c_i + \frac{1}{2}G,$$

[(3)]{.eqno}
:::

in which every $\mathit{tp}$ term cancels. As a function of the kept set, PQ is therefore

::: equation
$$\mathrm{PQ}(K) = \frac{\sum_{i \in K} v_i}{\tfrac{1}{2}\sum_{i \in K} c_i + \tfrac{1}{2}G},$$

[(4)]{.eqno}
:::

where $v_i$ is the candidate's copy-weighted IoU sum. This is a *linear-fractional* function of an
indicator vector and hence separable. Dinkelbach's theorem (Dinkelbach, 1967) gives its maximiser
in closed form as the per-candidate threshold

::: equation
$$\text{keep } i \iff v_i > \tfrac{1}{2}\, c_i \, \mathrm{PQ}^{*},$$

[(5)]{.eqno}
:::

which is exactly the break-even rule derived from the metric in §1.2 and deployed at the gate.
There is no coupling term for a set-level learner to exploit.

Four numerical checks confirm this [`src/eda/setlevel_pq_separability.py`]. The closed form
reproduces the exact scorer on the deployed kept set to within 3·10⁻⁵ PQ, so the one assumption
that could fail, namely that de-overlap between kept candidates changes their IoUs, has no effect.
The Dinkelbach set scores 0.5182, and local search from eight random restarts cannot improve it by a
single unit in the last place. Letting every image choose its own threshold, with perfect knowledge
of each candidate's true value, is worth +0.00002 over a single global threshold. Finally, the
check that explains why the idea appeared attractive: on the selector's *estimated* scores, a
per-image threshold oracle appears to be worth +0.038, a gain that consists entirely of the oracle
fitting estimation noise, since the same freedom with perfect information is worth nothing. It is
the ranking oracle of §4.4 in another form, and it would have transferred equally badly.

The remaining hypotheses, stated with their preconditions, are: (a) **more labelled frames**, since
every axis we measured is saturated at 707 images and the two quantities that still discriminate
(crowding and contrast) require more data; (b) an end-to-end objective aimed at **generation**
rather than selection, since the separability result implies that the IoU-0.5 threshold can only be
addressed by moving candidate masks across it, not by choosing among candidates more effectively,
and since §4.7 leaves the generation stage as the only place where the regime dependence has not
been shown to be annotator-specific;
and (c) annotator-conditioned prediction that models the distribution of conventions and emits its
median, which targets the population of §4.3, although our own transfer test indicates that most
of the apparent headroom there is one annotator's judgement.
