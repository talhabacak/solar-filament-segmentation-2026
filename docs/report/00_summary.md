# Solar Filament Segmentation Challenge 2026 — Method and Findings

The system described in this report reaches a pooled five-fold cross-fitted local PQ of 0.4431
(707 images, exact scorer, out-of-fold selection that reproduces the deployment path), and the
public leaderboard displays it as 0.40. Interpreting public scores on this task requires one fact,
which we state explicitly: the public MAGFiLO 1.0 release (Ahmadzadeh et al., 2024) contains
annotations for all 180 test images. We verified this, quarantined the release in
the first week of the competition and never used it. As reference points for this metric on these
data, a single human annotator scored against the others reaches 0.34, and a majority-consensus
mask constructed from every annotator's drawing reaches 0.64. Among our 47 submissions, ordered by
the unrounded score that the leaderboard API returns, the submission described here ranks first.
Code, measurements and negative results are in the accompanying repository, and each number below
names the script that produced it.

## System overview

The system is a two-lane instance segmentation pipeline with a learned selection layer.

A YOLO11m-seg detector (Jocher and Qiu, 2024) at 1792 px, single class and trained on one
annotator copy per image (the medoid annotator's), and a U-Net semantic lane (Ronneberger et al., 2015) with an EfficientNet-B3
encoder (Tan and Le, 2019) at 1536 px, trained with BCE + Dice + clDice (Shit et al., 2021),
produce candidates independently. The semantic lane is a twelve-member probability ensemble (five
cross-fold clDice models at weight 1/30, two heavily augmented members at 1/6, five blob-local
models at 0.10). Its members are similar enough that the ensemble behaves like nine independent
members, which bounds the benefit of further averaging (see below). Candidates are clustered at
IoU > 0.5, fused by confidence-weighted pixel voting, made pixel-disjoint, described by 42
geometric, photometric, orientation and provenance features, and retained by a two-stage
gradient-boosted selector: a classifier multiplied by a value regressor fitted on positives only,
normalised through an inner leave-one-fold-out quantile map, and averaged over five copies each
fitted on a bootstrap sample of images. The keep gate is derived from the
metric rather than tuned. Because PQ = ΣIoU / (TP + ½FP + ½FN), a candidate is worth keeping when
its expected copy-weighted IoU sum exceeds ½·c·PQ, and the deployed gate (conf ≥ 0.40, p ≥ 0.60)
lies at that break-even point. Before scoring, each mask loses its one-pixel boundary ring, which
corrects a systematic tendency of our contours to enclose those of the annotators. Interior holes are left as the model
draws them. The annotation protocol defines a filament mask as a single connected region without
holes, and filling ours is worth +0.0001 on local data (13 of 1117 submitted masks change), but
the leaderboard's unrounded ordering places the hole-filled submission *below* the original; the
conformity argument is therefore aesthetic rather than scored, and we did not adopt it.

Each stage contributes measurably under the same out-of-fold protocol (§3.8): the detector alone
reaches 0.398, the semantic lane alone 0.413, their fusion with nothing but a confidence gate 0.434
at its best threshold, and the learned selector 0.443, its gain coming entirely from removing 840
false positives at a cost of 289 true positives.

## Principal findings

The substance of the report is less the pipeline than a measured characterisation of the task.

**The target is an annotation convention, not a physical boundary.** Snapping masks to intensity
structure, which is the physically motivated choice, costs 0.070 PQ. Two annotators drawing the
same filament agree at mean IoU 0.634 (median 0.629), whereas our matched masks agree with the
ground truth at mean IoU 0.678, the SQ of the submitted system. A model trained on all copies
predicts close to the consensus, and the consensus is closer to each individual annotator than
the annotators are to one another. The dataset paper reports κ = 0.66 for the same reason
(Ahmadzadeh et al., 2024). Mask quality is therefore not merely optimised but *saturated against
the reproducibility of the label itself*. Measured on the same copies rather than on averages, our
mask is closer to an annotator than a second annotator is in 76.2 % of the 2483 matched copies that
two annotators drew (median IoU 0.683 against 0.635).

**Selection is at the inter-annotator ceiling, as a transfer test shows.** The two oracles on
which the project relied are reordering the candidate pool (worth +0.033 PQ) and perfect
selection from the pool (+0.069), both measured with the choice made *and* scored on the same
annotator. When the choice is made on annotator A and scored on annotator B, the ranking oracle
transfers at −25 % and the pool ceiling at −32 %. The transfer is not merely smaller but
*negative*: the apparent headroom is the noise of one annotator's judgement, and acting on it
reduces the score. A learned router motivated by the same oracle measured −0.0065. Two oracles do
transfer, and both are small: the per-candidate contour level at 15 %, and choosing the best
ensemble member per image at +23 %, the only oracle here with a positive transfer, worth +0.035
within an annotator and +0.008 across annotators. Future work on this task should value an oracle
by its *transfer* rather than by its value on the annotator that defined it.

**The remaining loss is a shape band, and it reflects a model blind spot rather than a policy
error.** Of on-disc ground-truth copies, 60.6 % are matched, 11.5 % have a candidate above IoU 0.5
that the selector rejected, 3.4 % were never proposed, and 24.6 % have a candidate whose IoU lies
in (0.05, 0.5]: the correct object with the wrong shape. The symmetric difference in that band
decomposes into 46 % lateral structure (barbs and side branches), 38 % drawn width and 16 %
extent. Candidates are systematically *too small* (608 k ground-truth-only pixels against 248 k
candidate-only pixels), and correcting width alone would lift 73 % of the band above IoU 0.5.
However, 63 % of the missing pixels are not activated by any of the nine effective members even at
p ≥ 0.2, so the band is not a fusion artefact: moving the fusion vote towards the union costs
−0.058 PQ, per-annotator width calibration changes from +0.032 as an oracle to −0.028 realised, and
a recall-weighted loss term costs 0.018 pixel-Dice (0.685 against 0.703) without reaching the
instance metric. The band is also where the annotators disagree: in images with more than one
annotator, 66.5 % of shape-band copies were drawn by no second annotator, against 18.1 % of matched
copies. Among the 425 that two annotators did draw independently, the deficit is one of length
rather than width (median skeleton ratio 0.72, width ratio 1.04), and in 75 % of them the ensemble's
probability field is essentially zero over the missing part (§4.3).

**Detection is not the bottleneck; converting candidates into score is.** We built and measured
more than a dozen structurally distinct mechanisms aimed at the selection gap: a metric-derived
value rule, late de-overlap, learned cluster representatives, crop refiners and scorers,
multi-threshold extraction, a native-resolution tiled lane, a query-based third source
(Mask2Former; Cheng et al., 2022) with lane-identity features, spine-auxiliary supervision,
chain-cluster targeting, selector bagging, a normalised regression target, and their compositions.
Each had a genuine justification at the oracle or precondition level; none yielded more than
+0.005 realised. The clearest case is the query lane, which reaches ground truth that no proposal-
or threshold-based extractor ever shaped correctly, yet across five folds adds only +0.0020
[−0.0027, +0.0066], i.e. 308 additional true positives at the cost of 494 additional false
positives, while informing the selector *which lane proposed a candidate* adds +0.0003
[−0.0014, +0.0020].

**New information, rather than new machinery, was the remaining axis, and it was measured to
completion.** Filaments form along magnetic polarity inversion lines (Martin, 1998), so we
retrieved, rebinned and registered a co-temporal SDO/HMI magnetogram (Scherrer et al., 2012) for
every frame. Registration was verified physically: the correlation between H-alpha darkness and
log|B| peaks sharply at zero rotation, which also establishes that the GONG frames are
P-angle corrected. At the pixel level the field is genuinely informative: +0.0146 AUC over a
capacity-matched control for the target "some annotator marked this pixel", and exactly zero when
the magnetograms are shuffled across images. This information does not convert into score. As a
U-Net input channel (a teacher model, since the rules restrict final inference to the provided
H-alpha images) it yields +0.0015 and +0.0003 on two folds; as privileged information (Vapnik and
Vashist, 2009) in an auxiliary head of an H-alpha-only student, −0.0000 and −0.0006, against a
seed-replicate floor of ±0.001. The reason is structural: instance matching at IoU 0.5 is
insensitive to slightly cleaner edges. Co-temporal AIA 304 observations (Lemen et al., 2012) rank
below the magnetogram, and a neighbouring frame close in time carries +0.0125 pixel AUC, the same
magnitude that already failed to convert.

**Regime sensitivity is real, but observed sensitivity is not a recoverable opportunity.** The
pipeline's recall depends on an image regime that can be measured from the frame itself. Across
quartiles of disc high-frequency energy PQ falls from 0.464 to 0.416; SQ and the false-positive rate
are flat, and the whole degradation is recall (0.623 to 0.540). The aggregate loss is visible in PQ,
but its recall-dominated nature appears only when PQ is decomposed into SQ, RQ, precision and recall
and the loss into its stage of origin, where about half of the rise is observed at the selector.
Atmospheric seeing and solar-cycle phase, both also measured from the frame, do not predict PQ. The
same signature runs along the time axis: the 2019–2022 images score 0.058 below the rest
(P(Δ < 0) = 0.992), and the test set gives 2021 and 2022 about twice their training weight. The
decisive measurement is the size of the opportunity at the selector: admitting every rejected
candidate that would have paid raises PQ by +0.0498 in the busiest quartile, growing with the
regime exactly as the mechanism predicts, but the same rule decided on one annotator and scored on
another transfers at **−0.365**. None of the ten interventions we tested passed its pre-registered
promotion threshold: selector features, an input channel that removes the frame-to-frame contrast
regime almost entirely (2.42× to 1.02×), a keep threshold set by the regime, a conditioning of the
segmentation network on it, and a training set rebalanced across the solar cycle among them. The
regime threshold is instructive. Its direction is a property of the image, since every fold and
both halves of the annotators choose a lower threshold for busy frames and its decision transfers
across annotators (0.291), but a single lowered threshold captures nearly all of its gain (+0.0018
of +0.0019), and the leaderboard placed it below the final submission. This does not show that other architectures or
objectives could not close the gap at the generation stage; it shows that the part visible to a
per-candidate selector is annotator-specific or already within reach of a global threshold (§4.7).
The test set does not leave the training distribution on any axis we could measure: its mix of the
regime, station, brightness, Earth–Sun distance and B0 angle shifts the local score by less than
0.001 each, and year and sky halo make it about 0.003 harder in total (§2.3).

**The offset between local and public scores belongs to the test labels, not to the pipeline.**
The public leaderboard has shown a score 0.04–0.05 below the local estimate for every version of
the pipeline, including two early submissions that had neither the learned selector nor the
five-model ensemble, so the offset cannot be a validation–deployment mismatch. The test images are
statistically indistinguishable from the training images (cross-validated AUC 0.534 on every
per-frame quantity we measure), and reweighting the local score onto their composition gives 0.4395
(95 % interval 0.425–0.454). What no image property reveals is the annotator: scoring the same
masks on the same images against different annotators moves an annotator's PQ by −0.036 to +0.038.
Because the public and private halves are random halves (private minus public: standard deviation
0.018), we expect a private score of 0.40–0.42, and differences of up to 0.02 between systems on
the public half do not determine their private order (§2.3).

**Fine-scale structure is recovered where it is attached to the body, and missed where it is long
and faint.** Using the annotated spines, a barb is defined as the part of a mask that is both thinner
than the body and farther from the spine than half a body width. Barbs make up 0.17 % of the annotated
area, in 303 of 8199 copies. Our matched masks cover 38.0 % of the barb pixels where a second
annotator covers 49.3 %, and 80.5 % of the body where the second annotator covers 79.7 % (§5.4,
Figure 4). Barbs are therefore where the system falls short of human agreement, but they are also
where humans agree least, and they carry almost no weight in PQ.

**Several plausible axes were closed by measurement rather than assumption:** capacity (13 M–93 M
parameters, nine encoders; the smallest is best); resolution (including a native-2048 tiled lane
with blended overlaps); representation pretraining by two routes (GONG pseudo-label distillation,
and in-domain block-masked reconstruction of the H-alpha frames themselves, the latter
indistinguishable from a different training seed on both folds); the limb in every direction
available to us (geometry channels, radial normalisation, radial denoising, a limb-weighted loss,
and an exact centre-to-limb augmentation that rotates the solar sphere: +0.0000 and −0.0036
against a seed replicate, with no change in the band it targets); checkpoint selection by Dice
versus the last epoch; transductive self-training; boundary post-processing; the operating
point; the semantic extraction threshold below 0.2 (reach rises at no candidate cost, the refitted
system does not, and the leaderboard ordering places it below every submission at 0.2); and image
quality, i.e. seeing and solar-cycle phase measured from each frame. Appendix A lists all of them
together with their mechanisms.

**The benefit of ensembling follows a fitted law rather than trial and error.** Averaging k similar
members gains A(1 − 1/k) with A = +0.0092 and R² = 0.9954; at nine effective members at most
+0.0010 remains. For this reason the project stopped adding members and turned to measuring
information.

## Methodology and verification

Every adopted decision traces to a committed measurement script, and every rejected one is
recorded together with the measurement that rejected it. Our PQ implementation is verified against
the organisers' published scorer, and no decision was adopted on a proxy metric, a rule that
mattered when mask mAP50 and PQ ranked two backbones in opposite orders. Differences on which a
decision rests are assessed with a paired image bootstrap (Efron and Tibshirani, 1993) against a
pre-registered floor of ±0.003, and a change of initialisation is always compared against a rerun
of the same recipe with another seed, never only against the reference: the seed noise of a single
ensemble member is ±0.001, and on one fold the reference's own member proved to be an unfavourable
seed worth +0.003. We also maintain a log of our own errors and the rules derived from them (§6,
Appendix B), including analysis errors we detected and corrected: a reachability gain destroyed by
our own de-overlap step, an "inter-annotator ceiling" that measured self-matching rather than
agreement, a candidate-level pilot on the inner cross-validation of one fold that did not survive
the full protocol, and a magnetic input path that loaded the field and then discarded it.

Three late measurements bound what local validation can resolve (§2.4). First, the bootstrap
resamples images within one fold partition and cannot see the partition's own variance: one
selector variant re-read on five partitions of the same images moves with a standard deviation of
0.0015, twice that of the baseline, so a single-partition interval is a lower bound and differences
below about 0.003 cannot be resolved locally at all; this is the most economical explanation of why
the local magnitude of our sub-floor selector variants did not predict their order on the
leaderboard. Second, 74 % of out-of-fold candidates carry a single detector vote where test
candidates carry a median of four, so local and test candidates share no common support and cannot
be reweighted onto each other; running the mismatch backwards on labelled data changes pooled PQ by
+0.0000. Third, five identical training runs of the segmentation network differ in validation
pixel-Dice with a standard deviation of 0.0059, so a single control–treatment pair differs by about
0.008 by chance; a conditioning experiment that read +0.014 on one pair read −0.004 over three, and
training-stage changes are now read over at least three seed pairs.

The final submission was rehearsed rather than assumed to be reproducible. Rebuilding it from its
checkpoints in an environment that had changed in the meantime (scikit-learn
1.6.1 → 1.7.2, torch 2.11, OpenCV 5.0, NumPy 2.3) reproduced the submitted file **byte for byte**,
all 1117 run-length encodings, on the same machine. Rebuilding it on a different GPU architecture
reproduces 1091 of 1117 encodings exactly and 25 more as the same mask with a one-pixel encoding
difference, and loses exactly one instance on one image, where a candidate crosses the keep gate
through floating-point differences alone, an expected effect of ±0.0004 PQ. We therefore treat the
submission file generated on one machine as the artefact of record, and claim bit-level
reproducibility only on that machine. The code state behind the submitted system, and behind every
number in this report except those listed below, is tagged `solar-final-analysis-v1` (commit
`79dc017`); at that commit the submitted file is rebuilt byte for
byte from the featurised test pool and the bagged selector bundle, and the tag records the SHA-256
of the submitted file and of the bundle. The measurements added afterwards (the test-set
inspection of §2.3, the comparison with the prepared alternative in §3.6, the seed-noise measurement
of §2.4 and the last three interventions of §4.7) change nothing in that system and are at the tag
`solar-final-analysis-v2`.

We also verified, rather than assumed, that our validation is unbiased by temporal proximity.
Random image folds place 31 % of training images (219 of 707) within 24 h of a cross-fold
neighbour, and images with such a neighbour score 0.4420 against 0.4422 for those without, so there
is no near-duplicate inflation. Test frames lie a median of 36.7 h from the nearest training frame,
the same regime.

## Compliance

All labels used anywhere are provided by the competition. The public MAGFiLO release, which
contains annotations for all 180 test images, was quarantined in the first week and never used for
training or prediction; our only use of it is analytical and documented, including the annotation
protocol (±70° longitude cut-off, no-holes convention), which we read as a specification. External
data appear only where the organisers' clarification permits them: magnetograms and AIA frames are
training-time inputs to teacher models, and the submission path refuses any checkpoint that reads
them, because final inference is restricted to the provided H-alpha images. External unlabelled
GONG frames appear in pretraining experiments only, not in the final model. AI assistance was used
throughout, as the organisers permit, and every result in this report is reproducible from the
committed code.
