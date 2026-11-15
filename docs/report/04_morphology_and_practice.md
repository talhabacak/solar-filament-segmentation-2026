# Final Report — Draft Sections 5–8

## 5. Morphology of the predictions

*The second component of the evaluation rubric asks how the predicted segmentations appear on the
H-alpha images. We report the two properties the organisers named, the many-to-many relation
distribution and the detection of filaments irrespective of whether a human annotated them, both
measured on our own out-of-fold output before submission.*

### 5.1 Relation structure

The relation structure is computed with the Self-Evaluation notebook's own definition (one IoU
matrix per annotator-image entry, with a link being any overlap), on the production out-of-fold
output at the submission policy [`src/eda/m2n_report.py`]:

| Side | degree 1 | degree ≥ 2 | degree 0 |
|---|---|---|---|
| GT → predictions (fragmentation) | 69.9% | **2.1%** | 28.0% |
| Predictions → GT (merging) | 87.8% | **2.6%** | 9.6% |

: Degree distribution of the many-to-many relation between ground truth and predictions.

On this axis the output is structurally clean: there is one prediction per filament in seven cases
out of ten on the ground-truth side and in nearly nine out of ten on the prediction side, with
fragmentation and merging each near 2–3%. The remaining mass is not structural: it consists of pure
misses (28.0%) and pure extras (9.6%), i.e. a question of coverage, which is where §4 locates the
entire gap. The run behind this table reproduces the PQ of the unbagged operating point, 0.4421, before counting
degrees, which confirms that it follows the deployed path and not a neighbouring one. A
methodological note is retained because it changes the numbers: an initial version accumulated
prediction degrees across annotator entries, inflating them by annotator multiplicity; the official
routine builds one matrix per entry, and the table above is the corrected form.

### 5.2 Extras that are real filaments

Our scored false positives are not uniformly errors. Rendering the sixteen scored false positives
with the highest selector confidence together with their images
[`outputs/gallery/fp_unannotated.png`] shows that all sixteen are unmistakable filaments: dark,
elongated, curvilinear absorption features that no annotator of the image drew. Given the
organisers' stated interest in *"algorithms which detect all filaments whether they were annotated
by human or not"* (discussion 735256), we report this explicitly: part of our false-positive count
is recall of unannotated filaments, which the metric penalises. Measured on one fold of the
submitted pool, so that all four numbers describe the same object, the detector lane reaches
78.7 % of ground-truth rows at IoU > 0.5, the semantic lane 66.6 %, their union 82.3 %, and the
fused pool that the selector receives 73.6 %: fusion gives up 8.7 points of reach in exchange for
candidates that are single objects rather than overlapping proposals. The selector is best
understood, and is documented in §3.5, as a deliberate and measured convention-matching layer on top
of a detector that finds more than the convention records.

### 5.3 Boundary behaviour

Matched-mask deficits are concentrated at the rim (74% of surplus and 82% of missing pixels lie
within 3 px of the ground-truth boundary), and our SQ (0.678) exceeds inter-annotator agreement
(0.634 on the same statistic). Visually, the contours follow the filament spine correctly and
differ from any individual annotator mainly in how far the boundary extends into the penumbral
gradient, which is the same disagreement that two annotators have with each other.

### 5.4 Fine-scale structure

The organisers list fine-scale structure, and barbs in particular, as the first challenge of the
task: thin, thread-like features that leave the filament body at a characteristic angle to its
spine. We measured how much of it our masks recover, on the annotators' own terms, using the
manually drawn spines [`src/eda/e220_fine_structure.py`]. A barb pixel is a pixel of an annotated
mask that is both thinner than the body (removed by a morphological opening at 0.35 of the body
width, with the tapering ends excluded) and farther from the annotated spine than half a body
width plus 2 px. Each criterion alone misclassifies part of the body: the opening also removes thin
necks of the body, which the spine runs through, and the spine distance also counts wide body
regions when the spine is drawn off-centre. Recovery is measured on the 8199 annotator copies,
for our matched masks and, where another annotator drew the same filament, for that annotator's
mask.

| region of the annotated mask | share of annotated area | our mask | second annotator |
|---|---|---|---|
| barbs (both criteria) | 0.17 % (303 copies) | 38.0 % | 49.3 % |
| beyond the body tube around the spine | 11.1 % | 46.0 % | 52.4 % |
| body | — | 80.5 % | 79.7 % |

: Share of each region of an annotated mask that is covered by our matched mask and by a second
annotator's drawing of the same filament (copies drawn by both; 114 copies with barbs, 2112 with
lateral structure beyond the body tube).

Three conclusions follow. First, the body is recovered as completely as a second annotator draws
it. Second, barbs are where the system falls short of human agreement, by 11 points, and they are
also where humans agree least: a second annotator covers only half of another annotator's barbs.
Third, barbs are rare in the annotations, 0.17 % of the annotated area in 303 of 8199 copies, so
they carry almost no weight in PQ, and a training objective set by PQ has no incentive to recover
them. Figure 4 shows both outcomes. When a barb is short and attached to a dark body, our mask
follows it; when it is long and faint, as in the bottom-right panel, the mask stops at the body.
Recovering long faint barbs is the same problem as recovering the faint length of a filament
(§4.3), and the measurements of §4.6 locate it in the model's response rather than in the
post-processing.

![Barbs (shaded) of annotated filaments, defined as above, with the annotator's outline (solid),
our out-of-fold mask (dashed) and the annotated spine (thin line). Top row: the three matched
copies, from distinct images, whose barbs our masks recover best; bottom row: the three recovered
least. In the bottom-left panel the shaded region is the faint end of the filament rather than a
side branch, a limit of the definition near spine ends [`src/eda/e220b_barb_figure.py`]](figures/fig4_fine_structure.png)

## 6. Negative results

*These results are reported at length deliberately: in a saturated regime they are the findings,
and each one closed an axis that would otherwise have consumed effort. The full ledger is given in
Appendix A; the nine results below carry general lessons.*

1. **Snapping masks to intensity structure costs 0.070 PQ.** This is the clearest demonstration
   that the target is an annotation convention rather than a physically optimal boundary, and the
   rest of the design follows from taking it seriously.
2. **Reachability gains can be negative.** Multi-threshold extraction raised raw candidate
   reachability by +0.0318 and *lowered* realised PQ by 0.0161, because build-time de-overlap lets
   nested candidates carve each other, so the improvement was destroyed at construction. Measuring
   a ceiling is not measuring a gain.
3. **Proxy metrics can invert.** A crack-pretrained yolov8x outperformed yolo11m on mask mAP50
   (0.6538 vs 0.6333) and underperformed it on PQ (0.4118 vs 0.4193). No decision in this project
   was adopted on a proxy metric.
4. **An unexamined precondition can invalidate an experiment in advance.** Transductive
   self-training was built, run and submitted before we connected it to our own measurement that
   training and test sets are i.i.d.; without distribution shift, adaptation has nothing to adapt
   to, and the method scored exactly at baseline. This produced the working rule that now gates
   every experiment: *state the mechanism's precondition, and check whether an existing measurement
   already evaluates it.*
5. **Agreement statistics require independence.** We initially reported a "human ceiling" of
   SQ 0.84 obtained by scoring the public MAGFiLO labels against the competition ground truth,
   without noticing that every public row has an exact counterpart there; the figure measured
   self-matching degraded by our own de-overlap. The correct figure is 0.629, and it reverses the
   conclusion: our masks agree with the ground truth better than two annotators agree with each
   other.
6. **An oracle must be valued by its transfer, not by its in-sample value.** The two oracles on
   which the project relied, reordering the pool and perfect selection from it, are worth +0.033
   and +0.069 when the choice is made and scored on the same annotator, and transfer at −25 % and
   −32 % when the choice is made on one annotator and scored on another. Every oracle reported here
   now carries a transfer test; two headline numbers did not survive theirs, and the one that did
   (choosing the best ensemble member per image, +23 %) was the one we least expected to.
7. **Decorrelation is necessary but not sufficient.** A transformer member is the only family we
   measured to be genuinely decorrelated from the production member (error correlation 0.871
   against 0.926–0.940); the decorrelation replicates on a second fold, and the member still loses
   to a seed replicate there. An ensemble gains from members that are both different *and* strong,
   and on 707 images every strong model is similar.
8. **Attribution locates information flow, not missing capacity.** Zeroing the 96 px encoder skip
   connection costs faint filaments 0.104 IoU and bright ones nothing, a genuine and reproducible
   localisation; yet adding a context module at exactly that point changed nothing, either on the
   pooled metric or on the targeted population. Knowing where information travels does not reveal
   where capacity is lacking.
9. **Gains below the measurement floor do not accumulate.** Three changes from different
   mechanisms, each between +0.0003 and +0.0011 alone, measure −0.0004 together against a
   member-count control. A strategy of accumulating many small gains is therefore unavailable, even
   when the gains are uncorrelated.

## 7. Reproducibility and engineering practice

- **Repository layout.** `src/filament` (metric, RLE, primitives), `src/data` (cache builders),
  `src/models` (training, inference, submission), `src/eda` (one script per question, named for the
  question), `scripts` (chained runs), `docs` (the record of reasoning), and `experiments` (ledger
  and error log).
- **One function per decision.** Candidate featurisation exists once and is shared by the pool
  builder and the submission path, because a duplicated implementation once flipped 5.9% of keep
  decisions.
- **Every adopted number is traceable** to a committed measurement script and its logged output,
  and every rejected idea is recorded with the measurement that rejected it. Appendix D indexes the
  scripts by the question each one answers.
- **Scorer parity.** Our PQ implementation is verified against the organisers' published scorer,
  and all local numbers in this report are computed with it (Appendix C).
- **Submission validation** is mechanical: disjointness, RLE round-trip identity, the 1,024-mask
  bound, coverage and area sanity are checked before every upload
  [`src/models/validate_submission.py`].
- **The final submission was rehearsed rather than assumed to be reproducible.** Rebuilding it from
  its checkpoints in an environment that had changed in the meantime (scikit-learn 1.6.1 → 1.7.2,
  torch 2.11, OpenCV 5.0, NumPy 2.3) reproduced the submitted file **byte for byte**, all 1117
  run-length encodings, on the same machine. Rebuilding it on a different GPU architecture
  reproduces 1091 of 1117 exactly and 25 more as the same mask with a one-pixel encoding difference,
  and loses exactly one instance, where a candidate crosses the keep gate through floating-point
  differences alone (an expected effect of ±0.0004 PQ). We therefore treat the file generated on one
  machine as the artefact of record and claim bit-level reproducibility only on that machine
  [`outputs/eda/_rehearsal_gpu.sh`, `outputs/eda/_compare_submissions.py`].
- **Inference cost, measured.** Building a submission from cached probability maps takes the
  semantic ensemble 12 members × ~21 s for the 180 test images on one GPU (flip test-time
  augmentation, fp16), then a CPU-bound fusion and featurisation step. That step was written
  against the full 2048² frame and cost 1.41 s per image; restricting every operation to the
  candidate's bounding box (`src/models/maskops.py`) brings it to 0.48 s, and distributing
  images over eight processes brings the whole build to 85 s from 651 s. The change is lossless
  in the strict sense: the rebuilt test pool matches the reference pool in all 42 features and
  every mask encoding, and the rebuilt submission is byte-identical to the submitted one
  [`outputs/eda/_verify_lossless_fusion.py`]. The detector lane was not timed separately on the
  GPU; on a 14-thread CPU without a GPU, one of its five models takes 5.6 s per 2048² image at
  1792 px [`src/eda/e218_nms_tighter.py`]. The complete test route from the images (five
  detectors, twelve semantic members, fusion, features, selector), rehearsed from a fresh clone on
  2026-11-15, takes 17.5 min on one RTX 5070 Ti (16 GB); training the 17 networks took about 30
  GPU-hours by their training logs (repository README).
- **`experiments/MISTAKES.md`** records engineering failures and the general rules derived from
  them (output-filter pipelines silently terminating training, Windows process-spawn semantics,
  process trees of chained jobs, variant runs overwriting production artefacts, and the analysis
  errors of §6). A curated selection is included as Appendix B: the practices exist because these
  failures occurred.

## 8. Data provenance and compliance

- **Ground truth.** Every label used for training, validation or selection is provided by the
  competition. Following the organisers' clarification (discussion 736543), the dataset's other
  manual metadata (spine, bbox, area, category_id) are permitted supervision; we used the spine in
  one auxiliary-head experiment (§4 and Appendix A, not adopted) and no other metadata.
- **The public MAGFiLO release.** We verified that it contains annotations for all 180 test images
  and quarantined it in the first week (`outputs/external/magfilo_quarantine/`, excluded from
  version control). It was never used for training or prediction. Our only use is analytical: an
  ID-level overlap count and a label-value measurement restricted to training stems, with no
  test-image geometry decoded [`src/eda/public_gt_value.py`]. The organisers' ruling that external
  ground truth is prohibited is respected in full.
- **External unlabelled data.** GONG H-alpha archive frames (confirmed as permissible by the
  organisers, discussion 737215) were used in the pretraining experiments reported in §4. They are
  **not** part of the final model.
- **External observations used as teachers only.** The organisers' answer in discussion 729396
  permits external data for obtaining additional information while stating that *for the final
  test, inference takes place solely on the H-alpha images*. Co-temporal SDO/HMI magnetograms and
  SDO/AIA 304 frames were retrieved from the Joint Science Operations Center (JSOC), registered into
  our image frame and measured under that constraint: as inputs to teacher models, and as privileged
  information for an H-alpha-only student. The submission path refuses any checkpoint whose channel
  configuration reads them, so no submission has ever depended on data outside the provided images
  [`src/models/submit_v2.py`]. The measurements and their negative outcome are reported in §4 and
  Appendix A.5.
- **Pretrained weights.** ImageNet weights are used in production; COCO (Lin et al., 2014),
  Sentinel-1, Satlas (Bastani et al., 2023) and crack-segmentation checkpoints were used in
  experiments only. All are publicly and freely available under their respective licences.
- **Test images.** Test images were used for inference. One experiment (§4) additionally trained on
  test *images* with self-generated soft targets, declared as transductive self-training; it is not
  part of the default submission path.
- **AI assistance.** AI assistance was used for coding, analysis and documentation throughout, as
  the organisers explicitly permit (discussion 724503). All experimental designs, measurements and
  conclusions in this report were executed and verified in this repository, and every number is
  reproducible from the committed code.
