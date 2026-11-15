# Appendix B — Engineering errors and derived practices

*A curated selection from `experiments/MISTAKES.md`. It is included because the practices described
in §7 are not stylistic preferences: each exists because something failed in a way that would
otherwise have corrupted a result. The full log is in the repository.*

## B.1 Analysis errors detected and corrected

**Reachability measured on the wrong object.** Multi-threshold extraction raised raw candidate
reachability by +0.0318 and lowered realised PQ by 0.0161. The reachability had been measured on
components *before* the pool's build-time de-overlap, which lets nested candidates carve each
other, so the improvement was destroyed at construction. *Rule: measure a ceiling on the object the
pipeline will actually keep, not on its precursor.*

**Average contribution mistaken for marginal contribution.** When deriving the metric's keep rule,
we first charged a cluster `SQ·k/(k + (N−k)/2)` and concluded that minority-consensus filaments
should be skipped. However, not predicting a cluster is not free, since its k rows become false
negatives worth 0.5k; the marginal value is therefore `2·SQ·k/N`, and at our operating point every
consensus tier clears the bar. *Rule: price the counterfactual, not the action.*

**An agreement statistic computed between dependent sets.** We reported human inter-annotator
agreement as SQ 0.8393, obtained by scoring the public MAGFiLO labels against the competition
ground truth, without noticing that every public row has an exact counterpart there. The figure
measured self-matching degraded by our own de-overlap step. Measured correctly between different
annotators' copies, agreement is 0.629, which reverses the conclusion: our masks agree with the
ground truth better than two people agree with each other. *Rule: verify independence before
computing any agreement statistic.*

**A silent guard that turned an experiment into a no-op.** In testing whether augmenting the
selector's *inputs* helps, the augmented-training branch was guarded with
`if len(cols) == len(feats): concat(...)`. One feature was missing from the jittered rows, so the
branch never ran and the experiment compared the baseline against itself, reporting "+0.0000, 95 %
CI [+0.0000, +0.0000]". A zero-width confidence interval is the signature of a comparison that did
not take place, and to a later reader it is indistinguishable from a genuine null result. *Rule: a
guard that can silently change what an experiment measures must be an assertion, never a
conditional.*

**A refiner trained towards one annotator.** In building a model to improve candidate masks, we set
its target to the single ground-truth copy that each candidate matched best. The scorer, however,
matches a prediction against every annotator entry, and the project had already measured that the
consensus lies closer to each annotator (SQ 0.79 for a consensus mask scored against each
annotator, against a mean of 0.634 for two annotators scored against each other). The target
therefore taught the model to imitate one convention, which is measurably worse than the consensus
the candidate was already close to. The project's earlier refiner had used the majority mask for
exactly this reason; we repeated a known error without checking. *Rule: when a target has to
represent several annotators, derive it from the metric's own accounting, and read the earlier
attempt before rebuilding one.*

**A targeted change read only where it was targeted.** Five model changes were designed against a
described population (166 unambiguous missed copies whose distinguishing property is crowding), and
each was read both on the pooled metric and on that population. None of them recovered a single one
of the 41 fold-0 members of that population, a result that became visible only because the
population-level reading existed. *Rule: a change aimed at a measured population must be read on
that population, or its failure will be hidden inside a pooled number dominated by other rows.*

**A proxy metric trusted across model families.** A crack-pretrained yolov8x outperformed yolo11m
on mask mAP50 (0.6538 vs 0.6333) and underperformed it on PQ (0.4118 vs 0.4193). *Rule: no
adoption on a proxy metric; the competition metric decides.*

**An experiment whose precondition our own data had already evaluated.** We built, ran and
submitted transductive self-training before connecting it to our measurement that the train/test
split is i.i.d.; without distribution shift, adaptation has nothing to adapt to, and the method
scored exactly at baseline. *Rule: state the mechanism's precondition and check whether an existing
measurement already answers it.* This rule subsequently saved two GPU-days by eliminating candidate
experiments before they were run.

**A test suite that mislabelled its own cases.** The first version of the parity suite in Appendix C
described predictions at IoU 0.665 and 0.855 as "below" and "above" the matching threshold, so the
strict-inequality semantics it claimed to verify were never exercised. *Rule: assert the property a
test claims to test.*

**Three claims read as one.** After the frame-relative selector block missed its pre-registered
criterion we wrote that the regime mechanism had been closed. An outside review pointed out that
"the mechanism is real", "this intervention works" and "this becomes a submission" are separate
claims, that the criterion (a ratio of two noisy quartiles) was not a principled test of an
interaction, and that the decisive measurement was the size of the opportunity. Measuring it
reversed the reading twice: the ceiling was large (+0.0498), and its transfer across annotators was
negative (−0.365). *Rule: before closing a mechanism, measure the ceiling of the stage it acts on and
the transfer of that ceiling; a failed intervention closes the intervention, not the mechanism.*

**A prediction identifier that matched no image.** A shrink sweep named its predictions `p_{i}`. The
scorer matches predictions to images by the part of the identifier before the first underscore, so
every arm scored exactly 0.0000 without raising an error. *Rule: every harness asserts that its
prediction image identifiers intersect the ground-truth image identifiers before scoring.*

**An operation applied twice.** The corrected sweep then showed that removing the shrink would gain
0.0487. The labelled pool already holds the shrunk masks, so the sweep had shrunk them a second
time; reproducing the deployed PQ from the unmodified pool exposed it. *Rule: reproduce the deployed
number from an artefact before measuring a change to it.*

**A rule inferred from one case.** After one submission lost on the leaderboard with its test and
local footprints moving in opposite directions, we recorded that sign flip as a pre-submission
kill criterion. Tested against every earlier submission with both numbers, it failed in both
directions. *Rule: a proposed gate is tested on all past cases before it is written down as a
rule.*

**A distributional claim made from medians.** We reported that frame-relative features close the
train/test gap because their medians matched (0.533 against 0.531). A classifier still separated the
two sets at AUC 0.952. *Rule: "the same distribution" is established with a multivariate two-sample
test, not with marginal statistics.*

**Edits that did not do what they appeared to do.** Code patched by text substitution produced a
file that did not parse, which we reported three times as running while reading an earlier log, and
in a second case an unterminated substitution loop, which we reported as stopped and which a
process listing found still running about 27 CPU-hours later. *Rule: after every edit, parse the
file; after every launch, confirm the process and the timestamp of its log before reporting; after
stopping a process, confirm by its identifier that it is gone.*

**"Never measured", from a search in the wrong language.** We stated that the physical input stack
had never been tried. The ledger, which is written in Turkish, recorded it; the search had used
English terms. *Rule: search the ledger in both languages and by result value, and check Appendix A,
before claiming an experiment was never run.*

**A geometric quantity quoted for the wrong object.** An earlier version of §2.1 gave the disc a
median radius of 991.7 px (914–1033). That number is the equivalent-area radius of the bright part of
each frame, computed as a by-product of a tiling script, and in some frames it includes the bright
sky halo around the limb. The disc fitted to the limb and checked on overlays is 899.95 ± 0.73 px in
every frame. *Rule: a geometric quantity is quoted from the fit that was validated for it, not from
a by-product of another measurement.*

**A correction planned from a thumbnail.** Reviewing the test frames on contact sheets at 1/5.4
scale, we placed an unproposed dark structure beyond the ±70° annotation limit and planned to
withdraw an earlier finding. Measured on the full frame, its centre lies at 54° longitude, inside
the limit, and the earlier finding stood. Near the limb longitude changes fastest with radius, and
a thumbnail cannot place a structure to ten degrees there. *Rule: a positional claim is measured on
the full-resolution frame before it is written down or used to overturn an earlier one.*

**A noise floor estimated from one repeat.** We set the ±0.01 band for training-stage pairs after
one control run had reproduced an earlier one to within 0.0006. Five identical controls later
spread with a standard deviation of 0.0059, so the band was about 1.2 standard deviations of a
pair; the conditioning experiment passed it on one pair and failed on three (§2.4). The replicate
step caught this before any five-fold training was spent, but the band itself had been
mis-calibrated. *Rule: a noise floor is measured from at least three repeats before a threshold is
set on it, and epochs of one run are not independent samples.*

## B.2 Infrastructure failures

**Piping a long-running job through a filter.** `... | head` closed the pipe and terminated a
training run at epoch zero with exit status 0; a `grep` suppressed a traceback; and `tr`/`tail`
block-buffered their output so that progress never appeared. *Rule: jobs write their own
`train.log`; filter when reading, never when running.*

**Committing a refactor without an import test.** Five lines of a removed loop were left at module
scope (`return df` outside a function), which made the module unimportable for days until a
submission script first needed it. *Rule: run `python -c "import <module>"` before committing a
refactor.*

**Terminating a chained job's child process instead of the chain.** The shell loop counted the
terminated process as finished and launched the next arm; the orphaned loop continued, and three
jobs ended up resident on one GPU. *Rule: terminate the shell running the loop first, then the
process, then the workers, and verify by command line rather than by memory usage.*

**A variant run overwriting a production artefact.** When `--pool/--tag` options were added to the
evaluation harness, the results JSON was tagged but the model dump was not, so an experimental pool
retrained the production selector. *Rule: when a harness gains a variant flag, bind every file it
writes to that flag.*

**Parallelism that slowed execution.** Sharing a 16 GB card between a 10 GB and a 7 GB job pushed
both into host-memory spill: 345→488 s/epoch and 45→450 s/epoch, making the parallel plan take
11.3 h against 4.7 h when serialised. *Rule: parallelism is free only while the card has headroom;
re-measure rather than assume.*

**The environment as an experimental variable.** Windows Smart App Control blocked an unsigned
`cv2.pyd` overnight (the same file that had worked for ten days), halting every measurement. The
issue was resolved with a fresh signed wheel rather than by disabling the protection. *Rule: after
repairing a package, smoke-test the whole stack, not only the repaired package.*

# Appendix C — Scorer parity verification

*Every local number in this report is computed by `src/filament/pq.py`. This appendix provides the
evidence that it computes the same quantity as the competition scorer. The organisers published
their exact implementation in the Self-Evaluation notebook; it is extracted to
`outputs/research/official_pq_scorer_extracted.py`, and both implementations are run on identical
inputs by `src/eda/scorer_parity.py`.*

The official scorer is a two-step composition (`get_overlap_df` builds one IoU matrix per
annotator-image entry, and `get_pq_score` reduces it), and the parity run invokes that composition
itself rather than a reimplementation of it.

| Case | Ours | Official | Δ |
|---|---|---|---|
| Perfect copy of the ground truth | 1.000000 | 1.000000 | 0 |
| Empty prediction set | 0.000000 | 0.000000 | 0 |
| IoU 0.4950 — must not match | 0.000000 | 0.000000 | 0 |
| **IoU 0.5000 exactly — strict `>`, must not match** | 0.000000 | 0.000000 | 0 |
| IoU 0.5050 — must match | 0.505000 | 0.505000 | 0 |
| One prediction, three annotator copies | 0.955863 | 0.955863 | 0 |
| Real fold-0 subset (40 images, 400 candidates) | 0.254194 | 0.254194 | 0 |

: Parity between our PQ implementation and the organisers' scorer.

The largest absolute difference is **0.00e+00**.

The cases are chosen to exercise the semantics that shaped our design rather than typical inputs
only: the strict inequality at the matching threshold (a prediction covering exactly half of the
ground truth does *not* match), the independent scoring of one prediction against several
annotator copies, the empty-prediction case in which every ground-truth instance becomes a false
negative, and a real fold subset on which the two implementations must agree over several hundred
overlapping candidates simultaneously.
