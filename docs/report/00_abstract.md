# Abstract

We present a two-lane instance segmentation system for solar filaments in full-disk H-alpha
images from the Global Oscillation Network Group (GONG), together with a measured account of where
Panoptic Quality (PQ) is lost on this task. A YOLO11m-seg detector and a twelve-member U-Net
semantic ensemble propose candidates, which are clustered, fused, made pixel-disjoint and filtered
by a gradient-boosted selector whose keep threshold is derived from the metric rather than tuned.
The submitted system reaches a pooled five-fold cross-fitted PQ of 0.4431 on the 707 training
images; a single human annotator scored against the other annotators reaches 0.34 on the same
metric. Five findings organise the report. First, matched-mask quality (SQ 0.678) already exceeds
inter-annotator agreement: on copies drawn by two annotators, our mask is the closer of the two to
an annotator in 76.2 % of cases, and SQ is flat across the solar disc, so the remaining loss is
carried by recognition. Second, the selection stage is at its information limit: oracles computed
on one annotator transfer negatively to another, and a set-level PQ objective provably reduces to
the deployed per-candidate rule. Third, eighteen structurally distinct mechanisms improved an
intermediate quantity without improving realised PQ. Fourth, recall depends on an image regime
measurable from the frame itself: across quartiles of disc high-frequency energy PQ falls from
0.464 to 0.416, entirely through recall, and the selection ceiling that grows with it (+0.0498)
transfers across annotators at −0.365, so observed regime sensitivity is not a recoverable
opportunity for the interventions we tested. Fifth, co-temporal SDO/HMI magnetograms carry genuine
pixel-level information (+0.0146 AUC) that does not reach the instance-level metric. Finally, the
public leaderboard sits 0.04–0.05 below the local estimate for every version of the pipeline, while
the test images are indistinguishable from the training images (AUC 0.534); we attribute the offset
to the test annotations and estimate the private score at 0.40–0.42. All results are reproducible
from the accompanying repository at tagged commits.

**Keywords:** solar filaments; instance segmentation; panoptic quality; annotator disagreement;
H-alpha; GONG
