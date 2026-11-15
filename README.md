# Solar Filament Segmentation Challenge 2026 — team TalhaBacak

Code, measurements and reports behind our final submission to the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(instance segmentation of filaments on full-disc GONG H-alpha images, scored by Panoptic Quality).

| | |
|---|---|
| Final submission | `outputs/submissions/sub036_t020pox_bag5.csv` (1117 instances) |
| Prepared alternative, not selected | `outputs/submissions/sub042_t020pox_bag5_cascade085.csv` (sub036 plus a marginal-band cascade; how it is built: `experiments/sub042.md`) |
| Local score | PQ 0.443, SQ 0.678, RQ 0.653 — pooled five-fold out-of-fold, all 707 training images |
| Public leaderboard | 0.40 |
| Technical report (4 pages) | [`report/Technical_Report_EN.pdf`](report/Technical_Report_EN.pdf), compiled from the organisers' template ([`report/overleaf/main.tex`](report/overleaf/main.tex)) · Turkish translation: [`report/Technical_Report_TR.pdf`](report/Technical_Report_TR.pdf) |
| Full report (63 pages) | [`report/Full_Report_EN.pdf`](report/Full_Report_EN.pdf) · Turkish: [`report/Full_Report_TR.pdf`](report/Full_Report_TR.pdf) |
| Pipeline notebook | [`notebooks/00_end_to_end_pipeline.ipynb`](notebooks/00_end_to_end_pipeline.ipynb) |

## The system in brief

```
GONG H-alpha frame (2048 x 2048)
  |-- detector lane   YOLO11m-seg @1792, one model per fold, medoid annotator copy
  |-- semantic lane   U-Net (EfficientNet-B3) @1536, BCE + Dice + clDice, 12-member ensemble
  +-> fusion          cluster at IoU > 0.5, confidence-weighted vote, pixel-disjoint masks
       -> 42 features per candidate
       -> selector    two-stage gradient boosting, mean of 5 image-bootstrap copies
       -> keep        conf >= 0.40 and p_sel >= 0.60 (derived from the PQ formula)
       -> submission  run-length encodings
```

Every parameter is listed in Table 1 of the technical report and set in the first code cell of the
notebook.

### Why 17 networks, and what they cost

Fifteen of the networks are the five fold models of each lane: per fold one detector, one clDice
U-Net and one instance-balanced U-Net. They exist because the selector is trained on out-of-fold
candidates for all 707 training images, which takes one model per fold in each lane. At test time
the same models are used together, so the ensemble costs inference only. The two
heavy-augmentation U-Nets are the only networks trained for the test-time ensemble alone.

What the parts buy, measured out of fold:

- **Fusing the two lanes:** +0.021 PQ over the semantic lane alone (report, Table 3).
- **The selector:** +0.009 on top of that.
- **The instance-balanced members:** +0.0061, CI [+0.0051, +0.0071]. This was the largest single gain
  of the project, and it appeared only once all five folds were trained (`docs/15_ABLATION_LEDGER.md`, §7).
- **Ensemble size:** averaging *k* similar semantic members gains +0.0092·(1 − 1/*k*) on fold 0 (§23b).
  The twelve members are weighted unequally and act as nine effective members (§27), which collect
  89 % of what averaging can give; more members of the same kind would add at most +0.0010.

Training took about 30 GPU-hours. The training logs give 0.7 h per detector, 2.6–3.9 h per
instance-balanced member, and 0.8 h and 2.5 h for the two heavy members; the clDice members did not
log their time and are estimated at about 1.2 h each. The test route runs in about 17 minutes for
the 180 test images on one 16 GB consumer GPU.

## Reproducing the submitted predictions

Install the pinned environment into a fresh Python 3.13 virtual environment (CUDA 12.8 for the GPU
routes). The file names the PyTorch package index itself, because the CUDA builds of torch are not
on PyPI:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The competition data are needed for routes 2 and 3. They are downloaded with the Kaggle API, which
needs an API token (`~/.kaggle/kaggle.json`) and the competition rules accepted on the Kaggle page;
the notebook then fetches and unpacks them itself. The expected layout is

```
data/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg                    707 images
data/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg                      180 images
```

There are three levels, and the notebook runs each of them (`MODE` in its first code cell).

| Route | What it shows | Rehearsed |
|---|---|---|
| 1. `FINAL_STAGE` | the selector and keep rule turn the shipped test candidate pool into the submitted file | yes, from a fresh clone |
| 2. `REPRODUCE` | the stored networks turn the test images into the submitted file | yes, 2026-11-15: fresh clone, fresh environment from `requirements.txt`, data downloaded by the notebook, weights from the release archive (the same files, by hash, that the repository now stores); 17.5 min on one RTX 5070 Ti (16 GB, Windows); the result is byte-identical to the submitted file (run twice; the second time with nothing installed or created by hand) |
| 3. `RETRAIN` | the raw data turn into the networks, the training pool and the selector | no (about 40 h of GPU time). Every step is the command that produced the shipped artefact. Its last stage was checked on its own: regenerated from the out-of-fold pool, the labelled pool equals the shipped one exactly and the ten extra features agree to within 4·10⁻⁵ (float rounding in two of them); a selector refitted on these regenerated inputs keeps 1109 of the 1117 submitted masks and adds 10, so a retrain is expected to come close to the submitted file, not to equal it |

**1. Final stage, from this repository alone (CPU, seconds).** The repository ships the featurised
test candidate pool and the fitted selector, so the final submission can be rebuilt and compared
with the uploaded file byte for byte:

```bash
python src/models/submit_from_pool.py --pool outputs/eda/test_pool_t020pox.parquet \
    --selector outputs/selector/selector_t020pox_bag5.joblib --conf 0.40 --p-min 0.60 \
    --out outputs/submissions/final_reproduced.csv \
    --check outputs/submissions/sub036_t020pox_bag5.csv
```

The command ends with `byte-identical ... (line endings normalised): True`. The submitted file
has SHA-256 `63b570002a2be27ddb813c5ad249100df3b5f9fc27ada7f817a9ef974fd2e527` with CRLF line
endings, as uploaded, and `43c9663abc4e91b4f8dc4081f86b04ffc86ffff80094727f92ae3879cfdcb97e` with LF
endings, as a Linux checkout holds it. The selector itself can be refitted from the shipped
labelled training pool in about a minute (`python src/models/fit_selector_bagged.py --out
<new file>`); the refit also rebuilds the submission byte for byte.

**2. Full inference from the stored weights (one 16 GB GPU, about 20 min).** The 17 network
weights are in this repository (see below); run the notebook with `MODE = 'REPRODUCE'`. It loads
them from `outputs/runs*/`, downloads the data if `data/MAGFiLO_1.0_Kaggle_2026/` is missing, runs the five detectors and the
twelve semantic members on the test images, fuses and featurises the candidates into
`outputs/eda/test_pool_rebuilt.parquet`, applies the shipped selector, and compares the result
with the submitted file. The shipped pool, selector and submission are never overwritten; the
evaluation cell rewrites `outputs/eda/pq_uncertainty.json` with the same values. The test-time detector candidates are
made with NMS IoU 0.05, the out-of-fold training candidates with 0.5; this is how the submitted
system was built (report, Section 2.4). The 0.05 is the optimum of the first-week sweep on one
detector before fusion (`src/models/nms_sweep.py`: +0.0105 PQ, flat from 0.01 to 0.15;
`experiments/LOG.md`, 20 Aug), a gain the fused ensemble absorbed (sub005 at 0.05 read the same 0.36
as sub004); in the fused pool, NMS 0.3 to 0.7 move the matchable annotations by at most 5 of 1510
(`src/eda/e218_nms_tighter.py`, ledger §54a). The mismatch itself was found when the system was
reproduced, and then measured end to end on all 707 training images: each fold's detector was
rerun once with NMS off, NMS applied offline at 0.05, 0.1, 0.3, 0.5 and 0.7, and every pool scored
through that fold's out-of-fold selector (`src/eda/nms_sweep_5fold.py`,
`outputs/eda/nms_sweep_5fold.json`). Against 0.5 (PQ 0.4421, which the rebuilt pool reproduces to
−0.0001), the test setting 0.05 reads −0.0016, 95 % CI [−0.0037, +0.0006], 2 of 5 folds up; 0.1,
0.3 and 0.7 read −0.0001, −0.0002 and −0.0003. The axis is flat through the selector and the
mismatch's cost is below the ±0.003 floor, so the submission is unchanged. Out of fold each image
has one detector's candidates and at test time five, so this prices NMS on the out-of-fold pool.
(A fold-0 reading made first, `src/eda/nms_mismatch_f0.py`, gave +0.0007 for 0.05 and +0.0045 for
0.1; the latter does not survive five folds.) On another GPU architecture expect a few encodings to
differ: an earlier rehearsal of the predecessor system on a different card reproduced 1091 of 1117
exactly and changed one instance (±0.0004 PQ), floating-point noise at the keep gate.

**3. Complete retrain (one A100-80GB, about 40 h).** Notebook with `MODE = 'RETRAIN'`: trains the
17 networks from the raw competition data (the five detectors on the medoid annotator copy of each
image, as the shipped ones were), rebuilds and labels the out-of-fold candidate pool, computes its
extra features, refits the bagged selector into a new file and writes the submission. This route
has not been run end to end.

### Network weights

The 17 network checkpoints of the submitted system (829 MB, 43 to 51 MB each) are stored in this
repository, as the competition's Final Submission rules ask, so a clone holds everything the notebook
needs and nothing has to be retrained:

```
outputs/runs/e013_yolo11m_f{0..4}_1792/weights/best.pt          detector, one per fold
outputs/runs_semantic/sem_b3_f{0..4}_1536_cldice/best.pt        semantic lane, base recipe
outputs/runs_aug/bloblocal_f{0..4}_s1536_w1.0_r40/best.pt       semantic lane, instance-balanced
outputs/runs_aug/aug_heavy_f0_{768,1536}_60ep/best.pt           semantic lane, heavy augmentation
```

The same 17 files are mirrored as a release asset (release tag `weights-v1`, file
`artifacts_v1.tar`, unpacked at the repository root); `artifacts_v1.sha256` in that release lists
the hash of every file, and the last cell of the notebook prints the hashes of what is on disk.

## Repository layout

| Path | Content |
|---|---|
| `src/filament/` | The metric (`pq.py`, tested against the organisers' scorer), run-length encoding, shared constants |
| `src/data/` | Cache builders and training-set exports |
| `src/models/` | Training, inference, fusion (`build_fused_pool.py`), selector (`fit_selector_bagged.py`), submission (`submit_v2.py`, `submit_from_pool.py`, `validate_submission.py`) |
| `src/eda/` | One script per measurement, named for the question it answers; indexed in Appendix D of the full report |
| `scripts/` | Chained training runs and the report builders (`build_overleaf_preview.py` checks the LaTeX source and prints a layout preview without a LaTeX engine) |
| `notebooks/` | The end-to-end pipeline notebook |
| `tests/` | Scorer tests, including equality with the organisers' published scorer |
| `data/splits/` | The frozen fold assignment (`folds_v1.parquet`), grouped by physical image |
| `outputs/eda/`, `outputs/selector/` | Committed measurement results (JSON, Markdown, small tables) that the reports cite by path |
| `outputs/submissions/` | The final submission, the prepared alternative and rehearsal rebuilds |
| `report/` | The technical report and the full report, English and Turkish; `report/overleaf/` is the Overleaf project of the 4-page report (organisers' template, bibliography, figures) |
| `docs/report/` | Source of the full report |
| `docs/15_ABLATION_LEDGER.md`, `experiments/` | The working record of every experiment and of our own mistakes, as it was kept during the competition (in Turkish) |

Run the tests with `python -m pytest tests`.

## Data and compliance

- **Labels.** Every label used for training, validation or selection is provided by the
  competition. The public MAGFiLO 1.0 release contains annotations for all 180 test images; we
  verified this at the identifier level in the first week, quarantined the release outside version
  control, and never used it for training, model selection or prediction.
- **External data.** The submitted models use ImageNet encoder weights and COCO-pretrained detector
  weights and nothing else. Unlabelled GONG frames, SDO/HMI magnetograms and SDO/AIA 304 frames
  appear only in experiments that were measured and not adopted; inference reads only the provided
  H-alpha images, and the submission path refuses a checkpoint that would read anything else.
- **The competition data are not redistributed here.** `outputs/eda/test_pool_t020pox.parquet`
  holds our own candidate masks and features for the test images, not images or labels.
- **AI assistance.** AI tools were used for coding and drafting, as the organisers permit.

## About this snapshot

This repository is a cleaned snapshot of the development repository at its final state. Internal
planning notes, machine-specific run scripts and third-party material were left out; the source
code and the measurement results are otherwise unchanged. The full report refers to two
tags of the development history (`solar-final-analysis-v1`, `solar-final-analysis-v2`); this
snapshot corresponds to the second. Added for this release: the 4-page technical report with
its figure and measurement scripts (`src/eda/report_pipeline_figure.py`,
`report_examples_figure.py`, `pq_uncertainty.py`, `matched_iou_dice_distribution.py`), the
detector NMS measurements (`src/eda/nms_mismatch_f0.py`, `src/eda/nms_sweep_5fold.py`), the small artefacts that make the final
stage reproducible from a clone, and a notebook that reproduces the final submission.
The full report was written on 2026-10-04, when two final submissions were planned (sub036 and
sub042); the competition page allowed one selection, and the selected one is sub036. Where the
reproduction check of this release found the full report wrong (the detectors' training targets and
optimizer, the test-time detector NMS, the single final selection), its sources were corrected
(`git log -- docs/report`).

## Licence

MIT, see [`LICENSE`](LICENSE). Pretrained weights from third parties (ImageNet, COCO) remain under
their own licences.
