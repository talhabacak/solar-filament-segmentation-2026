# Solar Filament Segmentation Challenge 2026 — team TalhaBacak

Code, measurements and reports behind our final submission to the
[Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(instance segmentation of filaments on full-disc GONG H-alpha images, scored by Panoptic Quality).

| | |
|---|---|
| Final submission | `outputs/submissions/sub036_t020pox_bag5.csv` (1117 instances) |
| Prepared alternative, not selected | `outputs/submissions/sub042_t020pox_bag5_cascade085.csv` (1182 instances; sub036 plus a marginal-band cascade) |
| Local score | PQ 0.443, SQ 0.678, RQ 0.653 — pooled five-fold out-of-fold, all 707 training images |
| Public leaderboard | 0.40 |
| Technical report (4 pages) | [`report/Technical_Report_EN.pdf`](report/Technical_Report_EN.pdf), compiled from the organisers' template ([`report/overleaf/main.tex`](report/overleaf/main.tex)) · Turkish translation: [`report/Technical_Report_TR.pdf`](report/Technical_Report_TR.pdf) |
| Full report (62 pages) | [`report/Full_Report_EN.pdf`](report/Full_Report_EN.pdf) · Turkish: [`report/Full_Report_TR.pdf`](report/Full_Report_TR.pdf) |
| Pipeline notebook | [`notebooks/00_end_to_end_pipeline.ipynb`](notebooks/00_end_to_end_pipeline.ipynb) |

## The system in brief

```
GONG H-alpha frame (2048 x 2048)
  |-- detector lane   YOLO11m-seg @1792, one model per fold, all annotator copies
  |-- semantic lane   U-Net (EfficientNet-B3) @1536, BCE + Dice + clDice, 12-member ensemble
  +-> fusion          cluster at IoU > 0.5, confidence-weighted vote, pixel-disjoint masks
       -> 42 features per candidate
       -> selector    two-stage gradient boosting, mean of 5 image-bootstrap copies
       -> keep        conf >= 0.40 and p_sel >= 0.60 (derived from the PQ formula)
       -> submission  run-length encodings
```

Every parameter is listed in Table 1 of the technical report and set in the first code cell of the
notebook.

## Reproducing the submitted predictions

Install the pinned environment (Python 3.13, CUDA 12.8):

```bash
pip install -r requirements.txt
```

There are three levels, and the notebook runs each of them (`MODE` in its first code cell).

**1. Final stage, from this repository alone (CPU, seconds).** The repository ships the featurised
test candidate pool and the fitted selector, so the final submission (and the prepared alternative)
can be rebuilt and compared with the uploaded files byte for byte:

```bash
python src/models/submit_from_pool.py --pool outputs/eda/test_pool_t020pox.parquet \
    --selector outputs/selector/selector_t020pox_bag5.joblib --conf 0.40 --p-min 0.60 \
    --out outputs/submissions/final1_reproduced.csv \
    --check outputs/submissions/sub036_t020pox_bag5.csv

python src/models/submit_from_pool.py --pool outputs/eda/test_pool_t020pox.parquet \
    --selector outputs/selector/selector_t020pox_bag5.joblib --conf 0.40 --p-min 0.60 \
    --cascade-scores outputs/eda/bag5_oof_scores.parquet --cascade-q 0.85 \
    --out outputs/submissions/final2_reproduced.csv \
    --check outputs/submissions/sub042_t020pox_bag5_cascade085.csv
```

Both commands end with `byte-identical ...: True`. (Submission files are checked out with the
platform's line ending, the one pandas writes; the uploaded files used CRLF and have SHA-256
`63b570002a2be27d…` and `c33fd83cc406854d…`.) The selector itself can be refitted from the
shipped labelled training pool in about a minute (`python src/models/fit_selector_bagged.py`); the
refit also rebuilds the first submission byte for byte.

**2. Full inference from the released weights (one 16 GB GPU, about 1.5 h).** Download the
competition data into `data/` (so that `data/MAGFiLO_1.0_Kaggle_2026/` exists), unpack the released
weights at the repository root (see below), then run the notebook with `MODE = 'REPRODUCE'`. It
runs the five detectors and the twelve semantic members on the test images, fuses and featurises
the candidates, and then runs the final stage above. Rebuilt from checkpoints, the predictions are
identical on the machine that produced them; on a different GPU architecture a rehearsal reproduced
1091 of 1117 encodings exactly and changed one instance (±0.0004 PQ), which is floating-point
noise at the keep gate (report, Section 7).

**3. Complete retrain (one A100-80GB, about 40 h).** Notebook with `MODE = 'RETRAIN'`: trains the
17 networks from the raw competition data, rebuilds the out-of-fold candidate pool, refits the
selector and writes the submissions.

### Released weights

The 17 network checkpoints (0.9 GB) are not in the Git history. They are published as a release
asset of this repository (release tag `weights-v1`, file `artifacts_v1.tar`); unpacking it at the
repository root restores

```
outputs/runs/e013_yolo11m_f{0..4}_1792/weights/best.pt          detector, one per fold
outputs/runs_semantic/sem_b3_f{0..4}_1536_cldice/best.pt        semantic lane, base recipe
outputs/runs_aug/bloblocal_f{0..4}_s1536_w1.0_r40/best.pt       semantic lane, instance-balanced
outputs/runs_aug/aug_heavy_f0_{768,1536}_60ep/best.pt           semantic lane, heavy augmentation
```

`artifacts_v1.sha256` in the same release lists the hash of every file, and the last cell of the
notebook prints the hashes of what is on disk.

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
code, the measurement results and the report sources are unchanged. The full report refers to two
tags of the development history (`solar-final-analysis-v1`, `solar-final-analysis-v2`); this
snapshot corresponds to the second. Added for this release: the 4-page technical report with
its three figure and uncertainty scripts (`src/eda/report_pipeline_figure.py`,
`report_examples_figure.py`, `pq_uncertainty.py`, `matched_iou_dice_distribution.py`), the small artefacts that make the final stage
reproducible from a clone, and a notebook that reproduces the final submission and the prepared alternative.
The full report (written 2026-10-04) plans two final submissions, sub036 and sub042; the
competition page allowed one selection, and the selected one is sub036.

## Licence

MIT, see [`LICENSE`](LICENSE). Pretrained weights from third parties (ImageNet, COCO) remain under
their own licences.
