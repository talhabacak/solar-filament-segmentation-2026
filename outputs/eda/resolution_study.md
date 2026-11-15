# Resolution degradation study — MAGFiLO 2048px masks

*Generated 2026-08-19 19:45 by `src/eda/resolution_study.py`*

- Sampling: all 1154 entries
- Entries analyzed: 1154, instances: 8199 (skipped bad segmentation: 0, zero-area at 2048: 0)
- Method: cv2.resize INTER_AREA on binary mask, threshold 0.5; 8-connectivity components; skimage skeletonize; width = 2*L2 dist on skeleton

Note: threshold is inclusive (coverage >= 0.5 -> foreground), so exactly-half covered pixels survive; total foreground can be preserved or even slightly inflated while thin structure is still destroyed (see skeleton/fragmentation).

## Per-scale instance damage (label downsample, INTER_AREA + 0.5 threshold)

| scale | % vanished | % fragmented (1→N comp, among survivors) | median skeleton retention | total fg px lost (union) |
|---|---|---|---|---|
| 1024 | 0.00% | 0.04% | 0.955 | -4.58% |
| 768 | 0.00% | 0.78% | 0.928 | -0.72% |
| 512 | 0.00% | 2.49% | 0.862 | -2.39% |

- Instances with >1 component already at 2048: 0.37%
- Local width at 2048 (px): median of per-instance mean width = 8.87, p10 = 5.70, median min-width = 2.00

## By instance area quartile (area at 2048)

Quartile edges (area px): [9–670] / [670–1228] / [1228–2438] / [2438–37739]

| quartile | metric | 1024 | 768 | 512 |
|---|---|---|---|---|
| Q1 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q1 | pct_fragmented | 0.05% | 0.98% | 3.07% |
| Q1 | median_skel_retention | 0.952 | 0.921 | 0.832 |
| Q2 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q2 | pct_fragmented | 0.00% | 0.83% | 2.40% |
| Q2 | median_skel_retention | 0.948 | 0.919 | 0.853 |
| Q3 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q3 | pct_fragmented | 0.05% | 0.54% | 2.35% |
| Q3 | median_skel_retention | 0.958 | 0.928 | 0.862 |
| Q4 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q4 | pct_fragmented | 0.05% | 0.78% | 2.11% |
| Q4 | median_skel_retention | 0.958 | 0.938 | 0.888 |

## By instance width quartile (mean local width at 2048)

Quartile edges (width px): [2–7] / [7–9] / [9–12] / [12–57]

| quartile | metric | 1024 | 768 | 512 |
|---|---|---|---|---|
| Q1 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q1 | pct_fragmented | 0.05% | 2.26% | 6.78% |
| Q1 | median_skel_retention | 0.942 | 0.902 | 0.813 |
| Q2 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q2 | pct_fragmented | 0.05% | 0.49% | 1.95% |
| Q2 | median_skel_retention | 0.952 | 0.921 | 0.849 |
| Q3 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q3 | pct_fragmented | 0.00% | 0.24% | 0.69% |
| Q3 | median_skel_retention | 0.957 | 0.931 | 0.872 |
| Q4 | pct_vanished | 0.00% | 0.00% | 0.00% |
| Q4 | pct_fragmented | 0.05% | 0.15% | 0.54% |
| Q4 | median_skel_retention | 0.964 | 0.951 | 0.908 |

## Inter-instance merge risk (same-entry instance pairs, 8-connected touch)

Pairs checked: 36064; already touching at 2048: 10

| scale | touching pairs | NEWLY touching pairs | entries affected | % of entries |
|---|---|---|---|---|
| 1024 | 22 | 12 | 10 | 0.87% |
| 768 | 23 | 14 | 12 | 1.04% |
| 512 | 32 | 23 | 20 | 1.73% |

![figure](resolution_study.png)
