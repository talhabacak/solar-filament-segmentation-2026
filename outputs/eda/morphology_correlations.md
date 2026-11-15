# Morphology correlation study

n = 8199 annotations across 707 files; fixed disk geometry center=(1023.5,1023.5) R=900.0.

## 1. Boundary complexity P^2/(4piA) by category, size-controlled

Hypothesis: Unidentifiable (U) drawn with less boundary detail than L/R at matched size.

| area quartile | med(L) | med(R) | med(U) | med(L+R) | U vs L+R Mann-Whitney p | U lower? |
|---|---|---|---|---|---|---|
| Q1 (n=2050) | 5.036 | 4.947 | 2.807 | 4.992 | 2.52e-140 | YES |
| Q2 (n=2050) | 6.041 | 6.135 | 3.055 | 6.076 | 5.95e-152 | YES |
| Q3 (n=2049) | 7.791 | 7.646 | 3.498 | 7.696 | 5.74e-150 | YES |
| Q4 (n=2050) | 9.894 | 10.880 | 4.170 | 10.242 | 1.69e-144 | YES |

Overall medians: L=7.095 R=7.235 U=3.164; U vs L+R one-sided p=0.00e+00.

Vertex density (polygon points per px of perimeter), median: Left=0.2193, Right=0.2215, Unidentifiable=0.1752

## 2. Spine skeleton statistics

Tortuosity = arclength/endpoint distance; net turning signed with canonical W->E traversal (positive = bows toward solar north).

| category       |   tortuosity |   mean_abs_turn_deg |
|:---------------|-------------:|--------------------:|
| Left           |        1.06  |              12.737 |
| Right          |        1.061 |              12.318 |
| Unidentifiable |        1.041 |              11.958 |

Hemispheric chirality rule: Right fraction in N = 0.690 (n=2474), Left fraction in S = 0.668 (n=2633); chi2 p=2.454e-143.

Within-hemisphere chirality vs bow-north chi2 p: N=3.916e-04, S=3.445e-01.

Bow-north fraction: Left=0.443 (n=2526), Right=0.522 (n=2581); chi2 p=1.434e-08.

Bow-north fraction: N-hemisphere=0.555 (n=3971), S=0.422 (n=4037); chi2 p=3.512e-32.

Hemisphere x chirality bow-north fractions:
|                |   mean |   size |
|:---------------|-------:|-------:|
| ('N', 'Left')  |  0.484 |    768 |
| ('N', 'Right') |  0.562 |   1706 |
| ('S', 'Left')  |  0.424 |   1758 |
| ('S', 'Right') |  0.445 |    875 |

## 3. Orientation vs latitude (tilt)

tilt_solar_deg = -PCA orientation (y-up convention); E-W aligned = |tilt| <= 30 deg.
Elongated subset: eccentricity >= 0.85 (n=7585 of 8199).

| |lat| bin | n | median |tilt| (deg) | frac E-W aligned | n elong | frac E-W (elong) |
|---|---|---|---|---|---|
| 0-15 | 2920 | 45.3 | 0.317 | 2701 | 0.313 |
| 15-30 | 3545 | 42.0 | 0.320 | 3308 | 0.317 |
| 30-45 | 1493 | 32.2 | 0.473 | 1357 | 0.480 |
| 45+ | 241 | 18.9 | 0.651 | 219 | 0.680 |

Spearman |tilt| vs |lat| (all): rho=-0.154 p=1.64e-44.

Signed tilt by hemisphere (elongated subset):

| |lat| bin | med tilt N | n N | med tilt S | n S |
|---|---|---|---|---|
| 0-15 | -28.4 | 1511 | 18.6 | 1190 |
| 15-30 | -32.2 | 1596 | 33.4 | 1712 |
| 30-45 | -16.8 | 610 | 23.2 | 747 |
| 45+ | -4.9 | 69 | -7.0 | 150 |

Spearman signed tilt vs signed lat (elongated): rho=-0.323 p=2.13e-183 (Joy's-law-like antisymmetry would give rho != 0).

## 4. Category (L/R/U) vs position and size

Proportions per |lat| bin:
| lat_bin   |   Left |   Right |   Unidentifiable |
|:----------|-------:|--------:|-----------------:|
| 0-15      |  0.289 |   0.345 |            0.366 |
| 15-30     |  0.339 |   0.327 |            0.335 |
| 30-45     |  0.297 |   0.249 |            0.454 |
| 45+       |  0.199 |   0.216 |            0.585 |

Proportions per r_norm bin:
| r_bin   |   Left |   Right |   Unidentifiable |
|:--------|-------:|--------:|-----------------:|
| 0-.3    |  0.305 |   0.355 |            0.34  |
| .3-.5   |  0.327 |   0.343 |            0.33  |
| .5-.7   |  0.306 |   0.317 |            0.377 |
| .7-.85  |  0.309 |   0.276 |            0.416 |
| .85+    |  0.261 |   0.29  |            0.449 |

U fraction by area decile (median area px^2 -> frac U):

316:0.64 | 491:0.56 | 670:0.46 | 860:0.38 | 1087:0.35 | 1386:0.31 | 1791:0.29 | 2434:0.26 | 3555:0.20 | 6942:0.28

Spearman is_U vs log area: rho=-0.258 p=2.46e-124.
Median area: U=869 px^2, L/R=1491 px^2.

Spearman is_U vs r_norm: rho=0.078 p=1.32e-12; is_U vs |lat|: rho=0.059 p=8.83e-08.

## 5. Spearman correlation matrix

(file_disagree_norm = per-file annotator count-disagreement proxy, multi-copy files only; pairwise-complete.)

|                     |   log_area |   width_proxy |   spine_len |   tortuosity |   mean_abs_turn_deg |   boundary_complexity |   eccentricity |   r_norm |   abs_lat_deg |   is_U |   file_disagree_norm |   cluster_support |   mean_pair_iou |
|:--------------------|-----------:|--------------:|------------:|-------------:|--------------------:|----------------------:|---------------:|---------:|--------------:|-------:|---------------------:|------------------:|----------------:|
| log_area            |      1     |         0.706 |       0.878 |        0.169 |              -0.024 |                 0.489 |          0.182 |    0.189 |         0.204 | -0.258 |               -0.033 |             0.279 |           0.038 |
| width_proxy         |      0.706 |         1     |       0.309 |       -0.137 |              -0.069 |                -0.147 |         -0.256 |    0.283 |         0.27  |  0.032 |                0.021 |             0.112 |           0.116 |
| spine_len           |      0.878 |         0.309 |       1     |        0.322 |               0.012 |                 0.776 |          0.431 |    0.073 |         0.1   | -0.377 |               -0.062 |             0.312 |          -0.028 |
| tortuosity          |      0.169 |        -0.137 |       0.322 |        1     |               0.543 |                 0.374 |         -0.249 |   -0.038 |        -0.049 | -0.181 |               -0.054 |             0.118 |          -0.072 |
| mean_abs_turn_deg   |     -0.024 |        -0.069 |       0.012 |        0.543 |               1     |                 0.072 |         -0.193 |    0.005 |        -0.007 | -0.049 |                0.087 |            -0.054 |          -0.143 |
| boundary_complexity |      0.489 |        -0.147 |       0.776 |        0.374 |               0.072 |                 1     |          0.458 |   -0.089 |        -0.05  | -0.619 |               -0.073 |             0.266 |          -0.124 |
| eccentricity        |      0.182 |        -0.256 |       0.431 |       -0.249 |              -0.193 |                 0.458 |          1     |   -0.072 |        -0.023 | -0.124 |               -0.042 |             0.127 |          -0.037 |
| r_norm              |      0.189 |         0.283 |       0.073 |       -0.038 |               0.005 |                -0.089 |         -0.072 |    1     |         0.453 |  0.078 |                0.056 |            -0.122 |           0.005 |
| abs_lat_deg         |      0.204 |         0.27  |       0.1   |       -0.049 |              -0.007 |                -0.05  |         -0.023 |    0.453 |         1     |  0.059 |                0.025 |            -0.028 |          -0     |
| is_U                |     -0.258 |         0.032 |      -0.377 |       -0.181 |              -0.049 |                -0.619 |         -0.124 |    0.078 |         0.059 |  1     |                0.033 |            -0.24  |          -0.01  |
| file_disagree_norm  |     -0.033 |         0.021 |      -0.062 |       -0.054 |               0.087 |                -0.073 |         -0.042 |    0.056 |         0.025 |  0.033 |                1     |            -0.273 |          -0.111 |
| cluster_support     |      0.279 |         0.112 |       0.312 |        0.118 |              -0.054 |                 0.266 |          0.127 |   -0.122 |        -0.028 | -0.24  |               -0.273 |             1     |           0.071 |
| mean_pair_iou       |      0.038 |         0.116 |      -0.028 |       -0.072 |              -0.143 |                -0.124 |         -0.037 |    0.005 |        -0     | -0.01  |               -0.111 |             0.071 |           1     |

Strongest pairs: log_area~spine_len=+0.878; spine_len~boundary_complexity=+0.776; log_area~width_proxy=+0.706; boundary_complexity~is_U=-0.619; tortuosity~mean_abs_turn_deg=+0.543; log_area~boundary_complexity=+0.489; boundary_complexity~eccentricity=+0.458; r_norm~abs_lat_deg=+0.453; spine_len~eccentricity=+0.431; spine_len~is_U=-0.377

## 6. Detectability: fold-0 GT matched vs missed (E010, conf=0.4, min_area=128, r<=0.97)

Matched 757 / 1510 GT instances (recall proxy 0.501).

| feature | mean(matched) | mean(missed) | Cohen's d | Spearman(feat, matched) |
|---|---|---|---|---|
| log_area | 3.265 | 3.064 | +0.499 | +0.256 |
| width_proxy | 14.347 | 12.606 | +0.255 | +0.197 |
| spine_len | 173.063 | 141.746 | +0.236 | +0.218 |
| tortuosity | 1.113 | 1.112 | +0.007 | +0.029 |
| mean_abs_turn_deg | 12.939 | 14.242 | -0.156 | -0.032 |
| boundary_complexity | 6.763 | 6.218 | +0.116 | +0.111 |
| eccentricity | 0.952 | 0.941 | +0.164 | +0.075 |
| r_norm | 0.554 | 0.589 | -0.173 | -0.094 |
| abs_lat_deg | 20.090 | 21.720 | -0.135 | -0.054 |
| is_U | 0.346 | 0.499 | -0.314 | -0.155 |

Logistic 5-fold CV AUC = 0.706 +/- 0.023.
Standardized logit coefficients: log_area=+1.74, width_proxy=-0.63, boundary_complexity=-0.59, spine_len=-0.52, r_norm=-0.30, is_U=-0.24, mean_abs_turn_deg=-0.14, abs_lat_deg=-0.14, eccentricity=-0.03, tortuosity=-0.02
