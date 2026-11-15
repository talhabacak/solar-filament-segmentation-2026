# False-negative profile at the current operating point

Operating point: `conf >= 0.3` and `p_sel >= 0.35` on the pooled 5-fold OOF candidates of E013@1792 (de-overlapped).

- exact scorer: **PQ 0.4105**, SQ 0.6641, TP 4768, FP 2461, FN 3431 over 8199 GT rows (4441 kept predictions)
- overall instance recall **0.5815** (4768/8199)
- raw candidate pool (conf>=0.05, de-overlapped) reaches 5671/8199 GT rows = recall ceiling **0.6917**; 2528 GT rows are not matched by ANY candidate at any threshold

## Headline

1. **Faintness is the strongest single physical driver.** Recall goes 0.370 -> 0.726 across contrast quartiles (n=8199); point-biserial r = 0.270, Cohen's d = 0.569.
2. **Size is second.** Recall 0.399 -> 0.664 across mask-area quartiles; the two interact - small AND faint is the worst cell.
3. **Annotator consensus dominates everything.** Recall by support k of N copies: 1/3 = 0.173 vs 3/3 = 0.739. Much of the FN mass is filaments only one annotator saw.
4. **We are NOT missing the stringy ones - we miss the compact ones.** Cohen's d of solidity is -0.177 (matched are LESS solid) and n_cc is 1 for 8169/8199 GT masks, so fragmentation is not a real axis in this dataset.
5. **903 of 3431 FN are already in the raw candidate pool** and were thrown away by the operating point (selector/threshold problem); 2528 are never proposed by the detector at any threshold (recall ceiling 0.6917).
6. **Greedy value of perfect recall**: recovering the tiers that pay, in descending order, stops after N=2,k=1 at PQ 0.5278 (+0.1173), recovering 2591 of 3431 FN. Chasing the remaining 840 LOWERS PQ.

## Feature discrimination (matched vs missed)


`mean_pos`/`median_pos` = matched, `mean_neg`/`median_neg` = missed. Positive Cohen's d = the feature is larger on filaments we find.

| feature        | label                    |    n |    mean_pos |     mean_neg |   median_pos |   median_neg |   cohens_d |   r_pointbiserial |         p_pb |   spearman |
|:---------------|:-------------------------|-----:|------------:|-------------:|-------------:|-------------:|-----------:|------------------:|-------------:|-----------:|
| mean_pair_iou  | cluster mean pair IoU    | 3966 |    0.598577 |    0.508159  |     0.608357 |    0.50947   |  0.847342  |         0.362215  | 3.03603e-123 |  0.353352  |
| support_frac   | support k/N              | 8199 |    0.955956 |    0.836005  |     1        |    1         |  0.614525  |         0.290145  | 8.97713e-159 |  0.282171  |
| contrast       | contrast (bg-fg)/bg      | 8199 |    0.123162 |    0.0991669 |     0.119209 |    0.0939753 |  0.568681  |         0.270138  | 4.34935e-137 |  0.280739  |
| contrast_v4    | contrast (8px ring)      | 8199 |    0.127339 |    0.10501   |     0.124102 |    0.100366  |  0.565425  |         0.268703  | 1.33229e-135 |  0.277619  |
| support        | annotator support k      | 8199 |    1.88779  |    1.49752   |     2        |    1         |  0.486237  |         0.233275  | 9.56792e-102 |  0.23703   |
| log10_area     | log10 mask area          | 8199 |    3.18355  |    3.02855   |     3.15229  |    2.98182   |  0.386856  |         0.187478  | 9.61756e-66  |  0.200125  |
| depth          | depth vs disk median     | 8199 |    0.107402 |    0.0916987 |     0.103746 |    0.0843911 |  0.222536  |         0.109136  | 3.75785e-23  |  0.125887  |
| fg_min         | fg min intensity         | 8199 |   89.0508   |   91.7616    |    89        |   93         | -0.218857  |        -0.107353  | 1.89402e-22  | -0.120645  |
| width_proxy    | width proxy (area/spine) | 8199 |   13.3508   |   11.8903    |    11.6261   |   10.2044    |  0.216263  |         0.106095  | 5.83321e-22  |  0.148905  |
| n_poly_points  | polygon vertices         | 8199 |   84.7909   |   71.1183    |    68        |   49         |  0.210948  |         0.103516  | 5.61365e-21  |  0.186892  |
| fg_mean        | fg mean intensity        | 8199 |  117.766    |  119.758     |   118.368    |  120.895     | -0.209661  |        -0.102891  | 9.63382e-21  | -0.119791  |
| fg_std         | fg std                   | 8199 |   11.2148   |   10.6213    |    10.9913   |   10.2006    |  0.201692  |         0.099019  | 2.54686e-19  |  0.12545   |
| ring_std       | ring std                 | 8199 |    8.33217  |    9.01884   |     7.69629  |    8.02357   | -0.1979    |        -0.0971755 | 1.15882e-18  | -0.0797443 |
| solidity       | solidity                 | 8199 |    0.569897 |    0.605456  |     0.560491 |    0.598608  | -0.177179  |        -0.087082  | 2.81285e-15  | -0.0828814 |
| n_gt_entry     | GT in this entry         | 8199 |    9.47106  |   10.2504    |     9        |   10         | -0.17573   |        -0.0863754 | 4.70333e-15  | -0.0880751 |
| n_kept_img     | kept preds in image      | 8199 |    8.29404  |    7.71408   |     8        |    8         |  0.173997  |         0.0855299 | 8.65329e-15  |  0.0792573 |
| bg_mean        | ring mean intensity      | 8199 |  134.32     |  132.95      |   135.739    |  134.97      |  0.156521  |         0.0769931 | 2.93307e-12  |  0.0603145 |
| mask_bbox_diag | bbox diagonal            | 8199 |  151.332    |  135.805     |   123.116    |   94.0479    |  0.14289   |         0.0703227 | 1.83529e-10  |  0.183158  |
| r_norm         | radial position r/R      | 8199 |    0.557755 |    0.582806  |     0.5739   |    0.60249   | -0.125053  |        -0.0615802 | 2.39929e-08  | -0.0618106 |
| eccentricity   | eccentricity             | 8199 |    0.948363 |    0.939877  |     0.969052 |    0.965411  |  0.124219  |         0.0611706 | 2.96914e-08  |  0.0439049 |
| spine_len      | spine length             | 8199 |  153.317    |  139.251     |   123.328    |   92.8586    |  0.118508  |         0.0583685 | 1.23071e-07  |  0.171984  |
| mask_area      | mask pixel area          | 8199 | 2253.96     | 1933.44      |  1420        |  959         |  0.117095  |         0.0576745 | 1.73323e-07  |  0.200125  |
| edge_grad      | boundary |grad|          | 8199 |   46.1578   |   44.4422    |    43.1991   |   40.8166    |  0.0994716 |         0.0490171 | 8.97896e-06  |  0.0775617 |
| n_copies       | annotator copies N       | 8199 |    2.00482  |    1.935     |     2        |    2         |  0.080297  |         0.0395849 | 0.000336815  |  0.0399423 |
| n_holes        | hole count               | 8199 |    0.15625  |    0.122413  |     0        |    0         |  0.0738848 |         0.0364282 | 0.000969953  |  0.0343467 |

## Recall by category

| level          |    n |   tp |   recall |
|:---------------|-----:|-----:|---------:|
| Left           | 2535 | 1571 | 0.619724 |
| Right          | 2590 | 1715 | 0.662162 |
| Unidentifiable | 3074 | 1482 | 0.482108 |

## Recall by consensus_tier

| level   |    n |   tp |   recall |
|:--------|-----:|-----:|---------:|
| 1/1     | 3206 | 1762 | 0.549595 |
| 1/2     |  518 |  144 | 0.277992 |
| 1/3     |  509 |   88 | 0.172888 |
| 2/2     | 1469 | 1077 | 0.733152 |
| 2/3     |  523 |  238 | 0.455067 |
| 3/3     | 1974 | 1459 | 0.739108 |

## Recall by support

|   level |    n |   tp |   recall |
|--------:|-----:|-----:|---------:|
|       1 | 4233 | 1994 | 0.471061 |
|       2 | 1992 | 1315 | 0.660141 |
|       3 | 1974 | 1459 | 0.739108 |

## Recall by n_copies

|   level |    n |   tp |   recall |
|--------:|-----:|-----:|---------:|
|       1 | 3206 | 1762 | 0.549595 |
|       2 | 1987 | 1221 | 0.614494 |
|       3 | 3006 | 1785 | 0.593812 |

## Recall by radial_zone

| level   |    n |   tp |   recall |
|:--------|-----:|-----:|---------:|
| 0-.35   | 1365 |  833 | 0.610256 |
| .35-.7  | 4305 | 2578 | 0.598839 |
| .7-.9   | 2422 | 1341 | 0.553675 |
| .9-1    |  107 |   16 | 0.149533 |

## Recall by size_quartile

| level    |    n |   tp |   recall |
|:---------|-----:|-----:|---------:|
| Q1_small | 2054 |  819 | 0.398734 |
| Q2       | 2046 | 1242 | 0.607038 |
| Q3       | 2049 | 1360 | 0.663738 |
| Q4_large | 2050 | 1347 | 0.657073 |

## Recall by width_quartile

| level    |    n |   tp |   recall |
|:---------|-----:|-----:|---------:|
| Q1_thin  | 2050 |  925 | 0.45122  |
| Q2       | 2050 | 1223 | 0.596585 |
| Q3       | 2049 | 1274 | 0.621767 |
| Q4_thick | 2050 | 1346 | 0.656585 |

## Recall by contrast_quartile

| level    |    n |   tp |   recall |
|:---------|-----:|-----:|---------:|
| Q1_faint | 2050 |  758 | 0.369756 |
| Q2       | 2050 | 1152 | 0.561951 |
| Q3       | 2049 | 1370 | 0.668619 |
| Q4_dark  | 2050 | 1488 | 0.725854 |

## Recall by site

| level        |    n |   tp |   recall |
|:-------------|-----:|-----:|---------:|
| Big Bear     | 1502 |  895 | 0.595872 |
| Cerro Tololo | 1522 |  871 | 0.572273 |
| Learmonth    | 1438 |  803 | 0.558414 |
| Mauna Loa    | 1298 |  747 | 0.575501 |
| Teide        | 1369 |  799 | 0.583638 |
| Udaipur      | 1070 |  653 | 0.61028  |

## Recall by year

|   level |    n |   tp |   recall |
|--------:|-----:|-----:|---------:|
|    2011 | 1066 |  612 | 0.574109 |
|    2012 | 1297 |  750 | 0.578258 |
|    2013 | 1368 |  807 | 0.589912 |
|    2014 | 1607 |  953 | 0.59303  |
|    2015 | 1266 |  752 | 0.593997 |
|    2016 |  826 |  476 | 0.576271 |
|    2017 |  385 |  229 | 0.594805 |
|    2018 |   66 |   39 | 0.590909 |
|    2019 |   43 |   16 | 0.372093 |
|    2020 |   30 |   12 | 0.4      |
|    2021 |  121 |   61 | 0.504132 |
|    2022 |  124 |   61 | 0.491935 |

## Recall by gt_per_image_quintile

| level        |    n |   tp |   recall |
|:-------------|-----:|-----:|---------:|
| (0.999, 9.0] | 1942 | 1162 | 0.598352 |
| (9.0, 13.0]  | 1480 |  818 | 0.552703 |
| (13.0, 19.0] | 1760 |  996 | 0.565909 |
| (19.0, 27.0] | 1414 |  866 | 0.612447 |
| (27.0, 58.0] | 1603 |  926 | 0.577667 |

## Recall by n_cc

|   level |    n |   tp |   recall |
|--------:|-----:|-----:|---------:|
|       1 | 8169 | 4751 | 0.581589 |
|       2 |   30 |   17 | 0.566667 |

## Recall by n_holes

|   level |    n |   tp |   recall |
|--------:|-----:|-----:|---------:|
|       0 | 7282 | 4192 | 0.575666 |
|       1 |  736 |  452 | 0.61413  |
|       2 |  135 |   95 | 0.703704 |
|       3 |   34 |   21 | 0.617647 |
|       4 |    9 |    6 | 0.666667 |
|       5 |    1 |    0 | 0        |
|       6 |    1 |    1 | 1        |
|      10 |    1 |    1 | 1        |

## Recall by full_support

| level   |    n |   tp |   recall |
|:--------|-----:|-----:|---------:|
| False   | 1550 |  470 | 0.303226 |
| True    | 6649 | 4298 | 0.646413 |

## Which misses matter

- FN total 3431: 3012 sit in 2581 clusters where **no** copy was matched (recoverable by a new detection); 419 sit in 354 clusters where at least one copy WAS matched (the prediction exists but its IoU with this copy is <=0.5 - a segmentation-quality problem, not recoverable by adding a disjoint mask).
- typical matched IoU used for the marginal arithmetic: 0.6641374867809499

Marginal rule (exact): adding a detection of a cluster drawn by k of N copies changes the numerator by IoU*k and the denominator by N/2, so it pays while `PQ < 2*IoU*k/N`. The briefed form `PQ < 2*IoU*k/(N+k)` ignores the 0.5*k FN credit that disappears and is therefore conservative; both are reported.

| rule                    |   n_clusters_worth |   n_fn_worth |   n_clusters_skip |   n_fn_skip |   pq_if_perfect_on_worth |   tp_after |   fp_after |   fn_after |   delta_pq |
|:------------------------|-------------------:|-------------:|------------------:|------------:|-------------------------:|-----------:|-----------:|-----------:|-----------:|
| marginal (2*IoU*k/N)    |               2581 |         3012 |                 0 |           0 |                 0.522367 |       7780 |       3804 |        419 |   0.111865 |
| briefed (2*IoU*k/(N+k)) |               2160 |         2591 |               421 |         421 |                 0.527796 |       7359 |       2962 |        840 |   0.117294 |

### Fully-missed clusters by (N, k)

|   N |   k |   n_clusters |   n_fn | worth_marginal   | worth_brief   |   thr_marginal |   thr_brief |
|----:|----:|-------------:|-------:|:-----------------|:--------------|---------------:|------------:|
|   1 |   1 |         1444 |   1444 | True             | True          |       1.32827  |    0.664137 |
|   2 |   1 |          374 |    374 | True             | True          |       0.664137 |    0.442758 |
|   2 |   2 |          129 |    258 | True             | True          |       1.32827  |    0.664137 |
|   3 |   1 |          421 |    421 | True             | False         |       0.442758 |    0.332069 |
|   3 |   2 |          127 |    255 | True             | True          |       0.886388 |    0.53131  |
|   3 |   3 |           86 |    260 | True             | True          |       1.32827  |    0.664137 |

### Greedy tier ladder

The marginal threshold is compared against the CURRENT PQ, which rises as misses are recovered - so the rule must be re-checked tier by tier. Tiers added in descending value order:

|   N |   k |   thr_marginal |   pq_before_tier | pays_at_this_pq   |   n_clusters_added |   n_fn_added |   cum_fn_recovered |   pq_after |   tp |   fp |   fn |
|----:|----:|---------------:|-----------------:|:------------------|-------------------:|-------------:|-------------------:|-----------:|-----:|-----:|-----:|
|   1 |   1 |       1.32827  |         0.410501 | True              |               1444 |         1444 |               1444 |   0.48905  | 6212 | 2461 | 1987 |
|   2 |   2 |       1.32827  |         0.48905  | True              |                129 |          258 |               1702 |   0.501689 | 6470 | 2461 | 1729 |
|   3 |   3 |       1.32827  |         0.501689 | True              |                 86 |          260 |               1962 |   0.514048 | 6730 | 2461 | 1469 |
|   3 |   2 |       0.886388 |         0.514048 | True              |                127 |          255 |               2217 |   0.522057 | 6985 | 2588 | 1214 |
|   2 |   1 |       0.664137 |         0.522057 | True              |                374 |          374 |               2591 |   0.527796 | 7359 | 2962 |  840 |
|   3 |   1 |       0.442758 |         0.527796 | False             |                421 |          421 |               3012 |   0.522367 | 7780 | 3804 |  419 |

- greedy optimum: stop after tier N=2,k=1 -> PQ 0.5278 (+0.1173) recovering 2591 FN

### All clusters by (N, k) - cluster-level detection rate

|   N |   k |   n_clusters |   n_gt_rows |   n_fn |   n_fully_missed |   cluster_detection_rate |
|----:|----:|-------------:|------------:|-------:|-----------------:|-------------------------:|
|   1 |   1 |         3206 |        3206 |   1444 |             1444 |                 0.549595 |
|   2 |   1 |          518 |         518 |    374 |              374 |                 0.277992 |
|   2 |   2 |          732 |        1469 |    392 |              129 |                 0.82377  |
|   3 |   1 |          509 |         509 |    421 |              421 |                 0.172888 |
|   3 |   2 |          261 |         523 |    285 |              127 |                 0.51341  |
|   3 |   3 |          655 |        1974 |    515 |               86 |                 0.868702 |

- Perfect recall of ALL fully-missed clusters: PQ 0.5224 (+0.1119)
- Upper bound if every FN row vanished (including the un-recoverable partial-cluster ones): PQ 0.5391

## FN split: dropped by the operating point vs never proposed

| group             |    n |   med_mask_area |   med_width_proxy |   med_spine_len |   med_contrast |   med_depth |   med_edge_grad |   med_fg_min |   med_r_norm |   med_abs_lat |   med_solidity |   med_n_cc |   med_n_holes |   med_eccentricity |   med_aspect_ratio |   med_support |   med_support_frac |   med_n_gt_image |   med_n_copies |
|:------------------|-----:|----------------:|------------------:|----------------:|---------------:|------------:|----------------:|-------------:|-------------:|--------------:|---------------:|-----------:|--------------:|-------------------:|-------------------:|--------------:|-------------------:|-----------------:|---------------:|
| matched(TP)       | 4768 |            1420 |           11.6261 |        123.328  |      0.119209  |   0.103746  |         43.1991 |           89 |     0.5739   |       19.4832 |       0.560491 |          1 |             0 |           0.969052 |            1.64335 |             2 |                  1 |               15 |              2 |
| FN_dropped        |  903 |             761 |           10.6475 |         72.8983 |      0.0941201 |   0.0875361 |         41.8276 |           94 |     0.640025 |       19.8658 |       0.707529 |          1 |             0 |           0.956303 |            1.64713 |             1 |                  1 |               15 |              2 |
| FN_never_proposed | 2528 |            1039 |           10.0218 |        104.411  |      0.0939558 |   0.083158  |         40.519  |           92 |     0.594299 |       19.9199 |       0.565269 |          1 |             0 |           0.968391 |            1.69648 |             1 |                  1 |               16 |              2 |

Reason a reachable FN was dropped: {
 "n": 903,
 "conf_below_only": 315,
 "psel_below_only": 127,
 "both_below": 461,
 "neither_below": 0,
 "conf_quantiles": {
  "0.1": 0.0835041955113411,
  "0.25": 0.1325783059000969,
  "0.5": 0.2015954852104187,
  "0.75": 0.2717067301273346,
  "0.9": 0.3367116391658783
 },
 "psel_quantiles": {
  "0.1": 0.09370472651610706,
  "0.25": 0.1744103349697086,
  "0.5": 0.2964424343119215,
  "0.75": 0.4064104675573721,
  "0.9": 0.5171174781212748
 },
 "pool_iou_quantiles": {
  "0.1": 0.5263198375701904,
  "0.25": 0.5645850300788879,
  "0.5": 0.6270270347595215,
  "0.75": 0.6929214596748352,
  "0.9": 0.7433309674263001
 }
}

### What separates a dropped FN from a never-proposed one

| feature        | label                 |    n |   mean_dropped |   mean_never |   median_dropped |   median_never |   cohens_d |   r_pointbiserial |        p_pb |   spearman |
|:---------------|:----------------------|-----:|---------------:|-------------:|-----------------:|---------------:|-----------:|------------------:|------------:|-----------:|
| solidity       | solidity              | 3431 |      0.692905  |    0.574219  |        0.707529  |       0.565269 |  0.556132  |         0.237936  | 2.29618e-45 |  0.238942  |
| spine_len      | spine length          | 3431 |     95.4748    |  154.888     |       72.8983    |     104.411    | -0.45462   |        -0.196358  | 3.63883e-31 | -0.207824  |
| mean_pair_iou  | cluster mean pair IoU | 1192 |      0.546919  |    0.497399  |        0.553253  |       0.496589 |  0.451143  |         0.183059  | 1.92219e-10 |  0.185121  |
| mask_bbox_diag | bbox diagonal         | 3431 |     97.072     |  149.64      |       75.8222    |     103.824    | -0.443314  |        -0.191657  | 9.61246e-30 | -0.200884  |
| n_poly_points  | polygon vertices      | 3431 |     50.7442    |   78.396     |       38         |      54        | -0.416595  |        -0.180493  | 1.63307e-26 | -0.205512  |
| mask_area      | mask pixel area       | 3431 |   1256.13      | 2175.37      |      761         |    1039        | -0.307682  |        -0.134304  | 2.79455e-15 | -0.127519  |
| log10_area     | log10 mask area       | 3431 |      2.93087   |    3.06345   |        2.88138   |       3.01662  | -0.302905  |        -0.132255  | 7.35768e-15 | -0.127519  |
| eccentricity   | eccentricity          | 3431 |      0.927252  |    0.944387  |        0.956303  |       0.968391 | -0.233424  |        -0.102282  | 1.9167e-09  | -0.12631   |
| n_kept_img     | kept preds in image   | 3431 |      7.21041   |    7.89399   |        7         |       8        | -0.202725  |        -0.0889451 | 1.80495e-07 | -0.0784937 |
| support        | annotator support k   | 3431 |      1.39313   |    1.53481   |        1         |       1        | -0.191625  |        -0.0841102 | 8.06926e-07 | -0.0819039 |
| n_holes        | hole count            | 3431 |      0.0697674 |    0.141218  |        0         |       0        | -0.173853  |        -0.0763571 | 7.55196e-06 | -0.0734046 |
| r_norm         | radial position r/R   | 3431 |      0.608138  |    0.573757  |        0.640025  |       0.594299 |  0.168341  |         0.0739498 | 1.45132e-05 |  0.0794286 |
| bg_mean        | ring mean intensity   | 3431 |    131.888     |  133.329     |      133.995     |     135.171    | -0.149194  |        -0.0655775 | 0.000121084 | -0.0528517 |
| fg_mean        | fg mean intensity     | 3431 |    118.87      |  120.075     |      120.533     |     121.166    | -0.116702  |        -0.0513385 | 0.0026295   | -0.0401366 |
| depth          | depth vs disk median  | 3431 |      0.0979097 |    0.0894801 |        0.0875361 |       0.083158 |  0.109635  |         0.0482372 | 0.00471194  |  0.0351084 |
| n_gt_entry     | GT in this entry      | 3431 |     10.5725    |   10.1353    |       10         |      10        |  0.0954348 |         0.0420012 | 0.0138783   |  0.0522559 |
| n_copies       | annotator copies N    | 3431 |      1.87818   |    1.9553    |        2         |       2        | -0.0877794 |        -0.0386373 | 0.023624    | -0.0399692 |
| n_cc           | connected components  | 3431 |      1         |    1.00514   |        1         |       1        | -0.0837332 |        -0.0368588 | 0.0308548   | -0.0368588 |
| support_frac   | support k/N           | 3431 |      0.820413  |    0.841574  |        1         |       1        | -0.0835984 |        -0.0367995 | 0.0311253   | -0.0393324 |
| fg_std         | fg std                | 3431 |     10.438     |   10.6867    |       10.0085    |      10.2886   | -0.0786922 |        -0.0346425 | 0.0424538   | -0.0377704 |

### Where the headroom is

```json
{
 "pq_now": 0.4105013659543128,
 "selector_oracle_on_pool": {
  "pq": 0.49181794975177634,
  "tp": 5671,
  "fp": 1333,
  "fn": 2528,
  "delta_pq": 0.08131658379746354
 },
 "add_missing_good_candidates_only": {
  "n_extra_candidates": 787,
  "pq": 0.445012991910264,
  "tp": 5671,
  "fp": 2932,
  "fn": 2528,
  "delta_pq": 0.0345116259559512
 },
 "detector_plus_selector_profitable_tiers": {
  "pq": 0.5277956549914699,
  "delta_pq": 0.11729428903715705
 }
}
```

## Reachability by group

### category

| level          |    n |   recall |   frac_reachable |
|:---------------|-----:|---------:|-----------------:|
| Left           | 2535 | 0.619724 |         0.694675 |
| Right          | 2590 | 0.662162 |         0.733977 |
| Unidentifiable | 3074 | 0.482108 |         0.653546 |

### size_quartile

| level    |    n |   recall |   frac_reachable |
|:---------|-----:|---------:|-----------------:|
| Q1_small | 2054 | 0.398734 |         0.593476 |
| Q2       | 2046 | 0.607038 |         0.718475 |
| Q3       | 2049 | 0.663738 |         0.743777 |
| Q4_large | 2050 | 0.657073 |         0.71122  |

### contrast_quartile

| level    |    n |   recall |   frac_reachable |
|:---------|-----:|---------:|-----------------:|
| Q1_faint | 2050 | 0.369756 |         0.52439  |
| Q2       | 2050 | 0.561951 |         0.692195 |
| Q3       | 2049 | 0.668619 |         0.763299 |
| Q4_dark  | 2050 | 0.725854 |         0.786829 |

### radial_zone

| level   |    n |   recall |   frac_reachable |
|:--------|-----:|---------:|-----------------:|
| 0-.35   | 1365 | 0.610256 |         0.70989  |
| .35-.7  | 4305 | 0.598839 |         0.687805 |
| .7-.9   | 2422 | 0.553675 |         0.698596 |
| .9-1    |  107 | 0.149533 |         0.457944 |

### consensus_tier

| level   |    n |   recall |   frac_reachable |
|:--------|-----:|---------:|-----------------:|
| 1/1     | 3206 | 0.549595 |         0.681535 |
| 1/2     |  518 | 0.277992 |         0.465251 |
| 1/3     |  509 | 0.172888 |         0.416503 |
| 2/2     | 1469 | 0.733152 |         0.780803 |
| 2/3     |  523 | 0.455067 |         0.632887 |
| 3/3     | 1974 | 0.739108 |         0.787741 |

## Figures

- `error_profile_fn_recall.png` recall vs decile for the 9 most discriminative features
- `error_profile_fn_crosstab.png` recall cross-tabs
- `error_profile_fn_split.png` FN anatomy (size/contrast/consensus/threshold plane)
