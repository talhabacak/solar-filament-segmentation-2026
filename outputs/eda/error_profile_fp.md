# Error profile at the current operating point

Operating point: **conf >= 0.3 AND p_sel >= 0.35** on the 10411 pooled 5-fold OOF de-overlapped candidates of E013@1792.

Exact official scorer (verified against `fast_pq`): **PQ 0.4105**, SQ 0.6641, RQ 0.6181, TP 4768, FP 2461, FN 3431, matched-IoU sum 3166.61, denominator 7714.0, over 8199 GT rows / 1154 annotator entries / 707 images. Kept candidates: 4441.

Every candidate is scored once per annotator copy of its image, so one kept candidate on a 3-annotator image occupies 3 GT-copy slots and can contribute up to 3 FP.


## 1. Anatomy of the 2461 false positives

| fp_class | n_kept | share_of_kept | fp_copies | share_of_FP | tp_copies | gt_slots | precision_copies |
|---|---|---|---|---|---|---|---|
| pure_hallucination | 1141 | 0.2569 | 1599 | 0.6497 | 0 | 1599 | 0.0 |
| partial_consensus | 690 | 0.1554 | 862 | 0.3503 | 935 | 1797 | 0.5203 |
| clean_tp | 2610 | 0.5877 | 0 | 0.0 | 3833 | 3832 | 1.0003 |

`fp_copies = n_copies - n_entries_hit` summed per class; the three classes partition the 4441 kept candidates and their FP sum is exactly 2461.


How close the FP predictions came to any GT in the same entry (best IoU with any GT of that annotator):

| lo | hi | n_fp | n_pure |
|---|---|---|---|
| 0.0 | 0.05 | 999 | 691 |
| 0.05 | 0.25 | 310 | 227 |
| 0.25 | 0.45 | 796 | 509 |
| 0.45 | 0.5 | 351 | 172 |

## 2. Feature separation (medians per class, Cohen's d ranked)

| feature | median_pure_hallucination | median_partial_consensus | median_clean_tp | d_pure_vs_cleanTP | d_partial_vs_cleanTP |
|---|---|---|---|---|---|
| conf | 0.5247 | 0.5729 | 0.6848 | -0.8642 | -0.5666 |
| p_sel | 0.6798 | 0.7381 | 0.8363 | -0.8206 | -0.5467 |
| contrast | 0.1012 | 0.1062 | 0.1208 | -0.5968 | -0.4444 |
| rank_conf | 5.0 | 5.0 | 3.0 | 0.579 | 0.4075 |
| boundary_complexity | 3.6829 | 3.6906 | 4.1527 | -0.2298 | -0.2267 |
| edge_grad | 39.4284 | 40.7324 | 42.4735 | -0.2242 | -0.1308 |
| area | 1353.0 | 1259.5 | 1627.0 | -0.2057 | -0.2479 |
| r_norm | 0.612 | 0.5881 | 0.5698 | 0.1505 | 0.0535 |
| width_proxy | 13.1505 | 12.9214 | 13.697 | -0.1465 | -0.1789 |
| depth | 0.0815 | 0.0823 | 0.0961 | -0.1127 | -0.158 |
| eccentricity | 0.9746 | 0.9751 | 0.9777 | -0.1029 | -0.0908 |
| fg_min | 91.0 | 92.0 | 89.0 | 0.1005 | 0.1976 |
| abs_lat | 20.9145 | 20.0123 | 19.5237 | 0.0959 | 0.0422 |
| n_cands_img | 19.0 | 18.0 | 18.0 | 0.072 | 0.0854 |

Positive d = larger in pure hallucinations than in clean TPs.


## 3. Spatial / context

| dimension | level | n_kept | n_images | n_pure | pure_share_of_kept | fp_copies | fp_per_image | precision_copies | share_of_all_FP |
|---|---|---|---|---|---|---|---|---|---|
| radial_zone | r<0.30 | 528 | 348 | 122 | 0.2311 | 289 | 0.8305 | 0.6727 | 0.1174 |
| radial_zone | r0.30-0.60 | 1799 | 593 | 427 | 0.2374 | 931 | 1.57 | 0.6794 | 0.3783 |
| radial_zone | r0.60-0.80 | 1565 | 595 | 395 | 0.2524 | 853 | 1.4336 | 0.6661 | 0.3466 |
| radial_zone | r0.80-0.90 | 519 | 362 | 179 | 0.3449 | 349 | 0.9641 | 0.5825 | 0.1418 |
| radial_zone | r>=0.90 | 30 | 30 | 18 | 0.6 | 39 | 1.3 | 0.2642 | 0.0158 |
| lat_band | |lat|<10 | 930 | 463 | 230 | 0.2473 | 517 | 1.1166 | 0.6649 | 0.2101 |
| lat_band | |lat|10-20 | 1289 | 549 | 302 | 0.2343 | 663 | 1.2077 | 0.6834 | 0.2694 |
| lat_band | |lat|20-30 | 1284 | 565 | 349 | 0.2718 | 720 | 1.2743 | 0.6538 | 0.2926 |
| lat_band | |lat|30-45 | 826 | 468 | 219 | 0.2651 | 476 | 1.0171 | 0.6421 | 0.1934 |
| lat_band | |lat|>=45 | 112 | 103 | 41 | 0.3661 | 85 | 0.8252 | 0.538 | 0.0345 |
| site | B | 792 | 115 | 204 | 0.2576 | 448 | 3.8957 | 0.6664 | 0.182 |
| site | C | 814 | 124 | 206 | 0.2531 | 429 | 3.4597 | 0.67 | 0.1743 |
| site | L | 787 | 122 | 232 | 0.2948 | 484 | 3.9672 | 0.6244 | 0.1967 |
| site | M | 729 | 121 | 194 | 0.2661 | 389 | 3.2149 | 0.6576 | 0.1581 |
| site | T | 723 | 109 | 176 | 0.2434 | 393 | 3.6055 | 0.6703 | 0.1597 |
| site | U | 596 | 95 | 129 | 0.2164 | 318 | 3.3474 | 0.6725 | 0.1292 |
| year | 2011 | 530 | 85 | 121 | 0.2283 | 307 | 3.6118 | 0.6659 | 0.1247 |
| year | 2012 | 691 | 92 | 176 | 0.2547 | 409 | 4.4457 | 0.6471 | 0.1662 |
| year | 2013 | 811 | 101 | 221 | 0.2725 | 420 | 4.1584 | 0.6577 | 0.1707 |
| year | 2014 | 902 | 102 | 238 | 0.2639 | 476 | 4.6667 | 0.6669 | 0.1934 |
| year | 2015 | 663 | 86 | 167 | 0.2519 | 370 | 4.3023 | 0.6702 | 0.1503 |
| year | 2016 | 438 | 79 | 101 | 0.2306 | 243 | 3.0759 | 0.663 | 0.0987 |
| year | 2017 | 221 | 69 | 67 | 0.3032 | 140 | 2.029 | 0.6206 | 0.0569 |
| year | 2018 | 37 | 21 | 11 | 0.2973 | 26 | 1.2381 | 0.6 | 0.0106 |
| year | 2019 | 14 | 10 | 5 | 0.3571 | 11 | 1.1 | 0.5926 | 0.0045 |
| year | 2020 | 10 | 6 | 1 | 0.1 | 5 | 0.8333 | 0.7059 | 0.002 |
| year | 2021 | 72 | 25 | 23 | 0.3194 | 35 | 1.4 | 0.6354 | 0.0142 |
| year | 2022 | 52 | 10 | 10 | 0.1923 | 19 | 1.9 | 0.7625 | 0.0077 |
| gt_quintile | Q1 | 295 | 133 | 105 | 0.3559 | 210 | 1.5789 | 0.5732 | 0.0853 |
| gt_quintile | Q2 | 730 | 152 | 215 | 0.2945 | 482 | 3.1711 | 0.6184 | 0.1959 |
| gt_quintile | Q3 | 873 | 124 | 223 | 0.2554 | 518 | 4.1774 | 0.6521 | 0.2105 |
| gt_quintile | Q4 | 1280 | 156 | 319 | 0.2492 | 681 | 4.3654 | 0.6694 | 0.2767 |
| gt_quintile | Q5 | 1263 | 121 | 279 | 0.2209 | 570 | 4.7107 | 0.7042 | 0.2316 |
| dens_quintile | D1 | 294 | 130 | 69 | 0.2347 | 154 | 1.1846 | 0.6876 | 0.0626 |
| dens_quintile | D2 | 658 | 141 | 155 | 0.2356 | 330 | 2.3404 | 0.6742 | 0.1341 |
| dens_quintile | D3 | 1076 | 162 | 266 | 0.2472 | 612 | 3.7778 | 0.6587 | 0.2487 |
| dens_quintile | D4 | 1092 | 129 | 288 | 0.2637 | 553 | 4.2868 | 0.6739 | 0.2247 |
| dens_quintile | D5 | 1321 | 124 | 363 | 0.2748 | 812 | 6.5484 | 0.6368 | 0.3299 |
| n_copies | 1 | 2582 | 395 | 820 | 0.3176 | 820 | 2.0759 | 0.6824 | 0.3332 |
| n_copies | 2 | 931 | 143 | 184 | 0.1976 | 641 | 4.4825 | 0.6557 | 0.2605 |
| n_copies | 3 | 928 | 148 | 137 | 0.1476 | 1000 | 6.7568 | 0.6412 | 0.4063 |

**Interpretation warning on `pure_share_of_kept` vs n_copies.** "Pure hallucination" is defined against the *union* of the annotators on that image, so it is mechanically easier to be a pure hallucination on a 1-annotator image (one chance to be confirmed) than on a 3-annotator image (three chances). The annotator-count-neutral metric is `precision_copies` = matched copies / GT-copy slots, which is essentially flat and slightly *worse* on more-annotated images: 0.6824 (1 copy) / 0.6557 (2) / 0.6412 (3). Use `precision_copies` for cross-n_copies comparisons and `pure_share_of_kept` only within a fixed n_copies (see the cross-tab below, where the near-limb effect survives the control).


`n_images` is the number of images that own at least one kept candidate in that level, so `fp_per_image` is FP-copies per *contributing* image (denominators differ across levels of the same dimension for candidate-level splits such as radial_zone and lat_band).


Regime cross-tab -- pure-hallucination share by n_copies x radial zone (checks whether the near-limb effect is just the 1-annotator effect):

| n_copies | radial_zone | n_kept | n_pure | pure_share |
|---|---|---|---|---|
| 1 | r<0.30 | 292 | 85 | 0.2911 |
| 1 | r0.30-0.60 | 1066 | 309 | 0.2899 |
| 1 | r0.60-0.80 | 911 | 283 | 0.3106 |
| 1 | r0.80-0.90 | 298 | 132 | 0.443 |
| 1 | r>=0.90 | 15 | 11 | 0.7333 |
| 2 | r<0.30 | 117 | 19 | 0.1624 |
| 2 | r0.30-0.60 | 364 | 67 | 0.1841 |
| 2 | r0.60-0.80 | 318 | 63 | 0.1981 |
| 2 | r0.80-0.90 | 125 | 32 | 0.256 |
| 2 | r>=0.90 | 7 | 3 | 0.4286 |
| 3 | r<0.30 | 119 | 18 | 0.1513 |
| 3 | r0.30-0.60 | 369 | 51 | 0.1382 |
| 3 | r0.60-0.80 | 336 | 49 | 0.1458 |
| 3 | r0.80-0.90 | 96 | 15 | 0.1562 |
| 3 | r>=0.90 | 8 | 4 | 0.5 |

...and by n_copies x GT-activity quintile:

| n_copies | gt_quintile | n_kept | n_pure | pure_share |
|---|---|---|---|---|
| 1 | Q1 | 163 | 70 | 0.4294 |
| 1 | Q2 | 382 | 153 | 0.4005 |
| 1 | Q3 | 466 | 146 | 0.3133 |
| 1 | Q4 | 763 | 229 | 0.3001 |
| 1 | Q5 | 808 | 222 | 0.2748 |
| 2 | Q1 | 67 | 20 | 0.2985 |
| 2 | Q2 | 163 | 35 | 0.2147 |
| 2 | Q3 | 201 | 43 | 0.2139 |
| 2 | Q4 | 254 | 53 | 0.2087 |
| 2 | Q5 | 246 | 33 | 0.1341 |
| 3 | Q1 | 65 | 15 | 0.2308 |
| 3 | Q2 | 185 | 27 | 0.1459 |
| 3 | Q3 | 206 | 34 | 0.165 |
| 3 | Q4 | 263 | 37 | 0.1407 |
| 3 | Q5 | 209 | 24 | 0.1148 |

Per-image correlations (686 images with >=1 kept candidate):

| pair | pearson | spearman | n |
|---|---|---|---|
| n_cands~fp | 0.5115 | 0.5379 | 686 |
| n_cands~n_pure | 0.4927 | 0.5011 | 686 |
| gt_per_entry~fp | 0.2774 | 0.3196 | 686 |
| gt_per_entry~n_pure | 0.3045 | 0.3552 | 686 |
| n_kept~fp | 0.5386 | 0.5776 | 686 |

## 4. Is it darkness? Photometric separability


**pure_vs_cleanTP** (n=3751, positives=1141):

| score | auc | auc_oriented |
|---|---|---|
| contrast | 0.3304 | 0.6696 |
| depth | 0.4552 | 0.5448 |
| fg_min | 0.5399 | 0.5399 |
| edge_grad | 0.432 | 0.568 |
| fg_mean | 0.5435 | 0.5435 |
| conf | 0.2684 | 0.7316 |
| p_sel | 0.2768 | 0.7232 |
| logit(contrast,depth,fg_min) OOF | 0.6837 | 0.6837 |
| logit(all 6 photometric) OOF | 0.6944 | 0.6944 |
| logit(conf,p_sel) OOF | 0.737 | 0.737 |
| logit(conf,p_sel + 6 photometric) OOF | 0.7615 | 0.7615 |

Incremental AUC from photometry on top of (conf, p_sel): **+0.0245** (0.7370 -> 0.7615, 5-fold GroupKFold by image).


**pure_vs_anyTP** (n=4441, positives=1141):

| score | auc | auc_oriented |
|---|---|---|
| contrast | 0.3546 | 0.6454 |
| depth | 0.466 | 0.534 |
| fg_min | 0.5271 | 0.5271 |
| edge_grad | 0.4403 | 0.5597 |
| fg_mean | 0.5324 | 0.5324 |
| conf | 0.2995 | 0.7005 |
| p_sel | 0.3075 | 0.6925 |
| logit(contrast,depth,fg_min) OOF | 0.6611 | 0.6611 |
| logit(all 6 photometric) OOF | 0.6708 | 0.6708 |
| logit(conf,p_sel) OOF | 0.7051 | 0.7051 |
| logit(conf,p_sel + 6 photometric) OOF | 0.7302 | 0.7302 |

Incremental AUC from photometry on top of (conf, p_sel): **+0.0252** (0.7051 -> 0.7302, 5-fold GroupKFold by image).


Pure-hallucination rate by contrast decile (kept set):

| dec | n | pure_rate | contrast_lo | contrast_hi |
|---|---|---|---|---|
| 0 | 445 | 0.4831 | 0.0102 | 0.076 |
| 1 | 444 | 0.3694 | 0.076 | 0.0879 |
| 2 | 444 | 0.3063 | 0.0879 | 0.0968 |
| 3 | 444 | 0.2635 | 0.0968 | 0.1051 |
| 4 | 444 | 0.2387 | 0.1051 | 0.1132 |
| 5 | 444 | 0.241 | 0.1133 | 0.1219 |
| 6 | 444 | 0.2005 | 0.122 | 0.1317 |
| 7 | 444 | 0.1734 | 0.1317 | 0.1443 |
| 8 | 444 | 0.1712 | 0.1444 | 0.1621 |
| 9 | 444 | 0.1216 | 0.1621 | 0.2706 |

## 5. Scorecard -- absolute PQ loss by stratum

Bin edges come from the GT distribution and are applied to GT rows (TP/FN) and to kept candidates (FP) alike; `category` is a GT-only property so all FP land in `unassigned_pred`. Every dimension reconstructs TP=4768/FP=2461/FN=3431.

| dimension | level | n_gt | tp | fp | fn | sq | rq_stratum | pq_stratum | recall_copies | lost_mass | lost_mass_share | delta_pq_if_perfect |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| category | unassigned_pred | 0 | 0 | 2461 | 0 |  | 0.0 | 0.0 |  | 1230.5 | 0.4177 | 0.0779 |
| radial_zone | r0.30-0.60 | 3308 | 1967 | 931 | 1341 | 0.6618 | 0.6339 | 0.4195 | 0.5946 | 1136.0 | 0.3856 | 0.2427 |
| consensus_tier | T1_single_annotator | 3206 | 1762 | 820 | 1444 | 0.6673 | 0.6088 | 0.4063 | 0.5496 | 1132.0 | 0.3842 | 0.237 |
| consensus_tier | T3_three_annotators | 3006 | 1785 | 1000 | 1221 | 0.6658 | 0.6165 | 0.4105 | 0.5938 | 1110.5 | 0.377 | 0.2265 |
| radial_zone | r0.60-0.80 | 2855 | 1694 | 853 | 1161 | 0.6657 | 0.6272 | 0.4175 | 0.5933 | 1007.0 | 0.3418 | 0.2115 |
| width_quartile | W3 | 2050 | 1275 | 1095 | 775 | 0.6751 | 0.5769 | 0.3895 | 0.622 | 935.0 | 0.3174 | 0.1661 |
| gt_quintile | Q5 | 2503 | 1357 | 570 | 1146 | 0.6633 | 0.6126 | 0.4064 | 0.5421 | 858.0 | 0.2912 | 0.1855 |
| gt_quintile | Q4 | 2386 | 1379 | 681 | 1007 | 0.6653 | 0.6203 | 0.4127 | 0.578 | 844.0 | 0.2865 | 0.1779 |
| size_quartile | A1_smallest | 2049 | 817 | 451 | 1232 | 0.6394 | 0.4926 | 0.315 | 0.3987 | 841.5 | 0.2856 | 0.1686 |
| category | cat3_Unidentifiable | 3074 | 1482 | 0 | 1592 | 0.677 | 0.6506 | 0.4404 | 0.4821 | 796.0 | 0.2702 | 0.2049 |
| size_quartile | A2 | 2050 | 1243 | 711 | 807 | 0.6581 | 0.6209 | 0.4086 | 0.6063 | 759.0 | 0.2576 | 0.1562 |
| width_quartile | W2 | 2049 | 1222 | 624 | 827 | 0.654 | 0.6275 | 0.4104 | 0.5964 | 725.5 | 0.2463 | 0.1546 |
| size_quartile | A3 | 2050 | 1361 | 730 | 689 | 0.6669 | 0.6573 | 0.4383 | 0.6639 | 709.5 | 0.2408 | 0.1496 |
| consensus_tier | T2_two_annotators | 1987 | 1221 | 641 | 766 | 0.6572 | 0.6345 | 0.417 | 0.6145 | 703.5 | 0.2388 | 0.149 |
| width_quartile | W4_thickest | 2050 | 1346 | 696 | 704 | 0.6977 | 0.6579 | 0.459 | 0.6566 | 700.0 | 0.2376 | 0.1437 |
| size_quartile | A4_largest | 2050 | 1347 | 569 | 703 | 0.6819 | 0.6793 | 0.4632 | 0.6571 | 636.0 | 0.2159 | 0.1419 |
| width_quartile | W1_thinnest | 2050 | 925 | 46 | 1125 | 0.6136 | 0.6124 | 0.3758 | 0.4512 | 585.5 | 0.1987 | 0.1528 |
| site | L | 1438 | 803 | 484 | 635 | 0.6726 | 0.5894 | 0.3964 | 0.5584 | 559.5 | 0.1899 | 0.1113 |
| gt_quintile | Q3 | 1539 | 969 | 518 | 570 | 0.6624 | 0.6404 | 0.4242 | 0.6296 | 544.0 | 0.1847 | 0.1145 |
| site | C | 1522 | 871 | 429 | 651 | 0.6589 | 0.6173 | 0.4067 | 0.5723 | 540.0 | 0.1833 | 0.1153 |
| site | B | 1502 | 895 | 448 | 607 | 0.667 | 0.6292 | 0.4196 | 0.5959 | 527.5 | 0.1791 | 0.1119 |
| gt_quintile | Q2 | 1265 | 781 | 482 | 484 | 0.6665 | 0.6179 | 0.4118 | 0.6174 | 483.0 | 0.164 | 0.0964 |
| category | cat1_Left | 2535 | 1571 | 0 | 964 | 0.6555 | 0.7652 | 0.5016 | 0.6197 | 482.0 | 0.1636 | 0.1595 |
| site | T | 1369 | 799 | 393 | 570 | 0.6605 | 0.624 | 0.4121 | 0.5836 | 481.5 | 0.1634 | 0.1032 |
| site | M | 1298 | 747 | 389 | 551 | 0.6605 | 0.6138 | 0.4054 | 0.5755 | 470.0 | 0.1595 | 0.099 |
| category | cat2_Right | 2590 | 1715 | 0 | 875 | 0.6609 | 0.7967 | 0.5266 | 0.6622 | 437.5 | 0.1485 | 0.1567 |
| radial_zone | r0.80-0.90 | 969 | 494 | 349 | 475 | 0.6734 | 0.5453 | 0.3672 | 0.5098 | 412.0 | 0.1399 | 0.0785 |
| site | U | 1070 | 653 | 318 | 417 | 0.6654 | 0.6399 | 0.4258 | 0.6103 | 367.5 | 0.1247 | 0.0792 |
| radial_zone | r<0.30 | 957 | 596 | 289 | 361 | 0.6596 | 0.6471 | 0.4269 | 0.6228 | 325.0 | 0.1103 | 0.0708 |
| gt_quintile | Q1 | 506 | 282 | 210 | 224 | 0.6617 | 0.5651 | 0.374 | 0.5573 | 217.0 | 0.0737 | 0.041 |
| radial_zone | r>=0.90 | 110 | 17 | 39 | 93 | 0.6685 | 0.2048 | 0.1369 | 0.1545 | 66.0 | 0.0224 | 0.0113 |

`lost_mass = 0.5*FP + 0.5*FN` (total 2946.0). `delta_pq_if_perfect` = global PQ change if that stratum were solved perfectly (all its FN recovered at IoU 1, all its FP removed).


Supplementary -- the same 2461 FP attributed to the category of the GT instance they overlap most inside their own entry (`no_gt_overlap` = best IoU < 0.10):

| bucket | fp_copies | share_of_FP |
|---|---|---|
| cat1_Left | 518 | 0.2105 |
| cat2_Right | 506 | 0.2056 |
| cat3_Unidentifiable | 394 | 0.1601 |
| no_gt_overlap_iou<0.10 | 1043 | 0.4238 |

GT vs kept-candidate marginals (quantiles 0/10/25/50/75/90/100):

| quantity | gt | kept_cand |
|---|---|---|
| area | [9.0, 409.8, 670.0, 1228.0, 2438.0, 4684.6, 37739.0] | [283.0, 613.0, 891.0, 1486.0, 2676.0, 4728.0, 21888.0] |
| width_proxy | [0.053, 6.47, 8.274, 11.074, 15.167, 21.038, 70.332] | [7.012, 9.847, 11.314, 13.468, 16.129, 19.564, 50.763] |
| r_norm | [0.012, 0.283, 0.422, 0.586, 0.731, 0.821, 0.966] | [0.014, 0.277, 0.424, 0.583, 0.725, 0.816, 0.941] |

Caveat: GT width_proxy = polygon_area / polygon_spine_length (morphology_instances); candidate width_proxy = deoverlapped_mask_area / skeletonize(mask).sum(). The two estimators are not identical, so the W1_thinnest FP count is a lower bound; the GT-side TP/FN split in that quartile is unaffected.


GT-side cross-annotator support (multi-copy files only, TP/FN only):

| support_tier | tp | iou_sum | fn | n_gt | sq | recall |
|---|---|---|---|---|---|---|
| full_support | 2536 | 1686.829 | 907 | 3443 | 0.6652 | 0.7366 |
| partial_support | 470 | 304.0727 | 1080 | 1550 | 0.647 | 0.3032 |

## 6. SQ losses

| dimension | level | n | p10 | p50 | p90 | mean |
|---|---|---|---|---|---|---|
| size_quartile | A1_smallest | 817 | 0.5299 | 0.6331 | 0.7556 | 0.6394 |
| size_quartile | A2 | 1243 | 0.5479 | 0.6565 | 0.7637 | 0.6581 |
| size_quartile | A3 | 1361 | 0.5546 | 0.6704 | 0.7691 | 0.6669 |
| size_quartile | A4_largest | 1347 | 0.5571 | 0.6882 | 0.7957 | 0.6819 |
| width_quartile | W1_thinnest | 925 | 0.5262 | 0.6088 | 0.7111 | 0.6136 |
| width_quartile | W2 | 1222 | 0.5483 | 0.6531 | 0.7551 | 0.654 |
| width_quartile | W3 | 1275 | 0.5612 | 0.6833 | 0.7748 | 0.6751 |
| width_quartile | W4_thickest | 1346 | 0.5731 | 0.7096 | 0.8041 | 0.6977 |
| category | cat1_Left | 1571 | 0.5428 | 0.652 | 0.7648 | 0.6555 |
| category | cat2_Right | 1715 | 0.5484 | 0.6644 | 0.765 | 0.6609 |
| category | cat3_Unidentifiable | 1482 | 0.5531 | 0.68 | 0.7944 | 0.677 |
| radial_zone | r<0.30 | 596 | 0.5437 | 0.6586 | 0.7645 | 0.6596 |
| radial_zone | r0.30-0.60 | 1967 | 0.5479 | 0.6625 | 0.7727 | 0.6618 |
| radial_zone | r0.60-0.80 | 1694 | 0.5441 | 0.6672 | 0.7772 | 0.6657 |
| radial_zone | r0.80-0.90 | 494 | 0.5564 | 0.678 | 0.7838 | 0.6734 |
| radial_zone | r>=0.90 | 17 | 0.5527 | 0.6706 | 0.7908 | 0.6685 |
| consensus_tier | T1_single_annotator | 1762 | 0.5499 | 0.6691 | 0.778 | 0.6673 |
| consensus_tier | T2_two_annotators | 1221 | 0.5425 | 0.6561 | 0.7668 | 0.6572 |
| consensus_tier | T3_three_annotators | 1785 | 0.547 | 0.6677 | 0.7764 | 0.6658 |

Counterfactuals:

| scenario | pq | delta_pq | extra |
|---|---|---|---|
| baseline | 0.4105 |  | {"iou_sum": 3166.607536971569, "tp": 4768, "fp": 2461, "fn": 3431} |
| all_FP_removed | 0.4884 | 0.0779 | {} |
| only_pure_halluc_FP_removed | 0.458 | 0.0475 | {"n_fp_removed": 1599} |
| all_FN_recovered_at_iou1 | 0.6997 | 0.2892 | {} |
| all_FN_recovered_at_current_SQ | 0.5775 | 0.167 | {} |
| iou_plus_0.02 | 0.4229 | 0.0124 | {"iou_sum": 3261.9675369715687, "n_capped": 0} |
| iou_plus_0.05 | 0.4414 | 0.0309 | {"iou_sum": 3405.0075369715687, "n_capped": 0} |
| iou_plus_0.1 | 0.4723 | 0.0618 | {"iou_sum": 3643.4075369715683, "n_capped": 0} |
| nearmiss_flip_bestiou_gt_0.4 | 0.4509 | 0.0404 | {"n_near_miss_fn": 617, "n_fp_removed": 617} |
| nearmiss_flip_bestiou_gt_0.45 | 0.4348 | 0.0243 | {"n_near_miss_fn": 356, "n_fp_removed": 356} |
| nearmiss_flip_bestiou_gt_0.48 | 0.4212 | 0.0107 | {"n_near_miss_fn": 152, "n_fp_removed": 152} |

Unmatched GT (FN) best-IoU with any kept prediction: {"count": 3431.0, "mean": 0.1479, "std": 0.1888, "min": 0.0, "50%": 0.0, "75%": 0.342, "90%": 0.4526, "95%": 0.4765, "max": 0.5}; share of FN with zero overlap = 0.5695.


## Figures

- `error_profile_fp_anatomy.png` -- FP decomposition + Cohen's d ranking
- `error_profile_fp_context.png` -- hallucination rate by radius/latitude/density/site
- `error_profile_fp_scorecard.png` -- lost denominator mass + matched-IoU boxes