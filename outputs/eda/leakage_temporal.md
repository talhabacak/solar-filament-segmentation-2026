# Temporal / content leakage: train vs test, and CV folds

Generated 2026-08-19 23:40. Train n=707, test n=180. SHA-256 exact overlap = 0 (previously verified); 32x32 aHash degenerate on full-disk solar images.

## 1. Nearest-neighbour Delta-t distributions

| population | n | min h | p5 | p25 | p50 | p75 | p95 | <1h | <6h | <24h | <72h | <7d |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test->train any-site | 180 | 0.01 | 2.1 | 15.9 | 36.7 | 72.0 | 165.7 | 3 | 21 | 63 | 135 | 171 |
| test->train same-site | 180 | 21.25 | 24.0 | 88.9 | 190.3 | 462.1 | 1419.7 | 0 | 0 | 7 | 34 | 84 |
| CV val->trainfold (folds_v1, pooled) | 707 | 0.01 | 5.2 | 24.0 | 46.5 | 95.7 | 224.5 | 14 | 46 | 191 | 483 | 646 |
| temporal_holdout val->train | 142 | 239.99 | 650.5 | 4708.8 | 13041.7 | 34853.4 | 45676.4 | 0 | 0 | 0 | 0 | 0 |

(same-site: 0 test images had no same-site train image.)

### 20 closest test->train pairs (any site)

| file_name             | site   | nn_train              | nn_train_site   |       dt_h |
|:----------------------|:-------|:----------------------|:----------------|-----------:|
| 20130208063134Lh.jpeg | L      | 20130208063154Uh.jpeg | U               | 0.00555556 |
| 20150913082034Lh.jpeg | L      | 20150913082014Th.jpeg | T               | 0.00555556 |
| 20201228202510Mh.jpeg | M      | 20201228202430Ch.jpeg | C               | 0.0111111  |
| 20130609133454Bh.jpeg | B      | 20130609123334Ch.jpeg | C               | 1.02222    |
| 20200806072410Th.jpeg | T      | 20200806053830Lh.jpeg | L               | 1.76111    |
| 20130811121434Ch.jpeg | C      | 20130811140254Bh.jpeg | B               | 1.80556    |
| 20110429120534Ch.jpeg | C      | 20110429135554Bh.jpeg | B               | 1.83889    |
| 20130110085614Th.jpeg | T      | 20130110104734Ch.jpeg | C               | 1.85556    |
| 20111114063134Lh.jpeg | L      | 20111114082314Th.jpeg | T               | 1.86111    |
| 20171029171010Mh.jpeg | M      | 20171029150150Bh.jpeg | B               | 2.13889    |
| 20120218152654Bh.jpeg | B      | 20120218173614Mh.jpeg | M               | 2.15556    |
| 20160226151654Bh.jpeg | B      | 20160226173214Mh.jpeg | M               | 2.25556    |
| 20160327133734Ch.jpeg | C      | 20160327170714Mh.jpeg | M               | 3.49444    |
| 20160327133714Th.jpeg | T      | 20160327170714Mh.jpeg | M               | 3.5        |
| 20151117212434Ch.jpeg | C      | 20151117172314Mh.jpeg | M               | 4.02222    |
| 20171122022450Uh.jpeg | U      | 20171121222130Lh.jpeg | L               | 4.05556    |
| 20110603123034Ch.jpeg | C      | 20110603163414Mh.jpeg | M               | 4.06111    |
| 20171108222430Lh.jpeg | L      | 20171109023950Uh.jpeg | U               | 4.25556    |
| 20211231180850Bh.jpeg | B      | 20211231223430Lh.jpeg | L               | 4.42778    |
| 20140410195814Mh.jpeg | M      | 20140411013854Uh.jpeg | U               | 5.67778    |

## 2. CV folds vs the real train/test split

Fraction of val images with a train neighbour under 6h/24h/72h: test 11.7%/35.0%/75.0% ; CV folds_v1 6.5%/27.0%/68.3% ; temporal_holdout 0.0%/0.0%/0.0%.


## 3. Content similarity (512px, disk interior r<0.88, zero-mean/unit-std)

Near pairs (dt<48h): 171 total, 171 scored (all 22 with dt<6h kept, remainder sampled). Far baseline: 200 random pairs with dt>30d.

| set | corr | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| near | corr_direct | 0.479 | 0.481 | 0.630 | 0.770 | 0.817 |
| near | corr_aligned | 0.507 | 0.498 | 0.679 | 0.775 | 0.817 |
| far | corr_direct | 0.484 | 0.479 | 0.636 | 0.785 | 0.847 |
| far | corr_aligned | 0.490 | 0.483 | 0.652 | 0.786 | 0.847 |

### By Delta-t bucket (near pairs)

| bucket   |   n |   direct_med |   aligned_med |   aligned_max |   n_aligned_gt_0.9 |
|:---------|----:|-------------:|--------------:|--------------:|-------------------:|
| 0-1h     |   3 |     0.635388 |      0.679124 |      0.695494 |                  0 |
| 1-6h     |  19 |     0.529415 |      0.669956 |      0.78398  |                  0 |
| 6-24h    |  59 |     0.47053  |      0.518838 |      0.77101  |                  0 |
| 24-48h   |  90 |     0.469673 |      0.471267 |      0.81658  |                  0 |

Separability: far-baseline aligned-corr p99 = 0.786, max = 0.847. Near pairs above far-p99: 1/171; above far-max: 0.

Near pairs with aligned corr > 0.9 (essentially same content): 0 (0 unique test images; 1 unique test images have a partner > 0.8).

### Highest-correlation near pairs

| test_file             | train_file            |        dt_h | test_site   | train_site   |   corr_direct |   corr_aligned |
|:----------------------|:----------------------|------------:|:------------|:-------------|--------------:|---------------:|
| 20220621185332Bh.jpeg | 20220622205632Bh.jpeg | 26.05       | B           | B            |      0.81658  |       0.81658  |
| 20111114063134Lh.jpeg | 20111114082314Th.jpeg |  1.86111    | L           | T            |      0.658913 |       0.78398  |
| 20180507140210Th.jpeg | 20180506140230Ch.jpeg | 23.9944     | T           | C            |      0.77101  |       0.77101  |
| 20130213063454Uh.jpeg | 20130212153254Bh.jpeg | 15.0333     | U           | B            |      0.769812 |       0.769812 |
| 20111017113714Th.jpeg | 20111018063134Lh.jpeg | 18.9056     | T           | L            |      0.764895 |       0.764895 |
| 20110214175414Mh.jpeg | 20110215083814Th.jpeg | 14.7333     | M           | T            |      0.759142 |       0.759142 |
| 20171122022450Uh.jpeg | 20171121222130Lh.jpeg |  4.05556    | U           | L            |      0.669773 |       0.747484 |
| 20130627063134Lh.jpeg | 20130628063134Lh.jpeg | 24          | L           | L            |      0.738573 |       0.738573 |
| 20120218152654Bh.jpeg | 20120218173614Mh.jpeg |  2.15556    | B           | M            |      0.662125 |       0.716447 |
| 20130609133454Bh.jpeg | 20130609123334Ch.jpeg |  1.02222    | B           | C            |      0.574265 |       0.709623 |
| 20110214175414Mh.jpeg | 20110213093414Th.jpeg | 32.3333     | M           | T            |      0.695693 |       0.695693 |
| 20130208063134Lh.jpeg | 20130208063154Uh.jpeg |  0.00555556 | L           | U            |      0.695494 |       0.695494 |
| 20181012105930Ch.jpeg | 20181014105830Ch.jpeg | 47.9833     | C           | C            |      0.695382 |       0.695382 |
| 20140322200714Mh.jpeg | 20140323020654Uh.jpeg |  5.99444    | M           | U            |      0.546446 |       0.69033  |
| 20151117212434Ch.jpeg | 20151117172314Mh.jpeg |  4.02222    | C           | M            |      0.609911 |       0.690102 |

## 4. Cross-site near-simultaneous pairs (dt <= 30 min)

| test_file             | train_file            | test_site   | train_site   |   dt_s |       dt_h |   corr_direct |   corr_aligned |
|:----------------------|:----------------------|:------------|:-------------|-------:|-----------:|--------------:|---------------:|
| 20130208063134Lh.jpeg | 20130208063154Uh.jpeg | L           | U            |     20 | 0.00555556 |      0.695494 |       0.695494 |
| 20150913082034Lh.jpeg | 20150913082014Th.jpeg | L           | T            |     20 | 0.00555556 |      0.614564 |       0.614564 |
| 20201228202510Mh.jpeg | 20201228202430Ch.jpeg | M           | C            |     40 | 0.0111111  |      0.635388 |       0.679124 |

## 5. Verdict

### Supplementary: high-pass (filament-scale) correlation

Whole-disk Pearson at 512 is dominated by low-frequency limb-darkening/brightness structure
(far-pair baseline median ~0.48), which masks fine structure. Re-running the dt<6h pairs with a
high-pass filter (z - GaussianBlur(sigma=8), disk interior, after phase alignment) vs 40 far pairs:

- near<6h (n=22): hp_corr mean 0.382, median 0.368, p90 0.520, max 0.834
- far>30d (n=40): hp_corr mean 0.112, median 0.104, p90 0.152, max 0.314

18/22 dt<6h pairs exceed the far-pair maximum (0.314) — fine structure (filaments, plage) is
demonstrably shared for temporally adjacent pairs, but even the 20-second cross-site pair tops out
at 0.834: per-site seeing/focus/exposure differences prevent pixel-level duplication.
(Full table: outputs/eda/leakage_highpass_check.json.)

### (a) Hidden content leakage train->test beyond temporal adjacency?

**No.** Zero pairs reach aligned corr > 0.9; the near-pair aligned-corr distribution (mean 0.507)
is statistically indistinguishable from the far-pair baseline (mean 0.490) at the whole-disk level,
and only 1/171 near pairs exceeds the far p99. There are no disguised duplicates, no re-encoded
copies, no crops of the same frame. All content sharing in this dataset is exactly what the
timestamps predict: same Sun photographed again. The 3 near-simultaneous cross-site pairs
(dt = 20-40 s) are literally the same Sun, yet correlate only 0.61-0.70 raw / up to 0.83 high-pass
— GONG site-to-site imaging differences act as a natural augmentation layer.

### (b) Implications for CV design

The hidden test set is itself temporally adjacent to train: 11.7% of test images have a train frame
within 6h, 35.0% within 24h, 75.0% within 72h (any site); minimum gap is 20 seconds. Same-site
adjacency is much weaker (min 21.25h, 0% under 6h) — the organizers appear to have split so that a
site's own near-in-time frames don't cross the boundary, but cross-site simultaneity leaks through.

folds_v1 random CV is slightly HARDER than the real split in adjacency terms (6.5%/27.0%/68.3%
under 6h/24h/72h vs test's 11.7%/35.0%/75.0%) — i.e., random folds already match, and even slightly
understate, the temporal-adjacency regime of the leaderboard. Time-blocked folds are therefore NOT
required for unbiased LB estimation; they would make CV pessimistic relative to the test regime.
The temporal_holdout split (min gap 240h, 0% under 7d) is the clean generalization probe: keep both
— folds_v1 to mimic the LB, temporal_holdout to measure true out-of-time generalization. The gap
between the two scores estimates how much of LB performance rides on temporal adjacency.

### (c) Memorization risk for competitors

Low as a pixel-level strategy, real as a content prior. No test image is a pixel-near-duplicate of
a train image (0 pairs corr > 0.9; max 0.817). However, for the 3 pairs under 30 minutes and the
~21 test images with a train frame under 6h, the SAME physical filaments (rotated <=3.3 deg
longitude at 6h) appear in a train image with ground-truth masks — a competitor who matched
timestamps could warp train GT onto those test frames. That covers at most ~12% of the test set,
and per-annotator mask variability plus site-to-site rendering differences cap the payoff; it is an
edge, not a winning strategy. For modeling, the practical consequence is inflated confidence on
adjacent frames — nothing about the data enables lookup-table leakage.