# Annotator decision structure (MAGFiLO Kaggle 2026 subset)

## 1. Entry-id prefix semantics
Prefix `GGBBAA`: **GG = group** (01-06 in this subset, paper has 24), **BB = batch**
(01-04), **AA = annotator slot** within group (01-03). Evidence: all
296 multi-copy files share GG+BB across copies
(100%) and copies always carry distinct
AA slots (100%).
37 prefixes -> 16 distinct
annotators (group x slot), 19 group-batch pools.
Copies per file: {1: 411, 2: 145, 3: 151}.

Entries per annotator:
person
01-02    193
01-01    175
01-03    148
04-01    102
02-01     82
05-02     74
05-03     58
03-01     56
04-02     51
05-01     48
02-03     43
03-02     43
06-03     43
06-02     23
03-03     12
04-03      3

## 2. Cross-copy instance matching (IoU>0.3, union-find, per file)
2675 clusters on 296 files;
support distribution {1: 1027, 2: 993, 3: 655};
1387 full-support, 1027 singletons.
Mean pairwise IoU among matched members: 0.573.
-> `annotator_clusters.parquet`

## 3. Agreement
- Clusters with >=2 members: 1648;
  **60.6% all-same-category**.
- Pair-level (n=2984): 66.9% same category.
- Unordered pair confusion (rows/cols = L,R,U): [[727, 270, 347], [0, 755, 372], [0, 0, 513]]
- **Cohen kappa L-vs-R pairs: 0.466**
  (paper reports 0.66 overall); all-3-category pair kappa 0.252.
- Existence vs category: skip rate (support < n_copies) by majority category:
  {'Left': 38.622386223862236, 'Right': 37.03703703703704, 'Unidentifiable': 64.78048780487805} (chi2=184.5,
  p=8.57e-41) -> Unidentifiable are skipped far more often.

## 4. Annotator fingerprints (-> `annotator_fingerprints.parquet`)
person  n_entries  mean_inst_per_entry  pct_unidentifiable  median_area  median_pts_per_karea  median_width  generosity_resid  n_multi_entries
 04-02         51             7.294118           57.526882  1010.504093             38.271951     10.623668          1.666667              9.0
 02-01         82             7.804878           26.093750   985.420803             58.701586      9.886235          0.383721             43.0
 01-01        175             7.337143           26.947040  1170.091338             57.064986     10.341014          0.366870            164.0
 01-02        193             7.186528           39.870224  1354.132095             45.159220     11.778588          0.241252            181.0
 05-03         58             7.517241           49.311927  1359.272974             41.808272     11.342057          0.190476             49.0
 03-03         12             5.416667           15.384615  1510.263327             67.748949     10.684392          0.100000              5.0
 05-02         74             7.405405           46.532847  1443.020801             31.960623     12.355339          0.018116             46.0
 04-03          3             5.666667           47.058824  1276.936176             49.777082     11.255522          0.000000              3.0
 03-01         56             9.785714           32.481752  1054.432085             47.177740     12.159567         -0.100000              5.0
 05-01         48             6.625000           51.886792  1282.145155             38.213187     12.304576         -0.211806             48.0
 02-03         43             6.023256           23.938224  1523.399743             52.160487     11.767214         -0.383721             43.0
 01-03        148             5.432432           25.124378  1383.686575             52.350740     11.515401         -0.769136            135.0
 04-01        102             7.460784           44.940867  1052.109287             48.878636     10.100719         -1.250000             12.0
 03-02         43             6.906977           44.107744  1182.333713             53.282322      9.047689               NaN              NaN
 06-02         23             4.347826           22.000000  1315.149204             50.774027     10.695181               NaN              NaN
 06-03         43             8.441860           56.198347  1271.437782             33.317742     13.758585               NaN              NaN

Most generous (count residual within shared files): 04-02
(+1.67 inst/entry), least:
04-01 (-1.25).
ANOVA on per-entry count residuals across annotators: F=6.69,
p=9.62e-11 -> annotator identity is a significant factor.

## 5. Support prediction (-> `support_prediction_dataset.parquet`)
Target: consensus (support>=2) vs singleton; 2675 clusters,
consensus rate 0.616. 5-fold GroupKFold by file:
- LogisticRegression AUC 0.728
  (2-copy 0.715, 3-copy 0.736)
- GradientBoosting AUC 0.844
  (2-copy 0.818, 3-copy 0.864)
- GB importances: {"frac_unidentifiable": 0.31713571788319717, "mean_category": 0.2057922586894639, "spine_len": 0.11828915080073704, "r_norm": 0.10610830103551083, "complexity": 0.08004212140873651, "log10_area": 0.06336983057736417, "width_proxy": 0.047952903989592316, "eccentricity": 0.03193071687693411, "abs_lat_deg": 0.025912032460510624, "n_copies_file": 0.0034669662779532673}
- LR standardized coefs: {"log10_area": 1.5931453526838426, "spine_len": -0.5734792100565187, "width_proxy": -0.5696717805169184, "frac_unidentifiable": -0.4171076545916885, "r_norm": -0.3458626698169648, "complexity": -0.25718709250660216, "n_copies_file": 0.14824755601982215, "abs_lat_deg": -0.025268425916709907, "eccentricity": 0.01986544577418013, "mean_category": 0.018047070540128926}

## 6. Hemispheric chirality preference (L/R instances, fixed geometry)
Counts: {'north': {'Left': 771, 'Right': 1713}, 'south': {'Left': 1764, 'Right': 877}}. Left fraction north 31.0%
vs south 66.8% (chi2=653.2,
p=4.51e-144).
