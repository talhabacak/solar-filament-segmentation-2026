# MAGFiLO Kaggle 2026 - Data Integrity & Leakage Audit

Generated: 2026-08-19 19:35 | train imgs: 707 | test imgs: 180 | image entries: 1154 | annotations: 8199

## 1. Hash / leakage check

- Exact-duplicate (sha256) groups within TRAIN: 0
- Exact-duplicate (sha256) groups within TEST: 0
- **TRAIN/TEST sha256 overlap: 0 -> NO OVERLAP - no direct leakage**
- aHash(32x32) collisions ACROSS train/test with different sha256: 158
  - train 20110205082634Lh.jpeg ~ test 20151207173614Mh.jpeg (ahash 0000000000000000...)
  - train 20110205082634Lh.jpeg ~ test 20200310113230Ch.jpeg (ahash 0000000000000000...)
  - train 20110221173814Mh.jpeg ~ test 20151207173614Mh.jpeg (ahash 0000000000000000...)
  - train 20110221173814Mh.jpeg ~ test 20200310113230Ch.jpeg (ahash 0000000000000000...)
  - train 20110301082654Uh.jpeg ~ test 20110329082654Uh.jpeg (ahash 0000000000000000...)
  - train 20110824141054Bh.jpeg ~ test 20110120105534Ch.jpeg (ahash 0000000000000000...)
  - train 20110824141054Bh.jpeg ~ test 20140410195814Mh.jpeg (ahash 0000000000000000...)
  - train 20110824141054Bh.jpeg ~ test 20141009195814Mh.jpeg (ahash 0000000000000000...)
  - train 20110824141054Bh.jpeg ~ test 20210627163630Mh.jpeg (ahash 0000000000000000...)
  - train 20110824141054Bh.jpeg ~ test 20210701163730Mh.jpeg (ahash 0000000000000000...)
  - train 20110918112834Ch.jpeg ~ test 20151026171014Mh.jpeg (ahash 0000000000000000...)
  - train 20110923165814Mh.jpeg ~ test 20110120105534Ch.jpeg (ahash 0000000000000000...)
  - train 20110923165814Mh.jpeg ~ test 20140410195814Mh.jpeg (ahash 0000000000000000...)
  - train 20110923165814Mh.jpeg ~ test 20141009195814Mh.jpeg (ahash 0000000000000000...)
  - train 20110923165814Mh.jpeg ~ test 20210627163630Mh.jpeg (ahash 0000000000000000...)
  - train 20110923165814Mh.jpeg ~ test 20210701163730Mh.jpeg (ahash 0000000000000000...)
  - train 20111214132434Ch.jpeg ~ test 20130206063154Uh.jpeg (ahash 0000000000000000...)
  - train 20120114063154Uh.jpeg ~ test 20130206063154Uh.jpeg (ahash 0000000000000000...)
  - train 20120424120134Ch.jpeg ~ test 20151207173614Mh.jpeg (ahash 0000000000000000...)
  - train 20120424120134Ch.jpeg ~ test 20200310113230Ch.jpeg (ahash 0000000000000000...)
- aHash caveat: only 764 unique aHash values over 887 images; largest cluster has 23 members. The binarized 32x32 hash collapses to the centered solar-disk silhouette for full-disk H-alpha images, so identical aHash does NOT imply near-duplicate content here - colliding pairs come from different dates/sites and differ at pixel level. No true near-duplicate leakage evidence.
- aHash collision groups within train: 40, within test: 5
  - train group: ['20110824141054Bh.jpeg', '20110923165814Mh.jpeg', '20141105195814Mh.jpeg', '20150527161214Mh.jpeg', '20150716162114Mh.jpeg', '20150919165814Mh.jpeg', '20160223173514Mh.jpeg', '20170117175014Mh.jpeg', '20170827165310Mh.jpeg', '20171011170310Mh.jpeg', '20171011224130Lh.jpeg', '20171022170710Mh.jpeg', '20171106171510Mh.jpeg', '20181101230010Mh.jpeg', '20190925165810Mh.jpeg', '20190926165810Mh.jpeg', '20190927165910Mh.jpeg', '20210620163530Mh.jpeg']
  - train group: ['20120627162114Mh.jpeg', '20130705161714Mh.jpeg', '20140213024554Uh.jpeg', '20150719182314Mh.jpeg', '20170502233130Lh.jpeg', '20190629163710Mh.jpeg']
  - train group: ['20140615195814Mh.jpeg', '20170722235230Lh.jpeg', '20180109223930Lh.jpeg']
  - train group: ['20110301082654Uh.jpeg', '20140605195854Bh.jpeg']
  - train group: ['20110918112834Ch.jpeg', '20130609173414Mh.jpeg', '20160225173314Mh.jpeg', '20160226173214Mh.jpeg', '20160327170714Mh.jpeg']
  - train group: ['20120427162614Mh.jpeg', '20180924112030Ch.jpeg']
  - train group: ['20120116063254Uh.jpeg', '20151224133734Ch.jpeg', '20151226133734Ch.jpeg', '20160714235434Lh.jpeg']
  - train group: ['20170701235630Lh.jpeg', '20190412025430Lh.jpeg']
  - train group: ['20121207103034Ch.jpeg', '20140414195834Ch.jpeg', '20181102104630Ch.jpeg']
  - train group: ['20151105133734Ch.jpeg', '20161229133734Ch.jpeg', '20161231133734Ch.jpeg', '20171025133730Ch.jpeg']
  - test group: ['20110120105534Ch.jpeg', '20140410195814Mh.jpeg', '20141009195814Mh.jpeg', '20210627163630Mh.jpeg', '20210701163730Mh.jpeg']
  - test group: ['20171029171010Mh.jpeg', '20201228202510Mh.jpeg']
  - test group: ['20151207173614Mh.jpeg', '20200310113230Ch.jpeg']
  - test group: ['20130526063134Lh.jpeg', '20181030230030Lh.jpeg']
  - test group: ['20131027185354Bh.jpeg', '20160312020554Uh.jpeg']

## 2. Annotation geometry

- shapely-invalid polygons: 0 (0.00%)
- self-intersections: 0 (0.00%)
- coords outside [0,2048]: 0 (0.00%) (max excursion 0.0px)
- degenerate (<3 pts): 0 (0.00%); ~zero area: 0 (0.00%)
- gross COCO-area mismatch (ratio outside [0.5,2.0]): 0 (0.00%)
- area ratio (coco/computed) median 1.000, p1 0.982, p99 1.017, within 10% of 1.0: 100.0%
- bbox vs polygon-bounds deviation > 2.0px: 55 (0.67%) (max dev 63.5px, median 0.00px)
- spine points outside [0,2048]: 0 (0.00%)
- spine coverage (frac of points within 5px of polygon): median 1.000, mean 0.999; annotations with coverage < 1.0: 37; < 0.5: 2; == 0: 1
- spine endpoints NOT both within 5px: 37
- duplicate annotation ids: 0 {}
- iscrowd values: {0: 8199}
- multi-part segmentations: 0 (parts hist: {1: 8199})

Worst spine-vs-polygon offenders:

| ann_id                               | image_id                |   spine_frac_within5px |   spine_frac_inside |   n_spine_pts |   valid_area |
|:-------------------------------------|:------------------------|-----------------------:|--------------------:|--------------:|-------------:|
| f1167b5f-4c20-4a10-a64f-58b3c25eeebe | 050202-20120523163014Mh |              0         |           0         |             3 |     58.2966  |
| bfed9608-8ed0-4bdb-850b-fce9f8ae5a84 | 010403-20150228082414Th |              0.0526316 |           0.0526316 |            19 |      9.12579 |
| ffb525e8-2f31-4df1-8ba7-0375180af785 | 010401-20140216230134Lh |              0.6       |           0.6       |             5 |   1437.81    |
| 066bb9c0-e3ee-4f49-b6b1-f5d511484fc1 | 010401-20140216230134Lh |              0.666667  |           0.666667  |             6 |    460.335   |
| f8ac2898-3436-4605-92df-2a1bc6d2a23f | 010401-20141106195854Bh |              0.75      |           0.714286  |            28 |   1297.44    |
| 760e5872-4712-47fa-bb24-66563bca2897 | 010401-20160403170114Mh |              0.8       |           0.8       |            10 |   2770.08    |
| a9629ec4-3131-42b7-8d6f-3652e8540c20 | 010401-20150423120034Ch |              0.8       |           0.8       |            10 |   2019.59    |
| 293c743c-08ca-4521-a432-db50d101bd04 | 010401-20150325015554Uh |              0.818182  |           0.818182  |            11 |   2800.53    |
| 7506b166-da4b-4d29-aa06-e6b505d0be87 | 010401-20121113103634Ch |              0.833333  |           0.833333  |            12 |   2662.13    |
| cf1fc335-5614-45d4-b28f-266b15486bac | 010401-20150423120034Ch |              0.833333  |           0.75      |            12 |   4937.68    |

Worst area-ratio offenders:

| ann_id                               | image_id                |   coco_area |   valid_area |   area_ratio |
|:-------------------------------------|:------------------------|------------:|-------------:|-------------:|
| 60db9f9e-312d-4c08-a606-7a1a2d91601c | 050202-20160430133734Ch |          73 |      77.3794 |     0.943404 |
| 3cc45daf-a23f-40df-8365-d12fa839bbb1 | 030302-20131030185354Bh |         288 |     301.406  |     0.955521 |
| 63a73081-bb7b-4649-b3cd-94421da69f79 | 010401-20131107043054Uh |         103 |      98.5882 |     1.04475  |
| 012b8a11-c4ab-47a6-8e9b-03180d56d69d | 010101-20150927111834Ch |         163 |     169.908  |     0.95934  |
| 81ea475f-0d32-4b0c-b530-0d119b4fc5a3 | 050202-20160430133734Ch |         147 |     153.188  |     0.959608 |
| bb7fe641-04a4-4e80-b8e5-445c78b6c328 | 050101-20110826074214Th |         103 |     106.933  |     0.963223 |
| 972a92b8-ee9c-4f5b-ac30-e83dfc0146d1 | 060403-20141225172214Mh |         275 |     265.242  |     1.03679  |
| a445ee72-2ed4-4f10-82c9-1e9e5aefd086 | 010103-20110114105034Ch |         263 |     253.701  |     1.03665  |
| 8c7fe5c5-8e67-4bca-8ae4-9920dd8af832 | 040401-20151107133734Ch |         149 |     143.893  |     1.03549  |
| 94cd1a80-8075-48c4-a718-0ea985fb2dc3 | 010402-20141112195854Bh |         268 |     277.457  |     0.965915 |

Worst bbox offenders:

| ann_id                               | image_id                |   bbox_max_dev_px |
|:-------------------------------------|:------------------------|------------------:|
| 025579ba-1f22-4fc5-a70b-7d4653c844f3 | 030302-20131030185354Bh |           63.4584 |
| b207fb08-7e5e-473c-b41c-5da3491370e8 | 010403-20120519070814Th |           38.1145 |
| a505d428-0a1c-4cf3-881c-173abfcac61f | 010102-20120330063134Lh |           36.9368 |
| ef2d1191-a443-42d6-87b0-f8af951048e2 | 010102-20150527161214Mh |           29.0037 |
| f8ac2898-3436-4605-92df-2a1bc6d2a23f | 010401-20141106195854Bh |           26.1224 |
| d7703f8e-696a-4236-b03a-040ec59f074c | 010102-20190602025430Lh |           25.1466 |
| 73361d0d-fe59-472b-9198-cce6e18a5f81 | 010102-20170418013450Uh |           24.2251 |
| 6aa7ef09-b879-4e6f-81ab-3ecce12245c2 | 010302-20151213023454Uh |           23.8616 |
| 2aaf1ac7-ebff-48e6-942f-f1a0c82f31b7 | 020401-20120225172914Mh |           22.6497 |
| 4d032552-657f-4faa-9bce-728d9ca22b04 | 010102-20160526011254Uh |           22.6197 |

## 3. Coverage

- image entries: 1154; unique file_names in json: 707
- duplicate image ids: none
- entries with 0 annotations: 0 []
- annotations referencing missing image ids: none
- disk train files not in json: 0 []
- json file_names missing from train dir: 0 []
- json file_names that exist in TEST dir: 0 []
- annotator copies per file (n_entries: n_files): {1: 411, 3: 151, 2: 145}
- annotations per image entry: min 1, max 26

## 4. Image format

- images not 2048x2048: 0 []
- PIL modes: {'L': 887}
- decoded channel counts (cv2, all images): {1: 887}
- 30-image sample: channels {1: 30}, max cross-channel diff: 0 (0 => genuinely grayscale even if stored 3-channel)
- jpeg impression: bytes/megapixel median 188924 (range 166082-237287); blockiness score median 1.139 (1.0 = no 8x8 block artifacts, >1.1 = visible compression blocking)
- overall file size stats (bytes): min 481212, median 788998, max 1105945

## 5. Timestamps

- filename-stem vs date_captured mismatches: 0 []
- unparsable: 0 []
- image id suffix != filename stem: 0 []