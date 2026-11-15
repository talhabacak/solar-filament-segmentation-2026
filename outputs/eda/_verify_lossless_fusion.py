"""Is the bounding-box fusion code lossless? Compare a rebuilt test pool with the reference pool
(dumped by the full-frame code on 2026-09-16) column by column, and the CSV with sub033."""
import sys, filecmp, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
ref = pd.read_parquet("outputs/eda/test_pool_t020pox.parquet")
new = pd.read_parquet(sys.argv[1])
print("rows", len(ref), "->", len(new), "| columns equal:", sorted(ref.columns) == sorted(new.columns))
ref = ref.sort_values(["stem", "cand_id"]).reset_index(drop=True); new = new.sort_values(["stem", "cand_id"]).reset_index(drop=True)
bad = []
for c in ref.columns:
    a, b = ref[c].to_numpy(), new[c].to_numpy()
    if a.dtype.kind in "fc":
        same = np.array_equal(a, b, equal_nan=True)
    else:
        same = np.array_equal(a, b)
    if not same:
        n = int((~(a == b)).sum()) if a.dtype.kind not in "fc" else int((~np.isclose(a, b, rtol=0, atol=0, equal_nan=True)).sum())
        bad.append((c, n))
print("columns with any difference:", bad if bad else "none -- every feature, score and shrunk RLE is identical")
same_csv = filecmp.cmp(sys.argv[2], "outputs/submissions/sub033_t020_photoorient_p060.csv", shallow=False)
print("CSV byte-identical to sub033:", same_csv)
sys.exit(0 if (not bad and same_csv) else 1)
