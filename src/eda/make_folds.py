"""Freeze the leakage-free 5-fold CV split (v1).

Unit = physical image (file_name), 707 groups. Deterministic greedy round-robin
after sorting by composite stratum (site, annotator-copy count, year era,
median instance count) — balances all four marginals simultaneously.
Also defines a temporal stress holdout (last 20% by timestamp).
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SEED = 2026
N_FOLDS = 5

entries = pd.read_parquet(ROOT / "outputs" / "eda" / "manifest_entries.parquet")

# one row per physical file
grp = entries.groupby("file_name").agg(
    site=("site", "first"),
    year=("year", "first"),
    timestamp=("timestamp", "first"),
    n_copies=("entry_id", "nunique"),
    med_instances=("num_annotations", "median"),
    n_entries=("entry_id", "count"),
).reset_index()
assert len(grp) == 707, len(grp)

era = pd.cut(grp["year"], bins=[2010, 2013, 2017, 2022], labels=["2011-13", "2014-17", "2018-22"])
grp["era"] = era.astype(str)
grp["inst_q"] = pd.qcut(grp["med_instances"], 5, labels=False, duplicates="drop")

# deterministic shuffle within strata, then round-robin deal
rng = np.random.default_rng(SEED)
grp["_rand"] = rng.random(len(grp))
grp = grp.sort_values(["site", "n_copies", "era", "inst_q", "_rand"]).reset_index(drop=True)
grp["fold"] = np.arange(len(grp)) % N_FOLDS

# temporal stress holdout: last 20% by timestamp
grp = grp.sort_values("timestamp").reset_index(drop=True)
cut = int(len(grp) * 0.8)
grp["temporal_holdout"] = np.arange(len(grp)) >= cut
boundary = grp.loc[cut, "timestamp"]

out = grp[["file_name", "fold", "temporal_holdout", "site", "year", "era",
           "n_copies", "med_instances", "n_entries"]].sort_values("file_name").reset_index(drop=True)
dest = ROOT / "data" / "splits"
dest.mkdir(parents=True, exist_ok=True)
out.to_parquet(dest / "folds_v1.parquet", index=False)

sha = hashlib.sha256((dest / "folds_v1.parquet").read_bytes()).hexdigest()

# audit
lines = [f"# folds_v1 audit\n", f"seed={SEED}, n_folds={N_FOLDS}, files={len(out)}",
         f"sha256(folds_v1.parquet) = {sha}",
         f"temporal holdout boundary: {boundary} ({(~out['temporal_holdout']).sum()} train / {out['temporal_holdout'].sum()} holdout)\n"]
for col in ["site", "era", "n_copies"]:
    tab = pd.crosstab(out[col], out["fold"])
    lines.append(f"\n## {col} x fold\n\n{tab.to_markdown()}")
ent = out.groupby("fold")["n_entries"].sum()
med = out.groupby("fold")["med_instances"].mean()
files = out.groupby("fold").size()
lines.append(f"\n## per fold\n\nfiles: {files.to_dict()}\nentries: {ent.to_dict()}\nmean med_instances: {med.round(2).to_dict()}")
# leakage assertion
assert out.groupby("file_name")["fold"].nunique().max() == 1
lines.append("\nleakage check: every file_name in exactly one fold — OK")
(dest / "folds_v1_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:8]))
print("\nfiles per fold:", files.to_dict())
print("entries per fold:", ent.to_dict())
print("site x fold max dev:", (pd.crosstab(out['site'], out['fold']).std(axis=1).max()))
print("OK")
