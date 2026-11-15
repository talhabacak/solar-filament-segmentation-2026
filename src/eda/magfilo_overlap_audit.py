"""MAGFiLO v1.0 public dataset vs competition overlap audit — METADATA ONLY.

Compliance policy (organizer ruling, discussion 729396: external ground truth is
forbidden): this script reads the public MAGFiLO JSON from quarantine and extracts
ONLY image identifiers and per-image entry/annotation COUNTS. No polygon, spine,
bbox, or any label geometry is read into outputs. The raw file stays in
outputs/external/magfilo_quarantine/ and must never be used for training/labels.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "outputs" / "external" / "magfilo_quarantine"
DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026"

with open(Q / "magfilo_2024_v1.0.json") as f:
    m = json.load(f)

print("top-level keys:", list(m.keys()))
images = m["images"]
print("public MAGFiLO image entries:", len(images))

# per-entry: id + file_name only
stem_of = {}
for im in images:
    stem = im["file_name"].rsplit(".", 1)[0]
    stem_of[im["id"]] = stem

# annotation COUNTS per entry (only image_id field is read)
ann_count = Counter(a["image_id"] for a in m["annotations"])
del m  # drop geometry from memory immediately

entries_per_stem = defaultdict(list)
for entry_id, stem in stem_of.items():
    entries_per_stem[stem].append(entry_id)

pub_stems = set(entries_per_stem)
print("public MAGFiLO unique stems:", len(pub_stems))

train_stems = {p.stem for p in (DATA / "train" / "train_images").glob("*.jpeg")}
test_stems = {p.stem for p in (DATA / "test" / "test_images").glob("*.jpeg")}

test_hit = sorted(test_stems & pub_stems)
train_hit = sorted(train_stems & pub_stems)
print(f"TEST stems in public MAGFiLO:  {len(test_hit)} / {len(test_stems)}")
print(f"TRAIN stems in public MAGFiLO: {len(train_hit)} / {len(train_stems)}")

def copy_dist(stems):
    return Counter(len(entries_per_stem[s]) for s in stems)

test_dist = copy_dist(test_hit)
train_dist = copy_dist(train_hit)
print("copies distribution for matched TEST stems:", dict(sorted(test_dist.items())))
print("copies distribution for matched TRAIN stems:", dict(sorted(train_dist.items())))

# validation vs known competition-train distribution (411/145/151)
comp_entries = pd.read_parquet(ROOT / "outputs" / "eda" / "manifest_entries.parquet")
comp_copies = comp_entries.groupby("file_name")["entry_id"].nunique()
print("competition train copies distribution:",
      dict(sorted(Counter(comp_copies.values).items())))

# per-stem annotation-count summary for matched test stems (counts only, no geometry)
rows = []
for s in test_hit:
    counts = [ann_count.get(e, 0) for e in entries_per_stem[s]]
    rows.append(dict(stem=s, n_copies=len(counts), ann_counts=str(sorted(counts))))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "outputs" / "external" / "magfilo_test_multiplicity.csv", index=False)

report = [
    "# MAGFiLO v1.0 public overlap audit (metadata only)\n",
    f"- Public MAGFiLO: {len(images)} image entries, {len(pub_stems)} unique stems",
    f"- Competition TEST stems found in public MAGFiLO: **{len(test_hit)} / 180**",
    f"- Competition TRAIN stems found in public MAGFiLO: **{len(train_hit)} / 707**",
    f"- Matched TEST stems annotator-copy distribution: {dict(sorted(test_dist.items()))}",
    f"- Matched TRAIN stems annotator-copy distribution: {dict(sorted(train_dist.items()))}",
    f"- Competition train (for reference): {dict(sorted(Counter(comp_copies.values).items()))}",
    "",
    "Policy: the raw public JSON stays in quarantine. Only stems and entry/annotation",
    "counts were extracted. No label geometry has been read into any output, and none",
    "may ever be used for training or predictions (organizer ruling, discussion 729396).",
]
(ROOT / "outputs" / "external" / "magfilo_overlap_report.md").write_text(
    "\n".join(report), encoding="utf-8")
print("report written")
