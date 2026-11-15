"""How far apart are two submission CSVs? Written for the clean-environment rehearsal: the
champion was built in an older environment and its test probability map is not bit-reproducible,
so "can we rebuild the final submission" needs an answer in the units that matter -- instances
and masks, not pixels of an intermediate map.

Reports, for each pair: row counts, images with a different instance count, the exact-RLE
overlap, and for the rest a greedy IoU matching (how far a surviving mask moved, and how many
instances have no counterpart at all -- the only kind of difference that can change PQ).

Usage: python outputs/eda/_compare_submissions.py A.csv B.csv [C.csv ...]
"""
import sys, pathlib, itertools, numpy as np, pandas as pd
ROOT = pathlib.Path(".").resolve(); sys.path.insert(0, str(ROOT / "src"))
from filament.rle import decode_rle
from filament.pq import _entry_iou_matrix


def load(f):
    d = pd.read_csv(f)
    col = "segmentation_rle" if "segmentation_rle" in d.columns else d.columns[-1]
    idc = "filament_id" if "filament_id" in d.columns else d.columns[0]
    d = d.rename(columns={col: "rle", idc: "fid"})
    d["stem"] = d["fid"].astype(str).str.rsplit("_", n=1).str[0]
    return d[["fid", "stem", "rle"]]


def compare(A, B, na, nb):
    print(f"\n=== {na}  vs  {nb}")
    print(f"  rows {len(A)} vs {len(B)}; images {A['stem'].nunique()} vs {B['stem'].nunique()}")
    ca, cb = A.groupby("stem").size(), B.groupby("stem").size()
    idx = sorted(set(ca.index) | set(cb.index))
    ca, cb = ca.reindex(idx).fillna(0), cb.reindex(idx).fillna(0)
    print(f"  images with a different instance count: {int((ca != cb).sum())} of {len(idx)}"
          f" (count diff sum {int((cb - ca).sum()):+d})")
    exact = len(set(A["rle"]) & set(B["rle"]))
    print(f"  byte-identical RLEs: {exact} ({exact / max(len(A), 1):.1%} of A)")
    moved, lost_a, lost_b, ious = 0, 0, 0, []
    for stem in idx:
        ra = A.loc[A["stem"] == stem, "rle"].tolist()
        rb = B.loc[B["stem"] == stem, "rle"].tolist()
        same = set(ra) & set(rb)
        ra = [r for r in ra if r not in same]
        rb = [r for r in rb if r not in same]
        if not ra and not rb:
            continue
        if not ra or not rb:
            lost_a += len(ra); lost_b += len(rb); continue
        Ma = np.stack([decode_rle(r).astype(np.uint8) for r in ra])
        Mb = np.stack([decode_rle(r).astype(np.uint8) for r in rb])
        iou = _entry_iou_matrix(Ma, Mb)
        used_b = set()
        for i in range(len(ra)):
            j = int(np.argmax([iou[i, k] if k not in used_b else -1 for k in range(len(rb))]))
            if iou[i, j] > 0.5 and j not in used_b:
                used_b.add(j); ious.append(float(iou[i, j])); moved += 1
            else:
                lost_a += 1
        lost_b += len(rb) - len(used_b)
    if ious:
        q = np.percentile(ious, [5, 50, 95])
        print(f"  differing-but-matched masks: {moved}, IoU p5/median/p95 {q[0]:.3f}/{q[1]:.3f}/{q[2]:.3f}")
    print(f"  instances with NO counterpart at IoU>0.5: {lost_a} in A, {lost_b} in B"
          f"   <- the only differences that can move PQ")


files = sys.argv[1:]
subs = {pathlib.Path(f).stem: load(f) for f in files}
for (na, A), (nb, B) in itertools.combinations(subs.items(), 2):
    compare(A, B, na, nb)
print("\nCOMPARE_DONE")
