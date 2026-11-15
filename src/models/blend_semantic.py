"""Blend two cached semantic probability fields into one, per fold.

The production out-of-fold semantic field is an equal blend of the two members
of the semantic lane -- the clDice U-Net and the instance-balanced ("blob")
U-Net, each the fold's held-out model:

    fold{k}_probs_blob5ens.npz = round(0.5 * fold{k}_probs_blob1536.npz
                                     + 0.5 * fold{k}_probs_cldice.npz)

Fields are stored as uint8 (probability x 255) keyed by image stem, so the
blend is done in float32 and rounded back. This is the field
`build_fused_pool.py --sem-tag _blob5ens` reads; the test-time counterpart is
composed by `compose_test_probs.py` from per-member test maps.

Usage:
  python src/models/blend_semantic.py --folds 0,1,2,3,4 \
      --a blob1536 --b cldice --out blob5ens --weight 0.5
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEM = ROOT / "outputs" / "semantic"


def blend_fold(fold, tag_a, tag_b, tag_out, w):
    za = np.load(SEM / f"fold{fold}_probs_{tag_a}.npz")
    zb = np.load(SEM / f"fold{fold}_probs_{tag_b}.npz")
    stems = sorted(za.files)
    if set(stems) != set(zb.files):
        raise SystemExit(f"fold {fold}: stem sets differ between {tag_a} and {tag_b}")
    out = {s: np.rint(w * za[s].astype(np.float32) + (1 - w) * zb[s].astype(np.float32)).astype(np.uint8)
           for s in stems}
    dest = SEM / f"fold{fold}_probs_{tag_out}.npz"
    np.savez_compressed(dest, **out)
    return dest, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--a", default="blob1536", help="first field tag (weight --weight)")
    ap.add_argument("--b", default="cldice", help="second field tag (weight 1 - --weight)")
    ap.add_argument("--out", default="blob5ens")
    ap.add_argument("--weight", type=float, default=0.5)
    a = ap.parse_args()
    for f in [int(x) for x in a.folds.split(",")]:
        dest, n = blend_fold(f, a.a, a.b, a.out, a.weight)
        print(f"fold {f}: {n} stems -> {dest.name}", flush=True)


if __name__ == "__main__":
    main()
