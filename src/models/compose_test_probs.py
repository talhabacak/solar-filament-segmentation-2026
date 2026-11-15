"""Compose a test-time semantic field from per-member maps (test_member_maps.py)
on the CPU, and write it where submit_v2.semantic_test_probs looks for the
recipe's cache, so submit_v2 --semantic <recipe> uses it without a GPU.

recipe_committee (ledger section 17): the structure measured on fold 0 --
    0.5 * mean(blob folds 0-4)
  + 0.5 * mean( mean(clDice folds 0-4), wdec4832, b2_2048, upernet1536,
                b5_1536, saaf1536, gn8_1536, rdn08 )
The clDice slot is the five-fold mean (as in every production recipe), the
seven arms are the fold-0 models, which held out fold 0 only and are therefore
legitimate test-time members; the two heavy members are absent because the
out-of-fold pool the selector was fit on never contained them.

Usage: python src/models/compose_test_probs.py --recipe recipe_committee
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEM = ROOT / "outputs" / "semantic"

ARMS = ["wdec4832", "b2_2048", "upernet1536", "b5_1536", "saaf1536", "gn8_1536", "rdn08"]
CLD = [f"cldice_f{k}" for k in range(5)]
BLOB = [f"blob_f{k}" for k in range(5)]

# recipe -> {member tag: weight}; weights sum to 1
RECIPES = {
    "recipe_committee": {**{t: 0.5 / 5 for t in BLOB},
                         **{t: 0.5 / 8 / 5 for t in CLD},
                         **{t: 0.5 / 8 for t in ARMS}},
    "recipe_committee4": {**{t: 0.5 / 5 for t in BLOB},
                          **{t: 0.5 / 4 / 5 for t in CLD},
                          **{t: 0.5 / 4 for t in ["b2_2048", "b5_1536", "wdec4832"]}},
    "recipe_committee9": {**{t: 0.5 / 5 for t in BLOB},
                          **{t: 0.5 / 9 / 5 for t in CLD},
                          **{t: 0.5 / 9 for t in ARMS + ["silu1536"]}},
    # sub025's recipe_blob5_50 with ONLY the clDice third replaced by the committee
    # third: 0.5 blob5 + 1/6 committee(clDice5 mean, 7 arms) + 1/6 heavy768 + 1/6 heavy1536.
    # The board read sub019 (blob5_50, heavies in) above sub021 (blob5_pure, heavies
    # out) and sub027 (pure + committee) at #5 below sub025, so the heavies stay and
    # the committee becomes the single variable against sub025.
    "recipe_committee_h": {**{t: 0.5 / 5 for t in BLOB},
                           **{t: (1 / 6) / 8 / 5 for t in CLD},
                           **{t: (1 / 6) / 8 for t in ARMS},
                           "heavy768": 1 / 6, "heavy1536": 1 / 6},
    # committee4 inside sub025's structure: the three arms the queue is training on
    # folds 1-4 (2048/B2, B5, wide decoder) plus the clDice five-fold mean as the
    # third; heavies and blob unchanged. Board read against sub029 (committee8_h)
    # and sub025 tells whether fewer, stronger members are the better committee.
    "recipe_committee4_h": {**{t: 0.5 / 5 for t in BLOB},
                            **{t: (1 / 6) / 4 / 5 for t in CLD},
                            **{t: (1 / 6) / 4 for t in ["b2_2048", "b5_1536", "wdec4832"]},
                            "heavy768": 1 / 6, "heavy1536": 1 / 6},
    # sub025's field rebuilt from member maps (control against the cached recipe_blob5_50)
    "recipe_blob5_50_check": {**{t: 0.5 / 5 for t in BLOB}, **{t: (1 / 6) / 5 for t in CLD},
                              "heavy768": 1 / 6, "heavy1536": 1 / 6},
    # the pure blob5 field rebuilt from member maps: must reproduce
    # test_probs_recipe_blob5_pure.npz up to uint8 rounding (the control)
    "recipe_blob5_pure_check": {**{t: 0.5 / 5 for t in BLOB}, **{t: 0.5 / 5 for t in CLD}},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", required=True, choices=sorted(RECIPES))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    w = RECIPES[a.recipe]
    assert abs(sum(w.values()) - 1) < 1e-9, sum(w.values())
    Z = {t: np.load(SEM / f"test_probs_m_{t}.npz") for t in w}
    stems = sorted(next(iter(Z.values())).files)
    for t, z in Z.items():
        if set(z.files) != set(stems):
            sys.exit(f"{t}: stem set differs")
    out = {}
    for s in stems:
        acc = np.zeros((2048, 2048), np.float32)
        for t, wt in w.items():
            acc += wt * Z[t][s].astype(np.float32)
        out[s] = np.rint(acc).astype(np.uint8)
    dest = SEM / (a.out or f"test_probs_{a.recipe}.npz")
    np.savez_compressed(dest, **out)
    print(f"{a.recipe}: {len(w)} members, {len(out)} stems -> {dest}")


if __name__ == "__main__":
    main()
