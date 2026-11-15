"""E140 -- what actually pays in a committee: how many members, or how different?

The committee works (E139: +0.0090 fold 0, +0.0045 replicated over two folds
with four members) and no single member is essential (leave-one-out +9..+15).
That leaves the question that decides what to train NEXT: is the gain driven by
the NUMBER of members or by how DECORRELATED they are?

The two answers point opposite ways. If it is count, the cheapest path is more
seeds of the architecture we already trust. If it is decorrelation, seeds are
worthless and the eight shelved arms -- every one of which lost on its own --
are the resource, and the next arm should be chosen to be as unlike the others
as possible.

Design: subsets are chosen to break the confound. At a fixed member count the
MOST-correlated and LEAST-correlated subsets are both built, so count and
decorrelation move independently:

    k=2   base + nearest neighbour        base + furthest neighbour
    k=4   the four most similar           the four most dissimilar
    k=6   two subsets spanning the range
    k=8   all arms

Each subset becomes a field (0.5*blob + 0.5*mean(members), the production
blend), a pool at threshold 0.2, and a PQ against the same champion selector
fitted on folds 1-4. The reported curve is PQ against mean pairwise correlation
at each k.

Usage: python src/eda/committee_scaling_law.py [--build-only]
"""

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEM = ROOT / "outputs" / "semantic"
OUT = ROOT / "outputs" / "eda"
ARMS = ["cldice", "1536wdec4832", "2048b2", "upernet1536", "b5_1536", "saaf1536", "gn8_1536", "rdn08"]


def log(*a):
    print(*a, flush=True)


def corr_matrix(Z, stems):
    """Pearson between arms on an 8-px lattice over 40 images (the E139 protocol)."""
    V = []
    for t in ARMS:
        v = np.concatenate([Z[t][s][::8, ::8].ravel().astype(np.float32) for s in stems])
        V.append(v)
    V = np.stack(V)
    return np.corrcoef(V)


def main():
    build_only = "--build-only" in sys.argv
    Z = {t: np.load(SEM / f"fold0_probs_{t}.npz") for t in ARMS}
    stems = sorted(Z["cldice"].files)[:40]
    C = corr_matrix(Z, stems)
    log("pairwise correlation (fold 0, 40 images, 8-px lattice):")
    log("            " + " ".join(f"{t[:7]:>7s}" for t in ARMS))
    for i, t in enumerate(ARMS):
        log(f"{t:>12s} " + " ".join(f"{C[i, j]:7.3f}" for j in range(len(ARMS))))

    base = ARMS.index("cldice")
    order = np.argsort(C[base])          # least correlated with the base first
    subsets = {
        "k1_base": ["cldice"],
        "k2_near": ["cldice", ARMS[order[-2]]],
        "k2_far": ["cldice", ARMS[order[0]]],
    }

    def mean_corr(names):
        idx = [ARMS.index(n) for n in names]
        p = [C[i, j] for i, j in itertools.combinations(idx, 2)]
        return float(np.mean(p)) if p else 1.0

    for k in (4, 6):
        best = min(itertools.combinations(ARMS, k), key=mean_corr)      # most dissimilar
        worst = max(itertools.combinations(ARMS, k), key=mean_corr)     # most similar
        subsets[f"k{k}_far"] = list(best)
        subsets[f"k{k}_near"] = list(worst)
    subsets["k8_all"] = list(ARMS)

    log("\nsubsets:")
    for n, m in subsets.items():
        log(f"  {n:9s} corr {mean_corr(m):.4f}  {', '.join(m)}")
    json.dump({n: dict(members=m, mean_corr=mean_corr(m)) for n, m in subsets.items()},
              open(OUT / "committee_subsets.json", "w"), indent=1)

    blob = np.load(SEM / "fold0_probs_blob1536.npz")
    for name, members in subsets.items():
        f = SEM / f"fold0_probs_cs_{name}.npz"
        if f.exists():
            log(f"{name}: field exists"); continue
        out = {}
        for s in Z["cldice"].files:
            m = np.mean([Z[t][s].astype(np.float32) for t in members], axis=0)
            out[s] = np.clip(np.rint(0.5 * blob[s].astype(np.float32) + 0.5 * m), 0, 255).astype(np.uint8)
        np.savez_compressed(f, **out)
        log(f"{name}: wrote {f.name}")
    if build_only:
        return

    tags = []
    for name in subsets:
        p = OUT / f"fused_pool_f0_cs_{name}_t020_labelled.parquet"
        if not p.exists():
            subprocess.run([sys.executable, "src/models/build_fused_pool.py", "--folds", "0",
                            "--sem-tag", f"_cs_{name}", "--sem-thr", "0.2",
                            "--out", f"fused_pool_f0_cs_{name}_t020.parquet"],
                           cwd=ROOT, stdout=open(OUT / f"_pool_cs_{name}.log", "w"), stderr=subprocess.STDOUT)
            subprocess.run([sys.executable, "outputs/eda/_label_one_pool.py",
                            f"fused_pool_f0_cs_{name}_t020"], cwd=ROOT)
        tags.append(f"cs_{name}_t020")
        log(f"{name}: pool ready")
    subprocess.run([sys.executable, "outputs/eda/_score_arms_f0.py"] + tags, cwd=ROOT)


if __name__ == "__main__":
    main()
