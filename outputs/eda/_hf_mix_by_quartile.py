"""hf distribution of the training and test (leaderboard) images, and how we score where each lies.

Bins are hf quartiles and deciles with TRAINING edges, so every training bin holds the same number of
images and the test share per bin shows directly where the test set is heavier or lighter. Per bin:
sub036's out-of-fold PQ with its components (SQ, recall, precision) and an image-bootstrap 95 % CI,
the test share against the training share, and the PQ shift expected from the test mix. The
public/private split of the 180 test images is not disclosed, so "test" is both boards together.

Per-image scoring parts are cached in outputs/eda/sub036_oof_per_image.parquet (first run: minutes).
Writes outputs/eda/hf_mix_by_quartile.png.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from e115c_threshold_ci import per_image, pq  # noqa: E402

OUT = ROOT / "outputs/eda"
C4 = ["tp", "fp", "fn", "iou"]
cache = OUT / "sub036_oof_per_image.parquet"
if cache.exists():
    P = pd.read_parquet(cache)
else:
    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet").merge(
        pd.read_parquet(OUT / "bag5_oof_scores.parquet"), on="cand_id")
    folds = pd.read_parquet(ROOT / "data/splits/folds_v1.parquet")
    stems = np.array(sorted(folds["file_name"].str.replace(".jpeg", "", regex=False)))
    gt = pd.read_parquet(ROOT / "outputs/gt_cache/gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt = gt[gt["stem"].isin(set(stems))]
    ncopy = gt.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    k = (a.conf >= .40) & (a.p >= .60)
    st = per_image(a[k][["cand_id", "rle", "stem"]], gt, stems, ncopy)[0]
    P = pd.DataFrame({"stem": stems, **{c: np.asarray(st[c], float) for c in C4}})
    P.to_parquet(cache, index=False)

q = pd.read_parquet(OUT / "image_quality.parquet")
tr = q[q.split == "train"].merge(P, on="stem")
te = q[q.split == "test"].copy()
tot = pq(*tr[C4].sum().to_numpy())
print(f"hf, train n {len(tr)}: median {tr.hf.median():.0f}, IQR {tr.hf.quantile(.25):.0f}-{tr.hf.quantile(.75):.0f}, "
      f"range {tr.hf.min():.0f}-{tr.hf.max():.0f}")
print(f"hf, test  n {len(te)}: median {te.hf.median():.0f}, IQR {te.hf.quantile(.25):.0f}-{te.hf.quantile(.75):.0f}, "
      f"range {te.hf.min():.0f}-{te.hf.max():.0f}")
print(f"out-of-fold PQ {tot:.4f}\n")

rng = np.random.default_rng(0)
res = {}
for nb, labs in ((4, [f"Q{i}" for i in range(1, 5)]), (10, [f"D{i}" for i in range(1, 11)])):
    edges = tr.hf.quantile(np.linspace(0, 1, nb + 1)[1:-1]).to_numpy()
    cut = lambda x: pd.cut(x, [-np.inf, *edges, np.inf], labels=labs)
    tr["b"], te["b"] = cut(tr.hf), cut(te.hf)
    print(f"{'bin':4s}{'hf range':>14s}{'train':>7s}{'test':>6s}{'test/train':>11s}"
          f"{'PQ':>8s}{'95% CI':>18s}{'SQ':>7s}{'recall':>8s}{'prec':>7s}{'GT/img':>8s}")
    rows = []
    lo = [tr.hf.min(), *edges]
    hi = [*edges, tr.hf.max()]
    for i, lab in enumerate(labs):
        g = tr[tr.b == lab]
        X = g[C4].to_numpy()
        tp, fp, fn, iou = X.sum(0)
        boots = [pq(*X[rng.integers(0, len(X), len(X))].sum(0)) for _ in range(2000)]
        ntr, nte = len(g), int((te.b == lab).sum())
        sh_tr, sh_te = ntr / len(tr), nte / len(te)
        r = dict(bin=lab, lo=lo[i], hi=hi[i], n_train=ntr, n_test=nte, share_train=sh_tr, share_test=sh_te,
                 pq=pq(tp, fp, fn, iou), lo95=np.quantile(boots, .025), hi95=np.quantile(boots, .975),
                 sq=iou / max(tp, 1), recall=tp / (tp + fn), precision=tp / (tp + fp), gt_per_img=(tp + fn) / ntr)
        rows.append(r)
        print(f"{lab:4s}{r['lo']:>7.0f}-{r['hi']:<6.0f}{ntr:>7d}{nte:>6d}{sh_te / sh_tr:>11.2f}{r['pq']:>8.4f}"
              f"   [{r['lo95']:.4f}, {r['hi95']:.4f}]{r['sq']:>7.4f}{r['recall']:>8.4f}{r['precision']:>7.4f}"
              f"{r['gt_per_img']:>8.1f}")
    t = pd.DataFrame(rows)
    w = tr.b.map(dict(zip(t.bin, t.share_test / t.share_train))).astype(float).to_numpy()
    print(f"  test outside the training range: below {int((te.hf < tr.hf.min()).sum())}, "
          f"above {int((te.hf > tr.hf.max()).sum())}")
    print(f"  PQ expected under the test mix {pq(*(tr[C4].to_numpy() * w[:, None]).sum(0)):.4f} "
          f"(shift {pq(*(tr[C4].to_numpy() * w[:, None]).sum(0)) - tot:+.4f})\n")
    res[nb] = t

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
bins = np.linspace(min(tr.hf.min(), te.hf.min()), max(tr.hf.max(), te.hf.max()), 36)
ax[0].hist(tr.hf, bins=bins, density=True, alpha=.55, label=f"train (n={len(tr)})")
ax[0].hist(te.hf, bins=bins, density=True, alpha=.55, label=f"test (n={len(te)})")
for e in tr.hf.quantile([.25, .5, .75]):
    ax[0].axvline(e, color="k", lw=.8, ls="--")
ax[0].set_xlabel("hf (disc Laplacian variance)")
ax[0].set_ylabel("density")
ax[0].set_title("hf distribution; dashed = training quartile edges")
ax[0].legend()
t = res[4]
x = np.arange(4)
ax[1].bar(x - .2, t.share_train, .4, label="train share")
ax[1].bar(x + .2, t.share_test, .4, label="test share")
ax2 = ax[1].twinx()
ax2.errorbar(x, t.pq, yerr=[t.pq - t.lo95, t.hi95 - t.pq], fmt="o-", color="k", capsize=4, label="OOF PQ")
ax2.set_ylabel("out-of-fold PQ (95% CI)")
ax[1].set_xticks(x, t.bin)
ax[1].set_ylabel("share of images")
ax[1].set_title("image share and our PQ per hf quartile")
ax[1].legend(loc="lower left")
fig.tight_layout()
fig.savefig(OUT / "hf_mix_by_quartile.png", dpi=130)
print("wrote outputs/eda/hf_mix_by_quartile.png")
