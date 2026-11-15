"""E129 -- a deep selector against the boosted-tree one, on the same 42 features.

Talha's question: should the selector be deep learning rather than a gradient-
boosted tree? Three architectures are worth separating:

  1. a CNN over the image crop -- measured already: Spearman 0.604 against the
     tabular model's 0.678, blend gain ~0 (`train_mask_scorer.py`);
  2. an MLP over the same tabular features -- the honest "DL instead of ML"
     comparison, and the one this file runs;
  3. a set transformer over all candidates of an image, so a candidate is scored
     in the context of its competitors -- structurally the most interesting, and
     also run here, because the hand-engineered version of that context
     (neighbour counts, mass share, nearest strong candidate) measured -0.0013.

Both deep arms predict the same two-stage quantity the production selector
predicts, p(match) and E[iou_sum | match], from the same features, under the
same honest protocol: fit on four folds, quantile reference from inner
leave-one-fold-out scores, score the held-out fold, exact PQ at (0.40, 0.60).
Standardisation is fitted on the training folds only.

Sizes are chosen for the data: 9,917 candidates, ~4,100 positives, 42 features.
The MLP is 42-128-64-2 with dropout 0.2; the set transformer is one 4-head
attention block over each image's candidates (up to 64) on a 64-dim embedding.
Both are trained with AdamW for 60 epochs, batch 512 (MLP) / 32 images (set),
BCE for the head and masked MSE on positives for the value head.

Usage: python src/eda/selector_dl.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))
sys.path.insert(0, str(ROOT / "src" / "models"))
from filament.pq import score  # noqa: E402
from e115c_threshold_ci import per_image, pq  # noqa: E402
from extra_features import EXTRA  # noqa: E402

OUT = ROOT / "outputs" / "eda"
BUNDLE = ROOT / "outputs" / "selector" / "selector_v5_shrunkblob5prodts.joblib"
A_POOL = OUT / "e098d_fused_pool_all_blob5_rebuild_labelled.parquet"
POLICY, REPS, EQUIV = (0.40, 0.60), 4000, 0.005
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(*a, flush=True)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2))
        self.head_p = nn.Linear(64, 1)
        self.head_v = nn.Linear(64, 1)

    def forward(self, x, mask=None):
        h = self.body(x)
        return self.head_p(h).squeeze(-1), self.head_v(h).squeeze(-1)


class SetTransformer(nn.Module):
    """One self-attention block over the candidates of an image: each candidate
    is re-scored knowing its competitors, which is what a learned NMS does."""

    def __init__(self, d, dim=64, heads=4):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(d, dim), nn.GELU())
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(0.2))
        self.head_p = nn.Linear(dim, 1)
        self.head_v = nn.Linear(dim, 1)

    def forward(self, x, pad_mask):
        h = self.emb(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask)
        h = self.norm(h + a)
        h = h + self.ff(h)
        return self.head_p(h).squeeze(-1), self.head_v(h).squeeze(-1)


def fit_predict_mlp(Xtr, ytr, vtr, Xte, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEV)
    yt = torch.tensor(ytr, dtype=torch.float32, device=DEV)
    vt = torch.tensor(vtr, dtype=torch.float32, device=DEV)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        for i in range(0, n, 512):
            idx = perm[i:i + 512]
            p, v = m(Xt[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(p, yt[idx])
            pos = yt[idx] > 0.5
            if pos.any():
                loss = loss + nn.functional.mse_loss(v[pos], vt[idx][pos])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p, v = m(torch.tensor(Xte, dtype=torch.float32, device=DEV))
        return (torch.sigmoid(p) * v.clamp(min=0)).cpu().numpy(), torch.sigmoid(p).cpu().numpy()


def pack_images(X, y, v, groups, max_n=64):
    out = []
    for g in pd.unique(groups):
        idx = np.where(groups == g)[0][:max_n]
        out.append((idx, X[idx], y[idx], v[idx]))
    return out


def fit_predict_set(Xtr, ytr, vtr, gtr, Xte, gte, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = SetTransformer(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2)
    tr = pack_images(Xtr, ytr, vtr, gtr)
    for ep in range(epochs):
        np.random.shuffle(tr)
        for i in range(0, len(tr), 32):
            batch = tr[i:i + 32]
            L = max(len(b[0]) for b in batch)
            xb = torch.zeros(len(batch), L, Xtr.shape[1], device=DEV)
            yb = torch.zeros(len(batch), L, device=DEV)
            vb = torch.zeros(len(batch), L, device=DEV)
            pad = torch.ones(len(batch), L, dtype=torch.bool, device=DEV)
            for j, (_, x, y_, v_) in enumerate(batch):
                k = len(x)
                xb[j, :k] = torch.tensor(x, dtype=torch.float32, device=DEV)
                yb[j, :k] = torch.tensor(y_, dtype=torch.float32, device=DEV)
                vb[j, :k] = torch.tensor(v_, dtype=torch.float32, device=DEV)
                pad[j, :k] = False
            p, v = m(xb, pad)
            valid = ~pad
            loss = nn.functional.binary_cross_entropy_with_logits(p[valid], yb[valid])
            pos = valid & (yb > 0.5)
            if pos.any():
                loss = loss + nn.functional.mse_loss(v[pos], vb[pos])
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    raw = np.zeros(len(Xte)); pm = np.zeros(len(Xte))
    with torch.no_grad():
        for g in pd.unique(gte):
            idx = np.where(gte == g)[0]
            xb = torch.tensor(Xte[idx], dtype=torch.float32, device=DEV).unsqueeze(0)
            pad = torch.zeros(1, len(idx), dtype=torch.bool, device=DEV)
            p, v = m(xb, pad)
            pp = torch.sigmoid(p)[0].cpu().numpy()
            raw[idx] = pp * np.clip(v[0].cpu().numpy(), 0, None)
            pm[idx] = pp
    return raw, pm


def oof_dl(kind, a, feats, fold):
    X = a[feats].to_numpy(np.float64)
    y = a["y"].to_numpy(); v = a["iou_sum"].to_numpy(np.float64); grp = a["stem"].to_numpy()
    p, pmatch = np.zeros(len(a)), np.zeros(len(a))
    for f in range(5):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = (X - mu) / sd
        inner = np.zeros(len(tr))
        for g in [k for k in range(5) if k != f]:
            itr = np.where((fold != f) & (fold != g))[0]
            msk = fold[tr] == g
            if kind == "mlp":
                inner[msk], _ = fit_predict_mlp(Z[itr], y[itr], v[itr], Z[tr][msk])
            else:
                inner[msk], _ = fit_predict_set(Z[itr], y[itr], v[itr], grp[itr], Z[tr][msk], grp[tr][msk])
        ref = np.sort(inner)
        if kind == "mlp":
            raw, pm = fit_predict_mlp(Z[tr], y[tr], v[tr], Z[te])
        else:
            raw, pm = fit_predict_set(Z[tr], y[tr], v[tr], grp[tr], Z[te], grp[te])
        p[te] = np.searchsorted(ref, raw, side="right") / len(ref)
        pmatch[te] = pm
        log(f"    fold {f} done")
    return p, pmatch


def main():
    from sklearn.metrics import roc_auc_score
    feats = list(joblib.load(BUNDLE)["features"])
    a = pd.read_parquet(A_POOL)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    for c in ["shape_photometry_features.parquet", "orientation_context_features.parquet"]:
        ex = pd.read_parquet(OUT / c)
        a = a.merge(ex[["cand_id"] + [k for k in EXTRA if k in ex.columns]], on="cand_id", how="left")
    feats = feats + list(EXTRA)
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    gt["fold"] = gt["stem"].map(fold_of)
    gt = gt[gt["fold"].notna()].copy(); gt["fold"] = gt["fold"].astype(int)
    stems = np.array(sorted(set(a["stem"])))
    gts = gt[gt["stem"].isin(stems)]
    ncopy = gts.groupby("stem")["entry_id"].nunique().reindex(stems).fillna(1).to_numpy()
    log(f"pool {len(a)} rows, {len(feats)} features, device {DEV}")

    def exact(col, fold=None):
        sub = a[(a["conf"] >= POLICY[0]) & (a[col] >= POLICY[1])]
        g = gts
        if fold is not None:
            sub, g = sub[sub["fold"] == fold], gts[gts["fold"] == fold]
        return score(g[["filament_id", "segmentation_rle"]],
                     sub[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"}))

    # the deployed GBDT scores, recomputed here so the comparison is like for like
    from selector_regularization import oof as oof_gbdt
    res, parts = {}, {}
    for name, fn in (("gbdt", lambda: oof_gbdt("base", a, feats, a["fold"].to_numpy())),
                     ("mlp", lambda: oof_dl("mlp", a, feats, a["fold"].to_numpy())),
                     ("set", lambda: oof_dl("set", a, feats, a["fold"].to_numpy()))):
        log(f"\n{name}:")
        a[f"p_{name}"], pm = fn()
        per = [exact(f"p_{name}", f)["pq"] for f in range(5)]
        r = exact(f"p_{name}")
        sub = a[(a["conf"] >= POLICY[0]) & (a[f"p_{name}"] >= POLICY[1])]
        st, _ = per_image(sub, gts, stems, ncopy)
        parts[name] = st
        res[name] = dict(pq=r["pq"], auc=float(roc_auc_score(a["y"], pm)), kept=int(len(sub)),
                         tp=r["tp"], fp=r["fp"], fn=r["fn"], per_fold=per)
        log(f"{name:5s} AUC {res[name]['auc']:.4f} | kept {len(sub)} PQ {r['pq']:.4f} "
            f"TP {r['tp']} FP {r['fp']} FN {r['fn']} | per fold " + " ".join(f"{x:.4f}" for x in per))
    rng = np.random.default_rng(0)
    n = len(stems)
    for name in ("mlp", "set"):
        boot = np.empty(REPS)
        for i in range(REPS):
            idx = rng.integers(0, n, n)
            vals = [pq(s["tp"][idx].sum(), s["fp"][idx].sum(), s["fn"][idx].sum(), s["iou"][idx].sum())
                    for s in (parts["gbdt"], parts[name])]
            boot[i] = vals[1] - vals[0]
        d = res[name]["pq"] - res["gbdt"]["pq"]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res[name].update(delta=float(d), ci=[float(lo), float(hi)], p_gt0=float((boot > 0).mean()))
        log(f"  {name} vs gbdt: {d:+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  P(>0) {(boot > 0).mean():.3f}")
    json.dump(res, open(OUT / "selector_dl.json", "w"), indent=1, default=float)
    log(f"\nwrote {OUT / 'selector_dl.json'}")


if __name__ == "__main__":
    main()
