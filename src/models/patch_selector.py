"""E151 -- a selector that looks at the image.

E150 located the only large open axis. Holding the candidates, the masks, the
budget and the operating point exactly as deployed, and improving ONLY the
ranking:

    deployed                      0.4422
    perfect ranking, same budget  0.4743      +0.0321
    perfect ranking and budget    0.5127      (+0.0384 more, unavailable: E143
                                               showed those candidates are below
                                               the break-even individually)

and the change is small and targeted: a perfect ranking swaps 571 of the 4001
kept candidates. What it drops is 71.5% false with mean realised value 0.181;
what it takes is 23.1% false with value 0.594.

The production selector cannot make those swaps because it never sees the image.
It receives 42 summary numbers -- areas, confidences, field statistics, the
photometry and orientation features of E123 -- and every one of them is a scalar
someone chose. Talha put it plainly: "secici goruntuden beslenmiyor".

This gives it pixels. For each candidate a three-channel patch is cropped around
its bounding box:

    channel 0   the H-alpha image, the evidence a human annotator uses
    channel 1   the semantic probability field, the evidence the pipeline uses
    channel 2   the candidate's own mask, so the network knows which structure
                in the patch it is being asked about

A ResNet-18 (ImageNet weights; cross-domain pretraining is permitted, external
filament labels are not) predicts two things per patch: whether the candidate
matches a ground-truth row, and the iou_sum it realises. Those two out-of-fold
predictions become two extra columns for the existing two-stage selector, which
keeps every earlier result intact and makes the comparison single-variable: the
same GBDT, the same gate, the same policy, 42 features against 44.

Honest by construction: the network for fold k is trained only on the other four
folds, and its predictions on fold k are therefore out of fold, exactly as the
GBDT's inner quantile reference already is.

Usage:
  python src/models/patch_selector.py --folds 0,1,2,3,4 --epochs 12 --size 128
  python src/models/patch_selector.py --emit          # write the OOF columns only
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in ("src", "src/eda", "src/models"):
    sys.path.insert(0, str(ROOT / p))
from filament.rle import decode_rle  # noqa: E402

OUT = ROOT / "outputs" / "eda"
SEM = ROOT / "outputs" / "semantic"
IMG = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
GATE_CONF = 0.40


def log(*a):
    print(*a, flush=True)


def crop(img, prob, mask, box, size, pad_frac=0.35):
    """A square crop around the candidate, padded so context is visible, resized."""
    y0, y1, x0, x1 = box
    h, w = y1 - y0, x1 - x0
    side = int(max(h, w) * (1 + 2 * pad_frac))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    ay0, ax0 = cy - side // 2, cx - side // 2
    ay1, ax1 = ay0 + side, ax0 + side
    out = np.zeros((3, side, side), np.float32)
    sy0, sx0 = max(ay0, 0), max(ax0, 0)
    sy1, sx1 = min(ay1, img.shape[0]), min(ax1, img.shape[1])
    if sy1 > sy0 and sx1 > sx0:
        dy0, dx0 = sy0 - ay0, sx0 - ax0
        dy1, dx1 = dy0 + (sy1 - sy0), dx0 + (sx1 - sx0)
        out[0, dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
        out[1, dy0:dy1, dx0:dx1] = prob[sy0:sy1, sx0:sx1]
        out[2, dy0:dy1, dx0:dx1] = mask[sy0:sy1, sx0:sx1]
    return np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) for c in out])


def build_patches(a, size, cache):
    """One patch per candidate; cached because the crop is the expensive part."""
    if cache.exists():
        z = np.load(cache)
        if len(z["cand_id"]) == len(a) and set(z["cand_id"]) == set(a["cand_id"]):
            log(f"patch cache hit: {cache.name} {z['x'].shape}")
            return z["x"], list(z["cand_id"])
    xs, ids, t0 = [], [], time.time()
    for (fold, stem), gg in a.groupby(["fold", "stem"], sort=False):
        img = cv2.imread(str(IMG / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        prob = np.load(SEM / f"fold{int(fold)}_probs_blob5ens.npz")[stem].astype(np.float32) / 255.0
        for cid, rle in zip(gg["cand_id"], gg["rle"]):
            m = decode_rle(rle).astype(np.float32)
            ys, xs_ = np.nonzero(m)
            if len(ys) == 0:
                continue
            xs.append(crop(img, prob, m, (ys.min(), ys.max() + 1, xs_.min(), xs_.max() + 1), size))
            ids.append(cid)
        if len(ids) % 2000 < len(gg):
            log(f"  cropped {len(ids)} patches ({time.time() - t0:.0f}s)")
    x = np.stack(xs).astype(np.float16)
    np.savez_compressed(cache, x=x, cand_id=np.array(ids))
    log(f"wrote {cache.name} {x.shape} ({time.time() - t0:.0f}s)")
    return x, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tag", default="patch")
    # the local RTX belongs to another project, and an empty CUDA_VISIBLE_DEVICES does
    # not disable CUDA on Windows, so the device is an explicit argument rather than
    # something inferred from the environment
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a_ = ap.parse_args()

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision

    a = pd.read_parquet(OUT / "fused_pool_all_blob5_t020_labelled.parquet")
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    fold_of = dict(zip(folds["file_name"].str.replace(".jpeg", "", regex=False), folds["fold"]))
    a["fold"] = a["stem"].map(fold_of).astype(int)
    # the ranking only matters where the gate is live, so the network is trained on
    # the candidates the confidence gate admits rather than on the whole pool
    a = a[a["conf"] >= GATE_CONF].reset_index(drop=True)
    log(f"{len(a)} candidates past the confidence gate, {a['y'].mean():.1%} true")

    x, ids = build_patches(a, a_.size, OUT / f"_patches_{a_.size}.npz")
    order = {c: i for i, c in enumerate(ids)}
    a = a[a["cand_id"].isin(order)].copy()
    a["row"] = a["cand_id"].map(order)
    X = torch.from_numpy(x[a["row"].to_numpy()].astype(np.float32))
    y = torch.from_numpy(a["y"].to_numpy().astype(np.float32))
    v = torch.from_numpy(a["iou_sum"].to_numpy().astype(np.float32))
    f = a["fold"].to_numpy()
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a_.device == "auto" else a_.device
    log(f"patches {tuple(X.shape)} on {dev}")

    oof_p, oof_v = np.zeros(len(a)), np.zeros(len(a))
    for F in [int(k) for k in a_.folds.split(",")]:
        tr, te = np.where(f != F)[0], np.where(f == F)[0]
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Linear(512, 2)
        net = net.to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a_.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a_.lr, total_steps=a_.epochs * max(len(tr) // a_.batch, 1))
        bce, mse = nn.BCEWithLogitsLoss(), nn.MSELoss()
        dl = DataLoader(TensorDataset(X[tr], y[tr], v[tr]), batch_size=a_.batch, shuffle=True,
                        drop_last=True)
        t0 = time.time()
        for ep in range(a_.epochs):
            net.train()
            tot = 0.0
            for xb, yb, vb in dl:
                xb, yb, vb = xb.to(dev), yb.to(dev), vb.to(dev)
                if np.random.rand() < 0.5:                      # the only augmentation the
                    xb = torch.flip(xb, dims=[3])               # geometry allows for free
                out = net(xb)
                # the value head is trained on the positives only, as the GBDT regressor is
                pos = yb > 0.5
                loss = bce(out[:, 0], yb)
                if pos.any():
                    loss = loss + mse(out[pos, 1], vb[pos])
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                tot += float(loss) * len(xb)
            log(f"  fold {F} epoch {ep}: loss {tot / len(tr):.4f} ({time.time() - t0:.0f}s)")
        net.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(te), 256):
                preds.append(net(X[te[i:i + 256]].to(dev)).cpu().numpy())
            pr = np.concatenate(preds)
        oof_p[te] = 1 / (1 + np.exp(-pr[:, 0]))
        oof_v[te] = np.clip(pr[:, 1], 0, None)
        from sklearn.metrics import roc_auc_score
        log(f"fold {F}: patch AUC {roc_auc_score(a['y'].to_numpy()[te], oof_p[te]):.4f} "
            f"({time.time() - t0:.0f}s)")

    out = pd.DataFrame({"cand_id": a["cand_id"].to_numpy(),
                        "patch_p": oof_p, "patch_v": oof_v})
    out.to_parquet(OUT / f"{a_.tag}_oof.parquet", index=False)
    from sklearn.metrics import roc_auc_score
    log(f"\npooled patch AUC {roc_auc_score(a['y'].to_numpy(), oof_p):.4f} "
        f"(the production selector's is 0.907 on the full pool)")
    log(f"wrote {OUT / f'{a_.tag}_oof.parquet'} ({len(out)} rows)")


if __name__ == "__main__":
    main()
