"""D1b precondition -- what does the production decoder's scSE attention
already do with the up-sampled branch versus the skip branch, on thin
filaments?

SAAF-lite (docs/24 D1b, _review_dfas_yolo.md §3.2) proposes re-weighting the
up-sampled map BEFORE it is concatenated with the skip, so that thin structure
in the skip is not overwhelmed. The production U-Net already has scSE
attention AFTER the concatenation, and scSE's channel gate (cSE) assigns one
weight per channel, so it can already down-weight the up-branch channels and
up-weight the skip channels if that is what the data wants. If it already does
so systematically on thin filaments, SAAF-lite is doing a job that is being
done, and the A100 hour is not worth spending. That is the measurement here;
no GPU is needed (a handful of fold-0 images through the fold-0 model on CPU).

Per decoder block, on each image, we read
  cSE  -> per-channel weights (B, C, 1, 1); split into the up-branch channels
          (the first `up` channels of the concatenation) and the skip channels
  sSE  -> the spatial gate map (B, 1, H, W)
and report the mean cSE weight of the up channels vs the skip channels, and
the mean sSE gate on thin-filament pixels (GT local width < 6 px) vs the rest
of the GT vs background. Pre-registered reading, from the review: if the up
branch's mean cSE weight is systematically below the skip branch's on every
full-resolution stage (say up < 0.5 x skip on the last two blocks), scSE is
already suppressing the up branch and D1b is NOT run; otherwise D1b stays
eligible for the A100.

Usage: python src/eda/scse_gate_probe.py --n 12
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle
from semantic_candidates import load_model

OUT = ROOT / "outputs" / "eda"
WEIGHTS = ROOT / "outputs" / "runs_semantic" / "sem_b3_f0_1536_cldice" / "best.pt"


def log(*a):
    print(*a, flush=True)


def thin_mask(gt_rles, imgsz):
    """GT pixels whose local width is below 6 px (distance transform x 2),
    plus the whole GT, both resized to the model grid."""
    full = np.zeros((IMG_H, IMG_W), np.uint8)
    thin = np.zeros((IMG_H, IMG_W), np.uint8)
    for r in gt_rles:
        m = decode_rle(r).astype(np.uint8)
        d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        full |= m
        thin |= ((m > 0) & (2 * d < 6)).astype(np.uint8)
    rs = lambda x: cv2.resize(x, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST).astype(bool)
    return rs(full), rs(thin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    # semantic_candidates.load_model builds on CUDA; the local card belongs to
    # another project, so mirror it on CPU (same keys, same architecture)
    import segmentation_models_pytorch as smp
    ck = torch.load(str(WEIGHTS), map_location="cpu", weights_only=False)
    kw = {}
    if ck.get("decoder_channels"):
        kw["decoder_channels"] = tuple(ck["decoder_channels"])
    if ck.get("decoder_use_norm"):
        kw["decoder_use_norm"] = ck["decoder_use_norm"]
    model = smp.Unet(encoder_name=ck["encoder"], encoder_weights=None, in_channels=int(ck.get("in_ch", 3)),
                     classes=1, decoder_attention_type="scse", **kw)
    model.load_state_dict(ck["model"])
    model = model.cpu().eval()
    imgsz, channels = ck["imgsz"], ck.get("channels", "rep")
    log(f"model {ck['encoder']} imgsz {imgsz} channels {channels} (CPU)")
    blocks = list(model.decoder.blocks)
    enc_ch = list(model.encoder.out_channels)
    # up-branch width per block: block 0 takes the deepest encoder map, later
    # blocks take the previous block's output
    ups = [enc_ch[-1]] + [b.conv2[0].out_channels for b in blocks[:-1]]
    totals = [b.conv1[0].in_channels for b in blocks]
    log("decoder blocks (in = up + skip):")
    for i, (t, u) in enumerate(zip(totals, ups)):
        log(f"  block {i}: total {t} = up {u} + skip {t - u}")

    caught = {}

    def hook(name):
        def f(mod, inp, out):
            caught[name] = out.detach()
        return f
    # smp wraps SCSEModule in an Attention module: block.attention1.attention.{cSE,sSE}.
    # The last block has no skip (skip_channels 0) and smp gives it an Identity
    # instead of scSE, so there is nothing to read there: the finest attention
    # stage is the 1/2-resolution block.
    probed = []
    for i, b in enumerate(blocks):
        scse = getattr(b.attention1, "attention", None)
        if scse is None or not hasattr(scse, "cSE"):
            log(f"  block {i}: no scSE (Identity, skip 0) -- not probed")
            continue
        scse.cSE.register_forward_hook(hook(f"c{i}"))
        scse.sSE.register_forward_hook(hook(f"s{i}"))
        probed.append(i)

    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)
    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    stems = sorted(folds.loc[folds["fold"] == 0, "file_name"].str.replace(".jpeg", "", regex=False))[: a.n]
    from semantic_candidates import DATA as IMG_DIR
    from train_semantic import local_contrast, radial_channel

    acc = {i: dict(c_up=[], c_skip=[], s_thin=[], s_gt=[], s_bg=[]) for i in range(len(blocks))}
    for k, stem in enumerate(stems):
        img = cv2.imread(str(IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
        raw = small.astype(np.float32) / 255.0
        if channels == "phys":
            stack = np.stack([raw, local_contrast(small), radial_channel(imgsz)])
        else:
            stack = np.stack([raw, raw, raw])
        x = torch.from_numpy(stack)[None]
        with torch.no_grad():
            model(x)
        if k == 0:
            # smp builds the last block's scSE module but never calls it (no skip
            # to concatenate), so a registered hook is not a fired hook: keep
            # only the blocks whose attention actually ran
            probed = [i for i in probed if f"c{i}" in caught]
            log(f"  scSE fired on blocks {probed}")
        full, thin = thin_mask(gt.loc[gt["stem"] == stem, "segmentation_rle"].tolist(), imgsz)
        for i in probed:
            c = caught[f"c{i}"][0, :, 0, 0].numpy()
            u = ups[i]
            acc[i]["c_up"].append(float(c[:u].mean()))
            acc[i]["c_skip"].append(float(c[u:].mean()) if len(c) > u else float("nan"))
            s = caught[f"s{i}"][0, 0].numpy()
            sh = s.shape[0]
            rs = lambda m: cv2.resize(m.astype(np.uint8), (sh, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
            t, g = rs(thin), rs(full)
            acc[i]["s_thin"].append(float(s[t].mean()) if t.any() else float("nan"))
            acc[i]["s_gt"].append(float(s[g & ~t].mean()) if (g & ~t).any() else float("nan"))
            acc[i]["s_bg"].append(float(s[~g].mean()))
        log(f"  {k+1}/{len(stems)} {stem}")

    log(f"\n  {'block':<6}{'grid':>6}{'cSE up':>9}{'cSE skip':>10}{'up/skip':>9}{'sSE thin':>10}{'sSE gt':>8}{'sSE bg':>8}")
    rows = {}
    for i in probed:
        cu, cs = np.nanmean(acc[i]["c_up"]), np.nanmean(acc[i]["c_skip"])
        st, sg, sb = np.nanmean(acc[i]["s_thin"]), np.nanmean(acc[i]["s_gt"]), np.nanmean(acc[i]["s_bg"])
        grid = caught[f"s{i}"].shape[-1]
        rows[i] = dict(grid=grid, c_up=cu, c_skip=cs, ratio=cu / cs if cs == cs and cs > 0 else float("nan"),
                       s_thin=st, s_gt=sg, s_bg=sb)
        log(f"  {i:<6}{grid:>6}{cu:>9.3f}{cs:>10.3f}{cu/cs if cs else float('nan'):>9.2f}{st:>10.3f}{sg:>8.3f}{sb:>8.3f}")
    last2 = [rows[i]["ratio"] for i in probed[-2:] if rows[i]["c_skip"] == rows[i]["c_skip"]]
    suppress = bool(last2) and all(r < 0.5 for r in last2)
    log(f"\n  pre-registered reading: up/skip cSE ratio on the last two scSE blocks ({probed[-2:]}) = "
        + ", ".join(f"{r:.2f}" for r in last2)
        + (" -> scSE ALREADY suppresses the up branch: D1b NOT run" if suppress
           else " -> scSE does not suppress the up branch: D1b stays eligible"))
    import json
    json.dump(dict(n=len(stems), blocks=rows, d1b_eligible=not suppress), open(OUT / "scse_gate_probe.json", "w"),
              indent=1, default=float)
    log(f"wrote {OUT/'scse_gate_probe.json'}")


if __name__ == "__main__":
    main()
