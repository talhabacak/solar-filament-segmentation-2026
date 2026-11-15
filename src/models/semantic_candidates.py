"""Turn the semantic model into instance candidates and measure what it ADDS.

The decisive question is not whether the semantic model is good on its own, but
whether it proposes the 2528 GT instances the detector never proposes. So this
reports, on a fold's val split:
  1. reachability of the semantic candidate pool alone,
  2. reachability of the detector pool alone (recomputed here),
  3. reachability of the UNION, and specifically how many detector-unreachable
     GT rows the semantic pool covers.
Then it scores fused policies with the exact PQ scorer.

Instance extraction: probability -> threshold -> remove small -> connected
components (optionally split by distance-transform watershed, off by default
because merge errors are only 4.9% of our FN).

Usage:
  python src/models/semantic_candidates.py --weights outputs/runs_semantic/<name>/best.pt --fold 0
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
DEVICE = "cuda"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.pq import score, _entry_iou_matrix
from filament.rle import decode_rle, encode_mask, rles_to_layers

DATA = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "train" / "train_images"
CACHE = ROOT / "outputs" / "oof_cache"
OUT = ROOT / "outputs" / "semantic"
DISK_C, DISK_R = 1023.5, 900.0
R_MAX = 0.97


def log(*a):
    print(*a, flush=True)


def load_model(weights):
    """Returns (model, imgsz, channels) — channels must match training exactly.

    Architecture, band count and encoder all come from the checkpoint, so the
    arch-experiment runs (in_ch=1, UPerNet, swin) load through the same path as
    the reference U-Net. A 1-band model reports channels="one", which
    predict_prob uses to build a matching input — the return arity stays a
    3-tuple so existing callers need no change.
    """
    import segmentation_models_pytorch as smp
    ck = torch.load(weights, map_location=DEVICE, weights_only=False)
    enc, in_ch = ck["encoder"], int(ck.get("in_ch", 3))
    kw = {"img_size": ck["imgsz"]} if enc.startswith("tu-swin") else {}
    if ck.get("arch", "unet") == "unetpp":
        dc = ck.get("decoder_channels")
        kw2 = {"decoder_channels": tuple(dc)} if dc else {}
        m = smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=in_ch,
                             classes=int(ck.get("classes", 1)), decoder_attention_type="scse", **kw2)
    elif ck.get("arch", "unet") == "upernet":
        m = smp.UPerNet(encoder_name=enc, encoder_weights=None,
                        in_channels=in_ch, classes=int(ck.get("classes", 1)), **kw)
    else:
        # decoder_channels defaults to smp's (256,128,64,32,16) for every
        # checkpoint written before the width became a swept axis; newer ones
        # carry their own, and rebuilding with the wrong width fails to load
        dc = ck.get("decoder_channels")
        if dc:
            kw["decoder_channels"] = tuple(dc)
        if ck.get("decoder_use_norm"):
            kw["decoder_use_norm"] = ck["decoder_use_norm"]
        if ck.get("decoder_interp", "nearest") != "nearest":
            kw["decoder_interpolation"] = ck["decoder_interp"]
        m = smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=in_ch,
                     classes=int(ck.get("classes", 1)), decoder_attention_type="scse", **kw)
        if ck.get("decoder_gn"):
            from decoder_norm import swap_decoder_bn_to_gn
            swap_decoder_bn_to_gn(m, int(ck["decoder_gn"]))
        if ck.get("decoder_att", "scse") == "coordatt":
            from coord_att import swap_decoder_scse_to_coordatt
            swap_decoder_scse_to_coordatt(m)
        if ck.get("decoder_act", "relu") != "relu":
            from decoder_norm import swap_decoder_act
            swap_decoder_act(m, ck["decoder_act"])
        if ck.get("saaf"):
            from saaf import apply_saaf
            apply_saaf(m)
        if ck.get("blurpool"):
            from blurpool import apply_blurpool
            apply_blurpool(m)
        if int(ck.get("ctx_stage", -1)) >= 0:
            # E174: the dilated-context module lives on the encoder skip and its weights
            # are in the checkpoint, so it has to exist before load_state_dict
            from stage_ctx import attach_stage_context
            attach_stage_context(m, int(ck["ctx_stage"]))
        if ck.get("hf_film"):
            # E213: the regime conditioning and its hf normalisation (mu, sd) are in the
            # checkpoint; the module must exist before load_state_dict
            from hf_film import attach
            attach(m, int(ck["imgsz"]))
    m = m.to(DEVICE).eval()
    m.load_state_dict(ck["model"])
    channels = "one" if in_ch == 1 else ck.get("channels", "rep")
    return m, ck["imgsz"], channels


@torch.no_grad()
def predict_prob(model, imgsz, stem, tta=True, img_dir=None, channels="rep",
                 denoise_input=False, denoise_radial=0.0):
    from train_semantic import local_contrast, radial_channel, denoise, radial_denoise
    path = Path(img_dir or DATA) / f"{stem}.jpeg"
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    small = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_AREA)
    if denoise_input:
        # A model trained with --denoise has only ever seen the filtered image.
        # Feeding it the raw one measures the preprocessing mismatch rather than
        # the mechanism, so this must match CachedSemantic exactly: bilateral
        # AFTER the resize to training size, never before.
        small = denoise(small)
    elif denoise_radial and denoise_radial > 0:
        # must match CachedSemantic: filtered outside r0, raw inside, same ramp
        small = radial_denoise(small, float(denoise_radial))
    raw = small.astype(np.float32) / 255.0
    if channels == "phys":
        stack = np.stack([raw, local_contrast(small), radial_channel(imgsz)])
    elif channels == "mag":
        # E162: the registered magnetogram, encoded exactly as build_magnetic_cache
        # does for training; a frame without one gets the neutral channels the
        # trainer's channel dropout already taught the network to accept
        sys.path.insert(0, str(ROOT / "src" / "data"))
        from magnetogram_features import load_registered
        from build_magnetic_cache import encode, NEUTRAL
        B = load_registered(stem)
        if B is None:
            c0 = np.full((imgsz, imgsz), NEUTRAL[0], np.uint8)
            c1 = np.full((imgsz, imgsz), NEUTRAL[1], np.uint8)
        else:
            c0, c1 = encode(B)
            c0 = cv2.resize(c0, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            c1 = cv2.resize(c1, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        stack = np.stack([raw, (c0.astype(np.float32) - 128.0) / 127.0, c1.astype(np.float32) / 255.0])
    elif channels == "one":
        stack = raw[None]
    else:
        stack = np.stack([raw, raw, raw])
    x = torch.from_numpy(stack).float()[None].to(DEVICE)
    with torch.amp.autocast("cuda"):
        p = torch.sigmoid(model(x)[:, :1])
        if tta:
            p = p + torch.flip(torch.sigmoid(model(torch.flip(x, dims=[3]))[:, :1]), dims=[3])
            p = p / 2
    p = p[0, 0].float().cpu().numpy()
    return cv2.resize(p, (IMG_W, IMG_H), interpolation=cv2.INTER_LINEAR)


def prob_to_instances(prob, thr, min_area):
    binm = (prob >= thr).astype(np.uint8)
    # Each component is compared and reduced on its own bounding box (from the stats) instead
    # of over the whole frame; the label image is the same, so the mask, its pixel order, and
    # every statistic over it are the same. The centroid is still the mean of the int64
    # coordinate arrays, not the float centroid OpenCV returns.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    out = []
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if a < min_area:
            continue
        x0, y0 = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        x1, y1 = x0 + int(stats[k, cv2.CC_STAT_WIDTH]), y0 + int(stats[k, cv2.CC_STAT_HEIGHT])
        mc = lab[y0:y1, x0:x1] == k
        ys, xs = np.nonzero(mc)
        ys += y0
        xs += x0
        r = float(np.hypot(xs.mean() - DISK_C, ys.mean() - DISK_C) / DISK_R)
        if r > R_MAX:
            continue
        m = np.zeros(lab.shape, dtype=bool)
        m[y0:y1, x0:x1] = mc
        out.append(dict(mask=m, area=a, r_norm=r, conf=float(prob[y0:y1, x0:x1][mc].mean())))
    return out


def reachability(cands_by_stem, gt_val):
    """Fraction of GT rows matched at IoU>0.5 by ANY candidate (upper bound)."""
    hit = set()
    for entry_id, sub in gt_val.groupby("entry_id"):
        stem = sub["file_name"].iloc[0].rsplit(".", 1)[0]
        cands = cands_by_stem.get(stem, [])
        if not cands:
            continue
        gt_layers = rles_to_layers(sub["segmentation_rle"].tolist())
        pred_layers = np.stack([c["mask"] for c in cands]).astype(np.uint8)
        iou = _entry_iou_matrix(gt_layers, pred_layers)
        for gi in np.nonzero((iou > 0.5).any(axis=1))[0]:
            hit.add(sub["filament_id"].iloc[int(gi)])
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thrs", default="0.3,0.5,0.7")
    ap.add_argument("--min-areas", default="64,256")
    ap.add_argument("--denoise", action="store_true",
                    help="bilateral-filter the input, for checkpoints trained "
                         "with train_semantic --denoise; the filter must match "
                         "training or the measurement is a domain shift")
    ap.add_argument("--denoise-radial", type=float, default=0.0,
                    help="radial bilateral denoise outside this r/R_sun, for checkpoints "
                         "trained with train_semantic --denoise-radial (must match)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tag", default="", help="suffix for the output npz")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    folds = pd.read_parquet(ROOT / "data" / "splits" / "folds_v1.parquet")
    val_files = sorted(folds.loc[folds["fold"] == a.fold, "file_name"])
    stems = [f.rsplit(".", 1)[0] for f in val_files]
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt_val = gt[gt["file_name"].isin(val_files)]
    log(f"fold {a.fold}: {len(val_files)} images, {len(gt_val)} GT rows")

    global DEVICE
    DEVICE = a.device
    model, imgsz, channels = load_model(a.weights)
    log(f"model imgsz {imgsz}, channels '{channels}'")
    probs = {}
    t0 = time.time()
    for i, s in enumerate(stems):
        probs[s] = predict_prob(model, imgsz, s, channels=channels,
                                denoise_input=a.denoise, denoise_radial=a.denoise_radial)
        if (i + 1) % 50 == 0:
            log(f"  predicted {i+1}/{len(stems)} ({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()

    # detector pool (champion model at NMS 0.05) for the comparison
    det = pd.read_parquet(CACHE / f"fold{a.fold}_preds_e013comp.parquet") \
        if (CACHE / f"fold{a.fold}_preds_e013comp.parquet").exists() \
        else pd.read_parquet(CACHE / f"fold{a.fold}_preds_e013.parquet")
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    if "view" in det.columns:
        det = det[det["view"] == "o"]
    det_by_stem = {}
    for stem, g in det.groupby("stem"):
        det_by_stem[stem] = [dict(mask=decode_rle(r.rle).astype(bool), area=int(r.area),
                                  r_norm=float(r.r_norm), conf=float(r.conf))
                             for r in g.itertuples()]
    det_hit = reachability(det_by_stem, gt_val)
    log(f"detector pool reachability: {len(det_hit)}/{len(gt_val)} "
        f"({len(det_hit)/len(gt_val):.4f})")

    results = {}
    total_gt = len(gt_val)
    for thr in [float(x) for x in a.thrs.split(",")]:
        for min_area in [int(x) for x in a.min_areas.split(",")]:
            sem_by_stem = {s: prob_to_instances(probs[s], thr, min_area) for s in stems}
            n_cand = sum(len(v) for v in sem_by_stem.values())
            sem_hit = reachability(sem_by_stem, gt_val)
            new = sem_hit - det_hit
            key = f"thr{thr}_area{min_area}"
            results[key] = dict(
                n_candidates=n_cand,
                sem_reach=len(sem_hit) / total_gt,
                union_reach=len(sem_hit | det_hit) / total_gt,
                new_over_detector=len(new),
                det_reach=len(det_hit) / total_gt)
            log(f"{key}: {n_cand} cands | sem reach {len(sem_hit)/total_gt:.4f} | "
                f"union {len(sem_hit|det_hit)/total_gt:.4f} | "
                f"NEW over detector {len(new)}")

            # standalone PQ of the semantic pool (greedy de-overlap by conf)
            rows = []
            for s, cands in sem_by_stem.items():
                for k, c in enumerate(sorted(cands, key=lambda d: -d["conf"]), start=1):
                    rows.append((f"{s}_{k}", encode_mask(c["mask"].astype(np.uint8))))
            if rows:
                pred_df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
                r = score(gt_val[["filament_id", "segmentation_rle"]], pred_df)
                results[key]["standalone_pq"] = r["pq"]
                results[key].update({f"standalone_{k}": v for k, v in r.items()
                                     if k in ("sq", "rq", "tp", "fp", "fn")})
                log(f"    standalone PQ {r['pq']:.4f} SQ {r['sq']:.4f} "
                    f"TP {r['tp']} FP {r['fp']} FN {r['fn']}")

    with open(OUT / f"fold{a.fold}_semantic_candidates.json", "w") as f:
        json.dump(results, f, indent=1)
    np.savez_compressed(OUT / f"fold{a.fold}_probs{a.tag}.npz",
                        **{s: (probs[s] * 255).astype(np.uint8) for s in stems})
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
