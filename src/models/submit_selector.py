"""Test submission with the verified component-selector policy.

Pipeline (mirrors the verified fold-0 evaluation exactly):
  predict @imgsz, conf>=0.05, max_det 300 -> greedy conf-desc pixel de-overlap
  -> r_norm<=0.97 -> selector features -> keep conf>=CONF_MIN and p>=P_MIN
Policy defaults from the adversarial verification: mid-plateau (0.25, 0.20);
honest expected gain +0.0084 PQ over the conf/area baseline.

Usage: python src/models/submit_selector.py --weights W --imgsz 1792 --out outputs/submissions/subN.csv
"""

import argparse
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.rle import write_submission
from selector import FEATURES, mask_features

R_MAX = 0.97


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1792)
    ap.add_argument("--conf-min", type=float, default=0.25)
    ap.add_argument("--p-min", type=float, default=0.20)
    ap.add_argument("--model", default="outputs/selector/selector_hgb.joblib")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(a.weights)
    bundle = joblib.load(ROOT / a.model)
    sel = bundle["model"] if isinstance(bundle, dict) else bundle
    feat_order = bundle.get("features", FEATURES) if isinstance(bundle, dict) else FEATURES
    assert list(feat_order) == list(FEATURES), f"feature order mismatch: {feat_order}"

    test_dir = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"
    preds_out = {}
    n_cand_total = n_kept_total = 0
    for p in sorted(test_dir.glob("*.jpeg")):
        res = yolo.predict(str(p), imgsz=a.imgsz, conf=0.05, iou=0.5,
                           max_det=300, verbose=False)[0]
        cands = []
        if res.masks is not None and len(res.masks.data):
            raw = res.masks.data.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            order = np.argsort(-confs)
            taken = np.zeros((IMG_H, IMG_W), dtype=bool)
            for i in order:
                m = cv2.resize(raw[i], (IMG_W, IMG_H),
                               interpolation=cv2.INTER_LINEAR) >= 0.5
                mm = m & ~taken
                if not mm.any():
                    continue
                f = mask_features(mm)
                if f["r_norm"] > R_MAX:
                    continue
                taken |= mm
                cands.append((float(confs[i]), mm, f))
        n_cand_total += len(cands)
        if not cands:
            preds_out[p.stem] = []
            continue
        rows = []
        for rank, (c, mm, f) in enumerate(cands, start=1):
            rows.append(dict(conf=c, log10_area=np.log10(f["area"]), **{
                k: f[k] for k in ("r_norm", "abs_lat", "bbox_aspect", "eccentricity",
                                  "boundary_complexity", "skel_len", "width_proxy")},
                n_cands_img=len(cands), rank_conf=rank))
        X = pd.DataFrame(rows)[FEATURES].to_numpy(dtype=np.float64)
        pr = sel.predict_proba(X)[:, 1]
        kept = [mm.astype(np.uint8) for (c, mm, f), pp in zip(cands, pr)
                if c >= a.conf_min and pp >= a.p_min]
        n_kept_total += len(kept)
        preds_out[p.stem] = kept

    n_empty = sum(1 for v in preds_out.values() if not v)
    print(f"test: {n_cand_total} candidates -> {n_kept_total} kept "
          f"({n_empty} empty images)")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = write_submission({k: v for k, v in preds_out.items() if v}, out)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
