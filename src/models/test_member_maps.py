"""Per-member TEST probability maps, one npz per model, so any ensemble recipe
can be composed afterwards on a CPU without touching a GPU again.

submit_v2.semantic_test_probs caches only the finished weighted mean of a
recipe; changing a weight or adding a member means re-inferring every model.
This writes outputs/semantic/test_probs_m_<tag>.npz (uint8, 180 stems, flip
TTA, each model at the size it was trained at, radial-denoise input restored
from the checkpoint for the RDN arm) and submit_v2's committee recipes read
these back through compose_test_probs().

Usage (on the GPU box):
  python src/models/test_member_maps.py --members \
      wdec4832=outputs/runs_semantic/sem_b3_f0_1536_cldice_wdec4832/best.pt \
      b2_2048=outputs/runs_semantic/sem_b3_f0_2048_cldice_b2/best.pt ...
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from semantic_candidates import load_model, predict_prob  # noqa: E402

SEM = ROOT / "outputs" / "semantic"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="tag=path/to/best.pt ...")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    stems = sorted(p.stem for p in TEST.glob("*.jpeg"))
    log(f"{len(stems)} test images")
    for spec in a.members:
        tag, w = spec.split("=", 1)
        out = SEM / f"test_probs_m_{tag}.npz"
        if out.exists() and not a.overwrite:
            log(f"{tag}: cached"); continue
        ck = torch.load(ROOT / w, map_location="cpu", weights_only=False)
        rdn = float(ck.get("denoise_radial", 0.0) or 0.0)
        dn = bool(ck.get("denoise", False))
        model, imgsz, channels = load_model(str(ROOT / w))
        t0, acc = time.time(), {}
        for s in stems:
            acc[s] = (predict_prob(model, imgsz, s, tta=True, img_dir=TEST, channels=channels,
                                   denoise_input=dn, denoise_radial=rdn) * 255).astype(np.uint8)
        np.savez_compressed(out, **acc)
        del model
        torch.cuda.empty_cache()
        log(f"{tag}: {len(acc)} maps at {imgsz}, rdn {rdn}, denoise {dn} -> {out.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
