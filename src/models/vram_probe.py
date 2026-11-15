"""Which SwinV2+UPerNet configuration actually fits in 16 GB?

The 1024/batch-2 run OOMed in the decoder, not the encoder: UPerNet upsamples
every pyramid level to stride 4 (256x256 here) and concatenates them before the
head, so decoder cost grows with input area and with decoder_channels, and the
encoder's own attention is the cheap part.

Rather than guess a smaller number, measure peak allocation for a real
forward+backward at each candidate setting and pick the largest that fits with
headroom. Each config runs in a fresh process-level allocator state (empty_cache
+ reset_peak_memory_stats), and OOM is caught so one failure does not end the
sweep.

Usage: python src/models/vram_probe.py
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

BUDGET_GB = 15.9


def log(*a):
    print(*a, flush=True)


def build(arch, encoder, imgsz, dec_ch, ckpt):
    import segmentation_models_pytorch as smp
    kw = {"img_size": imgsz} if encoder.startswith("tu-swin") else {}
    if dec_ch is not None:
        kw["decoder_channels"] = dec_ch
    m = smp.UPerNet(encoder_name=encoder, encoder_weights=None, in_channels=3,
                    classes=1, **kw)
    if ckpt:
        # timm encoders expose this; it trades ~30% step time for activations
        inner = getattr(m.encoder, "model", m.encoder)
        if hasattr(inner, "set_grad_checkpointing"):
            inner.set_grad_checkpointing(True)
        else:
            return None
    return m.cuda()


def probe(encoder, imgsz, batch, dec_ch, ckpt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        m = build("upernet", encoder, imgsz, dec_ch, ckpt)
        if m is None:
            return "no-ckpt-support", None
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        x = torch.zeros(batch, 3, imgsz, imgsz, device="cuda")
        y = torch.zeros(batch, 1, imgsz, imgsz, device="cuda")
        t0 = time.time()
        for _ in range(2):                      # 2 steps: allocator reaches steady state
            with torch.amp.autocast("cuda"):
                loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        ms = (time.time() - t0) / 2 * 1000
        del m, opt, x, y
        return "ok", (peak, ms)
    except torch.OutOfMemoryError:
        return "OOM", None
    finally:
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="tu-swinv2_cr_tiny_ns_224")
    a = ap.parse_args()

    # batch stays >= 2: UPerNet's pyramid branch reduces to 1x1 and BatchNorm
    # cannot compute statistics from a single sample.
    grid = [(imgsz, 2, dec, ck)
            for imgsz, dec, ck in itertools.product((1024, 768), (None, 128, 64),
                                                    (False, True))]
    results = []
    for imgsz, batch, dec, ck in grid:
        tag = f"{imgsz} b{batch} dec={dec or 'default'} ckpt={int(ck)}"
        status, out = probe(a.encoder, imgsz, batch, dec, ck)
        if status == "ok":
            peak, ms = out
            fits = "FITS" if peak < BUDGET_GB * 0.85 else "tight"
            log(f"{tag:38s} peak {peak:5.2f} GB  {ms:6.0f} ms/step  {fits}")
            results.append(dict(imgsz=imgsz, batch=batch, decoder_channels=dec,
                                ckpt=ck, peak_gb=peak, ms_step=ms))
        else:
            log(f"{tag:38s} {status}")
            results.append(dict(imgsz=imgsz, batch=batch, decoder_channels=dec,
                                ckpt=ck, status=status))

    dest = ROOT / "outputs" / "eda" / "vram_probe_swin.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(results, f, indent=1)
    ok = [r for r in results if "peak_gb" in r and r["peak_gb"] < BUDGET_GB * 0.85]
    if ok:
        best = max(ok, key=lambda r: (r["imgsz"], -(r["ms_step"])))
        log(f"\npick: imgsz={best['imgsz']} decoder_channels={best['decoder_channels']} "
            f"ckpt={best['ckpt']} -> {best['peak_gb']:.2f} GB, {best['ms_step']:.0f} ms/step")
    log(f"saved -> {dest}")


if __name__ == "__main__":
    main()
