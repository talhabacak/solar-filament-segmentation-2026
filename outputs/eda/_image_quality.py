"""E205 -- does image QUALITY move with time, and does it explain the late-year deficit?

Talha: is there a quality difference by time or by solar cycle phase, could we regress on cycle
phase (sin/cos), and is there anything to do about atmospheric blur / seeing?

The limb is a step function in truth: outside the disk there is sky. Everything that widens that
step -- atmospheric seeing, focus, tracking jitter -- is instrumental, so the measured 10-90 %
transition width across the limb is a direct per-frame seeing proxy, independent of the filaments
we are trying to find. Measured alongside it: high-frequency energy on the disk (a sharpness
proxy that does depend on content), the disk's own brightness and contrast, and the date.

R_SUN = 904.1 px and the centre offset 1.7 px are the fitted values of ledger 26c.
"""
import sys, pathlib, time, numpy as np, pandas as pd, cv2
ROOT = pathlib.Path(".").resolve()
IMG = ROOT / "data/MAGFiLO_1.0_Kaggle_2026/train/train_images"
TEST = ROOT / "data/MAGFiLO_1.0_Kaggle_2026/test/test_images"
R_SUN, C = 904.1, 1023.5
NTH = 720                                   # angular samples
RR = np.linspace(0.90, 1.10, 201)           # radial samples around the limb

th = np.linspace(0, 2 * np.pi, NTH, endpoint=False)
GX = (C + np.outer(RR * R_SUN, np.cos(th))).astype(np.float32)
GY = (C + np.outer(RR * R_SUN, np.sin(th))).astype(np.float32)
yy, xx = np.mgrid[0:2048, 0:2048]
DISK = ((xx - C) ** 2 + (yy - C) ** 2) < (0.85 * R_SUN) ** 2

def measure(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    f = img.astype(np.float32)
    prof = cv2.remap(f, GX, GY, cv2.INTER_LINEAR).mean(axis=1)      # mean radial profile
    inside, outside = prof[:20].mean(), prof[-20:].mean()           # disk level, sky level
    step = inside - outside
    # hf / disk statistics do not need the limb, so a frame whose limb step is too weak to fit
    # still contributes them -- an earlier version dropped nine frames entirely and left 77
    # candidates without the block. A failed limb fit is itself a quality flag, kept as limb_ok.
    if step >= 5:
        lvl = (prof - outside) / step
        r90 = np.interp(-0.90, -lvl, RR)    # interp needs increasing x: negate
        r10 = np.interp(-0.10, -lvl, RR)
        width, ok = float((r10 - r90) * R_SUN), 1
    else:
        width, ok = float("nan"), 0
    d = f[DISK]
    return dict(limb_width=width, limb_ok=ok, step=float(step),
                disk_mean=float(d.mean()), disk_sd=float(d.std()),
                hf=float(cv2.Laplacian(f, cv2.CV_32F, ksize=3)[DISK].var()))

rows = []
for tag, folder in (("train", IMG), ("test", TEST)):
    t0 = time.time()
    for k, p in enumerate(sorted(folder.glob("*.jpeg"))):
        m = measure(p)
        if m is None:
            continue
        m.update(stem=p.stem, split=tag, year=int(p.stem[:4]), month=int(p.stem[4:6]),
                 site=p.stem[-2])
        rows.append(m)
        if k % 200 == 0:
            print(f"  {tag} {k} ({time.time()-t0:.0f}s)", flush=True)
d = pd.DataFrame(rows)
d["date"] = pd.to_datetime(d["stem"].str[:8], format="%Y%m%d")
d.to_parquet("outputs/eda/image_quality.parquet", index=False)
print(f"\n{len(d)} goruntu olculdu ({(d.split=='train').sum()} egitim, {(d.split=='test').sum()} test)\n")
print("YILA GORE (limb_width = seeing vekili, px; hf = yuksek frekans enerjisi)")
g = d.groupby(["split", "year"]).agg(n=("stem", "size"), limb=("limb_width", "median"),
                                     hf=("hf", "median"), parlaklik=("disk_mean", "median"),
                                     kontrast=("disk_sd", "median")).round(2)
print(g.to_string())
print("\nGOZLEMEVINE GORE (egitim)")
print(d[d.split == "train"].groupby("site").agg(n=("stem", "size"), limb=("limb_width", "median"),
                                                hf=("hf", "median")).round(2).to_string())
print("\negitim vs test bir butun olarak:")
print(d.groupby("split").agg(limb=("limb_width", "median"), hf=("hf", "median"),
                             parlaklik=("disk_mean", "median"), kontrast=("disk_sd", "median")).round(2).to_string())
