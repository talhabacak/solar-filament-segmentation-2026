"""Is the shape deficit THICKNESS or LENGTH? -- the question that decides whether a boundary
operator is the right tool at all.

Talha's warning: a morphological change to a thin ribbon is a structure change, so "dilate until
IoU crosses 0.5" can repair the METRIC while making the structure worse. The signature of that
trap is measurable: if our mask is a short piece of a long filament (a fragment), growing it
crosses 0.5 only by engulfing background, and precision |P^G|/|P| collapses while IoU rises. If
our mask is a thin copy of the right filament, growing it raises IoU AND precision stays put.

Read on two populations: the 838 near-miss copies, and the 425 that at least two annotators
independently drew (the only non-noise target left, ledger 39).
"""
import sys, pathlib, numpy as np, pandas as pd, cv2
from scipy.ndimage import binary_dilation, label as cc
ROOT = pathlib.Path(".").resolve()
sys.path.insert(0, str(ROOT / "src"))
from filament.rle import rles_to_layers
from filament import IMG_H, IMG_W

def skel_len(m):
    s = cv2.ximgproc.thinning((m.astype(np.uint8))*255).sum()/255 if hasattr(cv2,"ximgproc") else 0
    if s < 1:
        cs,_ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        s = max(sum(cv2.arcLength(c, True) for c in cs)/2.0, 1.0)
    return float(s)

atlas = pd.read_parquet("outputs/eda/error_atlas.parquet")
sh = atlas[atlas["outcome"]=="shape"]
targ = {"yakin-isabet (0.4-0.5)": set(sh[sh["best_pool"]>=0.4]["fid"]),
        "mutabakatli (>=2 anotator)": set(sh[sh["agree_iou"]>0.5]["fid"])}
want = set().union(*targ.values())
gt = pd.read_parquet("outputs/gt_cache/gt_full.parquet")
gt["stem"] = gt["file_name"].str.replace(".jpeg","",regex=False)
gt = gt[gt["filament_id"].isin(want)]
pool = pd.read_parquet("outputs/eda/fused_pool_all_blob5_t020_labelled.parquet")
rows = []
for stem, g in gt.groupby("stem"):
    c = pool[pool["stem"]==stem]
    if not len(c): continue
    gall = pd.read_parquet("outputs/gt_cache/gt_full.parquet")
    G = rles_to_layers(g["segmentation_rle"].tolist(), IMG_H, IMG_W).astype(bool)
    P = rles_to_layers(c["rle"].tolist(), IMG_H, IMG_W).astype(bool)
    pa = P.sum((1,2))
    for i,(fid) in enumerate(g["filament_id"]):
        it = (G[i]&P).sum((1,2)).astype(float)
        j = int(np.argmax(it/(G[i].sum()+pa-it)))
        gm, pm = G[i], P[j]
        ys,xs = np.nonzero(gm|pm)
        y0,y1,x0,x1 = max(0,ys.min()-8),min(IMG_H,ys.max()+9),max(0,xs.min()-8),min(IMG_W,xs.max()+9)
        gc,pc = gm[y0:y1,x0:x1], pm[y0:y1,x0:x1]
        inter = float((gc&pc).sum())
        d = dict(fid=fid, iou=inter/(gc.sum()+pc.sum()-inter),
                 prec=inter/max(pc.sum(),1), rec=inter/max(gc.sum(),1),
                 area_r=pc.sum()/max(gc.sum(),1),
                 skel_r=skel_len(pc)/max(skel_len(gc),1e-6),
                 ncc_pred=int(cc(pc)[1]), ncc_gt=int(cc(gc)[1]))
        d["width_r"] = d["area_r"]/max(d["skel_r"],1e-6)
        # what a +2px growth does: does IoU rise by engulfing background?
        q = binary_dilation(pc, iterations=2)
        it2 = float((gc&q).sum())
        d["iou_d2"] = it2/(gc.sum()+q.sum()-it2); d["prec_d2"] = it2/max(q.sum(),1)
        rows.append(d)
d = pd.DataFrame(rows)
d.to_parquet("outputs/eda/deficit_kind.parquet")
for name, ids in targ.items():
    s = d[d["fid"].isin(ids)]
    print(f"\n=== {name}  n={len(s)} ===")
    print(f"  IoU {s['iou'].median():.3f}   kesinlik |P^G|/|P| {s['prec'].median():.3f}   "
          f"duyarlilik |P^G|/|G| {s['rec'].median():.3f}")
    print(f"  alan orani {s['area_r'].median():.3f}   ISKELET orani {s['skel_r'].median():.3f}   "
          f"GENISLIK orani {s['width_r'].median():.3f}")
    print(f"  bagli bilesen: tahmin {s['ncc_pred'].median():.0f}  GT {s['ncc_gt'].median():.0f}")
    thin = (s["skel_r"]>0.8)&(s["width_r"]<0.85); frag = (s["skel_r"]<0.8)
    fat = (s["area_r"]>1.25)&(s["prec"]<0.7)
    other = ~(thin|frag|fat)
    print(f"  SINIFLAR: ince {thin.mean():.1%}  parca/kisa {frag.mean():.1%}  "
          f"sisman {fat.mean():.1%}  diger {other.mean():.1%}")
    print(f"  +2px buyutme: IoU {s['iou'].median():.3f} -> {s['iou_d2'].median():.3f}, "
          f"kesinlik {s['prec'].median():.3f} -> {s['prec_d2'].median():.3f}  "
          f"(0.5'i gecen {(s['iou_d2']>0.5).mean():.1%})")
    for cls, mask in (("ince", thin), ("parca", frag)):
        if mask.sum():
            t = s[mask]
            print(f"    {cls:6s}: +2px sonrasi 0.5 ustu {(t['iou_d2']>0.5).mean():5.1%}, "
                  f"kesinlik {t['prec'].median():.3f} -> {t['prec_d2'].median():.3f}")
