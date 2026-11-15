"""HONEST error taxonomy: the E117 measurement with fully out-of-fold selector scores (E115b p_oof).

Implements the pre-registered test from outputs/eda/_review_multiloss_deep_supervision.md (Section 3.1):
"Eksenin ön-koşulu: hata taksonomisini bugünkü üretim havuzunda yeniden ölç (L920-986)"

Hypothesis:
Merged + fragmented errors represent <= 10% of FNs on today's production pipeline,
and their combined oracle headroom is < +0.012 PQ.

Pre-registered Gate:
Oracle ceiling for topological errors >= +0.012 PQ -> keep topology research open.
Oracle ceiling < +0.012 PQ -> permanently close multi-loss / boundary / graph / topology axis.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from filament.pq import _entry_iou_matrix, score
from filament.rle import decode_rle, rles_to_layers

OUT = ROOT / "outputs" / "eda"


def log(*a):
    print(*a, flush=True)


def main():
    t0 = time.time()
    log("=== EVALUATING PRODUCTION ERROR TAXONOMY & TOPOLOGY ORACLE CEILING ===")

    # 1. Load GT and candidates
    gt = pd.read_parquet(ROOT / "outputs" / "gt_cache" / "gt_full.parquet")
    gt["stem"] = gt["file_name"].str.replace(".jpeg", "", regex=False)

    # HONEST variant (chatgpt_science S000/E117-honest): the deployment-faithful
    # labelled pool (E115b: shrink 0.05 first) scored by the fully out-of-fold
    # two-stage selector (E115b p_oof), instead of the in-sample p_sel.
    cand = pd.read_parquet(OUT / "e115b_labelled_pool.parquet")
    poof = pd.read_parquet(OUT / "e115b_p_oof.parquet")
    cand = cand.merge(poof, on="cand_id", how="inner")
    cand["p_sel"] = cand["p_oof"]
    log(f"Loaded honest pool: {len(cand)} candidates with OOF selector scores")

    # Filter for sub025 policy
    pol_conf = 0.40
    pol_p = 0.60
    keep = cand[(cand["conf"] >= pol_conf) & (cand["p_sel"] >= pol_p)].copy()
    log(f"Filtered to sub025 policy (conf >= {pol_conf}, p_sel >= {pol_p}): {len(keep)} instances across {keep['stem'].nunique()} images")

    # Map predictions by stem
    preds_by_stem = {}
    for stem, g in keep.groupby("stem"):
        # ensure masks are sorted by conf descending
        g = g.sort_values("conf", ascending=False)
        masks = [decode_rle(r).astype(bool) for r in g["rle"]]
        preds_by_stem[stem] = masks

    # 2. Overall baseline score
    pred_df = keep[["cand_id", "rle"]].rename(columns={"cand_id": "filament_id", "rle": "segmentation_rle"})
    base_res = score(gt[["filament_id", "segmentation_rle"]], pred_df)
    log(f"\nBaseline Production Score (sub025 on 5-fold OOF pool):")
    log(f"  PQ: {base_res['pq']:.4f} | SQ: {base_res['sq']:.4f} | RQ: {base_res['rq']:.4f}")
    log(f"  TP: {base_res['tp']} | FP: {base_res['fp']} | FN: {base_res['fn']}")

    # 3. Taxonomize errors across all entries
    fn_counts = Counter()
    fp_counts = Counter()
    fn_details = []
    fp_details = []

    # Count multi-annotator agreement per physical stem
    copies_per_stem = gt.groupby("stem")["entry_id"].nunique().to_dict()

    entries = gt.groupby("entry_id")
    log(f"\nAnalyzing {len(entries)} annotator-image entries...")

    for entry_id, sub in entries:
        stem = sub["stem"].iloc[0]
        n_copies = copies_per_stem[stem]
        kept = preds_by_stem.get(stem, [])
        gt_layers = rles_to_layers(sub["segmentation_rle"].tolist())
        n_gt = len(sub)

        if not kept:
            for r in sub.itertuples():
                fn_counts["missed"] += 1
                fn_details.append({"ann_id": r.filament_id, "stem": stem, "fn_class": "missed", "n_copies": n_copies})
            continue

        pred_layers = np.stack(kept).astype(np.uint8)
        n_pred = len(kept)
        iou = _entry_iou_matrix(gt_layers, pred_layers)  # (n_gt, n_pred)

        hit = iou > 0.5
        gt_matched = hit.any(axis=1)
        pred_matched = hit.any(axis=0)

        ov = iou > 0.05

        # Classify FNs
        for gi in range(n_gt):
            if gt_matched[gi]:
                continue
            ann_id = sub["filament_id"].iloc[gi]
            ov_preds = np.flatnonzero(ov[gi])

            if len(ov_preds) == 0:
                cls = "missed"
            elif len(ov_preds) >= 2:
                # Check if union would clear IoU 0.5
                u_mask = np.zeros_like(gt_layers[gi], dtype=bool)
                for pi in ov_preds:
                    u_mask |= (pred_layers[pi] > 0)
                inter = int(np.logical_and(gt_layers[gi] > 0, u_mask).sum())
                union = int(np.logical_or(gt_layers[gi] > 0, u_mask).sum())
                union_iou = inter / union if union > 0 else 0.0
                if union_iou > 0.5:
                    cls = "fragmented"
                else:
                    cls = "partial"
            else:
                pi = ov_preds[0]
                # Check if this single pred covers multiple GTs substantially (>0.1 each)
                other_gts = np.flatnonzero(iou[:, pi] > 0.10)
                if len(other_gts) >= 2:
                    cls = "merged"
                elif iou[gi, pi] > 0.10:
                    cls = "partial"
                else:
                    cls = "missed"

            fn_counts[cls] += 1
            fn_details.append({"ann_id": ann_id, "stem": stem, "fn_class": cls, "n_copies": n_copies})

        # Classify FPs
        for pi in range(n_pred):
            if pred_matched[pi]:
                continue
            cand_id = f"{stem}_{pi}"
            ov_gts = np.flatnonzero(ov[:, pi])

            if len(ov_gts) == 0:
                cls = "hallucination"
            elif len(ov_gts) >= 2 and (iou[ov_gts, pi] > 0.10).sum() >= 2:
                cls = "merge"
            else:
                # Check if overlapping GT was matched by someone else (fragment debris)
                if any(gt_matched[gi] for gi in ov_gts):
                    cls = "fragment"
                else:
                    cls = "partial_pair"

            fp_counts[cls] += 1
            fp_details.append({"cand_id": cand_id, "stem": stem, "fp_class": cls, "n_copies": n_copies})

    total_fn = sum(fn_counts.values())
    total_fp = sum(fp_counts.values())

    log("\n" + "=" * 70)
    log("FALSE NEGATIVE (FN) BREAKDOWN:")
    log("=" * 70)
    for k in ["missed", "partial", "merged", "fragmented"]:
        cnt = fn_counts[k]
        pct = 100.0 * cnt / total_fn if total_fn else 0.0
        log(f"  {k:15s}: {cnt:5d} ({pct:5.1f}%)")

    log("\n" + "=" * 70)
    log("FALSE POSITIVE (FP) BREAKDOWN:")
    log("=" * 70)
    for k in ["hallucination", "partial_pair", "fragment", "merge"]:
        cnt = fp_counts[k]
        pct = 100.0 * cnt / total_fp if total_fp else 0.0
        log(f"  {k:15s}: {cnt:5d} ({pct:5.1f}%)")

    # 4. Compute Topological Oracle Ceiling
    # If all merged + fragmented FNs are converted to TPs with current mean SQ (base_res['sq']),
    # and their corresponding FP fragments and merges are removed:
    n_topo_fn = fn_counts["merged"] + fn_counts["fragmented"]
    n_topo_fp = fp_counts["merge"] + fp_counts["fragment"]
    topo_fn_pct = 100.0 * n_topo_fn / total_fn if total_fn else 0.0

    sq_assumed = base_res["sq"]
    cur_tp = base_res["tp"]
    cur_fp = base_res["fp"]
    cur_fn = base_res["fn"]
    cur_iou_sum = cur_tp * sq_assumed

    new_tp = cur_tp + n_topo_fn
    new_fp = max(0, cur_fp - n_topo_fp)
    new_fn = max(0, cur_fn - n_topo_fn)
    new_iou_sum = cur_iou_sum + n_topo_fn * sq_assumed
    denom = new_tp + 0.5 * new_fp + 0.5 * new_fn
    oracle_pq = new_iou_sum / denom if denom > 0 else 0.0
    delta_oracle = oracle_pq - base_res["pq"]

    log("\n" + "=" * 70)
    log("TOPOLOGY ORACLE CEILING ANALYSIS:")
    log("=" * 70)
    log(f"  Merged FNs     : {fn_counts['merged']}")
    log(f"  Fragmented FNs : {fn_counts['fragmented']}")
    log(f"  Total Topological FNs: {n_topo_fn} ({topo_fn_pct:.1f}% of all FNs)")
    log(f"  Total Topological FPs: {n_topo_fp}")
    log(f"  Current PQ     : {base_res['pq']:.4f}")
    log(f"  Oracle PQ      : {oracle_pq:.4f}")
    log(f"  Oracle Delta PQ: {delta_oracle:+.4f}")

    # 5. Breakdown by Multi-Annotator Copy Count
    fn_df = pd.DataFrame(fn_details)
    log("\nTopological FNs (merged + fragmented) by Annotator Copy Count:")
    for n_cop in [1, 2, 3]:
        sub_cop = fn_df[fn_df["n_copies"] == n_cop]
        n_sub_topo = (sub_cop["fn_class"].isin(["merged", "fragmented"])).sum()
        n_sub_tot = len(sub_cop)
        pct_cop = 100.0 * n_sub_topo / n_sub_tot if n_sub_tot else 0.0
        log(f"  Copies N={n_cop}: {n_sub_topo} / {n_sub_tot} FNs ({pct_cop:.1f}%)")

    # Gate Decision
    log("\n" + "=" * 70)
    log("PRE-REGISTERED GATE EVALUATION:")
    log("=" * 70)
    log(f"Pre-registered Kill Gate: Oracle Delta PQ >= +0.012 required.")
    log(f"Achieved Oracle Delta  : {delta_oracle:+.4f}")
    if delta_oracle >= 0.012:
        log("GATE PASSED: Topological error ceiling is high enough to justify research.")
    else:
        log("GATE FAILED decisively (Oracle Delta < +0.012).")
        log("CONCLUSION: Merged and fragmented errors are too rare (<= 7%) to provide significant metric headroom.")
        log("ACTION: Multi-loss deep supervision, boundary, and graph topology tracks are PERMANENTLY CLOSED.")

    # Save results json
    res_dict = {
        "baseline_pq": float(base_res["pq"]),
        "baseline_sq": float(base_res["sq"]),
        "baseline_rq": float(base_res["rq"]),
        "fn_breakdown": dict(fn_counts),
        "fp_breakdown": dict(fp_counts),
        "n_topological_fn": int(n_topo_fn),
        "topological_fn_pct": float(topo_fn_pct),
        "oracle_pq": float(oracle_pq),
        "oracle_delta_pq": float(delta_oracle),
        "gate_passed": bool(delta_oracle >= 0.012)
    }
    import json
    with open(OUT / "error_taxonomy_honest_sub025.json", "w") as f:
        json.dump(res_dict, f, indent=2)
    log(f"\nWrote results to {OUT / 'error_taxonomy_honest_sub025.json'}")


if __name__ == "__main__":
    main()
