"""Generate Appendix D: every script indexed by the question it answers.

Each script in this project opens with a docstring whose first sentence states the question it
exists to answer. Generating the index from those sentences means the appendix cannot drift from
the code -- a reviewer can check any row by opening the file.

A docstring is written for the person working on the code, and the appendix is read by someone
who is not. The first sentence therefore passes through a fixed, mechanical cleanup: it is taken
whole (never cut mid-sentence), the internal experiment identifier that opens many docstrings
("E154b --") and parenthetical pointers into working notes ("(docs/32 §5.2)", "(ledger 19a)") are
dropped, capitals used for emphasis are lowered while acronyms are kept, and ASCII arrows and
dashes become typographic ones. The few rows whose first sentence still does not read on its own
-- a pointer to an internal note, a personal name, or a sentence too long for a table -- are
reworded in OVERRIDES below, in the open, so the difference from the code is itself reviewable.

Usage: python src/eda/gen_script_index.py
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "report" / "06_appendix_d_scripts.md"
DIRS = [("src/data", "cache builders"),
        ("src/models", "training, inference and submission"),
        ("src/eda", "measurements")]

# acronyms and proper names written in capitals; every other all-capital word is emphasis
ACRONYMS = {"AIA", "AUC", "CNN", "CPU", "CRF", "CSV", "CV", "CVPR", "EDA", "FITS", "FN", "FP",
            "GONG", "GPU", "GT", "HMI", "ID", "IOU", "JPEG", "JSON", "LB", "LR", "M2F", "MIM",
            "NMS", "OOF", "PIL", "PQ", "RLE", "ROI", "RQ", "SAAF", "SDO", "SNR", "SQ", "TP", "TTA",
            "UNET", "VRAM", "YOLO"}

LEAD_TAG = re.compile(
    r"^(?:"
    r"E\d{3}[a-z]?(?:-lite)?(?:\s*/\s*E\d{3}[a-z]?(?:-lite)?)*(?:\s+v\d)?(?:\s+§\w+)?"
    r"|SI-S\d|WA-\d+(?:\s+(?:pilot|prerequisite))?|P\d(?:\+P\d)*|D1b precondition"
    r"|Experiment [A-Z]|Rubric arm [A-Z]"
    r")\s*(?:\([^)]*\))?\s*(?:--|:)\s*")
INTERNAL_PAREN = re.compile(
    r"\s*\((?:[^()]*?(?:docs/|ledger|chatgpt|saf_bilim|_review|S00\d|E\d{3}|Talha|§\d)[^()]*)\)")

OVERRIDES = {
    "candidate_bank_area_rule.py": "Area-conditioned representation rule on the candidate bank: the one form "
                                   "of the hypothesis that small masks should shrink and large ones grow that "
                                   "can add pixels to a mask in this pipeline.",
    "category_analysis.py": "Is the Unidentifiable chirality class noise to discard, or the largest "
                            "untapped population?",
    "champion_stress.py": "The final submission's PQ broken down by slice, with the exact scorer.",
    "compose_test_probs.py": "Compose a test-time semantic field from per-member probability maps on the "
                             "CPU, in the cache location the submission script reads.",
    "contour_level_null.py": "Is the contour-level ceiling real, or is it the maximum of eight noisy draws?",
    "decoder_norm.py": "A GroupNorm decoder for the U-Net, built as a BatchNorm decoder whose normalisation "
                       "layers are replaced after construction.",
    "detector_limb_recall.py": "Detector recall at the limb, the precondition for a limb-specific detector change.",
    "board_ordering.py": "Read the leaderboard's unrounded ordering of our own submissions.",
    "build_semantic_cache.py": "Pre-render the semantic lane's inputs so that training does not recompute them.",
    "deoverlap_order.py": "Does de-overlapping after selection, rather than before, raise the ceiling?",
    "eval_nodeov.py": "Score the reordered pipeline: select first, make candidates disjoint afterwards.",
    "fused_shrink_eval.py": "Are the gains of mask shrinking and of the selector additive?",
    "fusion_loss.py": "Which fusion step discards reachable ground truth?",
    "magnetogram_probe.py": "Feasibility of co-temporal HMI magnetograms as a second modality.",
    "mask_shrink.py": "Are our masks systematically too large? PQ under morphological shrinking.",
    "overlap_density.py": "Does candidate density explain why the leaderboard favours more shrink than "
                          "cross-fitting does?",
    "perfect_mask_ceiling.py": "The PQ ceiling if the exact annotator mask were drawn at every detection.",
    "phys_channel_regate.py": "Re-test the physical-channel negative result against the instance-level gate.",
    "train_test_area_shift.py": "Why the leaderboard favours more shrink than cross-fitting does.",
    "e098b_carve_reeval.py": "Why the carve-retention filter improved the local score and lowered the "
                             "leaderboard score.",
    "e098d_pool_twin.py": "A local replicate of two leaderboard submissions that differ only in the "
                          "carve-retention rule.",
    "e098e_twin_fixed_selector.py": "The same local replicate, with the selector held fixed.",
    "e115b_oof_threshold.py": "The selector-threshold sweep (p 0.55 → 0.60) repeated with out-of-fold "
                              "selector scores.",
    "e115c_threshold_ci.py": "A paired confidence interval on the out-of-fold p 0.55 vs 0.60 contrast.",
    "eval_m2f.py": "Fold-0 PQ for the Mask2Former lane, the gate that decides whether the lane is kept.",
    "latitude_structure.py": "Latitude structure of true positives, false negatives and false positives.",
    "limb_failure_anatomy.py": "Where does the limb fail: proposal, selection or boundary?",
    "opening_point_hdjojo.py": "Is the advantage of the public 0.40 recipe in the model or in the operating point?",
    "pretrain_crack_yolo11m.py": "Stage 1 of crack-segmentation pretraining on the selected backbone: "
                                 "yolo11m on crack-seg.",
    "resume_crack_pre.py": "Resume the crack-segmentation pretraining from its last checkpoint.",
    "saaf.py": "SAAF-lite: re-weight the upsampled decoder map before it is concatenated with the skip "
               "connection.",
    "scse_gate_probe.py": "What does the production decoder's scSE attention do with the upsampled branch "
                          "versus the skip branch on thin filaments?",
    "selector.py": "Component-level candidate selector on the fold-0 out-of-fold candidates of the "
                   "baseline detector.",
    "selector_context_features.py": "Let each candidate see the others: the simplest form of attention in "
                                    "the selector.",
    "semantic_ensemble_gate.py": "Does combining semantic ensemble members pay, measured on existing "
                                 "probability maps?",
    "temporal_pair_ceiling.py": "Does a second H-alpha frame close in time carry information about the "
                                "first frame's annotation?",
    "tp_fp_fn_map.py": "Map of true positives, false negatives and false positives in disc coordinates "
                       "for the production policy on the out-of-fold pool.",
    "train_tiled.py": "Native-resolution tiled semantic lane.",
    "train_yolov8x.py": "yolov8x-seg at 1792 px, differing from the baseline detector only in the backbone.",
    "what_would_055_take.py": "What would have to be true for a PQ of 0.53–0.55?",
    # 2026-09-24: docstrings of the regime and measurement-instrument scripts are written as
    # lab notes (capitalised emphasis, "the board", questions to ourselves); the index gets
    # the same content in the report's register.
    "fold_design_audit.py": "Whether the fold partition itself is a source of the measured differences "
                            "between variants, read on five partitions of the same images.",
    "hf_oracle_ceiling.py": "Selection ceiling by quartile of disc high-frequency energy, under the "
                            "metric's break-even rule with realised values.",
    "hf_oracle_transfer.py": "Whether that selection ceiling transfers from one annotator to another.",
    "mismatch_cost.py": "The cost, on labelled data, of the one-detector against five-detector mismatch "
                        "between validation and deployment.",
    "quality_features_test.py": "Image-quality and frame-relative feature blocks read through the "
                                "deployed selector protocol.",
    "regime_relative_features.py": "Each candidate's evidence expressed relative to its own frame "
                                   "(percentile and z-score), as selector features.",
    "relational_selector_test.py": "The neighbourhood-geometry features read through the selector over "
                                   "five folds.",
    "selector_committee.py": "Whether averaging several sub-floor selector variants yields a gain that "
                             "none of them has alone.",
    "test_carve_filter.py": "Whether discarding heavily carved candidates after greedy de-overlap "
                            "removes false-positive fragments.",
    "threshold_pool_twin.py": "The semantic extraction threshold below 0.20, each pool read with its "
                              "own refitted selector.",
    "transfer_instrument.py": "Covariate-shift reweighting of local validation towards the test set, and "
                              "why it is not possible here.",
    # 2026-09-25: the regime-gate, conditioning and final-pair scripts (§3.6, §4.7).
    "final_pair_selection.py": "Which submission should take the second final slot, measured by what it "
                               "adds to the first on test-sized subsets.",
    "film_regime_check.py": "Where the gain of the conditioned network lands, by quartile of disc "
                            "high-frequency energy.",
    "hf_film.py": "Conditions every encoder skip on the frame's disc high-frequency energy (FiLM).",
    "regime_gate.py": "A keep threshold set by the frame's disc high-frequency energy, chosen by nested "
                      "cross-validation.",
}


def first_sentence(path):
    txt = path.read_text(encoding="utf-8")[:6000]
    m = re.match(r'\s*"""(.+?)"""', txt, re.S)
    if not m:
        return ""
    para = m.group(1).strip().split("\n\n")[0]
    para = " ".join(ln.strip() for ln in para.splitlines())
    for _ in range(2):                                   # "E166 -- WA-05 prerequisite: ..."
        para = LEAD_TAG.sub("", para)
    para = INTERNAL_PAREN.sub("", para)
    # the first sentence, whole: a terminator followed by a capital, or the end of the paragraph
    s = re.split(r"(?<=[.?!])\s+(?=[A-Z\"'])", para, maxsplit=1)[0].strip()
    s = s.replace(" -- ", " — ").replace("--", "—").replace("->", "→").replace("^2", "²")
    s = re.sub(r"(\d)x(\d)", r"\1×\2", s)
    s = re.sub(r"(?<=[\w)]) x (?=[\w(])", " × ", s)
    s = re.sub(r"(?<=\d)-(?=\d)", "–", s)                 # numeric ranges take an en dash
    s = re.sub(r"\bhonest\s+", "", s, flags=re.I)
    s = re.sub(r"\borganiz", "organis", s)

    def lower(w):
        word = w.group(0)
        parts = re.split(r"(-)", word)
        return "".join(p if (p == "-" or p.upper() in ACRONYMS or not p.isalpha()) else p.lower()
                       for p in parts)

    s = re.sub(r"\b[A-Z][A-Z]{2,}(?:-[A-Z]+)*\b", lower, s)
    # capitalise the opening word unless it is an identifier (yolov8x-seg, smp-compatible, oof_cache)
    if s and s[0].islower() and not re.match(r"^(?:yolo|smp\b|smp-|\S*[_.]\S*\s)", s):
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".?!":
        s += "."
    return s


def main():
    lines = ["# Appendix D — Script index", "",
             "*Every script opens with a docstring stating the question it answers. Each row below",
             "is generated from the first sentence of that docstring, so the index cannot drift from",
             "the code. For a reader outside the project, internal experiment identifiers and",
             "pointers into working notes are omitted, and a small number of rows are reworded; the",
             "reworded rows are listed openly in `src/eda/gen_script_index.py`.*", ""]
    total = 0
    for k, (d, label) in enumerate(DIRS, 1):
        paths = sorted((ROOT / d).glob("*.py"))
        rows = [(p, OVERRIDES.get(p.name) or first_sentence(p)) for p in paths]
        rows = [(p, q) for p, q in rows if q]
        if not rows:
            continue
        lines += [f"## D.{k} `{d}`: {label}", "", "::: index",
                  "| Script | Question it answers |", "|---|---|"]
        for p, q in rows:
            lines.append(f"| `{p.name}` | {q.replace('|', '/')} |")
        lines += [":::", ""]
        total += len(rows)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{total} scripts indexed -> {OUT}")


if __name__ == "__main__":
    main()
