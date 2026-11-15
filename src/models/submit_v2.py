"""Submission with the two-lane fused pool + selector v5 + mask shrink.

Differences from sub #6 (LB 0.37):
  * selector trained on all five folds' fused pools (9610 candidates) instead of
    fold 0's 1658, with the source-aware and cross-lane features
    (cross-fitted pooled PQ 0.4251 vs 0.3954 conf/area baseline)
  * candidate features built by the SAME function used for the training pool, so
    train/apply features cannot drift
  * final masks shrunk by the measured-optimal 5-10% of area (fold-0 +0.0053;
    the visual gallery showed our contours systematically encircling the GT)

Usage:
  python src/models/submit_v2.py --out outputs/submissions/sub007.csv \
      --conf 0.40 --p-min 0.40 --shrink 0.07
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "models"))

from filament import IMG_H, IMG_W
from filament.rle import decode_rle, encode_mask, write_submission
from build_fused_pool import fuse_and_featurise, R_MAX
from mask_shrink import shrink_frac, shrink_true_frac
from semantic_candidates import load_model, predict_prob
from extra_features import EXTRA, extra_features, structure_tensor
from scipy import ndimage

CACHE = ROOT / "outputs" / "oof_cache"
SEM = ROOT / "outputs" / "semantic"
TEST = ROOT / "data" / "MAGFiLO_1.0_Kaggle_2026" / "test" / "test_images"


def log(*a):
    print(*a, flush=True)


# Recipe-diverse semantic ensemble, measured on fold 0 at native 2048 where all
# three members were trained on folds 1-4 (src/eda/inference_size.py):
#
#   prod_1536   0.6969      prod+heavy_768               0.7007
#   heavy_768   0.6980      prod+heavy_1536              0.7012
#   heavy_1536  0.6975      all three                    0.7034  (+0.0054)
#
# Pairwise probability-map correlations are 0.908-0.932, so the diversity is real.
# The ledger's rule is that an ensemble gains from members that make DIFFERENT
# mistakes, and every earlier attempt failed it because the members differed only
# in seed or size (8-model diversity ensemble: 0.4880 vs 0.4879). These differ in
# training recipe and resolution.
#
# The GONG-pretrained model is deliberately absent: adding it as a fourth member
# measured 0.7015, i.e. -0.0019. It is both the weakest single (0.6864) and a
# non-contributor, which is the fair test of the pretraining hypothesis -- not
# "is it better alone" but "does it add anything".
#
# Each member is inferred at the size it was TRAINED at. Running a larger input
# was measured and it hurts uniformly -- prod_1536 -0.0070 at 2048, heavy_1536
# -0.0067 at 2048, heavy_768 -0.0072 at 1024 -- because the filters are fitted to
# a particular apparent filament width and a 1.33x input moves every filament off
# it. The output is resized back to 2048 regardless, so nothing is lost by this.
# the seven A100 arms of ledger 8f-8n, ordered by their leave-one-out
# contribution to the committee (17a): 2048/B2, B5, wide decoder, GN, RDN, SAAF, UPerNet
_ARMS = ["outputs/runs_semantic/sem_b3_f0_2048_cldice_b2/best.pt",
         "outputs/runs_semantic/sem_b5_f0_1536_cldice/best.pt",
         "outputs/runs_semantic/sem_b3_f0_1536_cldice_wdec4832/best.pt",
         "outputs/runs_semantic/sem_b3_f0_1536_cldice_gn8/best.pt",
         "outputs/runs_semantic/sem_b3_f0_1536_cldice_rdn08/best.pt",
         "outputs/runs_semantic/sem_b3_f0_1536_cldice_saaf/best.pt",
         "outputs/runs_semantic/sem_b3_f0_1536_cldice_upernet/best.pt"]
ENSEMBLES = {
    "prod5": [("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 5)
              for k in range(5)],
    "recipe3": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 15)
                 for k in range(5)]
                + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 3),
                   ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 3)]),
    # recipe3 with the heavy_768 slot upgraded to the transductive member: the
    # same heavy-60 recipe trained on ALL 707 images plus the 180 test frames
    # with soft ensemble pseudo-labels (train_mixed.py). Its training data is a
    # strict superset of heavy_768's, and the fold-0 sanity arm held val 0.7024
    # against the 0.7028 baseline, so the swap risks nothing on the train
    # distribution; the transductive benefit is only measurable on the LB.
    "recipe3_mixed": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 15)
                       for k in range(5)]
                      + [("outputs/runs_aug/mixed_all707/best.pt", 1 / 3),
                         ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 3)]),
    # recipe3 with the heavy_768 slot widened from one fold-0 model to the full
    # five-fold heavy set (folds trained 28-29 Aug, val 0.6933-0.7056). Same
    # third-per-model-type weighting; strictly more diverse than recipe3 with
    # everything else equal. Default semantic set for future submissions -- not
    # submission-worthy alone (sub008: the board resolves only >= 0.005).
    # recipe3 with the instance-balanced model added as a FOURTH model type, at
    # the same equal-weight-per-type rule. Its case is diversity measured the way
    # the ledger's one successful diversity attempt was measured: correlation
    # with the production clDice field is 0.9137 over all pixels, inside the
    # 0.908-0.932 band that earned +0.0054, and 0.68 over the pixels either model
    # lights. On the fold-0 harness, averaging it with the production field at
    # equal weight gains +0.0039 with a PAIRED standard error of 0.0012 (t=3.24,
    # better on 15 of 20 splits) and lifts the pool oracle 0.5271 -> 0.5319.
    # Replacing the lane instead of joining it LOSES 0.0083, so the member form
    # is the one that pays. Here it takes a quarter rather than the half it had
    # on the harness, so the deployed effect is smaller than the measured one.
    "recipe4blob": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 20)
                     for k in range(5)]
                    + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 4),
                       ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 4),
                       ("outputs/runs_aug/bloblocal_f0_s1536_w1.0_r40/best.pt", 1 / 4)]),
    # The blob member at HALF the semantic field rather than the quarter
    # recipe4blob gave it. The weight was swept on the fold-0 harness and is not
    # flat: blob weight 0.00 -> 0.4448, 0.30 -> 0.4445, 0.50 -> 0.4488,
    # 0.65 -> 0.4470, 1.00 -> 0.4365, with the pool oracle peaking at the same
    # place (0.5319). A quarter therefore sits in the region measured to gain
    # nothing, which is what recipe4blob shipped in sub012. Paired against the
    # production-only field at this weight the gain is +0.0039 with a paired
    # standard error of 0.0012 (t = 3.24, better on 15 of 20 splits).
    "recipe_blob50": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 30)
                       for k in range(5)]
                      + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 6),
                         ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 6),
                         ("outputs/runs_aug/bloblocal_f0_s1536_w1.0_r40/best.pt", 1 / 2)]),
    # recipe_blob50 with the blob slot widened from ONE fold-0 model to all five
    # folds, at the same total half-weight. This is the arm that separates scope
    # from mechanism. sub014 shipped the single-model form and lost (0.38 against
    # sub015's 0.39), which reads as "the instance-balanced member does not help"
    # -- but a lone fold-0 model carrying half the semantic field is a
    # high-variance thing to average into an otherwise five-fold ensemble, and
    # that is a defect of scope, not of the loss.
    #
    # Measured on the full production pool rather than the fold-0 harness:
    # paired cross-fit over 703 stems and 20 identical splits, selector refit
    # out-of-fold on each pool, threshold policy chosen on the training half
    # only. fused_pool_all 0.4331 -> fused_pool_all_blob5 0.4392, difference
    # +0.0061 with a PAIRED standard error of 0.0005 (t = 11.86, better on 20 of
    # 20 splits, 95% interval [+0.0051, +0.0071]). The single-model form measured
    # +0.0039 +/- 0.0012 on fold 0 alone and did not survive the board; this is
    # five times the sample and a twentieth of the error bar.
    "recipe_blob5_50": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 30)
                         for k in range(5)]
                        + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 6),
                           ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 6)]
                        + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k,
                            1 / 10) for k in range(5)]),
    # recipe_blob5_50 with the two heavy members REMOVED, which makes the
    # submitted field identical to the one the out-of-fold pool is built from:
    # five clDice folds and five blob folds, half the weight each.
    #
    # Two reasons, one measured and one structural.
    #
    # Measured: the board's true ordering puts prod5 (sub007, #8) above recipe3
    # (sub008, #9) and recipe3 above recipe3_mixed (sub009, #11). The heavy
    # members were added on a +0.0054 fold-0 measurement and the board has them
    # slightly negative.
    #
    # Structural, and the reason to take the board seriously here: both heavy
    # models were trained on fold 0 ALONE, so they have seen the validation
    # images of folds 1-4. They cannot appear in an out-of-fold pool at all, and
    # they never have -- fused_pool_all is built from the five clDice folds only.
    # Their contribution was measured once, on one fold, and was never
    # validatable again, while shipping in every submission since sub008. That is
    # a train/deploy mismatch in the semantic field itself: we optimise the
    # selector against one field and submit another. This recipe closes it.
    "recipe_blob5_pure": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 10)
                           for k in range(5)]
                          + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k,
                              1 / 10) for k in range(5)]),
    # recipe_blob5_50 with the blob share raised from 0.50 to 0.65, keeping the
    # 1:2 clDice:heavy split inside the remaining 0.35.
    #
    # The single-model sweep put the optimum at 0.50 and had it falling above
    # (0.65 -> 0.4470 against 0.50's 0.4488), but the board later showed that
    # curve was shaped by the VARIANCE of one fold-0 model: for a single model a
    # quarter beat a half (sub012 over sub013), while for the five-fold ensemble a
    # half is the best field we have shipped. Averaging five folds removes the
    # noise that made more weight costly, so the optimum should move up -- and it
    # does. Paired cross-fit on the full pool, 703 stems, 20 identical splits:
    # 0.4354 -> 0.4363, +0.0009 with a paired standard error of 0.0003 (t = 2.75,
    # better on 14 of 20). Pushing to 0.80 turns it round: -0.0008, 6 of 20.
    #
    # Small, but the two-stage selector was +0.0011 on the same protocol and
    # carried a display step on the board.
    "recipe_blob5_65": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k,
                          0.35 / 3 / 5) for k in range(5)]
                        + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 0.35 / 3),
                           ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 0.35 / 3)]
                        + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k,
                            0.65 / 5) for k in range(5)]),
    "recipe3h5": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 1 / 15)
                   for k in range(5)]
                  + [("outputs/runs_aug/aug_heavy_f%d_768_60ep/best.pt" % k, 1 / 15)
                     for k in range(5)]
                  + [("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 3)]),
    # Ledger 17: the committee. Seven fold-0 arms that each LOST as replacements
    # (8f-8n) averaged with the production model: fold-0 fixed-selector twin,
    # deployment-faithful (label_pool order), 0.4358 -> 0.4451, +0.0093,
    # 95% CI [+0.0026, +0.0161]. Structure = recipe_blob5_pure with the clDice
    # half replaced by mean(clDice five-fold mean, seven arms). The arms held
    # out fold 0 only, so they are legal test-time members. Normally served
    # from the cache composed by compose_test_probs.py (per-member maps from
    # test_member_maps.py); this entry is the GPU fallback and the record.
    "recipe_committee": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 0.5 / 8 / 5)
                          for k in range(5)]
                         + [(w, 0.5 / 8) for w in _ARMS]
                         + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k, 0.5 / 5)
                            for k in range(5)]),
    "recipe_committee4": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 0.5 / 4 / 5)
                           for k in range(5)]
                          + [(w, 0.5 / 4) for w in _ARMS[:3]]
                          + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k, 0.5 / 5)
                             for k in range(5)]),
    "recipe_committee9": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, 0.5 / 9 / 5)
                           for k in range(5)]
                          + [(w, 0.5 / 9) for w in _ARMS
                             + ["outputs/runs_semantic/sem_b3_f0_1536_cldice_silu/best.pt"]]
                          + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k, 0.5 / 5)
                             for k in range(5)]),
    # Ledger 17d: the board put sub027 (pure structure + committee) at #5, below
    # sub025, while the single-variable pair sub019 > sub021 says removing the
    # heavy members costs. So keep sub025's field and change ONE thing: the
    # clDice third becomes the committee third (clDice five-fold mean + 7 arms).
    "recipe_committee_h": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, (1 / 6) / 8 / 5)
                            for k in range(5)]
                           + [(w, (1 / 6) / 8) for w in _ARMS]
                           + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 6),
                              ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 6)]
                           + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k, 0.5 / 5)
                              for k in range(5)]),
    # committee4 (2048/B2, B5, wide decoder + clDice five-fold mean) as the clDice
    # third of sub025's field; heavies and blob unchanged. sub030 candidate.
    "recipe_committee4_h": ([("outputs/runs_semantic/sem_b3_f%d_1536_cldice/best.pt" % k, (1 / 6) / 4 / 5)
                             for k in range(5)]
                            + [(w, (1 / 6) / 4) for w in _ARMS[:3]]
                            + [("outputs/runs_aug/aug_heavy_f0_768_60ep/best.pt", 1 / 6),
                               ("outputs/runs_aug/aug_heavy_f0_1536_60ep/best.pt", 1 / 6)]
                            + [("outputs/runs_aug/bloblocal_f%d_s1536_w1.0_r40/best.pt" % k, 0.5 / 5)
                               for k in range(5)]),
}


class _LazyProbs:
    """Decompress one probability map at a time instead of all 180 at once.

    The cache is uint8, but the caller wants float32 in [0,1] and consumes the
    maps strictly one stem at a time. Materialising the whole dict costs
    180 * 2048^2 * 4 bytes = 3.0 GB resident, which on this 31 GiB box -- shared
    with other work, and already once at 0.1 GB free -- is the difference
    between a submission that builds and one that dies on an allocation.
    Reading from the open npz per access keeps the peak at a single 16 MB map.
    """

    def __init__(self, z, path=None):
        self._z = z
        self.path = path        # so a worker process can open the same file itself

    def __getitem__(self, stem):
        return self._z[stem].astype(np.float32) / 255.0

    def __contains__(self, stem):
        return stem in self._z.files

    def __len__(self):
        return len(self._z.files)

    # __iter__ is NOT optional here. The caller does `sorted(probs)`, and without
    # __iter__ Python falls back to the old sequence protocol -- probs[0],
    # probs[1], ... -- which asks the npz for a member named "0" and dies. The
    # dict this replaced iterated its stems, so this must too.
    def __iter__(self):
        return iter(self._z.files)

    def keys(self):
        return list(self._z.files)

    def items(self):
        return ((s, self[s]) for s in self._z.files)


def semantic_test_probs(which="prod5"):
    """Weighted mean probability map over the ensemble members (flip TTA each).

    recipe3 keeps the three model TYPES at equal weight, matching what was
    measured, so the five production folds share a third between them. Note the
    honest limitation: the submission's five-fold average cannot be validated on
    any fold -- four of those models have seen it -- so the weighting is carried
    over from a measurement made with a single production model.
    """
    cache = SEM / f"test_probs_{which}.npz"
    if cache.exists():
        z = np.load(cache)
        log(f"reusing cached test probability maps [{which}] ({len(z.files)} images)")
        return _LazyProbs(z, cache)
    stems = sorted(p.stem for p in TEST.glob("*.jpeg"))
    acc = {s: np.zeros((IMG_H, IMG_W), np.float32) for s in stems}
    for w, weight in ENSEMBLES[which]:
        model, imgsz, channels = load_model(str(ROOT / w))
        # 2026-09-10, Talha: the organiser's clarification limits FINAL inference to the
        # provided H-alpha frames. A magnetogram-fed model (E162) may only ever be a
        # teacher; nothing that reads outputs/magnetograms may enter a submission.
        if channels == "mag":
            raise SystemExit(f"refusing {w}: channels='mag' reads external magnetograms at "
                             "inference, which the rules do not allow in a submission")
        t0 = time.time()
        for s in stems:
            acc[s] += predict_prob(model, imgsz, s, tta=True, img_dir=TEST,
                                   channels=channels) * weight
        del model
        torch.cuda.empty_cache()
        log(f"  {Path(w).parent.name} @{imgsz} w={weight:.3f} "
            f"({time.time()-t0:.0f}s)")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{s: (v * 255).astype(np.uint8) for s, v in acc.items()})
    # drop the 3.0 GB accumulator before the fusion loop, which needs the room
    acc.clear()
    return _LazyProbs(np.load(cache), cache)


def featurise_one(stem, det_stem, cache_path, sem_thr, carve_tau, shrink, true_frac, need_extra):
    """Fuse and featurise one test image: the selector's feature row per candidate, the
    candidate's mask RLE and the RLE of its shrunk mask. Runs in a worker process when
    --workers > 1, so it opens the probability cache itself and returns only encodings.
    Returns (DataFrame or None, carve drops)."""
    import build_fused_pool as _bfp
    prob = np.load(cache_path)[stem].astype(np.float32) / 255.0
    _bfp._CARVE_DROPS = 0
    rows = fuse_and_featurise(stem, det_stem, prob, sem_specs=((sem_thr, 256),), carve_tau=carve_tau)
    drops = _bfp._CARVE_DROPS
    if not rows:
        return None, drops
    df = pd.DataFrame(rows)
    df["log10_area"] = np.log10(df["area"])
    # Order matters and must match how the selector was trained. In fused_shrink_eval the masks
    # were shrunk FIRST and only `area` / `log10_area` were recomputed from the shrunk mask;
    # every other shape feature (skel_len, width_proxy, eccentricity, ...) still described the
    # un-shrunk mask. Replicate that exactly, or the selector sees a feature vector it was never
    # fit on.
    shrunk = None
    if shrink > 0:
        _sh = shrink_true_frac if true_frac else shrink_frac
        shrunk = [_sh(decode_rle(r).astype(bool), shrink) for r in df["rle"]]
        df["area"] = [int(m.sum()) for m in shrunk]
        df["log10_area"] = np.log10(np.maximum(df["area"], 1))
    if need_extra:
        # same code as the pool side (src/models/extra_features.py), on the SAME object the
        # selector was fitted on: the shrunk mask. Train/deploy parity is enforced by sharing
        # the function, per the ledger's uint8-drift lesson.
        raw_img = cv2.imread(str(TEST / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        tensors = structure_tensor(raw_img)
        masks_for_feats = shrunk if shrunk is not None else [decode_rle(r).astype(bool) for r in df["rle"]]
        ef = pd.DataFrame([extra_features(m, raw_img, tensors) for m in masks_for_feats], index=df.index)
        for c in need_extra:
            df[c] = ef[c]
    df["rle_shrunk"] = [encode_mask(m.astype(np.uint8)) for m in shrunk] if shrunk is not None else list(df["rle"])
    return df, drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    # Defaults are the measured optimum on the shrunk fused pool, rebuilt from
    # the same uint8 probability caches this script reads (cross-fitted PQ
    # 0.4286 vs 0.4240 un-shrunk vs 0.3958 conf/area baseline). --shrink must
    # stay at the value the selector was fit with, since shrinking moves the
    # area features the selector reads.
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--p-min", type=float, default=0.35)
    ap.add_argument("--shrink", type=float, default=0.05)
    ap.add_argument("--true-frac", action="store_true",
                    help="use mask_shrink.shrink_true_frac, whose parameter is the "
                         "area fraction actually removed. shrink_frac's is inert "
                         "below 0.19 -- it deletes a whole tied boundary ring for "
                         "any smaller value -- so the deployed 0.05 removes ~19 percent. "
                         "Swept properly for the first time, the optimum is 0.10 "
                         "(cross-fitted 0.4331 against 0.4286). Must match the "
                         "shrink the selector bundle was fitted with.")
    ap.add_argument("--det-models", default="0n,1n,2n,3n,4n")
    ap.add_argument("--selector", default="outputs/selector/selector_v5_shrunk.joblib")
    ap.add_argument("--semantic", default="prod5", choices=["prod5", "recipe3", "recipe3_mixed", "recipe3h5", "recipe4blob", "recipe_blob50", "recipe_blob5_50", "recipe_blob5_pure", "recipe_blob5_65", "recipe_committee", "recipe_committee4", "recipe_committee9", "recipe_committee_h", "recipe_committee4_h"],
                    help="semantic ensemble; recipe3 adds the two heavy-recipe "
                         "models measured at +0.0054 on fold 0")
    ap.add_argument("--sem-thr", type=float, default=0.3,
                    help="semantic extraction threshold at fusion (ledger 19: 0.2 with the selector "
                         "refit on the 0.2 pool pays +0.0050 on five folds; 0.3 is sub025)")
    ap.add_argument("--fill-holes", action="store_true",
                    help="close interior holes in every SELECTED mask before writing. The "
                         "annotation protocol (MAGFiLO paper: 'a single connected piece with "
                         "no holes') makes this a conformity fix, and it was measured on the "
                         "deployed fold-0 selection: +0.0001 PQ [+0.0000, +0.0001] over 750 "
                         "masks, 14 of which had holes (ledger 15 section 27n). Free, so it "
                         "rides the final submission; never worth a submission of its own.")
    ap.add_argument("--carve-tau", type=float, default=0.0,
                    help="minimum area retention ratio (carved_area / orig_area) during "
                         "greedy de-overlap. Candidates losing more than (1 - tau) fraction "
                         "of their area to higher-confidence candidates are dropped. "
                         "0.70 measured +0.0051 PQ across 5 folds.")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes for the per-image fusion and featurisation (the CPU-bound "
                         "step); 1 keeps the serial loop. The output does not depend on it.")
    ap.add_argument("--dump-pool", default="",
                    help="also write the featurised test candidate pool (every column the "
                         "selector reads, conf, and the shrunk mask as RLE) to this parquet, so "
                         "selector-side variants can be applied by submit_from_pool.py without "
                         "repeating fusion and featurisation")
    a = ap.parse_args()
    dumped = []

    bundle = joblib.load(ROOT / a.selector)
    sel, feats = bundle["model"], bundle["features"]
    iou_model = bundle.get("iou_model")   # present only for two-stage bundles
    log(f"selector: {len(feats)} features")

    dfs = []
    for mid in a.det_models.split(","):
        d = pd.read_parquet(CACHE / f"cross_m{mid}_test.parquet")
        dfs.append(d)
    det = pd.concat(dfs, ignore_index=True)
    det = det[(det["conf"] >= 0.05) & (det["r_norm"] <= R_MAX)]
    log(f"detector candidates: {len(det)} over {det['stem'].nunique()} images")

    probs = semantic_test_probs(a.semantic)
    stems = sorted(probs)
    need_extra = [f for f in feats if f in EXTRA]
    job_args = dict(cache_path=str(probs.path), sem_thr=a.sem_thr, carve_tau=a.carve_tau,
                    shrink=a.shrink, true_frac=a.true_frac, need_extra=need_extra)

    preds, t0, carve_drops = {}, time.time(), 0
    if a.workers > 1:
        # Images are independent, so the CPU-bound fusion and featurisation is spread over
        # processes; results come back in stem order, so the CSV is the same as the serial one
        # (verified byte for byte by outputs/eda/_verify_lossless_fusion.py).
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=a.workers, backend="loky")(
            delayed(featurise_one)(stem, det[det["stem"] == stem], **job_args) for stem in stems)
    else:
        results = (featurise_one(stem, det[det["stem"] == stem], **job_args) for stem in stems)

    for n, (stem, (df, drops)) in enumerate(zip(stems, results)):
        carve_drops += drops
        if df is None:
            preds[stem] = []
            continue
        X = df[feats].to_numpy(np.float64)
        p = sel.predict_proba(X)[:, 1]
        if iou_model is not None:
            # two-stage bundle: the threshold it was tuned with lives on
            # p(match) x E[iou_sum|match], so the same product has to be formed
            # here or the keep gate means something different at deploy than it
            # did in the harness
            p = p * np.clip(iou_model.predict(X), 0.0, None)
            # the TRAINING quantiles, not this pool's: the keep threshold was
            # tuned as a quantile of the training score distribution and has to
            # keep that meaning here
            ref = bundle["two_stage_cdf"]
            p = np.searchsorted(ref, p, side="right") / len(ref)
        if a.dump_pool:
            keep_cols = df.drop(columns=["rle"]).copy()
            keep_cols["stem"] = stem
            keep_cols["p_bundle"] = p
            dumped.append(keep_cols)
        sel_keep = ((df["conf"] >= a.conf) & (p >= a.p_min)).to_numpy()
        masks = []
        for i in np.flatnonzero(sel_keep):
            m = decode_rle(df["rle_shrunk"].iloc[i]).astype(bool)
            if a.fill_holes:
                m = ndimage.binary_fill_holes(m)
            if m.sum() >= 64:
                masks.append(m.astype(np.uint8))
        preds[stem] = masks
        if (n + 1) % 40 == 0:
            log(f"  {n+1}/{len(probs)} images ({time.time()-t0:.0f}s)")

    if a.dump_pool:
        pool = pd.concat(dumped, ignore_index=True)
        pool.to_parquet(ROOT / a.dump_pool, index=False)
        log(f"dumped test pool: {len(pool)} candidates over {pool['stem'].nunique()} images -> {a.dump_pool}")
    n_inst = sum(len(v) for v in preds.values())
    n_empty = sum(1 for v in preds.values() if not v)
    log(f"test: {n_inst} instances over {len(preds)-n_empty} images ({n_empty} empty)")
    if a.carve_tau > 0.0:
        log(f"carve rule (tau {a.carve_tau}) rejected {carve_drops} cluster masks at fusion on the test pool")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = write_submission({k: v for k, v in preds.items() if v}, out)
    log(f"wrote {len(df)} rows -> {a.out}")


if __name__ == "__main__":
    main()
