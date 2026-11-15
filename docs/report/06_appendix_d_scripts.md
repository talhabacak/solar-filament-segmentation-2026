# Appendix D — Script index

*Every script opens with a docstring stating the question it answers. Each row below
is generated from the first sentence of that docstring, so the index cannot drift from
the code. For a reader outside the project, internal experiment identifiers and
pointers into working notes are omitted, and a small number of rows are reworded; the
reworded rows are listed openly in `src/eda/gen_script_index.py`.*

## D.1 `src/data`: cache builders

::: index
| Script | Question it answers |
|---|---|
| `build_label_cache.py` | Per-entry instance label maps, the target an instance-balanced loss needs. |
| `build_magnetic_cache.py` | The magnetogram as a training input: a cache aligned with the semantic cache. |
| `build_semantic_cache.py` | Pre-render the semantic lane's inputs so that training does not recompute them. |
| `build_spine_cache.py` | Rasterise the manually-annotated spines into a per-entry cache. |
| `build_test_pseudo.py` | Soft pseudo-labels for the 180 test images, from our own ensemble — the transductive lane. |
| `build_tile_cache.py` | Native-resolution cache for the tiled lane: 2048² images, masks, foreground anchors. |
| `build_tube_cache.py` | Tubed-skeleton targets for Skeleton Recall Loss, from the human spine. |
| `export_yolo.py` | Export competition data to Ultralytics YOLO-seg format (fold-aware). |
| `export_yolo_allcopies.py` | Export YOLO dataset with all annotator copies as separate samples (1154 rows). |
| `export_yolo_consensus.py` | Export YOLO-seg labels with consensus targets from cross-annotator clusters. |
| `export_yolo_crop.py` | Export disk-ROI-cropped YOLO dataset: constant crop [112:1936] → 1824×1824. |
| `fetch_gong_unlabeled.py` | Pilot download of unlabelled GONG H-alpha frames, with the safety checks first. |
| `fetch_magnetograms.py` | Fetch a co-temporal HMI magnetogram for every image, train and test. |
| `register_magnetograms.py` | Put the magnetogram into the H-alpha frame, and prove it landed there. |
| `sample_gong_archive.py` | Stratified, time-separated sample of unlabelled GONG H-alpha frames. |
:::

## D.2 `src/models`: training, inference and submission

::: index
| Script | Question it answers |
|---|---|
| `apply_seed_refiner.py` | Run the seed refiner on detector boxes and score the lane it produces. |
| `augment.py` | Augmentation levels for the fine-tune stage, and why each one is here. |
| `blend_semantic.py` | Blend two cached semantic probability fields into one, per fold. |
| `blurpool.py` | Anti-aliased downsampling for the production encoder. |
| `build_detbox_crops.py` | Seed crops built from real detector boxes, not jittered GT boxes. |
| `build_fused_pool.py` | Build the fused (detector + semantic) candidate pool for every fold, with cross-lane agreement features — the input to selector v5. |
| `build_fusion_crops.py` | Crops for a refiner that sees both lanes, not just their fused result. |
| `build_refine_data.py` | Crops for the mask refiner, at native 2048 resolution. |
| `build_seed_crops.py` | Crops for a seed-refiner: segment from a box, not from an existing mask. |
| `cache_m2f_preds.py` | Cache M2F fold candidates in the oof_cache schema, so the fusion machinery can consume them as a third source without any schema surgery. |
| `cache_tiled_preds.py` | Cache the native-resolution tiled lane's instances in the oof_cache schema. |
| `cluster_ranker.py` | Pick the best member of a cluster instead of always emitting the consensus. |
| `compose_test_probs.py` | Compose a test-time semantic field from per-member probability maps on the CPU, in the cache location the submission script reads. |
| `coord_att.py` | Coordinate Attention (Hou, Zhou, Feng, CVPR 2021) as a drop-in for smp's scSE. |
| `cross_cache.py` | Cross-prediction cache: every fold model predicts all train + test images. |
| `decoder_norm.py` | A GroupNorm decoder for the U-Net, built as a BatchNorm decoder whose normalisation layers are replaced after construction. |
| `elastic_aug.py` | Elastic deformation, the one augmentation family this project never tried. |
| `ensemble.py` | 5-fold instance ensemble: cross-model mask clustering + consensus fusion. |
| `eval_m2f.py` | Fold-0 PQ for the Mask2Former lane, the gate that decides whether the lane is kept. |
| `eval_nodeov.py` | Score the reordered pipeline: select first, make candidates disjoint afterwards. |
| `eval_seldeov.py` | Move de-overlap from pool build time to selection time. |
| `eval_tiled.py` | The gate: does the tiled lane beat our own oracle on the real metric? |
| `extra_features.py` | The photometry + orientation feature family, as one function. |
| `finetune_aug.py` | Fine-tune augmentation study: does a 707-image dataset want more augmentation? |
| `fit_selector_bagged.py` | Fit the deployed two-stage selector with image-bootstrap bagging, as a deployable bundle. |
| `fuse_semantic.py` | Fuse detector + semantic candidate pools and measure the real PQ gain. |
| `fused_shrink_eval.py` | Are the gains of mask shrinking and of the selector additive? |
| `hf_film.py` | Conditions every encoder skip on the frame's disc high-frequency energy (FiLM). |
| `limb_aug.py` | Centre-to-limb augmentation by rotating the solar sphere. |
| `magnetogram_features.py` | Polarity inversion lines from the registered magnetogram, and per-candidate features from them. |
| `mask_shrink.py` | Are our masks systematically too large? PQ under morphological shrinking. |
| `maskops.py` | Bounding-box-local versions of the full-frame mask operations the fusion step repeats. |
| `nms_sweep.py` | NMS-threshold ablation (inference-only axis, never tested before). |
| `otsu_snap.py` | Intensity-guided mask snapping: push near-miss predictions over the IoU 0.5 line. |
| `patch_selector.py` | A selector that looks at the image. |
| `pixel_stacker.py` | A per-pixel gradient-boosted stacker over the two lanes. |
| `pooled_oof.py` | Pooled 5-fold OOF evaluation + selector v2 (all-folds training data). |
| `pretrain_crack_yolo11m.py` | Stage 1 of crack-segmentation pretraining on the selected backbone: yolo11m on crack-seg. |
| `pretrain_gong.py` | Stage 1: pretrain the encoder on pseudo-labelled GONG frames. |
| `pretrain_mim.py` | Masked-image pretraining of the segmentation U-Net on H-alpha. |
| `pseudo_label_gong.py` | Teacher pass: soft pseudo-labels for the unlabelled GONG frames. |
| `raw_iou_target.py` | Regress a smooth IoU, not the discontinuous value the metric pays. |
| `rebuild_fusion_targets.py` | Fix the refiner's target: the consensus mask, not one annotator's copy. |
| `resume_crack_pre.py` | Resume the crack-segmentation pretraining from its last checkpoint. |
| `saaf.py` | SAAF-lite: re-weight the upsampled decoder map before it is concatenated with the skip connection. |
| `selector.py` | Component-level candidate selector on the fold-0 out-of-fold candidates of the baseline detector. |
| `selector_v3.py` | Selector v3 target ablation on the pooled 5-fold candidates (CPU only). |
| `selector_v4.py` | Selector v4: add image-appearance features (physics the model never saw). |
| `selector_v5.py` | Selector v5: does cross-lane agreement close part of the +0.088 selection gap? |
| `selector_v6.py` | Selector v6: predict expected PQ value, not "does it match anything". |
| `semantic_candidates.py` | Turn the semantic model into instance candidates and measure what it adds. |
| `stage_ctx.py` | A context module on the stage-4 skip, aimed at the faint population. |
| `submit_from_pool.py` | Write a submission from a dumped test candidate pool, so selector-side variants cost seconds. |
| `submit_selector.py` | Test submission with the verified component-selector policy. |
| `submit_twolane.py` | Two-lane test submission: 5 detector folds + 5 semantic folds, fused. |
| `submit_v2.py` | Submission with the two-lane fused pool + selector v5 + mask shrink. |
| `sweep_oof.py` | OOF postprocess calibration: cache low-conf predictions once, sweep thresholds on PQ. |
| `swin_satlas_encoder.py` | smp-compatible encoder around TorchGeo's Satlas Sentinel-1 Swin-V2. |
| `test_member_maps.py` | Per-member test probability maps, one npz per model, so any ensemble recipe can be composed afterwards on a CPU without touching a GPU again. |
| `tiling.py` | Tiling geometry and blended mosaic reassembly for the native-resolution lane. |
| `train_blobloss.py` | Instance-balanced (blob) loss, aimed at the one deficit still open. |
| `train_blobloss_local.py` | Instance-balanced loss with a local background, fixing the measured failure. |
| `train_fusion_refiner.py` | The two-lane refiner: learn the annotator's mask from both lanes' evidence. |
| `train_m2f.py` | The last untried structural family: query-based instance segmentation. |
| `train_mask_scorer.py` | Look at the crop to judge a candidate, instead of describing it with numbers. |
| `train_mixed.py` | Transductive self-training: heavy-60 recipe on real GT + soft test pseudo-labels. |
| `train_refiner.py` | Refine a candidate mask against the consensus, at native resolution. |
| `train_seed_refiner.py` | Segment a filament from a box, in a scale-normalised crop. |
| `train_semantic.py` | Semantic segmentation lane — a complementary proposal source for the detector. |
| `train_semantic_arch.py` | Architecture / input-band experiments for the semantic lane. |
| `train_skelrecall.py` | Skeleton Recall Loss on the main mask output, with the human spine as the skeleton. |
| `train_spine_aux.py` | Spine-auxiliary training: the winning recipe plus a centreline head. |
| `train_tiled.py` | Native-resolution tiled semantic lane. |
| `train_yolo.py` | Single-stage YOLO-seg baseline training (fold-aware). |
| `train_yolov8x.py` | yolov8x-seg at 1792 px, differing from the baseline detector only in the backbone. |
| `tta_fuse.py` | TTA with consensus fusion (not naive pooling), measured on one fold. |
| `validate_submission.py` | Pre-flight checks on a submission CSV, before a slot is spent on it. |
| `vram_probe.py` | Which SwinV2+UPerNet configuration actually fits in 16 GB? |
| `yolo_infer.py` | YOLO-seg inference → pixel-disjoint instance masks → submission CSV / OOF scoring. |
:::

## D.3 `src/eda`: measurements

::: index
| Script | Question it answers |
|---|---|
| `agreement_boundary.py` | Is our inclusion policy at the annotator-agreement optimum? |
| `annotator_ceiling.py` | Measure the PQ ceiling imposed by annotator disagreement. |
| `annotator_ceiling_part2.py` | Ceiling part 2: union & strict consensus variants (lean memory), merge into JSON. |
| `annotator_count_sensitivity.py` | Does the number of annotators on an image move our optimal operating point? |
| `annotator_station_breakdown.py` | Where does our model do badly — which annotators, which stations? |
| `annotator_structure.py` | Annotator decision-structure mining for MAGFiLO Kaggle 2026. |
| `aug_calibration.py` | Are the augmentation ranges calibrated to the data's own variation? |
| `aug_gallery.py` | Look at what each augmentation level actually does to a filament. |
| `aug_onthefly.py` | Should augmentation run live at native 2048, or on the 768 cache as it does now? |
| `aug_results.py` | Read the augmentation study's trajectories, not just its final numbers. |
| `aug_sanity.py` | Does the augmentation preserve image/mask alignment, and what does it cost? |
| `band_cascade.py` | A second-stage selector for the marginal band (a cascade). |
| `band_failure_anatomy.py` | Disc and limb as two separate problems: where exactly does each lose? |
| `board_ordering.py` | Read the leaderboard's unrounded ordering of our own submissions. |
| `build_cand_rle_cache.py` | Rebuild the de-overlapped candidate RLEs for the pooled OOF candidate set. |
| `build_gt_cache.py` | Build the full GT RLE cache and smoke-test the scorer (PQ=1 self-score). |
| `build_manifest.py` | Build the canonical dataset manifest for the MAGFiLO Kaggle 2026 competition data. |
| `candidate_bank_area_rule.py` | Area-conditioned representation rule on the candidate bank: the one form of the hypothesis that small masks should shrink and large ones grow that can add pixels to a mask in this pipeline. |
| `candidate_bank_capture.py` | How much of the legal-oracle gain is capturable without GT? |
| `candidate_bank_decompose.py` | Decompose the legal-oracle ceiling into its four ingredients, each measured by realised PQ after the legal (non-overlap) pass. |
| `candidate_bank_oracle.py` | Immutable candidate bank and the legal best-variant oracle. |
| `candidate_bank_ranker.py` | Asymmetric, cluster-relative variant ranker on the candidate bank. |
| `candidate_bank_rescue.py` | Rescue ranker on the candidate bank. |
| `candidate_bank_ring_features.py` | Boundary-band probability profile of every candidate-bank variant. |
| `candidate_thresholds.py` | How much reachable GT are we leaving on the table at candidate generation? |
| `category_analysis.py` | Is the Unidentifiable chirality class noise to discard, or the largest untapped population? |
| `champion_stress.py` | The final submission's PQ broken down by slice, with the exact scorer. |
| `classical_baseline.py` | Model-free floor: how far does plain darkness thresholding get on PQ? |
| `cluster_representative.py` | Is the consensus mask the right representative of a cluster? |
| `combined_feature_families.py` | The two feature families that pointed the right way, together. |
| `committee_pool_twin.py` | Committee pool twin — does the fold-0 committee map pay at selector PQ? |
| `committee_refit_twin.py` | Selector refit on committee pools — the last committee hypothesis. |
| `committee_scaling_law.py` | What actually pays in a committee: how many members, or how different? |
| `construction_loss_split.py` | Which construction step destroys the recoverable rows — fusion or de-overlap? |
| `contour_level_centered.py` | The level predictor with the between-candidate variance removed. |
| `contour_level_null.py` | Is the contour-level ceiling real, or is it the maximum of eight noisy draws? |
| `contour_level_oracle.py` | How much is left in the contour level, one per candidate? |
| `contour_level_predictor.py` | Learn the per-candidate contour level, and see how much of the ceiling it takes. |
| `contrast_boundary_oracle.py` | Relative-contrast boundary oracle. |
| `coverage_geometry.py` | Where is the missing coverage — at the ends, or along the flanks? |
| `crowding_mechanism.py` | Why do we miss the unambiguous copies? |
| `deoverlap_order.py` | Does de-overlapping after selection, rather than before, raise the ceiling? |
| `det_conf_band_recal.py` | Band-wise recalibration of detector confidence for the selector. |
| `det_prob_fusion.py` | Fuse the detector's soft evidence into the semantic probability map. |
| `detector_limb_recall.py` | Detector recall at the limb, the precondition for a limb-specific detector change. |
| `disk_geometry_audit.py` | The solar disk is not where the pipeline thinks it is. |
| `disk_refit.py` | Trusted per-image solar disk geometry via full-resolution limb fitting. |
| `e098b_carve_reeval.py` | Why the carve-retention filter improved the local score and lowered the leaderboard score. |
| `e098c_carve_on_production.py` | Does the carve-retention rule have anything left to remove once the selector threshold has run? |
| `e098d_pool_twin.py` | A local replicate of two leaderboard submissions that differ only in the carve-retention rule. |
| `e098e_twin_fixed_selector.py` | The same local replicate, with the selector held fixed. |
| `e115b_oof_threshold.py` | The selector-threshold sweep (p 0.55 → 0.60) repeated with out-of-fold selector scores. |
| `e115c_threshold_ci.py` | A paired confidence interval on the out-of-fold p 0.55 vs 0.60 contrast. |
| `e215_conditional_unshrink.py` | Is the uniform one-pixel ring erosion wrong for thin candidates? |
| `e215b_shape_band_fragments.py` | Are shape-band misses fragments of a filament we did find? |
| `e216_inbox_closer_ceiling.py` | What is the most a gap closer working inside a kept ROI's bounding box could recover? |
| `e216b_inbox_closer_transfer.py` | Does the in-box closer's ceiling transfer across annotators? |
| `e217_yolo_soft_masks.py` | Cache the detector's soft mask probabilities out of fold (they were never kept). |
| `e217b_soft_extension.py` | Does the detector's soft field carry the length our masks miss, and does using it pay? |
| `e218_nms_tighter.py` | Detector NMS tighter than 0.5 (0.3, 0.4) in today's fused pool. |
| `nms_mismatch_f0.py` | Does the test caches' NMS 0.05, against 0.5 out of fold, cost PQ through the selector? (fold 0) |
| `e219_test_generalisation.py` | What does the unlabelled test set let us predict about our private score? |
| `e220_fine_structure.py` | Fine-scale structure: how much of the annotated lateral structure (barbs, side branches) do our masks recover, compared with a second annotator drawing the same filament? |
| `e220b_barb_figure.py` | Figure for report §5.4: barbs of annotated filaments and our masks. |
| `e221_reference_systems.py` | Reference systems on the same protocol: what does each stage of the pipeline add? |
| `e223_axial_continuation.py` | Can any evidence we have tell where a filament continues beyond our mask's end? |
| `e224_spine_curvature.py` | How far can a filament's continuation be predicted from its own shape? |
| `ensemble_heavy.py` | Does the heavy-recipe model add anything to the production semantic ensemble? |
| `ensemble_weight_opt.py` | The one knob never optimised out of fold: the blob/clDice balance of the field. |
| `error_atlas.py` | The error atlas: which of our misses are our fault, and what do they look like? |
| `error_profile_fn.py` | Profile the false negatives of the current operating point. |
| `error_profile_fp.py` | Error profile of the pooled-OOF operating point: FP anatomy + strength/weakness scorecard. |
| `error_taxonomy.py` | OOF error taxonomy: attribute PQ loss to failure modes. |
| `expert_gate_ceiling.py` | Would a per-instance gate beat the fixed blend, and could anything drive it? |
| `extract_kernel_sources.py` | Extract markdown+code cells from pulled Kaggle kernels (.ipynb) into plain .txt files. |
| `extraction_schemes.py` | Which extraction scheme recovers the unreachable rows, and what does it cost? |
| `failure_gallery.py` | Failure gallery: what we never see, and what we see badly. |
| `fetch_kaggle_community.py` | Re-runnable scout of Kaggle community knowledge for filament-segmentation-2026. |
| `field_smoothness.py` | Is the test-time semantic field a different object from the OOF field? |
| `film_regime_check.py` | Where the gain of the conditioned network lands, by quartile of disc high-frequency energy. |
| `final_pair_selection.py` | Which submission should take the second final slot, measured by what it adds to the first on test-sized subsets. |
| `fits_at_train_res.py` | Does the FITS advantage survive the resize the model actually applies? |
| `fits_precondition.py` | Does the original FITS recover the faint filaments the JPEG loses? |
| `fits_vs_denoise.py` | Can a denoiser on the existing JPEG buy what the FITS buys? |
| `focal_precondition.py` | Would focal loss up-weight signal, or annotator idiosyncrasy? |
| `fold_design_audit.py` | Whether the fold partition itself is a source of the measured differences between variants, read on five partitions of the same images. |
| `fp_hallucination_taxonomy.py` | What are the hallucination false positives? |
| `fusion_loss.py` | Which fusion step discards reachable ground truth? |
| `fusion_weighting.py` | Whose boundary should a two-lane cluster keep? |
| `gate_instance_quality.py` | The gate for an instance-level change: per-instance quality, not pixel-Dice. |
| `gen_script_index.py` | Generate Appendix D: every script indexed by the question it answers. |
| `hf_oracle_ceiling.py` | Selection ceiling by quartile of disc high-frequency energy, under the metric's break-even rule with realised values. |
| `hf_oracle_transfer.py` | Whether that selection ceiling transfers from one annotator to another. |
| `human_ceiling.py` | What does a human score on this metric, and what is the ceiling? |
| `hysteresis_precondition.py` | Can hysteresis recover the faint filaments a single threshold cannot? |
| `image_quality_residual.py` | Does image quality explain the selector's residual errors? |
| `inference_size.py` | Should a 1536-trained model be run at 2048 for inference, and who belongs in the ensemble? |
| `integrity_audit.py` | Data integrity + leakage audit for the MAGFiLO Kaggle 2026 dataset. |
| `invisible_gallery.py` | What do the consensus filaments our model cannot see actually look like? |
| `label_downsample_loss.py` | How much of the label does the training cache destroy? |
| `lane_miss_profile.py` | Does a cached lane reach the ground truth nothing in our pool ever reached? |
| `latitude_structure.py` | Latitude structure of true positives, false negatives and false positives. |
| `leakage_temporal.py` | Temporal / content leakage analysis between train and test (and across CV folds). |
| `limb_admitted_anatomy.py` | What are the candidates a looser limb threshold admits — and what separates the true ones from the false ones? |
| `limb_failure_anatomy.py` | Where does the limb fail: proposal, selection or boundary? |
| `limb_geometry_test.py` | Is the limb failure geometric (foreshortening) or photometric (signal)? |
| `limb_relative_features.py` | Limb-relative selector features. |
| `limb_selector_threshold.py` | A limb-specific selector threshold on the out-of-fold score. |
| `m2f_miss_profile.py` | Does the query lane reach the GT that nothing else in our pool ever reached? |
| `m2n_report.py` | The organiser's many-to-many plot on our output, plus the unannotated-real-filament gallery. |
| `magfilo_overlap_audit.py` | MAGFiLO v1.0 public dataset vs competition overlap audit — metadata only. |
| `magnetic_pixel_test.py` | Does the magnetic field, at the pixel level, know where filaments are? |
| `magnetogram_probe.py` | Feasibility of co-temporal HMI magnetograms as a second modality. |
| `make_folds.py` | Freeze the leakage-free 5-fold CV split (v1). |
| `marginal_band_value.py` | Is a false positive or a missed filament more expensive, empirically? |
| `mask_ratio_ceiling.py` | Representational ceiling imposed by YOLO's mask_ratio (segmentation target grid). |
| `merge_proposer_precondition.py` | Can a GT-free proposer find the merge pairs the oracle found? |
| `mismatch_cost.py` | The cost, on labelled data, of the one-detector against five-detector mismatch between validation and deployment. |
| `morphology_correlations.py` | Morphology correlation study for MAGFiLO Kaggle 2026. |
| `morphology_eda.py` | Instance morphology EDA for MAGFiLO Kaggle 2026 train annotations. |
| `multithresh_pool.py` | Does multi-threshold semantic extraction raise the pool's reachability? |
| `multithresh_reach.py` | Can multi-threshold extraction reach the faint filaments no candidate ever proposed? |
| `nearmiss_offsets.py` | The population the scorer never shows us: candidates that just miss IoU 0.5. |
| `nearmiss_predictability.py` | Is the per-instance offset a signal, or is it the annotator's drawing noise? |
| `opening_point_hdjojo.py` | Is the advantage of the public 0.40 recipe in the model or in the operating point? |
| `orientation_selector_features.py` | Orientation and context features in the selector. |
| `overlap_density.py` | Does candidate density explain why the leaderboard favours more shrink than cross-fitting does? |
| `pair_union_recovery.py` | Do near-miss GT rows become reachable when two overlapping candidates merge? |
| `paired_crossfit.py` | Paired cross-fit comparison of two pools, because the reported spread is not the error. |
| `patch_selector_eval.py` | Does the image-fed selector actually rank better? |
| `per_source_gates.py` | One confidence gate is applied to two different confidence scales. |
| `perfect_mask_ceiling.py` | The PQ ceiling if the exact annotator mask were drawn at every detection. |
| `phys_channel_regate.py` | Re-test the physical-channel negative result against the instance-level gate. |
| `pil_diagnostics.py` | Why are filaments 50 px from the nearest PIL? |
| `pil_hypothesis.py` | Do the filaments we never propose sit on polarity inversion lines? |
| `pil_selector_test.py` | The cheap magnetogram test: nine PIL features into the selector, single variable. |
| `pixel_vs_instance_errors.py` | Pixel-level and instance-level FP/FN side by side, and how the two connect. |
| `pool_ceiling.py` | What is left in the pool, and is the selector or the candidate set the cap? |
| `pool_complementarity.py` | Selector-free pool reading: what does each configuration find that the others do not? |
| `pq_decomposition.py` | Where does the loss come from on the easy disc (r 0.2–0.7), and is it RQ or SQ? |
| `pq_single_yolo.py` | Fold-0 PQ for a single YOLO checkpoint, organiser-parity scorer, conf sweep. |
| `private_lb_risk.py` | How much can the private half differ from the public one, by chance alone? |
| `public_gt_value.py` | What would the public MAGFiLO annotations score under the competition metric? |
| `quality_features_test.py` | Image-quality and frame-relative feature blocks read through the deployed selector protocol. |
| `quality_selector_features.py` | Image-quality features in the selector. |
| `radial_calibration.py` | Does a confidence of 0.6 mean the same thing at disk centre and at the limb? |
| `radial_cut.py` | Drop every prediction whose body sits outside r 0.9 (and 0.8). |
| `radial_denoise_profile.py` | Where, by radius, does the denoised model beat the baseline — and is the model's size error systematic? |
| `radial_threshold.py` | Is the limb actually under-selected, and would a radial keep-threshold help? |
| `random_split_reference.py` | Where does a single random validation split of n images put our pipeline? |
| `ranker_ceiling.py` | How much is left in the ranking, as opposed to the operating point? |
| `ranker_ceiling_null.py` | Is the +0.0321 ranking ceiling real, or does it only exist for the annotator it was fitted to? |
| `recall_headroom.py` | How much PQ headroom is left in recall, and is it worth chasing? |
| `regime_gate.py` | A keep threshold set by the frame's disc high-frequency energy, chosen by nested cross-validation. |
| `regime_relative_features.py` | Each candidate's evidence expressed relative to its own frame (percentile and z-score), as selector features. |
| `rejected_true_anatomy.py` | The 1029 rows the pool already reaches and the selector throws away. |
| `relational_features.py` | Does a candidate's neighbourhood geometry carry signal the selector does not have? |
| `relational_selector_test.py` | The neighbourhood-geometry features read through the selector over five folds. |
| `report_figures.py` | Figures for the final report, generated from committed measurement artefacts. |
| `resolution_study.py` | Resolution degradation study for MAGFiLO filament masks. |
| `router_learned.py` | Can a router actually be learned? |
| `router_oracle.py` | Would a learnable router beat plain averaging? |
| `scorer_parity.py` | Appendix C: verify our PQ implementation against the organisers' own scorer. |
| `scorer_parity_adversarial.py` | Parity cases built to break the agreement, not to confirm it. |
| `scse_gate_probe.py` | What does the production decoder's scSE attention do with the upsampled branch versus the skip branch on thin filaments? |
| `seed_lane_union.py` | Does the seed-refined lane reach GT the other two lanes miss? |
| `selector_committee.py` | Whether averaging several sub-floor selector variants yields a gain that none of them has alone. |
| `selector_context_features.py` | Let each candidate see the others: the simplest form of attention in the selector. |
| `selector_dl.py` | A deep selector against the boosted-tree one, on the same 42 features. |
| `selector_feature_augmentation.py` | Augmentation for the selector, which this project never tried. |
| `selector_feature_importance.py` | What each selector feature is worth, measured out of fold. |
| `selector_image_bagging.py` | Real bagging of the selector, by image bootstrap. |
| `selector_information_limit.py` | Are the 63 near-gate selector drops a feature problem or an information limit? |
| `selector_loss_variants.py` | The selector's loss, which has never been changed. |
| `selector_regularization.py` | Is the selector overfit, and does a physics-shaped prior fix it? |
| `selector_target_normalised.py` | Normalise the regressor target by the annotator copy count. |
| `semantic_ensemble_gate.py` | Does combining semantic ensemble members pay, measured on existing probability maps? |
| `semmap_limb_reach.py` | Reach by radius band for a semantic probability map, against a baseline map. |
| `setlevel_pq_separability.py` | Is a set-level PQ objective actually a source of headroom for selection? |
| `shape_photometry_features.py` | Mask-shape and photometric features the selector does not have. |
| `shrink_rule_damage.py` | What the two shrink rules actually do to a thin filament. |
| `shrink_topology.py` | Does the production shrink break filaments at the waist, and does protecting it pay? |
| `size_calibration.py` | Is the selector calibrated across candidate size, and what is the exact keep rule? |
| `size_conditioned_threshold.py` | A per-component threshold that depends on the component's own size. |
| `snr_binding.py` | Is the model's blindness to faint filaments actually SNR-limited? |
| `spine_agreement.py` | Precondition for spine-supervised training: do annotators agree on the centreline better than on the mask? |
| `spine_aux_reach.py` | Did the spine head change what the model can shape, even though pixel-Dice fell? |
| `spine_ribbon.py` | Is an annotated filament just a constant-width ribbon around its spine? |
| `spine_vs_mask_iou.py` | Settle the spine question in one unit: ribbon IoU vs mask IoU. |
| `sq_anatomy.py` | Where exactly does a matched mask lose its IoU, and would adaptive shrink recover any of it? |
| `sq_headroom.py` | How much of the SQ gap is width calibration, and how much is shape? |
| `stability_features.py` | Threshold-stability features: new information for the selector, not a new fitter. |
| `stage_attribution.py` | Which encoder stage carries which filament, and does it depend on type or position? |
| `target_policy_oracle.py` | Which annotator target could a perfect model be trained on? |
| `temporal_pair_ceiling.py` | Does a second H-alpha frame close in time carry information about the first frame's annotation? |
| `test_carve_filter.py` | Whether discarding heavily carved candidates after greedy de-overlap removes false-positive fragments. |
| `test_set_inference.py` | What does the leaderboard's ordering of our own submissions say about the test set? |
| `three_lane_pool.py` | Does the seed-refined lane survive fusion, or get absorbed like the others? |
| `threshold_5fold_twin.py` | Semantic extraction threshold 0.3 → 0.2 for the production field, five folds. |
| `threshold_for_test_pq.py` | The keep threshold is calibrated to the wrong PQ, and only the leaderboard can show it. |
| `threshold_pool_twin.py` | The semantic extraction threshold below 0.20, each pool read with its own refitted selector. |
| `tile_geometry.py` | Choose the tiling geometry from our data, rather than copying a public one. |
| `tile_sample_check.py` | Do the tiled training samples actually look right, and is the mask still on the filament? |
| `tp_fp_fn_map.py` | Map of true positives, false negatives and false positives in disc coordinates for the production policy on the out-of-fold pool. |
| `train_test_area_shift.py` | Why the leaderboard favours more shrink than cross-fitting does. |
| `train_test_shift.py` | Is the leaderboard test set drawn from the same distribution as the 707 train images? |
| `transfer_instrument.py` | Covariate-shift reweighting of local validation towards the test set, and why it is not possible here. |
| `true_unreachable.py` | The precondition gate has been measured against a pool that no longer exists. |
| `two_by_two_gate.py` | Two seeds × two folds: the only gate that can see a decoder change. |
| `two_stage_selector.py` | Rank candidates by p(match) × expected IoU, not by p(match) alone. |
| `unmerge_oracle.py` | Unmerge: the applicable ceiling and a GT-free cut proposer. |
| `what_would_055_take.py` | What would have to be true for a PQ of 0.53–0.55? |
| `zero_overlap_profile.py` | What are the 282 ground-truth rows nothing we keep even touches? |
:::

