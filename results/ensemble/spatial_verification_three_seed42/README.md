# Three simple spatial candidate verifiers

All three use the SAME frozen spatial CNN context (RF3,7,15,31,63,127), candidate pool,
feature vectors, seed42, 20 epochs by default, batch512 and AdamW1e-3.
- verify: one 3-class head (reject/BC/TC), ordinary candidate cross entropy.
- balanced: identical architecture, equal total weights for existing matched candidates,
  existing false candidates, recoverable misses and extra false candidates.
- dual: same balanced loss; separate keep/reject/relabel head for existing spatial detections
  and rescue/reject head for extra proposals. Both heads predict reject/BC/TC.

Frozen encoders and gates are loaded from ensemble_ablation_five_seed42 with hash checks.
Only LayerNorm -> Linear64 -> ReLU -> Linear32 -> ReLU -> class head(s) are trained.
Features: raw15, local max3 cell10, centered cell10, mean/softmax/spatial6,
baseline flag and score2, spatial context192; total235. Spatial vectors are cached as fp16.
This is post-detection verification/rescoring, NOT another dense pixel-weighted-average gate.
Proposals are the union of five D4 base maps, mean, softmax and spatial maps, using
Gaussian3, minimum-distance3, threshold.05 plus exact existing spatial detections.
Locations within3px are merged, prioritizing existing spatial detections. Coordinates are
never supplied by GT; no learned position regression is used. Cells with no candidate
within15px cannot be recovered. Candidate coverage is audited in each split.
Training labels use class-agnostic 15px score-ordered one-to-one matching, prioritizing
existing spatial candidates; matched candidates receive the correct BC/TC target.
Unmatched candidates receive reject, including duplicate proposals. This is an imperfect
supervision assignment; final evaluation uses the existing classwise official-style matcher.
Postprocessing has separate keep/rescue thresholds and class-agnostic point suppression
radius3/5/7, selected only on validation tune IDs. These settings are frozen before test.
Baseline scores use their original per-method validation postprocessing unchanged.
Candidate audits and per-pair records distinguish rescued GTs, lost GTs, FP and FN.
The same source validation fit/tune patient IDs are reused. The feature encoder itself was
trained on those fit IDs. This is exploratory validation-trained stacking, not a train-only
benchmark; previous test scores were already observed. No new base-model training or TTA
inference is performed. D4 cell/tissue caches are reused, and tissue resolution stays256.

Literature inspiration, not exact reproductions:
- Wang et al. 2011, learning-based wrapper correction: https://doi.org/10.1016/j.neuroimage.2011.01.006
- Xie et al. SESV, TMI2021: https://doi.org/10.1109/TMI.2020.3025308
- Huang et al. Mask Scoring R-CNN, CVPR2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html
