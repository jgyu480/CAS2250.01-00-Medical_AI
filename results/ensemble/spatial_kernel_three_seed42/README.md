# Three minimal context-kernel changes to spatial

All models start from the SAME original spatial checkpoint, checked by SHA256.
Initialization audit verifies the three modifications have the same prediction as spatial.
Original input15 + learned context192 + centered cell10 =217 channels, LayerNorm,
217->16->10 MLP, class-specific expert softmax and weighted original BC/TC maps stay intact.
No candidate verifier, label replacement, extra classification objective or routing loss is used.

| Method | Change |
|---|---|
| spatial | Frozen original reference |
| spatial_ft | Unmodified matched-budget fine-tuning control |
| scale_select | Per-pixel six-scale softmax attention for cell/tissue separately; weighted concat; uniform identity initialization |
| local_detail | Add a zero-initialized dense3x3 path beside each dilation2..32 convolution, summed before ReLU |
| disagreement | Add an independent spatial encoder of centered model scores, cell/tissue separately; six branch strengths start at0 |

The three changes are INDEPENDENT, not stacked together. The first two preserve the
maximum RF3/7/15/31/63/127; the third also uses the same six receptive fields.
Our original powers-of-two stack already has contiguous theoretical RF support. This
experiment does NOT assume proven gridding holes; dense paths test additional local detail.
Centered predictions are deterministic re-expressions of the original maps, not new data.
Scale attention is spatially local; SKNet's original global pooled mechanism is not reproduced.
Additional paths increase parameter counts (see initialization_audit.csv), so gains cannot be
attributed to information type alone without later parameter-matched controls.
All four fine-tuning runs use AdamW1e-4, fixed final12 epochs by default, seed42, same
four128px cores/image/epoch with63px halos, same Gaussian-sigma6 sampled BCE and raw D4
caches. Context encoder and the existing fusion head are both fine-tuned. No shared trainable
state is used between runs. Additional spatial_ft controls the effect of training longer.
Gate diagnostics, scale_attention.csv and kernel_adapters.csv measure use of context; per-pair files track rescue,
lost GTs, BC/TC precision/recall and FP/FN relative to original spatial. These are diagnostics,
not training objectives. The original references are also tuned on the same validation grid;
shared uses the same TTA-derived postprocessing for every method, per_method uses own val PP.
Validation freezes postprocessing and method selection before this run's test evaluation.
Source validation fit/tune patient IDs are reused; this is exploratory stacking on part of
validation, not a train-only official benchmark. Prior test exposure remains applicable.
TTA is the original joint cell/tissue D4 cache; final gates are not separately D4-averaged.
Tissue stays at the original cache256 resolution. G/GA and other base models are frozen.

Literature inspiration, not exact reproductions:
- Selective Kernel Networks, CVPR2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html
- Understanding Convolution for Semantic Segmentation, WACV2018: https://arxiv.org/abs/1702.08502
- Disagreement-based region-specific ensemble,2026: https://doi.org/10.3389/frai.2026.1812932
