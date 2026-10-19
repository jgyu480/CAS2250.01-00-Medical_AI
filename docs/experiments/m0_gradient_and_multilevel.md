# M0 training-gradient diagnostics and M0-1

This package adds experiment entry points without modifying original M0 files.
No dataset training is run during installation or synthetic checks.

## 1. Measurement

At EVERY training minibatch, on the SAME stochastic forward as the training loss,
compute g_c = grad L_cell and g_t = grad(lambda L_tissue). Only trainable parameter
tensors reached by both losses enter the comparison. Task-private tensors are excluded.
All-shared scope includes both encoders and exchange/decoder tensors that both losses reach;
component scopes are recorded separately. No gradient surgery is applied.

Published primitives (Yu et al., NeurIPS 2020, PCGrad, Definitions 1 and 2):
- cosine = dot(g_c,g_t)/(||g_c|| ||g_t||); conflict if cosine < 0.
- magnitude similarity = 2||g_c||||g_t||/(||g_c||^2+||g_t||^2).

Primary epoch statistic: fraction of valid minibatches with negative cosine.
Also record mean cosine, mean max(0,-cosine), magnitude similarity, both norms,
and first-order adverse-direction fractions on the same shared parameter scope.
These epoch aggregates are explicitly defined descriptive statistics, not a paper's
standardized epoch severity index. A zero norm makes the angle undefined: report
invalid observation count rather than replacing it with zero conflict.

Derived auxiliary statistic: epoch cancellation fraction =
sum_b 2 max(0,-dot_b) / sum_b (||g_cb||^2 + ||g_tb||^2).
This captures norm-weighted cancellation, NOT a universal measure of harm; a small
norm task can be dominated while total cancellation is small. First-order adverse
flags for -g_c-g_t are NOT measurements of actual AdamW update harm or curvature.

Every minibatch is measured; no subsampling. This is batch-gradient analysis,
not individual-image analysis. Batch size matters. Grad accumulation is fixed at 1
in this entry point to keep each observation aligned to an attempted update.
FP16 GradScaler may skip an update; observations still cover its training minibatch.
Raw gradients are measured BEFORE loss scaling, clipping, or optimizer updates.
Reductions use float32; forward/backward use the run's recorded AMP precision.
For FP32 diagnostics pass --fp32 to the launcher (more GPU memory required).
Additional two task-gradient reverse passes per batch increase time and memory.
Encoder-frozen epochs have a different trainable scope; plot them separately.
Negative cosine alone does not establish lower accuracy or slower convergence.

Outputs under each run/gradient_conflict:
- batches.csv: every epoch/batch/group, image IDs, dot/norms, angle, invalid flags.
- epochs.csv: epoch/group aggregates, observed/valid/invalid batch counts.
- protocol.json: definitions and scope.
- plot_epoch_conflict.py: PNG/PDF/SVG from the CSV, independent of model/data.

## 2. M0-1 architecture

M0 already has a shallow stage-1 skip. M0-1 replaces ASPP + decoder with
SegFormer-style projections and fusion of stages 1,2,3,4. Per-pixel linear 1x1
projections, upsampling to stage-1 resolution, concatenation, linear fusion,
GroupNorm/ReLU/dropout, and a 3-class head are used for each branch.

The SSL ResNet-50 architecture and weights are retained, including frozen BN
statistics and output stride 16. Stage resolutions are 1/4,1/8,1/16,1/16 because
M0 dilates stage 4; do not report them as the original MiT hierarchy.
The model is a ResNet + SegFormer-style decoder hybrid, NOT original SegFormer.
Three exchanges remain: Cell->Tissue after deepest encoder stage; Tissue->Cell
at deepest projected feature (replaces ASPP port); Tissue->Cell after fusion.

The trained M0's BOTH encoders are loaded strictly. Other modules initialize anew.
An M0 control loads the same encoders and reinitializes its other modules, then
uses the same loss, augmentation, epochs, learning rates, seed and batch size.
New module parameter counts differ by design. Use multiple seeds for conclusions.
A full M0 continuation is supported separately with --init-scope full, but that
comparison alone does not isolate decoder architecture from initialization.

Current defaults: M0 gradient rerun = original 50 epochs and 3 frozen epochs;
M0 control/M0-1 = matched 30 epochs, encoders trainable from epoch 1,
encoder LR 1e-5 / new module LR 1e-4; inherited current M0 focal and tissue losses.
30 epochs is a starting budget, not a validated convergence requirement.
Both matched runs must use identical M0_CHECKPOINT, seed, batch size and precision.
Original best.pt is initialization only; optimizer/scheduler are reset.
Resume with last.pt and original arguments; the script rejects setting changes.
Resume restores optimization state, but not all RNG states; an interrupted run
need not be bit-for-bit identical to an uninterrupted stochastic trajectory.

## 3. Local preparation (no dataset needed)

python -m compileall -q common/training/gradient_conflict.py common/training/experiment_models.py models/M0_1 scripts/train_m0_experiments.py scripts/eval_m0_experiments.py scripts/check_m0_experiments.py scripts/plot_epoch_conflict.py scripts/run_m0_experiment.py
python scripts/check_m0_experiments.py
python scripts/run_m0_experiment.py --job gradient
python scripts/run_m0_experiment.py --job control
python scripts/run_m0_experiment.py --job m0_1

Synthetic checks test forward/backward, BN behavior, exact encoder transfer,
parameter groups, geometry, monitor non-mutation, and incomplete-epoch cleanup.
They do not estimate performance or 1024px CUDA memory usage.

## 4. Later on VESSL

Choose a PyTorch CUDA image with matching torchvision. Additional packages:
python -m pip install numpy Pillow scipy matplotlib
Set these paths to your mounted files (examples only):
export OCELOT_DATA_ROOT=/mnt/data/ocelot2023_v1.0.1
export SSL_WEIGHTS_PATH=/mnt/weights/bt_rn50_ep200.torch
export OUTPUT_ROOT=/mnt/outputs
export M0_CHECKPOINT=/mnt/checkpoints/M0/best.pt

python scripts/check_data.py
python scripts/check_m0_experiments.py

Only the following commands start training:
python scripts/run_m0_experiment.py --job gradient --execute
python scripts/run_m0_experiment.py --job control --execute
python scripts/run_m0_experiment.py --job m0_1 --execute

Optional: --gradient-conflict on control and m0_1 records their own trajectories.
Optional: --fp32 disables AMP; keep precision matched across comparison runs.
If reducing batch size for GPU memory, match it across compared runs.
Results: $OUTPUT_ROOT/m0/gradient_seed42, $OUTPUT_ROOT/m0/control_seed42,
$OUTPUT_ROOT/m0_1/m0_1_seed42.
Resume example (reuse all original precision/batch/seed flags):
python scripts/run_m0_experiment.py --job gradient --resume "$OUTPUT_ROOT/m0/gradient_seed42/last.pt" --execute

## 5. Analysis after training

python scripts/plot_epoch_conflict.py --run "$OUTPUT_ROOT/m0/gradient_seed42"

First compare FIXED identical postprocessing on validation, for EACH matched run:
python scripts/eval_m0_experiments.py --checkpoint "$OUTPUT_ROOT/m0/control_seed42/best.pt" --split val
python scripts/eval_m0_experiments.py --checkpoint "$OUTPUT_ROOT/m0_1/m0_1_seed42/best.pt" --split val

The original cached-analysis scripts operate on saved predictions and can process
M0-1 without loading the M0 architecture. For EACH run:
python scripts/diagnose_m0.py cohorts --run "$OUTPUT_ROOT/m0_1/m0_1_seed42"
python scripts/analyze_m0_next.py --run "$OUTPUT_ROOT/m0_1/m0_1_seed42" --cached-only

Compare cell_groups.csv: BC/TC miss rate in 14-30 vs 30+ bins, and misclassification
of BC in CA / TC in BG. Matching is location-first diagnostic matching, separate
from official class-wise F1. Include total F1, BC recall, FP counts and tissue mIoU;
reducing misses by detecting everything is not sufficient evidence of improvement.

Optional later: tune postprocessing on val for each model, freeze it, evaluate test.
Tuning overwrites the run's eval_val files, so retain copies of fixed-postprocessing
results before tuning if that comparison must be kept.
python scripts/eval_m0_experiments.py --checkpoint "$OUTPUT_ROOT/m0_1/m0_1_seed42/best.pt" --split val --tune
python scripts/eval_m0_experiments.py --checkpoint "$OUTPUT_ROOT/m0_1/m0_1_seed42/best.pt" --split test

## References

Yu et al. Gradient Surgery for Multi-Task Learning. NeurIPS 2020.
https://arxiv.org/abs/2001.06782
Xie et al. SegFormer: Simple and Efficient Design for Semantic Segmentation
with Transformers. NeurIPS 2021, Section 3.2.
https://arxiv.org/abs/2105.15203
