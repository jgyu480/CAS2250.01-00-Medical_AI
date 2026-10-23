# Modulewise, meta-learned partial PCGrad

Two new experiments: `static` (one learned intercept per task/group; 10 parameters)
and `conditional` (intercept + severity + log norm ratio + preceding severity EMA;
40 parameters). Model architecture, input/label handling, focal cell loss, CE+Dice
tissue loss, and postprocessing are unchanged. The previous branch-weight gates
are NOT combined with this method.

For each disjoint shared group k, compute c=grad(L_cell), t=grad(lambda L_tissue).
On dot(c,t)<0, define c_pc=c-dot(c,t)/||t||^2*t and
t_pc=t-dot(c,t)/||c||^2*c, always using ORIGINAL anchors. Apply
d=(1-rho_c)*c+rho_c*c_pc+(1-rho_t)*t+rho_t*t_pc.
Otherwise d=c+t. Private tensors retain their only available task gradient;
frozen tensors have no update. This is modulewise PCGrad, not global-vector
PCGrad. For two tasks this correction can also be expressed as task reweighting;
novelty or superior granularity is not established by this algebra alone.

Groups: cell.encoder, tissue.encoder, encoder_c2t, tissue.aspp, tissue.decoder.
These change parameter updates, not individual backward paths through an edge.
The real-data preflight verifies the groups cover all doubly-reached parameters.

Conditional inputs: max(0,-cos(c,t)), log(||other||/||own||) clipped [-8,8],
previous EMA of severity, and intercept. EMA decay=.95; it advances once per
training batch, after application, for valid active groups only. Statistics and
task gradients are detached. Initial rho=.1 (intercept=logit(.1), slopes=0).
Sigmoid outputs range (0,1); exact endpoints are numerical test conditions.

## Data and optimization

Same train-case split as the previous dual-gate experiments: requested meta=40
of the 400 official train pairs, seed 2042; remaining pairs fit the model.
Official validation 137 is used only for checkpoint selection; test untouched.
The trainer verifies the split and common settings against the old fixed run
when present. Meta images have no augmentation.

Gate objective: J=.5*(L_cell_meta/a_cell + L_tissue_meta/a_tissue).
The two positive scales are mean losses over the entire meta subset before any
training, fixed thereafter and shared across modes. Calibration restores RNG
and loader generator state. This normalized joint objective differs from the
older two gates' diagonal meta objectives; static-vs-conditional is the clean
comparison isolating conditioning among the new runs.

Every 5 model batches, use the current AdamW moments, step counters, betas, epsilon,
decay, learning rates, global clipping and current detached gradients for a
differentiable virtual update. Evaluate a meta batch with dropout disabled and
cloned buffers; optimize only gate coefficients with Adam(lr=.01), gate gradient
clip=1. A batch with no conflict-dependent update has no meta learning signal.
There is no differentiation through past optimization. Real model/moments/modes
must remain unchanged during the virtual update. The applied ratios are
recomputed after the gate step. Main optimizer: existing two-rate AdamW, cosine,
batch 2, 1024px, freeze 3 epochs, BF16 when supported, 50 epochs, seed42.

## Logs and recovery

Each run writes history.csv, full best.pt AND last.pt (including model/gate
optimizers, EMA, fixed scales, RNG and settings), raw gradient_conflict logs,
and pcgrad_gate/{batches,epochs,coefficients}.csv. Actual applied ratios, pre/post
shared-gradient dot products, pair IDs and meta IDs are logged every batch.
Adverse rates are local SGD geometry, NOT observed AdamW loss increases.
No raw gradient vectors or intermediate model states are stored per batch.

Checkpoint resume is at epoch boundaries; uncheckpointed log rows are trimmed.
Exact source hashes are saved even before a git commit. Existing outputs are
never silently overwritten. Suite status distinguishes checks/GPU preflight/
running/completed/failed. A failed check prevents full training; the second
experiment starts only after the first has completed and its records validate.

## Verification and interpretation

CPU checks: independent full/partial projection, private/frozen behavior,
feature dependence, EMA, fresh/warmed clipped AdamW agreement, finite-difference
gate hypergradients, virtual nonmutation, checkpoint round-trip. Integration test
runs the actual trainer with tiny synthetic data and checks exact uninterrupted
vs resumed weights, moments, EMA and journals. Real GPU preflight runs both modes
at 1024px/batch2 for two epochs, frozen then unfrozen, including actual resume.
Full jobs run only after these checks pass on the user's GPU/environment.

A higher validation mF1 in one seed is an observed improvement, not statistical
significance. Conditional > static supports the utility of batch conditioning
in these tested settings. Repeated seeds and locked final test evaluation are
needed before a robust generalization claim. Removing all conflicts is not the
gate objective, and lower raw conflict alone is not evidence of improvement.

## References

- Yu et al. Gradient Surgery for Multi-Task Learning. NeurIPS 2020.
  https://arxiv.org/abs/2001.06782
- Wang et al. Gradient Vaccine. ICLR 2021.
  https://arxiv.org/abs/2010.05874
- Dey and Ning. Enhancing Molecular Property Prediction with Auxiliary Learning
  and Task-Specific Adaptation (RCGrad/BLO+RCGrad). 2024.
  https://arxiv.org/abs/2401.16299
- Liu et al. Auto-Lambda: Disentangling Dynamic Task Relationships. TMLR 2022.
  https://shikun.io/projects/auto-lambda

The particular module partition, conditional sigmoid controller and normalized
joint meta objective here are an experimental combination, not a reproduction
or guarantee from any single cited paper.
