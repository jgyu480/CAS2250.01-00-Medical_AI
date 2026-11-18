# Five controlled ensemble experiments

| Method | Change |
|---|---|
| pointwise | Re-trained old 15-input gate, softmax(0.1*tanh(z)) |
| softmax | Same input/head/training; softmax(z), T=1 |
| pool | softmax(z) plus previous 3x3-convolution/large-average context |
| spatial | Six stacked dilated spatial convolutions, RF 3/7/15/31/63/127; disagreement maps; channel LayerNorm |
| gain | Same spatial encoder plus learned mean fallback alpha and local competence/preservation losses |

All five gates train from scratch with seed42, AdamW 1e-3, 12 epochs by default,
four identical sampled 128px cores/image/epoch, including image borders, and up to 63px halos. Base G/GA/M0/M1/M1-1 are frozen;
existing joint cell/tissue D4 caches are reused. No AE is trained. Each spatial scale retains
16 learned cell and 16 tissue channels; receptive fields are in cell-image pixels.
Tissue resolution remains that of the source 256px cache; upsampling adds no detail.
The spatial-vs-pool contrast changes encoder, explicit disagreement inputs and normalization together.
The gain-vs-spatial contrast changes fallback and auxiliary losses together; it cannot isolate their individual effects.
The gain teacher uses local max-3 scores and binary center events (<=2px positive, >=15px negative)
on disagreeing candidates. It is a soft competence proxy, not an official detection-match oracle.
Routing weight=.2; safe-agreement preservation weight=1; primary sampled Gaussian-sigma6 BCE is held constant.
Gate checkpoints are the fixed-budget final epoch, not selected by training loss.
Source validation fit/tune patient IDs are reused; this is exploratory stacking on part of official validation,
not a train-only official benchmark. Earlier official test results were already observed.
Postprocessing and selected method are frozen on validation before accessing test labels in this run.
summary.csv contains shared-TTA-postprocessing and per-method-val-postprocessing results.
gate_diagnostics.csv reports weight variability, tanh saturation, alpha and sensitivity to shifted context.
