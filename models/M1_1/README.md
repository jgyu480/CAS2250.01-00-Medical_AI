# M1-1: Spatial Gated Dual-Head (Frozen-M1 experiment)

## Research question

Does learning **how much** to trust cross-attention-enhanced tissue context at each cell pixel improve classification relative to M1's tissue-aware head alone, or a fixed 50/50 mixture?

This is the design documented in Report 1: two three-class cell heads (background/BC/TC), joined by a scalar spatial gate. The tissue segmentation head remains unchanged. Only the cell classification path is modified.

## Model

- **Frozen M1:** separate SSL cell/tissue encoders, ASPP, cell-to-tissue global cross-attention at ASPP, decoder-level `TissueToCell` coordinate fusion, original cell prediction head, original tissue decoder/head. Its exact inference is preserved (only `eval()` mode).
- **New local head:** a deep copy of the trained M1 cell decoder/head **after loading the M1 checkpoint**, driven by raw cell ASPP output and low-level cell features, with **no access to tissue**.
- **Gate:** 16-channel projections of the local and fused 256-channel decoder features, their absolute difference, both 3-class probability maps (at decoder resolution), then 3x3 Conv + GroupNorm + SiLU + 1x1 Conv + sigmoid. One gate value per pixel, initially 0.5.
- **Prediction:** `P_final = (1-g)*softmax(local_logits) + g*softmax(m1_logits)`. Returns `log(P_final)` under `cell_logits` so all current detection, focal loss and postprocessing code works unchanged; `tissue_logits` is the frozen M1 output.
- **Training:** stage A (5 epochs): local decoder/head, `FL(local)`; stage B (20 epochs): local decoder/head and gate, `FL(final) + 0.2*FL(local)`; all M1 weights remain frozen. AdamW LR 1e-4, weight decay 0.01, separate cosine schedules per stage, gradient clip 10, standard augmentation and batch size 2.

**GT tissue masks are never given as model input.** Offline analysis may read GT tissue masks for region-specific diagnostics only.

## Tests and training

From **repository root**, after unpacking the overlay (see `INSTALL_M1_1.md`):

```bash
# Checks without OCELOT dataset or SSL checkpoint
python scripts/check_m1_1.py

# Optional real M1 strict state/forward equivalence check
python scripts/check_m1_1.py --real-m1-checkpoint outputs/m1/M1_seed42/best.pt

# Small smoke run (requires trained M1 checkpoint + OCELOT data + SSL weights)
python scripts/train_m1_1.py --m1-checkpoint outputs/m1/M1_seed42/best.pt --smoke

# Full training, seed 42
python scripts/train_m1_1.py --m1-checkpoint outputs/m1/M1_seed42/best.pt --seed 42 --run-name M1_1_seed42

# Resume after interruption
python scripts/train_m1_1.py --resume outputs/m1_1/M1_1_seed42/last.pt

# Select dynamic model's postprocessing on validation and compare branches
python scripts/eval_m1_1.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split val --tune --ablation

# Region-specific error rescue/harm + gate distribution (uses GT only offline)
python scripts/analyze_m1_1_errors.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split val

# When experiment is finalized: evaluate test ONCE with val-chosen settings
python scripts/eval_m1_1.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split test --ablation
```

Change checkpoint paths to match the actual M1 run. **Do not use M0 weights**: the M1 teacher checkpoint must load with `strict=True`. The output directory is `OUTPUT_ROOT/m1_1/<run>`, or the default `outputs/m1_1/<run>`.

The smoke test uses 128px images / two train and val examples, runs one epoch per stage, and is **not a meaningful experiment**. For full training use 1024px and complete official train/val splits. On a different GPU, adjust only hardware-related arguments as appropriate.

## Ablation and inference details

`eval_m1_1.py --ablation` computes `dynamic`, `local`, `m1`, and `fixed_0_5` using **one shared postprocessing selection chosen by dynamic on validation**. This controls for postprocessing differences; previously reported independently tuned M1 results may not be directly identical to this `m1` ablation, even though the underlying logits match. The official-style mF1 always uses `common.metrics.m0_eval` and matches 15px per-class.

For valid test practice, `--split test` refuses to run without a validation-created `postprocess.json`. Do not repeatedly tune against test. The diagnosis script uses a *class-agnostic greedy 15px* matching to categorize missing vs wrong-class cells; this is explicitly **diagnostic**, not the official evaluation metric.

Use `scripts/check_m1_1.py --real-m1-checkpoint ...` to ensure original M1 `cell_logits` / `tissue_logits` are exactly preserved. An M1 output discrepancy invalidates the experiment.

## Files and outputs

New code only: `common/modules/gated_fusion.py`, `models/M1_1/`, `configs/m1_1_train.json`, `scripts/{train,eval,check,analyze}_m1_1.py`. Existing M0/M1 code is not modified.

`outputs/m1_1/<run>` includes:

- `config.json`, `history.csv` (stage, losses, F1, mIoU), `warmup.pt`, `last.pt`, `best.pt` (best stage B validation mF1)
- `postprocess.json`, `eval_{val,test}.json`, `eval_{val,test}_per_image.json`
- `eval_{val,test}_ablation.json`, `eval_{val,test}_ablation_per_image.json`, `error_analysis_val.json`

Checkpoints include a full copy of the frozen M1 teacher so that model evaluation/resume does not depend on the original checkpoint file. Configuration saves the original M1 checkpoint absolute path and SHA256 digest for provenance. The existing SSL weights are still needed during `build_model()` initialization (same as the current M1 evaluation script).

## Limitations and next steps

This first experiment **freezes** M1 to isolate the effect of a locally trained head and gate. Thus it tests a frozen-teacher gated mixture, not full end-to-end fine-tuning. Any gain should be compared against the local-only and fixed-0.5 ablations; repeat seeds 42/43/44 before claiming robustness. The cell-only decoder might itself explain gains, so do not attribute improvement to the spatial gate unless dynamic outperforms fixed 0.5.
