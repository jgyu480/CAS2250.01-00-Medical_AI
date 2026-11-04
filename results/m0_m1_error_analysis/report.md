# M0 vs M1 — OCELOT validation error analysis

## Evaluation design
- 137 validation pairs, 24712 GT cells; no test set used.
- Official mF1: class-wise score-ordered 1:1 matching within 15px.
- Diagnostic categories: correct / wrong_class / missed (mutually exclusive).
- Tissue category is sampled from **GT tissue annotations**, not model predictions.
- Stable-context subset requires a non-UNK label and neighborhood purity threshold.
- These diagnostic confusion/miss rates are **not the official F1**.

## Official evaluation (reproduced)

| Metric | M0 | M1 | Difference |
|---|---:|---:|---:|
| mF1 | 0.7033 | 0.6850 | -0.0183 |
| BC | 0.6272 | 0.5998 | -0.0273 |
| TC | 0.7795 | 0.7703 | -0.0093 |

## Key context-sensitive cases (ground-truth cell denominator)

| Case | subset | GT count | M0 wrong class | M1 wrong class | M0 miss | M1 miss |
|---|---|---:|---:|---:|---:|---:|
| BC_in_CA | all | 1152 | 29.95% | 30.03% | 29.43% | 30.64% |
| BC_in_CA | stable | 1116 | 29.84% | 30.02% | 29.75% | 30.91% |
| TC_in_BG | all | 784 | 38.90% | 33.93% | 20.92% | 25.00% |
| TC_in_BG | stable | 720 | 40.97% | 35.00% | 20.42% | 25.00% |

## Interpretation and caveats

- Wrong-class detection is a *secondary*, one-to-one, opposite-label matching after official correct-label matching; it cannot be directly equated with the official false-positive count.
- Missed means no available opposite-label detection within 15 px after correct matches; ambiguity and dense-cell effects may remain.
- A local tissue center label is a region-level annotation, not proof of histological cell identity; inspect near-boundary cases and UNK.
- The two models have independently validation-tuned postprocessing. To isolate architectural effects, consider an additional common-postprocessing comparison.
- M1 also removes the M0 cell-to-tissue pathway; observed differences cannot be attributed exclusively to cross-attention.
- Validation is also used to choose the checkpoint and postprocessing, so these results are exploratory, not a final test-set estimate.

## Other outputs

- `analysis_summary.json`: machine-readable aggregate results
- `per_gt_comparison.csv`: paired GT-level errors, tissue context and M0→M1 transitions
- `extra_predictions.csv`: unmatched predictions (near-GT extra vs isolated FP)
- `per_image_comparison.csv`: per-image mF1/F1 changes
