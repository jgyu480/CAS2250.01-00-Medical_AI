# Frozen spatial ensembles with final-stage D4 TTA

No retraining. Frozen spatial, spatial_ft, scale_select, local_detail and disagreement
checkpoints are loaded and checked against the preceding frozen run SHA256 records.
Base model caches already contain joint cell/tissue 8-direction D4 TTA.
Each final gate is now evaluated in four rotations, with and without reflection.
All aligned cell/tissue probability maps are transformed together. Each resulting BC/TC
probability map is inversely transformed, then the EIGHT PROBABILITY MAPS are averaged.
This is F_D4(mean_g g^-1 P(g x)), not an average of logits or detection coordinates,
and not base-model inference over 64 independent transformations.
The full tissue FoV is unused by these gates; spatially aligned tissue is resized once,
then jointly transformed. Class/model axes stay fixed. No changes to the encoder or weights.

Both original single-orientation and new *_d4 methods are evaluated side by side.
Tiled evaluation uses 128px cores and 63px halos, covering maximum 127px receptive field.
Rotations/reflections occur within halo tiles and are equivalent to full-frame D4 evaluation;
no averaging of seams or zero-padding is introduced. Native image boundaries retain
per-layer replication padding. A CPU synthetic test checked dense/tiled equality and D4
equivariance, including rectangular inputs and original trained spatial weights.

tta_mean is unchanged. shared: all methods use val-chosen tta_mean postprocessing;
per_method: each method has its own postprocessing chosen on val_tune, then frozen.
Final method selection uses val_tune. summary.csv includes both policies and both splits;
val_per_pair.csv/test_per_pair.csv include classwise metrics, rescue/lost against original
single-orientation spatial. selected_test_score.csv reports the validation-chosen method.
fit/tune IDs are unchanged and the gates were originally fitted on part of official val.
This is exploratory stacking with prior test exposure, not a train-only benchmark claim.
Detailed prediction coordinates remain in local *.jsonl.gz files; GitHub contains code,
configuration, hashes and report tables. Existing checkpoints remain in their source run
and its published branch; they are not duplicated by this evaluation-only run.
