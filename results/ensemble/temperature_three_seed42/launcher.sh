#!/usr/bin/env bash
set -euo pipefail
root=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
source_run=${M3_SOURCE_RUN:-"$root/experiments/m3_context_d4_retrained_val_seed42"}
out="$root/experiments/${TEMP_RUN_ID:-temperature_three_seed42}"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo 'Temperature experiment already running.' >&2; exit 1; }
[[ "${TEMP_RUN_ID:-temperature_three_seed42}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || exit 2
test -s "$source_run/complete.ok"
test -s "$source_run/m3_context.py"
if [[ -f "$out/launcher.sh" ]]; then
  cmp -s "$0" "$out/launcher.sh" || { echo 'Script changed; set a new TEMP_RUN_ID.' >&2; exit 1; }
else
  cp "$0" "$out/launcher.sh"
fi
if [[ -x "$root/venv/bin/python" ]]; then
  default_python="$root/venv/bin/python"
elif [[ -x /opt/conda/envs/cas2250-medical-ai/bin/python ]]; then
  default_python=/opt/conda/envs/cas2250-medical-ai/bin/python
else
  default_python=python3
fi
export TEMP_SOURCE_RUN="$source_run" TEMP_OUTPUT="$out" M3_STATE_ROOT="$root"
export M3_REPO="$root/m3_runtime/repo" M3_OUTPUT="$out"
"${M3_PYTHON:-$default_python}" -u - <<'PY' 2>&1 | tee -a "$out/console.log"
import csv
import gzip
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

source = Path(os.environ["TEMP_SOURCE_RUN"]).resolve()
out = Path(os.environ["TEMP_OUTPUT"]).resolve()
# Load the completed run's exact helpers and model adapters.
os.environ["M3_OUTPUT"] = str(source)
spec = importlib.util.spec_from_file_location("prior", source / "m3_context.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.OUT = out
config = json.loads((source / "run.json").read_text())
device = torch.device(os.environ.get("M3_DEVICE", "cuda"))
torch.set_num_threads(2)
if device.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")
epochs = int(os.environ.get("TEMP_EPOCHS", "12"))
ae_epochs = int(os.environ.get("AE_EPOCHS", "5"))
ratio = float(os.environ.get("WEIGHT_RADIUS", "0.1"))
if min(epochs, ae_epochs) < 1 or not 0 < ratio <= 2:
    raise ValueError("Positive epochs and WEIGHT_RADIUS in (0,2] are required.")
paths = [Path(config["checkpoint_paths"][n]) for n in m.MODEL_NAMES]
for n, p in zip(m.MODEL_NAMES, paths):
    if m.sha256(p) != config["checkpoint_sha256"][n]:
        raise ValueError(f"Source checkpoint changed: {n}")
    if n in m.M2_FINALS:
        m.verify_m2(n, p)
if m.sha256(source / "m3_context.py") != config["implementation_sha256"]:
    raise ValueError("Source implementation changed.")
m.VARIANTS = ("pointwise", "fixed", "learned", "ae")
m.METHODS = ("tta_mean", *m.VARIANTS)

class MultiscaleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList(nn.Sequential(
            nn.Conv2d(15, 12, 3, padding=d, dilation=d), nn.ReLU(),
            nn.Conv2d(12, 3, 3, padding=1)
        ) for d in (1, 4, 12))
        self.decoder = nn.Sequential(nn.Conv2d(9, 16, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(16, 15, 1), nn.Sigmoid())

    def encode(self, x):
        return torch.cat([encoder(x) for encoder in self.encoders], dim=1)

    def forward(self, x):
        return self.decoder(self.encode(x))

def low_maps(cells, aligned):
    return torch.cat((F.adaptive_avg_pool2d(cells, (256,256)), aligned), dim=1)

class Gate(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        channels = 15
        if variant == "fixed":
            channels = 25
        elif variant == "learned":
            channels = 25
            self.cell_context = m.LearnedContext(10, (1,14,29))
            self.tissue_context = m.LearnedContext(5, (1,3,7))
        elif variant == "ae":
            channels = 24
            self.autoencoder = MultiscaleAE()
            state = torch.load(out / "ae_pretrain.pt", map_location="cpu", weights_only=True)
            self.autoencoder.load_state_dict(state["model"])
            # Only the encoder participates in supervised ensemble fitting.
            for p in self.autoencoder.decoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Sequential(nn.Conv2d(channels,16,1), nn.ReLU(), nn.Conv2d(16,10,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, cells, tissue, base, aligned, box, extent):
        size = cells.shape[-2:]
        inputs = [cells, F.interpolate(aligned,size=size,mode="bilinear",align_corners=False)]
        if self.variant == "fixed":
            inputs.extend((m.fixed_cell_context(cells),
                           m.crop_to_cell(m.fixed_tissue_context(tissue,extent),box,size)))
        elif self.variant == "learned":
            inputs.extend((self.cell_context(cells), m.crop_to_cell(self.tissue_context(tissue),box,size)))
        elif self.variant == "ae":
            encoded = self.autoencoder.encode(low_maps(cells,aligned))
            inputs.append(F.interpolate(encoded,size=size,mode="bilinear",align_corners=False))
        logits = self.head(torch.cat(inputs,dim=1)).reshape(cells.shape[0],5,2,*size)
        weights = (ratio * logits.tanh()).softmax(dim=1)
        scores = (weights*cells.reshape_as(weights)).sum(dim=1)
        return torch.logit(scores,eps=1e-6)

def pretrain(pairs):
    m.seed_everything(42)
    autoencoder = MultiscaleAE().to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(),lr=.001)
    path = out / "ae_pretrain.pt"
    start, history = 0, []
    if path.is_file():
        state = torch.load(path,map_location="cpu",weights_only=True)
        autoencoder.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start, history = state["epoch"], state["history"]
    began = time.monotonic()
    for epoch in range(start,ae_epochs):
        total = 0.
        for position in np.random.default_rng(42+epoch).permutation(len(pairs)):
            inputs, _ = pairs.read(int(position))
            x = low_maps(inputs[0],inputs[3]).detach()
            pred = autoencoder(x)
            # Reconstruct cell and tissue channels at equal task weight.
            weight = 1+9*x[:,:10]
            cell_loss = (weight*(pred[:,:10]-x[:,:10]).square()).sum()/weight.sum()
            loss = cell_loss+(pred[:,10:]-x[:,10:]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item()
        history.append({"epoch":epoch+1,"reconstruction_loss":total/len(pairs)})
        m.legacy.atomic_save({"model":autoencoder.state_dict(),"optimizer":optimizer.state_dict(),
                             "epoch":epoch+1,"history":history},path)
        print(f"[AE pretrain] {epoch+1}/{ae_epochs} loss={history[-1]['reconstruction_loss']:.6f} "
              f"elapsed={time.monotonic()-began:.1f}s",flush=True)
    m.write_csv(out/"ae_history.csv",history)

identity = {"source_run_sha256":m.sha256(source/"run.json"),"epochs":epochs,
            "ae_epochs":ae_epochs,"weight_radius":ratio,"device":str(device),
            "source_script_sha256":m.sha256(source/"m3_context.py"),
            "torch":str(torch.__version__),"methods":list(m.METHODS)}
if (out/"run.json").is_file() and json.loads((out/"run.json").read_text()) != identity:
    raise ValueError("Run settings changed. Use a new TEMP_RUN_ID.")
m.write_json(out/"run.json",identity)
m.Ensemble = Gate
cache = Path(config["cache"])
models, history = {}, []
with m.data_paths({"data_root":config["data_root"]}) as (_,_,_):
    val = m.TorchOcelotDataset(split="val")
    lookup = {r["pair_id"]:i for i,r in enumerate(val.source.rows)}
    fit_idx = [lookup[p] for p in config["fit_ids"]]
    tune_idx = [lookup[p] for p in config["tune_ids"]]
    fit_pairs = m.Pairs(val,fit_idx,"val",cache,device)
    pretrain(fit_pairs)
    targets = out/"fit_targets"
    targets.mkdir(exist_ok=True)
    for variant in m.VARIANTS:
        model, record = m.fit(variant,fit_pairs,epochs,device,targets)
        models[variant] = model
        history.extend(record)
    m.write_csv(out/"epoch_history.csv",history)
    tune_pairs = m.Pairs(val,tune_idx,"val",cache,device)
    pp = out/"postprocess.json"
    if pp.is_file():
        chosen = {n:tuple(v) for n,v in json.loads(pp.read_text())["per_method"].items()}
    else:
        chosen, grid = m.tune(tune_pairs,models)
        m.write_csv(out/"validation_grid.csv",grid)
        m.write_json(pp,m.group_settings(chosen))
    with gzip.open(out/"val_predictions.jsonl.gz","wt",encoding="utf-8") as stream:
        val_rows, val_records = m.evaluate("val_tune",tune_pairs,models,chosen,stream)
m.write_csv(out/"validation.csv",val_rows)
m.write_csv(out/"val_per_pair.csv",val_records)
candidates = [r for r in val_rows if r["policy"]=="per_method"]
selected = max(candidates,key=lambda r:r["mF1"])["method"]
frozen = {**identity,"selected_method":selected,"postprocess":m.group_settings(chosen),
          "base_sha256":config["checkpoint_sha256"],
          "gate_sha256":{n:m.sha256(out/f"{n}.pt") for n in models}}
if (out/"frozen_before_test.json").is_file() and json.loads((out/"frozen_before_test.json").read_text()) != frozen:
    raise ValueError("Frozen selection changed.")
m.write_json(out/"frozen_before_test.json",frozen)
if not (out/"test_complete.ok").is_file():
    test_audit = out/"test_audit"
    audit = test_audit/"audit"
    audit.mkdir(parents=True,exist_ok=True)
    data = Path(config["data_root"])
    metadata,_ = m.load_metadata(data/"metadata.json")
    report = {"errors":[],"warnings":[]}
    info,ids,_,_,records = m.inspect_split(data,"test",metadata,report)
    if report["errors"] or len(records)!=126:
        raise ValueError("Official test audit failed.")
    m.write_json(audit/"report.json",{"ok":True,"splits":{"test":info}})
    m.write_csv(audit/"manifest.csv",records)
    os.environ.update(OCELOT_DATA_ROOT=str(data),OUTPUT_ROOT=str(test_audit))
    test = m.TorchOcelotDataset(split="test")
    indices = list(range(len(test)))
    # Reuse the earlier test D4 cache where all arrays are available.
    prior_cache = source/"official_test_m3/cache"
    test_cache = prior_cache if all(m.cache_file(prior_cache,"test",test,i).is_file() for i in indices) else out/"test_cache"
    m.create_cache(test_cache,[("test",test,indices)],paths,device)
    pairs = m.Pairs(test,indices,"test",test_cache,device)
    with gzip.open(out/"test_predictions.jsonl.gz","wt",encoding="utf-8") as stream:
        test_rows, test_records = m.evaluate("test",pairs,models,chosen,stream)
    m.write_csv(out/"summary.csv",val_rows+test_rows)
    m.write_csv(out/"test_per_pair.csv",test_records)
    (out/"test_complete.ok").write_text("complete\n")
with (out/"summary.csv").open() as f:
    final = next(r for r in csv.DictReader(f) if r["split"]=="test" and
                 r["policy"]=="per_method" and r["method"]==selected)
m.write_csv(out/"selected_test_score.csv",[final])
(out/"README.md").write_text(
    "Three temperature-gated cell ensembles. fixed uses 10 predefined context channels. "
    "learned uses 10 convolutional context channels. ae uses 9 spatial latent channels, "
    "three per receptive-field scale, pretrained at 256x256 for five epochs by default, "
    "then jointly optimized with the gate. pointwise is the 15-channel control. "
    "All use identical five-model D4 outputs. Gate logits are bounded by tanh; "
    "the effective logit amplitude divided by temperature defaults to 0.1. "
    "Weights start at exactly 0.2 and are bounded approximately to 0.170-0.234. "
    "The three models use the same fixed training duration and postprocessing grid. "
    "Selection uses val_tune, then the frozen selections are evaluated on test. "
    "Previous exposure to this test set must be disclosed in research reporting. "
    "Original base models are reused without retraining.\n",encoding="utf-8")
print(f"[validation-selected] {selected}; test Cell mF1={float(final['mF1']):.5f}",flush=True)
PY
if [[ ${TEMP_PUBLISH:-1} == 1 ]]; then
  gh auth status --hostname github.com
  if gh auth setup-git --help >/dev/null 2>&1; then gh auth setup-git --hostname github.com; fi
  (
    cd "$root/m3_runtime/repo"
    tmp=$(mktemp -d)
    trap 'rm -rf -- "$tmp"' EXIT
    export GIT_INDEX_FILE="$tmp/index"
    branch="results/${TEMP_RUN_ID:-temperature_three_seed42}"
    ref="refs/heads/$branch"
    url=https://github.com/jgyu480/CAS2250.01-00-Medical_AI.git
    parent=$(git rev-parse HEAD)
    remote=$(git ls-remote "$url" "$ref")
    if [[ -n "$remote" ]]; then
      git fetch --no-tags "$url" "$ref"
      parent=$(git rev-parse FETCH_HEAD)
    fi
    git read-tree "$parent"
    dst="results/ensemble/${TEMP_RUN_ID:-temperature_three_seed42}"
    for name in README.md launcher.sh run.json summary.csv validation.csv validation_grid.csv postprocess.json       frozen_before_test.json selected_test_score.csv val_per_pair.csv test_per_pair.csv       epoch_history.csv ae_history.csv ae_pretrain.pt pointwise.pt fixed.pt learned.pt ae.pt; do
      test -s "$out/$name"
      blob=$(git hash-object -w -- "$out/$name")
      git update-index --add --cacheinfo "100644,$blob,$dst/$name"
    done
    tree=$(git write-tree)
    login=$(gh api user --jq .login)
    numeric_id=$(gh api user --jq .id)
    commit=$(git -c "user.name=$login" -c "user.email=$numeric_id+$login@users.noreply.github.com"       commit-tree "$tree" -p "$parent" -m 'Save temperature-gated ensemble comparison')
    git push "$url" "$commit:$ref"
    echo "[GitHub] https://github.com/jgyu480/CAS2250.01-00-Medical_AI/tree/$commit/$dst"
  )
fi
