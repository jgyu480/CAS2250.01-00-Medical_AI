#!/usr/bin/env bash
set -euo pipefail
root=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
source_run=${M3_SOURCE_RUN:-"$root/experiments/m3_context_d4_retrained_val_seed42"}
out="$root/experiments/${TEMP_RUN_ID:-context_3_7_15_31_63_127_seed42}"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo 'Temperature experiment already running.' >&2; exit 1; }
[[ "${TEMP_RUN_ID:-context_3_7_15_31_63_127_seed42}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || exit 2
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

SCALES=(3,7,15,31,63,127)
VECTOR_SIZE=16
HALO=63
CROP=128
CROPS_PER_IMAGE=4

def mean_window(x,k):
    if k==1:
        return x
    p=k//2
    x=F.avg_pool2d(F.pad(x,(p,p,0,0),mode="replicate"),(1,k),stride=1)
    return F.avg_pool2d(F.pad(x,(0,0,p,p),mode="replicate"),(k,1),stride=1)

class ContextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.Sequential(nn.Conv2d(10,16,3,padding=1,padding_mode="replicate"),nn.ReLU())
        self.tissue=nn.Sequential(nn.Conv2d(5,16,3,padding=1,padding_mode="replicate"),nn.ReLU())

    def forward(self,x):
        cell=self.cell(x[:,:10])
        tissue=self.tissue(x[:,10:])
        return torch.cat([feature for side in SCALES
                          for feature in (mean_window(cell,side-2),
                                          mean_window(tissue,side-2))],dim=1)

class MultiscaleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=ContextEncoder()
        self.decoder=nn.Sequential(nn.Conv2d(192,32,1),nn.ReLU(),nn.Conv2d(32,15,1),nn.Sigmoid())

    def forward(self,x):
        return self.decoder(self.encoder(x))

def raw_maps(cells,aligned):
    return torch.cat((cells,F.interpolate(aligned,size=cells.shape[-2:],
                                         mode="bilinear",align_corners=False)),dim=1)

class Gate(nn.Module):
    def __init__(self,variant):
        super().__init__()
        self.variant=variant
        if variant=="fixed":
            channels=15+len(SCALES)*3*2
        elif variant in ("learned","ae"):
            channels=207
            self.context=ContextEncoder()
            if variant=="ae":
                saved=torch.load(out/"ae_pretrain.pt",map_location="cpu",weights_only=True)
                self.context.load_state_dict({key.removeprefix("encoder."):value
                    for key,value in saved["model"].items() if key.startswith("encoder.")})
        else:
            channels=15
        self.head=nn.Sequential(nn.Conv2d(channels,16,1),nn.ReLU(),nn.Conv2d(16,10,1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self,cells,tissue,base,aligned,box,extent):
        raw=raw_maps(cells,aligned)
        features=[raw]
        if self.variant=="fixed":
            mean_cell=cells.reshape(1,5,2,*cells.shape[-2:]).mean(dim=1)
            ca=raw[:,10:].mean(dim=1,keepdim=True)
            x=torch.cat((mean_cell,ca),dim=1)
            for side in SCALES:
                avg=mean_window(x,side)
                std=(mean_window(x.square(),side)-avg.square()).clamp_min(1e-8).sqrt()
                features.extend((avg,std))
        elif self.variant in ("learned","ae"):
            features.append(self.context(raw))
        logits=self.head(torch.cat(features,dim=1)).reshape(1,5,2,*cells.shape[-2:])
        weights=(ratio*logits.tanh()).softmax(dim=1)
        score=(weights*cells.reshape_as(weights)).sum(dim=1)
        return torch.logit(score,eps=1e-6)

def crop_inputs(inputs,y,x,size):
    cells,tissue,base,aligned,box,extent=inputs
    aligned=F.interpolate(aligned,size=cells.shape[-2:],mode="bilinear",align_corners=False)
    region=(slice(y,y+size),slice(x,x+size))
    return (cells[...,region[0],region[1]],tissue,base[...,region[0],region[1]],
            aligned[...,region[0],region[1]],box,extent)

def pretrain(pairs):
    m.seed_everything(42)
    net=MultiscaleAE().to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    checkpoint=out/"ae_pretrain.pt"
    history=[]
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        net.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history=state["history"]
    began=time.monotonic()
    for epoch in range(len(history),ae_epochs):
        rng=np.random.default_rng(42+epoch)
        torch.manual_seed(42+epoch)
        total=0.
        for position in rng.permutation(len(pairs)):
            inputs,_=pairs.read(int(position))
            for _ in range(CROPS_PER_IMAGE):
                y,x=rng.integers(0,1024-CROP-2*HALO+1,size=2)
                patch=crop_inputs(inputs,int(y),int(x),CROP+2*HALO)
                clean=raw_maps(patch[0],patch[3]).detach()
                corrupted=(clean+torch.randn_like(clean)*.03).clamp(0,1)
                corrupted=corrupted*(torch.rand_like(clean)>.1)
                pred=net(corrupted)[...,HALO:HALO+CROP,HALO:HALO+CROP]
                target=clean[...,HALO:HALO+CROP,HALO:HALO+CROP]
                weight=1+9*target[:,:10]
                loss=((pred[:,:10]-target[:,:10]).square()*weight).sum()/weight.sum()
                loss=loss+(pred[:,10:]-target[:,10:]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total+=loss.item()
        history.append({"epoch":epoch+1,"reconstruction_loss":total/(len(pairs)*CROPS_PER_IMAGE)})
        m.legacy.atomic_save({"model":net.state_dict(),"optimizer":optimizer.state_dict(),
                             "epoch":epoch+1,"history":history},checkpoint)
        print(f"[AE pretrain] {epoch+1}/{ae_epochs} loss={history[-1]['reconstruction_loss']:.6f} "
              f"elapsed={time.monotonic()-began:.1f}s",flush=True)
    m.write_csv(out/"ae_history.csv",history)

def fit(variant,pairs,epochs,device,targets_dir):
    m.seed_everything(42)
    model=Gate(variant).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    checkpoint=out/f"{variant}.pt"
    history=[]
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history=state["history"]
    for epoch in range(len(history),epochs):
        model.train()
        generator=torch.Generator().manual_seed(42+epoch)
        rng=np.random.default_rng(42+epoch)
        total=0.
        for position in rng.permutation(len(pairs)):
            inputs,points=pairs.read(int(position))
            target_path=targets_dir/f"{pairs.rows[position]['pair_id']}.npy"
            if not target_path.is_file():
                target=m.points_to_heatmaps(points,6.,0.,mode="gaussian")
                m.legacy.save_array(target_path,target.astype(np.float16))
            target=torch.from_numpy(np.load(target_path,allow_pickle=False)).to(device,dtype=torch.float32)
            for _ in range(CROPS_PER_IMAGE):
                y,x=rng.integers(0,1024-CROP-2*HALO+1,size=2)
                patch=crop_inputs(inputs,int(y),int(x),CROP+2*HALO)
                logits=model(*patch)[...,HALO:HALO+CROP,HALO:HALO+CROP]
                truth=target[:,y+HALO:y+HALO+CROP,x+HALO:x+HALO+CROP]
                base=patch[2][...,HALO:HALO+CROP,HALO:HALO+CROP]
                loss=m.sampled_loss(logits,base,truth,generator)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total+=loss.item()
        history.append({"method":variant,"epoch":epoch+1,"weighted_BCE":total/(len(pairs)*CROPS_PER_IMAGE)})
        m.legacy.atomic_save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                             "history":history,"epoch":epoch+1,"variant":variant},checkpoint)
        print(f"[{variant}] epoch {epoch+1}/{epochs} weighted BCE={history[-1]['weighted_BCE']:.6f}",flush=True)
    return model.eval(),history

@torch.inference_mode()
def output_maps(models,inputs):
    yield "tta_mean",inputs[2][0]
    for name,model in models.items():
        height,width=inputs[0].shape[-2:]
        result=torch.empty((1,2,height,width),device=device)
        for y in range(0,height,128):
            for x in range(0,width,128):
                y0,x0=max(0,y-HALO),max(0,x-HALO)
                y1,x1=min(height,y+128+HALO),min(width,x+128+HALO)
                patch=crop_inputs(inputs,y0,x0,max(y1-y0,x1-x0))
                # Boundary tiles can be rectangular after cropping.
                h,w=y1-y0,x1-x0
                patch=(patch[0][...,:h,:w],patch[1],patch[2][...,:h,:w],
                       patch[3][...,:h,:w],patch[4],patch[5])
                pred=model(*patch).sigmoid()
                hh,ww=min(128,height-y),min(128,width-x)
                result[...,y:y+hh,x:x+ww]=pred[...,y-y0:y-y0+hh,x-x0:x-x0+ww]
        yield name,result[0]

identity = {"scales":list(SCALES),"vector_size":VECTOR_SIZE,"crop":CROP,"crops_per_image":CROPS_PER_IMAGE,
            "source_run_sha256":m.sha256(source/"run.json"),"epochs":epochs,
            "ae_epochs":ae_epochs,"weight_radius":ratio,"device":str(device),
            "source_script_sha256":m.sha256(source/"m3_context.py"),
            "torch":str(torch.__version__),"methods":list(m.METHODS)}
if (out/"run.json").is_file() and json.loads((out/"run.json").read_text()) != identity:
    raise ValueError("Run settings changed. Use a new TEMP_RUN_ID.")
m.write_json(out/"run.json",identity)
m.Ensemble = Gate
m.fit = fit
m.output_maps = output_maps
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
    "Dense context windows 3,7,15,31,63,127 in cell-image coordinates. "
    "Cell and tissue branches each apply spatial 3x3 convolution into 16 channels, "
    "then sliding average pooling with kernel side-2 to produce the requested receptive fields. "
    "Each scale yields 16 cell plus 16 tissue dimensions. "
    "The 192 context features concatenate with 15 original maps. "
    "AE uses the same CNN encoder pretrained by noisy-input reconstruction. "
    "All vectors are evaluated at each pixel, using inference tiles with a 63-pixel halo. "
    "Supervised training uses four 128x128 central crops per image per epoch with that same halo. "
    "Source tissue maps retain the existing 256-resolution limitation. "
    "The original G/GA and D4 caches are reused. "
    "Validation freezes postprocessing before test evaluation. Prior test exposure applies.\n",encoding="utf-8")
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
    branch="results/${TEMP_RUN_ID:-context_3_7_15_31_63_127_seed42}"
    ref="refs/heads/$branch"
    url=https://github.com/jgyu480/CAS2250.01-00-Medical_AI.git
    parent=$(git rev-parse HEAD)
    remote=$(git ls-remote "$url" "$ref")
    if [[ -n "$remote" ]]; then
      git fetch --no-tags "$url" "$ref"
      parent=$(git rev-parse FETCH_HEAD)
    fi
    git read-tree "$parent"
    dst="results/ensemble/${TEMP_RUN_ID:-context_3_7_15_31_63_127_seed42}"
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
