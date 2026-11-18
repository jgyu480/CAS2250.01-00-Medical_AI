#!/usr/bin/env bash
set -euo pipefail
root=${ENSEMBLE_ROOT:-/root/ocelot_ensemble}
source_run=${M3_SOURCE_RUN:-"$root/experiments/m3_context_d4_retrained_val_seed42"}
out="$root/experiments/${CORRECT_RUN_ID:-spatial_verification_three_seed42}"
mkdir -p "$out"
exec 9>"$out/run.lock"
flock -n 9 || { echo 'Correction experiment already running.' >&2; exit 1; }
[[ "${CORRECT_RUN_ID:-spatial_verification_three_seed42}" =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$ ]] || exit 2
test -s "$source_run/complete.ok"
test -s "$source_run/m3_context.py"
if [[ -f "$out/launcher.sh" ]]; then
  cmp -s "$0" "$out/launcher.sh" || { echo 'Script changed; set a new CORRECT_RUN_ID.' >&2; exit 1; }
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
cat > "$out/experiment.py" <<'PY'
"""Three compact spatial candidate verifiers on frozen D4 prediction maps."""
import ast,csv,gzip,importlib.util,itertools,json,os,time
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
root=Path(os.environ['M3_STATE_ROOT']);source=Path(os.environ['TEMP_SOURCE_RUN']).resolve()
out=Path(os.environ['TEMP_OUTPUT']).resolve()
prior=root/'experiments'/os.environ.get('CORRECT_SOURCE_ID','ensemble_ablation_five_seed42')
os.environ['M3_OUTPUT']=str(source)
spec=importlib.util.spec_from_file_location('prior_helpers',source/'m3_context.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
config=json.loads((source/'run.json').read_text());m.OUT=out
old_config=json.loads((prior/'run.json').read_text())
frozen_old=json.loads((prior/'frozen_before_test.json').read_text())
device=torch.device(os.environ.get('M3_DEVICE','cuda'));torch.set_num_threads(2)
if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
if old_config['source_run_sha256']!=m.sha256(source/'run.json'):raise ValueError('Source run mismatch')
if m.sha256(source/'m3_context.py')!=config['implementation_sha256']:raise ValueError('Changed source helper')
for name,path in config['checkpoint_paths'].items():
    if m.sha256(Path(path))!=config['checkpoint_sha256'][name]:raise ValueError(f'Changed base {name}')
# Import only architecture definitions, never execute old training/evaluation entry points.
if m.sha256(prior/'launcher.sh')!=old_config['script_sha256']:raise ValueError('Changed original launcher')
program=(prior/'experiment.py').read_text()
embedded=(prior/'launcher.sh').read_text().split("<<'PY'",1)[1].split('\n',1)[1].split('\nPY\n',1)[0]
if program.strip()!=embedded.strip():raise ValueError('Original experiment.py no longer matches its launcher')
tree=ast.parse(program)
names={'mean_window','raw_maps','PooledEncoder','SpatialEncoder','Gate','crop_inputs'}
ns=dict(torch=torch,nn=nn,F=F,SCALES=(3,7,15,31,63,127),HALO=63,device=device)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[]),str(prior/'experiment.py'),'exec'),ns)
gates={}
for name in ('softmax','spatial'):
    path=prior/f'{name}.pt'
    if m.sha256(path)!=frozen_old['gate_sha256'][name]:raise ValueError(f'Changed gate {name}')
    gate=ns['Gate'](name).to(device)
    gate.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)['model'])
    gate.eval().requires_grad_(False);gates[name]=gate
old_pp={n:tuple(v) for n,v in frozen_old['postprocess']['per_method'].items()}
epochs=int(os.environ.get('CORRECT_EPOCHS','20'))
if epochs<1:raise ValueError('Positive epochs required')
METHODS=('tta_mean','softmax','spatial','verify','balanced','dual')
VARIANTS=('verify','balanced','dual')
PROPOSAL_PP=(3.,3,.05)
MERGE_RADIUS=3.;MATCH_RADIUS=15.
DIM=15+10+10+6+2+192
# raw15, max3-cell10, disagreement10, mean/softmax/spatial6, source flag+score2, context192.
FEATURES=out/'feature_cache';FEATURES.mkdir(exist_ok=True)
identity=dict(version=1,source_run_sha256=m.sha256(source/'run.json'),prior_run_sha256=m.sha256(prior/'run.json'),
    implementation_sha256=m.sha256(out/'experiment.py'),gate_sha256={n:m.sha256(prior/f'{n}.pt') for n in gates},
    epochs=epochs,seed=42,feature_dim=DIM,proposal_pp=list(PROPOSAL_PP),merge_radius=MERGE_RADIUS,
    matching_radius=MATCH_RADIUS,test_enabled=os.environ.get('CORRECT_TEST','1')=='1',
    prior_test_exposure=True,feature_encoder='frozen spatial RF3,7,15,31,63,127,cell16+tissue16 each')
if (out/'run.json').is_file() and json.loads((out/'run.json').read_text())!=identity:raise ValueError('Settings changed; use new CORRECT_RUN_ID')
m.write_json(out/'run.json',identity)

def detections(maps,pp):return next(m.legacy.candidate_points(maps,False,[pp]))[1]

@torch.inference_mode()
def map_features(inputs):
    cells,tissue,base,aligned,box,extent=inputs
    aligned=F.interpolate(aligned,size=cells.shape[-2:],mode='bilinear',align_corners=False)
    height,width=cells.shape[-2:]
    maps={n:torch.empty((2,height,width),device=device) for n in gates}
    # 384 MiB transient CPU buffer; only vectors at candidate locations are saved.
    contexts=np.empty((192,height,width),dtype=np.float16)
    for y in range(0,height,128):
        for x in range(0,width,128):
            y0,x0=max(0,y-63),max(0,x-63);y1,x1=min(height,y+191),min(width,x+191)
            c=cells[...,y0:y1,x0:x1];a=aligned[...,y0:y1,x0:x1]
            feats=gates['spatial'].features(c,a)
            qs=torch.logit(gates['spatial'].combine(c,feats)[0][0],eps=1e-6).sigmoid()
            qp=torch.logit(gates['softmax'].details(c,a)[0][0],eps=1e-6).sigmoid()
            sy,sx=y-y0,x-x0;hh,ww=min(128,height-y),min(128,width-x)
            region=(slice(sy,sy+hh),slice(sx,sx+ww))
            maps['spatial'][:,y:y+hh,x:x+ww]=qs[:,region[0],region[1]]
            maps['softmax'][:,y:y+hh,x:x+ww]=qp[:,region[0],region[1]]
            contexts[:,y:y+hh,x:x+ww]=feats[1][0,:,region[0],region[1]].half().cpu().numpy()
    maps['tta_mean']=base[0]
    return maps,contexts,aligned

def merge_candidates(baseline,extra):
    # Keep exact baseline coordinates first; merge only very close duplicate proposals.
    candidates=[];flags=[];classes=[];scores=[];buckets=defaultdict(list)
    ordered=[(p,True) for p in sorted(baseline,key=lambda p:-p[3])]
    ordered.extend((p,False) for p in sorted(extra,key=lambda p:-p[3]))
    for p,is_base in ordered:
        x,y,cls,score=p;bx,by=int(x//4),int(y//4)
        duplicate=False
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for k in buckets[(bx+dx,by+dy)]:
                    if (candidates[k][0]-x)**2+(candidates[k][1]-y)**2<=MERGE_RADIUS**2:duplicate=True;break
                if duplicate:break
            if duplicate:break
        if duplicate:continue
        k=len(candidates);candidates.append((int(x),int(y)));flags.append(is_base);classes.append(int(cls));scores.append(float(score))
        buckets[(bx,by)].append(k)
    return np.asarray(candidates,dtype=np.int32).reshape(-1,2),np.asarray(flags,bool),np.asarray(classes,np.int64),np.asarray(scores,np.float32)

def label_candidates(coords,is_base,scores,truth):
    # Class-agnostic localization matching supplies the correct BC/TC target.
    # One target per cell, score-ordered with existing baseline candidates first.
    labels=np.zeros(len(coords),np.int64);groups=np.full(len(coords),3,np.int64)
    groups[is_base]=1 # baseline FP; changed to group0 if matched
    if not len(coords) or not len(truth):return labels,groups
    distances=np.linalg.norm(coords[:,None,:].astype(np.float32)-truth[None,:,:2].astype(np.float32),axis=2)
    available=distances<=MATCH_RADIUS
    order=sorted(range(len(coords)),key=lambda k:(not is_base[k],-float(scores[k])))
    for k in order:
        neighbors=np.flatnonzero(available[k])
        if not len(neighbors):continue
        j=neighbors[np.argmin(distances[k,neighbors])]
        labels[k]=int(truth[j,2]);groups[k]=0 if is_base[k] else 2
        available[:,j]=False
    return labels,groups

def save_npz(path,**data):
    tmp=path.with_suffix('.part')
    with tmp.open('wb') as f:np.savez_compressed(f,**data)
    tmp.replace(path)

@torch.inference_mode()
def prepare(pairs,split):
    packs=[]
    for position,row in enumerate(pairs.rows):
        pair_id=row['pair_id'];path=FEATURES/f'{split}-{pair_id}.npz'
        if not path.is_file():
            inputs,truth=pairs.read(position);truth=np.rint(np.asarray(truth,dtype=np.float32).reshape(-1,3))
            maps,contexts,aligned=map_features(inputs)
            refs={n:detections(maps[n],old_pp[n]) for n in ('tta_mean','softmax','spatial')}
            extra=[]
            for i in range(5):extra.extend(detections(inputs[0][0,2*i:2*i+2],PROPOSAL_PP))
            for name in maps:extra.extend(detections(maps[name],PROPOSAL_PP))
            coords,flags,classes,scores=merge_candidates(refs['spatial'],extra)
            yi,xi=coords[:,1],coords[:,0]
            c=inputs[0][0];support=F.max_pool2d(c[None],3,stride=1,padding=1)[0]
            raw=torch.cat((c,aligned[0]),0).cpu().numpy()[:,yi,xi].T
            local=support.cpu().numpy()[:,yi,xi].T
            p=c.reshape(5,2,1024,1024);diff=(p-p.mean(0,keepdim=True)).flatten(0,1).cpu().numpy()[:,yi,xi].T
            fused=torch.cat([maps[n] for n in ('tta_mean','softmax','spatial')],0).cpu().numpy()[:,yi,xi].T
            context=contexts[:,yi,xi].T.copy();del contexts
            feat=np.concatenate((raw,local,diff,fused,flags[:,None].astype(np.float32),scores[:,None],context),1)
            assert feat.shape==(len(coords),DIM)
            if not np.isfinite(feat).all():raise ValueError('Nonfinite candidate features')
            labels,groups=label_candidates(coords,flags,scores,truth)
            save_npz(path,features=feat.astype(np.float16),coords=coords,flags=flags,classes=classes,scores=scores,
                labels=labels,groups=groups,truth=truth,references=np.array(json.dumps(refs,separators=(',',':'))))
            print(f'[features] {split} {position+1}/{len(pairs)} candidates={len(coords)} '
                f'base_true={sum(groups==0)} base_false={sum(groups==1)} rescue={sum(groups==2)}',flush=True)
        with np.load(path,allow_pickle=False) as d:pack={k:d[k].copy() for k in d.files}
        pack['references']=json.loads(str(pack['references']));pack['pair_id']=pair_id;packs.append(pack)
    return packs

class Verifier(nn.Module):
    def __init__(self,dual=False):
        super().__init__();self.dual=dual
        self.trunk=nn.Sequential(nn.LayerNorm(DIM),nn.Linear(DIM,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.keep=nn.Linear(32,3)
        if dual:self.rescue=nn.Linear(32,3)
    def forward(self,x,flags):
        h=self.trunk(x);k=self.keep(h)
        return torch.where(flags[:,None],k,self.rescue(h)) if self.dual else k

def fit(variant,packs):
    m.seed_everything(42)
    x=torch.from_numpy(np.concatenate([p['features'] for p in packs]).astype(np.float32)).to(device)
    y=torch.from_numpy(np.concatenate([p['labels'] for p in packs])).to(device)
    groups=torch.from_numpy(np.concatenate([p['groups'] for p in packs])).to(device)
    flags=torch.from_numpy(np.concatenate([p['flags'] for p in packs])).to(device)
    if not len(x):raise ValueError('Empty training candidate set')
    # Group-balanced training gives equal TOTAL influence to four error/keep groups.
    counts=torch.bincount(groups,minlength=4).float()
    weights=(1/counts.clamp_min(1))[groups] if variant!='verify' else torch.ones_like(y,dtype=torch.float32)
    weights=weights/weights.mean()
    net=Verifier(variant=='dual').to(device);optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    checkpoint=out/f'{variant}.pt';history=[]
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        net.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history']
    for epoch in range(len(history),epochs):
        generator=torch.Generator().manual_seed(42+epoch);order=torch.randperm(len(x),generator=generator)
        total=0.;net.train();began=time.monotonic()
        for indices in order.split(512):
            indices=indices.to(device);logits=net(x[indices],flags[indices])
            loss=(F.cross_entropy(logits,y[indices],reduction='none')*weights[indices]).mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite verifier loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=loss.item()*len(indices)
        record=dict(method=variant,epoch=epoch+1,loss=total/len(x),seconds=time.monotonic()-began)
        history.append(record)
        m.legacy.atomic_save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),history=history,
            variant=variant,epoch=epoch+1,group_counts=counts.cpu()),checkpoint)
        print(f'[{variant}] epoch {epoch+1}/{epochs} CE={record["loss"]:.6f} {record["seconds"]:.1f}s',flush=True)
    return net.eval(),history

@torch.inference_mode()
def probabilities(net,pack):
    x=torch.from_numpy(pack['features'].astype(np.float32)).to(device)
    flags=torch.from_numpy(pack['flags']).to(device)
    return torch.cat([net(x[i:i+1024],flags[i:i+1024]).softmax(1).cpu() for i in range(0,len(x),1024)]).numpy() if len(x) else np.empty((0,3),np.float32)

def decode(pack,probs,setting):
    keep_thr,rescue_thr,distance=setting
    labels=probs.argmax(1);scores=probs[:,1:].max(1)
    thresholds=np.where(pack['flags'],keep_thr,rescue_thr)
    indices=np.flatnonzero((labels>0)&(scores>=thresholds))
    order=indices[np.argsort(-scores[indices],kind='quicksort')]
    tree=cKDTree(pack['coords']) if len(pack['coords']) else None
    suppressed=np.zeros(len(pack['coords']),bool);result=[]
    for k in order:
        if suppressed[k]:continue
        x,y=pack['coords'][k];result.append([int(x),int(y),int(labels[k]),float(scores[k])])
        suppressed[tree.query_ball_point((x,y),r=float(distance))]=True
    return result

def matched_ids(pred,truth):
    result=set()
    for cls in (1,2):
        pp=np.asarray([p for p in pred if p[2]==cls],np.float32).reshape(-1,4)
        ii=np.flatnonzero(truth[:,2]==cls);gt=truth[ii,:2]
        if not len(pp) or not len(gt):continue
        dist=np.linalg.norm(pp[:,None,:2]-gt[None],axis=2);available=dist<=15
        for i in np.argsort(-pp[:,3],kind='quicksort'):
            neighbors=np.flatnonzero(available[i])
            if not len(neighbors):continue
            j=neighbors[np.argmin(dist[i,neighbors])];result.add(int(ii[j]));available[:,j]=False
    return result

def metric_row(metric):return {**metric.compute(),**m.metric_counts(metric)}

def tune(packs,probs):
    settings=list(itertools.product((.2,.35,.5,.65,.8),(.4,.55,.7,.85,.95),(3,5,7)))
    chosen={};rows=[]
    for method in VARIANTS:
        best=None;best_f1=-1
        for setting in settings:
            metric=m.CellF1()
            for p,prob in zip(packs,probs[method]):metric.update(decode(p,prob,setting),p['truth'])
            score=metric.compute()['mF1']
            rows.append(dict(method=method,keep_threshold=setting[0],rescue_threshold=setting[1],min_distance=setting[2],**metric_row(metric)))
            if score>best_f1:best,best_f1=setting,score
        chosen[method]=best
        print(f'[validation] {method}: mF1={best_f1:.6f}, setting={best}',flush=True)
    return chosen,rows

def evaluate(split,packs,probs,chosen):
    totals={n:m.CellF1() for n in METHODS};paired=[];actions=[]
    with gzip.open(out/f'{split}_predictions.jsonl.gz','wt') as stream:
        for i,p in enumerate(packs):
            predictions=dict(p['references'])
            predictions.update({n:decode(p,probs[n][i],chosen[n]) for n in VARIANTS})
            base_ids=matched_ids(predictions['spatial'],p['truth'])
            for name,pred in predictions.items():
                metric=m.CellF1();metric.update(pred,p['truth']);totals[name].update(pred,p['truth'])
                ids=matched_ids(pred,p['truth'])
                paired.append(dict(split=split,pair_id=p['pair_id'],method=name,**metric_row(metric),
                    rescued_vs_spatial=len(ids-base_ids),lost_vs_spatial=len(base_ids-ids)))
                stream.write(json.dumps(dict(pair_id=p['pair_id'],method=name,predictions=pred),separators=(',',':'))+'\n')
            oracle=int(sum(p['labels']>0));total=len(p['truth'])
            actions.append(dict(split=split,pair_id=p['pair_id'],GT=total,candidates=len(p['coords']),
                candidate_matched_GT=oracle,candidate_coverage=oracle/total if total else None,
                baseline_localized=int(np.count_nonzero(p['groups']==0)),baseline_false=int(np.count_nonzero(p['groups']==1)),
                recoverable_misses=int(np.count_nonzero(p['groups']==2))))
    rows=[]
    for name,metric in totals.items():
        related=[r for r in paired if r['method']==name]
        row=dict(split=split,policy='validation_frozen',method=name,**metric_row(metric),
            rescued_vs_spatial=sum(r['rescued_vs_spatial'] for r in related),lost_vs_spatial=sum(r['lost_vs_spatial'] for r in related))
        rows.append(row);print(f'[result] {split} {name:15s} Cell mF1={row["mF1"]:.6f} '
            f'FP={row["FP_BC"]+row["FP_TC"]} rescue={row["rescued_vs_spatial"]} lost={row["lost_vs_spatial"]}',flush=True)
    m.write_csv(out/f'{split}_per_pair.csv',paired);m.write_csv(out/f'{split}_candidate_audit.csv',actions)
    return rows

cache=Path(config['cache']);models={};history=[]
with m.data_paths({'data_root':config['data_root']}):
    val=m.TorchOcelotDataset(split='val');lookup={r['pair_id']:i for i,r in enumerate(val.source.rows)}
    fit_pairs=m.Pairs(val,[lookup[p] for p in config['fit_ids']],'val',cache,device)
    tune_pairs=m.Pairs(val,[lookup[p] for p in config['tune_ids']],'val',cache,device)
    print('[stage] extracting frozen spatial vectors for fit',flush=True);train=prepare(fit_pairs,'fit')
    # Release base maps/feature encoders only after tune/test extraction; networks stay frozen.
    for name in VARIANTS:
        models[name],record=fit(name,train);history.extend(record);m.write_csv(out/'epoch_history.csv',history)
    del train
    print('[stage] extracting validation vectors',flush=True);validation=prepare(tune_pairs,'val_tune')
    val_probs={n:[probabilities(net,p) for p in validation] for n,net in models.items()}
    if (out/'postprocess.json').is_file():chosen={n:tuple(s) for n,s in json.loads((out/'postprocess.json').read_text()).items()}
    else:
        chosen,grid=tune(validation,val_probs);m.write_csv(out/'validation_grid.csv',grid);m.write_json(out/'postprocess.json',chosen)
    val_rows=evaluate('val_tune',validation,val_probs,chosen)
m.write_csv(out/'validation.csv',val_rows)
selected=max(val_rows,key=lambda r:r['mF1'])['method']
frozen=dict(identity,selected_method=selected,postprocess=chosen,baseline_postprocess={n:old_pp[n] for n in ('tta_mean','softmax','spatial')},
    verifier_sha256={n:m.sha256(out/f'{n}.pt') for n in VARIANTS})
# JSON roundtrip normalizes tuple values before restart comparison.
frozen=json.loads(json.dumps(frozen))
if (out/'frozen_before_test.json').is_file() and json.loads((out/'frozen_before_test.json').read_text())!=frozen:raise ValueError('Frozen settings changed')
m.write_json(out/'frozen_before_test.json',frozen)
test_rows=[]
if identity['test_enabled']:
    audit_root=out/'test_audit';(audit_root/'audit').mkdir(parents=True,exist_ok=True)
    data=Path(config['data_root']);metadata,_=m.load_metadata(data/'metadata.json');report={'errors':[],'warnings':[]}
    info,ids,_,_,records=m.inspect_split(data,'test',metadata,report)
    if report['errors'] or len(records)!=126:raise ValueError('Official test audit failed')
    m.write_json(audit_root/'audit/report.json',{'ok':True,'splits':{'test':info}});m.write_csv(audit_root/'audit/manifest.csv',records)
    os.environ.update(OCELOT_DATA_ROOT=str(data),OUTPUT_ROOT=str(audit_root))
    test=m.TorchOcelotDataset(split='test');indices=list(range(len(test)));tc=source/'official_test_m3/cache'
    if not all(m.cache_file(tc,'test',test,i).is_file() for i in indices):raise FileNotFoundError('Missing D4 test cache')
    packs=prepare(m.Pairs(test,indices,'test',tc,device),'test')
    test_probs={n:[probabilities(net,p) for p in packs] for n,net in models.items()}
    test_rows=evaluate('test',packs,test_probs,chosen)
    final=next(r for r in test_rows if r['method']==selected)
    m.write_csv(out/'selected_test_score.csv',[final]);print(f'[validation-selected] {selected}; test Cell mF1={final["mF1"]:.6f}',flush=True)
m.write_csv(out/'summary.csv',val_rows+test_rows)
(out/'README.md').write_text('''# Three simple spatial candidate verifiers

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
''',encoding='utf-8')
(out/'complete.ok').write_text('complete\n')

PY
"${M3_PYTHON:-$default_python}" -u "$out/experiment.py" 2>&1 | tee -a "$out/console.log"
if [[ ${CORRECT_PUBLISH:-1} == 1 ]]; then
  gh auth status --hostname github.com
  if gh auth setup-git --help >/dev/null 2>&1; then gh auth setup-git --hostname github.com; fi
  (
    cd "$root/m3_runtime/repo"
    tmp=$(mktemp -d)
    trap 'rm -rf -- "$tmp"' EXIT
    export GIT_INDEX_FILE="$tmp/index"
    branch="results/${CORRECT_RUN_ID:-spatial_verification_three_seed42}"
    ref="refs/heads/$branch"
    url=https://github.com/jgyu480/CAS2250.01-00-Medical_AI.git
    parent=$(git rev-parse HEAD)
    remote=$(git ls-remote "$url" "$ref")
    if [[ -n "$remote" ]]; then
      git fetch --no-tags "$url" "$ref"
      parent=$(git rev-parse FETCH_HEAD)
    fi
    git read-tree "$parent"
    dst="results/ensemble/${CORRECT_RUN_ID:-spatial_verification_three_seed42}"
    for name in README.md launcher.sh experiment.py run.json summary.csv validation.csv validation_grid.csv postprocess.json frozen_before_test.json val_tune_per_pair.csv val_tune_candidate_audit.csv epoch_history.csv verify.pt balanced.pt dual.pt; do
      test -s "$out/$name"
      blob=$(git hash-object -w -- "$out/$name")
      git update-index --add --cacheinfo "100644,$blob,$dst/$name"
    done
    for name in test_per_pair.csv test_candidate_audit.csv selected_test_score.csv; do
      if [[ -s "$out/$name" ]]; then
        blob=$(git hash-object -w -- "$out/$name")
        git update-index --add --cacheinfo "100644,$blob,$dst/$name"
      fi
    done
    tree=$(git write-tree)
    login=$(gh api user --jq .login)
    numeric_id=$(gh api user --jq .id)
    commit=$(git -c "user.name=$login" -c "user.email=$numeric_id+$login@users.noreply.github.com"       commit-tree "$tree" -p "$parent" -m 'Save three spatial candidate-verification experiments')
    git push "$url" "$commit:$ref"
    echo "[GitHub] https://github.com/jgyu480/CAS2250.01-00-Medical_AI/tree/$commit/$dst"
  )
fi
