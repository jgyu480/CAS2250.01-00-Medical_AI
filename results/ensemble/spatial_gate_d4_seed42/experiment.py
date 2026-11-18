"""Inference-only D4 symmetrization of five frozen spatial ensemble checkpoints."""
import ast,copy,csv,gzip,importlib.util,json,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
root=Path(os.environ['M3_STATE_ROOT']);source=Path(os.environ['TEMP_SOURCE_RUN']).resolve()
out=Path(os.environ['TEMP_OUTPUT']).resolve()
prior=root/'experiments'/os.environ.get('KERNEL_SOURCE_ID','ensemble_ablation_five_seed42')
os.environ['M3_OUTPUT']=str(source)
spec=importlib.util.spec_from_file_location('prior_helpers',source/'m3_context.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.OUT=out
config=json.loads((source/'run.json').read_text());old=json.loads((prior/'run.json').read_text())
frozen_old=json.loads((prior/'frozen_before_test.json').read_text())
device=torch.device(os.environ.get('M3_DEVICE','cuda'));torch.set_num_threads(2)
if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
if old['source_run_sha256']!=m.sha256(source/'run.json'):raise ValueError('Different base source run')
if m.sha256(source/'m3_context.py')!=config['implementation_sha256']:raise ValueError('Changed source helpers')
for name,path in config['checkpoint_paths'].items():
    if m.sha256(Path(path))!=config['checkpoint_sha256'][name]:raise ValueError(f'Changed base checkpoint: {name}')
if m.sha256(prior/'launcher.sh')!=old['script_sha256']:raise ValueError('Changed original launcher')
program=(prior/'experiment.py').read_text()
embedded=(prior/'launcher.sh').read_text().split("<<'PY'",1)[1].split('\n',1)[1].split('\nPY\n',1)[0]
if program.strip()!=embedded.strip():raise ValueError('Changed original experiment.py')
tree=ast.parse(program)
names={'mean_window','raw_maps','PooledEncoder','SpatialEncoder','Gate','crop_inputs'}
ns=dict(torch=torch,nn=nn,F=F,SCALES=(3,7,15,31,63,127),HALO=63,device=device)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names],type_ignores=[]),str(prior/'experiment.py'),'exec'),ns)
BaseGate=ns['Gate'];SpatialEncoder=ns['SpatialEncoder'];crop_inputs=ns['crop_inputs']
path=prior/'spatial.pt'
if m.sha256(path)!=frozen_old['gate_sha256']['spatial']:raise ValueError('Changed original spatial checkpoint')
initial=torch.load(path,map_location='cpu',weights_only=True)['model']
SCALES=(3,7,15,31,63,127);HALO=63;CROP=128;CROPS=4
variants=('spatial_ft','scale_select','local_detail','disagreement')
m.METHODS=('tta_mean','spatial',*variants)
epochs=int(os.environ.get('KERNEL_EPOCHS','12'));lr=float(os.environ.get('KERNEL_LR','0.0001'))
if epochs<1 or not 0<lr<=.001:raise ValueError('Require positive epochs and learning rate <=.001')

class ScaleSelect(nn.Module):
    """Per-pixel scale attention; preserve the 192-channel concat and identity init."""
    def __init__(self,base):
        super().__init__();self.base=base
        self.cell_select=nn.Conv2d(96,6,1);self.tissue_select=nn.Conv2d(96,6,1)
        for head in (self.cell_select,self.tissue_select):nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
    def attention(self,f):
        parts=f.reshape(f.shape[0],6,2,16,*f.shape[-2:])
        c=parts[:,:,0].flatten(1,2);t=parts[:,:,1].flatten(1,2)
        return self.cell_select(c).softmax(1),self.tissue_select(t).softmax(1)
    def forward(self,x):
        f=self.base(x);ac,at=self.attention(f)
        # Six scales stay concatenated; 6*uniform attention equals one at init.
        weights=torch.stack((ac,at),2)[:,:,:,None]*6
        return (f.reshape(f.shape[0],6,2,16,*f.shape[-2:])*weights).flatten(1,3)

class DetailBlock(nn.Module):
    def __init__(self,original):
        super().__init__();self.wide=original[0];self.activation=original[1]
        self.local=nn.Conv2d(self.wide.in_channels,self.wide.out_channels,3,padding=1,padding_mode='replicate')
        nn.init.zeros_(self.local.weight);nn.init.zeros_(self.local.bias)
    def forward(self,x):return self.activation(self.wide(x)+self.local(x))

def add_local_detail(context):
    # Keep the first dilation1 layer; add dense local paths at dilation2..32.
    for branch in (context.cell,context.tissue):
        for i in range(1,len(branch)):branch[i]=DetailBlock(branch[i])
    return context

class DisagreementContext(nn.Module):
    def __init__(self,base):
        super().__init__();self.base=base;self.difference=SpatialEncoder()
        self.strength=nn.Parameter(torch.zeros(1,6,2,1,1,1))
    def forward(self,x):
        p=x[:,:10].reshape(x.shape[0],5,2,*x.shape[-2:]);t=x[:,10:]
        difference=torch.cat(((p-p.mean(1,keepdim=True)).flatten(1,2),t-t.mean(1,keepdim=True)),1)
        main=self.base(x).reshape(x.shape[0],6,2,16,*x.shape[-2:])
        extra=self.difference(difference).reshape_as(main)
        return (main+self.strength*extra).flatten(1,3)

# This module wrapper retains BaseGate's exact feature/head/normalization/fusion code.
def build(variant):
    m.seed_everything(42)
    model=BaseGate('spatial');model.load_state_dict(initial)
    if variant=='scale_select':model.context=ScaleSelect(model.context)
    elif variant=='local_detail':model.context=add_local_detail(model.context)
    elif variant=='disagreement':model.context=DisagreementContext(model.context)
    elif variant not in ('spatial_ft','spatial'):raise ValueError(variant)
    return model.to(device)


kernel=root/'experiments'/os.environ.get('GATE_D4_SOURCE_ID','spatial_kernel_three_seed42')
if not (kernel/'complete.ok').is_file():raise FileNotFoundError(f'Finish kernel run first: {kernel}')
kr=json.loads((kernel/'run.json').read_text());kf=json.loads((kernel/'frozen_before_test.json').read_text())
if kr['source_run_sha256']!=m.sha256(source/'run.json') or kr['original_spatial_sha256']!=m.sha256(prior/'spatial.pt'):
    raise ValueError('Kernel/source checkpoint mismatch')
if m.sha256(kernel/'launcher.sh')!=kr['script_sha256']:raise ValueError('Changed kernel launcher')
models={'spatial':build('spatial').eval().requires_grad_(False)}
for name in variants:
    checkpoint=kernel/f'{name}.pt'
    if m.sha256(checkpoint)!=kf['adapted_sha256'][name]:raise ValueError(f'Changed kernel checkpoint: {name}')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=build(name);model.load_state_dict(state['model']);models[name]=model.eval().requires_grad_(False)
m.METHODS=('tta_mean',*models,*(name+'_d4' for name in models))
identity=dict(version=1,source_run_sha256=m.sha256(source/'run.json'),kernel_run_sha256=m.sha256(kernel/'run.json'),
    checkpoint_sha256={'spatial':m.sha256(prior/'spatial.pt'),**{n:m.sha256(kernel/f'{n}.pt') for n in variants}},
    script_sha256=m.sha256(out/'launcher.sh'),methods=list(m.METHODS),test_enabled=os.environ.get('GATE_D4_TEST','1')=='1',
    inference='base-model D4 cached maps, followed by joint spatial-gate D4 probability average',retraining=False,
    prior_test_exposure=True,tile_core=128,halo=63)
if (out/'run.json').is_file() and json.loads((out/'run.json').read_text())!=identity:raise ValueError('Settings changed; use new GATE_D4_RUN_ID')
m.write_json(out/'run.json',identity)


def transform(x,k,flip):
    x=torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if flip else x


def inverse(x,k,flip):
    if flip:x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))


@torch.inference_mode()
def predict_pair(model,cells,aligned,core=128):
    # Resize tissue ONCE in native coordinates, then transform it jointly with cells.
    aligned=F.interpolate(aligned,size=cells.shape[-2:],mode='bilinear',align_corners=False)
    height,width=cells.shape[-2:]
    single=torch.empty((cells.shape[0],2,height,width),device=cells.device,dtype=torch.float32)
    average=torch.empty_like(single)
    for y in range(0,height,core):
        for x in range(0,width,core):
            y0,x0=max(0,y-HALO),max(0,x-HALO)
            y1,x1=min(height,y+core+HALO),min(width,x+core+HALO)
            c=cells[...,y0:y1,x0:x1];a=aligned[...,y0:y1,x0:x1]
            hh,ww=min(core,height-y),min(core,width-x)
            cut=(slice(None),slice(None),slice(y-y0,y-y0+hh),slice(x-x0,x-x0+ww))
            total=torch.zeros((cells.shape[0],2,hh,ww),device=cells.device,dtype=torch.float32)
            for flip in (False,True):
                for k in range(4):
                    q=model.details(transform(c,k,flip),transform(a,k,flip))[0]
                    # Match previous forward()->sigmoid(), including numerical clipping.
                    q=torch.logit(q,eps=1e-6).sigmoid()
                    restored=inverse(q,k,flip)[cut]
                    if k==0 and not flip:single[...,y:y+hh,x:x+ww]=restored
                    total.add_(restored.float())
            average[...,y:y+hh,x:x+ww]=total/8
    return single,average


@torch.inference_mode()
def output_maps(models,inputs):
    yield 'tta_mean',inputs[2][0]
    for name,model in models.items():
        start=time.monotonic();single,average=predict_pair(model,inputs[0],inputs[3])
        print(f'[gate D4] {name}: {time.monotonic()-start:.1f}s',flush=True)
        yield name,single[0]
        yield name+'_d4',average[0]
m.output_maps=output_maps

def matched_ids(pred,truth):
    result=set()
    for cls in (1,2):
        pp=np.asarray([p for p in pred if p[2]==cls],np.float32).reshape(-1,4)
        ii=np.flatnonzero(truth[:,2]==cls);gt=truth[ii,:2]
        if not len(pp) or not len(gt):continue
        distance=np.linalg.norm(pp[:,None,:2]-gt[None],axis=2);available=distance<=15
        for i in np.argsort(-pp[:,3],kind='quicksort'):
            neighbors=np.flatnonzero(available[i])
            if not len(neighbors):continue
            j=neighbors[np.argmin(distance[i,neighbors])];result.add(int(ii[j]));available[:,j]=False
    return result

@torch.inference_mode()
def evaluate(split,pairs,models,chosen,stream):
    policies=m.group_settings(chosen)
    totals={policy:{n:m.CellF1() for n in m.METHODS} for policy in policies};records=[]
    for position,row in enumerate(pairs.rows):
        inputs,points=pairs.read(position);truth=np.rint(points).astype(np.int32)
        all_predictions={policy:{} for policy in policies}
        for name,maps in output_maps(models,inputs):
            parameters=sorted({settings[name] for settings in policies.values()})
            candidates=dict(m.legacy.candidate_points(maps,False,parameters))
            for policy,settings in policies.items():all_predictions[policy][name]=candidates[settings[name]]
        for policy,predictions in all_predictions.items():
            base_ids=matched_ids(predictions['spatial'],truth)
            for name,pred in predictions.items():
                totals[policy][name].update(pred,truth);pm=m.CellF1();pm.update(pred,truth);ids=matched_ids(pred,truth)
                r=dict(split=split,policy=policy,method=name,pair_id=row['pair_id'],**pm.compute(),**m.metric_counts(pm),
                    rescued_vs_spatial=len(ids-base_ids),lost_vs_spatial=len(base_ids-ids));records.append(r)
                stream.write(json.dumps(dict(split=split,policy=policy,method=name,pair_id=row['pair_id'],predictions=pred),separators=(',',':'))+'\n')
        if (position+1)%10==0 or position+1==len(pairs):print(f'[metrics] {split} {position+1}/{len(pairs)}',flush=True)
    rows=[]
    for policy,metrics in totals.items():
        for name,metric in metrics.items():
            related=[r for r in records if r['policy']==policy and r['method']==name]
            r=dict(split=split,policy=policy,method=name,**metric.compute(),**m.metric_counts(metric),
                rescued_vs_spatial=sum(x['rescued_vs_spatial'] for x in related),lost_vs_spatial=sum(x['lost_vs_spatial'] for x in related),
                delta_vs_spatial=metric.compute()['mF1']-metrics['spatial'].compute()['mF1'],**m.setting_fields(policies[policy][name]))
            rows.append(r);print(f'[result] {split} {policy:10s} {name:14s} mF1={r["mF1"]:.6f} FP={r["FP_BC"]+r["FP_TC"]} '
                f'rescue={r["rescued_vs_spatial"]} lost={r["lost_vs_spatial"]}',flush=True)
    return rows,records
cache=Path(config['cache'])
with m.data_paths({'data_root':config['data_root']}):
    val=m.TorchOcelotDataset(split='val');lookup={r['pair_id']:i for i,r in enumerate(val.source.rows)}
    tune_pairs=m.Pairs(val,[lookup[p] for p in config['tune_ids']],'val',cache,device)
    if (out/'postprocess.json').is_file():chosen={n:tuple(v) for n,v in json.loads((out/'postprocess.json').read_text())['per_method'].items()}
    else:
        chosen,grid=m.tune(tune_pairs,models);m.write_csv(out/'validation_grid.csv',grid);m.write_json(out/'postprocess.json',m.group_settings(chosen))
    with gzip.open(out/'val_predictions.jsonl.gz','wt') as stream:val_rows,records=evaluate('val_tune',tune_pairs,models,chosen,stream)
m.write_csv(out/'validation.csv',val_rows);m.write_csv(out/'val_per_pair.csv',records)
selected=max((r for r in val_rows if r['policy']=='per_method'),key=lambda r:r['mF1'])['method']
frozen=json.loads(json.dumps(dict(identity,selected_method=selected,postprocess=m.group_settings(chosen),
    checkpoint_sha256=identity['checkpoint_sha256'])))
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
    if not all(m.cache_file(tc,'test',test,i).is_file() for i in indices):raise FileNotFoundError('Incomplete D4 test cache')
    with gzip.open(out/'test_predictions.jsonl.gz','wt') as stream:test_rows,records=evaluate('test',m.Pairs(test,indices,'test',tc,device),models,chosen,stream)
    m.write_csv(out/'test_per_pair.csv',records)
    final=next(r for r in test_rows if r['policy']=='per_method' and r['method']==selected)
    m.write_csv(out/'selected_test_score.csv',[final]);print(f'[validation-selected] {selected}; test Cell mF1={final["mF1"]:.6f}',flush=True)
m.write_csv(out/'summary.csv',val_rows+test_rows)

(out/'README.md').write_text("""# Frozen spatial ensembles with final-stage D4 TTA

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
""",encoding='utf-8')
(out/'complete.ok').write_text('complete\n')

