"""Small context-encoder changes; keep spatial's pixelwise weighted-map fusion."""
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

identity=dict(version=1,source_run_sha256=m.sha256(source/'run.json'),original_spatial_sha256=m.sha256(path),
    original_postprocess=frozen_old['postprocess']['per_method'],script_sha256=m.sha256(out/'launcher.sh'),
    epochs=epochs,learning_rate=lr,optimizer='AdamW',seed=42,scales=list(SCALES),crop=128,crops_per_image=4,
    methods=list(m.METHODS),test_enabled=os.environ.get('KERNEL_TEST','1')=='1',prior_test_exposure=True,
    objective='unchanged sampled weighted BCE to sigma6 Gaussian',initialization='exact original spatial predictions')
if (out/'run.json').is_file() and json.loads((out/'run.json').read_text())!=identity:raise ValueError('Settings changed; use new KERNEL_RUN_ID')
m.write_json(out/'run.json',identity)

@torch.inference_mode()
def check_initialization():
    c=torch.rand((1,10,130,138),device=device);a=torch.rand((1,5,33,35),device=device)
    base=build('spatial').eval();reference=base.details(c,a)[0]
    rows=[]
    for name in variants:
        net=build(name).eval();difference=(net.details(c,a)[0]-reference).abs().max().item()
        if difference>2e-6:raise RuntimeError(f'Initialization changed predictions: {name}: {difference}')
        rows.append(dict(method=name,max_initial_prediction_difference=difference,parameters=sum(p.numel() for p in net.parameters())))
        print(f'[initialization] {name} delta={difference:.3g} params={rows[-1]["parameters"]}',flush=True)
    m.write_csv(out/'initialization_audit.csv',rows)

# Identical crops, loss sampling and fine-tuning budget for the control and three changes.
def fit(variant,pairs,targets):
    model=build(variant);optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    checkpoint=out/f'{variant}.pt';history=[]
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history']
        print(f'[resume] {variant}: {len(history)}/{epochs}',flush=True)
    for epoch in range(len(history),epochs):
        model.train();rng=np.random.default_rng(42+epoch);generator=torch.Generator().manual_seed(42+epoch)
        total=0.;steps=0;began=time.monotonic()
        for position in rng.permutation(len(pairs)):
            inputs,points=pairs.read(int(position));target_path=targets/f"{pairs.rows[position]['pair_id']}.npy"
            if not target_path.is_file():m.legacy.save_array(target_path,m.points_to_heatmaps(points,6.,0.,mode='gaussian').astype(np.float16))
            target=torch.from_numpy(np.load(target_path,allow_pickle=False)).to(device,dtype=torch.float32)
            for _ in range(CROPS):
                y,x=map(int,rng.integers(0,1024-CROP+1,size=2))
                y0,x0=max(0,y-HALO),max(0,x-HALO);y1,x1=min(1024,y+CROP+HALO),min(1024,x+CROP+HALO)
                patch=crop_inputs(inputs,y0,x0,max(y1-y0,x1-x0));ph,pw=y1-y0,x1-x0
                patch=(patch[0][...,:ph,:pw],patch[1],patch[2][...,:ph,:pw],patch[3][...,:ph,:pw],patch[4],patch[5])
                cy,cx=y-y0,x-x0;cut=(slice(None),slice(None),slice(cy,cy+CROP),slice(cx,cx+CROP))
                q=model.details(patch[0],patch[3])[0][cut]
                truth=target[:,y:y+CROP,x:x+CROP]
                loss=m.sampled_loss(torch.logit(q,eps=1e-6),patch[2][cut],truth,generator)
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {variant}')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();total+=loss.item();steps+=1
        record=dict(method=variant,epoch=epoch+1,weighted_BCE=total/steps,seconds=time.monotonic()-began)
        history.append(record);m.legacy.atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            history=history,epoch=epoch+1,variant=variant,initial_spatial_sha256=identity['original_spatial_sha256']),checkpoint)
        print(f'[{variant}] epoch {epoch+1}/{epochs} BCE={record["weighted_BCE"]:.6f} {record["seconds"]:.1f}s',flush=True)
    return model.eval(),history

@torch.inference_mode()
def output_maps(models,inputs):
    yield 'tta_mean',inputs[2][0]
    for name,model in models.items():
        height,width=inputs[0].shape[-2:];result=torch.empty((1,2,height,width),device=device)
        for y in range(0,height,128):
            for x in range(0,width,128):
                y0,x0=max(0,y-HALO),max(0,x-HALO);y1,x1=min(height,y+128+HALO),min(width,x+128+HALO)
                patch=crop_inputs(inputs,y0,x0,max(y1-y0,x1-x0));h,w=y1-y0,x1-x0
                patch=(patch[0][...,:h,:w],patch[1],patch[2][...,:h,:w],patch[3][...,:h,:w],patch[4],patch[5])
                q=model(*patch).sigmoid();hh,ww=min(128,height-y),min(128,width-x)
                result[...,y:y+hh,x:x+ww]=q[...,y-y0:y-y0+hh,x-x0:x-x0+ww]
        yield name,result[0]
m.output_maps=output_maps

# Official-style classwise matching identities, to distinguish rescue from damage.
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

@torch.inference_mode()
def diagnostics(pairs,models):
    rows=[];scales=[];adapters=[]
    for name,net in models.items():
        if name=='disagreement':
            for i,side in enumerate(SCALES):
                for j,branch in enumerate(('cell','tissue')):
                    adapters.append(dict(method=name,branch=branch,scale=side,metric='difference_strength',value=net.context.strength[0,i,j,0,0,0].item()))
        if name=='local_detail':
            for branch in ('cell','tissue'):
                for i,block in enumerate(getattr(net.context,branch)):
                    if isinstance(block,DetailBlock):
                        ratio=block.local.weight.norm()/(block.wide.weight.norm()+1e-8)
                        adapters.append(dict(method=name,branch=branch,scale=SCALES[i],metric='dense_to_dilated_weight_norm',value=ratio.item()))
    for position in range(min(8,len(pairs))):
        inputs,_=pairs.read(position);patch=crop_inputs(inputs,385,385,254)
        for name,net in models.items():
            features=net.features(patch[0],patch[3]);q,w,_,_,_=net.combine(patch[0],features)
            shifted=list(features);shifted[1]=torch.roll(shifted[1],(32,32),(-2,-1))
            qs=net.combine(patch[0],shifted)[0]
            r=dict(pair_id=pairs.rows[position]['pair_id'],method=name,
                context_shift_mean_abs_change=(q-qs)[...,63:191,63:191].abs().mean().item(),
                weight_spatial_std=w[...,63:191,63:191].flatten(3).std(-1).mean().item())
            for i,base_name in enumerate(m.MODEL_NAMES):
                for c,label in enumerate(('BC','TC')):r[f'weight_{base_name}_{label}']=w[:,i,c,63:191,63:191].mean().item()
            rows.append(r)
            if name=='scale_select':
                raw=ns['raw_maps'](patch[0],patch[3]);f=net.context.base(raw);ac,at=net.context.attention(f)
                for i,side in enumerate(SCALES):scales.append(dict(pair_id=pairs.rows[position]['pair_id'],scale=side,
                    cell_attention=ac[:,i,63:191,63:191].mean().item(),tissue_attention=at[:,i,63:191,63:191].mean().item()))
    m.write_csv(out/'gate_diagnostics.csv',rows);m.write_csv(out/'scale_attention.csv',scales)
    m.write_csv(out/'kernel_adapters.csv',adapters)

check_initialization()
models={'spatial':build('spatial').eval().requires_grad_(False)};history=[]
cache=Path(config['cache']);targets=out/'fit_targets';targets.mkdir(exist_ok=True)
with m.data_paths({'data_root':config['data_root']}):
    val=m.TorchOcelotDataset(split='val');lookup={r['pair_id']:i for i,r in enumerate(val.source.rows)}
    fit_pairs=m.Pairs(val,[lookup[p] for p in config['fit_ids']],'val',cache,device)
    tune_pairs=m.Pairs(val,[lookup[p] for p in config['tune_ids']],'val',cache,device)
    for name in variants:
        print(f'[stage] fine-tuning {name}',flush=True)
        net,record=fit(name,fit_pairs,targets);models[name]=net;history.extend(record);m.write_csv(out/'epoch_history.csv',history)
    diagnostics(tune_pairs,models)
    if (out/'postprocess.json').is_file():chosen={n:tuple(v) for n,v in json.loads((out/'postprocess.json').read_text())['per_method'].items()}
    else:
        chosen,grid=m.tune(tune_pairs,models);m.write_csv(out/'validation_grid.csv',grid);m.write_json(out/'postprocess.json',m.group_settings(chosen))
    with gzip.open(out/'val_predictions.jsonl.gz','wt') as stream:val_rows,records=evaluate('val_tune',tune_pairs,models,chosen,stream)
m.write_csv(out/'validation.csv',val_rows);m.write_csv(out/'val_per_pair.csv',records)
selected=max((r for r in val_rows if r['policy']=='per_method'),key=lambda r:r['mF1'])['method']
frozen=json.loads(json.dumps(dict(identity,selected_method=selected,postprocess=m.group_settings(chosen),
    adapted_sha256={n:m.sha256(out/f'{n}.pt') for n in variants})))
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
(out/'README.md').write_text('''# Three minimal context-kernel changes to spatial

All models start from the SAME original spatial checkpoint, checked by SHA256.
Initialization audit verifies the three modifications have the same prediction as spatial.
Original input15 + learned context192 + centered cell10 =217 channels, LayerNorm,
217->16->10 MLP, class-specific expert softmax and weighted original BC/TC maps stay intact.
No candidate verifier, label replacement, extra classification objective or routing loss is used.

| Method | Change |
|---|---|
| spatial | Frozen original reference |
| spatial_ft | Unmodified matched-budget fine-tuning control |
| scale_select | Per-pixel six-scale softmax attention for cell/tissue separately; weighted concat; uniform identity initialization |
| local_detail | Add a zero-initialized dense3x3 path beside each dilation2..32 convolution, summed before ReLU |
| disagreement | Add an independent spatial encoder of centered model scores, cell/tissue separately; six branch strengths start at0 |

The three changes are INDEPENDENT, not stacked together. The first two preserve the
maximum RF3/7/15/31/63/127; the third also uses the same six receptive fields.
Our original powers-of-two stack already has contiguous theoretical RF support. This
experiment does NOT assume proven gridding holes; dense paths test additional local detail.
Centered predictions are deterministic re-expressions of the original maps, not new data.
Scale attention is spatially local; SKNet's original global pooled mechanism is not reproduced.
Additional paths increase parameter counts (see initialization_audit.csv), so gains cannot be
attributed to information type alone without later parameter-matched controls.
All four fine-tuning runs use AdamW1e-4, fixed final12 epochs by default, seed42, same
four128px cores/image/epoch with63px halos, same Gaussian-sigma6 sampled BCE and raw D4
caches. Context encoder and the existing fusion head are both fine-tuned. No shared trainable
state is used between runs. Additional spatial_ft controls the effect of training longer.
Gate diagnostics, scale_attention.csv and kernel_adapters.csv measure use of context; per-pair files track rescue,
lost GTs, BC/TC precision/recall and FP/FN relative to original spatial. These are diagnostics,
not training objectives. The original references are also tuned on the same validation grid;
shared uses the same TTA-derived postprocessing for every method, per_method uses own val PP.
Validation freezes postprocessing and method selection before this run's test evaluation.
Source validation fit/tune patient IDs are reused; this is exploratory stacking on part of
validation, not a train-only official benchmark. Prior test exposure remains applicable.
TTA is the original joint cell/tissue D4 cache; final gates are not separately D4-averaged.
Tissue stays at the original cache256 resolution. G/GA and other base models are frozen.

Literature inspiration, not exact reproductions:
- Selective Kernel Networks, CVPR2019: https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Selective_Kernel_Networks_CVPR_2019_paper.html
- Understanding Convolution for Semantic Segmentation, WACV2018: https://arxiv.org/abs/1702.08502
- Disagreement-based region-specific ensemble,2026: https://doi.org/10.3389/frai.2026.1812932
''',encoding='utf-8')
(out/'complete.ok').write_text('complete\n')

