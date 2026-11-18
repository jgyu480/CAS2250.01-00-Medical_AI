import csv,gzip,importlib.util,json,math,os,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
source=Path(os.environ['TEMP_SOURCE_RUN']).resolve()
out=Path(os.environ['TEMP_OUTPUT']).resolve()
os.environ['M3_OUTPUT']=str(source)
spec=importlib.util.spec_from_file_location('prior',source/'m3_context.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.OUT=out
config=json.loads((source/'run.json').read_text())
device=torch.device(os.environ.get('M3_DEVICE','cuda'))
torch.set_num_threads(2)
if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
epochs=int(os.environ.get('ABLATE_EPOCHS','12'))
if epochs<1: raise ValueError('epochs must be positive')
variants=('pointwise','softmax','pool','spatial','gain')
m.METHODS=('tta_mean',*variants)
SCALES=(3,7,15,31,63,127)
HALO=63;CROP=128;CROPS=4
paths=[Path(config['checkpoint_paths'][n]) for n in m.MODEL_NAMES]
for name,path in zip(m.MODEL_NAMES,paths):
    if m.sha256(path)!=config['checkpoint_sha256'][name]: raise ValueError(f'Changed base model {name}')
if m.sha256(source/'m3_context.py')!=config['implementation_sha256']: raise ValueError('Changed helpers')

def mean_window(x,k):
    if k==1:return x
    p=k//2
    return F.avg_pool2d(F.pad(F.avg_pool2d(F.pad(x,(p,p,0,0),mode='replicate'),(1,k),stride=1),(0,0,p,p),mode='replicate'),(k,1),stride=1)

def raw_maps(cells,aligned):
    return torch.cat((cells,F.interpolate(aligned,size=cells.shape[-2:],mode='bilinear',align_corners=False)),1)

class PooledEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell=nn.Sequential(nn.Conv2d(10,16,3,padding=1,padding_mode='replicate'),nn.ReLU())
        self.tissue=nn.Sequential(nn.Conv2d(5,16,3,padding=1,padding_mode='replicate'),nn.ReLU())
    def forward(self,x):
        c,t=self.cell(x[:,:10]),self.tissue(x[:,10:])
        return torch.cat([z for side in SCALES for z in (mean_window(c,side-2),mean_window(t,side-2))],1)

class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        def branch(channels):
            return nn.ModuleList(nn.Sequential(nn.Conv2d(channels if i==0 else 16,16,3,
                dilation=2**i,padding=2**i,padding_mode='replicate'),nn.ReLU()) for i in range(6))
        self.cell=branch(10);self.tissue=branch(5)
    def forward(self,x):
        c,t=x[:,:10],x[:,10:];features=[]
        for cb,tb in zip(self.cell,self.tissue):
            c,t=cb(c),tb(t);features.extend((c,t))
        return torch.cat(features,1)

class Gate(nn.Module):
    def __init__(self,variant):
        super().__init__();self.variant=variant
        channels=15
        if variant=='pool':self.context=PooledEncoder();channels=207
        if variant in ('spatial','gain'):self.context=SpatialEncoder();channels=217
        self.norm=nn.LayerNorm(channels) if variant in ('spatial','gain') else nn.Identity()
        self.head=nn.Sequential(nn.Conv2d(channels,16,1),nn.ReLU(),nn.Conv2d(16,10,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        if variant=='gain':
            self.alpha_head=nn.Conv2d(16,2,1)
            nn.init.zeros_(self.alpha_head.weight);nn.init.constant_(self.alpha_head.bias,-2.197224577)
    def features(self,cells,aligned):
        raw=raw_maps(cells,aligned);features=[raw]
        if self.variant in ('pool','spatial','gain'):features.append(self.context(raw))
        if self.variant in ('spatial','gain'):
            p=cells.reshape(cells.shape[0],5,2,*cells.shape[-2:])
            features.append((p-p.mean(1,keepdim=True)).flatten(1,2))
        return features
    def combine(self,cells,features):
        x=torch.cat(features,1)
        if self.variant in ('spatial','gain'):x=self.norm(x.permute(0,2,3,1)).permute(0,3,1,2)
        hidden=self.head[1](self.head[0](x))
        z=self.head[2](hidden).reshape(cells.shape[0],5,2,*cells.shape[-2:])
        a=.1*z.tanh() if self.variant=='pointwise' else z
        v=a.softmax(1)
        p=cells.reshape_as(v);base=p.mean(1)
        alpha=self.alpha_head(hidden).sigmoid() if self.variant=='gain' else torch.ones_like(base)
        weights=(1-alpha[:,None])/5+alpha[:,None]*v
        q=(weights*p).sum(1)
        return q,weights,v,alpha,z
    def details(self,cells,aligned):return self.combine(cells,self.features(cells,aligned))
    def forward(self,cells,tissue,base,aligned,box,extent):return torch.logit(self.details(cells,aligned)[0],eps=1e-6)

def crop_inputs(inputs,y,x,size):
    c,t,b,a,box,extent=inputs
    a=F.interpolate(a,size=c.shape[-2:],mode='bilinear',align_corners=False)
    return c[...,y:y+size,x:x+size],t,b[...,y:y+size,x:x+size],a[...,y:y+size,x:x+size],box,extent

def gain_losses(q,v,alpha,cells,truth,generator):
    # Event supervision: <=2px from GT center is positive; >=15px is negative.
    # This is a soft local competence target, NOT the official detection matching oracle.
    p=cells.reshape(1,5,2,*cells.shape[-2:])
    support=F.max_pool2d(p.flatten(0,1),3,stride=1,padding=1).reshape_as(p)
    positive=truth>=math.exp(-4/72)
    negative=truth<=math.exp(-225/72)
    disputed=(support.max(1).values-support.min(1).values)>.05
    hard=(support.max(1).values>.05)
    pos=(positive[None]&disputed).flatten().nonzero().flatten()
    neg=(negative[None]&disputed&hard).flatten().nonzero().flatten()
    def sample(indices,n):
        if indices.numel()<=n:return indices
        order=torch.randperm(indices.numel(),generator=generator)[:n].to(indices.device)
        return indices[order]
    selected=torch.cat((sample(pos,2048),sample(neg,2048)))
    routing=q.sum()*0
    if selected.numel():
        y=positive[None].float()
        cost=F.binary_cross_entropy(support.clamp(1e-6,1-1e-6),y[:,None].expand_as(support),reduction='none')
        teacher=(-cost/.25).softmax(1).detach()
        ce=-(teacher*v.clamp_min(1e-8).log()).sum(1).flatten()
        routing=ce[selected].mean()
    # Keep the mean where models agree and the event label confirms it is safe.
    base=p.mean(1)
    safe=((positive[None]&(base>.5))|(negative[None]&(base<.05)))&((p.max(1).values-p.min(1).values)<.05)
    preserve=(q-base).square()[safe].mean() if safe.any() else q.sum()*0
    return routing,preserve,int(pos.numel()),int(neg.numel())

def fit(variant,pairs,targets):
    m.seed_everything(42)
    model=Gate(variant).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    checkpoint=out/f'{variant}.pt';history=[]
    if checkpoint.is_file():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);history=state['history']
        print(f'[resume] {variant}: {len(history)}/{epochs}',flush=True)
    began=time.monotonic();start_epoch=len(history)
    for epoch in range(len(history),epochs):
        model.train();rng=np.random.default_rng(42+epoch)
        generator=torch.Generator().manual_seed(42+epoch)
        sums=np.zeros(5);steps=0;epoch_start=time.monotonic()
        for position in rng.permutation(len(pairs)):
            inputs,points=pairs.read(int(position))
            target_path=targets/f"{pairs.rows[position]['pair_id']}.npy"
            if not target_path.is_file():m.legacy.save_array(target_path,m.points_to_heatmaps(points,6.,0.,mode='gaussian').astype(np.float16))
            target=torch.from_numpy(np.load(target_path,allow_pickle=False)).to(device,dtype=torch.float32)
            for _ in range(CROPS):
                y,x=map(int,rng.integers(0,1024-CROP+1,size=2))
                y0,x0=max(0,y-HALO),max(0,x-HALO)
                y1,x1=min(1024,y+CROP+HALO),min(1024,x+CROP+HALO)
                patch=crop_inputs(inputs,y0,x0,max(y1-y0,x1-x0))
                ph,pw=y1-y0,x1-x0
                patch=(patch[0][...,:ph,:pw],patch[1],patch[2][...,:ph,:pw],patch[3][...,:ph,:pw],patch[4],patch[5])
                cy,cx=y-y0,x-x0
                q,w,v,alpha,z=model.details(patch[0],patch[3])
                cut=(slice(None),slice(None),slice(cy,cy+CROP),slice(cx,cx+CROP))
                q=q[cut];truth=target[:,y:y+CROP,x:x+CROP]
                b=patch[2][cut]
                loss_main=m.sampled_loss(torch.logit(q,eps=1e-6),b,truth,generator)
                route=keep=q.sum()*0;pos=neg=0
                if variant=='gain':
                    route,keep,pos,neg=gain_losses(q,v[...,cy:cy+CROP,cx:cx+CROP],
                        alpha[cut],patch[0][cut],truth,generator)
                loss=loss_main+.2*route+keep
                if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss: {variant}')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
                sums+=np.array([loss_main.item(),route.item(),keep.item(),pos,neg]);steps+=1
        duration=time.monotonic()-epoch_start
        values=sums/steps
        history.append(dict(method=variant,epoch=epoch+1,weighted_BCE=float(values[0]),
            routing_loss=float(values[1]),preserve_loss=float(values[2]),positive_route_pixels=float(values[3]),
            negative_route_pixels=float(values[4]),seconds=duration))
        m.legacy.atomic_save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,
            epoch=epoch+1,variant=variant),checkpoint)
        remaining=(time.monotonic()-began)/(epoch+1-start_epoch)*(epochs-epoch-1)
        print(f'[{variant}] epoch {epoch+1}/{epochs} BCE={values[0]:.6f} route={values[1]:.6f} '
            f'keep={values[2]:.8f} {duration:.1f}s remaining_this_model~{remaining/60:.1f}min',flush=True)
    return model.eval(),history

@torch.inference_mode()
def output_maps(models,inputs):
    yield 'tta_mean',inputs[2][0]
    for name,model in models.items():
        h,w=inputs[0].shape[-2:];result=torch.empty((1,2,h,w),device=device)
        for y in range(0,h,128):
            for x in range(0,w,128):
                y0,x0=max(0,y-HALO),max(0,x-HALO)
                y1,x1=min(h,y+128+HALO),min(w,x+128+HALO)
                patch=crop_inputs(inputs,y0,x0,max(y1-y0,x1-x0))
                ph,pw=y1-y0,x1-x0
                patch=(patch[0][...,:ph,:pw],patch[1],patch[2][...,:ph,:pw],patch[3][...,:ph,:pw],patch[4],patch[5])
                pred=model(*patch).sigmoid();hh,ww=min(128,h-y),min(128,w-x)
                result[...,y:y+hh,x:x+ww]=pred[...,y-y0:y-y0+hh,x-x0:x-x0+ww]
        yield name,result[0]
m.output_maps=output_maps

@torch.inference_mode()
def diagnostics(pairs,models):
    records=[]
    for position in range(min(8,len(pairs))):
        inputs,_=pairs.read(position);patch=crop_inputs(inputs,385,385,254)
        for name,model in models.items():
            features=model.features(patch[0],patch[3]);q,w,v,alpha,z=model.combine(patch[0],features)
            core=w[...,63:191,63:191]
            shifted=list(features)
            if len(features)>1:shifted[1]=torch.roll(features[1],(32,32),(-2,-1))
            qs=model.combine(patch[0],shifted)[0]
            record=dict(pair_id=pairs.rows[position]['pair_id'],method=name,
                context_shift_mean_abs_change=(qs-q)[...,63:191,63:191].abs().mean().item(),
                effective_weight_spatial_std=core.flatten(3).std(-1).mean().item(),
                tanh_saturated_fraction=(z.abs()>3).float().mean().item() if name=='pointwise' else 0.,
                alpha_mean=alpha.mean().item())
            for i,base_name in enumerate(m.MODEL_NAMES):
                for c,label in enumerate(('BC','TC')):record[f'weight_{base_name}_{label}']=core[:,i,c].mean().item()
            records.append(record)
    m.write_csv(out/'gate_diagnostics.csv',records)
identity=dict(source_run_sha256=m.sha256(source/'run.json'),script_sha256=m.sha256(out/'launcher.sh'),
    epochs=epochs,seed=42,methods=list(m.METHODS),scales=list(SCALES),crop=128,crops_per_image=4,
    crop_sampling="uniform cores including image borders",routing_weight=.2,preserve_weight=1.,router_target='local center-event BCE soft teacher; positives <=2px, negatives >=15px',
    test_enabled=os.environ.get('ABLATE_TEST','1')=='1',prior_test_exposure=True)
if (out/'run.json').is_file() and json.loads((out/'run.json').read_text())!=identity:raise ValueError('Settings changed: choose new ABLATE_RUN_ID')
m.write_json(out/'run.json',identity)
models={};history=[]
cache=Path(config['cache']);targets=out/'fit_targets';targets.mkdir(exist_ok=True)
with m.data_paths({'data_root':config['data_root']}):
    val=m.TorchOcelotDataset(split='val');lookup={r['pair_id']:i for i,r in enumerate(val.source.rows)}
    fit_pairs=m.Pairs(val,[lookup[p] for p in config['fit_ids']],'val',cache,device)
    tune_pairs=m.Pairs(val,[lookup[p] for p in config['tune_ids']],'val',cache,device)
    for variant in variants:
        print(f'[stage] training {variant}',flush=True)
        model,record=fit(variant,fit_pairs,targets);models[variant]=model;history.extend(record)
        m.write_csv(out/'epoch_history.csv',history)
    diagnostics(tune_pairs,models)
    if (out/'postprocess.json').is_file():chosen={n:tuple(v) for n,v in json.loads((out/'postprocess.json').read_text())['per_method'].items()}
    else:
        chosen,grid=m.tune(tune_pairs,models);m.write_csv(out/'validation_grid.csv',grid)
        m.write_json(out/'postprocess.json',m.group_settings(chosen))
    with gzip.open(out/'val_predictions.jsonl.gz','wt') as stream:val_rows,records=m.evaluate('val_tune',tune_pairs,models,chosen,stream)
m.write_csv(out/'validation.csv',val_rows);m.write_csv(out/'val_per_pair.csv',records)
selected=max((r for r in val_rows if r['policy']=='per_method'),key=lambda r:r['mF1'])['method']
frozen=dict(identity,selected_method=selected,postprocess=m.group_settings(chosen),
    gate_sha256={n:m.sha256(out/f'{n}.pt') for n in variants},base_sha256=config['checkpoint_sha256'])
if (out/'frozen_before_test.json').is_file() and json.loads((out/'frozen_before_test.json').read_text())!=frozen:raise ValueError('Frozen selection changed')
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
    if not all(m.cache_file(tc,'test',test,i).is_file() for i in indices):raise FileNotFoundError(f'Incomplete D4 test cache: {tc}')
    pairs=m.Pairs(test,indices,'test',tc,device)
    with gzip.open(out/'test_predictions.jsonl.gz','wt') as stream:test_rows,records=m.evaluate('test',pairs,models,chosen,stream)
    m.write_csv(out/'test_per_pair.csv',records)
    final=next(r for r in test_rows if r['policy']=='per_method' and r['method']==selected)
    m.write_csv(out/'selected_test_score.csv',[final]);print(f'[validation-selected] {selected}; test Cell mF1={final["mF1"]:.6f}',flush=True)
m.write_csv(out/'summary.csv',val_rows+test_rows)
(out/'README.md').write_text('''# Five controlled ensemble experiments

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
''',encoding='utf-8')
(out/'complete.ok').write_text('complete\n')

