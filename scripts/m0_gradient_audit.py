#!/usr/bin/env python3
"""M0: per-image Cell/Tissue gradient alignment and publication-style figures.
Run from the M0 repository root. No training or checkpoint mutation.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

GROUPS = [
    ('all_shared', 'All shared'),
    ('cell.encoder.', 'Cell encoder'),
    ('tissue.encoder.', 'Tissue encoder'),
    ('cell.aspp.', 'Cell ASPP'),
    ('tissue.aspp.', 'Tissue ASPP'),
    ('cell.decoder.', 'Cell decoder'),
    ('tissue.decoder.', 'Tissue decoder'),
    ('encoder_c2t.', 'Cell → Tissue'),
    ('aspp_t2c.', 'Tissue → Cell / ASPP'),
    ('decoder_t2c.', 'Tissue → Cell / decoder'),
]
BLUE, RED, GREY = '#245B78', '#A84E4E', '#8B949B'

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': .7,
        'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.dpi': 450, 'figure.facecolor': 'white',
    })

def valid(rows, group='all_shared'):
    return [r for r in rows if r['group'] == group and np.isfinite(r['cosine'])]

def decorate(ax, letter, title):
    ax.set_title(title, loc='left', pad=11)
    ax.text(-.16, 1.045, letter, transform=ax.transAxes, weight='bold',
            fontsize=12, va='bottom')
    ax.tick_params(direction='out')

def save_figure(fig, path):
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(path.with_suffix('.' + extension), bbox_inches='tight',
                    pad_inches=.12)
    plt.close(fig)

def plot_checkpoint(rows, directory, tag):
    setup_style()
    all_rows = valid(rows)
    if not all_rows:
        print('[주의] 유효한 shared gradient가 없어 그림 생성을 생략합니다.')
        return
    rng = np.random.default_rng(2026)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.1))
    fig.subplots_adjust(left=.08, right=.98, bottom=.16, top=.86, wspace=.85)
    ax = axes[0]
    values = np.array([r['cosine'] for r in all_rows])
    edges = np.linspace(-1, 1, 26)
    ax.hist(values, bins=edges, weights=np.ones(len(values))/len(values)*100,
            color=BLUE, alpha=.85, edgecolor='white', linewidth=.45)
    ax.plot(values, np.full(len(values), -.7), '|', color=GREY,
            markersize=5, alpha=.55)
    ax.axvline(0, color=RED, lw=.9, ls=(0,(3,3)))
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.set_xlabel('Gradient cosine similarity')
    ax.set_ylabel('Image pairs per bin (%)')
    ax.text(.02, 1.02,
            f'n = {len(values)}   |   cosine < 0: {(values < 0).mean()*100:.1f}%',
            transform=ax.transAxes, fontsize=9)
    decorate(ax,'A','Cell–Tissue gradient alignment')
    ax = axes[1]
    names=[]
    for key,label in GROUPS:
        rs=valid(rows,key)
        if not rs:
            continue
        values=np.array([r['cosine'] for r in rs])
        y=len(names)
        if len(values)>2 and np.ptp(values)>1e-8:
            parts=ax.violinplot(values,positions=[y],vert=False,widths=.65,
                                showextrema=False,points=100)
            for body in parts['bodies']:
                body.set_facecolor(BLUE)
                body.set_edgecolor('none')
                body.set_alpha(.16)
        ax.scatter(values,y+rng.uniform(-.16,.16,len(values)),s=6,
                   color=GREY,alpha=.45,linewidths=0,rasterized=True)
        lo,median,hi=np.quantile(values,[.25,.5,.75])
        ax.plot([lo,hi],[y,y],color=BLUE,lw=2.1)
        ax.scatter(median,y,s=21,color=BLUE,zorder=4)
        names.append(label)
    ax.set_yticks(range(len(names)),names)
    ax.set_ylim(len(names)-.5,-.5)
    ax.set_xlim(-1.03,1.03)
    ax.set_xticks([-1,-.5,0,.5,1])
    ax.axvline(0,color=RED,lw=.9,ls=(0,(3,3)))
    ax.set_xlabel('Gradient cosine similarity')
    decorate(ax,'B','Distribution by model component')
    save_figure(fig,directory/f'{tag}_gradient_conflict')

def scalar_metrics(dot, c2, t2):
    nc, nt = math.sqrt(max(0,c2)), math.sqrt(max(0,t2))
    if nc <= 1e-12 or nt <= 1e-12:
        return dict(cosine=np.nan)
    return dict(cosine=float(np.clip(dot/(nc*nt),-1,1)))

def gradient_stats(names, gc, gt, key, torch):
    # Compare exactly the same parameter coordinates. Exclude task-private tensors.
    indices = [i for i, name in enumerate(names)
               if (key=='all_shared' or name.startswith(key))
               and gc[i] is not None and gt[i] is not None]
    result = dict(shared_tensors=len(indices), shared_parameters=0)
    if not indices:
        return {**result, **scalar_metrics(0,0,0)}
    totals = torch.zeros(3, device=gc[indices[0]].device, dtype=torch.float32)
    for i in indices:
        c, t = gc[i].detach().float(), gt[i].detach().float()
        totals += torch.stack(((c*t).sum(), (c*c).sum(), (t*t).sum()))
        result['shared_parameters'] += c.numel()
    dot, c2, t2 = totals.cpu().tolist()
    if not np.all(np.isfinite([dot,c2,t2])):
        raise RuntimeError('Non-finite gradient: use FP32 and check checkpoint.')
    return {**result, **scalar_metrics(dot,c2,t2)}

def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    text_fields = {'checkpoint','pair_id','group'}
    for r in rows:
        for k in r:
            if k not in text_fields:
                r[k] = float(r[k])
    return rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', help='M0 best.pt path')
    parser.add_argument('--split', choices=['train','val'], default='val')
    parser.add_argument('--limit', type=int, default=0, help='0: all images; otherwise seeded random subset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--size', type=int, help='default: original checkpoint input size')
    parser.add_argument('--out', default='outputs/gradient_audit')
    parser.add_argument('--plot-only', help='replot an existing raw CSV; no torch required')
    args = parser.parse_args()
    if args.limit < 0:
        parser.error('--limit must be >= 0')
    directory = Path(args.out).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        rows = read_csv(args.plot_only)
        for ck in sorted(set(r['checkpoint'] for r in rows)):
            subset = [r for r in rows if r['checkpoint']==ck]
            tag = Path(ck).stem + '_' + hashlib.sha1(ck.encode()).hexdigest()[:6]
            plot_checkpoint(subset, directory, tag)
        return
    root = Path.cwd().resolve()
    if not (root/'models/M0/model.py').is_file():
        raise SystemExit('Run this script from the M0 repository root.')
    sys.path.insert(0, str(root))
    import torch
    from torch.utils.data import DataLoader, Subset
    from common.backbones.ssl_resnet import SSLResNet50
    from common.data.tensors import TorchOcelotDataset, collate_pairs
    from common.training.m0_loop import prepare, pick_device, seed_everything
    from common.losses.m0_losses import m0_task_losses
    from models.M0.model import M0
    from common.paths import resolve_paths
    if not args.checkpoint:
        base = resolve_paths()['outputs']/'m0'
        preferred = base/'M0_official_default_seed42/best.pt'
        found = [preferred] if preferred.is_file() else sorted(base.glob('*/best.pt'))
        if len(found)!=1:
            raise SystemExit('Specify --checkpoint PATH. Candidates:\n' + '\n'.join(map(str,found)))
        args.checkpoint = str(found[0])
    device = pick_device(args.device)
    seed_everything(args.seed)
    dataset = TorchOcelotDataset(split=args.split, label_source='official', augment=False, seed=args.seed)
    selected = np.arange(len(dataset))
    if args.limit and args.limit < len(selected):
        selected = np.sort(np.random.default_rng(args.seed).choice(selected, args.limit, replace=False))
    loader = DataLoader(Subset(dataset, selected.tolist()), batch_size=1,
                        shuffle=False, num_workers=0, collate_fn=collate_pairs)
    rows, metadata = [], []
    for cp in [args.checkpoint]:
        path = Path(cp).expanduser().resolve()
        state = torch.load(path, map_location='cpu', weights_only=False)
        settings = state['settings']
        config_path = Path(settings['model_config'])
        if not config_path.is_absolute():
            config_path = root/config_path
        config = json.loads(config_path.read_text(encoding='utf-8'))
        # No separate SSL weight file is necessary: checkpoint contains both encoders.
        model = M0(SSLResNet50(), SSLResNet50(), config)
        model.load_state_dict(state['model'], strict=True)
        epoch = int(state['epoch'])+1
        frozen = int(state['epoch']) < int(settings.get('encoder_freeze_epochs',0))
        for p in model.encoder_parameters():
            p.requires_grad_(not frozen)
        model.to(device).eval()
        del state['model']
        named = [(n,p) for n,p in model.named_parameters() if p.requires_grad]
        names, parameters = zip(*named)
        size = args.size or int(settings['size'])
        if size < 32:
            raise SystemExit('--size must be at least 32.')
        loss_cfg = settings['cell_loss']
        loss_kwargs = {k:loss_cfg[k] for k in ('gamma','normalize','min_foreground_fraction') if k in loss_cfg}
        if 'min_foreground_fraction' not in loss_kwargs:
            # Mirror eval_m0.py for older checkpoints.
            loss_kwargs['min_foreground_fraction'] = 0.0
        checkpoint_rows = []
        print(f'[진단] {path.name} | epoch={epoch} | {args.split}={len(selected)} | size={size} | {device}',flush=True)
        for i,batch in enumerate(loader):
            images, targets, boxes = prepare(batch,device,size)
            out = model(*images,boxes)
            losses = m0_task_losses(out['cell_logits'],out['tissue_logits'],*targets,**loss_kwargs)
            if not all(torch.isfinite(v).item() for v in losses.values()):
                raise RuntimeError('Non-finite task loss.')
            gc = torch.autograd.grad(losses['cell'],parameters,retain_graph=True,allow_unused=True)
            gt = torch.autograd.grad(losses['tissue'],parameters,allow_unused=True)
            pid = str(batch['pair_id'][0])
            base = dict(checkpoint=str(path),epoch=epoch,pair_id=pid,
                        encoder_frozen=int(frozen), input_size=size)
            for key,_ in GROUPS:
                checkpoint_rows.append({**base,'group':key,
                                        **gradient_stats(names,gc,gt,key,torch)})
            del gc,gt,losses,out,images,targets,boxes
            if (i+1)%10==0 or i+1==len(loader):
                print(f'  {i+1}/{len(loader)}',flush=True)
        rows.extend(checkpoint_rows)
        write_csv(directory/'gradient_raw.csv',rows)
        tag = f'epoch{epoch:03d}_'+hashlib.sha1(str(path).encode()).hexdigest()[:6]
        plot_checkpoint(checkpoint_rows,directory,tag)
        metadata.append(dict(checkpoint=str(path),epoch=epoch,
                             model_config=config,encoder_frozen=frozen,input_size=size))
        del model,parameters,named
        if device.type=='cuda':
            torch.cuda.empty_cache()
    summaries=[]
    for key,label in GROUPS:
        rs=valid(rows,key)
        if not rs:
            continue
        vals=np.array([r['cosine'] for r in rs])
        summaries.append(dict(group=label,n=len(rs),mean_cosine=float(vals.mean()),
                              median_cosine=float(np.median(vals)),
                              negative_fraction=float((vals<0).mean())))
    if summaries:
        write_csv(directory/'gradient_summary.csv',summaries)
    (directory/'protocol.json').write_text(json.dumps(dict(
        split=args.split,seed=args.seed,pair_ids=[dataset.source.rows[i]['pair_id'] for i in selected],
        mode='eval; gradients enabled; FP32; one image pair per observation; no optimizer step',
        parameter_scope='intersection of tensors reached by both task losses; excludes task-private heads',
        warning='fixed-checkpoint diagnostic; no chronological training trend; negative cosine does not prove performance degradation',
        checkpoints=metadata),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'[완료] {directory}\nPNG/PDF/SVG + raw CSV + summary CSV + protocol.json',flush=True)

if __name__=='__main__':
    main()
