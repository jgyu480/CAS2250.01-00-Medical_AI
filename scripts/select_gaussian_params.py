#!/usr/bin/env python3
"""Training-only, CPU-only geometric screening; does NOT estimate model accuracy.

Run from the OCELOT repository root, or pass --repo /path/to/repository.
Dependencies: numpy scipy matplotlib Pillow. Existing files/configs are never edited.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def nearest(tree, xy):
    return tree.query(xy, k=2)[0]


def maps(dist, sigma, lam):
    d1, d2 = dist[..., 0], dist[..., 1]
    g = np.exp(-d1**2 / (2*sigma**2))
    # A=0 at a center; A=1 at equal distances. Singleton: no suppression.
    a = np.divide(2*d1, d1+d2, out=np.zeros_like(d1),
                  where=np.isfinite(d2) & ((d1+d2) > 0))**2
    o = np.exp(-d2**2 / (2*sigma**2))
    return g, g*(1-lam*a), g*(1-lam*a*o)


def macro_fraction(values, image_ids, mask, threshold, less=True):
    if not mask.any():
        return None
    good = values <= threshold if less else values >= threshold
    den = np.bincount(image_ids[mask])
    num = np.bincount(image_ids[mask], weights=good[mask], minlength=len(den))
    return float(np.mean(num[den > 0] / den[den > 0]))


def choose(rows, radius, valley_target, core_target, min_pairs):
    possible = []
    for r in rows:
        active = [name for name in ('BC-BC', 'TC-TC', 'BC-TC')
                  if r['n_'+name] >= min_pairs]
        if not active:
            continue
        # Require each adequately represented pair type to satisfy the rule.
        if (all(r['valley_'+name] >= .90 for name in active)
                and r['core_keep_fraction'] >= .90):
            possible.append(r)
    if not possible:
        return None
    # First match existing supervised support, then weakest sufficient suppression.
    r = min(possible, key=lambda r: (
        abs(np.sqrt(2*np.log(2))*r['sigma']-radius), r['lambda']))
    return {'sigma': r['sigma'], 'lambda': r['lambda'],
            'valley_threshold': valley_target, 'core_ratio_threshold': core_target,
            'rule': '>=90% pass for each pair type with sufficient n; image-macro fractions; '
                    'closest Gaussian half-height radius to existing disk radius, then smallest lambda'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, default=Path.cwd())
    p.add_argument('--out', type=Path, default=Path('outputs/gaussian_parameter_audit'))
    p.add_argument('--scope', choices=['pooled', 'classwise'], default='pooled',
                   help='pooled: all centers share geometry; classwise: separate BC/TC heatmaps')
    p.add_argument('--sigmas', nargs='+', type=float, default=[4, 6, 8, 10, 12, 16])
    p.add_argument('--lambdas', nargs='+', type=float, default=[0, .25, .5, .75, .9, 1])
    p.add_argument('--core-radius', type=float, default=3)
    p.add_argument('--reference-radius', type=float, default=None)
    p.add_argument('--min-pairs', type=int, default=30)
    p.add_argument('--examples', type=int, default=12)
    args = p.parse_args()
    if min(args.sigmas) <= 0 or min(args.lambdas) < 0 or max(args.lambdas) > 1:
        p.error('sigma>0, 0<=lambda<=1 required')
    if args.core_radius <= 0 or args.min_pairs < 1 or args.examples < 0:
        p.error('Invalid radius/count')
    root = args.repo.expanduser().resolve()
    config = json.loads((root/'configs/common.json').read_text())
    import os
    def resolve(key, env):
        value = Path(os.environ.get(env) or config['paths'][key]).expanduser()
        return value if value.is_absolute() else root/value
    data = resolve('data_root', 'OCELOT_DATA_ROOT')
    outputs = resolve('outputs', 'OUTPUT_ROOT')
    manifest = outputs/'audit/manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = sorted([r for r in csv.DictReader(f) if r['split']=='train'],
                      key=lambda r:r['pair_id'])
    if not rows:
        raise SystemExit('No train rows. Run the existing scripts/check_data.py first.')
    radius = args.reference_radius
    if radius is None:
        radius = float(json.loads((root/'configs/dataset.json').read_text())['cell_radius_px'])
    if radius <= 0:
        p.error('reference radius must be positive')
    out = args.out if args.out.is_absolute() else root/args.out
    out.mkdir(parents=True, exist_ok=True)
    # Statistics use ALL training annotations. Only selected examples load RGB images.
    pair_records, mid_d, ring_d, ring_images, records, provenance = [], [], [], [], [], []
    angles = np.arange(16)*2*np.pi/16
    offsets = args.core_radius*np.c_[np.cos(angles), np.sin(angles)]
    counts = {'cells': 0, 'boundary_core_exclusions': 0, 'singleton_groups': 0}
    for im, row in enumerate(rows):
        ann = data/row['cell_points']
        raw = ann.read_bytes()
        provenance.append({'pair_id': row['pair_id'], 'organ': row.get('organ',''),
                           'cell_points': row['cell_points'], 'sha256': hashlib.sha256(raw).hexdigest()})
        with ann.open(encoding='utf-8-sig', newline='') as f:
            pts = np.array([[float(v) for v in r] for r in csv.reader(f)
                            if r and any(v.strip() for v in r)], float).reshape(-1,3)
        if not np.isfinite(pts).all() or not np.isin(pts[:,2], [1,2]).all():
            raise ValueError(f'Invalid annotation: {ann}')
        if len(np.unique(pts[:,:2],axis=0)) != len(pts):
            raise ValueError(f'Duplicate coordinates: {ann}; inspect rather than silently discard')
        # Official OCELOT cell FoVs are 1024x1024, and repo reader uses this convention.
        if np.any(pts[:,:2]<0) or np.any(pts[:,:2]>1023):
            raise ValueError(f'Expected original 1024x1024 coordinates: {ann}')
        counts['cells'] += len(pts)
        records.append((row, pts))
        subsets = [np.arange(len(pts))] if args.scope=='pooled' else [
            np.flatnonzero(pts[:,2]==k) for k in (1,2)]
        for inds in subsets:
            xy = pts[inds,:2]
            if not len(xy):
                continue
            tree = cKDTree(xy)
            if len(xy)==1:
                counts['singleton_groups'] += 1
            ds, ix = tree.query(xy, k=2)
            core = xy[:,None,:]+offsets
            valid = ((core>=0)&(core<=1023)).all(axis=(1,2))
            counts['boundary_core_exclusions'] += int((~valid).sum())
            if valid.any():
                ring_d.append(nearest(tree, core[valid].reshape(-1,2)).reshape(-1,16,2))
                ring_images.extend([im]*int(valid.sum()))
            edges = sorted({tuple(sorted((i,int(ix[i,1])))) for i in range(len(xy))
                            if np.isfinite(ds[i,1])})
            for i,j in edges:
                a,b = inds[i], inds[j]
                gap = float(np.linalg.norm(xy[i]-xy[j]))
                pair_type = ('BC-BC' if pts[a,2]==pts[b,2]==1 else
                             'TC-TC' if pts[a,2]==pts[b,2]==2 else 'BC-TC')
                pair_records.append({'image_index': im, 'pair_id': row['pair_id'],
                                     'organ': row.get('organ',''), 'a': int(a), 'b': int(b),
                                     'distance_px': gap, 'type': pair_type})
                mid_d.append(nearest(tree, ((xy[i]+xy[j])/2)[None,:])[0])
        if (im+1)%50==0 or im+1==len(rows):
            print(f'[annotations] {im+1}/{len(rows)} training images', flush=True)
    if not pair_records or not ring_d:
        raise SystemExit('Insufficient training centers/pairs; no recommendation.')
    md = np.asarray(mid_d)
    rd = np.concatenate(ring_d)
    image_ids = np.array([r['image_index'] for r in pair_records])
    ring_images = np.asarray(ring_images)
    gaps = np.array([r['distance_px'] for r in pair_records])
    types = np.array([r['type'] for r in pair_records])
    target = (gaps>=14)&(gaps<30)
    all_metrics, sensitivity, primary = [], [], None
    metric_sets = {(vt,ct): [] for vt in (.2,.3,.4) for ct in (.75,.80,.85)}
    for s in sorted(set(args.sigmas)):
        for l in sorted(set(args.lambdas)):
            g, ha, hao = maps(md,s,l)
            rg, _, rh = maps(rd,s,l)
            retention = (rh/np.maximum(rg,1e-30)).min(axis=1)
            core_fractions = {ct: macro_fraction(retention,ring_images,
                             np.ones(len(retention),bool),ct,False) for ct in (.75,.80,.85)}
            for vt in (.2,.3,.4):
                item = {'sigma':s, 'lambda':l,
                        'half_height_radius_px':float(s*np.sqrt(2*np.log(2)))}
                for name in ('BC-BC','TC-TC','BC-TC'):
                    mask = target&(types==name)
                    item['n_'+name] = int(mask.sum())
                    item['valley_'+name] = macro_fraction(hao,image_ids,mask,vt)
                item['G_valley_fraction'] = macro_fraction(g,image_ids,target,vt)
                item['HA_valley_fraction'] = macro_fraction(ha,image_ids,target,vt)
                item['HAO_valley_fraction'] = macro_fraction(hao,image_ids,target,vt)
                for ct in (.75,.80,.85):
                    metric_sets[vt,ct].append({**item,'core_keep_fraction':core_fractions[ct]})
    for (vt,ct),metrics in metric_sets.items():
        rec = choose(metrics,radius,vt,ct,args.min_pairs)
        sensitivity.append({'valley_threshold':vt,'core_ratio_threshold':ct,
                            'sigma': rec['sigma'] if rec else None,
                            'lambda':rec['lambda'] if rec else None})
        if vt==.3 and ct==.80:
            all_metrics, primary = metrics, rec
    write_csv(out/'candidate_metrics.csv',all_metrics)
    write_csv(out/'threshold_sensitivity.csv',sensitivity)
    write_csv(out/'training_pairs.csv',pair_records)
    write_csv(out/'training_inputs.csv',provenance)
    # Example selection occurs by distance/type, with NO candidate-score selection.
    picked, used_images = [], set()
    for lo,hi in ((8,14),(14,20),(20,30),(30,60)):
        for name in ('BC-BC','TC-TC','BC-TC'):
            options = [r for r in pair_records if lo<=r['distance_px']<hi and r['type']==name]
            if not options:
                continue
            median = np.median([r['distance_px'] for r in options])
            options.sort(key=lambda r:(r['image_index'] in used_images,
                                      abs(r['distance_px']-median),r['pair_id'],r['a']))
            picked.append(options[0]); used_images.add(options[0]['image_index'])
    picked = picked[:args.examples]
    write_csv(out/'selected_examples.csv',picked)
    fig, ax = plt.subplots(figsize=(8,4))
    for name,col in zip(('BC-BC','TC-TC','BC-TC'),('#277da8','#e07a42','#744da9')):
        ax.hist(gaps[types==name],bins=np.arange(0,102,2),histtype='step',label=name,color=col)
    ax.axvspan(14,30,color='grey',alpha=.12,label='Target interval')
    ax.set(xlabel='Nearest-neighbor pair distance (original pixels)',ylabel='Unique pair count')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'pair_distances.png',dpi=180); plt.close(fig)
    # A provisional fallback is shown only for inspection, never called a recommendation.
    show = primary or {'sigma':12.,'lambda':.9}
    s,l = show['sigma'],show['lambda']
    for k,r in enumerate(picked,1):
        row,pts = records[r['image_index']]
        with Image.open(data/row['cell_image']) as src:
            if src.size!=(1024,1024):
                raise ValueError('Original-resolution coordinates required')
            rgb = np.asarray(src.convert('RGB'))
        a,b = pts[r['a'],:2],pts[r['b'],:2]
        center=(a+b)/2
        x0,y0=np.maximum(0,np.floor(center-64)).astype(int)
        x1,y1=np.minimum(1024,[x0+128,y0+128]).astype(int)
        xy=pts[:,:2]
        if args.scope=='classwise':
            xy=xy[pts[:,2]==pts[r['a'],2]]
        tree=cKDTree(xy)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        dm=nearest(tree,np.c_[xx.ravel(),yy.ravel()]).reshape(y1-y0,x1-x0,2)
        g,ha,hao=maps(dm,s,l)
        oldg,_,oldh=maps(dm,12,.9)
        fig=plt.figure(figsize=(14,7),layout='constrained')
        grid=fig.add_gridspec(2,4)
        axs=[fig.add_subplot(grid[0,i]) for i in range(4)]
        axs[0].imshow(rgb[y0:y1,x0:x1]); axs[0].set_title('Original + all annotated centers')
        for cls,col in ((1,'#39cbe6'),(2,'#ffad4d')):
            q=pts[(pts[:,2]==cls)&(pts[:,0]>=x0)&(pts[:,0]<x1)&(pts[:,1]>=y0)&(pts[:,1]<y1)]
            axs[0].scatter(q[:,0]-x0,q[:,1]-y0,s=20,facecolors='none',edgecolors=col)
        axs[0].plot([a[0]-x0,b[0]-x0],[a[1]-y0,b[1]-y0],color='white',lw=1)
        for ax,heat,title in zip(axs[1:],[g,ha,hao],['Gaussian','HA','HAO']):
            ax.imshow(heat,cmap='magma',vmin=0,vmax=1)
            ax.set_title(f'{title}: sigma={s:g}, lambda={l:g}')
        for ax in axs:
            ax.set_xticks([]);ax.set_yticks([])
        ax=fig.add_subplot(grid[1,:2])
        t=np.linspace(-.4,1.4,240)
        line=a[None,:]+t[:,None]*(b-a)[None,:]
        ld=nearest(tree,line)
        for values,label,col in zip(maps(ld,s,l),['G','HA','HAO'],['#9879b5','#219e9a','#e27636']):
            ax.plot(t*r['distance_px'],values,label=label,color=col,lw=2)
        ax.plot(t*r['distance_px'],maps(ld,12,.9)[2],label='HAO 12 / 0.9',c='#555',ls='--')
        ax.axvline(0,c='grey',ls=':'); ax.axvline(r['distance_px'],c='grey',ls=':')
        ax.axhline(.3,c='grey',ls='--',lw=.7)
        ax.set(xlabel='Distance along center-to-center line (px)',ylabel='Target score',ylim=(-.02,1.05))
        ax.legend(ncol=4,fontsize=8)
        for idx,heat,title in ((2,oldg,'Previous Gaussian: sigma=12'),(3,oldh,'Previous HAO: 12 / 0.9')):
            ax=fig.add_subplot(grid[1,idx]);ax.imshow(heat,cmap='magma',vmin=0,vmax=1)
            ax.set_title(title);ax.set_xticks([]);ax.set_yticks([])
        fig.suptitle(f"TRAIN {r['pair_id']} | {r['organ']} | {r['type']} | distance={r['distance_px']:.1f}px"
                     + (' | NO FEASIBLE RECOMMENDATION' if primary is None else ''))
        fig.savefig(out/f'example_{k:02d}_{r["pair_id"]}.png',dpi=160)
        plt.close(fig)
    summary={'status':'provisional_geometric_choice' if primary else 'no_feasible_candidate',
             'recommendation':primary,'scope':args.scope,'split':'train','images':len(rows),
             'counts':counts,'reference_disk_radius_px':radius,'core_ring_radius_px':args.core_radius,
             'sigma_anchor':radius/np.sqrt(2*np.log(2)),
             'geometry':'original 1024x1024 pixels; no resizing or augmentation; unique NN edges',
             'target_pairs':'14 <= center distance < 30 px; midpoint evaluated against ALL relevant centers',
             'sources':['https://arxiv.org/abs/1904.07850','https://arxiv.org/abs/1711.07752'],
             'formula':'G=exp(-d1^2/(2*sigma^2)); A=(2*d1/(d1+d2))^2; O=exp(-d2^2/(2*sigma^2)); HA=G*(1-lambda*A); HAO=G*(1-lambda*A*O)',
             'limitations':['HA/HAO and screening thresholds are project heuristics, not published methods.',
                           'Disk radius is existing target support, NOT measured biological cell radius.',
                           'Midpoint valley/core retention are target geometry, NOT model detection accuracy.',
                           'Very narrow Gaussian or strong suppression can be harder to learn; not assessed.',
                           'Point labels alone cannot identify optimal sigma/lambda or true nuclear sizes.',
                           'Classwise and pooled geometry are different; match the actual target implementation.',
                           'If output stride is R, use sigma/R and all distances/R on that grid.',
                           'Train all G/HA/HAO comparisons at the SAME chosen sigma; lambda shared by HA/HAO.']}
    (out/'recommendation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    text = ['TRAINING-ONLY GAUSSIAN PARAMETER SCREENING',json.dumps(primary,indent=2),
            f'Existing target disk radius={radius:g}px; matching Gaussian sigma={summary["sigma_anchor"]:.3f}px.',
            'Selection: midpoint score <=0.30 for >=90% of target pairs in each adequately sampled type;',
            'core ring HAO/G >=0.80 for >=90% of centers. Fractions macro-averaged over images.',
            'Priority: closest half-height support to existing disk, then weakest lambda.',
            'Thresholds are declared design choices. Inspect threshold_sensitivity.csv and example PNGs.',
            'No feasible choice => do not force a recommendation.',*summary['limitations']]
    (out/'READ_ME.txt').write_text('\n\n'.join(text)+'\n')
    print('\n[Result]',json.dumps(primary,ensure_ascii=False))
    print('[Files]',out)
    print('Inspect recommendation.json, threshold_sensitivity.csv and example_*.png before training.')


if __name__=='__main__':
    main()
