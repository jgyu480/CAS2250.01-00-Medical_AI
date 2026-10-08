"""증강된 train 샘플을 파일로 저장한다 (확인·공유용).

학습에서는 Dataset이 매 epoch 증강을 새로 만든다(온라인 증강).
이 스크립트는 그 결과를 눈으로 검토하거나 M0 실험 담당자에게 경로로 전달할 때 쓴다.

python scripts/export_augmented.py --preset default --ids 001 002 --copies 4
python scripts/export_augmented.py --preset default --all --copies 1

저장 위치: outputs/augmented/<preset>/  (Git 제외 폴더)
  cell/<ID>_e<k>.png            증강된 세포 사진 (무손실 PNG)
  tissue/<ID>_e<k>.png          증강된 조직 사진
  cell_points/<ID>_e<k>.csv     변환된 점 x,y,label (BC=1, TC=2)
  tissue_mask/<ID>_e<k>.png     변환된 조직 마스크, 원본 번호(BG=1, CA=2, UNK=255)
  manifest.csv                  위 경로 + 위치 상자 + 적용된 증강 파라미터
k는 epoch 번호이며, 같은 seed로 학습하면 그 epoch에 Dataset이 만드는 것과 같다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths  # noqa: E402
from common.data.dataset import OcelotDataset  # noqa: E402

TARGET_TO_RAW = np.array([1, 2, 255], dtype=np.uint8)  # BG, CA, UNK


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preset', default='default')
    parser.add_argument('--labels', default='official', help='official 또는 annotator:<ID>')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--copies', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    dataset = OcelotDataset('train', args.labels, augment=True, seed=args.seed,
                            augmentation=args.preset)
    ids = [r['pair_id'] for r in dataset.rows]
    wanted = ids if args.all else [i.zfill(3) for i in (args.ids or ['001'])]
    missing = set(wanted) - set(ids)
    if missing:
        raise SystemExit(f'[오류] train에 없는 ID: {sorted(missing)}')

    out = resolve_paths()['outputs'] / 'augmented' / args.preset
    for sub in ('cell', 'tissue', 'cell_points', 'tissue_mask'):
        (out / sub).mkdir(parents=True, exist_ok=True)

    records = []
    for epoch in range(args.copies):
        dataset.set_epoch(epoch)
        for pair_id in wanted:
            sample = dataset[ids.index(pair_id)]
            name = f'{pair_id}_e{epoch}'
            Image.fromarray(sample['cell_image']).save(out / 'cell' / f'{name}.png')
            Image.fromarray(sample['tissue_image']).save(out / 'tissue' / f'{name}.png')
            Image.fromarray(TARGET_TO_RAW[sample['tissue_target']]).save(
                out / 'tissue_mask' / f'{name}.png')
            with (out / 'cell_points' / f'{name}.csv').open('w', newline='') as h:
                csv.writer(h).writerows(
                    [[f'{x:g}', f'{y:g}', int(c)] for x, y, c in sample['cell_points']])
            records.append(dict(
                pair_id=pair_id, epoch=epoch, label_source=sample['label_source'],
                cell=f'cell/{name}.png', tissue=f'tissue/{name}.png',
                cell_points=f'cell_points/{name}.csv', tissue_mask=f'tissue_mask/{name}.png',
                cell_box=json.dumps([round(v, 6) for v in sample['cell_box'].tolist()]),
                transform=json.dumps(sample['transform'], ensure_ascii=False),
            ))
        print(f'[저장] epoch {epoch}: {len(wanted)}쌍')

    with (out / 'manifest.csv').open('w', encoding='utf-8', newline='') as h:
        writer = csv.DictWriter(h, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print('[경로]', out)


if __name__ == '__main__':
    main()
