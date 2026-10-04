"""두 FoV 위치와 정답을 그림으로 확인한다."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.dataset import OcelotDataset
from common.data.geometry import crop_tissue


def overlay(image, target, colors):
    result = image.astype(float).copy()
    for label, color in colors.items():
        selected = target == label
        result[selected] = (
            result[selected]*0.65 + np.array(color)*0.35
        )
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='001')
    parser.add_argument(
        '--split', choices=['train', 'val', 'test'], default='train'
    )
    parser.add_argument(
        '--labels', choices=['official', 'draft', 'reviewed'],
        default='official'
    )
    parser.add_argument('--augment', action='store_true')
    args = parser.parse_args()

    dataset = OcelotDataset(args.split, args.labels, args.augment)
    pair_id = args.id.zfill(3)
    indices = [
        i for i, row in enumerate(dataset.rows)
        if row['pair_id'] == pair_id
    ]
    if not indices:
        raise SystemExit(f'[오류] {args.split}에 {pair_id}가 없습니다.')

    sample = dataset[indices[0]]
    cell = Image.fromarray(sample['cell_image'])
    draw = ImageDraw.Draw(cell)

    for x, y, label in sample['cell_points']:
        color = (255, 220, 0) if label == 1 else (0, 100, 255)
        draw.ellipse((x-3, y-3, x+3, y+3), fill=color)

    tissue = overlay(
        sample['tissue_image'], sample['tissue_target'],
        {1: (0, 230, 80), 2: (230, 50, 230)}
    )
    box = sample['cell_box'] * 1024
    ImageDraw.Draw(tissue).rectangle(
        tuple(box), outline=(255, 30, 30), width=5
    )

    aligned = Image.fromarray(
        crop_tissue(sample['tissue_image'], sample['cell_box'])
    )
    target = overlay(
        sample['cell_image'], sample['cell_target'],
        {1: (255, 220, 0), 2: (0, 100, 255)}
    )

    # 그림의 글꼴 의존성을 줄이기 위해 짧은 영문 기호만 표시한다.
    titles = [
        'Cell: BC yellow / TC blue',
        'Tissue: CA green / UNK magenta',
        'Tissue ROI enlarged',
        f"Cell target: radius {dataset.settings['cell_radius_px']} px"
    ]

    canvas = Image.new('RGB', (1024, 1080), 'white')
    for index, panel in enumerate((cell, tissue, aligned, target)):
        x, y = (index % 2)*512, (index // 2)*540
        canvas.paste(panel.resize((512, 512)), (x, y+28))
        ImageDraw.Draw(canvas).text(
            (x+8, y+8), titles[index], fill='black'
        )

    folder = resolve_paths()['outputs'] / 'previews'
    folder.mkdir(parents=True, exist_ok=True)
    suffix = 'augmented' if args.augment else 'original'
    path = folder / f'{pair_id}_{args.labels}_{suffix}.png'
    canvas.save(path)

    info = dict(
        pair_id=pair_id,
        label_source=sample['label_source'],
        cell_box=sample['cell_box'].tolist(),
        transform=sample['transform']
    )
    path.with_suffix('.json').write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    print('[저장]', path)
    print('[위치]', sample['cell_box'].tolist())
    print('[변환]', sample['transform'])
    print('[안내] 공식 라벨 미리보기이며 직접 작성한 어노테이션은 아닙니다.')


if __name__ == '__main__':
    main()
