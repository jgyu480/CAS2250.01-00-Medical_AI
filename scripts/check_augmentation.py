"""증강 구현 검사 + 실제 사진으로 증강 결과를 눈으로 확인한다.

python scripts/check_augmentation.py                 # 자기 검사만 (데이터 불필요)
python scripts/check_augmentation.py --id 001 --repeat 6
    → outputs/augmentation_preview/001_*.png
      각 증강을 하나씩만 적용한 그림 + 전체 적용을 반복한 그림.
      세포 점(노랑 BC/파랑 TC)과 CA 경계(초록)가 사진과 맞는지 직접 본다.

검사 항목
  1. LAB 변환 왕복 오차, Reinhard 자기 자신 변환 = 원본
  2. 강도 0인 Color Jitter/HED Jitter = 원본
  3. 같은 seed·epoch·index → 같은 결과, epoch가 바뀌면 다른 결과
  4. 한 증강을 꺼도 나머지 증강의 파라미터가 같은지(ablation 공정성)
  5. 색 증강은 점·마스크·위치 상자를 바꾸지 않음 (실제 데이터가 있을 때)
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths  # noqa: E402
from common.data.label_compare import boundary_band  # noqa: E402
from common.data.photometric import (  # noqa: E402
    apply_photometric, color_jitter, hed_jitter, lab_to_rgb, load_augmentation,
    reinhard, rgb_to_lab, lab_stats, sample_photometric
)

FAKE_STATS = dict(color_space='lab', split='train',
                  mean=dict(mu=[70, 20, -10], sigma=[5, 4, 3]),
                  std=dict(mu=[12, 8, 6], sigma=[2, 1.5, 1]))


def synthetic_image(seed=0, size=128):
    rng = np.random.default_rng(seed)
    base = np.array([200, 120, 180], float) + rng.normal(0, 25, (size, size, 3))
    yy, xx = np.mgrid[:size, :size]
    for _ in range(20):  # 짙은 보라색 '핵'
        cy, cx = rng.integers(0, size, 2)
        base[(yy - cy) ** 2 + (xx - cx) ** 2 < 25] = [80, 40, 120]
    return np.clip(base, 0, 255).astype(np.uint8)


def self_check():
    image = synthetic_image()

    back = lab_to_rgb(rgb_to_lab(image))
    assert np.abs(back.astype(int) - image).max() <= 1, 'LAB 왕복 오차'

    m, s = lab_stats(image)
    assert np.abs(reinhard(image, m, s).astype(int) - image).max() <= 1, 'Reinhard 항등'
    shifted = reinhard(image, [m[0] - 10, m[1] + 8, m[2]], s)
    m2, _ = lab_stats(shifted)
    assert abs(m2[0] - (m[0] - 10)) < 1.5 and abs(m2[1] - (m[1] + 8)) < 1.5, 'Reinhard 목표 평균'

    assert np.abs(color_jitter(image).astype(int) - image).max() <= 1, 'Color Jitter 항등'
    assert np.abs(hed_jitter(image, [1, 1, 1], [0, 0, 0]).astype(int) - image).max() <= 2, 'HED 항등'

    config = load_augmentation('default')
    config['hed_jitter']['enabled'] = True
    config['gaussian_blur']['enabled'] = True

    def params(cfg, seed, epoch, index):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index]))
        rng.integers(2), rng.integers(4)
        return sample_photometric(rng, cfg, FAKE_STATS)

    assert params(config, 42, 0, 3) == params(config, 42, 0, 3), '재현성'
    assert params(config, 42, 0, 3) != params(config, 42, 1, 3), 'epoch별 변화'

    # 한 증강을 꺼도 다른 증강의 값은 그대로여야 공정한 ablation이다.
    for name in config['order']:
        off = copy.deepcopy(config)
        off[name]['enabled'] = False
        for index in range(30):
            full, part = params(config, 42, 0, index), params(off, 42, 0, index)
            assert name not in part
            for other in part:
                assert part[other] == full[other], f'{name} 끈 뒤 {other} 값 변함'

    out = apply_photometric(image, params(config, 42, 0, 0), config['order'])
    assert out.dtype == np.uint8 and out.shape == image.shape
    print('[완료] 색 공간·Reinhard·Jitter 항등·재현성·ablation 파라미터 고정 검사')


def draw_labels(sample):
    cell = Image.fromarray(sample['cell_image'])
    d = ImageDraw.Draw(cell)
    for x, y, label in sample['cell_points']:
        color = (255, 220, 0) if label == 1 else (0, 120, 255)
        d.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)

    tissue = sample['tissue_image'].copy()
    ca = sample['tissue_target'] == 1
    edge = boundary_band(ca, 2)  # 축소해도 보이도록 두껍게
    tissue[edge] = [0, 230, 80]
    tissue = Image.fromarray(tissue)
    ImageDraw.Draw(tissue).rectangle(tuple(sample['cell_box'] * 1024),
                                     outline=(255, 30, 30), width=4)
    return cell, tissue


def save_grid(panels, path, size=320):
    cols = len(panels)
    canvas = Image.new('RGB', (cols * size, 2 * size + 24), 'white')
    for i, (title, cell, tissue) in enumerate(panels):
        canvas.paste(cell.resize((size, size)), (i * size, 24))
        canvas.paste(tissue.resize((size, size)), (i * size, size + 24))
        ImageDraw.Draw(canvas).text((i * size + 6, 6), title, fill='black')
    canvas.save(path)


def preview(pair_id, repeat):
    from common.data.dataset import OcelotDataset

    base = OcelotDataset('train')
    index = [r['pair_id'] for r in base.rows].index(pair_id)
    original = base[index]

    singles = [('original', original)]
    for name in ('geometric_only', 'randstainna', 'color_jitter', 'hed_jitter', 'gaussian_blur'):
        if name == 'geometric_only':
            preset = 'geometric_only'
        else:
            preset = {k: {'enabled': k == name}
                      for k in ('randstainna', 'color_jitter', 'hed_jitter', 'gaussian_blur')}
            preset['geometric'] = {'horizontal_flip': False, 'quarter_turn_rotation': False}
            preset[name]['p'] = 1.0
        try:
            dataset = OcelotDataset('train', augment=True, augmentation=preset)
        except ValueError as error:
            print(f'[건너뜀] {name}: {error}')
            continue
        sample = dataset[index]
        if name != 'geometric_only':
            # 색 증강은 라벨을 바꾸면 안 된다.
            np.testing.assert_array_equal(sample['cell_points'], original['cell_points'])
            np.testing.assert_array_equal(sample['tissue_target'], original['tissue_target'])
            np.testing.assert_array_equal(sample['cell_box'], original['cell_box'])
        singles.append((name, sample))

    folder = resolve_paths()['outputs'] / 'augmentation_preview'
    folder.mkdir(parents=True, exist_ok=True)
    save_grid([(t, *draw_labels(s)) for t, s in singles], folder / f'{pair_id}_single.png')

    try:
        dataset = OcelotDataset('train', augment=True, augmentation='default')
        rows = [('original', *draw_labels(original))]
        for epoch in range(repeat):
            dataset.set_epoch(epoch)
            rows.append((f'default e{epoch}', *draw_labels(dataset[index])))
        save_grid(rows, folder / f'{pair_id}_default_repeat.png')
    except ValueError as error:
        print(f'[건너뜀] default 반복 그림: {error}')
    print('[저장]', folder)
    print('[확인] 위: 세포 사진·점, 아래: 조직 사진·CA 경계(초록)·세포 위치(빨강)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', help='실제 train 샘플로 미리보기 (예: 001)')
    parser.add_argument('--repeat', type=int, default=6)
    args = parser.parse_args()
    self_check()
    if args.id:
        preview(args.id.zfill(3), args.repeat)


if __name__ == '__main__':
    main()
