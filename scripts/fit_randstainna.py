"""RandStainNA용 LAB 염색 통계를 train 사진으로만 계산한다.

python scripts/fit_randstainna.py            # train 400쌍의 cell+tissue 800장
python scripts/fit_randstainna.py --max 200  # 일부만 무작위로 사용(빠른 확인)

결과: configs/randstainna_lab_stats.json (작은 숫자 파일이라 Git에 올린다)
val/test 사진은 사용하지 않는다(정보 누출 방지).

방법(Shen et al., MICCAI 2022):
  사진마다 조직 픽셀(빈 유리 제외)의 LAB 채널별 평균 a_i, 표준편차 d_i를 구하고,
  여러 사진에 걸친 a, d의 분포를 채널별 정규분포로 근사한다.
  학습 때 이 분포에서 가상 템플릿을 뽑아 Reinhard 변환으로 맞춘다.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths  # noqa: E402
from common.data.annotators import read_csv_rows  # noqa: E402
from common.data.photometric import lab_stats, load_augmentation  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=0, help='사용할 최대 사진 쌍 수(0=전체)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--downsample', type=int, default=2,
                        help='통계 계산 속도를 위해 가로세로를 1/N로 줄임')
    args = parser.parse_args()

    paths = resolve_paths()
    rows = [r for r in read_csv_rows(paths['outputs'] / 'audit/manifest.csv')
            if r['split'] == 'train']
    if not rows:
        raise SystemExit('[오류] manifest에 train이 없습니다. check_data.py를 먼저 실행하세요.')
    rows.sort(key=lambda r: r['pair_id'])
    if args.max and args.max < len(rows):
        rng = np.random.default_rng(args.seed)
        rows = [rows[i] for i in sorted(rng.choice(len(rows), args.max, replace=False))]

    means, stds, by_fov = [], [], {'cell': [], 'tissue': []}
    for k, row in enumerate(rows, 1):
        for fov in ('cell', 'tissue'):
            with Image.open(paths['data_root'] / row[f'{fov}_image']) as image:
                image = image.convert('RGB')
                if args.downsample > 1:
                    image = image.reduce(args.downsample)
                m, s = lab_stats(np.asarray(image))
            means.append(m)
            stds.append(s)
            by_fov[fov].append(np.r_[m, s].astype(float))
        if k % 50 == 0 or k == len(rows):
            print(f'[계산] {k}/{len(rows)}쌍')

    means, stds = np.array(means, float), np.array(stds, float)
    stats = dict(
        method='RandStainNA (Shen et al., MICCAI 2022), Gaussian, diagonal, tissue pixels only',
        color_space='lab', channels=['L', 'a', 'b'], split='train',
        images=len(means), pairs=len(rows), fovs=['cell', 'tissue'],
        downsample=args.downsample, created=str(date.today()),
        mean=dict(mu=means.mean(0).round(4).tolist(), sigma=means.std(0).round(4).tolist()),
        std=dict(mu=stds.mean(0).round(4).tolist(), sigma=stds.std(0).round(4).tolist()),
        # 참고용: 두 시야의 색 분포 차이가 큰지 보고서에서 확인할 수 있게 남긴다.
        # 값 순서: [평균L, 평균a, 평균b, 표준편차L, 표준편차a, 표준편차b]
        per_fov={fov: dict(average=np.array(v).mean(0).round(4).tolist(),
                           spread=np.array(v).std(0).round(4).tolist())
                 for fov, v in by_fov.items()},
    )

    out = ROOT / load_augmentation()['randstainna']['stats_file']
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('[저장]', out.relative_to(ROOT))
    print('  LAB 평균의 평균', stats['mean']['mu'], '/ 흩어짐', stats['mean']['sigma'])
    print('  LAB 표준편차의 평균', stats['std']['mu'], '/ 흩어짐', stats['std']['sigma'])


if __name__ == '__main__':
    main()
