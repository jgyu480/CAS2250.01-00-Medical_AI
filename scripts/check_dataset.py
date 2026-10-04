"""전체 데이터와 좌표 변환을 검사한다. 모델 학습은 실행하지 않는다."""

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.dataset import OcelotDataset
from common.data.geometry import points_in_tissue
from common.data.targets import points_to_target
from common.data.labels import tissue_to_target
from common.data.augment import transform_pair


def self_check():
    # 중앙이 아닌 영역으로 검사해 중앙 crop만 하는 오류를 잡는다.
    size = 16
    image = np.arange(size*size*3, dtype=np.uint8).reshape(size, size, 3)
    mask = np.arange(size*size).reshape(size, size) % 3
    points = np.array([[2., 3., 1.], [12., 11., 2.]])
    box = np.array([0.1, 0.3, 0.35, 0.55])
    original_positions = points_in_tissue(points, box, size)

    for flip in (False, True):
        for turns in range(4):
            cell, tissue, new_mask, new_points, new_box = transform_pair(
                image, image, mask, points, box, flip, turns
            )

            expected_image = image[:, ::-1] if flip else image
            expected_image = np.rot90(expected_image, turns)
            np.testing.assert_array_equal(cell, expected_image)
            np.testing.assert_array_equal(tissue, expected_image)

            expected_mask = np.rot90(
                mask[:, ::-1] if flip else mask, turns
            )
            np.testing.assert_array_equal(new_mask, expected_mask)

            # 큰 사진의 위치를 따로 변환한 결과와 비교한다.
            expected_positions = original_positions.copy()
            if flip:
                expected_positions[:, 0] = (
                    size - 1 - expected_positions[:, 0]
                )

            for _ in range(turns):
                x = expected_positions[:, 0].copy()
                y = expected_positions[:, 1].copy()
                expected_positions[:, 0] = y
                expected_positions[:, 1] = size - 1 - x

            np.testing.assert_allclose(
                points_in_tissue(new_points, new_box, size),
                expected_positions
            )

            for old, new in zip(points, new_points):
                np.testing.assert_array_equal(
                    cell[int(new[1]), int(new[0])],
                    image[int(old[1]), int(old[0])]
                )

    np.testing.assert_array_equal(
        tissue_to_target(np.array([[1, 2, 255]])), [[0, 1, 2]]
    )

    try:
        tissue_to_target(np.array([[0]]))
    except ValueError:
        pass
    else:
        raise AssertionError('알 수 없는 조직 번호를 거부하지 않았습니다.')

    near = np.array([[5., 5., 1.], [9., 5., 2.]])
    target = points_to_target(near, radius=3, size=size)
    assert target[5, 5] == 1
    assert target[5, 9] == 2
    assert target[0, 0] == 0

    np.testing.assert_array_equal(
        target, points_to_target(near[::-1], radius=3, size=size)
    )
    assert not points_to_target(np.empty((0, 3)), size=size).any()

    print('[완료] 클래스·빈 점 목록·원 겹침·8가지 위치 변환 검사')


def main():
    self_check()
    results, errors = {}, []

    for split, expected in {'train': 400, 'val': 137, 'test': 126}.items():
        dataset = OcelotDataset(split=split)
        if len(dataset) != expected:
            errors.append(
                f'{split}: {len(dataset)}쌍, 예상 {expected}쌍'
            )

        valid = 0
        for index in range(len(dataset)):
            pair_id = dataset.rows[index]['pair_id']

            try:
                sample = dataset[index]
                if (
                    sample['cell_target'].shape != (1024, 1024)
                    or not np.isin(sample['cell_target'], [0, 1, 2]).all()
                    or not np.isin(sample['tissue_target'], [0, 1, 2]).all()
                ):
                    raise ValueError('학습용 정답 형식이 다릅니다.')
                valid += 1

            except (ValueError, KeyError, OSError) as error:
                errors.append(f'{split}/{pair_id}: {error}')

            if (index+1) % 100 == 0 or index+1 == len(dataset):
                print(f'[점검] {split}: {index+1}/{len(dataset)}')

        results[split] = dict(total=len(dataset), valid=valid)

    output = resolve_paths()['outputs'] / 'dataset'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(
        json.dumps(
            dict(
                ok=not errors,
                splits=results,
                errors=errors,
                training_started=False
            ),
            ensure_ascii=False,
            indent=2
        ) + '\n',
        encoding='utf-8'
    )

    if errors:
        for message in errors[:15]:
            print('[오류]', message)
        raise SystemExit(
            '[FAIL] dataset/report.json을 확인하세요. '
            '진척도를 올리지 않았습니다.'
        )

    progress_path = ROOT / 'docs/progress.json'
    progress = json.loads(progress_path.read_text(encoding='utf-8'))

    if not {1, 2, 3} <= set(progress.get('completed_steps', [])):
        raise SystemExit('[오류] 어노테이션 준비 검사부터 완료하세요.')

    progress.update(
        completed_steps=sorted(set(progress['completed_steps']) | {4}),
        current_step=max(progress.get('current_step', 4), 5),
        status='dataset_geometry_checks_passed',
        training_started=False
    )
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    print('[PASS] 전체 Dataset 검사 완료. 4/10단계, 구현 진행률 40%.')
    print('[안내] 학습 실행 없음. 어노테이션 정확성은 별도 검토합니다.')


if __name__ == '__main__':
    main()
