"""GPU 없이 OCELOT 쌍을 읽는다. 네 모델이 동일한 Dataset을 사용한다."""

import csv
import json
import numpy as np
from PIL import Image

from common.paths import PROJECT_ROOT, resolve_paths
from .audit_metadata import load_metadata
from .labels import tissue_to_target
from .geometry import cell_box
from .targets import read_points, points_to_target
from .augment import transform_pair
from .photometric import (
    load_augmentation, load_stain_stats, sample_photometric, apply_photometric
)
from .annotators import (
    parse_label_source, load_annotation_config, label_paths, require_complete
)


def read_table(path):
    if not path.is_file():
        raise ValueError(f'필요한 CSV가 없습니다: {path}')

    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class OcelotDataset:
    def __init__(
        self, split='train', label_source='official', augment=False, seed=42,
        augmentation='default'
    ):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split은 train/val/test 중 하나입니다.')

        # annotator:<작성자ID>는 해당 작성자가 독립 작성한 라벨을 쓴다.
        self.annotator = parse_label_source(label_source)
        if (
            label_source not in ('official', 'draft', 'reviewed')
            and self.annotator is None
        ):
            raise ValueError(
                'label_source는 official/draft/reviewed/annotator:<ID>입니다.'
            )

        if split != 'train' and (augment or label_source != 'official'):
            raise ValueError('val/test는 공식 라벨과 원본 사진만 사용합니다.')

        paths = resolve_paths()
        report = json.loads(
            (paths['outputs'] / 'audit/report.json').read_text(encoding='utf-8')
        )
        if not report.get('ok'):
            raise ValueError('데이터 파일 검사를 먼저 통과해야 합니다.')

        self.root, self.split = paths['data_root'], split
        self.rows = sorted(
            (
                row for row in read_table(paths['outputs'] / 'audit/manifest.csv')
                if row['split'] == split
            ),
            key=lambda row: row['pair_id']
        )

        if (
            not self.rows
            or len({r['pair_id'] for r in self.rows}) != len(self.rows)
        ):
            raise ValueError('manifest에 데이터가 없거나 ID가 중복됐습니다.')

        self.metadata, _ = load_metadata(self.root / 'metadata.json')
        self.settings = json.loads(
            (PROJECT_ROOT / 'configs/dataset.json').read_text(encoding='utf-8')
        )

        if (
            self.settings['cell_target_classes'] != {'empty': 0, 'BC': 1, 'TC': 2}
            or self.settings['tissue_target_classes'] != {'BG': 0, 'CA': 1, 'UNK': 2}
            or self.settings['tissue_ignore_index'] != 2
        ):
            raise ValueError('학습용 클래스 설정이 공통 정의와 다릅니다.')

        self.label_source, self.augment = label_source, augment

        # augment=True일 때 configs/augmentation.json의 preset을 사용한다.
        self.aug_config, self.stain_stats = None, None
        if augment:
            self.aug_config = load_augmentation(augmentation)
            if self.aug_config['randstainna']['enabled']:
                self.stain_stats = load_stain_stats(
                    self.aug_config['randstainna']['stats_file']
                )
        self.seed, self.epoch, self.manual_ids = seed, 0, set()

        if self.annotator is not None:
            annotation = load_annotation_config()
            selected = read_table(PROJECT_ROOT / annotation['selection_csv'])
            self.manual_ids = {row['pair_id'] for row in selected}

            if (
                len(selected) != annotation['per_organ'] * 6
                or len(self.manual_ids) != len(selected)
                or not self.manual_ids <= {r['pair_id'] for r in self.rows}
            ):
                raise ValueError('직접 어노테이션 목록이 manifest와 다릅니다.')

            # 선택한 24쌍 전체가 완료돼야 한다. 공식 라벨로 대체하지 않는다.
            require_complete(self.annotator, self.manual_ids, annotation)

        elif label_source != 'official':
            annotation = json.loads(
                (PROJECT_ROOT / 'configs/annotation.json').read_text(
                    encoding='utf-8'
                )
            )
            selected = read_table(PROJECT_ROOT / annotation['selection_csv'])
            tracking = read_table(PROJECT_ROOT / annotation['tracking_csv'])
            self.manual_ids = {row['pair_id'] for row in selected}
            states = {row['pair_id']: row for row in tracking}

            if (
                len(selected) != annotation['per_organ'] * 6
                or len(self.manual_ids) != len(selected)
                or len(states) != len(tracking)
                or set(states) != self.manual_ids
                or not self.manual_ids <= {r['pair_id'] for r in self.rows}
            ):
                raise ValueError('직접 어노테이션 목록과 기록이 불일치합니다.')

            self.manual_root = (
                PROJECT_ROOT / annotation[f'{label_source}_root']
            )

            # 선택한 전체 24쌍이 완료돼야 해당 직접 라벨 모드를 허용한다.
            for pair_id in self.manual_ids:
                state = states[pair_id]
                status = (
                    'draft_status' if label_source == 'draft' else 'review_status'
                )

                if state[status] != 'completed' or not state['annotator'].strip():
                    raise ValueError(
                        f'{pair_id}: {label_source} 라벨이 미완료입니다.'
                    )

                if label_source == 'reviewed' and (
                    state['draft_status'] != 'completed'
                    or not state['reviewer'].strip()
                    or state['reviewer'].strip() == state['annotator'].strip()
                ):
                    raise ValueError(f'{pair_id}: 독립 검토 기록이 필요합니다.')

                for task, suffix in (('cell', 'csv'), ('tissue', 'png')):
                    path = self.manual_root / task / f'{pair_id}.{suffix}'
                    if not path.is_file():
                        raise ValueError(
                            f'{pair_id}: {task} 직접 라벨 파일이 없습니다.'
                        )

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        # 이후 학습에서 epoch마다 재현 가능한 증강을 바꾸는 데 사용한다.
        self.epoch = int(epoch)

    def __getitem__(self, index):
        row = self.rows[index]
        pair_id = row['pair_id']
        record = self.metadata[pair_id]

        if (
            record['subset'] != self.split
            or record['slide_name'] != row['slide_name']
        ):
            raise ValueError(f'{pair_id}: manifest와 metadata가 다릅니다.')

        images = []
        for key in ('cell_image', 'tissue_image'):
            with Image.open(self.root / row[key]) as image:
                if image.mode != 'RGB' or image.size != (1024, 1024):
                    raise ValueError(
                        f'{pair_id}: 1024x1024 RGB 사진이 아닙니다.'
                    )
                images.append(np.asarray(image).copy())

        cell_path = self.root / row['cell_points']
        mask_path = self.root / row['tissue_mask']
        used_source = 'official'

        # 직접 라벨 대상만 교체하고 나머지는 공식 라벨을 사용한다.
        if pair_id in self.manual_ids and self.annotator is not None:
            manual = label_paths(self.annotator, pair_id)
            cell_path, mask_path = manual['cell'], manual['tissue']
            used_source = self.label_source
        elif pair_id in self.manual_ids:
            cell_path = self.manual_root / 'cell' / f'{pair_id}.csv'
            mask_path = self.manual_root / 'tissue' / f'{pair_id}.png'
            used_source = self.label_source

        points = read_points(cell_path)
        with Image.open(mask_path) as image:
            tissue_target = tissue_to_target(np.asarray(image))

        if tissue_target.shape != (1024, 1024):
            raise ValueError(f'{pair_id}: 조직 라벨 크기가 다릅니다.')

        box = cell_box(record)
        flip, turns, color = False, 0, {}

        if self.augment:
            rng = np.random.default_rng(
                np.random.SeedSequence([self.seed, self.epoch, index])
            )
            # 기존 뒤집기·회전 난수 순서를 유지한다. 꺼진 증강도 난수는 소모한다.
            flip = bool(rng.integers(2))
            turns = int(rng.integers(4))
            geometric = self.aug_config['geometric']
            flip = flip and geometric['horizontal_flip']
            turns = turns if geometric['quarter_turn_rotation'] else 0

            # 색 증강은 사진만 바꾼다. 점·마스크·위치 상자는 그대로다.
            order = self.aug_config['order']
            color['cell'] = sample_photometric(
                rng, self.aug_config, self.stain_stats
            )
            color['tissue'] = (
                color['cell']
                if self.aug_config['shared_photometric_between_fovs']
                else sample_photometric(rng, self.aug_config, self.stain_stats)
            )
            images = [
                apply_photometric(images[0], color['cell'], order),
                apply_photometric(images[1], color['tissue'], order)
            ]

        cell, tissue, tissue_target, points, box = transform_pair(
            *images, tissue_target, points, box, flip, turns
        )
        cell_target = points_to_target(
            points, self.settings['cell_radius_px']
        )

        return dict(
            pair_id=pair_id,
            cell_image=cell,
            tissue_image=tissue,
            cell_points=points,
            cell_target=cell_target,
            tissue_target=tissue_target,
            cell_box=box,
            label_source=used_source,
            transform=dict(
                horizontal_flip=flip, quarter_turns=turns,
                preset=self.aug_config['preset'] if self.augment else None,
                photometric=color
            )
        )
