"""모델 생성과 학습 없이 폴더·설정·경로를 확인한다."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths

def require(condition, message):
    if not condition:
        raise SystemExit('[ERROR] ' + message)

def main():
    configs = {}

    for name in ('M0', 'M1', 'M2', 'M3'):
        folder = ROOT / 'models' / name

        for filename in ('__init__.py', 'config.json', 'model.py', 'README.md'):
            require(
                (folder / filename).is_file(),
                f'Missing: {name}/{filename}'
            )

        configs[name] = json.loads(
            (folder / 'config.json').read_text(encoding='utf-8')
        )
        require(configs[name]['model'] == name, f'Wrong model name: {name}')
        print(f'[OK] {name}: folder and configuration')

    # M2와 M3는 forward 구조가 같고 gradient 처리만 다르다.
    for key in ('tissue_to_cell', 'cell_to_tissue', 'attention'):
        require(
            configs['M2'][key] == configs['M3'][key],
            f'M2/M3 mismatch: {key}'
        )
    require(configs['M2']['gradient_method'] == 'sum', 'M2 gradient method')
    require(configs['M3']['gradient_method'] == 'pcgrad', 'M3 gradient method')

    # M1↔M2는 역방향 전달 효과를 비교한다.
    for key in ('tissue_to_cell', 'attention', 'gradient_method'):
        require(
            configs['M1'][key] == configs['M2'][key],
            f'M1/M2 mismatch: {key}'
        )

    require(
        configs['M1']['cell_to_tissue']['after_encoder'] == 'none',
        'M1 reverse path'
    )
    require(
        configs['M2']['cell_to_tissue']['after_encoder'] == 'coordinate_place',
        'M2 reverse path'
    )

    # M0↔M2는 ASPP 뒤 전달 방식의 효과를 비교한다.
    require(
        configs['M0']['cell_to_tissue'] == configs['M2']['cell_to_tissue'],
        'M0/M2 reverse path'
    )
    require(
        configs['M0']['tissue_to_cell']['after_decoder']
        == configs['M2']['tissue_to_cell']['after_decoder'],
        'Decoder sharing'
    )
    print('[OK] Comparison settings')

    for name, path in resolve_paths().items():
        state = 'FOUND' if path.exists() else 'MISSING'
        print(f'[{state}] {name}: {path}')

    print('[INFO] File contents will be checked in later steps.')
    print('[OK] Step 1 structure check passed. No training was executed.')

if __name__ == '__main__':
    main()
