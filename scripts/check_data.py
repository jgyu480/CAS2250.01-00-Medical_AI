"""전체 데이터 검사. 이미지·가중치를 학습에 사용하지 않는다."""

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.audit import audit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--zip',
        type=Path,
        help='Optional official ZIP checksum verification'
    )
    args = parser.parse_args()
    paths = resolve_paths()
    report, rows = audit_dataset(paths['data_root'])

    report['environment'] = {
        'python': platform.python_version(),
        'platform': platform.platform(),
        'numpy': np.__version__,
        'pillow': PIL.__version__
    }

    # 공식 페이지의 MD5와 비교해 다운로드 파일이 같은지 확인한다.
    if args.zip is not None:
        try:
            digest = hashlib.md5()
            with args.zip.open('rb') as handle:
                for chunk in iter(
                    lambda: handle.read(1024 * 1024), b''
                ):
                    digest.update(chunk)

            report['zip_md5'] = digest.hexdigest()
            if report['zip_md5'] != '215230295b0440b4dc356519ef9ff644':
                report['errors'].append(
                    'ZIP MD5 does not match the v1.0.1 release.'
                )
        except OSError as exc:
            report['errors'].append(f'ZIP: {exc}')

    report['ok'] = not report['errors']
    output = paths['outputs'] / 'audit'
    output.mkdir(parents=True, exist_ok=True)

    (output / 'report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8'
    )

    fields = [
        'pair_id', 'split', 'cell_image', 'tissue_image',
        'cell_points', 'tissue_mask', 'slide_name', 'organ'
    ]
    with (output / 'manifest.csv').open(
        'w', encoding='utf-8', newline=''
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    for split, info in report['splits'].items():
        print(
            f'[{split}] pairs={info["complete_pairs"]}, '
            f'valid={info["valid_file_pairs"]}, cells={info["cells"]}'
        )
        print('  mask pixel counts:', info['mask_pixel_counts'])
        print('  organs:', info['organs'])

    print(
        'Metadata example:',
        json.dumps(report.get('metadata_example'), ensure_ascii=False, indent=2)
    )

    for message in report['warnings']:
        print('[WARN]', message)
    for message in report['errors'][:15]:
        print('[ERROR]', message)

    print('Saved:', output / 'report.json')
    print('Saved:', output / 'manifest.csv')

    if report['ok']:
        print('[PASS] 파일 검사 통과. 조직 클래스 확인 완료. 좌표 검사는 별도 단계입니다.')
    else:
        print('[FAIL] See report.json for all errors.')

    raise SystemExit(0 if report['ok'] else 1)


if __name__ == '__main__':
    main()
