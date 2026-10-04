"""2단계 기록을 확정하고, 준비 실패 뒤의 연쇄 오류를 고친다."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.labels import (
    TISSUE_RAW, TISSUE_TARGET, TISSUE_IGNORE_INDEX, MAPPING_SOURCE
)


def main():
    report_path = resolve_paths()['outputs'] / 'audit/report.json'
    if not report_path.is_file():
        raise SystemExit(
            '[오류] 먼저 python scripts/check_data.py를 실행하세요.'
        )

    report = json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('ok') or report.get('errors'):
        raise SystemExit('[오류] 데이터 검사 FAIL을 먼저 해결해야 합니다.')

    # 공식 설명에서 확인한 의미와 실제 파일에서 관찰한 번호를 대조한다.
    if TISSUE_RAW != {'BG': 1, 'CA': 2, 'UNK': 255}:
        raise SystemExit('[오류] labels.py의 조직 클래스 정의가 다릅니다.')

    for split, count in {'train': 400, 'val': 137, 'test': 126}.items():
        info = report['splits'][split]
        if info['valid_file_pairs'] != count:
            raise SystemExit(f'[오류] {split}: 검증된 데이터 개수가 다릅니다.')
        if set(map(int, info['mask_pixel_counts'])) - set(TISSUE_RAW.values()):
            raise SystemExit(f'[오류] {split}: 알 수 없는 조직 번호가 있습니다.')

    edits = {}

    path = ROOT / 'scripts/prepare_annotation.py'
    text = path.read_text(encoding='utf-8')
    old = "if not report.get('ok') or not report.get('tissue_mapping_validated'):"
    new = (
        "if not report.get('ok') or "
        "json.loads((ROOT / 'configs/common.json').read_text(encoding='utf-8'))"
        ".get('tasks', {}).get('tissue_classes') != {'BG': 1, 'CA': 2, 'UNK': 255}:"
    )

    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(
            '[오류] prepare_annotation.py가 예상 버전과 다릅니다. '
            '파일을 수정하지 않았습니다.'
        )

    edits[path] = text.replace(
        'Complete step 2 and confirm the tissue mapping first.',
        '먼저 데이터 검사와 공식 조직 클래스 설정을 완료하세요.'
    )

    path = ROOT / 'scripts/check_annotation_setup.py'
    text = path.read_text(encoding='utf-8')
    old = (
        "def read_csv(path):\n"
        "    with path.open(encoding='utf-8', newline='') as handle:"
    )
    new = (
        "def read_csv(path):\n"
        "    if not path.is_file():\n"
        "        raise SystemExit(f'[오류] 파일 없음: {path}\\n"
        "먼저 python scripts/prepare_annotation.py를 실행하세요.')\n"
        "    with path.open(encoding='utf-8', newline='') as handle:"
    )

    if old in text:
        text = text.replace(old, new, 1)
    elif new not in text:
        raise SystemExit(
            '[오류] check_annotation_setup.py가 예상 버전과 다릅니다. '
            '파일을 수정하지 않았습니다.'
        )

    # 다시 검사해도 이미 완료한 이후 단계의 기록을 낮추지 않는다.
    marker = "    (ROOT / 'docs/progress.json').write_text("
    keep = (
        "    previous_path = ROOT / 'docs/progress.json'\n"
        "    previous = json.loads(previous_path.read_text(encoding='utf-8')) "
        "if previous_path.exists() else {}\n"
        "    progress['completed_steps'] = sorted("
        "set(previous.get('completed_steps', [])) | {1, 2, 3})\n"
        "    progress['current_step'] = max(previous.get('current_step', 4), 4)\n"
    )

    if keep not in text:
        if marker not in text:
            raise SystemExit(
                '[오류] 진행 기록 코드가 예상과 다릅니다. '
                '파일을 수정하지 않았습니다.'
            )
        text = text.replace(marker, keep + marker, 1)
    edits[path] = text

    # 데이터 검사를 다시 실행해도 공식 클래스 확인 기록을 유지한다.
    path = ROOT / 'common/data/audit.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace(
        "'tissue_mapping_validated': False",
        "'tissue_mapping_validated': True"
    )
    pending = (
        "    report['warnings'].append("
        "'Tissue IDs observed; semantic mapping needs release documentation.')"
    )
    pending_multi = (
        "    report['warnings'].append(\n"
        "        'Tissue IDs observed; semantic mapping needs release documentation.'\n"
        "    )"
    )
    confirmed = (
        "    report['tissue_class_mapping'] = {'BG': 1, 'CA': 2, 'UNK': 255}"
    )

    if pending in text:
        text = text.replace(pending, confirmed, 1)
    elif pending_multi in text:
        text = text.replace(pending_multi, confirmed, 1)
    elif confirmed not in text:
        raise SystemExit(
            '[오류] audit.py가 예상 버전과 다릅니다. '
            '파일을 수정하지 않았습니다.'
        )
    edits[path] = text

    path = ROOT / 'scripts/check_data.py'
    text = path.read_text(encoding='utf-8').replace(
        'File audit passed; tissue mapping and geometry still pending.',
        '파일 검사 통과. 조직 클래스 확인 완료. 좌표 검사는 별도 단계입니다.'
    )
    edits[path] = text

    # 조직 라벨에는 원본 번호 1, 2, 255만 허용한다.
    path = ROOT / 'common/data/audit_io.py'
    text = path.read_text(encoding='utf-8')
    marker = '        values, counts = np.unique(mask, return_counts=True)\n'

    if 'Unexpected tissue IDs:' not in text:
        if marker not in text:
            raise SystemExit(
                '[오류] audit_io.py가 예상과 다릅니다. '
                '파일을 수정하지 않았습니다.'
            )
        text = text.replace(
            marker,
            marker +
            '        if set(map(int, values)) - {1, 2, 255}:\n'
            '            raise ValueError(f"Unexpected tissue IDs: {values}")\n',
            1
        )
    edits[path] = text

    common_path = ROOT / 'configs/common.json'
    common = json.loads(common_path.read_text(encoding='utf-8'))
    common.setdefault('tasks', {}).update(
        tissue_classes=TISSUE_RAW,
        tissue_target_classes=TISSUE_TARGET,
        tissue_ignore_index=TISSUE_IGNORE_INDEX
    )
    edits[common_path] = json.dumps(common, ensure_ascii=False, indent=2) + '\n'

    report.update(
        tissue_mapping_validated=True,
        tissue_class_mapping=TISSUE_RAW,
        tissue_mapping_source=MAPPING_SOURCE
    )
    report['warnings'] = [
        w for w in report.get('warnings', [])
        if not w.startswith('Tissue IDs observed;')
    ]
    edits[report_path] = json.dumps(report, ensure_ascii=False, indent=2) + '\n'

    # 변경할 내용을 전부 확인한 뒤 저장한다.
    # 기존 selection.csv와 tracking.csv는 수정하지 않는다.
    for path, content in edits.items():
        path.write_text(content, encoding='utf-8')

    print('[완료] 조직 클래스 확정: BG=1, CA=2, UNK=255')
    print('[완료] 준비 조건과 파일 누락 안내를 수정했습니다.')
    print('[다음] prepare_annotation.py → check_annotation_setup.py')


if __name__ == '__main__':
    main()
