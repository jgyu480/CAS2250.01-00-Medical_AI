"""맥북과 VESSL에서 데이터·가중치·결과 경로를 처리한다."""

import json
import os
from pathlib import Path

# common/paths.py를 기준으로 프로젝트 최상위를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def resolve_paths():
    settings = json.loads(
        (PROJECT_ROOT / 'configs/common.json').read_text(encoding='utf-8')
    )
    env_names = {
        'data_root': 'OCELOT_DATA_ROOT',
        'ssl_weights': 'SSL_WEIGHTS_PATH',
        'outputs': 'OUTPUT_ROOT'
    }
    result = {}

    for key, env_name in env_names.items():
        # 환경변수가 없으면 공통 JSON의 기본 경로를 사용한다.
        value = os.environ.get(env_name) or settings['paths'][key]
        path = Path(value).expanduser()

        # 상대 경로는 터미널 위치가 아니라 프로젝트 기준이다.
        if path.is_absolute():
            result[key] = path.resolve()
        else:
            result[key] = (PROJECT_ROOT / path).resolve()

    return result
