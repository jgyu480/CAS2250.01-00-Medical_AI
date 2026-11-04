# M1-1 설치 안내 (기존 GitHub 저장소에 새 파일만 추가)

이 ZIP은 **전체 저장소가 아니라 추가 파일 패치(overlay)** 입니다. 기존 M0/M1 파일을 수정하지 않습니다.

## 1. 로컬 / VESSL에 적용

1. 최신 저장소를 받아서 루트 폴더(`CAS2250.01-00-Medical_AI`)를 엽니다.
2. ZIP을 저장소의 **루트 폴더**에 압축 해제합니다. ZIP 내부의 `common/`, `models/`, `configs/`, `scripts/`가 기존 동일한 이름의 폴더와 합쳐져야 합니다. 이미 존재하는 M0/M1 파일은 변경되지 않습니다.
3. 프로젝트 루트에서 실행합니다.

```bash
python scripts/check_m1_1.py
```

4. 실제 M1 가중치가 준비되어 있다면:

```bash
python scripts/check_m1_1.py --real-m1-checkpoint outputs/m1/M1_seed42/best.pt
python scripts/train_m1_1.py --m1-checkpoint outputs/m1/M1_seed42/best.pt --seed 42 --run-name M1_1_seed42
```

각자의 M1 checkpoint 경로는 실제 `best.pt` 위치로 바꾸세요. `OCELOT_DATA_ROOT`, `SSL_WEIGHTS_PATH`, `OUTPUT_ROOT`는 기존 M0/M1 환경변수 그대로 사용합니다.

5. Validation/ablation/오류 분석:

```bash
python scripts/eval_m1_1.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split val --tune --ablation
python scripts/analyze_m1_1_errors.py --checkpoint outputs/m1_1/M1_1_seed42/best.pt --split val
```

## 2. GitHub에 업로드 (터미널)

```bash
git status

git add common/modules/gated_fusion.py models/M1_1/ configs/m1_1_train.json scripts/train_m1_1.py scripts/eval_m1_1.py scripts/analyze_m1_1_errors.py scripts/check_m1_1.py INSTALL_M1_1.md

git commit -m "Implement M1-1 frozen M1 spatial gated dual-head"

git push origin main
```

`git push` 전에 `git status`를 보고 예상치 못한 데이터·가중치·결과 파일을 올리지 않는지 확인하세요. 이 패치에는 학습된 checkpoint나 OCELOT 원본 데이터가 들어 있지 않습니다.

## 3. 결과와 완료 판정

- 코드 단위 테스트: `scripts/check_m1_1.py` PASS
- 실제 M1 체크포인트 일치: `--real-m1-checkpoint` PASS
- Stage A (5 epoch), Stage B (20 epoch) 정상 종료
- `best.pt`로 Validation 후처리 선택, ablation 결과 및 Gate 분포 분석 완료
- 3개 seed 재현성 확인 후 Test 1회 평가

상세 설계는 `models/M1_1/README.md` 참고.
