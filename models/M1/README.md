# M1 — Global Tissue-to-Cell Cross-Attention

M0와 동일한 두 개의 독립적인 SSL ResNet-50/ASPP/Decoder/Head 사용.

- Encoder 뒤 **Cell→Tissue는 없음**.
- ASPP 뒤 **Cell(Q)→전체 Tissue(K/V)** cross-attention (Tissue 16×16 tokens, 4 heads).
- Cell Query 위치를 `cell_box`에 의해 Tissue 좌표계 [0,1]로 변환하여 두 토큰의 고정 Fourier 위치 인코딩을 통일.
- Decoder 뒤 Tissue→Cell 좌표 crop은 M0와 동일하게 유지.
- `model(*images, boxes)` 결과 key와 optimizer parameter group API는 M0와 동일.

단독 테스트: `python scripts/check_m1_attention.py`.

학습/평가 스크립트는 M0와 같은 데이터, loss, hyperparameter, seed를 유지해야 한다. M0 기존 실행 중인 학습 파일은 수정하지 말 것.

학습 준비가 끝난 뒤: `python scripts/train_m1.py --seed 42 --run-name M1_seed42`
(본 학습은 시작하지 않음. `--smoke`를 사용하면 작은 실제 데이터셋으로 테스트.)
