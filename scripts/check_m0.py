"""M0 점검: 좌표 정렬 → 출력 → 손실 → gradient 경로 → 짧은 학습 스텝.

본 학습이 아니다. 작은 크기(기본 256)로 줄인 train 2쌍만 사용한다.
공용 파일(docs/progress.json 등)은 수정하지 않는다.
결과: outputs/m0_check/report.json

python scripts/check_m0.py            # 기본 점검
python scripts/check_m0.py --steps 10 # 같은 batch로 10번 업데이트해 손실 감소 확인
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.data.tensors import TorchOcelotDataset, collate_pairs
from common.modules.exchange import crop_to_cell, place_in_tissue
from common.losses.m0_losses import focal_loss, m0_task_losses
from models.M0.model import build_model


def require(condition, message):
    if not condition:
        raise SystemExit("[FAIL] " + message)


def coordinate_map(size):
    """채널0 = 픽셀 중심의 x(0~1), 채널1 = y(0~1)."""
    centers = (torch.arange(size, dtype=torch.float64) + 0.5) / size
    y, x = torch.meshgrid(centers, centers, indexing="ij")
    return torch.stack([x, y])[None]


def check_alignment():
    tissue_size, cell_size = 64, 32
    boxes = torch.tensor([
        [0.5, 0.5, 0.75, 0.75],      # 001번 샘플 위치
        [0.0, 0.0, 0.2498, 0.2498],  # 224번 샘플 위치 (구석)
        [0.13, 0.61, 0.38, 0.86]
    ], dtype=torch.float64)

    for box in boxes:
        box = box[None]
        left, top, right, bottom = box[0].tolist()

        # 1) 조직→세포: 잘라낸 좌표가 상자 안의 기대 좌표와 같은가
        cropped = crop_to_cell(coordinate_map(tissue_size), box,
                               (cell_size, cell_size))
        u = (torch.arange(cell_size, dtype=torch.float64) + 0.5) / cell_size
        expect_x = left + u * (right - left)
        expect_y = top + u * (bottom - top)
        half = 0.5 / tissue_size
        keep_x = (expect_x >= half) & (expect_x <= 1 - half)
        keep_y = (expect_y >= half) & (expect_y <= 1 - half)
        error_x = (cropped[0, 0, 0] - expect_x)[keep_x].abs().max()
        error_y = (cropped[0, 1, :, 0] - expect_y)[keep_y].abs().max()
        require(max(error_x, error_y) < 1e-6, f"crop 좌표 오류 {box.tolist()}")

        # 2) 세포→조직: 상자 안에만 배치되고, 넓이가 상자와 같은가
        ones = torch.ones(1, 1, 128, 128, dtype=torch.float64)
        placed, coverage = place_in_tissue(ones, box,
                                           (tissue_size, tissue_size))
        area = (right - left) * (bottom - top) * tissue_size ** 2
        perimeter = 2 * ((right - left) + (bottom - top)) * tissue_size
        require(abs(coverage.sum().item() - area) <= perimeter,
                f"coverage 넓이 오류 {box.tolist()}")
        require(torch.equal(placed, coverage), "상자 밖 0 / 안 1 배치 오류")
        center = coordinate_map(tissue_size)[0]
        inside = coverage[0, 0].bool()
        require(((center[0][inside] >= left) & (center[0][inside] <= right)
                 & (center[1][inside] >= top)
                 & (center[1][inside] <= bottom)).all(),
                "coverage가 상자 밖에 있음")

        # 3) 배치 위치: 세포 좌표 지도를 배치하면 조직 좌표와 대응하는가
        cell_coords = coordinate_map(128)
        placed, coverage = place_in_tissue(cell_coords, box,
                                           (tissue_size, tissue_size))
        expect_u = (center[0] - left) / (right - left)
        pooled = round((right - left) * tissue_size)
        error = (placed[0, 0] - expect_u)[inside].abs().max().item()
        require(error <= 1.0 / pooled, f"배치 좌표 오류 {error:.4f}")

        # 4) 좌우 뒤집기와 순서를 바꿔도 결과가 같은가 (증강과 정렬 일치)
        feature = torch.randn(1, 4, tissue_size, tissue_size,
                              dtype=torch.float64)
        flipped_box = torch.tensor([[1 - right, top, 1 - left, bottom]],
                                   dtype=torch.float64)
        a = crop_to_cell(feature.flip(-1), flipped_box, (cell_size, cell_size))
        b = crop_to_cell(feature, box, (cell_size, cell_size)).flip(-1)
        require(torch.allclose(a, b, atol=1e-9), "뒤집기 후 crop 불일치")

    print("[완료] 좌표 정렬: crop·배치·coverage·뒤집기 (3개 위치)")


def check_losses():
    target = torch.zeros(1, 8, 8, dtype=torch.long)
    target[0, 2:4, 2:4] = 1
    target[0, 5:7, 5:7] = 2
    good = torch.full((1, 3, 8, 8), -8.)
    good.scatter_(1, target.unsqueeze(1), 8.)
    bad = good[:, [0, 2, 1]]
    require(focal_loss(good, target) < focal_loss(bad, target),
            "Focal Loss 방향 오류")
    print("[완료] Focal Loss 방향")


def resize_batch(batch, size):
    images = [
        F.interpolate(batch[key], size=(size, size), mode="bilinear",
                      align_corners=False)
        for key in ("cell_image", "tissue_image")
    ]
    targets = [
        F.interpolate(batch[key][:, None].float(), size=(size, size),
                      mode="nearest")[:, 0].long()
        for key in ("cell_target", "tissue_target")
    ]
    return images, targets, batch["cell_box"]


def grad_norm(parameters):
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().float().pow(2).sum().item()
    return total ** 0.5


def bn_stats(module):
    return {
        k: v.clone() for k, v in module.state_dict().items()
        if "running_" in k
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256,
                        choices=[128, 256, 512, 1024])
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    torch.manual_seed(42)
    start = time.time()

    check_alignment()
    check_losses()

    model = build_model().train()
    names = [n for n, m in (
        ("① encoder 뒤 세포→조직", model.encoder_c2t),
        ("② ASPP 뒤 조직→세포", model.aspp_t2c),
        ("③ decoder 뒤 조직→세포", model.decoder_t2c)
    ) if m is not None]
    print("[구성]", ", ".join(names))

    dataset = TorchOcelotDataset(split="train", label_source="official")
    batch = collate_pairs([dataset[0], dataset[1]])
    (cell_image, tissue_image), targets, boxes = resize_batch(
        batch, args.size
    )
    print("[입력] cell_box:", [[round(v, 4) for v in b] for b in
                              boxes.tolist()])

    before_bn = bn_stats(model)
    output = model(cell_image, tissue_image, boxes)
    for key, tensor in output.items():
        require(tuple(tensor.shape) == (2, 3, args.size, args.size),
                f"{key} 크기 오류: {tuple(tensor.shape)}")
        require(torch.isfinite(tensor).all(), f"{key}: NaN/Inf")
    print("[출력]", {k: list(v.shape) for k, v in output.items()})

    losses = m0_task_losses(output["cell_logits"], output["tissue_logits"],
                            *targets)
    require(all(torch.isfinite(v) for v in losses.values()), "손실 NaN/Inf")
    print("[손실] cell(Focal) {:.4f}, tissue(CE+Dice) {:.4f}".format(
        losses["cell"].item(), losses["tissue"].item()))

    # gradient 경로: 한 작업의 손실이 다른 갈래까지 전달되는가
    paths = {}
    model.zero_grad(set_to_none=True)
    losses["tissue"].backward(retain_graph=True)
    paths["tissue_loss→cell_encoder"] = grad_norm(model.cell.encoder.parameters())
    paths["tissue_loss→cell_head"] = grad_norm(model.cell.head.parameters())
    if model.encoder_c2t is not None:
        paths["tissue_loss→①"] = grad_norm(model.encoder_c2t.parameters())

    model.zero_grad(set_to_none=True)
    losses["cell"].backward()
    paths["cell_loss→tissue_encoder"] = grad_norm(model.tissue.encoder.parameters())
    paths["cell_loss→tissue_head"] = grad_norm(model.tissue.head.parameters())
    for name, module in (("②", model.aspp_t2c), ("③", model.decoder_t2c)):
        if module is not None:
            paths[f"cell_loss→{name}"] = grad_norm(module.parameters())

    for key, value in paths.items():
        print(f"[gradient] {key}: {value:.3e}")
    require(paths["tissue_loss→cell_head"] == 0, "조직 손실이 세포 head에 전달됨")
    require(paths["cell_loss→tissue_head"] == 0, "세포 손실이 조직 head에 전달됨")
    for key, value in paths.items():
        if "head" not in key:
            require(value > 0, f"gradient가 전달되지 않음: {key}")
    print("[완료] 전달 경로로 gradient 전달 / head는 각자 손실만 받음")

    after_bn = bn_stats(model)
    require(all(torch.equal(before_bn[k], after_bn[k]) for k in before_bn),
            "학습 모드에서 백본 BN 통계가 바뀜")
    print("[완료] 백본 BN 통계 고정")

    # 짧은 학습: 같은 batch를 반복해 손실이 줄어드는지 확인한다.
    optimizer = torch.optim.AdamW([
        {"params": model.encoder_parameters(), "lr": 1e-5},
        {"params": model.new_module_parameters(), "lr": 1e-4}
    ])
    watched = {
        "cell_encoder": next(model.cell.encoder.layer4.parameters()),
        "tissue_encoder": next(model.tissue.encoder.layer4.parameters()),
        "decoder": next(model.cell.decoder.parameters())
    }
    for name, module in (("①", model.encoder_c2t), ("②", model.aspp_t2c),
                         ("③", model.decoder_t2c)):
        if module is not None:
            watched[f"exchange{name}"] = next(module.parameters())
    snapshot = {k: v.detach().clone() for k, v in watched.items()}
    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(cell_image, tissue_image, boxes)
        step_losses = m0_task_losses(out["cell_logits"], out["tissue_logits"],
                                     *targets)
        total = step_losses["cell"] + 1.0 * step_losses["tissue"]
        require(torch.isfinite(total), f"{step}번째 스텝 손실 NaN/Inf")
        total.backward()
        optimizer.step()
        history.append({k: round(v.item(), 5)
                         for k, v in {**step_losses, "total": total}.items()})
        print(f"[스텝 {step + 1}/{args.steps}] total {total.item():.4f} "
              f"(cell {step_losses['cell'].item():.4f}, "
              f"tissue {step_losses['tissue'].item():.4f})")

    changed = {k: not torch.equal(snapshot[k], v.detach())
               for k, v in watched.items()}
    require(all(changed.values()), f"업데이트되지 않은 파라미터: {changed}")
    print("[완료] 두 백본과 새 부품 파라미터 업데이트")
    decreased = len(history) > 1 and history[-1]["total"] < history[0]["total"]
    print("[확인] 손실 감소:", "예" if decreased else "아니오 (스텝 수를 늘려 재확인)")

    folder = resolve_paths()["outputs"] / "m0_check"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": True,
        "size": args.size,
        "exchange": names,
        "initial_losses": {k: v.item() for k, v in losses.items()},
        "gradient_paths": paths,
        "steps": history,
        "loss_decreased": decreased,
        "torch": torch.__version__,
        "seconds": round(time.time() - start, 1),
        "training_started": False
    }
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print(f"[PASS] M0 점검 완료 ({report['seconds']}초). 본 학습은 시작하지 않음.")


if __name__ == "__main__":
    main()
