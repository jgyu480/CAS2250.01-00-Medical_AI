"""M2 점검. 데이터·SSL 가중치 없이 돌아가는 검사 + (있으면) 실제 데이터 검사.

python scripts/check_m2.py            # 합성 데이터만 사용 (CPU 1~2분)
python scripts/check_m2.py --data     # 실제 OCELOT train/val 쌍으로 정답 지도 확인
                                      # (check_data.py PASS 후, σ·λ는 configs/m2_train.json
                                      #  값 또는 --sigma/--lambda)

검사 항목
1. 정답 지도: select_gaussian_params.py의 식과 일치, 중심 값 1, 클래스별 분리,
   중심 1개·0개 처리, λ 범위, H_AO ≥ H_A, 보고서 그림 10 숫자, 뒤집기·회전 일치.
2. 손실: 가중치 마스크(골짜기 포함), 정규화, 기울기 전달.
3. 후처리: 정답 지도에서 만든 출력으로 모든 중심을 찾는지 (가까운 쌍 포함).
4. 모델: M0와 같은 가중치 이름, 세포 head만 2채널, 출력 크기, 몇 step 학습 시 손실 감소.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.backbones.ssl_resnet import SSLResNet50
from common.data.augment import transform_pair
from common.data.center_heatmaps import (
    class_heatmap, points_to_heatmaps, separation_maps
)
from common.losses.heatmap_losses import foreground_mask, weighted_mse
from common.metrics.heatmap_eval import detect_cells_heatmap
from common.metrics.m0_eval import CellF1
from models.M0.model import M0
from models.M2.model import M2


def screening_maps():
    """σ·λ 선정 스크립트의 maps 함수 (같은 식인지 비교용)."""
    path = ROOT / "scripts/select_gaussian_params.py"
    spec = importlib.util.spec_from_file_location("select_gaussian_params", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.maps


def brute(centers, sigma, lam, size, maps):
    yy, xx = np.mgrid[0:size, 0:size]
    pixels = np.c_[xx.ravel(), yy.ravel()].astype(float)
    if len(centers) == 0:
        return [np.zeros((size, size))] * 3
    tree = cKDTree(centers)
    if len(centers) >= 2:
        dist = tree.query(pixels, k=2)[0]
    else:
        dist = np.c_[tree.query(pixels, k=1)[0], np.full(len(pixels), np.inf)]
    return [m.reshape(size, size) for m in maps(dist, sigma, lam)]


def check_targets():
    maps = screening_maps()
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(25):
        size = 160
        pts = np.unique(rng.integers(0, size, (int(rng.integers(0, 40)), 2)), axis=0).astype(float)
        sigma = float(rng.choice([4, 6, 8, 12]))
        lam = float(rng.choice([0, .5, .9, .99]))
        for mode, ref in zip(("gaussian", "ha", "hao"), brute(pts, sigma, lam, size, maps)):
            heat = class_heatmap(pts, sigma, lam, mode, size)
            worst = max(worst, float(np.abs(heat - ref).max()))
            if len(pts):
                assert np.allclose(heat[pts[:, 1].astype(int), pts[:, 0].astype(int)], 1, atol=1e-6)
    assert worst < 1e-6, worst

    pts = np.array([[50, 50, 1], [60, 50, 2], [70, 50, 1]], float)
    heat = points_to_heatmaps(pts, 6, .9, "hao", 128)
    assert np.array_equal(heat[0], class_heatmap(pts[pts[:, 2] == 1, :2], 6, .9, "hao", 128))
    assert np.array_equal(heat[1], class_heatmap(pts[pts[:, 2] == 2, :2], 6, .9, "hao", 128))
    one = np.array([[30., 40.]])
    assert np.array_equal(class_heatmap(one, 6, .9, "hao", 128),
                          class_heatmap(one, 6, .9, "gaussian", 128))
    assert points_to_heatmaps(np.zeros((0, 3)), 6, .9).sum() == 0
    for bad in (1.0, -0.1):
        try:
            class_heatmap(one, 6, bad)
        except ValueError:
            pass
        else:
            raise AssertionError("λ 범위 검사가 동작하지 않습니다.")

    many = rng.integers(0, 256, (80, 2)).astype(float)
    assert (class_heatmap(many, 8, .9, "hao", 256) >= class_heatmap(many, 8, .9, "ha", 256) - 1e-7).all()

    for gap, expect in ((14.0, (0.084, 0.203)), (20 * np.sqrt(2), (0.050, 0.275))):
        d = cKDTree([[0, 0], [gap, 0]]).query([[gap / 2, 0]], k=2)[0]
        _, ha, hao = separation_maps(d[:, 0], d[:, 1], 12, .9)
        assert (round(float(ha[0]), 3), round(float(hao[0]), 3)) == expect, (gap, ha, hao)

    pts = np.c_[rng.integers(0, 1024, (150, 2)), rng.integers(1, 3, 150)].astype(float)
    pts = pts[np.unique(pts[:, :2], axis=0, return_index=True)[1]]
    base = points_to_heatmaps(pts, 6, .9)
    image = np.zeros((1024, 1024, 3), np.uint8)
    mask = np.zeros((1024, 1024), np.int64)
    for flip in (False, True):
        for turns in range(4):
            *_, moved, _ = transform_pair(image, image, mask, pts, [0, 0, 1, 1], flip, turns)
            expected = np.rot90(base[:, :, ::-1] if flip else base, turns, axes=(1, 2))
            assert np.allclose(points_to_heatmaps(moved, 6, .9), expected, atol=1e-6)
    print(f"PASS 정답 지도: 선정 스크립트 식과 최대 차이 {worst:.1e}, 중심 1, 클래스별, "
          "중심 1·0개, λ 범위, H_AO≥H_A, 그림 10 숫자, 뒤집기·회전")


def check_loss():
    pts = np.array([[40, 64, 1], [54, 64, 1], [100, 30, 2]], float)
    target = torch.from_numpy(points_to_heatmaps(pts, 6, .9, "hao", 128))[None]
    mask = foreground_mask(target)
    assert mask[0, 0, 64, 47], "가까운 두 BC 사이 골짜기가 가중치 영역에 포함돼야 합니다."
    assert not mask[0, 0, 5, 5]
    logits = torch.zeros_like(target, requires_grad=True)
    loss = weighted_mse(logits, target)
    expected = ((1 + 10 * mask.float()) * (0.5 - target) ** 2).sum() / max(
        mask.sum().item(), 0.005 * target.numel())
    assert torch.allclose(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    empty = torch.zeros(1, 2, 64, 64)
    assert torch.isfinite(weighted_mse(torch.zeros_like(empty), empty))
    print("PASS 손실: 골짜기 포함 가중치, 정규화, 기울기, 세포 없는 사진")


def check_postprocess():
    pts = np.array([[200, 200, 1], [214, 200, 1], [400, 380, 2], [409, 391, 2],
                    [600, 600, 1], [612, 600, 2], [800, 120, 2]], float)
    for mode in ("gaussian", "ha", "hao"):
        heat = torch.from_numpy(points_to_heatmaps(pts, 6, .9, mode)).clamp(1e-6, 1 - 1e-6)
        logits = torch.logit(heat)[None]
        preds = detect_cells_heatmap(logits, sigma=0.0, min_distance=3, threshold=0.3)[0]
        metric = CellF1()
        metric.update(preds, [tuple(int(v) for v in p) for p in pts])
        score = metric.compute()
        assert score["mF1"] > 0.999, (mode, preds)
    try:
        detect_cells_heatmap(logits, threshold=0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("threshold 0 검사가 동작하지 않습니다.")
    print("PASS 후처리: 세 정답 지도 모두 가까운 쌍(12~14px)까지 모든 중심을 찾음")


def check_model():
    torch.manual_seed(0)
    m0 = M0(SSLResNet50(), SSLResNet50(),
            json.loads((ROOT / "models/M0/config.json").read_text(encoding="utf-8")))
    m2 = M2(SSLResNet50(), SSLResNet50())
    a, b = m0.state_dict(), m2.state_dict()
    assert set(a) == set(b)
    different = [k for k in a if a[k].shape != b[k].shape]
    assert different == ["cell.head.weight", "cell.head.bias"], different
    assert tuple(b["cell.head.weight"].shape) == (2, 256, 1, 1)
    assert m0.config["tissue_to_cell"] == m2.config["tissue_to_cell"]
    assert m0.config["cell_to_tissue"] == m2.config["cell_to_tissue"]

    size = 128
    cell = torch.randn(2, 3, size, size)
    tissue = torch.randn(2, 3, size, size)
    box = torch.tensor([[0.375, 0.375, 0.625, 0.625]] * 2)
    pts = np.array([[40, 60, 1], [52, 60, 1], [90, 30, 2]], float)
    target = torch.from_numpy(points_to_heatmaps(pts, 4, .9, "hao", size))[None].repeat(2, 1, 1, 1)
    m2.train()
    optimizer = torch.optim.AdamW(m2.new_module_parameters(), lr=1e-3)
    for p in m2.encoder_parameters():
        p.requires_grad_(False)
    losses = []
    for _ in range(8):
        out = m2(cell, tissue, box)
        assert out["cell_logits"].shape == (2, 2, size, size)
        assert out["tissue_logits"].shape == (2, 3, size, size)
        loss = weighted_mse(out["cell_logits"], target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    print(f"PASS 모델: M0와 같은 가중치 이름, 세포 head만 2채널, 출력 크기, "
          f"8 step 손실 {losses[0]:.3f}→{losses[-1]:.3f}")


def check_data(sigma, lam):
    from common.training.m2_loop import HeatmapDataset
    target = {"mode": "hao", "scope": "classwise", "sigma": sigma, "lambda": lam}
    for split, augment in (("train", True), ("val", False)):
        data = HeatmapDataset(target, split=split, augment=augment, seed=42)
        for index in (0, len(data) // 2):
            data.set_epoch(1)
            sample = data[index]
            heat = sample["cell_heatmap"].numpy()
            pts = sample["cell_points"].numpy()
            assert heat.shape == (2, 1024, 1024) and heat.dtype == np.float32
            assert 0 <= heat.min() and heat.max() <= 1
            for x, y, c in pts:
                assert abs(heat[int(c) - 1, int(round(y)), int(round(x))] - 1) < 1e-6
            disk = sample["cell_target"].numpy()
            for x, y, c in pts:
                assert disk[int(round(y)), int(round(x))] == int(c)
            print(f"  {split} {sample['pair_id']}: 세포 {len(pts)}개, "
                  f"지도 최대 {heat.max():.3f}, 전경 픽셀 {(heat >= 0.2).mean():.2%}")
    print("PASS 실제 데이터: 증강 뒤 점 위치에서 해당 클래스 지도 값 1, M0 원형 정답과 같은 위치")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--lambda", dest="lam", type=float)
    args = parser.parse_args()
    check_targets()
    check_loss()
    check_postprocess()
    check_model()
    if args.data:
        config = json.loads((ROOT / "configs/m2_train.json").read_text(encoding="utf-8"))
        sigma = args.sigma if args.sigma is not None else config["cell_target"]["sigma"]
        lam = args.lam if args.lam is not None else config["cell_target"]["lambda"]
        if sigma is None or lam is None:
            raise SystemExit("[중단] σ·λ가 비어 있습니다. --sigma/--lambda를 주세요.")
        check_data(float(sigma), float(lam))
    print("M2 점검 완료")
