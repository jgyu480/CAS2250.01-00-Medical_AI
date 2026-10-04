import argparse
import hashlib
import json
import platform
import sys
import urllib.request
from pathlib import Path
import torch
import torchvision
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.paths import resolve_paths
from common.backbones.ssl_resnet import load_ssl_encoder
from common.modules.segmentation import TaskBranch
from common.losses.segmentation import dice_loss, task_losses
from common.data.tensors import TorchOcelotDataset, collate_pairs


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(
            value.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_losses():
    target = torch.tensor([[[1, 2], [1, 2]]], dtype=torch.long)
    good = torch.full((1, 3, 2, 2), -15.)
    good.scatter_(1, target.unsqueeze(1), 15.)
    bad = good[:, [0, 2, 1]]
    assert dice_loss(good, target, (1, 2)) < dice_loss(
        bad, target, (1, 2)
    )

    target = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    logits = torch.randn(1, 3, 2, 2)
    changed = logits.clone()
    changed[:, 0, 1, 0] = 100
    torch.testing.assert_close(
        dice_loss(logits, target, (0, 1), 2),
        dice_loss(changed, target, (0, 1), 2)
    )
    ignored = torch.full_like(target, 2)
    assert dice_loss(logits, ignored, (0, 1), 2).item() == 0
    print("[완료] 손실 방향·UNK 제외·전체 UNK 처리")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size", type=int,
        choices=[128, 256, 512, 1024], default=128
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)

    progress_path = ROOT / "docs/progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if not {1, 2, 3, 4} <= set(progress.get("completed_steps", [])):
        raise SystemExit("4단계 전체 Dataset 검사를 먼저 완료하세요.")

    paths = resolve_paths()
    settings = json.loads(
        (ROOT / "configs/components.json").read_text(encoding="utf-8")
    )
    weights = paths["ssl_weights"]
    expected_sha = settings["checkpoint_sha256"]

    print("[환경]", platform.python_version(),
          torch.__version__, torchvision.__version__)

    if not weights.is_file():
        url = (
            "https://github.com/lunit-io/benchmark-ssl-pathology/"
            "releases/download/pretrained-weights/bt_rn50_ep200.torch"
        )
        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_name(weights.name + ".download")
        print("[다운로드] Lunit 공식 BT 가중치")
        try:
            urllib.request.urlretrieve(url, temporary)
            if file_digest(temporary) != expected_sha:
                raise ValueError("다운로드한 가중치의 SHA256이 다릅니다.")
            temporary.replace(weights)
        finally:
            temporary.unlink(missing_ok=True)

    checkpoint_sha256 = file_digest(weights)
    if checkpoint_sha256 != expected_sha:
        raise ValueError("가중치의 SHA256이 공식 파일 대조값과 다릅니다.")
    print("[완료] 공식 BT 파일 SHA256 대조")

    cell = TaskBranch(load_ssl_encoder(weights)).eval()
    tissue = TaskBranch(load_ssl_encoder(weights)).eval()
    print("[완료] 두 SSL 백본 strict 로딩")

    left = dict(cell.encoder.named_parameters())
    right = dict(tissue.encoder.named_parameters())
    assert all(
        torch.equal(left[key], right[key])
        and left[key].data_ptr() != right[key].data_ptr()
        for key in left
    )
    print("[완료] 같은 백본 초기값 / 독립 파라미터")

    before = [state_digest(cell), state_digest(tissue)]
    dataset = TorchOcelotDataset(split="train", label_source="official")
    batch = collate_pairs([dataset[0], dataset[1]])
    size = args.size

    inputs = [
        F.interpolate(
            batch[key][:1], size=(size, size),
            mode="bilinear", align_corners=False
        )
        for key in ("cell_image", "tissue_image")
    ]
    targets = [
        F.interpolate(
            batch[key][:1, None].float(),
            size=(size, size), mode="nearest"
        )[:, 0].long()
        for key in ("cell_target", "tissue_target")
    ]

    shapes = {}
    predictions = []
    with torch.no_grad():
        for name, branch, image in zip(
            ("cell", "tissue"), (cell, tissue), inputs
        ):
            output = branch(image)
            expected = {
                "low": (1, 256, size // 4, size // 4),
                "high": (1, 2048, size // 16, size // 16),
                "context": (1, 256, size // 16, size // 16),
                "decoded": (1, 256, size // 4, size // 4),
                "logits": (1, 3, size, size)
            }
            for key, tensor in output.items():
                if tuple(tensor.shape) != expected[key]:
                    raise ValueError(f"{name}/{key}: 출력 크기 오류")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"{name}/{key}: NaN/Inf 발생")
            shapes[name] = {
                key: list(value.shape) for key, value in output.items()
            }
            predictions.append(output["logits"])
            print(f"[출력] {name}:", shapes[name])

        check_losses()
        losses = task_losses(*predictions, *targets)

    if not all(torch.isfinite(value) for value in losses.values()):
        raise ValueError("실제 데이터 손실에 NaN/Inf가 있습니다.")
    if before != [state_digest(cell), state_digest(tissue)]:
        raise ValueError("점검 중 파라미터 또는 BN 통계가 바뀌었습니다.")
    print("[완료] 파라미터·BN 통계 변경 없음")
    print("[안내] backward·optimizer 실행 없음")

    folder = paths["outputs"] / "components"
    folder.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": True,
        "test_size": size,
        "shapes": shapes,
        "smoke_losses": {
            key: value.item() for key, value in losses.items()
        },
        "checkpoint_sha256": checkpoint_sha256,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "training_started": False,
        "model_states_unchanged": True
    }
    (folder / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )

    progress.update(
        completed_steps=sorted(set(progress["completed_steps"]) | {5}),
        current_step=max(progress.get("current_step", 5), 6),
        status="ssl_and_components_checks_passed",
        training_started=False
    )
    progress_path.write_text(
        json.dumps(progress, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )
    print("[PASS] 5/10단계 완료. 실제 어노테이션 수는 그대로 유지.")
    print("[안내] 학습·성능 평가 미시작.")


if __name__ == "__main__":
    main()
