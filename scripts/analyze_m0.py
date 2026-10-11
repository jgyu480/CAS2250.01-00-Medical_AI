"""M0 결과 분석 (정리본 재료). 학습이 끝난 run 폴더를 대상으로 한다.

    python scripts/analyze_m0.py curves  --run outputs/m0/<run>
    python scripts/analyze_m0.py errors  --run outputs/m0/<run> --split test
    python scripts/analyze_m0.py cases   --run outputs/m0/<run> --split test
    python scripts/analyze_m0.py annotators --run outputs/m0/<run>

curves      history.csv → 학습 곡선 그림 (손실, 검증 mF1·mIoU, 학습률)
errors      eval_<split>_per_image.json → 세포 오류 유형 집계 (누락·중복·오분류·오검출), 장기별
cases       좋은/중간/나쁜 예측 이미지 그림 (세포 오류 표시 + 조직 정답·예측·오류 영역)
annotators  직접 라벨링한 24쌍에서 공식·이진호·신상우 라벨과 모델 예측을 서로 비교

결과는 <run>/analysis/ 에 저장한다. 원본 데이터와 라벨은 수정하지 않는다.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.metrics.m0_eval import CellF1, DISTANCE_CUTOFF

# 팀 미리보기와 같은 색 규칙: BC 노랑, TC 파랑, CA 초록
BC_COLOR, TC_COLOR = (255, 214, 0), (40, 110, 255)
CA_COLOR = (0, 170, 80)
ERROR_COLORS = {
    "missed": (255, 255, 255),      # 정답은 있는데 예측 없음 (누락)
    "misclassified": (230, 40, 40), # 위치는 맞는데 BC/TC 반대 (오분류)
    "duplicate": (255, 140, 0),     # 이미 맞힌 세포 근처에 또 찍음 (중복)
    "false_positive": (220, 0, 220) # 근처에 정답 세포가 없음 (오검출)
}
GROUP_EN = {"낮음": "low", "중간": "mid", "높음": "high"}
ERROR_NAMES = {"missed": "누락", "misclassified": "오분류",
               "duplicate": "중복", "false_positive": "오검출"}


def analysis_dir(run):
    folder = Path(run) / "analysis"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---------------------------------------------------------------- 오류 분류
def categorize(predictions, truths, cutoff=DISTANCE_CUTOFF):
    """클래스를 무시하고 위치로 먼저 매칭한 뒤 오류 유형을 나눈다.

    점수가 높은 예측부터 cutoff 안의 가장 가까운 남은 정답과 짝짓는다.
    - 짝 + 같은 클래스: correct / 짝 + 다른 클래스: misclassified
    - 짝 없음 + 근처 정답이 이미 짝지어짐: duplicate / 근처 정답 없음: false_positive
    - 짝 없는 정답: missed
    공식 F1은 클래스별로 따로 매칭하므로, 이 분류는 원인 파악용 보조 지표다.
    """
    preds = sorted(predictions, key=lambda p: -p[3])
    gt = np.asarray([t[:2] for t in truths], dtype=np.float32).reshape(-1, 2)
    used = np.zeros(len(gt), dtype=bool)
    result = defaultdict(list)
    for p in preds:
        if len(gt):
            d = np.linalg.norm(gt - np.asarray(p[:2], dtype=np.float32), axis=1)
            near = d <= cutoff
            free = near & ~used
        else:
            near = free = np.zeros(0, dtype=bool)
        if free.any():
            j = np.flatnonzero(free)[np.argmin(d[free])]
            used[j] = True
            kind = "correct" if p[2] == truths[j][2] else "misclassified"
            result[kind].append((p, truths[j]))
        elif near.any():
            result["duplicate"].append((p, None))
        else:
            result["false_positive"].append((p, None))
    for j in np.flatnonzero(~used):
        result["missed"].append((None, truths[j]))
    return result


def cmd_errors(args):
    records = json.loads((Path(args.run) / f"eval_{args.split}_per_image.json")
                         .read_text(encoding="utf-8"))
    kinds = ["correct", "misclassified", "duplicate", "false_positive", "missed"]
    totals = defaultdict(lambda: defaultdict(int))
    rows = []
    for r in records:
        cats = categorize([tuple(p) for p in r["predictions"]],
                          [tuple(t) for t in r["truths"]])
        counts = {k: len(cats[k]) for k in kinds}
        # 오분류 방향: 정답 TC를 BC로, 정답 BC를 TC로
        counts["TC→BC"] = sum(1 for p, t in cats["misclassified"] if t[2] == 2)
        counts["BC→TC"] = sum(1 for p, t in cats["misclassified"] if t[2] == 1)
        rows.append({"pair_id": r["pair_id"], "organ": r["organ"],
                     "num_gt": r["num_gt"], "num_pred": r["num_pred"],
                     "mF1": r["mF1"], "mIoU": r["mIoU"], **counts})
        for group in ("전체", r["organ"]):
            for k, v in counts.items():
                totals[group][k] += v
            totals[group]["num_gt"] += r["num_gt"]
            totals[group]["pairs"] += 1

    folder = analysis_dir(args.run)
    with (folder / f"errors_{args.split}_per_image.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["mF1"]))
    summary = {g: dict(v) for g, v in totals.items()}
    (folder / f"errors_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    print(f"[오류 유형] {args.split} (정답 세포 대비 비율)")
    header = f"{'':15s}{'쌍':>4s}{'정답':>7s}{'맞춤':>8s}{'누락':>8s}" \
             f"{'오분류':>8s}{'중복':>7s}{'오검출':>8s}  TC→BC / BC→TC"
    print(header)
    for group, v in sorted(summary.items(), key=lambda x: x[0] != "전체"):
        n = max(v["num_gt"], 1)
        print(f"{group:15s}{v['pairs']:4d}{v['num_gt']:7d}"
              f"{v['correct'] / n:8.1%}{v['missed'] / n:8.1%}"
              f"{v['misclassified'] / n:8.1%}{v['duplicate']:7d}"
              f"{v['false_positive']:8d}  {v['TC→BC']} / {v['BC→TC']}")
    print(f"[저장] {folder}/errors_{args.split}_*.csv/json")


# ---------------------------------------------------------------- 학습 곡선
def cmd_curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (Path(args.run) / "history.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    epoch = [int(r["epoch"]) for r in rows]
    col = lambda k: [float(r[k]) for r in rows]
    frozen = [e for e, r in zip(epoch, rows) if r["encoder_frozen"] == "True"]

    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink, muted, grid = "#0b0b0b", "#52514e", "#e6e5e1"
    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": grid, "axes.labelcolor": muted,
        "xtick.color": muted, "ytick.color": muted, "axes.titlecolor": ink,
        "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False
    })
    panels = [
        ("Total loss (cell + tissue)", [("train", "train_total", blue),
                                    ("val", "val_total", orange)]),
        ("Cell loss (Focal)", [("train", "train_cell", blue),
                            ("val", "val_cell", orange)]),
        ("Tissue loss (CE + Dice)", [("train", "train_tissue", blue),
                                ("val", "val_tissue", orange)]),
        ("Val cell F1", [("mF1", "val_mF1", blue), ("BC", "val_f1_BC", orange),
                       ("TC", "val_f1_TC", aqua)]),
        ("Val tissue IoU", [("mIoU", "val_mIoU", blue), ("BG", "val_iou_BG", orange),
                        ("CA", "val_iou_CA", aqua)]),
        ("Learning rate", [("encoder", "lr_encoder", blue),
                           ("new modules", "lr_new", orange)]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    best = max(range(len(rows)), key=lambda i: float(rows[i]["val_mF1"]))
    for ax, (title, series) in zip(axes.flat, panels):
        if frozen:
            ax.axvspan(min(frozen) - 0.5, max(frozen) + 0.5, color="#f1f0ec",
                       zorder=0, label="encoder frozen")
        for label, key, color in series:
            ax.plot(epoch, col(key), color=color, linewidth=2, label=label)
        ax.axvline(epoch[best], color=muted, linewidth=1, linestyle=":")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=8)
    axes.flat[3].annotate(
        f"best epoch {epoch[best]}\nmF1 {float(rows[best]['val_mF1']):.3f}",
        (epoch[best], float(rows[best]["val_mF1"])), xytext=(8, -28),
        textcoords="offset points", fontsize=8, color=ink)
    fig.suptitle(f"M0 training curves - {Path(args.run).name}", x=0.01,
                 ha="left", fontsize=12, color=ink)
    out = analysis_dir(args.run) / "curves.png"
    fig.savefig(out, dpi=150)
    print(f"[저장] {out} (best epoch {epoch[best]}, "
          f"val mF1 {float(rows[best]['val_mF1']):.4f})")


# ---------------------------------------------------------------- 모델 추론 (그림용)
def load_run_model(run, device_name="auto"):
    import torch
    from common.training.m0_loop import pick_device
    from models.M0.model import build_model
    checkpoint = Path(run) / "best.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    s = state["settings"]
    model = build_model(json.loads((ROOT / s["model_config"]).read_text(
        encoding="utf-8")))
    model.load_state_dict(state["model"])
    device = pick_device(device_name)
    post_path = Path(run) / "postprocess.json"
    post = (json.loads(post_path.read_text())["chosen"] if post_path.exists()
            else s["postprocess_during_training"])
    post = {k: post[k] for k in ("sigma", "min_distance", "threshold")}
    return model.to(device).eval(), device, post


def predict(model, device, post, tensors):
    import torch
    from common.metrics.m0_eval import detect_cells
    with torch.no_grad():
        out = model(tensors["cell_image"][None].to(device),
                    tensors["tissue_image"][None].to(device),
                    tensors["cell_box"][None].to(device))
    cells = detect_cells(out["cell_logits"], **post)[0]
    tissue = out["tissue_logits"][0, :2].argmax(0).cpu().numpy()
    return cells, tissue


def draw_points(image, cats):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    r = 6
    for p, t in cats.get("correct", []):
        x, y = p[:2]
        draw.ellipse([x - r, y - r, x + r, y + r],
                     fill=BC_COLOR if p[2] == 1 else TC_COLOR)
    for kind in ("misclassified", "duplicate", "false_positive"):
        for p, t in cats.get(kind, []):
            x, y = p[:2]
            draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2],
                         outline=ERROR_COLORS[kind], width=3)
    for _, t in cats.get("missed", []):
        x, y = t[:2]
        draw.rectangle([x - r, y - r, x + r, y + r],
                       outline=ERROR_COLORS["missed"], width=3)
    return canvas


def draw_truth(image, truths):
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for x, y, c in truths:
        draw.ellipse([x - 6, y - 6, x + 6, y + 6],
                     fill=BC_COLOR if c == 1 else TC_COLOR)
    return canvas


def overlay(image, mask_ca, valid=None, error=None):
    base = image.astype(np.float32)
    out = base.copy()
    ca = mask_ca.astype(bool)
    out[ca] = 0.55 * base[ca] + 0.45 * np.array(CA_COLOR)
    if valid is not None:
        out[~valid] = 0.5 * base[~valid] + 0.5 * np.array([200, 0, 200])
    if error is not None:
        out[error] = 0.4 * base[error] + 0.6 * np.array([230, 40, 40])
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))


def box_outline(canvas, box):
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    draw.rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h],
                   outline=(255, 0, 0), width=4)
    return canvas


def label(canvas, text):
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.size[0], 34], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0))
    return canvas


def cmd_cases(args):
    from common.data.tensors import TorchOcelotDataset
    records = json.loads((Path(args.run) / f"eval_{args.split}_per_image.json")
                         .read_text(encoding="utf-8"))
    usable = sorted([r for r in records if r["num_gt"] >= 10],
                    key=lambda r: r["mF1"])
    if args.ids:
        chosen = [("id", pid) for pid in args.ids]
    else:
        n = args.per_group
        mid = len(usable) // 2
        chosen = ([("낮음", r["pair_id"]) for r in usable[:n]]
                  + [("중간", r["pair_id"]) for r in usable[mid:mid + n]]
                  + [("높음", r["pair_id"]) for r in usable[-n:]])

    model, device, post = load_run_model(args.run, args.device)
    dataset = TorchOcelotDataset(split=args.split, label_source="official")
    index = {row["pair_id"]: i for i, row in enumerate(dataset.source.rows)}
    folder = analysis_dir(args.run) / f"cases_{args.split}"
    folder.mkdir(exist_ok=True)

    for group, pid in chosen:
        raw = dataset.source[index[pid]]
        tensors = dataset[index[pid]]
        preds, tissue_pred = predict(model, device, post, tensors)
        truths = [(int(round(x)), int(round(y)), int(c))
                  for x, y, c in raw["cell_points"].tolist()]
        cats = categorize(preds, truths)
        f1 = CellF1()
        f1.update(preds, truths)
        score = f1.compute()
        target = raw["tissue_target"]
        valid = target != 2
        error = valid & (tissue_pred != target)
        acc = 1 - error[valid].mean() if valid.any() else float("nan")

        size = 512
        tiles = [
            label(draw_truth(raw["cell_image"], truths),
                  "Cell GT (BC yellow / TC blue)"),
            label(draw_points(raw["cell_image"], cats),
                  "Cell pred: filled=correct, red=wrong class, orange=duplicate, "
                  "magenta=false pos, white box=missed"),
            label(box_outline(overlay(raw["tissue_image"], target == 1, valid),
                              raw["cell_box"]),
                  "Tissue GT (green CA, magenta UNK, red box=cell FoV)"),
            label(box_outline(overlay(raw["tissue_image"], tissue_pred == 1,
                                      None, error), raw["cell_box"]),
                  "Tissue pred (green CA, red=wrong pixel)"),
        ]
        sheet = Image.new("RGB", (size * 4 + 30, size + 60), "white")
        for i, tile in enumerate(tiles):
            sheet.paste(tile.resize((size, size)), (i * (size + 10), 50))
        counts = {k: len(cats[k]) for k in ERROR_NAMES}
        ImageDraw.Draw(sheet).text(
            (8, 12),
            f"[{GROUP_EN.get(group, group)}] {pid} "
            f"({dataset.source.metadata[pid]['organ']}) "
            f"| mF1 {score['mF1']:.3f} (BC {score['f1_BC']:.3f}, "
            f"TC {score['f1_TC']:.3f}) | GT {len(truths)}, pred {len(preds)} "
            f"| {counts} | tissue pixel acc {acc:.3f}",
            fill=(0, 0, 0))
        out = folder / f"{GROUP_EN.get(group, 'id')}_{pid}.png"
        sheet.save(out)
        print(f"[저장] {out}")


# ---------------------------------------------------------------- 어노테이션 비교
def cmd_annotators(args):
    from common.data.tensors import TorchOcelotDataset
    from common.data.annotators import label_paths, annotator_ids
    from common.data.targets import read_points
    from common.data.labels import tissue_to_target
    from common.metrics.m0_eval import TissueIoU
    import torch

    with (ROOT / "annotations/selection.csv").open(encoding="utf-8-sig") as f:
        selection = [r["pair_id"] for r in csv.DictReader(f)]
    names = annotator_ids()
    model, device, post = load_run_model(args.run, args.device)
    dataset = TorchOcelotDataset(split="train", label_source="official")
    index = {row["pair_id"]: i for i, row in enumerate(dataset.source.rows)}

    def cell_labels(source, pid, raw):
        if source == "official":
            pts = raw["cell_points"]
        else:
            pts = read_points(label_paths(source, pid)["cell"])
        return [(int(round(x)), int(round(y)), int(c)) for x, y, c in pts]

    def tissue_labels(source, pid, raw):
        if source == "official":
            return raw["tissue_target"]
        with Image.open(label_paths(source, pid)["tissue"]) as image:
            return tissue_to_target(np.asarray(image))

    sources = ["official", *names]
    pairs = [("model", s) for s in sources] + [
        (a, b) for i, a in enumerate(sources) for b in sources[i + 1:]]
    cell = {p: CellF1() for p in pairs}
    tissue = {p: TissueIoU() for p in pairs}
    per_image = []

    for pid in selection:
        raw = dataset.source[index[pid]]
        preds, tissue_pred = predict(model, device, post, dataset[index[pid]])
        cells = {s: cell_labels(s, pid, raw) for s in sources}
        masks = {s: tissue_labels(s, pid, raw) for s in sources}
        cells["model"] = [p[:3] for p in preds]
        masks["model"] = tissue_pred
        row = {"pair_id": pid, "organ": dataset.source.metadata[pid]["organ"],
               **{f"cells_{s}": len(cells[s]) for s in ["model", *sources]}}
        for a, b in pairs:
            # a를 "예측"(점수 1.0), b를 "정답"으로 두고 공식 규칙으로 채점
            pred = (list(preds) if a == "model"
                    else [(*q, 1.0) for q in cells[a]])
            cell[(a, b)].update(pred, cells[b])
            single = CellF1()
            single.update(pred, cells[b])
            row[f"mF1_{a}_vs_{b}"] = round(single.compute()["mF1"], 4)
            # 조직: b의 UNK는 제외, a의 UNK는 BG로 본다
            logits = torch.nn.functional.one_hot(
                torch.as_tensor(np.where(masks[a] == 2, 0, masks[a])), 3
            ).permute(2, 0, 1)[None].float()
            tissue[(a, b)].update(logits, torch.as_tensor(masks[b])[None])
        per_image.append(row)
        print(f"  {pid} 완료", flush=True)

    result = {f"{a} vs {b}": {
        **{k: round(v, 4) for k, v in cell[(a, b)].compute().items()},
        **{k: round(v, 4) for k, v in tissue[(a, b)].compute().items()}}
        for a, b in pairs}
    folder = analysis_dir(args.run)
    (folder / "annotators.json").write_text(json.dumps({
        "pairs": len(selection), "postprocess": post, "comparisons": result,
        "note": "24쌍은 train에 속한다. 모델은 이 사진의 공식 라벨로 학습했으므로 "
                "model vs official 점수는 학습 데이터 점수다. 사람 간 비교는 "
                "모델과 무관한 라벨 일치도다."
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (folder / "annotators_per_image.csv").open(
            "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)

    print(f"[비교] 24쌍 (세포 mF1 / 조직 mIoU), 앞쪽을 예측·뒤쪽을 정답으로 채점")
    for key, v in result.items():
        print(f"  {key:28s} mF1 {v['mF1']:.4f} (BC {v['f1_BC']:.4f}, "
              f"TC {v['f1_TC']:.4f}) | mIoU {v['mIoU']:.4f}")
    print("[주의] 24쌍은 학습에 사용된 train 사진입니다. "
          "model vs official은 일반화 성능이 아닙니다.")
    print(f"[저장] {folder}/annotators*.json/csv")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("curves", "errors", "cases", "annotators"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        if name in ("errors", "cases"):
            p.add_argument("--split", choices=["val", "test"], default="val")
        if name in ("cases", "annotators"):
            p.add_argument("--device", default="auto")
        if name == "cases":
            p.add_argument("--per-group", type=int, default=2)
            p.add_argument("--ids", nargs="*")
    args = parser.parse_args()
    {"curves": cmd_curves, "errors": cmd_errors, "cases": cmd_cases,
     "annotators": cmd_annotators}[args.command](args)


if __name__ == "__main__":
    main()
