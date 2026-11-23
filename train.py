#!/usr/bin/env python3
"""YOLOv8 학습 (맥 GPU = mps).

사용법 (프로젝트 폴더에서):
  .venv/bin/python train.py --data datasets/snacks/data.yaml --check   # 3 epoch 시험 (속도·loss 확인)
  .venv/bin/python train.py --data datasets/snacks/data.yaml           # 본 학습 (기본 100 epoch)
  .venv/bin/python train.py --data ... --model yolov8s.pt            # 더 큰 모델

학습이 끝나면
  - weights/best.pt            : GitHub에 올릴 모델
  - results/                   : 보고서용 그래프 (loss·mAP 곡선, confusion matrix, PR 곡선, 예측 예시)
  - results/metrics.md         : 전체·클래스별 성능 표
가 만들어진다. 원본 학습 기록은 runs/detect/<이름>/ 에 있다.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml

PLOTS = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png",
         "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "labels.jpg", "train_batch0.jpg",
         "val_batch0_labels.jpg", "val_batch0_pred.jpg", "results.csv", "args.yaml"]


def local_data_yaml(path):
    """Roboflow 내보내기의 data.yaml 은 경로가 '../train/images' 처럼 되어 있어 어긋나기 쉽다.
    실제로 있는 폴더를 찾아 절대 경로로 고친 사본을 만든다."""
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    root = Path(data.get("path") or path.parent)
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    for key in ("train", "val"):
        if not data.get(key):
            sys.exit(f"data.yaml 에 {key}: 경로가 없어요. Roboflow에서 Train/Valid로 나눠 내보냈는지 확인하세요.")
    for key in ("train", "val", "test"):
        v = data.get(key)
        if v is None:
            continue
        if not isinstance(v, str):
            sys.exit(f"data.yaml 의 {key}: 가 목록이에요. 이 스크립트는 폴더 하나만 받아요 — 한 폴더로 모아 주세요.")
        tries = [root / v, path.parent / v]
        if v.startswith("../"):
            tries += [root / v[3:], path.parent / v[3:]]
        found = next((t.resolve() for t in tries if t.exists()), None)
        if found is None:
            if key == "test":
                data.pop(key)
                continue
            sys.exit(f"data.yaml 의 {key}: {v} 폴더를 찾을 수 없어요. 내보낸 폴더 구조를 확인하세요.")
        data[key] = str(found)
    data.pop("path", None)
    out = path.parent / "data_local.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return out, data


def count_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in Path(folder).rglob("*") if f.suffix.lower() in exts)


def metric_table(m):
    """전체 + 클래스별 표. 그 분할에 박스가 없는 클래스도 n/a 로 남긴다."""
    names = m.names
    idx = [int(c) for c in m.box.ap_class_index]
    nt = getattr(m, "nt_per_class", None)
    total = int(sum(nt)) if nt is not None else "-"
    rows = ["| 클래스 | 박스 수 | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|---|",
            f"| **전체** | {total} | {m.box.mp:.3f} | {m.box.mr:.3f} | {m.box.map50:.3f} | {m.box.map:.3f} |"]
    for k, name in names.items():
        inst = int(nt[k]) if nt is not None and k < len(nt) else "-"
        if k in idx:
            p, r, ap50, ap = m.box.class_result(idx.index(k))
            rows.append(f"| {name} | {inst} | {p:.3f} | {r:.3f} | {ap50:.3f} | {ap:.3f} |")
        else:
            rows.append(f"| {name} | {inst} | n/a | n/a | n/a | n/a |")
    return rows


def write_metrics(model_path, data_yaml, data, device, out_md, info, run_dir):
    from ultralytics import YOLO
    model = YOLO(model_path)
    splits = [("val", "val — best.pt 를 고를 때 쓴 데이터라 실제보다 조금 높게 나올 수 있음")]
    if data.get("test"):
        splits.append(("test", "test — 학습·모델 선택에 쓰지 않은 데이터 (보고서에는 이 숫자를 권장)"))
    lines = ["# 학습 결과", "", *[f"- {k}: {v}" for k, v in info.items()]]
    last = None
    for split, title in splits:
        last = model.val(data=str(data_yaml), split=split, device=device, plots=False, verbose=False,
                         project=str(run_dir), name=f"final_{split}", exist_ok=True)
        lines += ["", f"## {title}", "", *metric_table(last)]
    lines += ["", "n/a = 그 분할에 해당 클래스 박스가 없음"]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return last, splits[-1][0]


def main():
    p = argparse.ArgumentParser(description="YOLOv8 학습 (맥 GPU)")
    p.add_argument("--data", required=True, help="Roboflow에서 YOLOv8 형식으로 내보낸 data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="시작 가중치 (yolov8n.pt / yolov8s.pt …)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--patience", type=int, default=30, help="이 epoch 동안 나아지지 않으면 일찍 멈춤")
    p.add_argument("--device", default="mps", help="mps(맥 GPU) / cpu / 0(엔비디아 GPU)")
    p.add_argument("--name", default="snacks")
    p.add_argument("--check", action="store_true", help="3 epoch만 돌려 속도와 loss 확인")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    from ultralytics import YOLO

    data_yaml, data = local_data_yaml(args.data)
    n_train, n_val = count_images(data["train"]), count_images(data["val"])
    print(f"클래스: {data.get('names')} | train {n_train}장 · val {n_val}장")

    epochs = 3 if args.check else args.epochs
    name = f"{args.name}_check" if args.check else args.name
    t0 = time.time()
    model = YOLO(args.model)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, device=args.device, name=name, exist_ok=True,
                workers=args.workers, seed=0, deterministic=False, plots=True)
    minutes = (time.time() - t0) / 60
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    done = model.trainer.epoch + 1            # 일찍 멈추면 계획한 epoch 보다 적다
    print(f"\n학습 시간 {minutes:.1f}분, {done} epoch (epoch당 {minutes * 60 / done:.0f}초) → {run_dir}")

    if args.check:
        print(f"시험 끝. 본 학습 {args.epochs} epoch 예상: 약 {minutes / done * args.epochs:.0f}분 "
              f"(patience {args.patience}로 일찍 끝날 수 있음)")
        return

    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(best, "weights/best.pt")
    res = Path("results")
    res.mkdir(exist_ok=True)
    for f in PLOTS:
        if (run_dir / f).exists():
            shutil.copy2(run_dir / f, res / f)
    info = {"모델": Path(args.model).name, "epochs": f"{done} / {epochs} (patience {args.patience})",
            "imgsz": args.imgsz,
            "batch": args.batch, "device": args.device, "train/val 이미지": f"{n_train} / {n_val}",
            "학습 시간": f"{minutes:.1f}분"}
    m, split = write_metrics(best, data_yaml, data, args.device, res / "metrics.md", info, run_dir)
    print(f"\n최종 ({split}): mAP50 {m.box.map50:.3f} · mAP50-95 {m.box.map:.3f}")
    print("weights/best.pt, results/ (그래프·metrics.md) 에 저장했어요.")


if __name__ == "__main__":
    main()
