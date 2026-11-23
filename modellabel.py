#!/usr/bin/env python3
"""학습한 모델(weights/best.pt)로 여러 과자가 함께 있는 사진에 박스 초안을 만든다.

prelabel.py 는 '사진마다 과자 하나'인 과자별 영상용이고, 이 스크립트는 3개를 함께 찍은 영상(mix)용이다.
과자 수가 예상과 다르거나, 같은 과자가 두 번 잡히거나, 박스끼리 크게 겹치거나, 신뢰도가 낮으면 '확인 필요'로 보낸다.
결과 폴더 구조는 prelabel.py 와 같다 (ok/, check/, review/check_*.jpg, review.csv).

사용법 (프로젝트 폴더에서):
  .venv/bin/python modellabel.py frames_mix --out datasets/prelabel_mix
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2

from prelabel import CLASSES, collect_files, make_tile, prepare_out, save_sheets, yolo_line


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reasons_for(dets, expect, min_conf, max_iou):
    reasons = []
    if expect and len(dets) != expect:
        reasons.append(f"과자 {len(dets)}개 (예상 {expect}개)")
    dup = [c for c, n in Counter(c for _, c in dets).items() if n > 1]
    if dup:
        reasons.append("같은 과자가 두 번 잡힘(" + ", ".join(dup) + ")")
    low = [f"{c} {b[4]:.2f}" for b, c in dets if b[4] < min_conf]
    if low:
        reasons.append("신뢰도 낮음(" + ", ".join(low) + ")")
    if any(iou(a[0], b[0]) > max_iou for i, a in enumerate(dets) for b in dets[i + 1:]):
        reasons.append("박스끼리 크게 겹침")
    return reasons


def main():
    p = argparse.ArgumentParser(description="학습한 모델로 박스 초안 만들기 (여러 과자가 함께 있는 사진)")
    p.add_argument("src", nargs="+", help="사진 폴더")
    p.add_argument("--out", default="datasets/prelabel_mix")
    p.add_argument("--model", default="weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25, help="박스로 쓸 최소 신뢰도 (기본 0.25)")
    p.add_argument("--min-conf", type=float, default=0.6, help="이보다 낮은 박스가 있으면 확인 묶음으로 (기본 0.6)")
    p.add_argument("--max-iou", type=float, default=0.3, help="박스끼리 이보다 많이 겹치면 확인 묶음으로 (기본 0.3)")
    p.add_argument("--expect", type=int, default=3, help="사진마다 예상하는 과자 수 (0=확인 안 함, 기본 3)")
    p.add_argument("--device", default="mps")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    files = collect_files(args.src)
    if not files:
        sys.exit(f"사진이 없어요: {args.src}")
    if not Path(args.model).exists():
        sys.exit(f"모델이 없어요: {args.model} (먼저 train.py 로 학습하세요)")
    out = Path(args.out)
    prepare_out(out, args.src, args.overwrite)

    from ultralytics import YOLO
    model = YOLO(args.model)
    if [model.names[i] for i in sorted(model.names)] != CLASSES:
        sys.exit(f"모델 클래스 {model.names} 가 {CLASSES} 와 달라요.")

    rows, tiles, where_count, cls_count, reason_count = [], [], Counter(), Counter(), Counter()
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"! 읽을 수 없어 건너뜀: {f}")
            continue
        h, w = img.shape[:2]
        r = model.predict(img, conf=args.conf, iou=0.5, agnostic_nms=True, device=args.device, verbose=False)[0]
        dets = [([*b, float(s)], CLASSES[int(c)]) for b, s, c in
                zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        reasons = reasons_for(dets, args.expect, args.min_conf, args.max_iou)
        where = "check" if reasons else "ok"
        shutil.copy2(f, out / where / "train/images" / f.name)
        (out / where / "train/labels" / f"{f.stem}.txt").write_text(
            "".join(yolo_line(c, b, w, h) + "\n" for b, c in dets))
        where_count[where] += 1
        cls_count.update(c for _, c in dets)
        for rs in reasons:
            reason_count[rs.split("(")[0].strip()] += 1
        rows.append([f.name, where, len(dets), " / ".join(f"{c} {b[4]:.2f}" for b, c in dets), "; ".join(reasons)])
        if where == "check":
            tiles.append(make_tile(len(tiles) + 1, f.name, img, [(b, c, "x" if b[4] < args.min_conf else "")
                                                                  for b, c in dets]))

    with open(out / "review.csv", "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["file", "where", "labels", "boxes(class conf)", "reasons"])
        wr.writerows(rows)
    if tiles:
        save_sheets(tiles, out / "review")
    print(f"사진 {len(rows)}장 → {out}/  ok {where_count['ok']}장 · check {where_count['check']}장")
    print("  박스: " + ", ".join(f"{c} {cls_count[c]}" for c in CLASSES))
    if reason_count:
        print("  확인 이유: " + ", ".join(f"{k} {v}" for k, v in reason_count.most_common()))


if __name__ == "__main__":
    main()
