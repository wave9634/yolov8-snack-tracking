#!/usr/bin/env python3
"""학습용 영상에서 프레임을 골라 Roboflow에 올릴 이미지로 저장한다.

사용법 (프로젝트 폴더에서):
  .venv/bin/python extract_frames.py              # raw_videos/train → frames/
  .venv/bin/python extract_frames.py --fps 3      # 1초에 3장
  .venv/bin/python extract_frames.py --clean      # 이전 결과를 지우고 다시

동작:
  - 1/fps 초 구간마다 가장 선명한 프레임 1장을 고른다 (손떨림·움직임 흐림 줄이기).
  - 같은 영상 안에서 유난히 흐린 장면, 직전 장과 거의 같은 장면은 frames_rejected/ 로 뺀다.
  - 파일 이름 앞부분(bigpie_, frenchpie_, piezza_, mix_, neg_)으로 클래스별 장수를 센다.
"""
import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}


def small_gray(frame, width=640):
    """해상도와 상관없이 같은 기준으로 비교하려고 폭 640 흑백으로 줄인다."""
    h, w = frame.shape[:2]
    size = (width, max(1, round(h * width / w)))
    return cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fit(frame, max_side):
    h, w = frame.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return frame
    s = max_side / max(h, w)
    return cv2.resize(frame, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def save(path, frame, max_side):
    cv2.imwrite(str(path), fit(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, 95])


def extract(video, out_dir, rej_dir, args):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 1.0 / args.fps
    stats = Counter()
    kept = []          # (경로, 선명도)
    prev_gray = None
    best = None        # (선명도, 프레임 번호, 프레임, 작은 흑백)
    slot_end = interval
    idx = 0

    def flush(b):
        nonlocal prev_gray
        score, n, frame, gray = b
        name = f"{video.stem}_{n:05d}.jpg"
        if (args.min_diff > 0 and prev_gray is not None
                and float(np.mean(cv2.absdiff(gray, prev_gray))) < args.min_diff):
            save(rej_dir / f"dup_{name}", frame, args.max_side)
            stats["duplicate"] += 1
            return
        save(out_dir / name, frame, args.max_side)
        kept.append((out_dir / name, score))
        prev_gray = gray

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        if t >= slot_end:
            if best is not None:
                flush(best)
                best = None
            while t >= slot_end:
                slot_end += interval
        gray = small_gray(frame)
        score = sharpness(gray)
        if best is None or score > best[0]:
            best = (score, idx, frame.copy(), gray)
        idx += 1
    if best is not None:
        flush(best)
    cap.release()

    if args.blur_ratio > 0 and len(kept) >= 3:
        cut = statistics.median(s for _, s in kept) * args.blur_ratio
        for path, score in kept:
            if score < cut:
                path.rename(rej_dir / f"blur_{path.name}")
                stats["blurry"] += 1
    stats["saved"] = len(kept) - stats["blurry"]
    stats["seconds"] = idx / fps
    return stats


def class_of(stem, classes):
    prefix = stem.split("_")[0].lower()
    if prefix in classes or prefix in ("mix", "neg"):
        return prefix
    return None


def main():
    p = argparse.ArgumentParser(description="학습용 영상 → 프레임 이미지")
    p.add_argument("input", nargs="?", default="raw_videos/train",
                   help="영상 파일 또는 폴더 (기본: raw_videos/train)")
    p.add_argument("--out", default="frames", help="저장 폴더 (기본: frames)")
    p.add_argument("--rejected", default="frames_rejected", help="제외한 장면 폴더")
    p.add_argument("--fps", type=float, default=2.0, help="1초에 뽑을 장수 (기본 2)")
    p.add_argument("--blur-ratio", type=float, default=0.35,
                   help="같은 영상 선명도 중앙값의 이 비율보다 흐리면 제외 (0=끄기)")
    p.add_argument("--min-diff", type=float, default=1.5,
                   help="직전 장과 평균 밝기 차이가 이보다 작으면 중복으로 제외 (0=끄기)")
    p.add_argument("--max-side", type=int, default=1280, help="긴 변 최대 픽셀 (0=원본 크기)")
    p.add_argument("--classes", default="bigpie,frenchpie,piezza", help="클래스 이름 (쉼표로 구분)")
    p.add_argument("--clean", action="store_true", help="이전 결과를 지우고 시작")
    args = p.parse_args()
    if args.fps <= 0:
        p.error("--fps는 0보다 커야 해요")

    src = Path(args.input)
    if src.is_dir():
        videos = sorted(f for f in src.iterdir() if f.suffix.lower() in VIDEO_EXTS)
    elif src.is_file():
        videos = [src]
    else:
        sys.exit(f"입력을 찾을 수 없어요: {src}")
    if not videos:
        sys.exit(f"영상이 없어요: {src}/ (확장자 {', '.join(sorted(VIDEO_EXTS))})")

    out_dir, rej_dir = Path(args.out), Path(args.rejected)
    for d in (out_dir, rej_dir):
        old = list(d.glob("*.jpg")) if d.exists() else []
        if old and not args.clean:
            sys.exit(f"{d}/ 에 이전 결과 {len(old)}장이 있어요. 지우고 다시 하려면 --clean 을 붙이세요.")
        for f in old:
            f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    classes = [c.strip().lower() for c in args.classes.split(",") if c.strip()]
    per_class, total = Counter(), Counter()
    unknown, failed = [], []

    for v in videos:
        st = extract(v, out_dir, rej_dir, args)
        if st is None:
            failed.append(v.name)
            print(f"  ! {v.name}: 열 수 없어요")
            continue
        kind = class_of(v.stem, classes)
        total.update({k: st[k] for k in ("saved", "blurry", "duplicate")})
        if kind == "mix":
            for c in classes:
                per_class[c] += st["saved"]
        elif kind:
            per_class[kind] += st["saved"]
        else:
            unknown.append(v.name)
        print(f"  {v.name}  {st['seconds']:.1f}초 → 저장 {st['saved']} · 흐림 {st['blurry']} · "
              f"중복 {st['duplicate']}  [{kind or '?'}]")

    print(f"\n이미지 {total['saved']}장 → {out_dir}/   "
          f"(제외 {total['blurry'] + total['duplicate']}장 → {rej_dir}/)")
    print("클래스별 등장 이미지 수 (mix 영상은 세 클래스 모두에 더함):")
    for c in classes:
        note = "OK" if per_class[c] >= 100 else "부족 — 영상을 더 찍거나 --fps 를 올리세요"
        print(f"  {c:<8}{per_class[c]:>5}장  {note}")
    if per_class["neg"]:
        print(f"  neg     {per_class['neg']:>5}장  (라벨 없이 올리기)")
    if unknown:
        print(f"\n분류 못 한 영상: {', '.join(unknown)}")
        print("  → 파일 이름을 bigpie_a.mp4, mix_desk.mov 처럼 바꾸면 클래스별로 셀 수 있어요.")
    if failed:
        print(f"\n열지 못한 영상: {', '.join(failed)}")
        print("  → 아이폰은 설정 > 카메라 > 포맷 > '높은 호환성'으로 다시 찍거나 H.264로 변환해 보세요.")


if __name__ == "__main__":
    main()
