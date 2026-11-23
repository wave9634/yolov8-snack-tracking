#!/usr/bin/env python3
"""학습한 모델로 영상 속 과자를 감지·추적하고, 결과 영상과 궤적(trajectory)을 저장한다.

사용법 (프로젝트 폴더에서):
  .venv/bin/python track.py raw_videos/test/test_01.mp4 --start 3      # 3초부터 5초 분량
  .venv/bin/python track.py raw_videos/test/test_01.mp4 --start 3 --tracker botsort.yaml

결과 (outputs/):
  result.mp4       H.264 결과 영상 (박스·클래스·ID + 최근 이동 경로 선)
  result.gif       README에 넣을 미리보기
  trajectory.csv   프레임마다 객체 중심 좌표
                   (frame·time_s = 이 클립 기준, video_frame·video_time_s = 원본 영상 기준)
  trajectory.png   ID별 전체 이동 경로 그림
"""
import argparse
import colorsys
import csv
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


def color_for(tid):
    """ID마다 다른 색 (BGR)."""
    r, g, b = colorsys.hsv_to_rgb((tid * 0.61803) % 1.0, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def save_plot(path, background, history, cls_of, n_frames):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, w = background.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 10 * h / w))
    ax.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB), alpha=0.45)
    for tid, pts in sorted(history.items()):
        xs, ys = [p[1] for p in pts], [p[2] for p in pts]
        b, g, r = color_for(tid)
        c = (r / 255, g / 255, b / 255)
        ax.plot(xs, ys, "-", color=c, lw=2.5)
        ax.scatter(xs[0], ys[0], color=c, marker="o", s=40, zorder=3)
        ax.scatter(xs[-1], ys[-1], color=c, marker="s", s=40, zorder=3)
        ax.text(xs[-1] + 6, ys[-1] - 6, f"{cls_of[tid]} #{tid}", color="white", fontsize=9,
                bbox=dict(facecolor=c, edgecolor="none", pad=1.5))
    ax.set_title(f"Trajectories: {len(history)} tracks over {n_frames} frames (o = start, square = end)")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def tracker_config(name, fps, conf, buffer_sec, out_dir):
    """추적기 설정을 영상에 맞게 고친 사본을 만든다.
    - track_buffer 는 '프레임 수'라서 60fps 영상이면 30fps보다 절반 시간만 기억한다 → 초 단위로 환산
    - 새 객체를 잡기 시작하는 신뢰도(track_high/new_track_thresh)를 --conf 로"""
    from ultralytics.utils import ROOT
    src = Path(name)
    if not src.exists():
        src = ROOT / "cfg" / "trackers" / name
    if not src.exists():
        sys.exit(f"추적기 설정을 찾을 수 없어요: {name} (bytetrack.yaml 또는 botsort.yaml)")
    cfg = yaml.safe_load(src.read_text())
    cfg["track_buffer"] = max(1, round(buffer_sec * fps))
    cfg["track_high_thresh"] = conf
    cfg["new_track_thresh"] = conf
    dst = Path(tempfile.gettempdir()) / f"yolo_tracker_{src.stem}.yaml"   # outputs/ 를 어지럽히지 않게
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst, cfg


def main():
    p = argparse.ArgumentParser(description="YOLOv8 감지 + 추적 + 궤적 저장")
    p.add_argument("video", help="입력 영상")
    p.add_argument("--model", default="weights/best.pt", help="학습한 모델 (기본 weights/best.pt)")
    p.add_argument("--start", type=float, default=0.0, help="시작 시각(초)")
    p.add_argument("--duration", type=float, default=5.0, help="길이(초, 기본 5)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml 또는 botsort.yaml")
    p.add_argument("--conf", type=float, default=0.25,
                   help="새 객체로 잡기 시작하는 신뢰도 (기본 0.25). 이미 추적 중인 객체는 0.1까지 이어 붙임")
    p.add_argument("--buffer-sec", type=float, default=1.0,
                   help="객체를 놓친 뒤 같은 ID로 다시 잡아 줄 시간(초, 기본 1)")
    p.add_argument("--device", default="mps", help="mps(맥 GPU) / cpu")
    p.add_argument("--trail", type=int, default=45, help="영상에 그릴 최근 경로 길이(프레임, 0=전체)")
    p.add_argument("--line-width", type=int, default=0,
                   help="박스 선 굵기 (0=영상 크기에 맞춰 자동 — 1080p는 5, 글자 크기도 따라 커짐)")
    p.add_argument("--out", default="outputs", help="저장 폴더")
    p.add_argument("--no-gif", action="store_true", help="GIF 만들지 않기")
    args = p.parse_args()

    from ultralytics import YOLO
    import imageio_ffmpeg

    if not Path(args.model).exists():
        sys.exit(f"모델이 없어요: {args.model} (먼저 train.py 로 학습하세요)")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없어요: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    length = frame_count / fps if frame_count > 0 else 0
    if length and args.start + args.duration > length + 0.05:
        print(f"! 영상 길이 {length:.1f}초 — {args.start}초부터 {args.duration}초를 다 채우지 못해요")
    start_frame = int(round(args.start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracker_yaml, tcfg = tracker_config(args.tracker, fps, args.conf, args.buffer_sec, out)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = out / "result.mp4"
    enc = None                        # 첫 프레임 크기를 알고 나서 ffmpeg 를 연다

    model = YOLO(args.model)
    history = defaultdict(list)       # id -> [(frame, cx, cy)]
    cls_of, rows = {}, []
    last_frame = None
    t0 = None
    n = 0
    for i in range(int(round(args.duration * fps))):
        ok, frame = cap.read()
        if not ok:
            break
        t_video = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000   # 가변 프레임레이트 영상도 실제 시각으로
        if t0 is None:
            t0 = t_video
        if enc is None:               # 회전 정보가 반영된 실제 크기로 인코딩 (한 번만 압축)
            h, w = frame.shape[:2]
            enc = subprocess.Popen(
                [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps:.6f}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(mp4)],
                stdin=subprocess.PIPE)
        r = model.track(frame, persist=True, tracker=str(tracker_yaml), conf=tcfg.get("track_low_thresh", 0.1),
                        device=args.device, verbose=False)[0]
        lw = args.line_width or max(2, round(max(frame.shape[:2]) / 380))
        img = r.plot(line_width=lw)
        boxes = r.boxes
        if boxes.id is not None:
            for (x1, y1, x2, y2), tid, c, cf in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(),
                                                    boxes.cls.int().tolist(), boxes.conf.tolist()):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                history[tid].append((i, cx, cy))
                cls_of[tid] = model.names[c]
                rows.append([i, round(t_video - t0, 3), start_frame + i, round(t_video, 3), tid, model.names[c],
                             round(cf, 3), round(cx, 1), round(cy, 1), round(x1, 1), round(y1, 1),
                             round(x2, 1), round(y2, 1)])
        for tid, pts in history.items():
            recent = [(x, y) for f, x, y in pts if args.trail <= 0 or f > i - args.trail]
            if len(recent) >= 2:
                cv2.polylines(img, [np.array(recent, np.int32).reshape(-1, 1, 2)], False, color_for(tid), lw + 1)
            if pts[-1][0] == i:
                cv2.circle(img, (int(pts[-1][1]), int(pts[-1][2])), lw * 2, color_for(tid), -1)
        enc.stdin.write(np.ascontiguousarray(img).tobytes())
        last_frame = frame
        n += 1
    cap.release()
    if enc is None:
        sys.exit("읽은 프레임이 없어요. --start 값을 확인하세요.")
    enc.stdin.close()
    if enc.wait() != 0:
        sys.exit("ffmpeg 인코딩에 실패했어요.")

    if not args.no_gif:               # 긴 변 480px, 초당 10장 (세로 영상도 커지지 않게)
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf",
                        "fps=10,scale='if(gt(iw,ih),480,-2)':'if(gt(iw,ih),-2,480)':flags=lanczos,"
                        "split[a][b];[a]palettegen[p];[b][p]paletteuse",
                        str(out / "result.gif")], check=True)

    with open(out / "trajectory.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "time_s", "video_frame", "video_time_s", "id", "class", "conf",
                     "cx", "cy", "x1", "y1", "x2", "y2"])
        wr.writerows(rows)
    if history:
        save_plot(out / "trajectory.png", last_frame, history, cls_of, n)

    per_class = Counter(cls_of.values())
    print(f"{n}프레임 ({n / fps:.1f}초, {fps:.0f}fps) 처리 → {out}/")
    print(f"  추적기 {Path(args.tracker).stem}: 새 객체 신뢰도 {args.conf}, 놓친 객체 기억 {tcfg['track_buffer']}프레임"
          f"({args.buffer_sec}초)")
    print(f"  추적한 객체 {len(history)}개: " + (", ".join(f"{k} {v}" for k, v in per_class.most_common()) or "없음"))
    print(f"  궤적 좌표 {len(rows)}줄 → trajectory.csv" + (", trajectory.png" if history else ""))
    print("  result.mp4" + ("" if args.no_gif else ", result.gif"))


if __name__ == "__main__":
    main()
