#!/usr/bin/env python3
"""박스 초안(ok)과 검토를 마친 check 사진(reviewed)을 모아 학습·검증·시험으로 나눈다 (Roboflow의 Generate 대신).

같은 영상에서 0.33초 간격으로 뽑은 사진은 거의 똑같아서, 한 장씩 무작위로 나누면 검증·시험 사진의
'쌍둥이'가 학습에 들어가 점수가 부풀어요. 그래서 영상마다 2초(60프레임) 구간을 묶음으로 나눠
묶음 단위로 train / valid / test 에 넣는다.

--rot90 을 주면 train 사진마다 90° 돌린 사본을 하나 더 넣는다 (시계·반시계 중 무작위).
과자별 영상은 봉지가 대부분 세로로 서 있어서, 바닥에 가로로 놓인 피에짜를 빅파이로 착각했다.
90° 회전은 박스가 정확히 따라 돌아가므로(축에 맞춘 박스 그대로) 라벨이 틀어지지 않는다. valid/test 는 그대로 둔다.

사용법 (프로젝트 폴더에서):
  .venv/bin/python make_dataset.py --rot90              # → datasets/snacks/ (data.yaml 포함)
  .venv/bin/python make_dataset.py --rot90 --overwrite  # 이전 결과를 지우고 다시
"""
import argparse
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["bigpie", "frenchpie", "piezza"]
SPLITS = ("train", "valid", "test")


def read_exclude(path):
    if not path or not Path(path).exists():
        return set()
    names = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        name = line.split("#")[0].strip()
        if name:
            names.add(name)
    return names


def collect(srcs, exclude):
    """(video, frame, image, label) 목록. 라벨 파일이 없는 사진은 쓰지 않는다."""
    items, seen, skipped = [], set(), Counter()
    for s in map(Path, srcs):
        img_dir, lab_dir = s / "train/images", s / "train/labels"
        if not img_dir.is_dir():
            sys.exit(f"사진 폴더가 없어요: {img_dir}")
        for img in sorted(img_dir.glob("*.jpg")):
            if img.name in exclude:
                skipped["제외 목록"] += 1
                continue
            lab = lab_dir / f"{img.stem}.txt"
            if not lab.exists():
                skipped["라벨 없음"] += 1
                continue
            if img.name in seen:
                sys.exit(f"같은 이름의 사진이 두 폴더에 있어요: {img.name}")
            seen.add(img.name)
            video, _, frame = img.stem.rpartition("_")
            if not frame.isdigit():
                sys.exit(f"파일 이름이 <영상>_<프레임번호>.jpg 꼴이 아니에요: {img.name}")
            items.append((video, int(frame), img, lab))
    return items, skipped


def assign(items, block_frames, ratio, seed):
    """영상마다 block_frames 구간 묶음을 섞어 val·test 몫을 먼저 떼고 나머지는 train."""
    rng = random.Random(seed)
    blocks = defaultdict(lambda: defaultdict(list))
    for it in items:
        blocks[it[0]][it[1] // block_frames].append(it)
    out = {s: [] for s in SPLITS}
    for video in sorted(blocks):
        ids = sorted(blocks[video])
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * ratio[2])) if n >= 5 else 0
        n_val = max(1, round(n * ratio[1])) if n >= 3 else 0
        for k, b in enumerate(ids):
            split = "test" if k < n_test else "valid" if k < n_test + n_val else "train"
            out[split] += blocks[video][b]
    return out


def rotate90(img_path, lab_path, out_img, out_lab, clockwise):
    """사진을 90° 돌리고 YOLO 박스(정규화 중심·크기)도 같이 돌린다."""
    import cv2
    img = cv2.imread(str(img_path))
    cv2.imwrite(str(out_img), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if clockwise else cv2.ROTATE_90_COUNTERCLOCKWISE),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    lines = []
    for line in lab_path.read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        nx, ny = (1 - y, x) if clockwise else (y, 1 - x)      # 시계: (x, y) → (1-y, x), 반시계: (y, 1-x)
        lines.append(f"{c} {nx:.6f} {ny:.6f} {h:.6f} {w:.6f}")
    out_lab.write_text("".join(l + "\n" for l in lines))


def main():
    p = argparse.ArgumentParser(description="초안 라벨 → 학습/검증/시험 데이터셋")
    p.add_argument("src", nargs="*", default=["datasets/prelabel/ok", "datasets/prelabel/reviewed",
                                              "datasets/prelabel_mix/ok", "datasets/prelabel_mix/reviewed"])
    p.add_argument("--exclude", default="docs/labeling/ok_excluded.txt", help="쓰지 않을 사진 이름 목록")
    p.add_argument("--out", default="datasets/snacks")
    p.add_argument("--block-frames", type=int, default=60, help="묶음 길이(원본 영상 프레임 수, 기본 60 = 30fps에서 2초)")
    p.add_argument("--ratio", default="0.7,0.2,0.1", help="train,valid,test 비율 (묶음 기준)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rot90", action="store_true", help="train 사진마다 90° 돌린 사본 추가")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    ratio = [float(x) for x in args.ratio.split(",")]
    if len(ratio) != 3 or abs(sum(ratio) - 1) > 1e-6:
        p.error("--ratio 는 합이 1인 숫자 3개예요 (예: 0.7,0.2,0.1)")
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            sys.exit(f"{out}/ 에 이전 결과가 있어요. 지우고 다시 하려면 --overwrite 를 붙이세요.")
        for name in (*SPLITS, "data.yaml", "data_local.yaml"):
            q = out / name
            if q.is_dir():
                shutil.rmtree(q)
            elif q.exists():
                q.unlink()

    items, skipped = collect(args.src, read_exclude(args.exclude))
    if not items:
        sys.exit("쓸 사진이 없어요.")
    split = assign(items, args.block_frames, ratio, args.seed)

    table = {s: Counter() for s in SPLITS}
    for s in SPLITS:
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
        for video, _, img, lab in split[s]:
            shutil.copy2(img, out / s / "images" / img.name)
            shutil.copy2(lab, out / s / "labels" / lab.name)
            for line in lab.read_text().splitlines():
                if line.strip():
                    table[s][CLASSES[int(line.split()[0])]] += 1
    n_rot = 0
    if args.rot90:
        rng = random.Random(args.seed + 1)
        for video, _, img, lab in split["train"]:
            name = f"{img.stem}_r90"
            rotate90(img, lab, out / "train/images" / f"{name}.jpg", out / "train/labels" / f"{name}.txt", rng.random() < 0.5)
            n_rot += 1
    (out / "data.yaml").write_text(
        "train: train/images\nval: valid/images\ntest: test/images\n\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n", encoding="utf-8")

    print(f"사진 {len(items)}장 → {out}/  (뺀 사진: " + ", ".join(f"{k} {v}" for k, v in skipped.items()) + ")")
    print(f"{'':<8}" + "".join(f"{c:>11}" for c in CLASSES) + f"{'합계':>8}")
    for s in SPLITS:
        print(f"{s:<8}" + "".join(f"{table[s][c]:>11}" for c in CLASSES) + f"{len(split[s]):>9}")
    if n_rot:
        print(f"(train 에 90° 돌린 사본 {n_rot}장 추가 — 위 표는 원본 기준)")


if __name__ == "__main__":
    main()
