#!/usr/bin/env python3
"""사진에 과자 박스 초안(YOLO 라벨)을 자동으로 만든다. Roboflow에 올린 뒤 고치기만 하면 된다.

1) YOLOE(찾을 물체를 글로 알려 주는 감지 모델)로 사진 속 과자 봉지를 찾고
2) CLIP(사진과 글을 짝짓는 모델)으로 박스마다 빅파이 / 후렌치파이 / 피에짜 / 기타(손·리모컨·이불) 중 무엇인지 고른다
3) 파일 이름 앞부분(bigpie_a_00012.jpg)이나 폴더 이름을 그 사진의 과자로 쓴다.
   과자별 영상에서 뽑은 사진이라 과자가 하나뿐이므로, 감지 신뢰도 × CLIP 확률이 가장 높은 박스 하나만 라벨로 쓴다.
   신뢰도가 낮거나, 같은 과자로 보이는 박스가 둘이거나, 박스가 없으면 '확인 필요' 묶음으로 따로 모은다
   (mix_ 처럼 과자가 정해지지 않은 프레임은 박스마다 CLIP 판단을 그대로 쓴다.
    neg_ 는 과자가 없어야 하는 사진이라 빈 라벨을 쓰고, 뭔가 잡히면 확인 묶음으로 보낸다)

사용법 (프로젝트 폴더에서):
  .venv/bin/python prelabel.py                     # frames/ → datasets/prelabel/
  .venv/bin/python prelabel.py --overwrite         # 이전 결과를 지우고 다시
  .venv/bin/python prelabel.py --multi             # 사진 한 장에 같은 과자가 여럿 있을 수 있을 때

결과 (datasets/prelabel/):
  ok/         초안이 그럴듯한 사진 → Roboflow 업로드 묶음 1
  check/      확인이 필요한 사진 → 업로드 묶음 2 (꼼꼼히 보기)
  review/     check 사진마다 박스를 그린 미리보기 (? 붙은 굵은 박스가 확인할 박스)
  review.csv  사진별 박스·확인 이유

처음 실행하면 모델을 내려받는다: yoloe-26x-seg.pt(172MB)·mobileclip2_b.ts(254MB)는 프로젝트 폴더에,
CLIP ViT-B/32(354MB)는 ~/.cache/clip 에.
"""
import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["bigpie", "frenchpie", "piezza"]    # Roboflow·data.yaml 클래스 순서 (0, 1, 2) — 빅파이, 후렌치파이, 피자과자(피에짜)
# 모두 '과자' 하나로 취급. 긴 봉지(피에짜)는 일반 문장만 쓰면 위쪽 2/3만 감싸는 일이 많아 '긴 봉지'·색 문장을 더했다
DETECT_PROMPTS = ["snack package", "snack bag", "food package", "candy bar wrapper", "plastic wrapper",
                  "long snack bar package", "yellow snack package", "red snack package", "white snack package"]
BRAND_PROMPTS = {     # 한국 과자 이름은 CLIP이 잘 몰라서 포장 색·모양으로 설명한다
    "bigpie": ["a red and yellow snack wrapper", "a red Big Pie chocolate pie package",
               "a red snack bag with white Korean letters"],
    "frenchpie": ["a white and green snack wrapper", "a white French Pie snack package with green letters",
                  "a plain white plastic snack wrapper with green edges"],      # 뒷면은 거의 흰색
    "piezza": ["a long yellow snack wrapper with pizza pictures", "a yellow Piezza pizza cracker package",
               "a yellow snack bar wrapper"],
    "other": ["a TV remote control", "a black phone case", "a human hand", "fingers", "a wrist with a watch",
              "a blanket", "a bed sheet with a pattern", "a pillow", "a wooden cabinet"],
}
COLORS = {"bigpie": (40, 40, 230), "frenchpie": (60, 200, 60), "piezza": (0, 210, 255)}    # BGR
ALT_COLOR = (170, 170, 170)                            # 라벨로 쓰지 않은 후보 박스 (미리보기)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNSUPPORTED = {".heic", ".heif", ".tif", ".tiff", ".gif", ".avif"}
OUTPUTS = ("ok", "check", "review", "review.csv")    # --overwrite 때 지우는 것은 이것뿐
MIN_SIDE = 2                                         # 이보다 좁은 박스는 버린다 (px)
TILE = 420                                           # 미리보기 칸 크기


def prior_class(path):
    """폴더 이름 → 파일 이름 앞부분 순으로 기본 클래스를 정한다 (대소문자 무시). neg 는 과자가 없어야 하는 사진."""
    folder = path.parent.name.lower()
    if folder in CLASSES or folder == "neg":
        return folder
    head = path.stem.split("_")[0].lower()
    return head if head in CLASSES or head == "neg" else None


def detect(model, img, sizes, conf, device):
    """여러 크기로 찾아 합친다. 화면을 꽉 채운 큰 물체는 작은 크기(320)에서, 멀리 있는 작은 물체는 큰 크기(960)에서 잘 잡힌다."""
    import torch
    from torchvision.ops import nms
    xyxy, scores = [], []
    for s in sizes:
        r = model.predict(img, conf=conf, imgsz=s, iou=0.5, agnostic_nms=True, device=device, verbose=False)[0]
        xyxy.append(r.boxes.xyxy.cpu())
        scores.append(r.boxes.conf.cpu())
    xyxy, scores = torch.cat(xyxy), torch.cat(scores)
    keep = nms(xyxy, scores, 0.5)
    boxes = [[*xyxy[i].tolist(), float(scores[i])] for i in keep]
    return [b for b in boxes if b[2] - b[0] >= MIN_SIDE and b[3] - b[1] >= MIN_SIDE]


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def _inside(inner, outer):
    """inner 박스가 outer 안에 80% 이상 들어 있는가."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy >= 0.8 * _area(inner)


def _covered(outer, inners, n=48):
    """inners 가 outer 넓이의 몇 %를 덮는지 (격자로 어림)."""
    X, Y = np.meshgrid(np.linspace(outer[0], outer[2], n), np.linspace(outer[1], outer[3], n))
    m = np.zeros_like(X, bool)
    for b in inners:
        m |= (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
    return float(m.mean())


def tidy_boxes(boxes, drop_groups=True):
    """여러 개를 한꺼번에 감싼 큰 박스를 빼고(안의 박스 2개 이상이 넓이 60% 이상을 덮을 때),
    그래도 남은 박스끼리 거의 통째로 겹치면 nested=True (한 과자에 박스 2개 / 과자 2개를 한 박스로 — 사람이 볼 것).
    drop_groups=False: 과자가 하나뿐인 사진 — 봉지를 감싼 박스가 안쪽 그림 박스들 때문에 지워지지 않게 빼기를 건너뛴다."""
    drop = set()
    for i, a in enumerate(boxes if drop_groups else []):
        inner = [b for j, b in enumerate(boxes) if j != i and _area(b) < _area(a) and _inside(b, a)]
        if len(inner) >= 2 and _covered(a, inner) >= 0.6:
            drop.add(i)
    kept = [b for i, b in enumerate(boxes) if i not in drop]
    nested = any(i != j and _inside(b, a) for i, a in enumerate(kept) for j, b in enumerate(kept))
    return kept, nested


class BrandClassifier:
    def __init__(self, device):
        import torch
        try:
            import clip
        except ImportError:             # ultralytics 가 쓰는 CLIP 패키지 (YOLOE 첫 실행 때 보통 이미 설치됨)
            from ultralytics.utils.checks import check_requirements
            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip
        from PIL import Image
        self.torch, self.Image, self.device = torch, Image, device
        self.model, self.pre = clip.load("ViT-B/32", device=device)
        self.keys = [k for k, v in BRAND_PROMPTS.items() for _ in v]
        texts = [t for v in BRAND_PROMPTS.values() for t in v]
        with torch.no_grad():
            t = self.model.encode_text(clip.tokenize(texts).to(device)).float()
        self.text = t / t.norm(dim=-1, keepdim=True)

    def __call__(self, img, boxes):
        """박스마다 {클래스…, other: 확률} (박스 주변 5% 여유를 두고 자른다 — 넓히면 옆 물체가 섞임)."""
        if not boxes:
            return []
        h, w = img.shape[:2]
        crops = []
        for x1, y1, x2, y2, _ in boxes:
            px, py = (x2 - x1) * 0.05, (y2 - y1) * 0.05
            xa, ya = min(w - 1, int(max(0, x1 - px))), min(h - 1, int(max(0, y1 - py)))
            xb, yb = max(xa + 1, int(min(w, x2 + px))), max(ya + 1, int(min(h, y2 + py)))
            crops.append(self.pre(self.Image.fromarray(cv2.cvtColor(img[ya:yb, xa:xb], cv2.COLOR_BGR2RGB))))
        with self.torch.no_grad():
            f = self.model.encode_image(self.torch.stack(crops).to(self.device)).float()
            f = f / f.norm(dim=-1, keepdim=True)
            logits = (100 * f @ self.text.T).cpu()
        # 묶음마다 가장 잘 맞는 문장 하나로 겨룬다 (문장 수가 많은 '기타'가 더하기로 유리해지지 않게)
        groups = list(BRAND_PROMPTS)
        idx = {g: [i for i, k in enumerate(self.keys) if k == g] for g in groups}
        best = self.torch.stack([logits[:, idx[g]].max(dim=1).values for g in groups], dim=1)
        return [dict(zip(groups, row.tolist())) for row in best.softmax(dim=1)]


def decide(prior, score, min_brand):
    """(--multi 또는 과자가 정해지지 않은 사진) 박스 하나의 클래스와 확인 이유(없으면 '')."""
    brand = max(CLASSES, key=lambda k: score[k])
    top = max(score, key=score.get)
    if prior == "neg":
        return brand, "과자가 없어야 하는 사진인데 잡힘(neg — 라벨은 비워 둠)"
    if top == "other":
        return (prior or brand), f"과자가 아닌 물체일 수 있음({score['other']:.2f})"
    if prior and top != prior:
        if score[top] >= min_brand:
            return top, f"파일은 {prior}인데 {top}로 보임({score[top]:.2f})"
        return prior, f"과자 종류 불확실({top} {score[top]:.2f})"
    if score[top] < min_brand:
        return top, f"과자 종류 불확실({top} {score[top]:.2f})"
    return top, ""


def pick_one(prior, boxes, probs, min_brand, min_det):
    """과자별 영상의 사진 — 그 과자 하나만 있으므로 감지 신뢰도 × CLIP 확률이 가장 높은 박스 하나를 고른다.
    돌려주는 것: (고른 박스 또는 None, 확인 이유 목록, 미리보기에 같이 그릴 다른 후보 박스)."""
    cand = sorted(((b[4] * p[prior], b, p) for b, p in zip(boxes, probs) if max(p, key=p.get) == prior),
                  key=lambda t: t[0], reverse=True)
    if not cand:
        return None, ["박스 없음 — 직접 그리기" if not boxes else "과자로 보이는 박스 없음 — 직접 그리기"], boxes
    s, best, p = cand[0]
    reasons, alts = [], []

    def flag(b, reason):
        if all(b is not a for a in alts):
            alts.append(b)
        if reason not in reasons:
            reasons.append(reason)

    if best[4] < min_det or p[prior] < min_brand:
        reasons.append(f"확신 낮음(감지 {best[4]:.2f} · CLIP {p[prior]:.2f})")
    # 같은 과자로 보이는 다른 박스: 고른 박스 안쪽의 작고 덜 확실한 박스(포장의 그림·접힌 끝)만 버리고,
    # 밖으로 벗어났거나, 안쪽이어도 크기가 비슷하거나, 점수가 거의 같으면(고른 박스가 너무 큼) 사람이 고르게 한다
    for s2, b, _ in cand[1:]:
        if (s2 >= 0.5 * s and (not _inside(b, best) or _area(b) >= 0.4 * _area(best))) or s2 >= 0.8 * s:
            flag(b, "같은 과자로 보이는 박스가 2개 이상 — 맞는 박스 고르기")
    for b, q in zip(boxes, probs):
        if b is best:
            continue
        top = max(q, key=q.get)
        # 고른 박스를 통째로 감싸는 박스 — 고른 박스가 봉지 일부만 감쌌을 수 있음
        if b[4] >= min(min_det, 0.6 * best[4]) and _area(b) > 1.1 * _area(best) and _inside(best, b):
            flag(b, "고른 박스를 감싸는 더 큰 박스가 있음 — 봉지 일부만 잡았는지 확인")
        # 사진에는 이 과자 하나뿐인데 다른 과자로 자신 있게 보이는 박스 — 고른 박스나 CLIP이 틀렸을 수 있음
        elif top in CLASSES and top != prior and b[4] * q[top] >= 0.5 * s and not _inside(b, best):
            flag(b, f"다른 과자({top})로 보이는 박스가 있음")
    return best, reasons, alts


def yolo_line(cls, box, w, h):
    x1, y1, x2, y2 = box[:4]
    return (f"{CLASSES.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")


def write_data_yaml(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "data.yaml").write_text(
        "train: train/images\nval: train/images\n\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n", encoding="utf-8")
    for sub in ("train/images", "train/labels"):
        (folder / sub).mkdir(parents=True, exist_ok=True)


def unique_name(f, used):
    """업로드 폴더 안에서 이름이 겹치지 않게 (확장자·대소문자가 달라도 라벨 파일 이름은 같아지므로 stem 기준)."""
    cands = [f.name, f"{f.parent.name}_{f.name}"]
    cands += [f"{f.parent.name}_{f.stem}_{n}{f.suffix}" for n in range(2, 100000)]
    for c in cands:
        key = Path(c).stem.lower()
        if key not in used:
            used.add(key)
            return c
    raise RuntimeError(f"이름을 정할 수 없어요: {f}")


def exif_rotated(f):
    """EXIF 회전 정보가 있는 사진인가 (cv2 는 회전해서 읽으므로 라벨도 회전된 기준 → 회전된 그림을 저장해야 맞음)."""
    try:
        from PIL import Image
        return Image.open(f).getexif().get(0x0112, 1) != 1
    except Exception:
        return False


def make_tile(no, name, img, dets):
    """미리보기 한 칸: 박스(클래스 색, 확인할 박스는 굵게 + ?, 라벨로 안 쓴 후보는 회색 alt)와 번호·파일 이름."""
    im = img.copy()
    h, w = im.shape[:2]
    t = max(2, round(max(h, w) / 300))
    for k, (box, cls, reason) in enumerate(dets, 1):
        x1, y1, x2, y2 = map(int, box[:4])
        col = COLORS.get(cls, ALT_COLOR)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, t * (3 if reason else 1))
        cv2.putText(im, f"{k}:{cls or 'alt'}" + ("?" if reason else ""), (x1 + 3, max(18, y1 + 3 + 12 * t)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45 * t, col, max(1, t // 2))
    s = TILE / max(h, w)
    im = cv2.resize(im, (max(1, int(w * s)), max(1, int(h * s))))
    cell = np.full((TILE + 24, TILE, 3), 30, np.uint8)
    cell[:im.shape[0], :im.shape[1]] = im
    cv2.putText(cell, f"{no}. {name}", (4, TILE + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return cell


def save_sheets(tiles, out_dir, per_sheet=12, cols=4):
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(range(0, len(tiles), per_sheet), 1):
        page = tiles[i:i + per_sheet]
        page += [np.full_like(tiles[0], 30)] * (-len(page) % cols)
        sheet = np.vstack([np.hstack(page[r:r + cols]) for r in range(0, len(page), cols)])
        cv2.imwrite(str(out_dir / f"check_{n:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def collect_files(srcs):
    files, skipped = {}, Counter()
    for s in map(Path, srcs):
        if not s.exists():
            sys.exit(f"없는 경로예요: {s}")
        for f in ([s] if s.is_file() else s.rglob("*")):
            if f.suffix.lower() in EXTS:
                files.setdefault(f.resolve(), f)          # 겹치는 폴더를 줘도 한 번만
            elif f.suffix.lower() in UNSUPPORTED:
                skipped[f.suffix.lower()] += 1
    if skipped:
        print("! 지원하지 않는 형식이라 건너뜀: " + ", ".join(f"{k} {v}장" for k, v in skipped.items())
              + " (jpg/png로 바꿔 주세요. 아이폰은 설정 > 카메라 > 포맷 > 높은 호환성)")
    return sorted(files.values())


def prepare_out(out, srcs, overwrite):
    """결과 폴더 준비. 사진 폴더와 겹치면 거절, 지울 때는 이 스크립트가 만든 것만 지운다."""
    o = out.resolve()
    for s in srcs:
        r = Path(s).resolve()
        if o == r or r in o.parents or o in r.parents:
            sys.exit(f"결과 폴더({out})와 사진 폴더({s})가 겹쳐요. --out 을 다른 곳으로 정하세요.")
    if out.exists() and any(out.iterdir()):
        others = [p.name for p in out.iterdir() if p.name not in OUTPUTS and p.name != ".DS_Store"]
        if others:
            sys.exit(f"{out}/ 에 이 스크립트가 만들지 않은 것이 있어요({', '.join(others[:5])}). 다른 --out 을 쓰세요.")
        if not overwrite:
            sys.exit(f"{out}/ 에 이전 결과가 있어요. 지우고 다시 하려면 --overwrite 를 붙이세요.")
        for name in OUTPUTS:
            p = out / name
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
    for sub in ("ok", "check"):
        write_data_yaml(out / sub)


def main():
    p = argparse.ArgumentParser(description="과자 박스 초안 만들기 (Roboflow 업로드용 YOLO 라벨)")
    p.add_argument("src", nargs="*", default=["frames"], help="사진 폴더 (기본 frames)")
    p.add_argument("--out", default="datasets/prelabel")
    p.add_argument("--model", default="yoloe-26x-seg.pt", help="YOLOE 감지 모델")
    p.add_argument("--conf", type=float, default=0.2, help="후보 박스로 볼 최소 감지 신뢰도 (기본 0.2)")
    p.add_argument("--min-det", type=float, default=0.5,
                   help="라벨로 쓸 박스의 감지 신뢰도가 이보다 낮으면 확인 묶음으로 (기본 0.5)")
    p.add_argument("--min-brand", type=float, default=0.5, help="CLIP 과자 판단을 믿을 최소 확률 (기본 0.5)")
    p.add_argument("--multi", action="store_true",
                   help="사진 한 장에 같은 과자가 여럿 있을 수 있음 (기본: 과자별 영상이라 사진마다 하나)")
    p.add_argument("--sizes", default="320,640,960", help="감지에 쓸 이미지 크기들 (쉼표로)")
    p.add_argument("--device", default="mps", help="mps(맥 GPU) / cpu")
    p.add_argument("--overwrite", action="store_true", help="이전 결과를 지우고 다시 만들기")
    args = p.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    files = collect_files(args.src)
    if not files:
        sys.exit(f"사진이 없어요: {args.src}")
    out = Path(args.out)
    prepare_out(out, args.src, args.overwrite)

    from ultralytics import YOLOE
    det = YOLOE(args.model)
    det.set_classes(DETECT_PROMPTS)
    clf = BrandClassifier(args.device)

    rows, tiles, used = [], [], set()
    cls_count, where_count, reason_count = Counter(), Counter(), Counter()
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"! 읽을 수 없어 건너뜀: {f}")
            continue
        h, w = img.shape[:2]
        prior = prior_class(f)
        dets, desc = [], ""
        try:
            single = prior in CLASSES and not args.multi
            boxes, nested = tidy_boxes(detect(det, img, sizes, args.conf, args.device), drop_groups=not single)
            probs = clf(img, boxes)
            desc = " / ".join(f"{max(s, key=s.get)} {b[4]:.2f}/{max(s.values()):.2f}" for b, s in zip(boxes, probs))
            if single:
                best, reasons, rivals = pick_one(prior, boxes, probs, args.min_brand, args.min_det)
                labels = [] if best is None else [(best, prior, "; ".join(reasons))]
                dets = labels + [(b, None, "") for b in rivals]
            else:
                for b, s in zip(boxes, probs):
                    c, rs = decide(prior, s, args.min_brand)
                    if not rs and b[4] < args.min_det:
                        rs = f"확신 낮음(감지 {b[4]:.2f})"
                    dets.append((b, c, rs))
                reasons = [rs for _, _, rs in dets if rs]
                if nested:
                    reasons.append("겹친 박스 — 한 과자에 박스 2개이거나 과자 2개를 한 박스로 묶었을 수 있음")
                if not dets and prior != "neg":
                    reasons.append("박스 없음 — 직접 그리기")
                labels = [] if prior == "neg" else dets      # neg 는 잡힌 게 있어도 빈 라벨 (배경 사진)
        except Exception as e:                           # 한 장이 실패해도 나머지는 계속
            dets, labels = [], None
            reasons = [f"처리 오류 — 직접 그리기 ({type(e).__name__}: {e})"]

        where = "check" if reasons else "ok"
        name = unique_name(f, used)
        dst = out / where / "train/images" / name
        if exif_rotated(f):
            cv2.imwrite(str(dst), img)                   # 회전된 그림을 저장해야 라벨 좌표와 맞음
        else:
            shutil.copy2(f, dst)
        if labels is not None and (labels or prior == "neg"):   # 라벨 파일이 없으면 Roboflow에서 '라벨 없음'으로 보임
            (out / where / "train/labels" / f"{Path(name).stem}.txt").write_text(
                "".join(yolo_line(c, b, w, h) + "\n" for b, c, _ in labels))
        for _, c, _ in (labels or []):
            cls_count[c] += 1
        for rs in reasons:
            reason_count[rs.split("(")[0].split(" —")[0].strip()] += 1
        where_count[where] += 1
        rows.append([name, str(f), prior or "", where, len(labels or []), desc, "; ".join(reasons)])
        if where == "check":
            tiles.append(make_tile(len(tiles) + 1, name, img, dets))
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)}장")

    with open(out / "review.csv", "w", newline="", encoding="utf-8-sig") as fh:   # 엑셀에서 한글이 깨지지 않게
        wr = csv.writer(fh)
        wr.writerow(["file", "source", "file_class", "where", "labels", "all_boxes(clip_top det/clip)", "reasons"])
        wr.writerows(rows)
    if tiles:
        save_sheets(tiles, out / "review")

    print(f"\n사진 {len(rows)}장 → {out}/")
    print(f"  ok {where_count['ok']}장 · check {where_count['check']}장")
    print("  라벨 박스: " + ", ".join(f"{c} {cls_count[c]}" for c in CLASSES) + f" (총 {sum(cls_count.values())}개)")
    if reason_count:
        print("  확인 이유: " + ", ".join(f"{k} {v}" for k, v in reason_count.most_common()))
    print("  미리보기: review/check_*.jpg, 사진별 내용: review.csv")


if __name__ == "__main__":
    main()
