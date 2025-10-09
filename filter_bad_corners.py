#!/usr/bin/env python3
"""Filter out skewed / inconsistent 4-corner annotation files before alignment.

Moves suspicious <idx>_corners.txt (and paired images) into a 'filtered_bad'
subfolder so subsequent warp/rectify steps skip them. Guarantees we do not
remove 3 consecutive frames: in any run of 3+ flagged frames we'll keep the
single frame with the lowest error score.

Usage:
  python3 filter_bad_corners.py --corners-dir out_annot_..._unpad40_v1 --indices 43-74

"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import math
import shutil

CornerMap = Dict[str, Tuple[float, float]]


def parse_indices(spec: str) -> List[int]:
    parts = spec.replace(',', ' ').split()
    out: List[int] = []
    for p in parts:
        if '-' in p:
            a,b = p.split('-',1)
            out.extend(range(int(a), int(b)+1))
        else:
            out.append(int(p))
    return sorted(set(out))


def read_corners(path: Path) -> CornerMap:
    pts: CornerMap = {}
    if not path.exists():
        return pts
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2])
        except Exception:
            continue
        pts[key] = (x, y)
    return pts


def corner_error_score(corners: CornerMap, img_shape: Tuple[int,int], vx_frac: float, hy_frac: float) -> float:
    """Compute a normalized score: deviations relative to the container bbox size.
    Lower is better."""
    h, w = img_shape
    tlx, tly = corners['TL']
    trx, try_ = corners['TR']
    brx, bry = corners['BR']
    blx, bly = corners['BL']

    dx_left = abs(tlx - blx)
    dx_right = abs(trx - brx)
    dy_top = abs(tly - try_)
    dy_bot = abs(bly - bry)

    # container bbox
    left_x = 0.5 * (tlx + blx)
    right_x = 0.5 * (trx + brx)
    top_y = 0.5 * (tly + try_)
    bot_y = 0.5 * (bly + bry)
    box_w = max(1.0, right_x - left_x)
    box_h = max(1.0, bot_y - top_y)

    nx_left = dx_left / box_w
    nx_right = dx_right / box_w
    ny_top = dy_top / box_h
    ny_bot = dy_bot / box_h

    order_pen = 0.0
    if not (tlx < trx and blx < brx and tly < bly and try_ < bry):
        order_pen = 1.0

    score = nx_left + nx_right + ny_top + ny_bot + order_pen
    return float(score)


def looks_bad(corners: CornerMap, img_shape: Tuple[int,int], vx_frac: float, hy_frac: float) -> Tuple[bool, float]:
    """Return (is_bad, score). Compare deviations relative to bbox size
    (box width/height) rather than the full image. This better detects
    mixed-container corners where one corner sits far off the container edge.
    """
    if not corners or any(k not in corners for k in ('TL','TR','BR','BL')):
        return True, 10.0
    h,w = img_shape
    tlx, tly = corners['TL']
    trx, try_ = corners['TR']
    brx, bry = corners['BR']
    blx, bly = corners['BL']

    left_x = 0.5*(tlx + blx)
    right_x = 0.5*(trx + brx)
    top_y = 0.5*(tly + try_)
    bot_y = 0.5*(bly + bry)
    box_w = max(1.0, right_x - left_x)
    box_h = max(1.0, bot_y - top_y)

    dx_left = abs(tlx - blx)
    dx_right = abs(trx - brx)
    dy_top = abs(tly - try_)
    dy_bot = abs(bly - bry)

    # thresholds are fractions of the box size
    dx_thresh = max(4.0, vx_frac * box_w)
    dy_thresh = max(3.0, hy_frac * box_h)

    fails = 0
    if dx_left > dx_thresh:
        fails += 1
    if dx_right > dx_thresh:
        fails += 1
    if dy_top > dy_thresh:
        fails += 1
    if dy_bot > dy_thresh:
        fails += 1

    # consider very narrow box (head-only) but only mark bad if combined with other
    narrow = (box_w < 0.08 * float(w))

    score = corner_error_score(corners, img_shape, vx_frac, hy_frac)

    is_bad = (fails >= 2) or (fails >= 1 and narrow)
    return is_bad, float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corners-dir', required=True)
    ap.add_argument('--indices', required=True)
    ap.add_argument('--img-dir', required=False, default=None,
                    help='If provided, move matching images too')
    ap.add_argument('--vx-thresh', type=float, default=0.18)
    ap.add_argument('--hy-thresh', type=float, default=0.12)
    args = ap.parse_args()

    corners_dir = Path(args.corners_dir)
    if not corners_dir.exists():
        print('Corners dir not found:', corners_dir); return

    idxs = parse_indices(args.indices)
    if not idxs:
        print('No indices parsed'); return

    bad_dir = corners_dir / 'filtered_bad'
    bad_dir.mkdir(parents=True, exist_ok=True)

    flags = {}
    scores = {}

    for i in idxs:
        p = corners_dir / f"{i}_corners.txt"
        if not p.exists():
            flags[i] = False; scores[i] = 0.0; continue
        corners = read_corners(p)
        # attempt to get image size from paired aligned image if exists
        img_h_w = (1080, 1920)
        if args.img_dir:
            imgp = Path(args.img_dir) / f"{i}_cropped_unpadded.jpg"
            if imgp.exists():
                try:
                    import cv2
                    im = cv2.imread(str(imgp))
                    if im is not None:
                        img_h_w = im.shape[:2]
                except Exception:
                    pass
        is_bad, score = looks_bad(corners, img_h_w, args.vx_thresh, args.hy_thresh)
        flags[i] = bool(is_bad)
        scores[i] = float(score)

    # enforce no 3 consecutive flags: for each run of >=3 flagged, keep the one with lowest score
    ordered = idxs
    i = 0
    to_remove = set()
    while i < len(ordered):
        if not flags[ordered[i]]:
            i += 1; continue
        j = i
        run = []
        while j < len(ordered) and flags[ordered[j]]:
            run.append(ordered[j]); j += 1
        if len(run) >= 3:
            # keep the one with smallest score, remove the rest
            best = min(run, key=lambda x: scores.get(x, 1e6))
            for idx in run:
                if idx == best:
                    flags[idx] = False
                else:
                    to_remove.add(idx)
        else:
            for idx in run:
                to_remove.add(idx)
        i = j

    # Move flagged files to filtered_bad
    moved = []
    for idx in ordered:
        if flags.get(idx, False) or idx in to_remove:
            src = corners_dir / f"{idx}_corners.txt"
            if src.exists():
                dst = bad_dir / src.name
                shutil.move(str(src), str(dst))
                moved.append(src.name)
            # also try to move annotated image and corners.jpg if present
            ann = corners_dir / f"{idx}_annot_cropped_padded.jpg"
            if ann.exists():
                shutil.move(str(ann), str(bad_dir / ann.name))
            anntxt = corners_dir / f"{idx}_annot_cropped_padded_jpg.txt"
            if anntxt.exists():
                shutil.move(str(anntxt), str(bad_dir / anntxt.name))
            # also move any _corners.jpg
            cj = corners_dir / f"{idx}_corners.jpg"
            if cj.exists():
                shutil.move(str(cj), str(bad_dir / cj.name))

    print('Filtered files moved to', bad_dir)
    print('Moved:', moved)


if __name__ == '__main__':
    main()
