#!/usr/bin/env python3
"""Crop exact left/right pixels (40px each) from a set of images and update
corresponding corner annotation files (TL/TR/BR/BL). Writes results into
separate image and corner output directories so each pipeline step is tracked.

Usage example:
  python3 crop_unpad40_update_corners.py \
      --src-dir mnt/2025-10-08_11-26-25 \
      --corners-dir out_annot_2025-10-08_11-26-25_pad_yolo_v1 \
      --out-img-dir images_unpad40_2025-10-08_11-26-25_v1 \
      --out-corners-dir out_annot_2025-10-08_11-26-25_unpad40_v1 \
      --indices 43-74
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List
import cv2
import numpy as np

CornerMap = Dict[str, Tuple[float, float]]


def parse_indices(spec: str) -> List[int]:
    parts = spec.replace(',', ' ').split()
    out = []
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
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        key = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2])
        except Exception:
            continue
        pts[key] = (x, y)
    return pts


def write_corners(path: Path, pts: CornerMap):
    order = ["TL","TR","BR","BL"]
    lines = []
    for k in order:
        if k not in pts:
            continue
        x,y = pts[k]
        lines.append(f"{k} {int(round(x))} {int(round(y))}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--corners-dir', required=True)
    ap.add_argument('--out-img-dir', required=True)
    ap.add_argument('--out-corners-dir', required=True)
    ap.add_argument('--indices', required=True)
    ap.add_argument('--crop-left', type=int, default=40)
    ap.add_argument('--crop-right', type=int, default=40)
    ap.add_argument('--src-suffix', default='_cropped_padded.jpg')
    ap.add_argument('--out-suffix', default='_cropped_unpadded.jpg')
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    corners_dir = Path(args.corners_dir)
    out_img_dir = Path(args.out_img_dir); out_img_dir.mkdir(parents=True, exist_ok=True)
    out_corners_dir = Path(args.out_corners_dir); out_corners_dir.mkdir(parents=True, exist_ok=True)

    idxs = parse_indices(args.indices)
    if not idxs:
        print('No indices parsed'); return

    left = int(args.crop_left); right = int(args.crop_right)

    for i in idxs:
        src_img = src_dir / f"{i}{args.src_suffix}"
        if not src_img.exists():
            print(f"Missing image: {src_img}"); continue
        img = cv2.imread(str(src_img))
        if img is None:
            print(f"Read fail: {src_img}"); continue
        h,w = img.shape[:2]
        if left + right >= w - 2:
            print(f"Skipping {i}: crop too large for width={w}")
            continue
        cropped = img[:, left: w-right]
        out_img = out_img_dir / f"{i}{args.out_suffix}"
        cv2.imwrite(str(out_img), cropped)
        print(f"Wrote image: {out_img.name} ({cropped.shape[1]}x{cropped.shape[0]})")

        # update corners (if exist)
        in_corner = corners_dir / f"{i}_corners.txt"
        if in_corner.exists():
            pts = read_corners(in_corner)
            if pts:
                new_pts: CornerMap = {}
                new_w = cropped.shape[1]
                for k,(x,y) in pts.items():
                    nx = float(x) - float(left)
                    nx = max(0.0, min(float(new_w-1), nx))
                    new_pts[k] = (nx, float(y))
                out_corner = out_corners_dir / f"{i}_corners.txt"
                write_corners(out_corner, new_pts)
                print(f"Wrote corners: {out_corner.name}")
        else:
            print(f"No corners for {i} at {in_corner}")

    print('Done.')


if __name__ == '__main__':
    main()
