#!/usr/bin/env python3
"""
Trim pure-black left/right borders for images in a folder and update the corresponding
"*_aligned_corners.txt" files by shifting X coordinates.

Usage examples:
  python3 trim_black_sides_update_corners.py --dir aligned_cropped
  python3 trim_black_sides_update_corners.py --dir aligned_cropped --dir aligned_viz_cropped
  python3 trim_black_sides_update_corners.py --dir aligned_cropped --black-thr 8 --min-nonblack-frac 0.01

Notes:
  - Only trims contiguous black columns from the left and right edges.
  - A column is considered black if the fraction of non-black pixels in that column is < min_nonblack_frac
    where a non-black pixel has max(channel) > black_thr.
  - Corner files are expected alongside the image as "<index>_aligned_corners.txt".
    Example image: 42_aligned.jpg -> corners: 42_aligned_corners.txt
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import re


def detect_black_side_crops(img: np.ndarray, black_thr: int = 8, min_nonblack_frac: float = 0.01):
    h, w = img.shape[:2]
    if img.ndim == 2:
        maxc = img
    else:
        maxc = img.max(axis=2)

    # per-column fraction of non-black pixels
    nonblack = (maxc > black_thr).astype(np.uint8)
    col_ratio = nonblack.sum(axis=0) / float(h)

    black_col = col_ratio < float(min_nonblack_frac)

    # leading black run
    left = 0
    while left < w and black_col[left]:
        left += 1

    # trailing black run
    right = 0
    while right < w and black_col[w - 1 - right]:
        right += 1

    # sanity: avoid over-cropping
    if left + right >= w - 5:
        return 0, 0
    return left, right


def read_corners_file(path: Path):
    corners = {}
    if not path.exists():
        return corners
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 3:
            key = parts[0]
            try:
                x = float(parts[1]); y = float(parts[2])
            except Exception:
                continue
            corners[key] = (x, y)
    return corners


def write_corners_file(path: Path, corners):
    keys = ['TL', 'TR', 'BR', 'BL']
    lines = []
    for k in keys:
        if k in corners:
            x, y = corners[k]
            lines.append(f"{k} {x:.2f} {y:.2f}")
    path.write_text("\n".join(lines) + "\n")


def shift_corners_x(corners, dx: float):
    out = {}
    for k, (x, y) in corners.items():
        out[k] = (x - dx, y)
    return out


def process_dir(root: Path, pattern: str = '*_aligned.jpg', black_thr: int = 8, min_nonblack_frac: float = 0.01, inplace: bool = True):
    files = sorted(root.glob(pattern))
    touched = 0
    for img_path in files:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print('Skip (read fail):', img_path)
            continue

        left, right = detect_black_side_crops(img, black_thr=black_thr, min_nonblack_frac=min_nonblack_frac)
        if left == 0 and right == 0:
            continue

        h, w = img.shape[:2]
        new_img = img[:, left:w - right] if right > 0 else img[:, left:]

        # write image
        out_path = img_path if inplace else img_path.with_name(img_path.stem + '_lrtrim' + img_path.suffix)
        ok = cv2.imwrite(str(out_path), new_img)
        if not ok:
            print('Write failed:', out_path)
            continue

        # update corners if present
        corn_path = img_path.with_name(img_path.stem.replace('_aligned', '_aligned_corners') + '.txt')
        if corn_path.exists():
            corners = read_corners_file(corn_path)
            if corners:
                shifted = shift_corners_x(corners, dx=left)
                write_corners_file(corn_path, shifted)

        touched += 1
        print(f"Trimmed {img_path.name}: left={left}, right={right} -> {new_img.shape[1]}px wide")
    print(f"Done {root}: {touched} files updated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', action='append', required=True, help='Directory to process (repeatable)')
    ap.add_argument('--black-thr', type=int, default=8, help='Pixel threshold (max channel) to consider black')
    ap.add_argument('--min-nonblack-frac', type=float, default=0.01, help='Min fraction of non-black pixels per column to keep')
    ap.add_argument('--no-inplace', action='store_true', help='Write new files with suffix instead of overwriting')
    args = ap.parse_args()

    dirs = [Path(d) for d in args.dir]
    for d in dirs:
        if not d.exists():
            print('Missing dir:', d)
            continue
        process_dir(d, inplace=not args.no_inplace, black_thr=args.black_thr, min_nonblack_frac=args.min_nonblack_frac)


if __name__ == '__main__':
    main()
