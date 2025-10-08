#!/usr/bin/env python3
"""Trim left/right black borders from images and keep corner annotations in sync."""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


CornerMap = Dict[str, Tuple[float, float]]


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
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        pts[key] = (x, y)
    return pts


def write_corners(path: Path, pts: CornerMap):
    order = ["TL", "TR", "BR", "BL"]
    lines = []
    for key in order:
        if key not in pts:
            continue
        x, y = pts[key]
        lines.append(f"{key} {int(round(x))} {int(round(y))}")
    path.write_text("\n".join(lines) + "\n")


def find_lr_bounds(gray: np.ndarray, thresh: int, min_row_frac: float) -> Tuple[int, int]:
    h, w = gray.shape[:2]
    min_rows = max(1, int(round(h * float(min_row_frac))))
    valid = (gray > thresh).sum(axis=0)

    left = 0
    while left < w and valid[left] < min_rows:
        left += 1

    right = w - 1
    while right >= 0 and valid[right] < min_rows:
        right -= 1

    if left >= right:
        return 0, w - 1
    return left, right


def update_corners(corners_path: Path, left_trim: int, new_width: int):
    pts = read_corners(corners_path)
    if not pts:
        return
    max_x = max(0, new_width - 1)
    for key, (x, y) in list(pts.items()):
        pts[key] = (min(max_x, max(0.0, x - left_trim)), y)
    write_corners(corners_path, pts)


def process_image(img_path: Path, corners_suffix: str, inplace: bool, thresh: int, min_row_frac: float) -> bool:
    img = cv2.imread(str(img_path))
    if img is None:
        print("Skip unreadable:", img_path)
        return False

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left, right = find_lr_bounds(gray, thresh=thresh, min_row_frac=min_row_frac)
    # ignore tiny adjustments (<= 1 px per side)
    if left <= 1 and (w - 1 - right) <= 1:
        return False

    cropped = img[:, left:right + 1]

    idx = img_path.name.split('_')[0]
    corners_path = img_path.parent / f"{idx}{corners_suffix}"
    if corners_path.exists():
        update_corners(corners_path, left, cropped.shape[1])

    if inplace:
        cv2.imwrite(str(img_path), cropped)
    else:
        out_path = img_path.with_name(img_path.stem + '_trimlr' + img_path.suffix)
        cv2.imwrite(str(out_path), cropped)

    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dirs', nargs='+', required=True, help='Directories to process (e.g. aligned aligned_viz)')
    ap.add_argument('--suffix', default='_aligned.jpg', help='Image filename suffix to match (default: _aligned.jpg)')
    ap.add_argument('--corners-suffix', default='_aligned_corners.txt', help='Corners filename suffix (default: _aligned_corners.txt)')
    ap.add_argument('--thresh', type=int, default=8, help='Grayscale threshold to treat as non-black (default: 8)')
    ap.add_argument('--min-row-frac', type=float, default=0.10, help='Fraction of rows that must be non-black for a column to be kept (default: 0.10)')
    ap.add_argument('--inplace', action='store_true', help='Overwrite images in place (default: write *_trimlr.jpg copies)')
    args = ap.parse_args()

    total = 0
    for d in args.dirs:
        root = Path(d)
        if not root.exists():
            print('Directory not found:', root)
            continue
        for img_path in sorted(root.glob(f'*{args.suffix}')):
            if process_image(img_path, args.corners_suffix, args.inplace, args.thresh, args.min_row_frac):
                total += 1

    print(f'Done. Trimmed {total} image(s).')


if __name__ == '__main__':
    main()
