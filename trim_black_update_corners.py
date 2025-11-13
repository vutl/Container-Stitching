#!/usr/bin/env python3
"""Trim black borders on all sides and keep corner annotations in sync.

This script detects non-black columns/rows using a grayscale threshold and a
minimum fraction of non-black pixels per column/row. It crops the image to the
detected bounds and updates corner annotation files by subtracting the left/top
offsets. By default it writes a new file with suffix `_trim` unless `--inplace`
is used.
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np



@dataclass
class CornerRecord:
    x: float
    y: float
    extra: List[str]


CornerMap = Dict[str, CornerRecord]


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
        pts[key] = CornerRecord(x, y, parts[3:])
    return pts


def write_corners(path: Path, pts: CornerMap):
    order = ["TL", "TR", "BR", "BL"]
    lines = []
    for key in order:
        if key not in pts:
            continue
        rec = pts[key]
        line = f"{key} {int(round(rec.x))} {int(round(rec.y))}"
        if rec.extra:
            line += " " + " ".join(rec.extra)
        lines.append(line)
    if lines:
        path.write_text("\n".join(lines) + "\n")


def find_bounds(gray: np.ndarray, thresh: int, min_row_frac: float, min_col_frac: float) -> Tuple[int, int, int, int]:
    """Return (left, right, top, bottom) inclusive bounds to keep.

    - thresh: grayscale value above which a pixel is considered non-black
    - min_row_frac: fraction of rows that must be non-black for a column to be kept
    - min_col_frac: fraction of columns that must be non-black for a row to be kept
    """
    h, w = gray.shape[:2]
    min_rows = max(1, int(round(h * float(min_row_frac))))
    min_cols = max(1, int(round(w * float(min_col_frac))))

    col_nonblack = (gray > thresh).sum(axis=0)
    row_nonblack = (gray > thresh).sum(axis=1)

    left = 0
    while left < w and col_nonblack[left] < min_rows:
        left += 1

    right = w - 1
    while right >= 0 and col_nonblack[right] < min_rows:
        right -= 1

    top = 0
    while top < h and row_nonblack[top] < min_cols:
        top += 1

    bottom = h - 1
    while bottom >= 0 and row_nonblack[bottom] < min_cols:
        bottom -= 1

    # if bounds invalid, return full image
    if left >= right or top >= bottom:
        return 0, w - 1, 0, h - 1
    return left, right, top, bottom


def update_corners(corners_path: Path,
                   left_trim: int,
                   top_trim: int,
                   new_width: int,
                   new_height: int,
                   write_path: Path) -> CornerMap:
    pts = read_corners(corners_path)
    if not pts:
        return {}
    max_x = max(0, new_width - 1)
    max_y = max(0, new_height - 1)
    for key, rec in list(pts.items()):
        nx = min(max_x, max(0.0, rec.x - left_trim))
        ny = min(max_y, max(0.0, rec.y - top_trim))
        pts[key] = CornerRecord(nx, ny, rec.extra)
    write_corners(write_path, pts)

    return pts


def draw_corners(img: np.ndarray,
                 corners: CornerMap,
                 radius: int,
                 thickness: int) -> np.ndarray:
    out = img.copy()
    for key, rec in corners.items():
        cx, cy = int(round(rec.x)), int(round(rec.y))
        cv2.circle(out, (cx, cy), radius, (0, 0, 255), -1)
        cv2.putText(out, key, (cx + radius + 2, cy - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def process_image(img_path: Path,
                  corners_suffix: str,
                  inplace: bool,
                  thresh: int,
                  min_row_frac: float,
                  min_col_frac: float,
                  corners_dir: Optional[Path],
                  corners_out_dir: Optional[Path],
                  out_dir: Optional[Path],
                  vis_dir: Optional[Path],
                  draw_radius: int,
                  draw_thickness: int):
    img = cv2.imread(str(img_path))
    if img is None:
        print("Skip unreadable:", img_path)
        return None, None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    left, right, top, bottom = find_bounds(gray, thresh=thresh, min_row_frac=min_row_frac, min_col_frac=min_col_frac)

    # ignore tiny adjustments (<= 1 px per side)
    if left <= 1 and (w - 1 - right) <= 1 and top <= 1 and (h - 1 - bottom) <= 1:
        return None, None

    cropped = img[top:bottom + 1, left:right + 1]

    # Use the full stem (e.g. 'img_0') so we match corner files like 'img_0_corners.txt'
    idx = img_path.stem
    corners_root = corners_dir if corners_dir is not None else img_path.parent
    corners_out_root = corners_out_dir if corners_out_dir is not None else corners_root
    corners_path = corners_root / f"{idx}{corners_suffix}"
    corners_out_path = corners_out_root / f"{idx}{corners_suffix}"
    updated_pts: CornerMap = {}
    if corners_path.exists():
        corners_out_path.parent.mkdir(parents=True, exist_ok=True)
        updated_pts = update_corners(corners_path,
                                     left,
                                     top,
                                     cropped.shape[1],
                                     cropped.shape[0],
                                     corners_out_path)

    if inplace:
        cv2.imwrite(str(img_path), cropped)
        out_path = img_path
    else:
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{img_path.stem}_trim{img_path.suffix}"
        else:
            out_path = img_path.with_name(img_path.stem + '_trim' + img_path.suffix)
        cv2.imwrite(str(out_path), cropped)

    vis_path = None
    if vis_dir is not None and updated_pts:
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_img = draw_corners(cropped, updated_pts, draw_radius, draw_thickness)
        vis_path = vis_dir / f"{img_path.stem}_trim_vis{img_path.suffix}"
        cv2.imwrite(str(vis_path), vis_img)

    return out_path, vis_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dirs', nargs='+', required=True, help='Directories to process')
    ap.add_argument('--suffix', default='.jpg', help='Image filename suffix to match (default: .jpg)')
    ap.add_argument('--corners-suffix', default='_corners.txt', help='Corners filename suffix (default: _corners.txt)')
    ap.add_argument('--thresh', type=int, default=8, help='Grayscale threshold to treat as non-black (default: 8)')
    ap.add_argument('--min-row-frac', type=float, default=0.02, help='Fraction of rows that must be non-black for a column to be kept (default: 0.02)')
    ap.add_argument('--min-col-frac', type=float, default=0.02, help='Fraction of columns that must be non-black for a row to be kept (default: 0.02)')
    ap.add_argument('--inplace', action='store_true', help='Overwrite images in place (default: write *_trim.jpg copies)')
    ap.add_argument('--corners-dir', type=str, default=None, help='Directory containing corner annotation files (default: alongside images)')
    ap.add_argument('--corners-out-dir', type=str, default=None, help='Directory to write updated corner files (default: same as corners-dir)')
    ap.add_argument('--out-dir', type=str, default=None, help='Directory to write trimmed images (default: alongside images)')
    ap.add_argument('--vis-dir', type=str, default=None, help='Directory to write corner visualization images')
    ap.add_argument('--draw-radius', type=int, default=6, help='Radius for visualization dots (default: 6)')
    ap.add_argument('--draw-thickness', type=int, default=1, help='Text thickness for visualization (default: 1)')
    args = ap.parse_args()

    corners_dir = Path(args.corners_dir) if args.corners_dir else None
    corners_out_dir = Path(args.corners_out_dir) if args.corners_out_dir else None
    out_dir = Path(args.out_dir) if args.out_dir else None
    vis_dir = Path(args.vis_dir) if args.vis_dir else None

    total = 0
    total_vis = 0
    for d in args.dirs:
        root = Path(d)
        if not root.exists():
            print('Directory not found:', root)
            continue
        pattern = f'*{args.suffix}' if args.suffix.startswith('.') else f'*{args.suffix}'
        for img_path in sorted(root.glob(pattern)):
            out_path, vis_path = process_image(img_path,
                                               args.corners_suffix,
                                               args.inplace,
                                               args.thresh,
                                               args.min_row_frac,
                                               args.min_col_frac,
                                               corners_dir,
                                               corners_out_dir,
                                               out_dir,
                                               vis_dir,
                                               args.draw_radius,
                                               args.draw_thickness)
            if out_path is not None:
                total += 1
            if vis_path is not None:
                total_vis += 1

    msg = f'Done. Trimmed {total} image(s).'
    if vis_dir is not None:
        msg += f' Wrote {total_vis} visualization image(s).'
    print(msg)


if __name__ == '__main__':
    main()
