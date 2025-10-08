#!/usr/bin/env python3
"""
Generate rectangular masks from saved 4-corner files for images in a folder.

For each image matching <stem><image_suffix> this script looks for a
corner file named <stem><corners_suffix> containing lines like:
  TL x y
  TR x y
  BR x y
  BL x y

It produces a mask image (uint8 PNG) where the rectangle covering the four
corners is filled with 255 and everything else is 0. Masks are saved using
the output suffix (default: _mask.png) next to the image.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


def read_corners(path: Path) -> Dict[str, Tuple[float, float]]:
    pts = {}
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


def mask_from_corners(shape: Tuple[int, int], corners: Dict[str, Tuple[float, float]], pad: int = 0) -> np.ndarray:
    h, w = shape[:2]
    keys = ('TL', 'TR', 'BR', 'BL')
    if not all(k in corners for k in keys):
        # fallback: if any corner missing, return empty mask
        return np.zeros((h, w), dtype=np.uint8)
    xs = [corners[k][0] for k in keys]
    ys = [corners[k][1] for k in keys]
    x0 = max(0, int(np.floor(min(xs))) - pad)
    x1 = min(w, int(np.ceil(max(xs))) + pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    y1 = min(h, int(np.ceil(max(ys))) + pad)
    if x1 <= x0 or y1 <= y0:
        return np.zeros((h, w), dtype=np.uint8)
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0:y1, x0:x1] = 255
    return m


def main():
    p = argparse.ArgumentParser(description='Generate rectangular masks from corner files')
    p.add_argument('--dir', default='aligned_cropped', help='Directory containing images and corner files')
    p.add_argument('--image-suffix', default='_aligned.jpg', help='Suffix of image files to process (default: _aligned.jpg)')
    p.add_argument('--corners-suffix', default='_aligned_corners.txt', help='Suffix of corner files (default: _aligned_corners.txt)')
    p.add_argument('--out-suffix', default='_mask.png', help='Suffix for generated mask files (default: _mask.png)')
    p.add_argument('--pad', type=int, default=0, help='Pixels to pad the rectangle on all sides')
    p.add_argument('--verbose', action='store_true')
    args = p.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print('Directory not found:', root)
        return

    imgs = sorted(root.glob(f'*{args.image_suffix}'))
    if not imgs:
        print('No images found with suffix', args.image_suffix, 'in', root)
        return

    count = 0
    for img_path in imgs:
        stem = img_path.stem
        # Robustly find a corners file: try exact stem + corners_suffix, then glob for '*corners*.txt',
        # then try removing common image suffix parts like '_aligned'
        candidates = []
        candidates.append(root / f"{stem}{args.corners_suffix}")
        # glob for any file containing 'corner' or 'corners' near the stem
        candidates.extend(sorted(root.glob(f"{stem}*corners*.txt")))
        # try removing trailing known parts
        if stem.endswith('_aligned'):
            base = stem[:-8]
            candidates.append(root / f"{base}{args.corners_suffix}")
            candidates.extend(sorted(root.glob(f"{base}*corners*.txt")))

        corners_path = None
        for c in candidates:
            if c and Path(c).exists():
                corners_path = Path(c)
                break
        corners = read_corners(corners_path) if corners_path is not None else {}
        img = cv2.imread(str(img_path))
        if img is None:
            if args.verbose:
                print('Failed to read image', img_path)
            continue
        mask = mask_from_corners(img.shape[:2], corners, pad=args.pad)
        out_name = root / f"{stem}{args.out_suffix}"
        cv2.imwrite(str(out_name), mask)
        count += 1
        if args.verbose:
            print('Wrote', out_name, 'corners_path=', str(corners_path) if corners_path is not None else 'None', 'corners_exist=', bool(corners))

    print(f'Done: wrote {count} masks to {root} (suffix={args.out_suffix})')


if __name__ == '__main__':
    main()
