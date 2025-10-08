#!/usr/bin/env python3
"""
Perspective-rectify aligned container images to perfect rectangles using their
four-corner annotations.

Example:
  python3 rectify_to_rectangle.py \
      --dir aligned_cropped \
      --indices 42-74 \
      --image-suffix _aligned.jpg \
      --corners-suffix _aligned_corners.txt \
      --out aligned_rectified \
      --write-mask

For each input frame i, reads <dir>/<i><image-suffix> and the corresponding
corner file, computes a perspective warp that maps TL/TR/BR/BL to a rectangle
whose height equals the length of the right edge (TR->BR) and whose width is
the maximum of the top and bottom edge lengths. The rectified image and updated
corners (axis-aligned rectangle) are written to the output directory.
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import cv2
import numpy as np


def parse_indices(spec: str) -> List[int]:
    parts = spec.replace(',', ' ').split()
    out: List[int] = []
    for token in parts:
        if '-' in token:
            a, b = token.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(token))
    return sorted(set(out))


def read_corners(path: Path) -> Dict[str, Tuple[float, float]]:
    pts: Dict[str, Tuple[float, float]] = {}
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
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        pts[key] = (x, y)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Input directory containing aligned images')
    ap.add_argument('--indices', required=True, help='Frame indices (e.g. "42-74" or "42 43 44")')
    ap.add_argument('--image-suffix', default='_aligned.jpg')
    ap.add_argument('--corners-suffix', default='_aligned_corners.txt')
    ap.add_argument('--out', default='aligned_rectified')
    ap.add_argument('--vertical-margin', type=int, default=20,
                    help='Number of extra pixels to keep above and below the container (default: 20)')
    ap.add_argument('--write-mask', action='store_true', help='Also write *_mask.png for each rectified frame')
    args = ap.parse_args()

    in_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = parse_indices(args.indices)
    if not idxs:
        print('No indices parsed')
        return

    margin = max(0, int(round(args.vertical_margin)))

    for idx in idxs:
        img_path = in_dir / f"{idx}{args.image_suffix}"
        corners_path = in_dir / f"{idx}{args.corners_suffix}"
        if not img_path.exists():
            print(f'Missing image: {img_path}')
            continue
        corners = read_corners(corners_path)
        if not all(k in corners for k in ('TL', 'TR', 'BR', 'BL')):
            print(f'Skipping {idx}: incomplete corners in {corners_path}')
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read {img_path}')
            continue

        tl = np.array(corners['TL'], dtype=np.float32)
        tr = np.array(corners['TR'], dtype=np.float32)
        br = np.array(corners['BR'], dtype=np.float32)
        bl = np.array(corners['BL'], dtype=np.float32)

        src = np.stack([tl, tr, br, bl], axis=0)
        h_in, w_in = img.shape[:2]
        height_right = np.linalg.norm(tr - br)
        height_left = np.linalg.norm(tl - bl)
        target_h_core = max(1, int(round(max(height_right, height_left))))

        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        target_w = max(1, int(round(max(width_top, width_bottom))))

        left_pad = max(0, int(np.floor(min(tl[0], bl[0]))))
        right_pad = max(0, int(np.ceil((w_in - 1) - max(tr[0], br[0]))))
        output_w = target_w + left_pad + right_pad

        dst = np.array(
            [
                [float(left_pad), float(margin)],
                [float(left_pad + target_w - 1), float(margin)],
                [float(left_pad + target_w - 1), float(margin + target_h_core - 1)],
                [float(left_pad), float(margin + target_h_core - 1)],
            ],
            dtype=np.float32,
        )

        output_h = target_h_core + 2 * margin

        M = cv2.getPerspectiveTransform(src, dst)
        rectified = cv2.warpPerspective(
            img,
            M,
            (output_w, output_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        out_img = out_dir / f"{idx}_aligned.jpg"
        cv2.imwrite(str(out_img), rectified)

        out_corners = out_dir / f"{idx}_aligned_corners.txt"
        with open(out_corners, 'w') as f:
            f.write(f"TL {left_pad} {margin}\n")
            f.write(f"TR {left_pad + target_w - 1} {margin}\n")
            f.write(f"BR {left_pad + target_w - 1} {margin + target_h_core - 1}\n")
            f.write(f"BL {left_pad} {margin + target_h_core - 1}\n")

        if args.write_mask:
            mask = np.zeros((output_h, output_w), dtype=np.uint8)
            mask[margin:margin + target_h_core, left_pad:left_pad + target_w] = 255
            cv2.imwrite(str(out_dir / f"{idx}_aligned_mask.png"), mask)

        print(f"Rectified {idx}: {img_path.name} -> {out_img.name} ({output_w}x{output_h}, margin={margin}, left_pad={left_pad}, right_pad={right_pad})")

    print('Done.')


if __name__ == '__main__':
    main()
