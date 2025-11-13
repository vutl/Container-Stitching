#!/usr/bin/env python3
"""
Align frames to a common upright rectangle while:
- preserving each frame's side background widths (left/right) from its corners,
- enforcing a uniform canvas size for all outputs,
- keeping exactly N pixels background above and below the container (default 20/20).

Inputs per frame i:
  image: <dir>/<i><img_suffix>
  corners: <corners_dir>/<i><corners_suffix> with lines: LABEL x y ... for TL TR BR BL

Outputs:
  <out>/<i>_aligned.jpg
  <out>/<i>_aligned_corners.txt (warped TL/TR/BR/BL)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import argparse
import cv2
import numpy as np


CornerMap = Dict[str, Tuple[float, float]]


def parse_indices(spec: str) -> List[int]:
    parts = spec.replace(',', ' ').split()
    out: List[int] = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
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
        q = parts[0]
        try:
            x = float(parts[1]); y = float(parts[2])
        except Exception:
            continue
        pts[q] = (x, y)
    return pts


@dataclass
class Frame:
    idx: int
    img: np.ndarray
    corners: CornerMap
    w: int
    h: int
    left_margin: int
    right_margin: int
    cont_h: float


def load_frames(img_dir: Path, corners_dir: Path, img_suffix: str, corners_suffix: str,
                indices: Iterable[int], img_prefix: str = '') -> List[Frame]:
    frames: List[Frame] = []
    for i in indices:
        ip = img_dir / f"{img_prefix}{i}{img_suffix}"
        cp = corners_dir / f"{i}{corners_suffix}"
        if not ip.exists() or not cp.exists():
            print(f"skip {i} (missing)")
            continue
        img = cv2.imread(str(ip))
        if img is None:
            print(f"skip {i} (read fail)")
            continue
        corners = read_corners(cp)
        if not all(k in corners for k in ('TL', 'TR', 'BR', 'BL')):
            print(f"skip {i} (incomplete corners)")
            continue
        h, w = img.shape[:2]
        left = int(round(min(corners['TL'][0], corners['BL'][0])))
        right = int(round((w - 1) - max(corners['TR'][0], corners['BR'][0])))
        top_y = min(corners['TL'][1], corners['TR'][1])
        bot_y = max(corners['BL'][1], corners['BR'][1])
        cont_h = float(bot_y - top_y)
        frames.append(Frame(i, img, corners, w, h, max(left, 0), max(right, 0), cont_h))
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True)
    ap.add_argument('--indices', required=True)
    ap.add_argument('--img-suffix', default='_cropped_padded.jpg')
    ap.add_argument('--img-prefix', default='', help='Prefix for image filenames (e.g., "img_" for img_0.jpg)')
    ap.add_argument('--corners-dir', default='out_annot')
    ap.add_argument('--corners-suffix', default='_corners.txt')
    ap.add_argument('--out', default='aligned')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--bottom', type=int, default=20)
    args = ap.parse_args()

    img_dir = Path(args.dir)
    corners_dir = Path(args.corners_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = parse_indices(args.indices)
    frames = load_frames(img_dir, corners_dir, args.img_suffix, args.corners_suffix, idxs, args.img_prefix)
    if not frames:
        print('No frames matched inputs; nothing to do.')
        return

    # Uniform canvas
    target_w = max(f.w for f in frames)
    cont_h_med = int(round(float(np.median([f.cont_h for f in frames]))))
    target_h = cont_h_med + args.top + args.bottom

    print(f"Canvas: {target_w}x{target_h} (container_h_med={cont_h_med}, top={args.top}, bottom={args.bottom})")

    for f in frames:
        # Preserve side background widths from this frame
        dst_left = int(np.clip(f.left_margin, 0, target_w - 2))
        dst_right = int(np.clip(target_w - 1 - f.right_margin, dst_left + 1, target_w - 1))
        # Exact background counts above/below the container region
        dst_top = int(args.top)
        dst_bottom = int(target_h - 1 - args.bottom)

        dst = np.array([
            (float(dst_left), float(dst_top)),
            (float(dst_right), float(dst_top)),
            (float(dst_right), float(dst_bottom)),
            (float(dst_left), float(dst_bottom)),
        ], dtype=np.float32)

        src = np.array([
            f.corners['TL'], f.corners['TR'], f.corners['BR'], f.corners['BL']
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            f.img, H, (target_w, target_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )

        # Do not force black bands; keep whatever background the homography brings

        out_img = out_dir / f"{f.idx}_aligned.jpg"
        cv2.imwrite(str(out_img), warped)

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        with open(out_dir / f"{f.idx}_aligned_corners.txt", 'w') as g:
            for name, (x, y) in zip(['TL', 'TR', 'BR', 'BL'], proj):
                g.write(f"{name} {int(round(float(x)))} {int(round(float(y)))}\n")
        print(f"Wrote: {out_img}  left={f.left_margin}px right={f.right_margin}px")

    print('Done.')


if __name__ == '__main__':
    main()
