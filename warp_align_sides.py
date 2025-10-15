#!/usr/bin/env python3
"""
Side-face alignment wrapper — copy of `warp_align.py` with defaults tuned for
side images processed by the side pipeline. Uses trimmed images and the
corresponding updated corner files produced by the trimming step.
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
                indices: Iterable[int]) -> List[Frame]:
    frames: List[Frame] = []
    for i in indices:
        ip = img_dir / f"{i}{img_suffix}"
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
    ap.add_argument('--dir', required=True, help='Directory with trimmed images (e.g. out_trimmed_sides)')
    ap.add_argument('--indices', required=True)
    ap.add_argument('--img-suffix', default='_no_warp_trim.jpg')
    ap.add_argument('--corners-dir', default='out_trimmed_sides_corners')
    ap.add_argument('--corners-suffix', default='_corners.txt')
    ap.add_argument('--out', default='aligned_sides')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--bottom', type=int, default=20)
    ap.add_argument('--center-container', action='store_true', help='Center container horizontally on canvas using median container width')
    args = ap.parse_args()

    img_dir = Path(args.dir)
    corners_dir = Path(args.corners_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = parse_indices(args.indices)
    frames = load_frames(img_dir, corners_dir, args.img_suffix, args.corners_suffix, idxs)
    if not frames:
        print('No frames matched inputs; nothing to do.')
        return

    # Uniform canvas
    target_w = max(f.w for f in frames)
    cont_h_med = int(round(float(np.median([f.cont_h for f in frames]))))
    target_h = cont_h_med + args.top + args.bottom

    center_container = bool(args.center_container)
    if center_container:
        # compute median container width from corners (TR.x - TL.x)
        cont_ws = []
        for f in frames:
            src = f.corners
            try:
                w = float(src['TR'][0]) - float(src['TL'][0])
                if w > 0:
                    cont_ws.append(w)
            except Exception:
                continue
        cont_w_med = int(round(float(np.median(cont_ws))) ) if cont_ws else None
        if cont_w_med is not None:
            print(f"Centering container horizontally using median container width = {cont_w_med}")

    print(f"Canvas: {target_w}x{target_h} (container_h_med={cont_h_med}, top={args.top}, bottom={args.bottom})")

    for f in frames:
        # Preserve side background widths from this frame (or center if requested)
        if center_container and cont_w_med is not None:
            # place centered container of width cont_w_med
            dst_left = int(round(max(0, (target_w - cont_w_med) / 2.0)))
            dst_right = int(round(min(target_w - 1, dst_left + cont_w_med - 1)))
            # ensure within bounds
            dst_left = int(np.clip(dst_left, 0, target_w - 2))
            dst_right = int(np.clip(dst_right, dst_left + 1, target_w - 1))
        else:
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
