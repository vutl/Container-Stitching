#!/usr/bin/env python3
"""
Refine container corner points from bounding boxes using gradient-based heuristics.
Reads bounding boxes from files like `out_annot/<idx>_annot_cropped.txt` and writes
corner points to `out_annot/<idx>_corners.txt` and annotated images `*_corners.jpg`.

Usage:
    python3 test2.py --dir mnt/2025-09-04_14-35-53 --indices 42 50 60 70 74

Algorithm summary (per bbox):
 - Crop bbox region, compute Sobel gx, gy (absolute).
 - Search for strong vertical edge peaks inside a left/right margin and within a
   central row band (so corner is near the box center along orthogonal axis).
 - Compute dynamic thresholds from local region statistics (percentile + median/std).
 - If vertical and horizontal peaks found -> intersection is corner (optionally refined
   by weighted centroid inside small window).
 - If only one axis found -> combine with bbox center or snapped top/bottom derived from
   global estimates.
 - If nothing found -> fallback to bbox corner coordinates.

Outputs (per image):
 - `out_annot/<idx>_corners.txt` with lines: LABEL x y score method box_x1 box_y1 box_x2 box_y2
 - `out_annot/<idx>_corners.jpg` annotated image (points and labels)

"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from corner_refinement import CornerRefinerParams, refine_corners


# ----------------- helpers -----------------


def read_boxes_from_txt(txt_path: Path):
    boxes = []
    if not txt_path.exists():
        return boxes
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        label = parts[0]
        try:
            x1 = int(parts[1]); y1 = int(parts[2]); x2 = int(parts[3]); y2 = int(parts[4])
        except Exception:
            continue
        score = float(parts[5]) if len(parts) >= 6 else None
        boxes.append((label, (x1, y1, x2, y2), score))
    return boxes


# ----------------- script entry -----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=str, default='mnt/2025-09-04_14-35-53')
    ap.add_argument('--indices', type=int, nargs='+', default=[42, 50, 60, 70, 74])
    ap.add_argument('--bbox-dir', type=str, default='out_annot')
    ap.add_argument('--bbox-suffix', type=str, default='_annot_cropped_padded_jpg.txt')
    ap.add_argument('--img-suffix', type=str, default='_cropped_padded.jpg')
    ap.add_argument('--out', type=str, default='out_annot')

    ap.add_argument('--side-margin-frac', type=float, default=0.12)
    ap.add_argument('--center-window-frac', type=float, default=0.45)
    ap.add_argument('--center-near-px', type=int, default=4,
                    help='radius in px around bbox center to search before margins')
    ap.add_argument('--center-thr-alpha', type=float, default=0.5,
                    help='alpha multiplier for center threshold relative to dynamic threshold')
    ap.add_argument('--min-side-px', type=int, default=6)
    ap.add_argument('--sobel-ksize', type=int, default=3)
    ap.add_argument('--perc', type=float, default=90.0)
    ap.add_argument('--perc-alpha', type=float, default=0.45)
    ap.add_argument('--std-mult', type=float, default=1.0)
    ap.add_argument('--refine-half-w', type=int, default=6)
    ap.add_argument('--refine-half-h', type=int, default=6)
    ap.add_argument('--refine-search-h', type=int, default=8)
    ap.add_argument('--refine-search-w', type=int, default=8)
    ap.add_argument('--min-accept-score', type=float, default=10.0)
    ap.add_argument('--container-head-area-thresh', type=float, default=2000.0,
                    help='Area threshold to classify box as container head (use bbox corner directly)')

    args = ap.parse_args()

    img_dir = Path(args.dir)
    bbox_dir = Path(args.bbox_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = CornerRefinerParams(
        side_margin_frac=args.side_margin_frac,
        center_window_frac=args.center_window_frac,
        center_near_px=args.center_near_px,
        center_thr_alpha=args.center_thr_alpha,
        min_side_px=args.min_side_px,
        sobel_ksize=args.sobel_ksize,
        perc=args.perc,
        perc_alpha=args.perc_alpha,
        std_mult=args.std_mult,
        refine_half_w=args.refine_half_w,
        refine_half_h=args.refine_half_h,
        refine_search_h=args.refine_search_h,
        refine_search_w=args.refine_search_w,
        min_accept_score=args.min_accept_score,
        container_head_area_thresh=args.container_head_area_thresh,
    )

    for i in args.indices:
        img_path = img_dir / f"{i}{args.img_suffix}"
        bbox_path = bbox_dir / f"{i}{args.bbox_suffix}"
        if not img_path.exists():
            print(f"Missing image: {img_path}")
            continue
        if not bbox_path.exists():
            print(f"Missing bbox file (skipping): {bbox_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed reading: {img_path}")
            continue

        boxes = read_boxes_from_txt(bbox_path)
        if not boxes:
            print(f"No boxes found in {bbox_path}; skipping")
            continue

        annotated = img.copy()
        candidates = refine_corners(img, [b for _, b, _ in boxes], params)

        out_txt = out_dir / f"{i}_corners.txt"
        with open(out_txt, 'w') as f:
            for cand, (_, box, _) in zip(candidates, boxes):
                x1, y1, x2, y2 = box
                cx, cy = cand.point
                f.write(f"{cand.quadrant} {cx} {cy} {cand.confidence:.4f} {cand.method} {x1} {y1} {x2} {y2}\n")
                cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)
                txt = f"{cand.quadrant} {cand.confidence:.2f} {cand.method}"
                cv2.putText(annotated, txt, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        out_img = out_dir / f"{i}_corners.jpg"
        cv2.imwrite(str(out_img), annotated)
        print(f"Wrote corners: {out_txt}  and annotated image: {out_img}")

    print("Done.")


if __name__ == '__main__':
    main()
