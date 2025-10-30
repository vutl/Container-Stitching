#!/usr/bin/env python3
"""Split wrapper: trim black borders for two segments using separate corners.

This wraps trim_black_update_corners.process_image to run on two independent
segments (e.g., out_annot_1_full and out_annot_2_full) from the same image
directory. It writes trimmed images and updated corners to separate outputs.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2  # only to validate readable images quickly

import trim_black_update_corners as base


def _iter_indices(img_dir: Path, suffix: str):
    for p in sorted(img_dir.glob(f"*{suffix}")):
        name = p.name
        stem = name[:-len(suffix)] if name.endswith(suffix) else p.stem
        if stem.isdigit():
            yield int(stem), p
        else:
            yield stem, p


def _process_segment(img_dir: Path,
                     corners_dir: Path,
                     out_img_dir: Optional[Path],
                     corners_suffix: str,
                     corners_out_dir: Optional[Path],
                     vis_dir: Optional[Path],
                     suffix: str,
                     thresh: int,
                     min_row_frac: float,
                     min_col_frac: float,
                     draw_radius: int,
                     draw_thickness: int) -> Tuple[int, int]:
    done = 0
    vis = 0
    for idx, img_path in _iter_indices(img_dir, suffix):
        corners_path = corners_dir / f"{idx}{corners_suffix}"
        if not corners_path.exists():
            continue
        # quick image check to avoid work on unreadable files
        if cv2.imread(str(img_path)) is None:
            print("Skip unreadable:", img_path)
            continue
        out_path, vis_path = base.process_image(
            img_path=img_path,
            corners_suffix=corners_suffix,
            inplace=False,
            thresh=thresh,
            min_row_frac=min_row_frac,
            min_col_frac=min_col_frac,
            corners_dir=corners_dir,
            corners_out_dir=corners_out_dir,
            out_dir=out_img_dir,
            vis_dir=vis_dir,
            draw_radius=draw_radius,
            draw_thickness=draw_thickness,
        )
        if out_path is not None:
            done += 1
        if vis_path is not None:
            vis += 1
    return done, vis


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--img-dir', required=True, help='Directory with source images (e.g., img2)')
    ap.add_argument('--suffix', default='_no_warp.jpg', help='Image filename suffix to match (default: _no_warp.jpg)')
    ap.add_argument('--corners-suffix', default='_corners.txt', help='Corners filename suffix (default: _corners.txt)')

    ap.add_argument('--corners1-dir', required=True, help='Corners directory for segment 1 (e.g., out_annot_1_full)')
    ap.add_argument('--corners2-dir', required=False, help='Corners directory for segment 2 (optional for single-container)')
    ap.add_argument('--out1', default='out_trimmed_sides_1', help='Output directory for trimmed images of segment 1')
    ap.add_argument('--out2', default='out_trimmed_sides_2', help='Output directory for trimmed images of segment 2 (optional)')
    ap.add_argument('--corners1-out', default='out_trimmed_sides_corners_1', help='Output directory for updated corners of segment 1')
    ap.add_argument('--corners2-out', default='out_trimmed_sides_corners_2', help='Output directory for updated corners of segment 2 (optional)')
    ap.add_argument('--vis1', default=None, help='Optional visualization output directory for segment 1')
    ap.add_argument('--vis2', default=None, help='Optional visualization output directory for segment 2')

    ap.add_argument('--thresh', type=int, default=8)
    ap.add_argument('--min-row-frac', type=float, default=0.02)
    ap.add_argument('--min-col-frac', type=float, default=0.02)
    ap.add_argument('--draw-radius', type=int, default=6)
    ap.add_argument('--draw-thickness', type=int, default=1)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    corners1_dir = Path(args.corners1_dir)
    corners2_dir = Path(args.corners2_dir) if args.corners2_dir else None
    out1 = Path(args.out1); out1.mkdir(parents=True, exist_ok=True)
    out2 = Path(args.out2) if args.out2 else None
    if out2:
        out2.mkdir(parents=True, exist_ok=True)
    c1o = Path(args.corners1_out); c1o.mkdir(parents=True, exist_ok=True)
    c2o = Path(args.corners2_out) if args.corners2_out else None
    if c2o:
        c2o.mkdir(parents=True, exist_ok=True)
    vis1 = Path(args.vis1) if args.vis1 else None
    vis2 = Path(args.vis2) if args.vis2 else None

    n1, v1 = _process_segment(
        img_dir, corners1_dir, out1, args.corners_suffix, c1o, vis1,
        args.suffix, args.thresh, args.min_row_frac, args.min_col_frac,
        args.draw_radius, args.draw_thickness,
    )
    
    if corners2_dir:
        n2, v2 = _process_segment(
            img_dir, corners2_dir, out2, args.corners_suffix, c2o, vis2,
            args.suffix, args.thresh, args.min_row_frac, args.min_col_frac,
            args.draw_radius, args.draw_thickness,
        )
        msg = f"Done. Trimmed seg1={n1} (vis={v1}), seg2={n2} (vis={v2})."
    else:
        msg = f"Done. Trimmed seg1={n1} (vis={v1}) [single-container, seg2 skipped]"
    
    print(msg)


if __name__ == '__main__':
    main()
