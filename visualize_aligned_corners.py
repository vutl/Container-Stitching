#!/usr/bin/env python3
"""
Draw only per-image aligned corners on the aligned images (no reference overlay).

For each aligned image i:
 - Prefer reading aligned/<i>_aligned_corners.txt (saved during warp_align).
 - If missing, compute the 4-point mapping from out_annot/<i>_corners.txt to
   the reference corners, transform the points, and draw them (magenta).
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def read_corners(path):
    """Read TL/TR/BR/BL from _corners.txt -> dict {Q: (x,y)}"""
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
        q = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        pts[q] = (int(round(x)), int(round(y)))
    return pts


def parse_range(s):
    parts = s.replace(',', ' ').split()
    out = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned-dir', default='aligned')
    ap.add_argument('--corners-dir', default='out_annot')
    ap.add_argument('--indices', required=True)
    ap.add_argument('--ref-index', type=int, default=43)
    ap.add_argument('--out', default='aligned_viz')
    args = ap.parse_args()

    idxs = parse_range(args.indices)
    aligned_dir = Path(args.aligned_dir)
    corners_dir = Path(args.corners_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reference corners are needed only for fallback mapping; do not draw them
    ref_corners = read_corners(corners_dir / f"{args.ref_index}_corners.txt")

    for i in idxs:
        img_path = aligned_dir / f"{i}_aligned.jpg"
        if not img_path.exists():
            print(f"Missing: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read: {img_path}")
            continue

        # Draw reference corners on this aligned image
        viz = img.copy()
        
        # Prefer using saved per-image aligned corners from warp_align if available
        aligned_corners_path = aligned_dir / f"{i}_aligned_corners.txt"
        this_aligned = read_corners(aligned_corners_path)
        if all(k in this_aligned for k in ('TL','TR','BR','BL')):
            poly_i = np.array([this_aligned[q] for q in ['TL','TR','BR','BL']], dtype=np.int32)
            cv2.polylines(viz, [poly_i.reshape(-1,1,2)], True, (255, 0, 255), 2)
            for q, pt in this_aligned.items():
                cv2.circle(viz, pt, 5, (255, 0, 255), -1)
                cv2.putText(viz, f"{q}", (pt[0]+6, pt[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
        else:
            # Fallback: compute on the fly via 4-pt mapping from original corners
            this_corners = read_corners(corners_dir / f"{i}_corners.txt")
            if all(k in this_corners for k in ('TL','TR','BR','BL')):
                if not all(k in ref_corners for k in ('TL','TR','BR','BL')):
                    cv2.putText(viz, "missing ref corners for fallback", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
                src_pts = np.array([this_corners[k] for k in ['TL','TR','BR','BL']], dtype=np.float32)
                dst_pts = np.array([ref_corners[k] for k in ['TL','TR','BR','BL']], dtype=np.float32)
                try:
                    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    src_pts_h = src_pts.reshape(-1,1,2)
                    proj = cv2.perspectiveTransform(src_pts_h, H).reshape(-1,2)
                    proj_int = np.round(proj).astype(int)
                    cv2.polylines(viz, [proj_int.reshape(-1,1,2)], True, (255, 0, 255), 2)
                    for (x,y), q in zip(proj_int, ['TL','TR','BR','BL']):
                        cv2.circle(viz, (int(x),int(y)), 5, (255, 0, 255), -1)
                        cv2.putText(viz, f"{q}", (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1, cv2.LINE_AA)
                except Exception as e:
                    cv2.putText(viz, f"warp_failed: {e}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
            else:
                cv2.putText(viz, "missing TL/TR/BR/BL for this frame", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

        out_path = out_dir / f"{i}_aligned_viz.jpg"
        cv2.imwrite(str(out_path), viz)
        print(f"Wrote: {out_path}")

    print("\nDone. Check aligned_viz/ for visualization.")


if __name__ == '__main__':
    main()
