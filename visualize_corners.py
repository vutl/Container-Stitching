#!/usr/bin/env python3
"""Draw corner points from corner files onto images and save to a vis folder.

Usage:
  python3 visualize_corners.py --dir <input_dir> --indices 210-261 --image-suffix _aligned.jpg \
    --corners-suffix _aligned_corners.txt --out-vis-dir aligned_rectified_sides_vis

The corner file is expected to contain four lines with 'x y' per line or a single line with 8 numbers.
"""
import argparse
import os
import re
import cv2
import glob
import numpy as np


def read_corners(path):
    """Read corner coordinates from a text file.

    Accepts files with numeric tokens possibly mixed with labels (e.g. 'TL 123 456').
    Returns list of (x,y) tuples or None if not enough numbers found.
    """
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        txt = f.read()
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+", txt)
    if not nums:
        return None
    vals = [float(x) for x in nums]
    if len(vals) < 8:
        return None
    vals = vals[:8]
    pts = [(vals[i], vals[i+1]) for i in range(0, 8, 2)]
    return pts


def draw_corners(img, corners, color=(0, 255, 0)):
    h, w = img.shape[:2]
    out = img.copy()
    for i, (x, y) in enumerate(corners):
        xi = int(round(x))
        yi = int(round(y))
        if 0 <= xi < w and 0 <= yi < h:
            cv2.circle(out, (xi, yi), 6, color, -1)
            cv2.putText(out, str(i+1), (xi+6, yi-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        else:
            # draw small cross at clamped position
            xc = min(max(xi, 0), w-1)
            yc = min(max(yi, 0), h-1)
            cv2.drawMarker(out, (xc, yc), (0,0,255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2)
    return out


def collect_indices(indices_arg):
    if not indices_arg:
        return []
    parts = indices_arg.split(',')
    inds = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-')
            inds.extend(list(range(int(a), int(b)+1)))
        else:
            inds.append(int(p))
    return sorted(set(inds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Input directory containing images')
    p.add_argument('--indices', default=None, help='Comma or dash separated indices, e.g. 210-261 or 210,212')
    p.add_argument('--image-suffix', default='_aligned.jpg')
    p.add_argument('--corners-suffix', default='_aligned_corners.txt')
    p.add_argument('--out-vis-dir', default=None, help='Where to write visualizations. If omitted, uses <dir>_vis')
    args = p.parse_args()

    inds = collect_indices(args.indices) if args.indices else None
    in_dir = args.dir
    out_vis = args.out_vis_dir or (in_dir.rstrip('/\\') + '_vis')
    os.makedirs(out_vis, exist_ok=True)

    if inds:
        files = [os.path.join(in_dir, f"{i}{args.image_suffix}") for i in inds]
    else:
        files = sorted(glob.glob(os.path.join(in_dir, f"*{args.image_suffix}")))

    processed = 0
    for f in files:
        if not os.path.exists(f):
            # try f as path with no dir
            continue
        base = os.path.basename(f)
        # Determine root name by removing the image_suffix if present, else use the stem
        suffix = args.image_suffix
        if base.endswith(suffix):
            root = base[:-len(suffix)]
        else:
            root = os.path.splitext(base)[0]
        corners_path = os.path.join(in_dir, root + args.corners_suffix)
        corners = read_corners(corners_path)
        img = cv2.imread(f)
        if img is None:
            print(f"Failed to read image: {f}")
            continue
        if corners is None:
            print(f"No corners for {f}, writing original to vis folder")
            out = img
        else:
            out = draw_corners(img, corners)
        out_path = os.path.join(out_vis, base)
        cv2.imwrite(out_path, out)
        processed += 1

    print(f"Wrote {processed} visualization(s) to {out_vis}")


if __name__ == '__main__':
    main()
