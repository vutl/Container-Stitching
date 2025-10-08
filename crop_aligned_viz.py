#!/usr/bin/env python3
"""
Crop each image in aligned_viz/ to the bounding box of its container (from aligned/<idx>_aligned_corners.txt),
extending 20px above and below the box, and save to aligned_viz_cropped/.
"""
import cv2
import numpy as np
from pathlib import Path

def read_corners(path):
    corners = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                corners[parts[0]] = (float(parts[1]), float(parts[2]))
    return corners

def main():
    in_dir = Path('aligned_viz')
    corners_dir = Path('aligned')
    out_dir = Path('aligned_viz_cropped')
    out_dir.mkdir(exist_ok=True)
    for img_path in sorted(in_dir.glob('*_aligned_viz.jpg')):
        idx = img_path.name.split('_')[0]
        corners_path = corners_dir / f'{idx}_aligned_corners.txt'
        if not corners_path.exists():
            print(f'Missing corners for {img_path.name}')
            continue
        corners = read_corners(corners_path)
        if not all(k in corners for k in ['TL','TR','BR','BL']):
            print(f'Incomplete corners for {img_path.name}')
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read {img_path}')
            continue
        pts = np.array([corners[k] for k in ['TL','TR','BR','BL']])
        x0 = 0
        x1 = img.shape[1]
        y0 = int(np.floor(pts[:,1].min()))
        y1 = int(np.ceil(pts[:,1].max()))
        # extend 20px above and below
        y0 = max(0, y0 - 20)
        y1 = min(img.shape[0], y1 + 20)
        crop = img[y0:y1, x0:x1]
        out_path = out_dir / img_path.name
        cv2.imwrite(str(out_path), crop)
        print(f'Cropped {img_path.name} -> {out_path}')

if __name__ == '__main__':
    main()
