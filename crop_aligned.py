#!/usr/bin/env python3
"""
Crop each aligned image (without drawn bounding boxes) to the container bounding box
with a 20px margin above and below, and save the updated bounding box coordinates.
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

def write_corners(path, corners):
    with open(path, 'w') as f:
        for key in ['TL', 'TR', 'BR', 'BL']:
            x, y = corners[key]
            f.write(f"{key} {int(round(x))} {int(round(y))}\n")


def main():
    in_dir = Path('aligned')
    out_dir = Path('aligned_cropped')
    out_dir.mkdir(exist_ok=True)

    for img_path in sorted(in_dir.glob('*_aligned.jpg')):
        idx = img_path.stem.split('_')[0]
        corners_path = in_dir / f'{idx}_aligned_corners.txt'
        if not corners_path.exists():
            print(f'Missing corners for {img_path.name}')
            continue

        corners = read_corners(corners_path)
        if not all(k in corners for k in ['TL', 'TR', 'BR', 'BL']):
            print(f'Incomplete corners for {img_path.name}')
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f'Failed to read {img_path}')
            continue

        pts = np.array([corners[k] for k in ['TL', 'TR', 'BR', 'BL']], dtype=np.float32)
        x0 = 0
        x1 = img.shape[1]
        y0 = int(np.floor(pts[:, 1].min()))
        y1 = int(np.ceil(pts[:, 1].max()))

        margin = 20
        y0 = max(0, y0 - margin)
        y1 = min(img.shape[0], y1 + margin)

        if y1 <= y0 or x1 <= x0:
            print(f'Skipping {img_path.name}: invalid crop bounds')
            continue

        crop = img[y0:y1, x0:x1]
        out_img_path = out_dir / img_path.name
        cv2.imwrite(str(out_img_path), crop)

        adjusted = {k: (corners[k][0] - x0, corners[k][1] - y0) for k in corners}
        out_corners_path = out_dir / f'{idx}_aligned_corners.txt'
        write_corners(out_corners_path, adjusted)
        print(f'Cropped {img_path.name} -> {out_img_path} (corners -> {out_corners_path})')


if __name__ == '__main__':
    main()
