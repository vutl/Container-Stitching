#!/usr/bin/env python3
"""Detect image sequence direction and optionally produce reversed copies.

Behavior:
- Load a YOLO model and run detection on the "index 0" image (or the image
  with the smallest numeric index if 0 not present).
- If mean gu_cor x < mean edge_cor x -> left-to-right
  else -> right-to-left.
- If direction is right-to-left, produce a sibling folder "img_rev" under the
  input directory containing copies of the images with their numeric indices
  reversed (e.g. img_0 -> img_72 if 72 is the largest index). Also write a
  text file `direction.txt` containing the direction string.

This script expects file stems that end with a number, e.g. "img_12.jpg" or
"12.jpg". Files without a trailing number will be skipped with a warning.
"""

from pathlib import Path
import argparse
import re
import shutil
import sys
from typing import List, Tuple, Optional

import cv2

# reuse detection helpers from detect_yolo_corners.py (present in repo)
from detect_yolo_corners import load_model, run_detect


NUM_RE = re.compile(r"^(.*?)(\d+)$")


def find_indexed_files(d: Path, suffix: str) -> List[Tuple[Path, int, str]]:
    """Return list of (path, index, prefix) for files in dir matching suffix
    and whose stem ends with digits. Sorted by index ascending.
    """
    out = []
    for p in sorted(d.glob(f"*{suffix}")):
        stem = p.stem
        m = NUM_RE.match(stem)
        if not m:
            # skip non-numeric-stem files
            continue
        prefix, num = m.group(1), m.group(2)
        out.append((p, int(num), prefix))
    out.sort(key=lambda t: t[1])
    return out


def pick_reference_file(indexed: List[Tuple[Path, int, str]]) -> Optional[Path]:
    if not indexed:
        return None
    # prefer exact index 0 if present
    for p, idx, _ in indexed:
        if idx == 0:
            return p
    # otherwise return smallest index
    return indexed[0][0]


def mean_x_for_class(boxes, classes, target_cls: str) -> Optional[float]:
    xs = []
    for b, c in zip(boxes, classes):
        if c == target_cls:
            cx = 0.5 * (b[0] + b[2])
            xs.append(cx)
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def make_rev_mapping(indices: List[int]) -> dict:
    # Map original index -> reversed index preserving min offset
    if not indices:
        return {}
    lo = indices[0]
    hi = indices[-1]
    mapping = {}
    for orig in indices:
        mapping[orig] = lo + (hi - orig)
    return mapping


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dir', required=True, help='Directory with images (container folder)')
    ap.add_argument('--suffix', default='.jpg', help='Image filename suffix to match')
    ap.add_argument('--model', default='last11scor_3class_31_07.pt', help='YOLO model path')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.6)
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists() or not d.is_dir():
        print(f"Directory not found: {d}")
        sys.exit(2)

    indexed = find_indexed_files(d, args.suffix)
    if not indexed:
        print(f"No numeric-indexed images found in {d} matching suffix '{args.suffix}'")
        sys.exit(3)

    ref = pick_reference_file(indexed)
    if ref is None:
        print("Failed to pick reference image")
        sys.exit(4)

    print(f"Using reference image: {ref.name}")

    model = load_model(args.model)

    img = cv2.imread(str(ref))
    if img is None:
        print(f"Failed to read reference image: {ref}")
        sys.exit(5)

    boxes, scores, classes = run_detect(model, img, conf=args.conf, iou=args.iou)

    # Normalize class names to strings
    classes = [str(c) for c in classes]

    gu_x = mean_x_for_class(boxes, classes, 'gu_cor')
    edge_x = mean_x_for_class(boxes, classes, 'edge_cor')

    if gu_x is None or edge_x is None:
        print('Could not find both gu_cor and edge_cor in reference image; direction undetermined')
        # still write a file indicating unknown
        (d / 'direction.txt').write_text('undetermined')
        sys.exit(0)

    direction = 'left-to-right' if gu_x < edge_x else 'right-to-left'
    print(f"Detected direction: {direction}")
    (d / 'direction.txt').write_text(direction)

    if direction == 'right-to-left':
        out_dir = d / 'img_rev'
        out_dir.mkdir(parents=True, exist_ok=True)

        indices = [t[1] for t in indexed]
        mapping = make_rev_mapping(indices)

        for p, idx, prefix in indexed:
            m = NUM_RE.match(p.stem)
            if not m:
                print(f"Skipping non-numeric stem: {p.name}")
                continue
            pre = m.group(1)
            new_idx = mapping.get(idx)
            if new_idx is None:
                print(f"No mapping for index {idx}; skipping")
                continue
            new_name = f"{pre}{new_idx}{p.suffix}"
            dst = out_dir / new_name
            shutil.copy2(str(p), str(dst))
        print(f"Wrote reversed images to: {out_dir}")

    print('Done.')


if __name__ == '__main__':
    main()
