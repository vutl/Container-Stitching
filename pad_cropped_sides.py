#!/usr/bin/env python3
"""Pad left and right sides of images whose filenames contain "_cropped".
Creates a new file next to each input with suffix "_padded" before the extension by default.

Usage examples:
  python3 pad_cropped_sides.py            # pads all *_cropped.jpg/.png under cwd with 30px per side
  python3 pad_cropped_sides.py --pad 50   # use 50px per side
  python3 pad_cropped_sides.py --root extracted/ --pad 20
  python3 pad_cropped_sides.py --inplace  # overwrite originals (use carefully)

The script defaults to non-destructive behavior and prints a short summary.
"""
import argparse
from pathlib import Path
import cv2
import sys


def pad_image_left_right(img, pad_px, color=(0,0,0)):
    h, w = img.shape[:2]
    new_w = w + 2*pad_px
    canvas = None
    if img.ndim == 2:
        # grayscale
        canvas = (np.zeros((h, new_w), dtype=img.dtype) + color[0]) if isinstance(color, tuple) else np.zeros((h, new_w), dtype=img.dtype)
    else:
        canvas = np.zeros((h, new_w, 3), dtype=img.dtype)
        canvas[:] = color
    canvas[:, pad_px:pad_px+w] = img
    return canvas


if __name__ == '__main__':
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='.', help='Root folder to search')
    ap.add_argument('--pad', type=int, default=30, help='Pixels to add on left and right')
    ap.add_argument('--ext', type=str, default='jpg', help='Image extension to process (jpg|png)')
    ap.add_argument('--inplace', action='store_true', help='Overwrite original files (dangerous)')
    ap.add_argument('--dry-run', action='store_true', help='Just list files that would be processed')
    ap.add_argument('--verbose', action='store_true', help='Verbose output')
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print('Root not found:', root)
        sys.exit(1)

    patterns = [f'*_cropped.{args.ext}', f'*_cropped.*{args.ext}']
    files = list(root.rglob(f'*_cropped.{args.ext}'))
    # also include uppercase extension variants
    files += list(root.rglob(f'*_cropped.{args.ext.upper()}'))
    files = sorted(set(files))

    if not files:
        print('No files found matching "*_cropped.%s" under %s' % (args.ext, root))
        sys.exit(0)

    touched = 0
    for p in files:
        try:
            if args.verbose:
                print('Found:', p)
            if args.dry_run:
                continue
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                print('Failed to read (skipping):', p)
                continue
            h, w = img.shape[:2]
            if args.pad <= 0:
                print('pad <=0, skipping:', p)
                continue
            # create padded image
            padded = pad_image_left_right(img, args.pad, color=(0,0,0))
            if args.inplace:
                outp = p
            else:
                outp = p.with_name(p.stem + '_padded' + p.suffix)
            ok = cv2.imwrite(str(outp), padded)
            if not ok:
                print('Write failed:', outp)
                continue
            touched += 1
            if args.verbose:
                print(f'Wrote: {outp} (orig {w}x{h} -> {padded.shape[1]}x{padded.shape[0]})')
        except Exception as e:
            print('Error processing', p, e)

    print(f'Done. Processed {touched} files (pad={args.pad}px, root={root}).')
