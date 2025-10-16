#!/usr/bin/env python3
"""Split wrapper: run side-face warp alignment separately for two segments.

This script wraps warp_align_sides: given two directories of trimmed images
and their updated corners, it runs alignment per segment and writes outputs
to independent destinations.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Segment 1
    ap.add_argument('--dir1', required=True, help='Trimmed images directory for segment 1')
    ap.add_argument('--indices1', required=True, help='Indices for segment 1 (e.g., 1-19)')
    ap.add_argument('--img-suffix1', default='_no_warp_trim.jpg')
    ap.add_argument('--corners-dir1', default='out_trimmed_sides_corners_1')
    ap.add_argument('--corners-suffix1', default='_corners.txt')
    ap.add_argument('--out1', default='aligned_sides_1')

    # Segment 2
    ap.add_argument('--dir2', required=True, help='Trimmed images directory for segment 2')
    ap.add_argument('--indices2', required=True, help='Indices for segment 2 (e.g., 20-58)')
    ap.add_argument('--img-suffix2', default='_no_warp_trim.jpg')
    ap.add_argument('--corners-dir2', default='out_trimmed_sides_corners_2')
    ap.add_argument('--corners-suffix2', default='_corners.txt')
    ap.add_argument('--out2', default='aligned_sides_2')

    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--bottom', type=int, default=20)
    ap.add_argument('--center-container', action='store_true')
    args = ap.parse_args()

    script = str((Path(__file__).parent / 'warp_align_sides.py').resolve())
    # Run segment 1
    cmd1 = [sys.executable, script,
            '--dir', args.dir1,
            '--indices', args.indices1,
            '--img-suffix', args.img_suffix1,
            '--corners-dir', args.corners_dir1,
            '--corners-suffix', args.corners_suffix1,
            '--out', args.out1,
            '--top', str(args.top),
            '--bottom', str(args.bottom)]
    if args.center_container:
        cmd1.append('--center-container')
    subprocess.run(cmd1, check=True)

    # Run segment 2
    cmd2 = [sys.executable, script,
            '--dir', args.dir2,
            '--indices', args.indices2,
            '--img-suffix', args.img_suffix2,
            '--corners-dir', args.corners_dir2,
            '--corners-suffix', args.corners_suffix2,
            '--out', args.out2,
            '--top', str(args.top),
            '--bottom', str(args.bottom)]
    if args.center_container:
        cmd2.append('--center-container')
    subprocess.run(cmd2, check=True)


if __name__ == '__main__':
    main()
