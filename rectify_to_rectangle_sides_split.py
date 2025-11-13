#!/usr/bin/env python3
"""Split wrapper: rectify two segments of aligned side images separately.

Runs rectify_to_rectangle_sides twice with independent input/output folders.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Segment 1
    ap.add_argument('--dir1', required=True, help='Aligned input dir for segment 1')
    ap.add_argument('--indices1', required=True)
    ap.add_argument('--out1', default='aligned_rectified_sides_1')

    # Segment 2
    ap.add_argument('--dir2', required=False, help='Aligned input dir for segment 2 (optional)')
    ap.add_argument('--indices2', required=False)
    ap.add_argument('--out2', default='aligned_rectified_sides_2')

    # Common knobs
    ap.add_argument('--image-suffix', default='_aligned.jpg')
    ap.add_argument('--corners-suffix', default='_aligned_corners.txt')
    ap.add_argument('--vertical-margin', type=int, default=20)
    ap.add_argument('--write-mask', action='store_true')
    args = ap.parse_args()

    script = str((Path(__file__).parent / 'rectify_to_rectangle_sides.py').resolve())
    cmd1 = [sys.executable, script,
            '--dir', args.dir1,
            '--indices', args.indices1,
            '--out', args.out1,
            '--image-suffix', args.image_suffix,
            '--corners-suffix', args.corners_suffix,
            '--vertical-margin', str(args.vertical_margin)]
    if args.write_mask:
        cmd1.append('--write-mask')
    subprocess.run(cmd1, check=True)

    if args.dir2 and args.indices2:
        cmd2 = [sys.executable, script,
                '--dir', args.dir2,
                '--indices', args.indices2,
                '--out', args.out2,
                '--image-suffix', args.image_suffix,
                '--corners-suffix', args.corners_suffix,
                '--vertical-margin', str(args.vertical_margin)]
        if args.write_mask:
            cmd2.append('--write-mask')
        subprocess.run(cmd2, check=True)
    else:
        print('Segment 2 skipped: single-container mode.')


if __name__ == '__main__':
    main()
