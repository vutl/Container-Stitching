#!/usr/bin/env python3
"""Split wrapper: stitch side panoramas for two segments separately.

Runs test3_sides twice with separate aligned dirs, indices, and output dirs.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    # Segment 1
    ap.add_argument('--aligned1', required=True, help='Aligned images dir for segment 1')
    ap.add_argument('--indices1', required=True)
    ap.add_argument('--corners1', required=True, help='Corners dir for segment 1')
    ap.add_argument('--out1', default='aligned_panorama_1')
    # Segment 2
    ap.add_argument('--aligned2', required=False, help='Aligned images dir for segment 2 (optional)')
    ap.add_argument('--indices2', required=False)
    ap.add_argument('--corners2', required=False, help='Corners dir for segment 2 (optional)')
    ap.add_argument('--out2', default='aligned_panorama_2')

    # Common knobs
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--mask-pad', type=int, default=0)
    ap.add_argument('--transform', type=str, default='translation', choices=['translation', 'affine', 'homography'])
    ap.add_argument('--blend', type=str, default='feather', choices=['feather', 'seam', 'none'])
    ap.add_argument('--lock-dy', action='store_true')
    ap.add_argument('--seam-width', type=int, default=1)
    ap.add_argument('--detector', type=str, default=None, choices=['sift', 'orb', 'akaze', 'kaze'])
    ap.add_argument('--loftr-weights', type=str, default=None, help='Path to EfficientLoFTR checkpoint to use for matching')
    args = ap.parse_args()

    script = str((Path(__file__).parent / 'test3_sides.py').resolve())
    # Segment 1
    cmd1 = [sys.executable, script,
            '--aligned-dir', args.aligned1,
            '--indices', args.indices1,
            '--corners-dir', args.corners1,
            '--out', args.out1,
            '--stride', str(args.stride),
            '--mask-pad', str(args.mask_pad),
            '--transform', args.transform,
            '--blend', args.blend,
            '--seam-width', str(args.seam_width)]
    if args.lock_dy:
        cmd1.append('--lock-dy')
    if args.detector:
        cmd1 += ['--detector', args.detector]
    if args.loftr_weights:
        cmd1 += ['--loftr-weights', args.loftr_weights]
    subprocess.run(cmd1, check=True)

    # Segment 2
    if args.aligned2 and args.indices2 and args.corners2:
        cmd2 = [sys.executable, script,
                '--aligned-dir', args.aligned2,
                '--indices', args.indices2,
                '--corners-dir', args.corners2,
                '--out', args.out2,
                '--stride', str(args.stride),
                '--mask-pad', str(args.mask_pad),
                '--transform', args.transform,
                '--blend', args.blend,
                '--seam-width', str(args.seam_width)]
        if args.lock_dy:
            cmd2.append('--lock-dy')
        if args.detector:
            cmd2 += ['--detector', args.detector]
        if args.loftr_weights:
            cmd2 += ['--loftr-weights', args.loftr_weights]
        subprocess.run(cmd2, check=True)
    else:
        print('Segment 2 skipped: single-container mode.')


if __name__ == '__main__':
    main()
