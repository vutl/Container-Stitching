#!/usr/bin/env python3
"""
Full pipeline runner for side-face container stitching (split double-container).

Runs the entire workflow from raw images to final panoramas:
1. Corner detection & split (test_sides_split.py)
2. Trim black borders per segment (trim_black_update_corners_split.py)
3. Align frames per segment (warp_align_sides_split.py)
4. Rectify to rectangles per segment (rectify_to_rectangle_sides_split.py)
5. Stitch panoramas per segment (test3_sides_split.py)

All intermediate outputs are stored within the input folder.
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path


def run_cmd(cmd: list, desc: str):
    """Run a subprocess command with error handling."""
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✓ {desc} completed successfully")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--img-dir', required=True, help='Input directory containing raw images (e.g., downloads/.../GVCU.../img2)')
    ap.add_argument('--img-suffix', default='_no_warp.jpg', help='Image filename suffix')
    ap.add_argument('--model', default='last11scor_3class_31_07.pt', help='YOLO model path')
    ap.add_argument('--conf', type=float, default=0.005, help='YOLO confidence threshold')
    ap.add_argument('--iou', type=float, default=0.15, help='YOLO IoU threshold')
    
    # Stitching parameters
    ap.add_argument('--stride', type=int, default=3, help='Stitching stride (default: 3)')
    ap.add_argument('--blend', type=str, default='seam', choices=['feather', 'seam', 'none'], help='Blend mode (default: seam)')
    ap.add_argument('--seam-width', type=int, default=8, help='Seam blend width (default: 8)')
    ap.add_argument('--lock-dy', action='store_true', help='Force vertical translation to zero')
    
    args = ap.parse_args()
    
    img_dir = Path(args.img_dir).resolve()
    if not img_dir.exists():
        print(f"ERROR: Image directory not found: {img_dir}")
        sys.exit(1)
    
    base_dir = Path(__file__).parent.resolve()
    
    # Define output folders (all within img_dir)
    split_c1 = img_dir / 'split_c1'
    split_c2 = img_dir / 'split_c2'
    trimmed_c1 = img_dir / 'trimmed_c1'
    trimmed_c2 = img_dir / 'trimmed_c2'
    trimmed_corners_c1 = img_dir / 'trimmed_corners_c1'
    trimmed_corners_c2 = img_dir / 'trimmed_corners_c2'
    aligned_c1 = img_dir / 'aligned_c1'
    aligned_c2 = img_dir / 'aligned_c2'
    rectified_c1 = img_dir / 'rectified_c1'
    rectified_c2 = img_dir / 'rectified_c2'
    panorama_c1 = img_dir / 'panorama_c1'
    panorama_c2 = img_dir / 'panorama_c2'
    
    # Step 1: Corner detection & split
    run_cmd([
        sys.executable, str(base_dir / 'test_sides_split.py'),
        '--model', args.model,
        '--dir', str(img_dir),
        '--suffix', args.img_suffix,
        '--conf', str(args.conf),
        '--iou', str(args.iou),
        '--out1', str(split_c1),
        '--out2', str(split_c2),
    ], "1. Corner detection & split")
    
    # Determine indices from split output
    c1_files = sorted(split_c1.glob('*_corners.txt'))
    c2_files = sorted(split_c2.glob('*_corners.txt'))
    
    if not c1_files:
        print("ERROR: No corners found in C1 after split. Check detection step.")
        sys.exit(1)
    
    # Extract indices from filenames like "img_0_corners.txt" or "0_corners.txt"
    def extract_index(f):
        parts = f.stem.split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return None
    
    c1_indices = [extract_index(f) for f in c1_files]
    c2_indices = [extract_index(f) for f in c2_files]
    c1_indices = [i for i in c1_indices if i is not None]
    c2_indices = [i for i in c2_indices if i is not None]
    
    c1_range = f"{min(c1_indices)}-{max(c1_indices)}" if c1_indices else ""
    c2_range = f"{min(c2_indices)}-{max(c2_indices)}" if c2_indices else ""
    
    has_c2 = len(c2_indices) > 0
    
    # Read direction info from detection step
    direction_file = img_dir / 'direction.txt'
    needs_reverse = False
    if direction_file.exists():
        direction = direction_file.read_text().strip()
        needs_reverse = (direction == 'right_to_left')
        if needs_reverse:
            print(f"\n[DIRECTION] Detected RIGHT→LEFT: Will stitch in reverse order (img_{max(c1_indices)}→img_{min(c1_indices)})")
        else:
            print(f"\n[DIRECTION] Detected LEFT→RIGHT: Will stitch in normal order (img_{min(c1_indices)}→img_{max(c1_indices)})")
    else:
        print("\nWarning: direction.txt not found, assuming left_to_right")
    
    print(f"\nDetected indices:")
    print(f"  C1: {c1_range} ({len(c1_indices)} frames)")
    if has_c2:
        print(f"  C2: {c2_range} ({len(c2_indices)} frames)")
    else:
        print(f"  C2: No frames detected (single container)")
    
    if not c1_indices:
        print("ERROR: C1 segment has no frames.")
        sys.exit(1)
    
    # Step 2: Trim black borders
    trim_cmd = [
        sys.executable, str(base_dir / 'trim_black_update_corners_split.py'),
        '--img-dir', str(img_dir),
        '--suffix', args.img_suffix,
        '--corners1-dir', str(split_c1),
        '--out1', str(trimmed_c1),
        '--corners1-out', str(trimmed_corners_c1),
    ]
    if has_c2:
        trim_cmd.extend([
            '--corners2-dir', str(split_c2),
            '--out2', str(trimmed_c2),
            '--corners2-out', str(trimmed_corners_c2),
        ])
    run_cmd(trim_cmd, "2. Trim black borders")
    
    # Step 3: Align frames
    align_cmd = [
        sys.executable, str(base_dir / 'warp_align_sides_split.py'),
        '--dir1', str(trimmed_c1),
        '--indices1', c1_range,
        '--img-suffix1', '_trim.jpg',
        '--corners-dir1', str(trimmed_corners_c1),
        '--out1', str(aligned_c1),
        '--top', '20',
        '--bottom', '20',
    ]
    if has_c2:
        align_cmd.extend([
            '--dir2', str(trimmed_c2),
            '--indices2', c2_range,
            '--img-suffix2', '_trim.jpg',
            '--corners-dir2', str(trimmed_corners_c2),
            '--out2', str(aligned_c2),
        ])
    run_cmd(align_cmd, "3. Align frames")
    
    # Step 4: Rectify to rectangles
    rectify_cmd = [
        sys.executable, str(base_dir / 'rectify_to_rectangle_sides_split.py'),
        '--dir1', str(aligned_c1),
        '--indices1', c1_range,
        '--out1', str(rectified_c1),
        '--vertical-margin', '20',
    ]
    if has_c2:
        rectify_cmd.extend([
            '--dir2', str(aligned_c2),
            '--indices2', c2_range,
            '--out2', str(rectified_c2),
        ])
    run_cmd(rectify_cmd, "4. Rectify to rectangles")
    
    # Determine actual aligned indices from rectified output
    rectified_c1_files = sorted(rectified_c1.glob('*_aligned.jpg'))
    rectified_c2_files = sorted(rectified_c2.glob('*_aligned.jpg')) if has_c2 else []
    
    def extract_aligned_index(f):
        # Extract index from "0_aligned.jpg" or "img_0_aligned.jpg"
        stem = f.stem  # "0_aligned" or "img_0_aligned"
        parts = stem.replace('_aligned', '').split('_')
        for p in parts:
            if p.isdigit():
                return int(p)
        return None
    
    rectified_c1_indices = [extract_aligned_index(f) for f in rectified_c1_files]
    rectified_c1_indices = [i for i in rectified_c1_indices if i is not None]
    
    if has_c2:
        rectified_c2_indices = [extract_aligned_index(f) for f in rectified_c2_files]
        rectified_c2_indices = [i for i in rectified_c2_indices if i is not None]
    
    # Build comma-separated indices for stitching (only actual frames)
    c1_stitch_indices = ','.join(map(str, sorted(rectified_c1_indices)))
    c2_stitch_indices = ','.join(map(str, sorted(rectified_c2_indices))) if has_c2 else ""
    
    print(f"\nRectified frames for stitching:")
    print(f"  C1: {len(rectified_c1_indices)} frames - indices: {c1_stitch_indices}")
    if has_c2:
        print(f"  C2: {len(rectified_c2_indices)} frames - indices: {c2_stitch_indices}")
    
    # Edge case handling: we can stitch only segments that have >=2 rectified frames.
    # If a segment has exactly 1 rectified frame, copy that rectified image into
    # a panorama folder as a fallback (single-frame "panorama") so the pipeline
    # doesn't fail entirely.
    if not rectified_c1_indices and not (has_c2 and rectified_c2_indices):
        print("ERROR: No rectified frames found for both C1 and C2. Cannot stitch.")
        sys.exit(1)

    # Prepare lists that indicate which segment will be stitched
    stitch_c1 = False
    stitch_c2 = False

    # Decide for C1
    if len(rectified_c1_indices) >= 2:
        stitch_c1 = True
    elif len(rectified_c1_indices) == 1:
        # If C2 is stitchable, prefer stitching C2 and skip C1. Otherwise copy
        # the single rectified image as a fallback panorama for C1.
        if has_c2 and len(rectified_c2_indices) >= 2:
            print(f"WARNING: C1 has only 1 rectified frame. Will skip C1 and stitch C2 only.")
            rectified_c1_indices = []
            c1_stitch_indices = ""
        else:
            print(f"INFO: C1 has only 1 rectified frame. Creating single-image panorama fallback for C1.")
            panorama_c1.mkdir(parents=True, exist_ok=True)
            idx = rectified_c1_indices[0]
            src = rectified_c1 / f"{idx}_aligned.jpg"
            dst = panorama_c1 / f"panorama_{idx}_{idx}_bbox.jpg"
            try:
                shutil.copy(str(src), str(dst))
                print(f"  Copied single rectified image to {dst}")
            except Exception as e:
                print(f"WARNING: failed to copy single rectified image for C1: {e}")
            # Do not set stitch_c1 True since there's nothing to stitch
            rectified_c1_indices = []
            c1_stitch_indices = ""

    # Decide for C2
    if has_c2 and len(rectified_c2_indices) >= 2:
        stitch_c2 = True
    elif has_c2 and len(rectified_c2_indices) == 1:
        # If C1 is stitchable, prefer C1; otherwise create single-image fallback for C2
        if stitch_c1:
            print(f"WARNING: C2 has only 1 rectified frame. Will skip C2 and stitch C1 only.")
            rectified_c2_indices = []
            c2_stitch_indices = ""
        else:
            print(f"INFO: C2 has only 1 rectified frame. Creating single-image panorama fallback for C2.")
            panorama_c2.mkdir(parents=True, exist_ok=True)
            idx = rectified_c2_indices[0]
            src = rectified_c2 / f"{idx}_aligned.jpg"
            dst = panorama_c2 / f"panorama_{idx}_{idx}_bbox.jpg"
            try:
                shutil.copy(str(src), str(dst))
                print(f"  Copied single rectified image to {dst}")
            except Exception as e:
                print(f"WARNING: failed to copy single rectified image for C2: {e}")
            rectified_c2_indices = []
            c2_stitch_indices = ""

    # Build and run stitch commands for segments that need stitching.
    # We call the split stitcher when both segments are being stitched together,
    # otherwise call the single-side stitcher for the stitchable segment.
    if stitch_c1 and (stitch_c2):
        stitch_cmd = [
            sys.executable, str(base_dir / 'test3_sides_split.py'),
            '--aligned1', str(rectified_c1),
            '--indices1', c1_stitch_indices,
            '--corners1', str(rectified_c1),
            '--out1', str(panorama_c1),
            '--aligned2', str(rectified_c2),
            '--indices2', c2_stitch_indices,
            '--corners2', str(rectified_c2),
            '--out2', str(panorama_c2),
            '--transform', 'translation',
            '--blend', args.blend,
            '--stride', str(args.stride),
            '--seam-width', str(args.seam_width),
            '--detector', 'sift',
        ]
    elif stitch_c1:
        stitch_cmd = [
            sys.executable, str(base_dir / 'test3_sides.py'),
            '--aligned-dir', str(rectified_c1),
            '--indices', c1_stitch_indices,
            '--corners-dir', str(rectified_c1),
            '--out', str(panorama_c1),
            '--transform', 'translation',
            '--blend', args.blend,
            '--stride', str(args.stride),
            '--seam-width', str(args.seam_width),
            '--detector', 'sift',
        ]
    elif stitch_c2:
        stitch_cmd = [
            sys.executable, str(base_dir / 'test3_sides.py'),
            '--aligned-dir', str(rectified_c2),
            '--indices', c2_stitch_indices,
            '--corners-dir', str(rectified_c2),
            '--out', str(panorama_c2),
            '--transform', 'translation',
            '--blend', args.blend,
            '--stride', str(args.stride),
            '--seam-width', str(args.seam_width),
            '--detector', 'sift',
        ]
    else:
        print("INFO: No segments require stitching (single-image fallbacks created where possible). Skipping stitching step.")
        stitch_cmd = None

    if stitch_cmd:
        if args.lock_dy:
            stitch_cmd.append('--lock-dy')
        if needs_reverse:
            stitch_cmd.append('--reverse')
        run_cmd(stitch_cmd, "5. Stitch panoramas")
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nOutput locations:")
    print(f"  Panorama C1: {panorama_c1}/panorama_*_bbox.jpg")
    if has_c2:
        print(f"  Panorama C2: {panorama_c2}/panorama_*_bbox.jpg")
    else:
        print(f"  Panorama C2: (skipped - single container)")
    print(f"\nAll intermediate outputs in: {img_dir}/")


if __name__ == '__main__':
    main()
