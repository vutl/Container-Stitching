#!/usr/bin/env python3
"""
End-to-end side pipeline runner (single command).

This script orchestrates the full split pipeline over a folder of images:
  1) Detect + split sequence into two segments using YOLO corners
  2) Trim black borders and update corners per segment
  3) Warp-align each segment to a uniform canvas
  4) Stitch each segment into a panorama
  5) Rectify aligned frames per segment

It automatically detects whether a second segment exists (double-container
case). If segment 2 is empty, it skips steps for segment 2.

Outputs are written under an output root folder, keeping all artifacts
isolated (annotations, trimmed, aligned, panoramas, rectified).

Example minimal run:
  python run_side_split_pipeline.py \
    --img-dir /path/to/img2 \
    --suffix _no_warp.jpg \
    --model last11scor_3class_31_07.pt \
    --out-root side_pipeline_out \
    --stitch-blend seam --lock-dy --stitch-transform affine --stitch-detector sift

Notes:
  - If you omit --indices, all files matching the suffix are processed
  - For typical datasets with 1..N numeric names, the script infers indices
    automatically for each stage by scanning the corresponding folders
  - You can tune alignment and stitching knobs via CLI flags
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _list_indices_from_dir(dir_path: Path, suffix: str, numeric_only: bool = True) -> List[int]:
    idxs: List[int] = []
    if not dir_path.exists():
        return idxs
    for p in sorted(dir_path.glob(f"*{suffix}")):
        name = p.name
        stem = name[:-len(suffix)] if name.endswith(suffix) else p.stem
        if numeric_only and not re.fullmatch(r"\d+", stem or ""):
            continue
        try:
            idxs.append(int(stem))
        except Exception:
            # ignore non-numeric stems when requested
            if not numeric_only:
                continue
    return sorted(set(idxs))


def _indices_to_spec(idxs: List[int]) -> Optional[str]:
    """Convert a list of ints to a compact comma-separated spec accepted by our CLIs.

    We use a simple comma-join to preserve gaps. All downstream scripts accept
    comma-separated integers ("parse_indices" splits on commas/spaces).
    """
    if not idxs:
        return None
    return ",".join(str(i) for i in idxs)


def _has_any_files(dir_path: Path, pattern: str) -> bool:
    return any(dir_path.glob(pattern))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--img-dir', required=True, help='Directory containing source images')
    ap.add_argument('--suffix', default='_no_warp.jpg', help='Image suffix to match (default: _no_warp.jpg)')
    ap.add_argument('--indices', default=None, help='Optional indices/ranges for detection (e.g., "1-60" or "1,2,5")')

    # Detection / split
    ap.add_argument('--model', default='last11scor_3class_31_07.pt')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.6)

    # Alignment canvas knobs
    ap.add_argument('--align-top', type=int, default=20)
    ap.add_argument('--align-bottom', type=int, default=20)
    ap.add_argument('--center-container', action='store_true', help='Center container horizontally on canvas')

    # Stitching knobs (defaults mimic the "last good" command)
    ap.add_argument('--stitch-stride', type=int, default=1)
    ap.add_argument('--stitch-mask-pad', type=int, default=0)
    ap.add_argument('--stitch-transform', choices=['translation', 'affine', 'homography'], default='translation')
    ap.add_argument('--stitch-blend', choices=['feather', 'seam', 'none'], default='seam')
    ap.add_argument('--lock-dy', action='store_true')
    ap.add_argument('--seam-width', type=int, default=3)
    ap.add_argument('--stitch-detector', choices=['sift', 'orb', 'akaze', 'kaze'], default=None)

    # Output root (default: parent directory of img-dir, i.e., the container folder)
    ap.add_argument('--out-root', default=None, help='Output root folder. Default: the parent of --img-dir (container folder).')
    args = ap.parse_args()

    img_dir = Path(args.img_dir).resolve()
    out_root = Path(args.out_root).resolve() if args.out_root else Path(img_dir).parent.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # 1) Detect + split
    out_annot1 = out_root / 'out_annot_1'
    out_annot2 = out_root / 'out_annot_2'
    out_annot1.mkdir(parents=True, exist_ok=True)
    out_annot2.mkdir(parents=True, exist_ok=True)

    detect_script = str((Path(__file__).parent / 'test_sides_split.py').resolve())
    cmd_detect = [
        sys.executable, detect_script,
        '--model', str(args.model),
        '--dir', str(img_dir),
        '--suffix', args.suffix,
        '--out1', str(out_annot1),
        '--out2', str(out_annot2),
        '--conf', str(args.conf),
        '--iou', str(args.iou),
    ]
    if args.indices:
        cmd_detect += ['--indices', args.indices]
    print('[1/5] Detect + split:', ' '.join(cmd_detect))
    subprocess.run(cmd_detect, check=True)

    # Decide if segment 2 exists
    seg2_exists = _has_any_files(out_annot2, '*_annot*.txt') or _has_any_files(out_annot2, '*_corners.txt')
    if seg2_exists:
        print('Detected a double-container sequence (segment 2 present).')
    else:
        print('No segment 2 detected; treating as single-container sequence.')

    # 2) Trim black borders + update corners
    out_trim1 = out_root / 'out_trimmed_sides_1'
    out_trim2 = out_root / 'out_trimmed_sides_2'
    out_trim_c1 = out_root / 'out_trimmed_sides_corners_1'
    out_trim_c2 = out_root / 'out_trimmed_sides_corners_2'
    out_trim1.mkdir(exist_ok=True)
    out_trim2.mkdir(exist_ok=True)
    out_trim_c1.mkdir(exist_ok=True)
    out_trim_c2.mkdir(exist_ok=True)

    trim_script = str((Path(__file__).parent / 'trim_black_update_corners_split.py').resolve())
    cmd_trim = [
        sys.executable, trim_script,
        '--img-dir', str(img_dir),
        '--suffix', args.suffix,
        '--corners1-dir', str(out_annot1),
        '--corners2-dir', str(out_annot2),
        '--out1', str(out_trim1), '--out2', str(out_trim2),
        '--corners1-out', str(out_trim_c1), '--corners2-out', str(out_trim_c2),
        '--corners-suffix', '_corners.txt',
    ]
    print('[2/5] Trim + update corners:', ' '.join(cmd_trim))
    subprocess.run(cmd_trim, check=True)

    # Build indices for alignment per segment based on trimmed outputs.
    # The trimming step historically wrote files ending with '_no_warp_trim.jpg',
    # but some variants produce '_trim.jpg'. Try both so the runner is robust.
    def _list_indices_flexible(dir_path: Path, suffix: str):
        """List numeric indices by extracting the first integer found in filenames.

        This handles names like 'img_0_trim.jpg' or '0_trim.jpg' and returns a
        sorted list of unique integers.
        """
        idxs = []
        if not dir_path.exists():
            return idxs
        for p in sorted(dir_path.glob(f"*{suffix}")):
            m = re.search(r"(\d+)", p.name)
            if m:
                try:
                    idxs.append(int(m.group(1)))
                except Exception:
                    continue
        return sorted(set(idxs))

    def _first_nonempty_indices(dir_path: Path, suffixes):
        for s in suffixes:
            idxs = _list_indices_flexible(dir_path, s)
            if idxs:
                return idxs, s
        return [], None

    idxs1, used_trim_suffix1 = _first_nonempty_indices(out_trim1, ['_no_warp_trim.jpg', '_trim.jpg'])
    idxs2, used_trim_suffix2 = _first_nonempty_indices(out_trim2, ['_no_warp_trim.jpg', '_trim.jpg'])
    spec1 = _indices_to_spec(idxs1)
    spec2 = _indices_to_spec(idxs2)
    if not spec1:
        print('No trimmed frames found for segment 1. Exiting.')
        return
    if not seg2_exists or not spec2:
        seg2_exists = False  # ensure we skip downstream if nothing to align

    # 3) Warp-align
    aligned1 = out_root / 'aligned_sides_1'
    aligned2 = out_root / 'aligned_sides_2'
    aligned1.mkdir(exist_ok=True)
    aligned2.mkdir(exist_ok=True)

    align_script = str((Path(__file__).parent / 'warp_align_sides_split.py').resolve())
    cmd_align = [
        sys.executable, align_script,
        '--dir1', str(out_trim1), '--indices1', spec1,
        '--corners-dir1', str(out_trim_c1), '--out1', str(aligned1),
        '--dir2', str(out_trim2), '--indices2', spec2 if spec2 else '',
        '--corners-dir2', str(out_trim_c2), '--out2', str(aligned2),
        '--top', str(args.align_top), '--bottom', str(args.align_bottom),
    ]
    # If trimming produced a non-default suffix, pass it to the align wrapper
    if used_trim_suffix1:
        cmd_align += ['--img-suffix1', used_trim_suffix1]
    if used_trim_suffix2:
        cmd_align += ['--img-suffix2', used_trim_suffix2]
    if args.center_container:
        cmd_align.append('--center-container')
    print('[3/5] Warp-align:', ' '.join(cmd_align))
    # If seg2 is empty, we still call align script; it will skip missing frames
    subprocess.run(cmd_align, check=True)

    # Build indices for stitching per segment based on aligned outputs
    aidxs1 = _list_indices_from_dir(aligned1, '_aligned.jpg')
    aidxs2 = _list_indices_from_dir(aligned2, '_aligned.jpg')
    aspec1 = _indices_to_spec(aidxs1)
    aspec2 = _indices_to_spec(aidxs2)
    if not aspec1:
        print('No aligned frames found for segment 1. Exiting.')
        return
    if not aidxs2:
        seg2_exists = False

    # 4) Rectify aligned frames (per-segment)
    rect1 = out_root / 'aligned_rectified_sides_1'
    rect2 = out_root / 'aligned_rectified_sides_2'
    rect1.mkdir(exist_ok=True)
    rect2.mkdir(exist_ok=True)
    rectify_script = str((Path(__file__).parent / 'rectify_to_rectangle_sides_split.py').resolve())
    # Prefer compact ranges for rectification (frames are typically contiguous)
    rect_spec1 = f"{aidxs1[0]}-{aidxs1[-1]}" if aidxs1 else ''
    rect_spec2 = f"{aidxs2[0]}-{aidxs2[-1]}" if aidxs2 else ''
    cmd_rectify = [
        sys.executable, rectify_script,
        '--dir1', str(aligned1), '--indices1', rect_spec1 or aspec1 or '', '--out1', str(rect1),
        '--dir2', str(aligned2), '--indices2', rect_spec2 or aspec2 or '', '--out2', str(rect2),
        '--image-suffix', '_aligned.jpg', '--corners-suffix', '_aligned_corners.txt',
        '--vertical-margin', '20'
    ]
    print('[4/5] Rectify:', ' '.join(cmd_rectify))
    subprocess.run(cmd_rectify, check=True)

    # 5) Stitch panoramas (run on rectified outputs)
    pano_suffix = '_seam' if args.stitch_blend == 'seam' else ''
    pano1 = out_root / f"out_panorama_1{pano_suffix}"
    pano2 = out_root / f"out_panorama_2{pano_suffix}"
    pano1.mkdir(exist_ok=True)
    pano2.mkdir(exist_ok=True)
    stitch_script = str((Path(__file__).parent / 'test3_sides_split.py').resolve())
    # Use rectified folders as the aligned input and corners source for stitching
    cmd_stitch = [
        sys.executable, stitch_script,
        '--aligned1', str(rect1), '--indices1', aspec1,
        '--corners1', str(rect1), '--out1', str(pano1),
        '--aligned2', str(rect2), '--indices2', aspec2 if aspec2 else '',
        '--corners2', str(rect2), '--out2', str(pano2),
        '--stride', str(args.stitch_stride), '--mask-pad', str(args.stitch_mask_pad),
        '--transform', args.stitch_transform, '--blend', args.stitch_blend,
        '--seam-width', str(args.seam_width)
    ]
    if args.lock_dy:
        cmd_stitch.append('--lock-dy')
    if args.stitch_detector:
        cmd_stitch += ['--detector', args.stitch_detector]
    print('[5/5] Stitch:', ' '.join(cmd_stitch))
    subprocess.run(cmd_stitch, check=True)

    print('\nCompleted side split pipeline.')
    print('Outputs:')
    print(f'  Annotations:  {out_annot1}  and  {out_annot2} (if any)')
    print(f'  Trimmed:      {out_trim1}  and  {out_trim2} (if any)')
    print(f'  Aligned:      {aligned1}   and  {aligned2} (if any)')
    print(f'  Panoramas:    {pano1}      and  {pano2} (if any)')
    print(f'  Rectified:    {rect1}      and  {rect2} (if any)')


if __name__ == '__main__':
    main()
