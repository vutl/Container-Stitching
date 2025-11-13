#!/usr/bin/env python3
"""Debug SIFT matching for frames 54-73 (sequential pairs).

Saves keypoint visualization for each pair in /home/atin-tts-1/Container-Stitching/debug_matches/
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def get_sift():
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT', cv2.NORM_L2
    elif hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT', cv2.NORM_L2
    else:
        return cv2.ORB_create(4000), 'ORB', cv2.NORM_HAMMING


def read_corners(path: Path):
    """Read corner file and return TL, TR, BR, BL points."""
    pts = {}
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        key = parts[0]
        try:
            x, y = float(parts[1]), float(parts[2])
            pts[key] = (x, y)
        except:
            continue
    if len(pts) < 4:
        return None
    return pts


def make_mask_from_corners(shape, corners, pad=0):
    """Create mask from corner bbox."""
    h, w = shape[:2]
    if not corners:
        return np.ones((h, w), dtype=np.uint8) * 255
    xs = [corners[k][0] for k in ('TL', 'TR', 'BR', 'BL') if k in corners]
    ys = [corners[k][1] for k in ('TL', 'TR', 'BR', 'BL') if k in corners]
    if not xs or not ys:
        return np.ones((h, w), dtype=np.uint8) * 255
    x0 = max(0, int(np.floor(min(xs))) - pad)
    x1 = min(w, int(np.ceil(max(xs))) + pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    y1 = min(h, int(np.ceil(max(ys))) + pad)
    mask = np.zeros((h, w), dtype=np.uint8)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def match_pair(img1_path: Path, img2_path: Path, 
               corners1_path: Path, corners2_path: Path,
               out_path: Path):
    """Match SIFT features between two images and save visualization."""
    img1 = cv2.imread(str(img1_path))
    img2 = cv2.imread(str(img2_path))
    if img1 is None or img2 is None:
        print(f"  ✗ Missing image: {img1_path.name} or {img2_path.name}")
        return
    
    # Get masks
    corners1 = read_corners(corners1_path)
    corners2 = read_corners(corners2_path)
    mask1 = make_mask_from_corners(img1.shape, corners1, pad=0)
    mask2 = make_mask_from_corners(img2.shape, corners2, pad=0)
    
    # SIFT detect
    det, det_name, norm = get_sift()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    
    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        print(f"  ✗ Not enough features: {img1_path.stem} kp={len(kp1) if kp1 else 0}, {img2_path.stem} kp={len(kp2) if kp2 else 0}")
        return
    
    # Match
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    
    if len(good) < 3:
        print(f"  ✗ Not enough matches: {img1_path.stem}→{img2_path.stem} matches={len(good)}")
        return
    
    # Filter inliers (translation only)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    deltas = pts2 - pts1
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))
    
    residuals = deltas - np.array([dx, dy], dtype=np.float32)
    dist = np.linalg.norm(residuals, axis=1)
    inliers = dist <= 12.0
    inlier_matches = [m for i, m in enumerate(good) if inliers[i]]
    
    # Draw matches
    match_img = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imwrite(str(out_path), match_img)
    print(f"  ✓ {img1_path.stem}→{img2_path.stem}: kp1={len(kp1)}, kp2={len(kp2)}, matches={len(good)}, inliers={len(inlier_matches)} | dx={dx:.1f} dy={dy:.1f} → {out_path.name}")


def main():
    # Fixed paths
    rectified_dir = Path('/home/atin-tts-1/Container-Stitching/downloads/gdrive_1sASbG/imgs_stit/right/MSNU713415145G1_10_10_2025_08_55_57_12253_42/rectified_c1')
    out_dir = Path('/home/atin-tts-1/Container-Stitching/debug_matches')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"SIFT Matching Debug: frames 54→73")
    print(f"Input dir: {rectified_dir}")
    print(f"Output dir: {out_dir}")
    print("=" * 70)
    
    # Process sequential pairs 54→55, 55→56, ..., 72→73
    for i in range(54, 73):
        idx1 = i
        idx2 = i + 1
        
        img1_path = rectified_dir / f"{idx1}_aligned.jpg"
        img2_path = rectified_dir / f"{idx2}_aligned.jpg"
        corners1_path = rectified_dir / f"{idx1}_aligned_corners.txt"
        corners2_path = rectified_dir / f"{idx2}_aligned_corners.txt"
        
        if not img1_path.exists() or not img2_path.exists():
            print(f"  ✗ Missing images: {idx1} or {idx2}")
            continue
        
        out_path = out_dir / f"{idx1:03d}_{idx2:03d}_matches.jpg"
        match_pair(img1_path, img2_path, corners1_path, corners2_path, out_path)
    
    print("=" * 70)
    print(f"✓ Done! Saved to: {out_dir}/")


if __name__ == '__main__':
    main()
