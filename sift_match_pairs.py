#!/usr/bin/env python3
"""Detect SIFT features (masked) and draw matches for given image pairs.

Saves results as <a>_<b>_sift_matches.jpg in the current folder.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np


def read_mask(path: Path):
    if not path.exists():
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    return (m > 0).astype('uint8')


def run_pair(a_path: Path, b_path: Path, a_mask: Path = None, b_mask: Path = None, out: Path = None):
    a = cv2.imread(str(a_path))
    b = cv2.imread(str(b_path))
    if a is None or b is None:
        print('Missing images', a_path, b_path)
        return None
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

    mask_a = read_mask(a_mask) if a_mask else None
    mask_b = read_mask(b_mask) if b_mask else None
    if mask_a is None:
        mask_a = np.ones(a_gray.shape, dtype=np.uint8)
    if mask_b is None:
        mask_b = np.ones(b_gray.shape, dtype=np.uint8)

    # prefer SIFT; fall back to ORB
    if hasattr(cv2, 'SIFT_create'):
        det = cv2.SIFT_create()
        norm = cv2.NORM_L2
    else:
        det = cv2.ORB_create(4000)
        norm = cv2.NORM_HAMMING

    kp1, des1 = det.detectAndCompute(a_gray, mask_a)
    kp2, des2 = det.detectAndCompute(b_gray, mask_b)
    if des1 is None or des2 is None:
        print('no descriptors')
        return None

    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    good = sorted(good, key=lambda x: x.distance)[:400]

    # compute geometry for filtering: angles and lengths
    pts1 = np.array([kp1[m.queryIdx].pt for m in good], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good], dtype=np.float32)
    vec = pts2 - pts1
    dx = vec[:, 0]
    dy = vec[:, 1]
    angles = np.degrees(np.arctan2(dy, dx))  # -180..180
    # normalize angles to [-90, 90]
    angles = ((angles + 90) % 180) - 90
    lengths = np.hypot(dx, dy)

    # find dominant angle bin
    if angles.size == 0:
        filt_matches = good
    else:
        ang_bin = 5.0
        ang_bins = np.arange(-90.0, 90.0 + ang_bin, ang_bin)
        hist, edges = np.histogram(angles, bins=ang_bins)
        dominant_idx = int(np.argmax(hist))
        dominant_center = (edges[dominant_idx] + edges[dominant_idx+1]) / 2.0
        angle_tol = 12.0
        keep_ang = np.abs(((angles - dominant_center + 180) % 360) - 180) <= angle_tol

        # length binning: choose bin size dynamically (use 20 px default)
        if lengths.size == 0:
            keep_len = np.ones_like(keep_ang, dtype=bool)
        else:
            bin_size = max(8.0, np.median(lengths) * 0.1)
            len_bins = np.arange(0, lengths.max() + bin_size, bin_size)
            len_hist, ledges = np.histogram(lengths, bins=len_bins)
            len_idx = int(np.argmax(len_hist)) if len_hist.size>0 else 0
            len_low = ledges[len_idx]
            len_high = ledges[len_idx+1] if len_idx+1 < ledges.size else ledges[-1]
            keep_len = (lengths >= len_low) & (lengths < len_high)

        keep = keep_ang & keep_len
        filt_matches = [m for k, m in enumerate(good) if keep[k]]

    img = cv2.drawMatches(a, kp1, b, kp2, filt_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if out is None:
        out = Path(f"{a_path.stem}_{b_path.stem}_sift_matches.jpg")
    cv2.imwrite(str(out), img)
    print('Saved', out)

    # Also compute homography from filtered matches and create a simple stitch
    stitched_out = None
    if len(filt_matches) >= 3:
        pts1 = np.float32([kp1[m.queryIdx].pt for m in filt_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in filt_matches])
        deltas = pts2 - pts1
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        residuals = deltas - np.array([dx, dy], dtype=np.float32)
        dist = np.linalg.norm(residuals, axis=1)
        inlier_mask = dist <= 12.0
        if np.any(inlier_mask):
            dx = float(np.median(deltas[inlier_mask, 0]))
            dy = float(np.median(deltas[inlier_mask, 1]))
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = dx
        H[1, 2] = dy

        h1, w1 = a.shape[:2]
        h2, w2 = b.shape[:2]
        corners_a = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(corners_a, H)
        all_pts = np.concatenate([mapped, np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)], axis=0)
        x_coords = all_pts[:, 0, 0]
        y_coords = all_pts[:, 0, 1]
        x_min = int(np.floor(x_coords.min()))
        x_max = int(np.ceil(x_coords.max()))
        y_min = int(np.floor(y_coords.min()))
        y_max = int(np.ceil(y_coords.max()))
        tx = -min(0, x_min)
        ty = -min(0, y_min)
        canvas_w = x_max + tx
        canvas_h = y_max + ty
        T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        H_full = T @ H
        warped_a = cv2.warpPerspective(a, H_full, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        mask_a_u8 = mask_a if mask_a is not None else np.ones((h1, w1), dtype=np.uint8) * 255
        warped_mask_a = cv2.warpPerspective(mask_a_u8, H_full, (canvas_w, canvas_h), flags=cv2.INTER_NEAREST,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        canvas_b = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas_b[ty:ty + h2, tx:tx + w2] = b
        mask_b_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask_b_src = mask_b if mask_b is not None else np.ones((h2, w2), dtype=np.uint8) * 255
        mask_b_canvas[ty:ty + h2, tx:tx + w2] = mask_b_src

        only_a = (warped_mask_a > 0) & (mask_b_canvas == 0)
        only_b = (mask_b_canvas > 0) & (warped_mask_a == 0)
        overlap = (warped_mask_a > 0) & (mask_b_canvas > 0)

        out = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        out[only_a] = warped_a[only_a].astype(np.float32)
        out[only_b] = canvas_b[only_b].astype(np.float32)

        if np.any(overlap):
            # Shrink masks slightly to preserve sharper core before blending
            kernel = np.ones((3, 3), np.uint8)
            mask_a_core = cv2.erode((warped_mask_a > 0).astype(np.uint8) * 255, kernel, iterations=1)
            mask_b_core = cv2.erode((mask_b_canvas > 0).astype(np.uint8) * 255, kernel, iterations=1)

            base_pixels = canvas_b.astype(np.float32)[overlap]
            new_pixels = warped_a.astype(np.float32)[overlap]
            mean_base = base_pixels.mean(axis=0)
            mean_new = new_pixels.mean(axis=0)
            std_base = base_pixels.std(axis=0)
            std_new = new_pixels.std(axis=0)
            scale = np.ones(3, dtype=np.float32)
            valid = std_new > 1.0
            scale[valid] = std_base[valid] / np.maximum(std_new[valid], 1e-3)
            scale = np.clip(scale, 0.7, 1.4)
            shift = mean_base - scale * mean_new
            warped_adj = np.clip(
                warped_a.astype(np.float32) * scale[None, None, :] + shift[None, None, :],
                0,
                255,
            )

            mask_a_u8 = mask_a_core
            mask_b_u8 = mask_b_core
            dist_a = cv2.distanceTransform(mask_a_u8, cv2.DIST_L2, 5)
            dist_b = cv2.distanceTransform(mask_b_u8, cv2.DIST_L2, 5)
            dist_a = cv2.GaussianBlur(dist_a, (0, 0), sigmaX=4, sigmaY=4)
            dist_b = cv2.GaussianBlur(dist_b, (0, 0), sigmaX=4, sigmaY=4)
            denom = dist_a + dist_b + 1e-6
            w_a = dist_a / denom
            w_b = 1.0 - w_a
            blended = warped_adj * w_a[..., None] + canvas_b.astype(np.float32) * w_b[..., None]

            # If the overlap is wide, apply a simple two-level multiband blend to retain contrast
            if np.count_nonzero(overlap) > 8000:
                sigma = 6
                warped_blur = cv2.GaussianBlur(warped_adj, (0, 0), sigma)
                canvas_blur = cv2.GaussianBlur(canvas_b.astype(np.float32), (0, 0), sigma)
                warped_detail = warped_adj - warped_blur
                canvas_detail = canvas_b.astype(np.float32) - canvas_blur
                blended = (warped_blur * w_a[..., None] + canvas_blur * w_b[..., None]
                           + warped_detail * w_a[..., None] + canvas_detail * w_b[..., None])

            out[overlap] = blended[overlap]

        stitched_out = Path(f"{a_path.stem}_{b_path.stem}_stitched.jpg")
        cv2.imwrite(str(stitched_out), np.clip(out, 0, 255).astype(np.uint8))
        inlier_ct = int(inlier_mask.sum()) if inlier_mask is not None else len(filt_matches)
        print('Saved stitch', stitched_out, 'inliers=', inlier_ct)

    return out, stitched_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pairs', nargs='+', help='Pairs like 42:44 or full paths')
    ap.add_argument('--dir', default='aligned_cropped')
    ap.add_argument('--mask-suffix', default='_mask.png')
    ap.add_argument('--rectify', action='store_true', help='Use rectified directory (overrides --dir with aligned_rectified)')
    args = ap.parse_args()

    base = Path('aligned_rectified') if args.rectify else Path(args.dir)
    for p in args.pairs:
        if ':' in p:
            a, b = p.split(':', 1)
            a_path = base / f"{a}_aligned.jpg"
            b_path = base / f"{b}_aligned.jpg"
        else:
            parts = p.split(',')
            if len(parts) >= 2:
                a_path = base / parts[0]
                b_path = base / parts[1]
            else:
                continue
        a_mask_path = base / f"{a_path.stem}_mask.png"
        b_mask_path = base / f"{b_path.stem}_mask.png"
        if not a_mask_path.exists():
            a_mask_path = None
        if not b_mask_path.exists():
            b_mask_path = None
        run_pair(a_path, b_path, a_mask=a_mask_path, b_mask=b_mask_path)


if __name__ == '__main__':
    main()
