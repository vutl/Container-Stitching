#!/usr/bin/env python3
import numpy as np
import cv2
from pathlib import Path
import argparse
from test3_sides import read_corners, mask_from_corners, init_loftr, run_loftr_match


def analyze_pair(img0_p, img1_p, c0_p, c1_p, matcher_weights, inlier_thresh=12.0):
    img0 = cv2.imread(str(img0_p))
    img1 = cv2.imread(str(img1_p))
    if img0 is None or img1 is None:
        return None
    corners0 = read_corners(Path(c0_p))
    corners1 = read_corners(Path(c1_p))
    mask0 = mask_from_corners(img0.shape[:2], corners0, pad=0)
    mask1 = mask_from_corners(img1.shape[:2], corners1, pad=0)
    if mask0 is None or mask1 is None:
        return None
    # ensure init
    init_loftr(matcher_weights)
    mk0, mk1, mconf = run_loftr_match(img0, img1, mask0, mask1)
    if mk0 is None or len(mk0) == 0:
        return None
    deltas = mk1 - mk0
    dxs = deltas[:,0]
    dys = deltas[:,1]
    med_dx = float(np.median(dxs))
    med_dy = float(np.median(dys))
    mean_dx = float(np.mean(dxs))
    mean_dy = float(np.mean(dys))
    std_dx = float(np.std(dxs))
    std_dy = float(np.std(dys))
    # residuals relative to median
    residuals = deltas - np.array([med_dx, med_dy], dtype=np.float32)
    dist = np.linalg.norm(residuals, axis=1)
    inliers_mask = dist <= inlier_thresh
    nin = int(np.count_nonzero(inliers_mask))
    n = len(dxs)
    # angle distribution
    angles = np.degrees(np.arctan2(dys, dxs))
    ang_mean = float(np.mean(angles))
    ang_std = float(np.std(angles))
    # length of vectors
    lengths = np.linalg.norm(deltas, axis=1)
    med_len = float(np.median(lengths))
    pct_within_5pct = float((np.abs(lengths - med_len) <= 0.05*med_len).sum())/n
    pct_within_10pct = float((np.abs(lengths - med_len) <= 0.10*med_len).sum())/n
    # vertical spread of matched points
    y0_std = float(np.std(mk0[:,1]))
    y1_std = float(np.std(mk1[:,1]))

    return {
        'n_matches': n,
        'n_inliers': nin,
        'med_dx': med_dx,
        'med_dy': med_dy,
        'mean_dx': mean_dx,
        'mean_dy': mean_dy,
        'std_dx': std_dx,
        'std_dy': std_dy,
        'ang_mean': ang_mean,
        'ang_std': ang_std,
        'med_len': med_len,
        'pct_len_5pct': pct_within_5pct,
        'pct_len_10pct': pct_within_10pct,
        'y0_std': y0_std,
        'y1_std': y1_std
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rect-dir', required=True)
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--inlier-thresh', type=float, default=12.0)
    args = ap.parse_args()

    rect = Path(args.rect_dir)
    init_loftr(args.weights)
    rows = []
    for i in range(args.start, args.end+1):
        j = i+1
        img0 = rect / f"{i}_aligned.jpg"
        img1 = rect / f"{j}_aligned.jpg"
        c0 = rect / f"{i}_aligned_corners.txt"
        c1 = rect / f"{j}_aligned_corners.txt"
        if not img0.exists() or not img1.exists():
            print(f"Missing: {img0} or {img1}")
            continue
        stat = analyze_pair(img0, img1, c0, c1, args.weights, inlier_thresh=args.inlier_thresh)
        if stat is None:
            print(f"No stats for {i}->{j}")
            continue
        rows.append((i,j,stat))
        print(f"{i}->{j}: n={stat['n_matches']} nin={stat['n_inliers']} med_dx={stat['med_dx']:.2f} std_dx={stat['std_dx']:.2f} ang_std={stat['ang_std']:.2f} pct_len10={stat['pct_len_10pct']*100:.1f}% y0_std={stat['y0_std']:.1f}")

    # summary
    print('\nSUMMARY:')
    for (i,j,stat) in rows:
        print(f"{i}->{j}: matches={stat['n_matches']}, inliers={stat['n_inliers']}, med_dx={stat['med_dx']:.2f}, std_dx={stat['std_dx']:.2f}, ang_std={stat['ang_std']:.2f}, pct_len10={stat['pct_len_10pct']*100:.1f}%, y0_std={stat['y0_std']:.1f}")

if __name__ == '__main__':
    main()
