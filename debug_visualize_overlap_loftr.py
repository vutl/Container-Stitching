#!/usr/bin/env python3
"""Visualize overlap for consecutive pairs using LoFTR with mask cropping.

Produces per-pair outputs:
 - {i}_{j}_matches.png : side-by-side with inlier match lines
 - {i}_{j}_warped.png  : canvas with warped image2 pasted
 - {i}_{j}_masks.png   : mask_old, mask_new (warped), overlap

Usage:
 python3 debug_visualize_overlap_loftr.py --dir rectified_c1 --start 54 --end 72 --weights EfficientLoFTR/weights/eloftr_outdoor.ckpt --out debug_overlap_loftr

"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import os

# Import helpers from test3_sides
from test3_sides import read_corners, mask_from_corners, init_loftr, run_loftr_match


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def draw_matches_side_by_side(img0, img1, pts0, pts1, inlier_mask):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    H = max(h0, h1)
    canvas = np.zeros((H, w0 + w1, 3), dtype=np.uint8)
    canvas[:h0, :w0] = img0 if img0.ndim == 3 else cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    canvas[:h1, w0:w0+w1] = img1 if img1.ndim == 3 else cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    # draw inlier lines
    for (x0,y0),(x1,y1),inl in zip(pts0.tolist(), pts1.tolist(), inlier_mask.tolist()):
        if not inl:
            continue
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1 + w0)), int(round(y1)))
        cv2.circle(canvas, p0, 2, (0,255,0), -1)
        cv2.circle(canvas, p1, 2, (0,255,0), -1)
        cv2.line(canvas, p0, p1, (0,200,255), 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', required=True, help='Directory with rectified images and corners')
    ap.add_argument('--start', type=int, required=True)
    ap.add_argument('--end', type=int, required=True)
    ap.add_argument('--weights', type=str, required=True, help='LoFTR checkpoint')
    ap.add_argument('--out', default='debug_overlap_loftr')
    ap.add_argument('--inlier-thresh', type=float, default=12.0)
    ap.add_argument('--min-inliers', type=int, default=3)
    args = ap.parse_args()

    rect_dir = Path(args.dir)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    # init loftr
    init_loftr(args.weights)

    for i in range(args.start, args.end):
        j = i + 1
        img0_p = rect_dir / f"{i}_aligned.jpg"
        img1_p = rect_dir / f"{j}_aligned.jpg"
        c0_p = rect_dir / f"{i}_aligned_corners.txt"
        c1_p = rect_dir / f"{j}_aligned_corners.txt"
        if not img0_p.exists() or not img1_p.exists():
            print(f"Missing pair: {img0_p} or {img1_p}")
            continue
        img0 = cv2.imread(str(img0_p))
        img1 = cv2.imread(str(img1_p))
        if img0 is None or img1 is None:
            print(f"Failed to read images {img0_p} or {img1_p}")
            continue

        corners0 = read_corners(c0_p)
        corners1 = read_corners(c1_p)
        mask0 = mask_from_corners(img0.shape[:2], corners0, pad=0)
        mask1 = mask_from_corners(img1.shape[:2], corners1, pad=0)
        if mask0 is None or mask1 is None:
            print(f"No corners for {i} or {j}")
            continue

        try:
            mkpts0, mkpts1, mconf = run_loftr_match(img0, img1, mask0, mask1)
        except Exception as e:
            print(f"LoFTR error for {i}->{j}: {e}")
            continue

        if mkpts0 is None or len(mkpts0) < args.min_inliers:
            print(f"LoFTR found too few matches for {i}->{j}: {None if mkpts0 is None else len(mkpts0)}")
            continue

        deltas = mkpts1 - mkpts0
        dx_med = float(np.median(deltas[:,0]))
        dy_med = float(np.median(deltas[:,1]))
        residuals = deltas - np.array([dx_med, dy_med], dtype=np.float32)
        dist = np.linalg.norm(residuals, axis=1)
        inliers = dist <= args.inlier_thresh
        nin = int(np.count_nonzero(inliers))
        if nin < args.min_inliers:
            print(f"Not enough inliers after filtering for {i}->{j}: {nin}")
            continue
        dx = float(np.median(deltas[inliers,0]))
        dy = float(np.median(deltas[inliers,1]))

        print(f"{i}->{j}: matches={len(mkpts0)} inliers={nin} dx={dx:.2f} dy={dy:.2f}")

        # draw matches image
        match_img = draw_matches_side_by_side(img0, img1, mkpts0, mkpts1, inliers)
        cv2.imwrite(str(out_dir / f"{i}_{j}_matches.png"), match_img)

        # Build canvas by placing img0 at (0,0) and warping img1 by translation H
        h0,w0 = img0.shape[:2]
        h1,w1 = img1.shape[:2]
        # H maps img1 -> img0 coords: pts0 + (dx,dy) = pts1 => H should move img1 by -dx? Wait: mkpts1 - mkpts0 = delta from img0->img1
        # We want to warp img1 into img0 coordinates: compute H so that point in img1 at (x,y) goes to (x - dx, y - dy) in img0 coords
        H = np.array([[1.0,0.0,-dx],[0.0,1.0,-dy],[0.0,0.0,1.0]], dtype=np.float64)

        # compute warped corners of img1
        corners = np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32)
        warped = cv2.perspectiveTransform(corners[None,:,:], H)[0]
        x_min = int(np.floor(min(0, warped[:,0].min())))
        x_max = int(np.ceil(max(w0, warped[:,0].max())))
        y_min = int(np.floor(min(0, warped[:,1].min())))
        y_max = int(np.ceil(max(h0, warped[:,1].max())))
        canvas_w = x_max - x_min
        canvas_h = y_max - y_min
        offset_x = -x_min
        offset_y = -y_min

        T = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)
        H_to_canvas = T @ H

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[offset_y:offset_y+h0, offset_x:offset_x+w0] = img0
        warped_img1 = cv2.warpPerspective(img1, H_to_canvas, (canvas_w, canvas_h))
        cv2.imwrite(str(out_dir / f"{i}_{j}_warped.png"), warped_img1)

        # masks
        full_mask1 = np.ones((h1,w1), dtype=np.uint8)*255
        mask1_warp = cv2.warpPerspective(full_mask1, H_to_canvas, (canvas_w, canvas_h))
        mask0_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        mask0_canvas[offset_y:offset_y+h0, offset_x:offset_x+w0] = (mask0>0).astype(np.uint8)*255
        overlap = cv2.bitwise_and(mask0_canvas, mask1_warp)

        # Compose mask visualization
        vis_masks = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        # old mask = blue channel
        vis_masks[mask0_canvas>0, 0] = 255
        # new mask (warped) = green channel
        vis_masks[mask1_warp>0, 1] = 255
        # overlap = red channel
        vis_masks[overlap>0, 2] = 255
        cv2.imwrite(str(out_dir / f"{i}_{j}_masks.png"), vis_masks)

        # also write overlap overlay on canvas
        overlay = canvas.copy()
        # tint warped_img1 red where overlap
        red = warped_img1.copy()
        red[overlap==0] = 0
        overlay = cv2.addWeighted(overlay, 0.7, warped_img1, 0.3, 0)
        # draw overlap boundary
        ys, xs = np.where(overlap>0)
        if len(ys)>0:
            y0b, y1b = ys.min(), ys.max()
            x0b, x1b = xs.min(), xs.max()
            cv2.rectangle(overlay, (x0b,x0b), (x1b,y1b), (0,0,255), 2)
        cv2.imwrite(str(out_dir / f"{i}_{j}_overlap.png"), overlay)

    print(f"Wrote outputs to {out_dir}")

if __name__ == '__main__':
    main()
