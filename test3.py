#!/usr/bin/env python3
"""
Stitch aligned images using the same incremental strategy as stitch_color_auto,
but build the feature masks from saved 4-corner bounding boxes instead of color
heuristics.
"""

import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np


# -------------------- helpers --------------------

def parse_range(spec: str) -> List[int]:
    parts = spec.replace(',', ' ').split()
    out: List[int] = []
    for p in parts:
        if '-' in p:
            a, b = p.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def read_corners(path: Path) -> Dict[str, Tuple[float, float]]:
    pts: Dict[str, Tuple[float, float]] = {}
    if not path.exists():
        return pts
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        q = parts[0]
        try:
            x = float(parts[1])
            y = float(parts[2])
        except Exception:
            continue
        pts[q] = (x, y)
    return pts


def mask_from_corners(shape: Tuple[int, int],
                      corners: Dict[str, Tuple[float, float]],
                      pad: int = 0) -> Optional[np.ndarray]:
    if not corners:
        return None
    xs = []
    ys = []
    for key in ('TL', 'TR', 'BR', 'BL'):
        if key not in corners:
            return None
        x, y = corners[key]
        xs.append(float(x))
        ys.append(float(y))
    h, w = shape
    x0 = max(0, int(np.floor(min(xs))) - pad)
    x1 = min(w, int(np.ceil(max(xs))) + pad)
    y0 = max(0, int(np.floor(min(ys))) - pad)
    y1 = min(h, int(np.ceil(max(ys))) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def rect_fill_if_close(mask: np.ndarray, min_fill_ratio: float = 0.72) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    areas = [cv2.contourArea(c) for c in cnts]
    idx = int(np.argmax(areas))
    cnt = cnts[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = max(1, w * h)
    comp_area = cv2.contourArea(cnt)
    ratio = float(comp_area) / float(rect_area)
    if ratio >= min_fill_ratio:
        filled = np.zeros_like(mask)
        filled[y:y + h, x:x + w] = 255
        return filled
    return mask


def get_detector():
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT'
    if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT'
    return cv2.ORB_create(4000), 'ORB'


def estimate_transform_affine_masked(src: np.ndarray,
                                     tar: np.ndarray,
                                     mask_src: np.ndarray,
                                     mask_tar: np.ndarray) -> Optional[np.ndarray]:
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    det, det_name = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask_src)
    kp2, des2 = det.detectAndCompute(gray2, mask_tar)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    norm = cv2.NORM_L2 if det_name == 'SIFT' else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    pairs = bf.knnMatch(des1, des2, 2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    A, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None:
        return None
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A
    return H


def estimate_translation_masked(src: np.ndarray,
                                 tar: np.ndarray,
                                 mask_src: Optional[np.ndarray],
                                 mask_tar: Optional[np.ndarray],
                                 inlier_thresh: float = 12.0) -> Optional[np.ndarray]:
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    det, det_name = get_detector()
    mask_src_u8 = mask_src if mask_src is not None else np.ones(gray1.shape, dtype=np.uint8) * 255
    mask_tar_u8 = mask_tar if mask_tar is not None else np.ones(gray2.shape, dtype=np.uint8) * 255

    kp1, des1 = det.detectAndCompute(gray1, mask_src_u8)
    kp2, des2 = det.detectAndCompute(gray2, mask_tar_u8)
    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return None

    norm = cv2.NORM_L2 if det_name == 'SIFT' else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, 2)
    good = []
    for pair in raw:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 3:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    deltas = pts2 - pts1
    dx = float(np.median(deltas[:, 0]))
    dy = float(np.median(deltas[:, 1]))

    residuals = deltas - np.array([dx, dy], dtype=np.float32)
    dist = np.linalg.norm(residuals, axis=1)
    inliers = dist <= inlier_thresh
    if np.count_nonzero(inliers) >= 3:
        dx = float(np.median(deltas[inliers, 0]))
        dy = float(np.median(deltas[inliers, 1]))

    H = np.eye(3, dtype=np.float32)
    H[0, 2] = dx
    H[1, 2] = dy
    return H


def is_reasonable_transform(H: Optional[np.ndarray],
                            max_scale: float = 1.8,
                            min_scale: float = 0.55) -> bool:
    if H is None or not np.isfinite(H).all():
        return False
    a, b, c, d = H[0, 0], H[0, 1], H[1, 0], H[1, 1]
    s = np.sqrt(max(1e-8, (a * a + b * b + c * c + d * d) / 2.0))
    return min_scale <= s <= max_scale


def expand_canvas(mosaic: np.ndarray,
                  h_to_canvas: np.ndarray,
                  new_img: np.ndarray,
                  max_size: int = 14000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = new_img.shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners, h_to_canvas)

    h_canvas, w_canvas = mosaic.shape[:2]
    base = np.float32([[0, 0], [w_canvas, 0], [w_canvas, h_canvas], [0, h_canvas]]).reshape(-1, 1, 2)

    all_pts = np.concatenate([base, warped_corners], axis=0)
    x_min, y_min = np.floor(all_pts.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_pts.max(axis=0).ravel()).astype(int)
    if not np.isfinite([x_min, y_min, x_max, y_max]).all():
        return None, None

    tx = -min(0, x_min)
    ty = -min(0, y_min)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
    new_w = int(x_max + tx)
    new_h = int(y_max + ty)
    if new_w <= 0 or new_h <= 0 or new_w > max_size or new_h > max_size:
        return None, None

    moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    h_final = T @ h_to_canvas
    warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask_src = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask_new = cv2.warpPerspective(mask_src, h_final, (new_w, new_h), flags=cv2.INTER_NEAREST,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_new = warped_mask_new > 0

    mask_mosaic = np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = moved_mask > 0

    out = moved.copy()
    only_new = mask_new & (~mask_old)
    overlap = mask_new & mask_old

    # if np.any(overlap):
    #     base_pixels = moved[overlap].astype(np.float32)
    #     new_pixels = warped_new[overlap].astype(np.float32)
    #     mean_base = base_pixels.mean(axis=0)
    #     mean_new = new_pixels.mean(axis=0)
    #     std_base = base_pixels.std(axis=0)
    #     std_new = new_pixels.std(axis=0)
    #     scale = np.ones(3, dtype=np.float32)
    #     valid = std_new > 1.0
    #     scale[valid] = std_base[valid] / np.maximum(std_new[valid], 1e-3)
    #     scale = np.clip(scale, 0.6, 1.6)
    #     shift = mean_base - scale * mean_new
    #     warped_new = np.clip(
    #         warped_new.astype(np.float32) * scale[None, None, :] + shift[None, None, :],
    #         0,
    #         255,
    #     ).astype(np.uint8)

    if np.any(only_new):
        out[only_new] = warped_new[only_new]
    if np.any(overlap):
        mask_new_u8 = warped_mask_new.astype(np.uint8)
        mask_old_u8 = moved_mask.astype(np.uint8)
        dist_new = cv2.distanceTransform(mask_new_u8, cv2.DIST_L2, 5).astype(np.float32)
        dist_old = cv2.distanceTransform(mask_old_u8, cv2.DIST_L2, 5).astype(np.float32)
        dist_new = cv2.GaussianBlur(dist_new, (0, 0), sigmaX=6, sigmaY=6)
        dist_old = cv2.GaussianBlur(dist_old, (0, 0), sigmaX=6, sigmaY=6)
        denom = dist_new + dist_old + 1e-6
        w_new = dist_new / denom
        w_old = 1.0 - w_new
        blended = moved.astype(np.float32) * w_old[..., None] + warped_new.astype(np.float32) * w_new[..., None]
        out[overlap] = np.clip(blended, 0, 255).astype(np.uint8)[overlap]

    return out, T


def crop_final_mosaic(mosaic: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mosaic
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return mosaic[y:y + h, x:x + w]


# -------------------- stitching --------------------

def stitch_incremental_with_corners(image_paths: List[Path],
                                    corner_paths: List[Path],
                                    stride: int = 2,
                                    mask_pad: int = 0,
                                    force_rect: bool = True,
                                    ecc_max_iter: int = 200,
                                    ecc_eps: float = 1e-4) -> Optional[np.ndarray]:
    if not image_paths:
        return None
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    # Initialize mosaic with first image
    mosaic = base.copy()
    G0 = np.eye(3, dtype=np.float32)  # Global transform for canvas tracking
    G = [np.eye(3, dtype=np.float32)]  # Individual transforms
    
    # Get base mask for reference
    base_corners = read_corners(corner_paths[0])
    base_mask = mask_from_corners(base.shape[:2], base_corners, pad=mask_pad)
    if base_mask is None:
        print("Warning: No base mask found, using full image")
        base_mask = np.ones(base.shape[:2], dtype=np.uint8) * 255
    if force_rect:
        base_mask = rect_fill_if_close(base_mask)

    n = len(image_paths)
    idxs = list(range(0, n, max(1, stride)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    # Process each subsequent frame
    for k in range(1, len(idxs)):
        cur_idx = idxs[k]
        cur_path = image_paths[cur_idx]
        print(f"Processing {cur_path.name} -> mosaic")

        cur_img = cv2.imread(str(cur_path))
        if cur_img is None:
            print("  -> Skip missing frame")
            continue

        # Get current frame mask: prefer precomputed mask PNG (<stem>_mask.png), else corners bbox
        cur_img_path = cur_path
        mask_path = cur_img_path.with_name(cur_img_path.stem + '_mask.png')
        cur_mask = None
        if mask_path.exists():
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape[:2] == cur_img.shape[:2]:
                cur_mask = (m > 0).astype(np.uint8) * 255
        if cur_mask is None:
            cur_corners = read_corners(corner_paths[cur_idx])
            cur_mask = mask_from_corners(cur_img.shape[:2], cur_corners, pad=mask_pad)
        if cur_mask is None:
            print("  -> Missing corners; skipping")
            continue
        if force_rect:
            cur_mask = rect_fill_if_close(cur_mask)
        # Exclude black columns/pixels from the mask (avoid blending black borders)
        cur_valid = (cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY) > 4).astype(np.uint8) * 255
        if cur_mask is None:
            cur_mask = cur_valid
        else:
            cur_mask = cv2.bitwise_and(cur_mask, cur_valid)

        # Create mosaic mask for feature matching (use union of all previous content)
        mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        mosaic_mask = (mosaic_gray > 0).astype(np.uint8) * 255

        # Estimate translation between current frame and mosaic to avoid curvature
        H = estimate_translation_masked(cur_img, mosaic, cur_mask, mosaic_mask)

        # Fallback to ECC if feature matching fails
        if H is None:
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY),
                    warp,
                    cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ecc_max_iter, ecc_eps),
                    inputMask=cur_mask,
                    gaussFiltSize=5,
                )
                H = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            except cv2.error:
                print("  -> ECC failed; skipping frame")
                continue

        # Ensure transform is pure translation (zero out any numerical drift)
        H[0, 0] = 1.0; H[0, 1] = 0.0
        H[1, 0] = 0.0; H[1, 1] = 1.0

        # Expand canvas and warp current frame into it
        mosaic_new, T = expand_canvas(mosaic, H, cur_img)
        if mosaic_new is None:
            print("  -> Canvas overflow; skipping frame")
            continue

        mosaic = mosaic_new
        print(f"  -> Stitched; canvas {mosaic.shape[1]}x{mosaic.shape[0]}")

    return mosaic


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--aligned-dir', type=str, default='aligned')
    ap.add_argument('--indices', type=str, required=True, help='e.g., "42-48" or "42 43 44"')
    ap.add_argument('--ref-index', type=int, default=None, help='Reference index to start the mosaic')
    ap.add_argument('--corners-dir', type=str, default='aligned')
    ap.add_argument('--corners-suffix', type=str, default='_aligned_corners.txt')
    ap.add_argument('--stride', type=int, default=2, help='Process every Nth frame (default: 2)')
    ap.add_argument('--mask-pad', type=int, default=0, help='Extra pixels added to bounding box masks')
    ap.add_argument('--no-force-rect', action='store_true', help="Disable rectangle fill heuristic")
    ap.add_argument('--out', type=str, default='aligned', help='Output directory for panorama')
    args = ap.parse_args()

    aligned_dir = Path(args.aligned_dir)
    corners_dir = Path(args.corners_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = parse_range(args.indices)
    if not idxs:
        print('No indices provided')
        return

    if args.ref_index is not None:
        if args.ref_index not in idxs:
            idxs.insert(0, args.ref_index)
        ordered = [args.ref_index] + [i for i in idxs if i != args.ref_index]
    else:
        ordered = sorted(idxs)

    image_paths = [aligned_dir / f"{i}_aligned.jpg" for i in ordered]
    corner_paths = [corners_dir / f"{i}{args.corners_suffix}" for i in ordered]

    for p in image_paths:
        if not p.exists():
            print(f"Missing image: {p}")
            return
    for c in corner_paths:
        if not c.exists():
            print(f"Missing corners: {c}")
            return

    mosaic = stitch_incremental_with_corners(
        image_paths,
        corner_paths,
        stride=max(1, args.stride),
        mask_pad=max(0, args.mask_pad),
        force_rect=not args.no_force_rect,
    )
    if mosaic is None:
        print('Stitching failed.')
        return

    first_idx, last_idx = ordered[0], ordered[-1]
    out_path = out_dir / f"panorama_{first_idx}_{last_idx}_bbox.jpg"
    final = crop_final_mosaic(mosaic)
    cv2.imwrite(str(out_path), final)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()