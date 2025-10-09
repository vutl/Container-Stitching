#!/usr/bin/env python3
"""
Stitch aligned images using an incremental strategy and masks from saved
4-corner bounding boxes. This variant avoids any Gaussian blur or
multiband pyramid blending: overlap blending uses raw distance transforms
only (no smoothing).
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


def expand_canvas(mosaic: np.ndarray,
                  h_to_canvas: np.ndarray,
                  new_img: np.ndarray,
                  max_size: int = 14000,
                  use_multiband: bool = True,
                  seam_trim: int = 0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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
    # debug logging to trace small off-by-one differences
    print(f"DEBUG expand_canvas: x_min={x_min} y_min={y_min} x_max={x_max} y_max={y_max} tx={tx} ty={ty} -> new_w={new_w} new_h={new_h}")
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
    # optionally trim seam by cropping 1..N columns from the left/right edges
    if seam_trim and seam_trim > 0:
        try:
            ys, xs = np.where(warped_mask_new > 0)
            if xs.size > 0:
                x0 = int(xs.min())
                x1 = int(xs.max())
                trim = int(seam_trim)
                # only trim if bbox is wider than 2*trim to avoid removing whole region
                if (x1 - x0 + 1) > 2 * trim:
                    # zero-out the left trim columns and right trim columns
                    l0 = max(0, x0)
                    l1 = min(warped_mask_new.shape[1], x0 + trim)
                    r0 = max(0, x1 - trim)
                    r1 = min(warped_mask_new.shape[1], x1)
                    warped_mask_new[:, l0:l1] = 0
                    warped_mask_new[:, r0:r1] = 0
        except Exception:
            # fallback: do nothing if any error
            pass
    mask_new = warped_mask_new > 0

    mask_mosaic = np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = moved_mask > 0

    out = moved.copy()
    only_new = mask_new & (~mask_old)
    overlap = mask_new & mask_old

    if np.any(only_new):
        out[only_new] = warped_new[only_new]
    if np.any(overlap):
        # Direct paste: new warped image overwrites overlap (no blending/smoothing)
        out[overlap] = warped_new[overlap]

    # return moved/warped output, translation T, and boolean masks (old, new)
    return out, T, mask_old, mask_new


def crop_final_mosaic_from_mask(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # mask is boolean or uint8 (True/255 where content exists)
    if mask is None:
        return crop_final_mosaic(mosaic)
    if mask.dtype != np.uint8:
        mask_u8 = (mask > 0).astype(np.uint8) * 255
    else:
        mask_u8 = mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mosaic
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return mosaic[y:y + h, x:x + w]


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
                                    ecc_eps: float = 1e-4,
                                    use_multiband: bool = True,
                                    seam_trim: int = 0) -> Optional[Tuple[np.ndarray, np.ndarray, list]]:
    if not image_paths:
        return None
    # normalize all input images to a common target height (median) to avoid
    # tiny off-by-one differences introduced by perspective warps and rounding.
    heights = []
    for p in image_paths:
        tmp = cv2.imread(str(p))
        if tmp is not None:
            heights.append(tmp.shape[0])
    if not heights:
        return None
    target_h = int(np.median(heights))

    def _normalize_height(img: np.ndarray, tgt: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h == tgt:
            return img
        if h > tgt:
            # crop vertically centered
            off = (h - tgt) // 2
            return img[off:off + tgt, :]
        # h < tgt: pad equally top/bottom (bottom gets extra if odd)
        pad_top = (tgt - h) // 2
        pad_bot = tgt - h - pad_top
        return cv2.copyMakeBorder(img, pad_top, pad_bot, 0, 0, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])
    base = _normalize_height(base, target_h)
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
    # track a boolean mosaic mask to avoid tiny interpolation artifacts
    mosaic_mask = (base_mask > 0)
    # collect per-frame vertical bounds (y0,y1) in mosaic coords
    y_bounds = []
    nz = cv2.findNonZero((base_mask > 0).astype(np.uint8))
    if nz is not None:
        x, y, ww, hh = cv2.boundingRect(nz)
        y_bounds.append((y, y + hh))

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
        cur_img = _normalize_height(cur_img, target_h)

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
            # If the detected bbox is very narrow (head-only), expand horizontally
            # to full image width while preserving vertical bounds. This helps SIFT
            # matching when only the container head is detected and background
            # left/right have no useful features.
            try:
                h_img, w_img = cur_img.shape[:2]
                if cur_mask is not None:
                    ys, xs = np.where(cur_mask > 0)
                    if ys.size > 0 and xs.size > 0:
                        x0, x1 = int(xs.min()), int(xs.max())
                        bbox_w = x1 - x0 + 1
                        # threshold: consider narrow if bbox occupies < 85% of width
                        if float(bbox_w) < 0.85 * float(w_img):
                            y0, y1 = int(ys.min()), int(ys.max())
                            new_mask = np.zeros_like(cur_mask)
                            new_mask[y0:y1 + 1, 0:w_img] = 255
                            cur_mask = new_mask
            except Exception:
                # fallback: keep original mask
                pass
        # Corners are assumed rectified and correct at this stage; proceed.

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
        mosaic_mask_u8 = (mosaic_mask.astype(np.uint8) * 255)

        # Estimate translation between current frame and mosaic to avoid curvature
        H = estimate_translation_masked(cur_img, mosaic, cur_mask, mosaic_mask_u8)

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
        # Lock vertical motion and round to integer pixels to avoid tiny drifts
        H[1, 2] = 0.0
        H[0, 2] = float(int(round(H[0, 2])))
        H[1, 2] = float(int(round(H[1, 2])))

        # Expand canvas and warp current frame into it
        mosaic_new, T, mask_old, mask_new = expand_canvas(mosaic, H, cur_img, use_multiband=use_multiband, seam_trim=seam_trim)
        if mosaic_new is None:
            print("  -> Canvas overflow; skipping frame")
            continue

        # update mosaic and mosaic_mask (boolean)
        mosaic = mosaic_new
        if mask_old is not None and mask_new is not None:
            # mask_old and mask_new are boolean arrays of size mosaic
            mosaic_mask = mask_old | mask_new
        else:
            # fallback: recompute from mosaic content
            mosaic_mask = (cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY) > 0)
        # record vertical bounds of the new warped mask in mosaic coords
        try:
            nz2 = cv2.findNonZero((mask_new.astype(np.uint8)))
            if nz2 is not None:
                x2, y2, w2, h2 = cv2.boundingRect(nz2)
                y_bounds.append((y2, y2 + h2))
        except Exception:
            pass
        print(f"  -> Stitched; canvas {mosaic.shape[1]}x{mosaic.shape[0]}")

    return mosaic, mosaic_mask, y_bounds


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
    ap.add_argument('--seam-trim', type=int, default=0, help='Horizontally trim new warped masks by N pixels to hide seams')
    ap.add_argument('--no-force-rect', action='store_true', help="Disable rectangle fill heuristic")
    ap.add_argument('--out', type=str, default='aligned', help='Output directory for panorama')
    ap.add_argument('--no-multiband', action='store_true', help='Disable multiband pyramid blending')
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

    res = stitch_incremental_with_corners(
        image_paths,
        corner_paths,
        stride=max(1, args.stride),
        mask_pad=max(0, args.mask_pad),
        force_rect=not args.no_force_rect,
        use_multiband=(not args.no_multiband),
        seam_trim=max(0, args.seam_trim),
    )
    if res is None:
        print('Stitching failed.')
        return
    mosaic, mosaic_mask, y_bounds = res

    first_idx, last_idx = ordered[0], ordered[-1]
    out_path = out_dir / f"panorama_{first_idx}_{last_idx}_bbox.jpg"
    # use the final mosaic mask and apply a small morphological closing to
    # remove single-pixel fringes that cause off-by-one cropping differences
    mask_u8 = (mosaic_mask.astype(np.uint8) * 255)
    if mask_u8 is not None and mask_u8.size > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask_closed = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            final = mosaic[y:y + h, x:x + w]
        else:
            final = crop_final_mosaic_from_mask(mosaic, mosaic_mask)
    else:
        final = crop_final_mosaic_from_mask(mosaic, mosaic_mask)
    cv2.imwrite(str(out_path), final)
    print(f'Saved -> {out_path}')


if __name__ == '__main__':
    main()
