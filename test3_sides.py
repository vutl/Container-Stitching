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
import sys
import torch
import matplotlib.pyplot as plt

from blending import ImageBlender, BlendResult


# Globals for optional LoFTR matcher (initialized lazily)
_LOFTR_MATCHER = None
_LOFTR_PRECISION = 'fp32'
_LOFTR_WEIGHTS = None
_FORCE_DET = None


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
    # Allow forcing a detector via global _FORCE_DET set by main()
    forced = globals().get('_FORCE_DET', None)
    if forced is not None:
        f = str(forced).lower()
        if f == 'loftr' or f == 'eloftr':
            return None, 'LOFTR'
        if f == 'sift' and hasattr(cv2, 'SIFT_create'):
            return cv2.SIFT_create(), 'SIFT'
        if f == 'akaze' and hasattr(cv2, 'AKAZE_create'):
            return cv2.AKAZE_create(), 'AKAZE'
        if f == 'kaze' and hasattr(cv2, 'KAZE_create'):
            return cv2.KAZE_create(), 'KAZE'
        if f == 'orb':
            return cv2.ORB_create(4000), 'ORB'

    # Default behaviour: prefer SIFT if available, else ORB
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT'
    if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT'
    return cv2.ORB_create(4000), 'ORB'


def init_loftr(weights_path: Optional[str], model_type: str = 'full', precision: str = 'fp32'):
    """Initialize EfficientLoFTR matcher from checkpoint path.
    weights_path: path to eloftr_outdoor.ckpt (string or Path)
    model_type: 'full' or 'opt'
    precision: 'fp32', 'mp', or 'fp16'
    Returns the matcher (on CUDA if available).
    """
    global _LOFTR_MATCHER, _LOFTR_PRECISION
    if weights_path is None:
        raise ValueError('weights_path must be provided to init LoFTR')
    if _LOFTR_MATCHER is not None:
        return _LOFTR_MATCHER

    # Add EfficientLoFTR package folder to sys.path if present
    base = Path(__file__).parent.resolve()
    eloftr_pkg = base / 'EfficientLoFTR'
    if str(eloftr_pkg) not in sys.path and eloftr_pkg.exists():
        sys.path.insert(0, str(eloftr_pkg))

    try:
        from EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter
    except Exception as e:
        raise ImportError('Cannot import LoFTR from EfficientLoFTR package: ' + str(e))

    # choose config
    if model_type == 'full':
        _cfg = full_default_cfg.copy()
    else:
        _cfg = opt_default_cfg.copy()

    if precision == 'mp':
        _cfg['mp'] = True
    elif precision == 'fp16':
        _cfg['half'] = True

    matcher = LoFTR(config=_cfg)

    ckpt_path = Path(weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'LoFTR checkpoint not found: {ckpt_path}')

    # Safe-loading: try weights_only if available
    try:
        ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    except TypeError:
        try:
            from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
            torch.serialization.add_safe_globals([ModelCheckpoint])
        except Exception:
            pass
        ckpt = torch.load(str(ckpt_path), map_location='cpu')

    state = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    def _strip_prefix(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    if isinstance(state, dict):
        if any(k.startswith('module.') for k in state.keys()):
            state = _strip_prefix(state, 'module.')
        elif any(k.startswith('model.') for k in state.keys()):
            state = _strip_prefix(state, 'model.')

    matcher.load_state_dict(state)
    matcher = reparameter(matcher)
    if precision == 'fp16':
        matcher = matcher.half()

    matcher = matcher.eval()
    if torch.cuda.is_available():
        matcher = matcher.cuda()

    _LOFTR_MATCHER = matcher
    _LOFTR_PRECISION = precision
    return _LOFTR_MATCHER


def run_loftr_match(img0: np.ndarray, img1: np.ndarray, mask0: Optional[np.ndarray] = None, mask1: Optional[np.ndarray] = None):
    """Run the initialized LoFTR matcher on two images with optional mask support.
    
    Args:
        img0, img1: RGB/grayscale images (H, W, 3) or (H, W)
        mask0, mask1: Optional binary masks (H, W) uint8. If provided, images are cropped to mask bbox before matching.
    
    Returns (mkpts0, mkpts1, mconf) as numpy arrays (Nx2, Nx2, N) in ORIGINAL image coordinates.
    """
    global _LOFTR_MATCHER
    if _LOFTR_MATCHER is None:
        raise RuntimeError('LoFTR matcher not initialized (call init_loftr)')

    # Crop to mask bbox if provided to exclude background
    x0_offset, y0_offset = 0, 0
    x1_offset, y1_offset = 0, 0
    
    if mask0 is not None and mask0.sum() > 0:
        ys, xs = np.where(mask0 > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        img0 = img0[y0:y1, x0:x1]
        x0_offset, y0_offset = x0, y0
    
    if mask1 is not None and mask1.sum() > 0:
        ys, xs = np.where(mask1 > 0)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        img1 = img1[y0:y1, x0:x1]
        x1_offset, y1_offset = x0, y0

    # LoFTR expects HxW grayscale input in [0,1] tensors, divisible by 32
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    nh0 = (h0 // 32) * 32
    nw0 = (w0 // 32) * 32
    nh1 = (h1 // 32) * 32
    nw1 = (w1 // 32) * 32
    nh0 = max(nh0, 32); nw0 = max(nw0, 32)
    nh1 = max(nh1, 32); nw1 = max(nw1, 32)

    img0c = cv2.resize(img0, (nw0, nh0)) if (nh0 != h0 or nw0 != w0) else img0
    img1c = cv2.resize(img1, (nw1, nh1)) if (nh1 != h1 or nw1 != w1) else img1

    if img0c.ndim == 3:
        img0c = cv2.cvtColor(img0c, cv2.COLOR_BGR2GRAY)
    if img1c.ndim == 3:
        img1c = cv2.cvtColor(img1c, cv2.COLOR_BGR2GRAY)

    t0 = torch.from_numpy(img0c)[None, None].float() / 255.0
    t1 = torch.from_numpy(img1c)[None, None].float() / 255.0
    if torch.cuda.is_available():
        t0 = t0.cuda(); t1 = t1.cuda()

    batch = {'image0': t0, 'image1': t1}
    with torch.no_grad():
        _LOFTR_MATCHER(batch)
    mkpts0 = batch['mkpts0_f'].cpu().numpy()
    mkpts1 = batch['mkpts1_f'].cpu().numpy()
    mconf = batch['mconf'].cpu().numpy()
    
    # Translate points back to original image coordinates
    if x0_offset != 0 or y0_offset != 0:
        mkpts0[:, 0] += x0_offset
        mkpts0[:, 1] += y0_offset
    if x1_offset != 0 or y1_offset != 0:
        mkpts1[:, 0] += x1_offset
        mkpts1[:, 1] += y1_offset
    
    return mkpts0, mkpts1, mconf


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

    # Use L2 for float descriptors (SIFT, KAZE), Hamming for binary (ORB, AKAZE)
    norm = cv2.NORM_L2 if det_name in ('SIFT', 'KAZE') else cv2.NORM_HAMMING
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


def estimate_transform_homography_masked(src: np.ndarray,
                                         tar: np.ndarray,
                                         mask_src: Optional[np.ndarray],
                                         mask_tar: Optional[np.ndarray]) -> Optional[np.ndarray]:
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    det, det_name = get_detector()
    ms = mask_src if mask_src is not None else np.ones(gray1.shape, dtype=np.uint8) * 255
    mt = mask_tar if mask_tar is not None else np.ones(gray2.shape, dtype=np.uint8) * 255
    kp1, des1 = det.detectAndCompute(gray1, ms)
    kp2, des2 = det.detectAndCompute(gray2, mt)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None
    # Use L2 for float descriptors (SIFT, KAZE), Hamming for binary (ORB, AKAZE)
    norm = cv2.NORM_L2 if det_name in ('SIFT', 'KAZE') else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    pairs = bf.knnMatch(des1, des2, 2)
    good = []
    for pair in pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, inliers = cv2.findHomography(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    return H


def estimate_translation_masked(src: np.ndarray,
                                 tar: np.ndarray,
                                 mask_src: Optional[np.ndarray],
                                 mask_tar: Optional[np.ndarray],
                                 inlier_thresh: float = 12.0,
                                 debug_save: bool = False,
                                 debug_path: Optional[Path] = None) -> Optional[np.ndarray]:
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    det, det_name = get_detector()

    # If LoFTR is selected, run that matcher path
    if det_name == 'LOFTR':
        # Ensure LoFTR matcher is initialized if weights were provided via global
        loftr_weights = globals().get('_LOFTR_WEIGHTS', None)
        if loftr_weights is not None and globals().get('_LOFTR_MATCHER', None) is None:
            try:
                init_loftr(loftr_weights)
            except Exception as e:
                print(f"  -> LoFTR init failed: {e}")
                return None

        try:
            mkpts0, mkpts1, mconf = run_loftr_match(src, tar, mask_src, mask_tar)
        except Exception as e:
            print(f"  -> LoFTR matching error: {e}")
            return None

        if mkpts0 is None or mkpts1 is None or len(mkpts0) < 3:
            return None

        deltas = mkpts1 - mkpts0
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        residuals = deltas - np.array([dx, dy], dtype=np.float32)
        dist = np.linalg.norm(residuals, axis=1)
        inliers = dist <= inlier_thresh
        nin = int(np.count_nonzero(inliers))
        if nin < 3:
            return None
        # Recompute dx/dy from inliers
        dx = float(np.median(deltas[inliers, 0]))
        dy = float(np.median(deltas[inliers, 1]))

        # Debug visualization: side-by-side with line segments for inliers
        if debug_save and debug_path is not None:
            pts0 = mkpts0[inliers]
            pts1 = mkpts1[inliers]
            h0, w0 = gray1.shape[:2]
            h1, w1 = gray2.shape[:2]
            canvas = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
            canvas[:h0, :w0, 0] = gray1
            canvas[:h1, w0:w0 + w1, 0] = gray2
            # draw lines
            for (x0, y0), (x1, y1) in zip(pts0.tolist(), pts1.tolist()):
                pt1 = (int(round(x0)), int(round(y0)))
                pt2 = (int(round(x1 + w0)), int(round(y1)))
                cv2.circle(canvas, pt1, 2, (0, 255, 0), -1)
                cv2.circle(canvas, pt2, 2, (0, 255, 0), -1)
                cv2.line(canvas, pt1, pt2, (255, 0, 0), 1)
            try:
                cv2.imwrite(str(debug_path), canvas)
            except Exception:
                pass

        H = np.eye(3, dtype=np.float32)
        H[0, 2] = dx
        H[1, 2] = dy
        return H

    # Fallback: classic detector route (SIFT/ORB/etc.)
    mask_src_u8 = mask_src if mask_src is not None else np.ones(gray1.shape, dtype=np.uint8) * 255
    mask_tar_u8 = mask_tar if mask_tar is not None else np.ones(gray2.shape, dtype=np.uint8) * 255

    kp1, des1 = det.detectAndCompute(gray1, mask_src_u8)
    kp2, des2 = det.detectAndCompute(gray2, mask_tar_u8)
    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        return None

    # Use L2 for float descriptors (SIFT, KAZE), Hamming for binary (ORB, AKAZE)
    norm = cv2.NORM_L2 if det_name in ('SIFT', 'KAZE') else cv2.NORM_HAMMING
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

    # Debug visualization: draw filtered matches
    if debug_save and debug_path is not None:
        inlier_matches = [m for i, m in enumerate(good) if inliers[i]]
        match_img = cv2.drawMatches(src, kp1, tar, kp2, inlier_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(str(debug_path), match_img)

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
                  blender: ImageBlender,
                  max_size: int = 14000,
                  debug_save: bool = False,
                  debug_dir: Optional[Path] = None,
                  step_name: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
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

    # Use the blender module for all blending operations
    if np.any(overlap):
        # Find overlap bounding box for optimization
        ys, xs = np.where(overlap)
        if len(ys) > 0:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            overlap_bbox = (y0, y1, x0, x1)
        else:
            overlap_bbox = None
        
        # Perform blending using the ImageBlender
        blend_result = blender.blend(moved, warped_new, mask_old, mask_new, overlap_bbox)
        out = blend_result.blended
        seam_mask_full = blend_result.seam_mask
    else:
        # No overlap, just copy new pixels
        if np.any(only_new):
            out[only_new] = warped_new[only_new]
        seam_mask_full = None

    # If requested, write debug files: moved (mosaic before), warped_new (new warped), seam mask
    if debug_save and debug_dir is not None:
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            if step_name is None:
                step_name = 'step'
            # Save moved (mosaic after warp of previous canvas but before blend)
            mv_path = debug_dir / f"{step_name}_moved.png"
            cv2.imwrite(str(mv_path), moved)
            # Save warped new image
            wn_path = debug_dir / f"{step_name}_warped_new.png"
            cv2.imwrite(str(wn_path), warped_new)
            # Save seam mask if computed
            if seam_mask_full is not None:
                sm_path = debug_dir / f"{step_name}_seam_mask.png"
                cv2.imwrite(str(sm_path), seam_mask_full)
        except Exception:
            pass

    return out, T, seam_mask_full


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
                                    transform: str = 'translation',
                                    blend: str = 'feather',
                                    lock_dy: bool = False,
                                    seam_width: int = 1,
                                    debug_save: bool = False,
                                    debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    if not image_paths:
        return None
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    # Initialize the blender with specified mode and parameters
    blender = ImageBlender(mode=blend, seam_width=seam_width, feather_sigma=6.0, min_seam_size=8)

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

        # Estimate transform between current frame and mosaic
        H = None
        tmode = (transform or 'translation').lower()
        
        # Prepare debug path for match visualization
        match_debug_path = None
        if debug_save and debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            match_debug_path = debug_dir / f"{k:03d}_{cur_path.stem}_matches.jpg"
        
        if tmode == 'translation':
            H = estimate_translation_masked(cur_img, mosaic, cur_mask, mosaic_mask,
                                           debug_save=debug_save, debug_path=match_debug_path)
        elif tmode == 'affine':
            H = estimate_transform_affine_masked(cur_img, mosaic, cur_mask, mosaic_mask)
        elif tmode in ('homography', 'h'): 
            H = estimate_transform_homography_masked(cur_img, mosaic, cur_mask, mosaic_mask)

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

        # For translation mode, ensure H is pure translation (zero drift). For other modes, keep as-is.
        if tmode == 'translation':
            H[0, 0] = 1.0; H[0, 1] = 0.0
            H[1, 0] = 0.0; H[1, 1] = 1.0

        if not is_reasonable_transform(H):
            print("  -> Transform seems unreasonable; skipping")
            continue

        # Optionally lock vertical translation to zero (prevent vertical drift)
        if lock_dy:
            try:
                H[1, 2] = 0.0
            except Exception:
                pass

        # Expand canvas and warp current frame into it
        step_name = f"{k}_{cur_path.stem}"
        mosaic_new, T, seam_mask = expand_canvas(
            mosaic, H, cur_img, blender,
            debug_save=debug_save, debug_dir=debug_dir, step_name=step_name)
        if mosaic_new is None:
            print("  -> Canvas overflow; skipping frame")
            continue

        mosaic = mosaic_new
        print(f"  -> Stitched; canvas {mosaic.shape[1]}x{mosaic.shape[0]}")
        if debug_save and debug_dir is not None:
            try:
                # save a copy of the mosaic after this step
                fname = debug_dir / f"{step_name}_mosaic.png"
                cv2.imwrite(str(fname), mosaic)
            except Exception:
                pass

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
    ap.add_argument('--transform', type=str, default='translation', choices=['translation', 'affine', 'homography'],
                    help='Global transform model between frames (default: translation)')
    ap.add_argument('--blend', type=str, default='feather', choices=['feather', 'seam', 'none'],
                    help='Blend mode in overlaps: feather (default), seam (min-error cut), or none (hard cut)')
    ap.add_argument('--lock-dy', action='store_true', help='Force vertical translation to zero to avoid vertical drift')
    ap.add_argument('--seam-width', type=int, default=1, help='Soft seam width in pixels (horizontal blur)')
    ap.add_argument('--detector', type=str, default=None, choices=['sift', 'orb', 'akaze', 'kaze'],
                    help='Force which feature detector to use during matching')
    ap.add_argument('--loftr-weights', type=str, default=None,
                    help='Path to EfficientLoFTR checkpoint to use (enables LOFTR)')
    ap.add_argument('--debug-steps', action='store_true', help='Save per-step debug images (moved/warped/seam/mosaic)')
    ap.add_argument('--debug-dir', type=str, default=None, help='Directory to save debug-step images')
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

    # Respect detector override if provided
    if args.detector:
        globals()['_FORCE_DET'] = args.detector
    # Respect LoFTR option: when weights provided, force LOFTR
    if args.loftr_weights:
        globals()['_LOFTR_WEIGHTS'] = str(args.loftr_weights)
        globals()['_FORCE_DET'] = 'loftr'

    mosaic = stitch_incremental_with_corners(
        image_paths,
        corner_paths,
        stride=max(1, args.stride),
        mask_pad=max(0, args.mask_pad),
        force_rect=not args.no_force_rect,
        transform=args.transform,
        blend=args.blend,
        lock_dy=args.lock_dy,
        seam_width=max(0, int(args.seam_width)),
        debug_save=args.debug_steps,
        debug_dir=Path(args.debug_dir) if args.debug_dir else None,
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
