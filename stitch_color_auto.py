import cv2
import numpy as np
from pathlib import Path
import argparse


def container_roi(img, top=0.18, bottom=0.88, left=0.08, right=0.92):
    h, w = img.shape[:2]
    r0, r1 = int(h * top), int(h * bottom)
    c0, c1 = int(w * left), int(w * right)
    mask = np.zeros((h, w), np.uint8)
    mask[r0:r1, c0:c1] = 255
    return mask


def sample_hsv_range(img, roi_frac=0.28, tol_h=15, tol_s=60, tol_v=60):
    """Robust sampling of HSV bounds.
    Strategy:
    - Compute HSV inside the `container_roi` and build a hue histogram weighted by saturation*value.
    - Pick the hue peak with highest weight (ignoring very low-saturation pixels).
    - Use median S and V within that hue bin to form a representative center HSV.
    - Fallback to central ROI median when ROI sampling is insufficient or ambiguous.
    Returns (lower, upper) as uint8 arrays.
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask ROI to focus on likely container area
    rmask = container_roi(img)
    coords = np.where(rmask > 0)

    # fallback: central crop median
    cx0 = int(w_img * (0.5 - roi_frac / 2.0))
    cy0 = int(h_img * (0.5 - roi_frac / 2.0))
    cx1 = int(w_img * (0.5 + roi_frac / 2.0))
    cy1 = int(h_img * (0.5 + roi_frac / 2.0))
    roi = hsv[cy0:cy1, cx0:cx1]
    med_center = np.median(roi.reshape(-1, 3), axis=0).astype(int)
    h_c, s_c, v_c = int(med_center[0]), int(med_center[1]), int(med_center[2])

    # If ROI mask has enough pixels, build hue histogram weighted by saturation*value
    if coords[0].size > 200:
        samples = hsv[coords]
        # filter out very low-saturation/value pixels (likely background/sky/ground)
        sat_val = (samples[:,1].astype(np.float32) * samples[:,2].astype(np.float32))
        good_idx = sat_val > (20.0 * 20.0)
        if good_idx.sum() < 50:
            # not enough high-quality pixels -> fallback to center
            h_med, s_med, v_med = h_c, s_c, v_c
        else:
            filt = samples[good_idx]
            hues = filt[:,0].astype(int)
            # histogram over 180 hue bins
            hist = np.bincount(hues, minlength=180)
            peak_h = int(np.argmax(hist))
            # get pixels within +/-3 hue bins around peak to compute median S/V
            low_bin = (peak_h - 3) % 180
            high_bin = (peak_h + 3) % 180
            if low_bin <= high_bin:
                sel = filt[(filt[:,0] >= low_bin) & (filt[:,0] <= high_bin)]
            else:
                # wrap-around
                sel = filt[(filt[:,0] >= low_bin) | (filt[:,0] <= high_bin)]
            if sel.shape[0] < 10:
                h_med, s_med, v_med = h_c, s_c, v_c
            else:
                med = np.median(sel, axis=0).astype(int)
                h_med, s_med, v_med = int(med[0]), int(med[1]), int(med[2])
    else:
        h_med, s_med, v_med = h_c, s_c, v_c

    # construct bounds with tolerances, handle wrap-around
    low_h = (h_med - tol_h) % 180
    high_h = (h_med + tol_h) % 180
    low = np.array([low_h, max(0, s_med - tol_s), max(0, v_med - tol_v)], dtype=np.uint8)
    high = np.array([high_h, min(255, s_med + tol_s), min(255, v_med + tol_v)], dtype=np.uint8)
    return low, high


def color_mask_from_bounds(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lh, ls, lv = lower
    uh, us, uv = upper
    if lh <= uh:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        # wrap-around
        lower1 = np.array([0, ls, lv], dtype=np.uint8)
        upper1 = np.array([uh, us, uv], dtype=np.uint8)
        lower2 = np.array([lh, ls, lv], dtype=np.uint8)
        upper2 = np.array([179, us, uv], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)

    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # keep only main connected component inside ROI to suppress background hits
    comp_mask = mask.copy()
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        big = np.zeros_like(mask)
        cv2.drawContours(big, contours, idx, 255, cv2.FILLED)
        mask = big
    return mask


def bbox_from_edges(img, roi_mask=None, min_area_frac=0.01, expand_px=12):
    """Fast heuristic: find strong edges, take largest contour inside ROI, return a rectangular mask.
    Works well when container has clear shape/edges vs background.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(g, 50, 150)
    # dilate to connect edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    if roi_mask is not None:
        edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(img.shape[:2], dtype=np.uint8)

    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    cnt = contours[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    # reject tiny regions
    img_area = img.shape[0] * img.shape[1]
    if (w * h) < max(16, img_area * min_area_frac):
        return np.zeros(img.shape[:2], dtype=np.uint8)

    x0 = max(0, x - expand_px)
    y0 = max(0, y - expand_px)
    x1 = min(img.shape[1], x + w + expand_px)
    y1 = min(img.shape[0], y + h + expand_px)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def rect_fill_from_mask(mask, pad=8, min_comp_area_px=500, require_area_fraction=0.02):
    """
    Fill a rectangle defined by min/max x,y of mask pixels.
    - mask: uint8 mask (0/255 or 0/1)
    - pad: pixels to expand rectangle on each side
    - min_comp_area_px: ignore if mask has too few pixels
    - require_area_fraction: only apply if mask area >= require_area_fraction * image_area
    Returns a uint8 mask (0/255) where the rectangle area is filled.
    """
    if mask is None or mask.size == 0:
        return mask

    m = (mask > 0).astype(np.uint8)
    total_px = int(m.sum())
    h, w = m.shape[:2]
    if total_px < int(min_comp_area_px):
        return (m * 255).astype(np.uint8)

    if (total_px / float(h * w)) < float(require_area_fraction):
        return (m * 255).astype(np.uint8)

    ys, xs = np.where(m > 0)
    if ys.size == 0 or xs.size == 0:
        return (m * 255).astype(np.uint8)

    y0 = max(0, int(ys.min()) - pad)
    y1 = min(h - 1, int(ys.max()) + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(w - 1, int(xs.max()) + pad)

    out = np.zeros_like(m, dtype=np.uint8)
    out[y0:y1+1, x0:x1+1] = 255

    return out


def trim_mask_vertically(mask, min_pixels_per_row_frac=0.01):
    """Keep the largest contiguous vertical run of rows that have at least a fraction of pixels set.
    This removes thin stray regions at top/bottom that expand bounding box."""
    if mask is None or mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    m = (mask > 0).astype(np.uint8)
    row_counts = m.sum(axis=1)
    thresh = max(3, int(w * float(min_pixels_per_row_frac)))
    good_rows = np.where(row_counts >= thresh)[0]
    if good_rows.size == 0:
        return mask
    # find largest contiguous run
    runs = []
    start = good_rows[0]
    prev = good_rows[0]
    for r in good_rows[1:]:
        if r == prev + 1:
            prev = r
            continue
        runs.append((start, prev))
        start = r
        prev = r
    runs.append((start, prev))
    best = max(runs, key=lambda x: x[1] - x[0])
    y0, y1 = best[0], best[1]
    out = np.zeros_like(m, dtype=np.uint8)
    out[y0:y1+1, :] = m[y0:y1+1, :]
    return (out * 255).astype(np.uint8)


def find_horizontal_rims(img, mask=None, central_width_frac=0.6, blur_ksize=5, canny_th1=40, canny_th2=120, smooth_k=11):
    """Detect likely top and bottom horizontal rims using per-row edge energy inside central columns.
    Returns (y_top, y_bottom) or (None, None) if not found.
    Implemented without external deps (uses cv2).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, canny_th1, canny_th2)
    h, w = edges.shape[:2]
    cx0 = int(w * (0.5 - central_width_frac / 2.0))
    cx1 = int(w * (0.5 + central_width_frac / 2.0))
    if cx0 >= cx1:
        cx0 = 0; cx1 = w
    roi = edges[:, cx0:cx1]
    row_sum = roi.sum(axis=1).astype(np.float32)
    # smooth with 1D Gaussian via cv2.GaussianBlur on column vector
    ks = smooth_k if (smooth_k % 2 == 1) else smooth_k + 1
    row_smooth = cv2.GaussianBlur(row_sum.reshape(-1, 1), (ks, 1), 0).flatten()
    if row_smooth.max() < 2.0:
        return None, None
    center = h // 2
    top_region = row_smooth[:center]
    bottom_region = row_smooth[center:]
    if top_region.size == 0 or bottom_region.size == 0:
        return None, None
    top_idx = int(np.argmax(top_region))
    bottom_idx = int(np.argmax(bottom_region)) + center
    if bottom_idx - top_idx < int(0.12 * h):
        return None, None
    return top_idx, bottom_idx


def get_detector():
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT'
    if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT'
    return cv2.ORB_create(4000), 'ORB'


def hybrid_mask(prev, cur, lower, upper):
    """Robust union of color mask and bbox edges, with rim-based cropping and vertical trimming.
    This reduces stray top/bottom regions and recovers when color-only mask is narrow.
    """
    # color-based mask and edge bbox
    mask_color = color_mask_from_bounds(prev, lower, upper)
    mask_bbox = bbox_from_edges(prev, container_roi(prev))
    union = cv2.bitwise_or(mask_color, mask_bbox)

    # morphological close to join nearby regions (smaller kernel to avoid overfilling)
    kernel = np.ones((5, 5), np.uint8)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, kernel, iterations=1)

    # connected components: prefer the largest component but lower min threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(union)
    if num_labels <= 1:
        out = union.copy()
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        idx = int(np.argmax(areas)) + 1
        # keep the largest and also any other components that are vertically overlapping (merge)
        out[labels == idx] = 255
        # merge others if their bbox intersects vertically with the main component (helps when a pole splits mask)
        main_y0 = stats[idx, cv2.CC_STAT_TOP]
        main_y1 = main_y0 + stats[idx, cv2.CC_STAT_HEIGHT]
        for i in range(1, num_labels):
            if i == idx:
                continue
            y0 = stats[i, cv2.CC_STAT_TOP]
            y1 = y0 + stats[i, cv2.CC_STAT_HEIGHT]
            # if vertical overlap > 10% of smaller height, merge
            overlap = max(0, min(main_y1, y1) - max(main_y0, y0))
            min_h = min(stats[idx, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_HEIGHT])
            if min_h > 0 and (overlap / float(min_h)) > 0.1:
                out[labels == i] = 255

    # optional rectangle fill when mask is holey
    rect = rect_fill_from_mask(out, pad=8, min_comp_area_px=600, require_area_fraction=0.003)
    if rect is not None and rect.sum() > 0:
        out_sum = int((out > 0).sum())
        rect_sum = int((rect > 0).sum())
        hole_frac = 1.0 - (out_sum / float(rect_sum + 1e-9))
        if hole_frac >= 0.08:
            out = rect

    # If mask still very small or narrow, try stronger fallbacks
    img_area = out.size
    area_px = int((out > 0).sum())
    # width fraction of bounding rect
    contours, _ = cv2.findContours((out > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox_w_frac = 0.0
    if contours:
        x,y,w,h = cv2.boundingRect(contours[int(np.argmax([cv2.contourArea(c) for c in contours]))])
        bbox_w_frac = float(w) / float(out.shape[1])

    # If area is tiny (<12% of image) or width small (<0.6), union with edge bbox and perform stronger closing
    try_union = False
    if area_px < 0.12 * img_area or bbox_w_frac < 0.60:
        try_union = True

    if try_union:
        bb = bbox_from_edges(prev, container_roi(prev))
        if bb is not None and bb.sum() > 0:
            out = cv2.bitwise_or(out, bb)
        # stronger closing/dilation to bridge gaps
        kernel2 = np.ones((11, 11), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel2, iterations=1)
        out = cv2.dilate(out, np.ones((7,7), np.uint8), iterations=1)

    # If still too small, compute convex hull of all contours and fill it
    area_px2 = int((out > 0).sum())
    if area_px2 < 0.12 * img_area:
        cnts, _ = cv2.findContours((out > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            all_pts = np.vstack(cnts)
            hull = cv2.convexHull(all_pts)
            full = np.zeros_like(out)
            cv2.drawContours(full, [hull], -1, 255, cv2.FILLED)
            out = full

    # If bounding rect is still narrow horizontally, expand to central fraction
    contours2, _ = cv2.findContours((out > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours2:
        x,y,w,h = cv2.boundingRect(contours2[int(np.argmax([cv2.contourArea(c) for c in contours2]))])
        w_frac = float(w) / float(out.shape[1])
        if w_frac < 0.65:
            # expand horizontally to center 80% or to container_roi width
            cx0 = int(out.shape[1] * 0.1)
            cx1 = int(out.shape[1] * 0.9)
            ext = np.zeros_like(out)
            ext[:, cx0:cx1] = 255
            out = cv2.bitwise_or(out, ext)

    # use rim detection to crop stray top/bottom if available
    try:
        y_top, y_bottom = find_horizontal_rims(prev, mask=out, central_width_frac=0.6, blur_ksize=5, canny_th1=40, canny_th2=110, smooth_k=9)
        if y_top is not None and y_bottom is not None:
            cropped = np.zeros_like(out)
            pad = 6
            y0 = max(0, y_top - pad)
            y1 = min(out.shape[0]-1, y_bottom + pad)
            cropped[y0:y1+1, :] = out[y0:y1+1, :]
            out = cropped
    except Exception:
        pass

    # vertical trim to keep largest continuous vertical band
    out_trim = trim_mask_vertically(out, min_pixels_per_row_frac=0.01)
    if out_trim is not None and out_trim.sum() > 0:
        out = out_trim

    # Temporal propagation: try warping previous color_mask into this frame and union
    try:
        if 'mask_color' in locals() and mask_color is not None:
            warped_prev = warp_mask_via_ecc(prev, cur, mask_color)
            if warped_prev is not None:
                # union and keep if it increases coverage meaningfully
                union2 = cv2.bitwise_or(out, warped_prev)
                if union2.sum() > out.sum():
                    out = union2
    except Exception:
        pass

    # Finally, produce a full-width mask but confined vertically to the detected rows.
    pad_vert = 6
    # find non-empty rows in the mask
    rows = np.where((out > 0).any(axis=1))[0]
    if rows.size > 0:
        y0 = max(0, int(rows.min()) - pad_vert)
        y1 = min(out.shape[0] - 1, int(rows.max()) + pad_vert)
        full = np.zeros_like(out, dtype=np.uint8)
        full[y0:y1+1, :] = 255
        out = full
    else:
        # fallback: use container_roi vertical span if available
        try:
            roi = container_roi(prev)
            ys = np.where(roi > 0)[0]
            if ys.size > 0:
                y0 = max(0, int(ys.min()) - pad_vert)
                y1 = min(out.shape[0] - 1, int(ys.max()) + pad_vert)
                full = np.zeros_like(out, dtype=np.uint8)
                full[y0:y1+1, :] = 255
                out = full
        except Exception:
            pass

    return out


def warp_mask_via_ecc(prev_img, cur_img, mask_prev):
    """Estimate translation via ECC (translation-only) and warp mask_prev into cur_img coords.
    Returns warped_mask (uint8) or None on failure."""
    try:
        gray1 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
        warp = np.eye(2, 3, dtype=np.float32)
        cv2.findTransformECC(gray1, gray2, warp, cv2.MOTION_TRANSLATION,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4),
                             inputMask=(mask_prev>0).astype(np.uint8), gaussFiltSize=5)
        H = np.vstack([warp, [0,0,1]]).astype(np.float32)
        h, w = cur_img.shape[:2]
        warped = cv2.warpPerspective(mask_prev, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return warped
    except Exception:
        return None


def rect_fill_if_close(mask, min_fill_ratio=0.72):
    """If the largest component already covers most of its bounding rectangle,
    replace the mask by the filled bounding rectangle. Returns new mask.
    """
    if mask is None or mask.size == 0:
        return mask
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    cnt = contours[idx]
    x, y, w, h = cv2.boundingRect(cnt)
    rect_area = max(1, w * h)
    comp_area = cv2.contourArea(cnt)
    ratio = float(comp_area) / float(rect_area)
    if ratio >= min_fill_ratio:
        m = np.zeros_like(mask)
        m[y:y+h, x:x+w] = 255
        return m
    return mask


def estimate_transform_affine_masked(src, tar, mask_src, mask_tar):
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    det, det_name = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask_src)
    kp2, des2 = det.detectAndCompute(gray2, mask_tar)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
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


def is_reasonable_transform(H, max_scale=1.8, min_scale=0.55):
    if H is None or not np.isfinite(H).all():
        return False
    a = H[0, 0]; b = H[0, 1]; c = H[1, 0]; d = H[1, 1]
    s = np.sqrt(max(1e-8, (a * a + d * d + b * b + c * c) / 2.0))
    if s > max_scale or s < min_scale:
        return False
    return True


def expand_canvas(mosaic, h_to_canvas, new_img, max_size=14000):
    h, w = new_img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, h_to_canvas)

    h_canvas, w_canvas = mosaic.shape[:2]
    base = np.float32([[0,0],[w_canvas,0],[w_canvas,h_canvas],[0,h_canvas]]).reshape(-1,1,2)

    all_c = np.concatenate([base, warped_corners], axis=0)
    x_min, y_min = np.floor(all_c.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_c.max(axis=0).ravel()).astype(int)
    if not np.isfinite([x_min, y_min, x_max, y_max]).all():
        return None, None

    tx, ty = -min(0, x_min), -min(0, y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
    new_w, new_h = int(x_max + tx), int(y_max + ty)
    if new_w <= 0 or new_h <= 0 or new_w > max_size or new_h > max_size:
        return None, None

    moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    h_final = T @ h_to_canvas
    warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    mask_src = (np.ones((h, w), dtype=np.uint8) * 255)
    warped_mask_new = cv2.warpPerspective(mask_src, h_final, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_new = (warped_mask_new > 0)

    mask_mosaic = (np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255)
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = (moved_mask > 0)

    out = moved.copy()
    only_new = mask_new & (~mask_old)
    overlap = mask_new & mask_old
    if np.any(only_new):
        out[only_new] = warped_new[only_new]
    if np.any(overlap):
        mask_new_u8 = (warped_mask_new > 0).astype(np.uint8) * 255
        mask_old_u8 = (moved_mask > 0).astype(np.uint8) * 255
        dist_new = cv2.distanceTransform(mask_new_u8, cv2.DIST_L2, 5).astype(np.float32)
        dist_old = cv2.distanceTransform(mask_old_u8, cv2.DIST_L2, 5).astype(np.float32)
        denom = dist_new + dist_old + 1e-6
        w_new = dist_new / denom
        w_old = 1.0 - w_new
        moved_f = moved.astype(np.float32)
        warped_f = warped_new.astype(np.float32)
        blended = moved_f * w_old[..., None] + warped_f * w_new[..., None]
        blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
        out[overlap] = blended_u8[overlap]

    return out, T


def crop_final_mosaic(mosaic):
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mosaic
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return mosaic[y:y+h, x:x+w]


def stitch_incremental_color(image_paths, suffix, debug=False, use_bbox=False, force_rect=True):
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    # sample HSV bounds from the first image center
    lower, upper = sample_hsv_range(base)
    print(f"Sampled HSV bounds: lower={lower.tolist()} upper={upper.tolist()}")

    mosaic = base.copy()
    G0 = np.eye(3, dtype=np.float32)
    G = [np.eye(3, dtype=np.float32)]

    # Process every 2nd image by default (stride=2). This reduces number of pairwise matches.
    n = len(image_paths)
    idxs = list(range(0, n, 2))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    for k in range(1, len(idxs)):
        prev_idx = idxs[k-1]
        cur_idx = idxs[k]
        print(f"Processing image {cur_idx+1}/{n}: {Path(image_paths[cur_idx]).name} (prev {Path(image_paths[prev_idx]).name})")
        img = cv2.imread(str(image_paths[cur_idx]))
        if img is None:
            print(f"  -> Skip missing {image_paths[cur_idx]}")
            continue

        prev = cv2.imread(str(image_paths[prev_idx]))
        if prev is None:
            print(f"  -> Skip missing prev {image_paths[prev_idx]}")
            continue

        if use_bbox:
            # try hybrid: union of color and bbox for robustness
            mask_prev = hybrid_mask(prev, img, lower, upper)
            mask_cur = hybrid_mask(img, prev, lower, upper)
        else:
            mask_prev = color_mask_from_bounds(prev, lower, upper)
            mask_cur = color_mask_from_bounds(img, lower, upper)
        # restrict to ROI to avoid background matches
        mask_prev = cv2.bitwise_and(mask_prev, container_roi(prev))
        mask_cur = cv2.bitwise_and(mask_cur, container_roi(img))

        # optionally force-fill near-rectangular masks to their bounding rect
        if force_rect:
            mask_prev_rect = rect_fill_if_close(mask_prev)
            mask_cur_rect = rect_fill_if_close(mask_cur)
            # if rect_fill_if_close returned a mask, replace
            if mask_prev_rect is not None:
                mask_prev = mask_prev_rect
            if mask_cur_rect is not None:
                mask_cur = mask_cur_rect

        H = estimate_transform_affine_masked(prev, img, mask_prev, mask_cur)
        if not is_reasonable_transform(H):
            H = None

        if H is None:
            # fallback to ECC translation using ROI
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    warp, cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-4),
                    inputMask=container_roi(prev),
                    gaussFiltSize=5,
                )
                H = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            except cv2.error:
                print("  -> ECC failed; skipping frame")
                continue

        try:
            g_cur = G[-1] @ np.linalg.inv(H)
            G.append(g_cur.astype(np.float32))
        except np.linalg.LinAlgError:
            print("  -> Singular matrix. Skipping frame.")
            continue

        h_to_canvas = G0 @ g_cur
        mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
        if mosaic_new is None:
            print("  -> Skip frame due to excessive canvas growth.")
            G.pop()
            continue

        mosaic = mosaic_new
        G0 = T @ G0
        print(f"  -> Stitched. New canvas size: {mosaic.shape[1]}x{mosaic.shape[0]}")
        if debug:
            dd = Path(image_paths[0]).parent / 'debug_color'
            dd.mkdir(exist_ok=True)
            cv2.imwrite(str(dd / f"mask_{prev_idx:03d}.png"), mask_prev)
            cv2.imwrite(str(dd / f"mask_{cur_idx:03d}.png"), mask_cur)
            if use_bbox:
                cv2.imwrite(str(dd / f"bbox_{prev_idx:03d}.png"), mask_prev)
                cv2.imwrite(str(dd / f"bbox_{cur_idx:03d}.png"), mask_cur)
            if force_rect:
                cv2.imwrite(str(dd / f"mask_rect_{prev_idx:03d}.png"), mask_prev)
                cv2.imwrite(str(dd / f"mask_rect_{cur_idx:03d}.png"), mask_cur)

    return mosaic


def main():
    p = argparse.ArgumentParser(description='Color-sampled stitching (auto HSV sample from first image)')
    p.add_argument('dir', help='directory with images')
    p.add_argument('start', type=int)
    p.add_argument('end', type=int)
    p.add_argument('--suffix', default='_cropped.jpg')
    p.add_argument('--debug', action='store_true')
    p.add_argument('--use-bbox', action='store_true', help='Detect fast rectangular bbox from edges and use it instead of color mask')
    args = p.parse_args()

    base = Path(args.dir)
    paths = [str(base / f"{i}{args.suffix}") for i in range(args.start, args.end + 1)]
    for q in paths:
        if not Path(q).exists():
            print(f"Error: missing input {q}")
            return

    print("Starting color-auto stitching...")
    # pass flag by reading args here
    mosaic = None
    # adapt stitch function behavior: we will pass debug and use_bbox flag
    mosaic = stitch_incremental_color(paths, args.suffix, debug=args.debug)
    if mosaic is None:
        print("Stitching failed.")
        return
    out = Path(paths[0]).parent / f"mosaic_color_{args.start}_{args.end}.jpg"
    final = crop_final_mosaic(mosaic)
    cv2.imwrite(str(out), final)
    print(f"Saved -> {out}")


if __name__ == '__main__':
    main()