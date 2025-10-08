import cv2
import numpy as np
from pathlib import Path

def container_roi(img, top=0.18, bottom=0.88, left=0.08, right=0.92):
    h, w = img.shape[:2]
    r0, r1 = int(h * top), int(h * bottom)
    c0, c1 = int(w * left), int(w * right)
    mask = np.zeros((h, w), np.uint8)
    mask[r0:r1, c0:c1] = 255
    return mask


def yellow_container_mask(img):
    """Mask pixels likely belonging to the yellow container roof.
    This suppresses static background (grey pillars, ground)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Broad yellow-ish range; tune if needed
    lower = np.array([15, 40, 60], dtype=np.uint8)
    upper = np.array([45, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    return mask


def get_detector():
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT'
    if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT'
    return cv2.ORB_create(4000), 'ORB'


def estimate_h(src, tar):
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    # ROI inside + yellow mask to avoid static pillars/background
    mask1 = cv2.bitwise_and(container_roi(src), yellow_container_mask(src))
    mask2 = cv2.bitwise_and(container_roi(tar), yellow_container_mask(tar))

    det, name = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    if name == 'SIFT':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    raw = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    H, _ = cv2.findHomography(pts1, pts2, cv2.LMEDS)
    return H


def estimate_transform_affine(src, tar):
    """Estimate a robust 2D transform limited to rotation+scale+translation.
    Returns a 3x3 matrix. Falls back to None if fails."""
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    mask1 = cv2.bitwise_and(container_roi(src), yellow_container_mask(src))
    mask2 = cv2.bitwise_and(container_roi(tar), yellow_container_mask(tar))

    det, _ = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # Choose norm based on detector
    norm = cv2.NORM_L2 if hasattr(cv2, 'SIFT_create') else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, 2)
    good = []
    for m, n in raw:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 6:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    A, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.LMEDS)
    if A is None:
        return None
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A
    return H


def is_reasonable_transform(H, max_scale=1.8, min_scale=0.55, max_persp=1e-3):
    """Heuristic guard: reject transforms that overly scale or add strong perspective."""
    if H is None or not np.isfinite(H).all():
        return False
    a = H[0, 0]; b = H[0, 1]; c = H[1, 0]; d = H[1, 1]
    # approximate isotropic scale from linear part
    s = np.sqrt(max(1e-8, (a * a + d * d + b * b + c * c) / 2.0))
    if s > max_scale or s < min_scale:
        return False
    if abs(H[2, 0]) > max_persp or abs(H[2, 1]) > max_persp:
        return False
    return True


def expand_canvas(mosaic, h_to_canvas, new_img):
    # Safety caps to avoid exploding canvas
    MAX_CANVAS_W = 12000
    MAX_CANVAS_H = 12000
    STEP_GROWTH = 1.8  # per step cap relative to current canvas size

    # compute bounds for new image when warped by H_to_canvas
    h, w = new_img.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, h_to_canvas)

    # existing mosaic bounds (0..W-1, 0..H-1) in canvas coords
    h_canvas, w_canvas = mosaic.shape[:2]
    base = np.float32([[0,0],[w_canvas,0],[w_canvas,h_canvas],[0,h_canvas]]).reshape(-1,1,2)

    all_c = np.concatenate([base, warped_corners], axis=0)
    x_min, y_min = np.floor(all_c.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_c.max(axis=0).ravel()).astype(int)

    # guard against NaNs/Infs or absurd ranges
    if not np.isfinite([x_min, y_min, x_max, y_max]).all():
        return None, None

    tx, ty = -min(0, x_min), -min(0, y_min)
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)

    new_w, new_h = int(x_max + tx), int(y_max + ty)

    # step growth guard relative to current canvas and new image size
    max_w = min(MAX_CANVAS_W, int(max(1, w_canvas) * STEP_GROWTH + w * STEP_GROWTH))
    max_h = min(MAX_CANVAS_H, int(max(1, h_canvas) * STEP_GROWTH + h * STEP_GROWTH))
    if new_w > max_w or new_h > max_h or new_w <= 0 or new_h <= 0:
        return None, None
    # move old mosaic into new canvas (keeps zeros where there's no data)
    moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # warp the new image with linear interpolation
    h_final = T @ h_to_canvas
    warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Create accurate coverage masks by warping an all-ones mask using INTER_NEAREST
    mask_src = (np.ones((h, w), dtype=np.uint8) * 255)
    warped_mask_new = cv2.warpPerspective(mask_src, h_final, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_new = (warped_mask_new > 0)

    # build mask of existing mosaic coverage by warping a ones-mask of the original mosaic size
    h_canvas, w_canvas = mosaic.shape[:2]
    mask_mosaic = (np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255)
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = (moved_mask > 0)

    out = moved.copy()

    # small morphological closing to fill tiny holes/gaps in masks (removes 1-pixel seams)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_new_u8 = (warped_mask_new > 0).astype(np.uint8) * 255
    mask_old_u8 = (moved_mask > 0).astype(np.uint8) * 255
    mask_new_u8 = cv2.morphologyEx(mask_new_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask_old_u8 = cv2.morphologyEx(mask_old_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    mask_new = (mask_new_u8 > 0)
    mask_old = (mask_old_u8 > 0)

    # pixels that only the new warped image covers: copy directly
    only_new = mask_new & (~mask_old)
    if np.any(only_new):
        out[only_new] = warped_new[only_new]

    # overlap: do feathered blending based on distance transforms to create smooth seam
    overlap = mask_new & mask_old
    if np.any(overlap):
        # compute distance transform inside each mask (larger near center)
        # distanceTransform expects 8-bit mask where non-zero is foreground
        dist_new = cv2.distanceTransform(mask_new_u8, cv2.DIST_L2, 3).astype(np.float32)
        dist_old = cv2.distanceTransform(mask_old_u8, cv2.DIST_L2, 3).astype(np.float32)

        # weights: use distance as confidence; add small epsilon to denom
        w_new = dist_new
        w_old = dist_old
        denom = (w_new + w_old)
        denom3 = denom[..., None]
        denom3[denom3 == 0] = 1.0

        moved_f = moved.astype(np.float32)
        warped_f = warped_new.astype(np.float32)
        blended = (moved_f * w_old[..., None] + warped_f * w_new[..., None]) / denom3
        blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)

        out[overlap] = blended_u8[overlap]

    # update canvas transform
    return out, T


def stitch_incremental(image_paths):
    # read first
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    mosaic = base.copy()
    # global transform from image-0 to canvas
    G0 = np.eye(3, dtype=np.float32)

    # cumulative transforms to image0 frame
    G = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(image_paths)):
        img = cv2.imread(str(image_paths[i]))
        if img is None:
            print(f"Skip missing {image_paths[i]}")
            continue
        # H between previous raw image and current raw image
        prev = cv2.imread(str(image_paths[i-1]))
        # Prefer an affine transform to avoid wild perspective
        H = estimate_transform_affine(prev, img)
        if not is_reasonable_transform(H):
            # Try full homography as second attempt
            H = estimate_h(prev, img)
        if H is None:
            # Fallback: estimate translation with ECC on ROI
            print(f"Transforms failed at {image_paths[i-1]} -> {image_paths[i]} (fallback to ECC translation)")
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    warp, cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4),
                    inputMask=container_roi(prev),
                    gaussFiltSize=5,
                )
            except cv2.error:
                warp = np.eye(2, 3, dtype=np.float32)
            H = np.vstack([warp, [0, 0, 1]]).astype(np.float32)

        # cumulative: current raw -> image0
        g_cur = G[-1] @ np.linalg.inv(H)
        G.append(g_cur.astype(np.float32))

        # map current raw to canvas: H_to_canvas = G0 @ G_cur
        h_to_canvas = G0 @ g_cur
        mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
        if mosaic_new is None:
            # If canvas would explode, try degrading to translation-only based on ECC
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    warp, cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-4),
                    inputMask=container_roi(prev),
                    gaussFiltSize=5,
                )
            except cv2.error:
                pass
            h_tr = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            g_cur = G[-2] @ np.linalg.inv(h_tr) if len(G) >= 2 else np.linalg.inv(h_tr).astype(np.float32)
            G[-1] = g_cur.astype(np.float32)
            h_to_canvas = G0 @ g_cur
            mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
            if mosaic_new is None:
                print(f"Skip frame due to excessive canvas growth: {image_paths[i]}")
                continue

        mosaic = mosaic_new
        # update canvas transform for future warps
        G0 = T @ G0

    return mosaic


def main():
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_14-35-53")
    paths = [str(base / f"{i}_cropped.jpg") for i in range(43, 75)]
    mosaic = stitch_incremental(paths)
    out = base / "mosaic_seq_43_74.jpg"
    cv2.imwrite(str(out), mosaic)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
