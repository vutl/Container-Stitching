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


def blue_container_mask(img):
    """Mask pixels likely belonging to the BLUE container roof."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV range for blue colors
    lower = np.array([100, 100, 20], dtype=np.uint8)
    upper = np.array([140, 255, 255], dtype=np.uint8)
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
    # ROI inside + BLUE mask to avoid static pillars/background
    mask1 = cv2.bitwise_and(container_roi(src), blue_container_mask(src))
    mask2 = cv2.bitwise_and(container_roi(tar), blue_container_mask(tar))

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

    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 4.0)
    return H


def estimate_transform_affine(src, tar):
    """Estimate a robust 2D transform limited to rotation+scale+translation."""
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    mask1 = cv2.bitwise_and(container_roi(src), blue_container_mask(src))
    mask2 = cv2.bitwise_and(container_roi(tar), blue_container_mask(tar))

    det, _ = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

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

    A, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
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
    s = np.sqrt(max(1e-8, (a * a + d * d + b * b + c * c) / 2.0))
    if s > max_scale or s < min_scale:
        return False
    if abs(H[2, 0]) > max_persp or abs(H[2, 1]) > max_persp:
        return False
    return True


def expand_canvas(mosaic, h_to_canvas, new_img):
    MAX_CANVAS_W = 12000
    MAX_CANVAS_H = 12000
    STEP_GROWTH = 1.8

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

    max_w = min(MAX_CANVAS_W, int(max(1, w_canvas) * STEP_GROWTH + w * STEP_GROWTH))
    max_h = min(MAX_CANVAS_H, int(max(1, h_canvas) * STEP_GROWTH + h * STEP_GROWTH))
    if new_w > max_w or new_h > max_h or new_w <= 0 or new_h <= 0:
        return None, None
    
    # warp existing mosaic into new canvas (use explicit border filling)
    moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # warp the incoming image
    h_final = T @ h_to_canvas
    warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # create accurate coverage masks by warping binary ones with INTER_NEAREST
    mask_src = (np.ones((h, w), dtype=np.uint8) * 255)
    warped_mask_new = cv2.warpPerspective(mask_src, h_final, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_new = (warped_mask_new > 0).astype(np.uint8)

    mask_mosaic = (np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255)
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = (moved_mask > 0).astype(np.uint8)

    # small closing to fill 1-2px holes and avoid thin black seams
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_new = cv2.morphologyEx((mask_new * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    mask_old = cv2.morphologyEx((mask_old * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0

    only_new = mask_new & (~mask_old)
    overlap = mask_new & mask_old

    out = moved.copy()
    if np.any(only_new):
        out[only_new] = warped_new[only_new]

    if np.any(overlap):
        # distance-based soft weights to smoothly blend overlap and avoid black pull-down
        mask_new_u8 = (mask_new.astype(np.uint8) * 255)
        mask_old_u8 = (mask_old.astype(np.uint8) * 255)
        dist_new = cv2.distanceTransform(mask_new_u8, cv2.DIST_L2, 5).astype(np.float32)
        dist_old = cv2.distanceTransform(mask_old_u8, cv2.DIST_L2, 5).astype(np.float32)
        denom = dist_new + dist_old + 1e-6
        w_new = dist_new / denom
        w_old = 1.0 - w_new

        moved_f = moved.astype(np.float32)
        warped_f = warped_new.astype(np.float32)
        w_new_3 = w_new[..., None]
        w_old_3 = w_old[..., None]

        blended = moved_f * w_old_3 + warped_f * w_new_3
        blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
        out[overlap] = blended_u8[overlap]

    return out, T


def stitch_incremental(image_paths):
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    mosaic = base.copy()
    G0 = np.eye(3, dtype=np.float32)
    G = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(image_paths)):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_paths[i]}")
        img = cv2.imread(str(image_paths[i]))
        if img is None:
            print(f"  -> Skip missing {image_paths[i]}")
            continue
        
        prev = cv2.imread(str(image_paths[i-1]))
        
        H = estimate_transform_affine(prev, img)
        if not is_reasonable_transform(H):
            H = estimate_h(prev, img)
        
        if H is None:
            print(f"  -> Transforms failed, fallback to ECC translation")
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

        try:
            g_cur = G[-1] @ np.linalg.inv(H)
            G.append(g_cur.astype(np.float32))
        except np.linalg.LinAlgError:
            print(f"  -> Singular matrix, cannot invert. Skipping frame.")
            continue

        h_to_canvas = G0 @ g_cur
        mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
        
        if mosaic_new is None:
            print(f"  -> Canvas would explode, trying with pure translation fallback.")
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
                print(f"  -> Skip frame due to excessive canvas growth.")
                continue

        mosaic = mosaic_new
        G0 = T @ G0
        print(f"  -> Stitched. New canvas size: {mosaic.shape[1]}x{mosaic.shape[0]}")

    return mosaic


def main():
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_14-46-47")
    paths = [str(base / f"{i}_cropped.jpg") for i in range(49, 108)]
    
    # Verify all paths exist before starting
    for p in paths:
        if not Path(p).exists():
            print(f"Error: Input file not found: {p}")
            return
            
    print("Starting stitching process for blue container...")
    mosaic = stitch_incremental(paths)
    
    if mosaic is not None:
        out = base / "mosaic_seq_49_107_final.jpg"
        cv2.imwrite(str(out), mosaic)
        print(f"\nStitching complete. Saved -> {out}")
    else:
        print("\nStitching failed.")


if __name__ == "__main__":
    main()
