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

def red_container_mask(img):
    """Mask các pixel có khả năng thuộc về nóc container màu đỏ."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Màu đỏ trong HSV nằm ở 2 khoảng (quanh 0 và quanh 180)
    # Khoảng dưới
    lower1 = np.array([0, 70, 50])
    upper1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower1, upper1)
    # Khoảng trên
    lower2 = np.array([170, 70, 50])
    upper2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower2, upper2)
    # Kết hợp 2 mask
    mask = mask1 + mask2
    
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=3)
    return mask

def get_detector():
    if hasattr(cv2, 'SIFT_create'):
        return cv2.SIFT_create(), 'SIFT'
    if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
        return cv2.xfeatures2d.SIFT_create(), 'SIFT'
    return cv2.ORB_create(4000), 'ORB'

def estimate_transform_affine(src, tar):
    """Ước lượng phép biến đổi affine (xoay, co giãn, dịch chuyển)."""
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
    # Sử dụng mask ROI và mask màu đỏ
    mask1 = cv2.bitwise_and(container_roi(src), red_container_mask(src))
    mask2 = cv2.bitwise_and(container_roi(tar), red_container_mask(tar))

    det, _ = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    norm = cv2.NORM_L2 if hasattr(cv2, 'SIFT_create') else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, 2)
    good = []
    for m_n in raw:
        if len(m_n) == 2:
            m, n = m_n
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

def is_reasonable_transform(H, max_scale=1.8, min_scale=0.55):
    """Kiểm tra xem phép biến đổi có hợp lệ không."""
    if H is None or not np.isfinite(H).all():
        return False
    a = H[0, 0]; b = H[0, 1]; c = H[1, 0]; d = H[1, 1]
    s = np.sqrt(max(1e-8, (a * a + d * d + b * b + c * c) / 2.0))
    if s > max_scale or s < min_scale:
        return False
    return True

def expand_canvas(mosaic, h_to_canvas, new_img):
    MAX_CANVAS_W = 15000
    MAX_CANVAS_H = 15000

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

    if new_w > MAX_CANVAS_W or new_h > MAX_CANVAS_H or new_w <= 0 or new_h <= 0:
        return None, None
    
    moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    h_final = T @ h_to_canvas
    warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask_src = (np.ones((h, w), dtype=np.uint8) * 255)
    warped_mask_new = cv2.warpPerspective(mask_src, h_final, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_new = (warped_mask_new > 0).astype(np.uint8)

    mask_mosaic = (np.ones((h_canvas, w_canvas), dtype=np.uint8) * 255)
    moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_old = (moved_mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_new = cv2.morphologyEx((mask_new * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0
    mask_old = cv2.morphologyEx((mask_old * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1) > 0

    only_new = mask_new & (~mask_old)
    overlap = mask_new & mask_old

    out = moved.copy()
    if np.any(only_new):
        out[only_new] = warped_new[only_new]

    if np.any(overlap):
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

def crop_final_mosaic(mosaic):
    """Cắt bỏ phần nền đen/thừa xung quanh container."""
    gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Could not find contours to crop.")
        return mosaic

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    cropped = mosaic[y:y+h, x:x+w]
    print(f"Cropped final mosaic from {mosaic.shape[1]}x{mosaic.shape[0]} to {w}x{h}")
    return cropped

def stitch_incremental_robust(image_paths):
    base = cv2.imread(str(image_paths[0]))
    if base is None:
        raise FileNotFoundError(image_paths[0])

    mosaic = base.copy()
    G0_to_canvas = np.eye(3, dtype=np.float32)
    G_cumulative = [np.eye(3, dtype=np.float32)]

    for i in range(1, len(image_paths)):
        print(f"Processing image {i+1}/{len(image_paths)}: {Path(image_paths[i]).name}")
        img = cv2.imread(str(image_paths[i]))
        if img is None:
            print(f"  -> Skip missing {image_paths[i]}")
            continue
        
        prev = cv2.imread(str(image_paths[i-1]))
        if prev is None:
            print(f"  -> Skip missing previous image {image_paths[i-1]}")
            continue

        # Ưu tiên Affine để tránh biến dạng phối cảnh
        H = estimate_transform_affine(prev, img)
        
        if not is_reasonable_transform(H):
            print(f"  -> Affine transform is unreasonable. Fallback to ECC Translation.")
            H = None # Force fallback

        if H is None:
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                cv2.findTransformECC(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                    warp, cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 500, 1e-5),
                    inputMask=container_roi(prev),
                    gaussFiltSize=5,
                )
                H = np.eye(3, dtype=np.float32)
                H[:2, :] = warp
            except cv2.error:
                print(f"  -> ECC also failed. Skipping frame.")
                continue

        try:
            g_cur = G_cumulative[-1] @ np.linalg.inv(H)
            G_cumulative.append(g_cur)
        except np.linalg.LinAlgError:
            print(f"  -> Singular matrix. Skipping frame.")
            continue

        h_to_canvas = G0_to_canvas @ g_cur
        mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
        
        if mosaic_new is None:
            print(f"  -> Skip frame due to excessive canvas growth.")
            G_cumulative.pop() # Rollback
            continue

        mosaic = mosaic_new
        G0_to_canvas = T @ G0_to_canvas
        print(f"  -> Stitched. New canvas size: {mosaic.shape[1]}x{mosaic.shape[0]}")

    return mosaic

def main():
    # --- Cấu hình cho container ĐỎ ---
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_15-25-13")
    paths = [str(base / f"{i}_cropped.jpg") for i in range(44, 91)]
    
    for p in paths:
        if not Path(p).exists():
            print(f"Lỗi: Không tìm thấy file đã cắt: {p}")
            return
            
    print("--- Bắt đầu ghép ảnh cho container ĐỎ (phương pháp Robust) ---")
    mosaic = stitch_incremental_robust(paths)
    
    if mosaic is not None and mosaic.size > 0:
        print("\nGhép ảnh thành công. Bắt đầu cắt nền thừa...")
        final_image = crop_final_mosaic(mosaic)
        
        out_path = base / "mosaic_red_container_robust.jpg"
        cv2.imwrite(str(out_path), final_image)
        print(f"\nHoàn tất! Đã lưu ảnh cuối cùng -> {out_path}")
    else:
        print("\nGhép ảnh thất bại.")

if __name__ == "__main__":
    main()
