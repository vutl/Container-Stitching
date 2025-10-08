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

def color_mask_from_bounds(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lh, ls, lv = lower
    uh, us, uv = upper
    if lh <= uh:
        mask = cv2.inRange(hsv, lower, upper)
    else:
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
    # Giới hạn mask trong ROI
    mask = cv2.bitwise_and(mask, container_roi(img))
    comp_mask = mask.copy()
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        big = np.zeros_like(mask)
        cv2.drawContours(big, contours, idx, 255, cv2.FILLED)
        mask = big
    return mask

def find_container_corners(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) == 4:
        corners = approx[:, 0, :]
    else:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        corners = box.astype(int)
    # Sắp xếp lại thứ tự: [top-left, top-right, bottom-right, bottom-left]
    s = corners.sum(axis=1)
    diff = np.diff(corners, axis=1)
    ordered = np.zeros((4,2), dtype=np.float32)
    ordered[0] = corners[np.argmin(s)]
    ordered[2] = corners[np.argmax(s)]
    ordered[1] = corners[np.argmin(diff)]
    ordered[3] = corners[np.argmax(diff)]
    # Ép các điểm ra sát mép ảnh (full width)
    # ordered: [top-left, top-right, bottom-right, bottom-left]
    h, w = mask.shape[:2]
    # Tìm 2 điểm có y nhỏ nhất (trên), 2 điểm có y lớn nhất (dưới)
    idxs = np.argsort(ordered[:,1])
    top = ordered[idxs[:2]]
    bot = ordered[idxs[2:]]
    # Trái/phải dựa vào x
    if top[0,0] < top[1,0]:
        top_left, top_right = top[0], top[1]
    else:
        top_left, top_right = top[1], top[0]
    if bot[0,0] < bot[1,0]:
        bot_left, bot_right = bot[0], bot[1]
    else:
        bot_left, bot_right = bot[1], bot[0]
    # Ép x=0 cho left, x=w-1 cho right
    top_left[0] = 0
    bot_left[0] = 0
    top_right[0] = w-1
    bot_right[0] = w-1
    new_ordered = np.array([top_left, top_right, bot_right, bot_left], dtype=np.float32)
    return new_ordered

def sample_hsv_range(img, roi_frac=0.28, tol_h=15, tol_s=60, tol_v=60):
    h, w = img.shape[:2]
    cx0 = int(w * (0.5 - roi_frac / 2.0))
    cy0 = int(h * (0.5 - roi_frac / 2.0))
    cx1 = int(w * (0.5 + roi_frac / 2.0))
    cy1 = int(h * (0.5 + roi_frac / 2.0))
    roi = img[cy0:cy1, cx0:cx1]
    hsv_center = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    med_center = np.median(hsv_center.reshape(-1, 3), axis=0).astype(int)
    h_c, s_c, v_c = int(med_center[0]), int(med_center[1]), int(med_center[2])
    return np.array([h_c-15, max(0, s_c-60), max(0, v_c-60)], dtype=np.uint8), np.array([(h_c+15)%180, min(255, s_c+60), min(255, v_c+60)], dtype=np.uint8)

def main():
    parser = argparse.ArgumentParser(description='Warp container to full width using 4 corners after ROI mask')
    parser.add_argument('dir', help='directory with images')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('--suffix', default='_cropped.jpg')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    base = Path(args.dir)
    paths = [str(base / f"{i}{args.suffix}") for i in range(args.start, args.end + 1)]
    imgs = [cv2.imread(p) for p in paths]
    if any([im is None for im in imgs]):
        print('Missing image!')
        return
    lower, upper = sample_hsv_range(imgs[0])
    debug_dir = base / 'debug_warp_fullwidth'
    if args.debug:
        debug_dir.mkdir(exist_ok=True)
    # --- Warp toàn bộ ảnh về reference rectangle ---
    warped_imgs = []
    warped_masks = []
    reference_corners = None
    for idx, img in enumerate(imgs):
        H, W = img.shape[:2]
        mask = color_mask_from_bounds(img, lower, upper)
        corners = find_container_corners(mask)
        if corners is None:
            print(f'Image {idx}: No corners found!')
            continue
        if reference_corners is None:
            reference_corners = corners.copy()
            dst_pts = np.array([[0,0], [W-1,0], [W-1,H-1], [0,H-1]], dtype=np.float32)
        # Đảm bảo ép 4 góc ra sát mép ảnh (x=0, x=w-1)
        corners[0][0] = 0; corners[1][0] = W-1; corners[2][0] = W-1; corners[3][0] = 0
        M = cv2.getPerspectiveTransform(corners, reference_corners)
        warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.warpPerspective(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_imgs.append(warped)
        warped_masks.append(warped_mask)
        if args.debug:
            img_draw = img.copy()
            for i, pt in enumerate(corners):
                cv2.circle(img_draw, tuple(pt.astype(int)), 8, (0,0,255), -1)
                cv2.putText(img_draw, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imwrite(str(debug_dir / f'corners_{idx:03d}.png'), img_draw)
            cv2.imwrite(str(debug_dir / f'warped_{idx:03d}.png'), warped)
            cv2.imwrite(str(debug_dir / f'warped_mask_{idx:03d}.png'), warped_mask)
    print(f'Warped {len(warped_imgs)} images to reference container.')

    # --- Stitch/blend toàn bộ ảnh đã warp bằng incremental affine + expand_canvas như stitch_color_auto.py ---
    if len(warped_imgs) < 2:
        print('Not enough warped images to stitch.')
        return
    if hasattr(cv2, 'SIFT_create'):
        detector = cv2.SIFT_create()
        norm_type = cv2.NORM_L2
    else:
        detector = cv2.ORB_create(4000)
        norm_type = cv2.NORM_HAMMING
    mosaic = warped_imgs[0].copy()
    mask_mosaic = warped_masks[0].copy()
    H_total = np.eye(3, dtype=np.float32)
    for i in range(1, len(warped_imgs)):
        img = warped_imgs[i]
        mask = warped_masks[i]
        # detect keypoints chỉ trên vùng container đã warp (không lấy background)
        kp1, des1 = detector.detectAndCompute(cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY), mask_mosaic)
        kp2, des2 = detector.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), mask)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            print(f'Not enough keypoints for image {i}')
            continue
        matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, 2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(good) < 6:
            print(f'Not enough good matches for image {i}')
            continue
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        # estimate affine
        M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            print(f'Affine estimation failed for image {i}')
            continue
        # convert to 3x3
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = M
        # expand canvas if needed
        h, w = mosaic.shape[:2]
        corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_c = np.concatenate([corners, warped_corners], axis=0)
        x_min, y_min = np.floor(all_c.min(axis=0).ravel()).astype(int)
        x_max, y_max = np.ceil(all_c.max(axis=0).ravel()).astype(int)
        tx, ty = -min(0, x_min), -min(0, y_min)
        new_w, new_h = int(x_max + tx), int(y_max + ty)
        T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float32)
        moved = cv2.warpPerspective(mosaic, T, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        moved_mask = cv2.warpPerspective(mask_mosaic, T, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        H_final = T @ H
        warped_img = cv2.warpPerspective(img, H_final, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        warped_mask = cv2.warpPerspective(mask, H_final, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # blend: ưu tiên vùng mask mới, còn lại giữ mosaic cũ
        mask_new = (warped_mask > 0)
        mask_old = (moved_mask > 0)
        out = moved.copy()
        only_new = mask_new & (~mask_old)
        overlap = mask_new & mask_old
        if np.any(only_new):
            out[only_new] = warped_img[only_new]
        if np.any(overlap):
            # blend vùng overlap bằng trung bình
            out[overlap] = ((moved[overlap].astype(np.float32) + warped_img[overlap].astype(np.float32))/2).astype(np.uint8)
        mosaic = out
        mask_mosaic = np.bitwise_or(moved_mask, warped_mask)
        if args.debug:
            cv2.imwrite(str(debug_dir / f'mosaic_{i:03d}.png'), mosaic)
            cv2.imwrite(str(debug_dir / f'mosaic_mask_{i:03d}.png'), mask_mosaic)
    out_path = Path(args.dir) / f'container_stitched_affine_full_{args.start}_{args.end}.jpg'
    cv2.imwrite(str(out_path), mosaic)
    print(f'Saved stitched image (all warped + affine, expand canvas): {out_path}')

if __name__ == '__main__':
    main()