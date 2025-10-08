import cv2
import numpy as np
from pathlib import Path
import argparse
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
    # Giới hạn mask trong vùng container ROI để loại bỏ background
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
def container_roi(img, top=0.18, bottom=0.88, left=0.08, right=0.92):
    h, w = img.shape[:2]
    r0, r1 = int(h * top), int(h * bottom)
    c0, c1 = int(w * left), int(w * right)
    mask = np.zeros((h, w), np.uint8)
    mask[r0:r1, c0:c1] = 255
    return mask
def find_container_corners(mask):
    # Tìm contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    # Dùng approxPolyDP để tìm 4 điểm góc
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    if len(approx) == 4:
        corners = approx[:, 0, :]
    else:
        # Nếu không ra đúng 4 điểm, dùng bounding box
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
    return ordered
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
    parser = argparse.ArgumentParser(description='Find 4 corners of container in each image')
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
    debug_dir = base / 'debug_corners'
    if args.debug:
        debug_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(imgs):
        mask = color_mask_from_bounds(img, lower, upper)
        corners = find_container_corners(mask)
        if corners is not None:
            print(f'Image {idx}: corners = {corners.tolist()}')
            if args.debug:
                img_draw = img.copy()
                for i, pt in enumerate(corners):
                    cv2.circle(img_draw, tuple(pt.astype(int)), 8, (0,0,255), -1)
                    cv2.putText(img_draw, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imwrite(str(debug_dir / f'corners_{idx:03d}.png'), img_draw)
        else:
            print(f'Image {idx}: No corners found!')
if __name__ == '__main__':
    main()