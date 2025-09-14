import cv2
import numpy as np
from pathlib import Path
import argparse

# --- Warp all images to the first image's container mask ---
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

def get_affine_from_mask(src_mask, dst_mask):
    # Find contours and get largest
    cnts1, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2, _ = cv2.findContours(dst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts1 or not cnts2:
        return None
    c1 = max(cnts1, key=cv2.contourArea)
    c2 = max(cnts2, key=cv2.contourArea)
    # Get 4 points from bounding rect
    x1, y1, w1, h1 = cv2.boundingRect(c1)
    x2, y2, w2, h2 = cv2.boundingRect(c2)
    src_pts = np.float32([[x1, y1], [x1+w1, y1], [x1, y1+h1]])
    dst_pts = np.float32([[x2, y2], [x2+w2, y2], [x2, y2+h2]])
    M = cv2.getAffineTransform(src_pts, dst_pts)
    return M

def main():
    parser = argparse.ArgumentParser(description='Warp all images to first container mask and blend')
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
    # Sample color from first image
    lower, upper = sample_hsv_range(imgs[0])
    mask0 = color_mask_from_bounds(imgs[0], lower, upper)
    h0, w0 = imgs[0].shape[:2]
    acc = np.zeros((h0, w0, 3), np.float32)
    count = np.zeros((h0, w0), np.float32)
    debug_dir = base / 'debug_warp'
    if args.debug:
        debug_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(debug_dir / f'mask_000.png'), mask0)
    for idx, img in enumerate(imgs):
        # Step 1: tạo mask sạch như stitch_color_auto
        mask = color_mask_from_bounds(img, lower, upper)
        if idx == 0:
            warped = img.copy()
            warped_mask = mask.copy()
        else:
            M = get_affine_from_mask(mask, mask0)
            if M is None:
                print(f'Cannot get affine for {paths[idx]}')
                continue
            warped = cv2.warpAffine(img, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            warped_mask = cv2.warpAffine(mask, M, (w0, h0), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Sau khi warp, lại lọc contour lớn nhất để loại bỏ background bị kéo dãn
            comp_mask = warped_mask.copy()
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                areas = [cv2.contourArea(c) for c in contours]
                idx_max = int(np.argmax(areas))
                big = np.zeros_like(warped_mask)
                cv2.drawContours(big, contours, idx_max, 255, cv2.FILLED)
                warped_mask = big
        if args.debug:
            cv2.imwrite(str(debug_dir / f'mask_{idx:03d}.png'), warped_mask)
        # Blend: accumulate only where mask
        mask_bool = (warped_mask > 0)
        acc[mask_bool] += warped[mask_bool]
        count[mask_bool] += 1
    # Normalize
    out = np.zeros_like(imgs[0])
    valid = count > 0
    out[valid] = (acc[valid] / count[valid, None]).astype(np.uint8)
    out_path = base / f'mosaic_warp_to_first_{args.start}_{args.end}.jpg'
    cv2.imwrite(str(out_path), out)
    print(f'Saved -> {out_path}')

if __name__ == '__main__':
    main()
