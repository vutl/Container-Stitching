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
    comp_mask = mask.copy()
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        big = np.zeros_like(mask)
        cv2.drawContours(big, contours, idx, 255, cv2.FILLED)
        mask = big
    return mask

def crop_to_mask(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, mask
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_img, cropped_mask

def main():
    parser = argparse.ArgumentParser(description='Stitch color-auto, crop each image to container mask before blending')
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
    debug_dir = base / 'debug_crop'
    if args.debug:
        debug_dir.mkdir(exist_ok=True)
    cropped_imgs = []
    for idx, img in enumerate(imgs):
        mask = color_mask_from_bounds(img, lower, upper)
        cropped_img, cropped_mask = crop_to_mask(img, mask)
        cropped_imgs.append(cropped_img)
        if args.debug:
            cv2.imwrite(str(debug_dir / f'mask_{idx:03d}.png'), cropped_mask)
            cv2.imwrite(str(debug_dir / f'crop_{idx:03d}.png'), cropped_img)
    # Resize all cropped images to the same size as the first crop for blending
    h0, w0 = cropped_imgs[0].shape[:2]
    acc = np.zeros((h0, w0, 3), np.float32)
    count = np.zeros((h0, w0), np.float32)
    for idx, img in enumerate(cropped_imgs):
        img_resized = cv2.resize(img, (w0, h0), interpolation=cv2.INTER_LINEAR)
        acc += img_resized
        count += 1
    out = (acc / count[..., None]).astype(np.uint8)
    out_path = base / f'mosaic_color_crop_{args.start}_{args.end}.jpg'
    cv2.imwrite(str(out_path), out)
    print(f'Saved -> {out_path}')

if __name__ == '__main__':
    main()
