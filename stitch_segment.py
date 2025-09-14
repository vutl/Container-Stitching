import cv2
import numpy as np
from pathlib import Path
import argparse

try:
    from stitch_generic import container_roi, expand_canvas, get_detector, is_reasonable_transform, crop_final_mosaic
except Exception:
    # minimal fallbacks
    def container_roi(img, top=0.18, bottom=0.88, left=0.08, right=0.92):
        h, w = img.shape[:2]
        r0, r1 = int(h * top), int(h * bottom)
        c0, c1 = int(w * left), int(w * right)
        mask = np.zeros((h, w), np.uint8)
        mask[r0:r1, c0:c1] = 255
        return mask

    def expand_canvas(mosaic, h_to_canvas, new_img):
        h, w = new_img.shape[:2]
        h_canvas, w_canvas = mosaic.shape[:2]
        T = np.eye(3, dtype=np.float32)
        new_w, new_h = w_canvas + w, max(h_canvas, h)
        moved = cv2.warpPerspective(mosaic, T, (new_w, new_h))
        h_final = T @ h_to_canvas
        warped_new = cv2.warpPerspective(new_img, h_final, (new_w, new_h))
        out = moved.copy()
        out[:warped_new.shape[0], :warped_new.shape[1]] = warped_new
        return out, T

    def get_detector():
        if hasattr(cv2, 'SIFT_create'):
            return cv2.SIFT_create(), 'SIFT'
        if hasattr(cv2, 'xfeatures2d') and hasattr(cv2.xfeatures2d, 'SIFT_create'):
            return cv2.xfeatures2d.SIFT_create(), 'SIFT'
        return cv2.ORB_create(4000), 'ORB'

    def is_reasonable_transform(H, max_scale=1.8, min_scale=0.55):
        if H is None or not np.isfinite(H).all():
            return False
        a = H[0, 0]; b = H[0, 1]; c = H[1, 0]; d = H[1, 1]
        s = np.sqrt(max(1e-8, (a * a + d * d + b * b + c * c) / 2.0))
        return not (s > max_scale or s < min_scale)

    def crop_final_mosaic(mosaic):
        gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return mosaic
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return mosaic[y:y+h, x:x+w]


def create_segmentation_mask(img, rect=None, iterCount=5, debug=False):
    """Create a binary mask for the container using GrabCut.
    - rect: (x,y,w,h) initial rectangle. If None, derive from `container_roi` bounding box.
    - iterCount: iterations for grabCut.
    Returns uint8 mask (0/255).
    """
    h, w = img.shape[:2]

    if rect is None:
        roi_mask = container_roi(img)
        ys, xs = np.where(roi_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            # full image fallback
            rect = (int(w*0.1), int(h*0.1), int(w*0.8), int(h*0.8))
        else:
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            # enlarge a bit
            pad_x = int(0.05 * (x1 - x0 + 1))
            pad_y = int(0.05 * (y1 - y0 + 1))
            x0 = max(0, x0 - pad_x)
            y0 = max(0, y0 - pad_y)
            x1 = min(w-1, x1 + pad_x)
            y1 = min(h-1, y1 + pad_y)
            rect = (x0, y0, x1 - x0, y1 - y0)

    mask_gc = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img, mask_gc, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
    except Exception:
        # fallback to simple ROI
        seg = np.zeros((h, w), np.uint8)
        x, y, rw, rh = rect
        seg[y:y+rh, x:x+rw] = 255
        return seg

    mask2 = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD), 255, 0).astype('uint8')

    # Morphological cleanup
    kernel = np.ones((7, 7), np.uint8)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)

    # Keep largest connected component to avoid stray bits
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        max_idx = int(np.argmax(areas))
        filled = np.zeros_like(mask2)
        cv2.drawContours(filled, contours, max_idx, (255,), thickness=cv2.FILLED)
        # small closing to fill holes
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((9,9), np.uint8), iterations=2)
        mask2 = filled

    if debug:
        return mask2, rect
    return mask2


def estimate_transform_affine_segment(src, tar, matcher_ratio=0.75, min_matches=6):
    gray1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)

    mask1 = create_segmentation_mask(src)
    mask2 = create_segmentation_mask(tar)

    det, name = get_detector()
    kp1, des1 = det.detectAndCompute(gray1, mask1)
    kp2, des2 = det.detectAndCompute(gray2, mask2)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    norm = cv2.NORM_L2 if name == 'SIFT' else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(norm, crossCheck=False)
    raw = bf.knnMatch(des1, des2, 2)
    good = []
    for m_n in raw:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < matcher_ratio * n.distance:
                good.append(m)
    if len(good) < min_matches:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    A, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if A is None:
        return None
    H = np.eye(3, dtype=np.float32)
    H[:2, :] = A
    return H


def stitch_incremental_segment(image_paths, debug_dir=None):
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
            print(f"  -> Skip missing previous {image_paths[i-1]}")
            continue

        H = estimate_transform_affine_segment(prev, img)
        if not is_reasonable_transform(H):
            print("  -> Segment-based affine unreasonable; fallback to ECC translation")
            H = None

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
                print("  -> ECC failed. Skipping frame.")
                continue

        try:
            g_cur = G_cumulative[-1] @ np.linalg.inv(H)
            G_cumulative.append(g_cur)
        except np.linalg.LinAlgError:
            print("  -> Singular matrix. Skipping frame.")
            continue

        h_to_canvas = G0_to_canvas @ g_cur
        mosaic_new, T = expand_canvas(mosaic, h_to_canvas, img)
        if mosaic_new is None:
            print("  -> Skip frame due to excessive canvas growth.")
            G_cumulative.pop()
            continue

        mosaic = mosaic_new
        G0_to_canvas = T @ G0_to_canvas
        print(f"  -> Stitched. New canvas size: {mosaic.shape[1]}x{mosaic.shape[0]}")

    return mosaic


def main():
    parser = argparse.ArgumentParser(description="Stitch using segmentation-based masks (GrabCut) for container region")
    parser.add_argument("image_dir", type=str)
    parser.add_argument("start_idx", type=int)
    parser.add_argument("end_idx", type=int)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--suffix", type=str, default="_cropped.jpg")
    parser.add_argument("--output", type=str, default="mosaic_segment.jpg")
    parser.add_argument("--grab-iter", type=int, default=5, help="GrabCut iterations")
    parser.add_argument("--debug", action="store_true", help="Save segmentation debug masks for first few frames")
    args = parser.parse_args()

    base_path = Path(args.image_dir)
    paths = [base_path / f"{args.prefix}{i}{args.suffix}" for i in range(args.start_idx, args.end_idx + 1)]
    existing = [p for p in paths if p.exists()]
    if len(existing) < 2:
        print("Need at least 2 images to stitch")
        return

    debug_dir = None
    if args.debug:
        debug_dir = base_path / "debug_segment"
        debug_dir.mkdir(exist_ok=True)

    print("--- Starting segmentation-based stitching ---")
    mosaic = stitch_incremental_segment(existing, debug_dir)

    if mosaic is not None and mosaic.size > 0:
        final = crop_final_mosaic(mosaic)
        out_path = base_path / args.output
        cv2.imwrite(str(out_path), final)
        print(f"Saved final mosaic -> {out_path}")
    else:
        print("Stitch failed or produced empty result")


if __name__ == "__main__":
    main()
