import cv2
import numpy as np
from pathlib import Path

# --- Config ---
THRESH_VAL = 12           # intensity threshold to treat as black (0-255)
ROW_ROI_TOP = 0.18        # only evaluate rows in [top, bottom] to focus on the container
ROW_ROI_BOTTOM = 0.88
COLUMN_FRACTION_THR = 0.98  # min fraction of non-black rows to keep a column
LEFT_EXTRA_PAD = 2
RIGHT_EXTRA_PAD = 2
MICRO_TRIM_MAX_PX = 4       # max extra pixels to trim per side for tiny slivers
MICRO_TRIM_FRACTION_THR = 0.99  # stricter threshold for micro-trim


def compute_horizontal_crop(img: np.ndarray):
    """Find left/right columns to remove curved black borders.

    Strategy:
    - Convert to grayscale and threshold to get non-black mask.
    - Consider only a vertical ROI where the container is visible (to ignore top/bottom background).
    - For each column, compute the fraction of rows that are non-black.
    - Choose first/last column where this fraction exceeds a high threshold.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Non-black mask
    non_black = gray > THRESH_VAL

    # Focus on container rows (avoid top/bottom clutter)
    r0 = int(h * ROW_ROI_TOP)
    r1 = int(h * ROW_ROI_BOTTOM)
    if r1 <= r0:
        r0, r1 = int(h * 0.2), int(h * 0.8)

    roi = non_black[r0:r1, :]

    # Column-wise fraction of non-black pixels
    col_frac = roi.mean(axis=0)  # values in [0,1]

    # Find leftmost column with enough non-black pixels
    left = 0
    for i in range(w):
        if col_frac[i] >= COLUMN_FRACTION_THR:
            left = max(0, i - LEFT_EXTRA_PAD)
            break

    # Find rightmost column similarly
    right = w - 1
    for i in range(w - 1, -1, -1):
        if col_frac[i] >= COLUMN_FRACTION_THR:
            right = min(w - 1, i + RIGHT_EXTRA_PAD)
            break

    # Micro-trim pass: remove tiny residual black columns at the very edges
    # Trim a few pixels if their non-black fraction is still too low
    # Left edge micro-trim
    lt = 0
    while (left + lt < w) and (lt < MICRO_TRIM_MAX_PX) and (col_frac[left + lt] < MICRO_TRIM_FRACTION_THR):
        lt += 1
    left = min(left + lt, w - 2)

    # Right edge micro-trim
    rt = 0
    while (right - rt >= 0) and (rt < MICRO_TRIM_MAX_PX) and (col_frac[right - rt] < MICRO_TRIM_FRACTION_THR):
        rt += 1
    right = max(right - rt, left + 1)

    # Ensure valid crop window and at least some width left
    if right - left < max(32, int(0.2 * w)):
        # Fallback: use 10% margins if detection failed
        left = int(0.1 * w)
        right = int(0.9 * w)

    return left, right


def crop_black_sides(image_path: Path, out_suffix="_cropped") -> Path:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    left, right = compute_horizontal_crop(img)

    cropped = img[:, left:right+1]

    out_path = image_path.with_name(image_path.stem + out_suffix + image_path.suffix)
    cv2.imwrite(str(out_path), cropped)

    print(f"Cropped {image_path.name}: left={left}, right={right} -> {out_path.name}")
    return out_path


def main():
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_14-35-53")
    # Process a continuous range 43..74 (inclusive)
    files = [base / f"{i}.jpg" for i in range(0, 2)]

    for fp in files:
        if not fp.exists():
            print(f"Missing: {fp}")
            continue
        crop_black_sides(fp)

    print("Done. You can now stitch using the *_cropped.jpg images (43..74).")


if __name__ == "__main__":
    main()
