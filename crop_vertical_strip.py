import cv2
import numpy as np
from pathlib import Path
import argparse

def find_vertical_bounds(imgs, threshold=8):
    """Tìm min-max hàng (y) có pixel khác 0 trên toàn bộ dải ảnh."""
    # Dùng profile dọc (mean theo trục ngang) để tìm vùng container
    profiles = []
    for img in imgs:
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        profiles.append(gray.mean(axis=1))
    # Trung bình profile qua tất cả ảnh
    mean_profile = np.mean(np.stack(profiles, axis=1), axis=1)
    # Tìm vùng liên tục lớn nhất có mean vượt ngưỡng (ngưỡng tự động: 60% giá trị max-min)
    minval, maxval = float(mean_profile.min()), float(mean_profile.max())
    thresh = minval + 0.6 * (maxval - minval)
    mask = mean_profile > thresh
    # Tìm đoạn liên tục dài nhất
    from itertools import groupby
    best_run = (0, 0)
    run_start = None
    for i, v in enumerate(mask):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if best_run == (0, 0) or (i - run_start) > (best_run[1] - best_run[0]):
                best_run = (run_start, i-1)
            run_start = None
    if run_start is not None:
        if best_run == (0, 0) or (len(mask) - run_start) > (best_run[1] - best_run[0]):
            best_run = (run_start, len(mask)-1)
    min_y, max_y = best_run
    print(f"[DEBUG] Vertical mean profile: min={minval:.1f}, max={maxval:.1f}, thresh={thresh:.1f}, crop rows {min_y}:{max_y}")
    return min_y, max_y

def main():
    parser = argparse.ArgumentParser(description='Crop all images in a range to the same vertical strip (container region)')
    parser.add_argument('dir', help='directory with images')
    parser.add_argument('--start', type=int, default=42)
    parser.add_argument('--end', type=int, default=74)
    parser.add_argument('--suffix', default='_cropped.jpg')
    parser.add_argument('--out', default='cropped_vertical')
    args = parser.parse_args()

    base = Path(args.dir)
    paths = [base / f"{i}{args.suffix}" for i in range(args.start, args.end+1)]
    imgs = [cv2.imread(str(p)) for p in paths if p.exists()]
    if not imgs:
        print('No images found!')
        return
    # --- CROP THỦ CÔNG: SỬA 2 GIÁ TRỊ NÀY CHO PHÙ HỢP ---
    min_y = 300  # <-- Sửa giá trị này nếu cần
    max_y = 1600 # <-- Sửa giá trị này nếu cần
    print(f"[MANUAL] Cropping all images to rows {min_y}:{max_y} (height={max_y-min_y+1}) and overwriting original files.")
    count = 0
    for p, img in zip(paths, imgs):
        crop = img[min_y:max_y+1, :]
        cv2.imwrite(str(p), crop)
        count += 1
    print(f"Overwrote {count} images with cropped versions.")

if __name__ == '__main__':
    main()
