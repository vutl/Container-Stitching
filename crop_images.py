import cv2
import numpy as np
from pathlib import Path
import argparse

# --- Config ---
THRESH_VAL = 12           # Cường độ để coi là màu đen (0-255)
ROW_ROI_TOP = 0.18        # Chỉ đánh giá các hàng trong khoảng [top, bottom] để tập trung vào container
ROW_ROI_BOTTOM = 0.88
COLUMN_FRACTION_THR = 0.98  # Tỷ lệ tối thiểu các hàng không phải màu đen để giữ lại một cột
# Điều chỉnh để cắt sát hơn theo yêu cầu
LEFT_EXTRA_PAD = 0
RIGHT_EXTRA_PAD = 0
MICRO_TRIM_MAX_PX = 4       # Số pixel tối đa để cắt tỉa thêm cho các dải đen nhỏ
MICRO_TRIM_FRACTION_THR = 0.99  # Ngưỡng chặt chẽ hơn cho việc cắt tỉa vi mô


def compute_horizontal_crop(img: np.ndarray):
    """Tìm các cột trái/phải để loại bỏ viền đen cong."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Mask cho các pixel không phải màu đen
    non_black = gray > THRESH_VAL

    # Tập trung vào các hàng của container
    r0 = int(h * ROW_ROI_TOP)
    r1 = int(h * ROW_ROI_BOTTOM)
    if r1 <= r0:
        r0, r1 = int(h * 0.2), int(h * 0.8)

    roi = non_black[r0:r1, :]

    # Tỷ lệ pixel không đen theo từng cột
    col_frac = roi.mean(axis=0)

    # Tìm cột ngoài cùng bên trái có đủ pixel không đen
    left = 0
    for i in range(w):
        if col_frac[i] >= COLUMN_FRACTION_THR:
            left = max(0, i - LEFT_EXTRA_PAD)
            break

    # Tìm cột ngoài cùng bên phải tương tự
    right = w - 1
    for i in range(w - 1, -1, -1):
        if col_frac[i] >= COLUMN_FRACTION_THR:
            right = min(w - 1, i + RIGHT_EXTRA_PAD)
            break

    # Cắt tỉa vi mô: loại bỏ các cột đen nhỏ còn sót lại ở các cạnh
    lt = 0
    while (left + lt < w) and (lt < MICRO_TRIM_MAX_PX) and (col_frac[left + lt] < MICRO_TRIM_FRACTION_THR):
        lt += 1
    left = min(left + lt, w - 2)

    rt = 0
    while (right - rt >= 0) and (rt < MICRO_TRIM_MAX_PX) and (col_frac[right - rt] < MICRO_TRIM_FRACTION_THR):
        rt += 1
    right = max(right - rt, left + 1)

    # Đảm bảo cửa sổ cắt hợp lệ
    if right - left < max(32, int(0.2 * w)):
        # Fallback: sử dụng lề 10% nếu phát hiện thất bại
        left = int(0.1 * w)
        right = int(0.9 * w)

    return left, right


def crop_single_image(image_path: Path, out_suffix="_cropped") -> Path:
    """Đọc, cắt và lưu một ảnh."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    left, right = compute_horizontal_crop(img)

    cropped = img[:, left:right+1]

    out_path = image_path.with_name(image_path.stem + out_suffix + image_path.suffix)
    cv2.imwrite(str(out_path), cropped)

    print(f"Đã cắt {image_path.name}: left={left}, right={right} -> {out_path.name}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Công cụ cắt viền đen cho ảnh container.")
    parser.add_argument("image_dir", type=str, help="Thư mục chứa ảnh cần cắt.")
    parser.add_argument("start_idx", type=int, help="Chỉ số bắt đầu của dãy ảnh.")
    parser.add_argument("end_idx", type=int, help="Chỉ số kết thúc của dãy ảnh (bao gồm).")
    parser.add_argument("--suffix", type=str, default=".jpg", help="Phần đuôi của file ảnh đầu vào (ví dụ: '.jpg').")
    
    args = parser.parse_args()

    base_path = Path(args.image_dir)
    if not base_path.is_dir():
        print(f"Lỗi: Không tìm thấy thư mục tại '{args.image_dir}'")
        return

    print(f"--- Bắt đầu quá trình cắt viền cho thư mục: {base_path.name} ---")
    
    for i in range(args.start_idx, args.end_idx + 1):
        fp = base_path / f"{i}{args.suffix}"
        if not fp.exists():
            print(f"Bỏ qua file không tồn tại: {fp.name}")
            continue
        try:
            crop_single_image(fp)
        except Exception as e:
            print(f"Lỗi khi xử lý file {fp.name}: {e}")
            
    print("\n--- Hoàn tất việc cắt viền. ---")


if __name__ == "__main__":
    main()
