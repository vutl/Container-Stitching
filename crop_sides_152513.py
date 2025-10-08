from pathlib import Path
# Import logic đã được kiểm chứng từ file crop gốc
from crop_black_sides import crop_black_sides


def main():
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_15-25-13")
    
    # Dãy ảnh cho container màu đỏ là từ 44 đến 90
    image_indices = range(44, 91)
    
    print(f"--- Bắt đầu quá trình cắt viền cho thư mục: {base.name} (sử dụng logic gốc) ---")
    for i in image_indices:
        fp = base / f"{i}.jpg"
        if not fp.exists():
            print(f"Bỏ qua file không tồn tại: {fp.name}")
            continue
        try:
            crop_black_sides(fp)
        except Exception as e:
            print(f"Lỗi khi xử lý file {fp.name}: {e}")
            
    print("\n--- Hoàn tất việc cắt viền. ---")


if __name__ == "__main__":
    main()
