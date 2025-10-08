from pathlib import Path
from crop_black_sides import crop_black_sides


def main():
    base = Path(r"/home/atin-tts-1/vutl/Container/mnt/2025-09-04_14-46-47")
    # Range 49..106 inclusive
    for i in range(48, 108):
        fp = base / f"{i}.jpg"
        if not fp.exists():
            print(f"Missing: {fp}")
            continue
        crop_black_sides(fp)
    print("Done cropping 49..106 to *_cropped.jpg")


if __name__ == "__main__":
    main()
