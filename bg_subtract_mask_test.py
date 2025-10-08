import cv2
import numpy as np
from pathlib import Path

# List of test frames (can adjust as needed)
test_indices = [49, 60, 80, 100, 107]
base_dir = Path("/home/atin-tts-1/vutl/Container/mnt/2025-09-04_14-46-47")
background_path = "0_cropped.jpg"

# Read background image
bg = cv2.imread(str(background_path))
if bg is None:
    raise FileNotFoundError(f"Background image not found: {background_path}")

for idx in test_indices:
    img_path = base_dir / f"{idx}_cropped.jpg"
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Missing {img_path}")
        continue
    # Resize background if needed
    if img.shape != bg.shape:
        bg_resized = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        bg_resized = bg
    # Absolute difference
    diff = cv2.absdiff(img, bg_resized)
    # Convert to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold to get mask
    _, mask = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
    # Morphology to clean up
    mask = cv2.medianBlur(mask, 7)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8), iterations=2)
    # Keep largest component
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx_max = int(np.argmax(areas))
        big = np.zeros_like(mask)
        cv2.drawContours(big, contours, idx_max, 255, cv2.FILLED)
        mask = big
    # Overlay for visualization
    overlay = img.copy()
    color_mask = np.zeros_like(img)
    color_mask[:,:,1] = 180  # green
    overlay = cv2.addWeighted(overlay, 0.7, color_mask, 0.3, 0)
    overlay[mask>0] = cv2.addWeighted(img[mask>0], 0.5, color_mask[mask>0], 0.5, 0)
    # Save results
    out_dir = base_dir / "bg_subtract_test"
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / f"mask_{idx}.png"), mask)
    cv2.imwrite(str(out_dir / f"overlay_{idx}.png"), overlay)
    print(f"Saved mask and overlay for frame {idx}")
