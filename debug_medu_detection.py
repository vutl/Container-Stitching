#!/usr/bin/env python3
"""Debug script to check detections at specific frames."""
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

model = YOLO('last11scor_3class_31_07.pt')
img_dir = Path('downloads/gdrive_1sASbG/imgs_stit/left/MEDU9530977L5G1_10_10_2025_08_58_23_5350_45')

# Check frames around boundary (38-43)
for i in range(38, 44):
    img_path = img_dir / f'img_{i}.jpg'
    if not img_path.exists():
        print(f"Missing: {img_path}")
        continue
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Run detection
    res = model.predict(img, conf=0.05, iou=0.6, verbose=False)
    r = res[0]
    
    boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []
    scores = r.boxes.conf.cpu().numpy() if r.boxes is not None else []
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
    
    classes = []
    for cid in cls_ids:
        cname = r.names.get(cid, f'class_{cid}')
        classes.append(cname)
    
    # Count by class
    gu_count = sum(1 for c in classes if c == 'gu_cor')
    edge_count = sum(1 for c in classes if c == 'edge_cor')
    
    print(f"\nFrame {i}: total={len(boxes)} boxes (gu_cor={gu_count}, edge_cor={edge_count})")
    
    # Show boxes and check for overlaps
    for j, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        print(f"  Box {j}: {cls:8s} conf={score:.3f} center=({cx:.1f}, {cy:.1f}) box=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")
    
    # Check if gu_cor in same row
    if gu_count >= 2:
        gu_boxes = [box for box, cls in zip(boxes, classes) if cls == 'gu_cor']
        for i1 in range(len(gu_boxes)):
            for i2 in range(i1+1, len(gu_boxes)):
                b1 = gu_boxes[i1]
                b2 = gu_boxes[i2]
                cy1 = (b1[1] + b1[3]) / 2
                cy2 = (b2[1] + b2[3]) / 2
                cx1 = (b1[0] + b1[2]) / 2
                cx2 = (b2[0] + b2[2]) / 2
                dy = abs(cy1 - cy2)
                dx = abs(cx1 - cx2)
                dy_frac = dy / h
                dx_frac = dx / w
                if dy_frac < 0.08:  # same row threshold
                    print(f"  ⚠️  Two gu_cor in same row: dy_frac={dy_frac:.3f}, dx_frac={dx_frac:.3f} (min_dx_frac=0.12)")
                    if dx_frac < 0.12:
                        print(f"      → TOO CLOSE! This triggers seam split condition.")
