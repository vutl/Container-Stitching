#!/usr/bin/env python3
"""Test if the updated _suppress function fixes the false split."""
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Import the updated suppress function
sys.path.insert(0, '/home/atin-tts-1/Container-Stitching')
from test_sides_split import _suppress_cross_class_overlaps

model = YOLO('last11scor_3class_31_07.pt')
img_dir = Path('downloads/gdrive_1sASbG/imgs_stit/left/MEDU9530977L5G1_10_10_2025_08_58_23_5350_45')

print("Testing frames 38-43 with updated suppress function:\n")

for i in range(38, 44):
    img_path = img_dir / f'img_{i}.jpg'
    if not img_path.exists():
        continue
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    # Run detection
    res = model.predict(img, conf=0.05, iou=0.6, verbose=False)
    r = res[0]
    
    boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
    scores = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
    
    classes = [r.names.get(cid, f'class_{cid}') for cid in cls_ids]
    
    # BEFORE suppress
    gu_before = sum(1 for c in classes if c == 'gu_cor')
    edge_before = sum(1 for c in classes if c == 'edge_cor')
    
    # AFTER suppress
    boxes_s, scores_s, classes_s = _suppress_cross_class_overlaps(boxes, scores, classes, iou_thresh=0.45)
    gu_after = sum(1 for c in classes_s if c == 'gu_cor')
    edge_after = sum(1 for c in classes_s if c == 'edge_cor')
    
    trigger = len(boxes_s) > 4
    status = "⚠️ SEAM TRIGGER" if trigger else "✓ OK"
    
    print(f"Frame {i}: BEFORE={len(boxes)} (gu={gu_before}, edge={edge_before}) → AFTER={len(boxes_s)} (gu={gu_after}, edge={edge_after}) {status}")
