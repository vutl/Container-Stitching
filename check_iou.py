#!/usr/bin/env python3
"""Check IoU between overlapping edge_cor boxes."""
import cv2
from pathlib import Path
from ultralytics import YOLO

def compute_iou(a, b):
    x1, y1, x2, y2 = a
    x1b, y1b, x2b, y2b = b
    xa1 = max(x1, x1b); ya1 = max(y1, y1b)
    xa2 = min(x2, x2b); ya2 = min(y2, y2b)
    if xa2 <= xa1 or ya2 <= ya1:
        return 0.0
    inter = (xa2 - xa1) * (ya2 - ya1)
    area_a = max(0, (x2 - x1)) * max(0, (y2 - y1))
    area_b = max(0, (x2b - x1b)) * max(0, (y2b - y1b))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

model = YOLO('last11scor_3class_31_07.pt')
img_path = Path('downloads/gdrive_1sASbG/imgs_stit/left/MEDU9530977L5G1_10_10_2025_08_58_23_5350_45/img_40.jpg')

img = cv2.imread(str(img_path))
res = model.predict(img, conf=0.05, iou=0.6, verbose=False)
r = res[0]

boxes = r.boxes.xyxy.cpu().numpy().tolist()
scores = r.boxes.conf.cpu().numpy().tolist()
cls_ids = r.boxes.cls.cpu().numpy().astype(int)
classes = [r.names.get(cid, f'class_{cid}') for cid in cls_ids]

print(f"Frame 40: {len(boxes)} boxes\n")
for i, (box, cls, score) in enumerate(zip(boxes, classes, scores)):
    x1, y1, x2, y2 = box
    print(f"Box {i}: {cls:8s} conf={score:.3f} box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

print("\nIoU matrix (only edge_cor pairs):")
edge_indices = [i for i, c in enumerate(classes) if c == 'edge_cor']
for i in edge_indices:
    for j in edge_indices:
        if i < j:
            iou_val = compute_iou(boxes[i], boxes[j])
            print(f"  Box {i} vs Box {j}: IoU = {iou_val:.4f} {'← OVERLAP!' if iou_val > 0.45 else ''}")
