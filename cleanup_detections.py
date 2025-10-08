#!/usr/bin/env python3
"""
Read existing out_annot/*_annot_cropped.txt files, remove overlapping boxes using IoU
and a priority ordering (det > pred_3 > pred_2 > pred_1), then write cleaned txt and
an annotated image for quick inspection.

Produces files:
- out_annot/<idx>_annot_cropped_dedup.txt
- out_annot/<idx>_annot_cropped_dedup.jpg

This is a non-destructive post-processing helper to address the "stacked/overlapping
predicted corners" symptom.
"""
import glob
import os
import cv2
import numpy as np
from pathlib import Path


def iou(boxA, boxB):
    # boxes are [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    inter = interW * interH
    boxAArea = (boxA[2]-boxA[0]+1) * (boxA[3]-boxA[1]+1)
    boxBArea = (boxB[2]-boxB[0]+1) * (boxB[3]-boxB[1]+1)
    denom = float(boxAArea + boxBArea - inter)
    if denom <= 0:
        return 0.0
    return inter / denom


def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    lab = parts[0]
    try:
        x1 = int(float(parts[1])); y1 = int(float(parts[2])); x2 = int(float(parts[3])); y2 = int(float(parts[4]))
    except Exception:
        return None
    score = None
    if len(parts) >= 6:
        try:
            score = float(parts[5])
        except Exception:
            score = None
    return (lab, [x1,y1,x2,y2], score, line.strip())


def draw_boxes(img, boxes, labels, scores=None):
    out = img.copy()
    color_map = {
        "det":     (0, 255,   0),   # green
        "pred_3":  (255,128,  0),   # orange
        "pred_2":  (0, 128, 255),   # blue
        "pred_1":  (180,  0, 180),  # purple
    }
    for i,b in enumerate(boxes):
        x1,y1,x2,y2 = b
        lab = labels[i]
        col = color_map.get(lab, (0,255,0))
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        txt = lab if scores is None or scores[i] is None else f"{lab} {scores[i]:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), col, -1)
        cv2.putText(out, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out


def cleanup_file(txt_path, out_dir, iou_thr=0.35):
    basename = os.path.basename(txt_path)
    if not basename.endswith('_annot_cropped.txt'):
        return
    idx = basename.split('_')[0]
    img_candidates = [
        f"mnt/2025-09-04_14-35-53/{idx}_cropped.jpg",
        f"mnt/2025-09-04_14-46-47/{idx}_cropped.jpg",
        f"mnt/2025-09-04_14-35-53/{idx}.jpg",
    ]
    img_path = None
    for c in img_candidates:
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        # try any jpg with idx prefix
        for p in glob.glob('mnt/**/*.jpg', recursive=True):
            if os.path.basename(p).startswith(idx + '_') or os.path.basename(p).startswith(idx + '.'):
                img_path = p; break
    if img_path is None:
        print(f"[skip] no image for {txt_path}")
        return

    with open(txt_path, 'r') as f:
        lines = [l for l in f.readlines() if l.strip()]
    parsed = [parse_line(l) for l in lines]
    parsed = [p for p in parsed if p is not None]
    if not parsed:
        print(f"[skip] no boxes in {txt_path}")
        return

    # priority: lower number means higher priority
    priority = {'det':0, 'pred_3':1, 'pred_2':2, 'pred_1':3}

    # sort by (priority, -score if present, area small->large) so best kept first
    def sort_key(item):
        lab, box, score, raw = item
        pr = priority.get(lab, 5)
        s = -score if score is not None else 0.0
        area = (box[2]-box[0])*(box[3]-box[1])
        return (pr, s, area)

    parsed_sorted = sorted(parsed, key=sort_key)

    kept = []
    for lab, box, score, raw in parsed_sorted:
        skip = False
        for _, kbox, _, _ in kept:
            if iou(box, kbox) > iou_thr:
                skip = True
                break
        if not skip:
            kept.append((lab, box, score, raw))

    # write deduped txt and image
    out_txt = os.path.join(out_dir, f"{idx}_annot_cropped_dedup.txt")
    with open(out_txt, 'w') as f:
        for lab, box, score, raw in kept:
            if score is None:
                f.write(f"{lab} {box[0]} {box[1]} {box[2]} {box[3]}\n")
            else:
                f.write(f"{lab} {box[0]} {box[1]} {box[2]} {box[3]} {score}\n")

    img = cv2.imread(img_path)
    boxes = [b for _, b, _, _ in kept]
    labels = [l for l, _, _, _ in kept]
    scores = [s for _, _, s, _ in kept]
    out_img = draw_boxes(img, boxes, labels, scores)
    out_img_path = os.path.join(out_dir, f"{idx}_annot_cropped_dedup.jpg")
    cv2.imwrite(out_img_path, out_img)
    print(f"Wrote dedup: {out_txt}  {out_img_path}")


def main():
    out_dir = 'out_annot'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    files = glob.glob(os.path.join(out_dir, '*_annot_cropped.txt'))
    if not files:
        print('No annotation txt files found in out_annot/')
        return
    for t in files:
        cleanup_file(t, out_dir)


if __name__ == '__main__':
    main()
