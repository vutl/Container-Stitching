#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Same as test.py but use YOLO class names to decide head vs regular corners.
- If a detection's class name is 'gu_cor' we treat it as container-head corner
    (use the head-offset rule to place the corner).
- If class name is 'edge_cor' we treat it as a regular corner and run the
    gradient-based refinement.
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from corner_completion import box_center, assign_quadrant, VectorCornerCompleter
from corner_refinement_sides import CornerRefinerParams, refine_corners, CornerCandidate


# ---------- drawing ----------
def draw_boxes(img, boxes, classes, scores=None):
    out = img.copy()
    color = {
        "det":    (0,255,0),      # green
        "pred_3": (255,128,0),    # orange
    }
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        cls = classes[i] if classes is not None and i < len(classes) else 'det'
        lab = 'det' if cls is None else cls
        col = color.get(lab, (0,255,0))
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        txt = lab if scores is None or scores[i] is None else f"{lab} {scores[i]:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), col, -1)
        cv2.putText(out, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out


# ---------- YOLO (get classes) ----------
def load_model(model_path):
    try:
        from ultralytics import YOLO
        return YOLO(str(model_path))
    except Exception as e:
        print("Ultralytics not available:", e)
        raise


def run_detect(model, img, conf, iou):
    res = model.predict(img, conf=conf, iou=iou, verbose=False)
    r = res[0] if isinstance(res, (list,tuple)) else res
    boxes, scores, classes = [], [], []
    try:
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()
        names = None
        try:
            names = model.model.names
        except Exception:
            try:
                names = model.names
            except Exception:
                names = None
        for bb, sc, cc in zip(b, s, cls):
            boxes.append(bb.tolist())
            scores.append(float(sc))
            if names is None:
                classes.append(str(int(cc)))
            else:
                classes.append(str(names.get(int(cc), str(int(cc)))))
    except Exception:
        pass
    return boxes, scores, classes


def keep_best_per_quadrant(boxes, scores, classes, w, h, max_per_quad=1):
    quads = {"TL":[], "TR":[], "BR":[], "BL":[]}
    for b, s, c in zip(boxes, scores, classes):
        cx, cy = box_center(b)
        q = assign_quadrant(cx, cy, w, h)
        quads[q].append((b, s, c))
    kept_boxes, kept_scores, kept_classes = [], [], []
    for q in quads:
        quads[q].sort(key=lambda x: x[1], reverse=True)
        for item in quads[q][:max_per_quad]:
            b, s, c = item
            kept_boxes.append(b); kept_scores.append(s); kept_classes.append(c)
    return kept_boxes, kept_scores, kept_classes


# ---------- main (sides-aware) ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='last11scor_3class_31_07.pt')
    ap.add_argument('--dir', type=str, required=True)
    ap.add_argument('--indices', type=str, nargs='+', required=True,
                    help="numbers or ranges like 54 60-65")
    ap.add_argument('--suffix', type=str, default='_cropped_padded.jpg')
    ap.add_argument('--out', type=str, required=True, default='out_annot_sides')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--max-per-quad', type=int, default=1)
    ap.add_argument('--debug', action='store_true')

    ap.add_argument('--no-corners', action='store_true',
                    help='Skip gradient-based corner refinement outputs')
    ap.add_argument('--corner-side-margin-frac', type=float, default=0.12)
    ap.add_argument('--corner-center-window-frac', type=float, default=0.45)
    ap.add_argument('--corner-center-near-px', type=int, default=4,
                    help='Radius around bbox center to check before margin search')
    ap.add_argument('--corner-center-thr-alpha', type=float, default=0.5,
                    help='Alpha multiplier for center thresholds')
    ap.add_argument('--corner-min-side-px', type=int, default=6)
    ap.add_argument('--corner-sobel-ksize', type=int, default=3)
    ap.add_argument('--corner-perc', type=float, default=90.0)
    ap.add_argument('--corner-perc-alpha', type=float, default=0.45)
    ap.add_argument('--corner-std-mult', type=float, default=1.0)
    ap.add_argument('--corner-refine-half-w', type=int, default=6)
    ap.add_argument('--corner-refine-half-h', type=int, default=6)
    ap.add_argument('--corner-refine-search-h', type=int, default=8)
    ap.add_argument('--corner-refine-search-w', type=int, default=8)
    ap.add_argument('--corner-min-accept-score', type=float, default=10.0)
    ap.add_argument('--corner-head-area-thresh', type=float, default=2000.0,
                    help='Area threshold to treat a bbox as container head')
    ap.add_argument('--corner-head-offset', type=int, default=20,
                    help='Offset (px) applied when extending container head corners')
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)

    # expand indices
    idxs = []
    for s in args.indices:
        if '-' in s:
            a,b = s.split('-',1)
            if a.isdigit() and b.isdigit():
                idxs.extend(range(int(a), int(b)+1))
        elif s.isdigit():
            idxs.append(int(s))
    if not idxs:
        print("No valid indices."); return

    corner_params = None
    if not args.no_corners:
        corner_params = CornerRefinerParams(
            side_margin_frac=args.corner_side_margin_frac,
            center_window_frac=args.corner_center_window_frac,
            center_near_px=args.corner_center_near_px,
            center_thr_alpha=args.corner_center_thr_alpha,
            min_side_px=args.corner_min_side_px,
            sobel_ksize=args.corner_sobel_ksize,
            perc=args.corner_perc,
            perc_alpha=args.corner_perc_alpha,
            std_mult=args.corner_std_mult,
            refine_half_w=args.corner_refine_half_w,
            refine_half_h=args.corner_refine_half_h,
            refine_search_h=args.corner_refine_search_h,
            refine_search_w=args.corner_refine_search_w,
            min_accept_score=args.corner_min_accept_score,
            container_head_area_thresh=args.corner_head_area_thresh,
            container_head_offset=args.corner_head_offset,
        )

    for i in idxs:
        p = Path(args.dir) / f"{i}{args.suffix}"
        if not p.exists():
            print(f"Missing: {p}"); continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Read fail: {p}"); continue
        h, w = img.shape[:2]

        # 1) detect (get classes)
        boxes, scores, classes = run_detect(model, img, conf=args.conf, iou=args.iou)
        boxes, scores, classes = keep_best_per_quadrant(boxes, scores, classes, w, h, args.max_per_quad)
        labels = [c if c is not None else 'det' for c in classes]

        # 2) only handle 3-corner case (vector completion remains class-agnostic)
        if len(boxes) == 3:
            completion = VectorCornerCompleter.infer_missing_three(boxes, (w, h))
            if completion is not None:
                boxes.append(completion.box)
                scores.append(None)
                classes.append('gu_cor')
                labels.append('pred_3')
        else:
            print(f"[{i}] detections = {len(boxes)} (skip vector-complete; need exactly 3)")

        # 3) save annotated boxes (show class names)
        annotated = draw_boxes(img, boxes, classes, scores)
        out_img = out_dir / f"{i}_annot{args.suffix}"
        cv2.imwrite(str(out_img), annotated)
        safe_suffix = args.suffix.replace('.', '_')
        out_txt = out_dir / f"{i}_annot{safe_suffix}.txt"
        boxes_int = [[int(round(v)) for v in b] for b in boxes]
        with open(out_txt, "w") as f:
            for b_int, lab, sc in zip(boxes_int, labels, scores):
                x1, y1, x2, y2 = b_int
                if sc is None:
                    f.write(f"{lab} {x1} {y1} {x2} {y2}\n")
                else:
                    f.write(f"{lab} {x1} {y1} {x2} {y2} {sc:.4f}\n")

        print(f"Wrote: {out_img}")

        if args.no_corners:
            continue

        # 4) refine corners for all boxes. For side-face images we pass a per-box
        # `force_head` list derived from detector class names so head-style
        # corners are returned when the detector already indicates a head.
        force_head = [(c == 'gu_cor') for c in classes]
        corner_candidates = refine_corners(img, boxes_int, corner_params, force_head=force_head)

        corner_txt = out_dir / f"{i}_corners.txt"
        with open(corner_txt, "w") as f:
            for cand, b_int in zip(corner_candidates, boxes_int):
                x1, y1, x2, y2 = b_int
                cx, cy = cand.point
                f.write(f"{cand.quadrant} {cx} {cy} {cand.confidence:.4f} {cand.method} {x1} {y1} {x2} {y2}\n")

        corners_vis = img.copy()
        for cand in corner_candidates:
            cx, cy = cand.point
            cv2.circle(corners_vis, (cx, cy), 6, (0, 0, 255), -1)
            txt = f"{cand.quadrant} {cand.confidence:.2f} {cand.method}"
            cv2.putText(corners_vis, txt, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)

        corner_img = out_dir / f"{i}_corners.jpg"
        cv2.imwrite(str(corner_img), corners_vis)
        print(f"Wrote corners: {corner_txt} and annotated image: {corner_img}")
    print("Done.")


if __name__ == "__main__":
    main()
