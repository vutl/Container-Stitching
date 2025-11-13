#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sequence splitter for side-face corners.

Behavior:
- Iterate frames in order. For folder 1, keep writing until the first frame
  that triggers a seam condition:
    * primary: total detections > 4 (5-6 box: 2 edge_cor + 2 gu_cor of C1,
      plus 1-2 gu_cor of C2), OR
    * guard: two gu_cor on the same row too close horizontally (unexpected),
      OR (optional) seam quartet found.
- On that boundary frame, save only C1 boxes (2 edge + 2 gu if possible) to
  folder 1.
- Start folder 2 from the next frame. For early frames that still show both
  containers (e.g. 4 gu present), keep only C2 boxes (2 gu + 2 edge) to
  folder 2. Otherwise keep normal behavior.
- For each saved image, run gradient-based corner refinement on the kept boxes.

Outputs per folder:
- {idx}_annot{suffix}: annotated boxes with class labels
- {idx}_annot{suffix}.txt: x1 y1 x2 y2 (+score) per box
- {idx}_corners.jpg / {idx}_corners.txt: refined corners visualization and data

This is a validation tool per your spec; it does not alter other scripts.
"""

import argparse
from collections import deque
from pathlib import Path
from typing import List, Tuple, Optional
import re
import cv2
import numpy as np

from corner_completion import box_center, assign_quadrant
from corner_refinement_sides import CornerRefinerParams, refine_corners


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


# ---------- seam detection helpers ----------
def _find_seam_x_from_gu(boxes, classes, h, w, eps_x_frac=0.02, pair_dx_frac=0.05):
    gu = [(b, i) for i, (b, c) in enumerate(zip(boxes, classes)) if c == 'gu_cor']
    if len(gu) < 4:
        return None
    tops = []
    bots = []
    for b, _ in gu:
        cx, cy = box_center(b)
        if cy < h * 0.5: tops.append((cx, cy))
        else:            bots.append((cx, cy))
    if not tops or not bots:
        return None
    epsx = max(4.0, eps_x_frac * w)
    # pair top-bottom by closest x
    col_x = []
    for (tx, ty) in tops:
        best = None
        for (bx, by) in bots:
            dx = abs(tx - bx)
            if dx <= epsx and (best is None or dx < best[0]):
                best = (dx, 0.5*(tx+bx))
        if best is not None:
            col_x.append(best[1])
    if len(col_x) < 2:
        return None
    col_x.sort()
    # nearest two columns -> seam
    col_x_pairs = [(abs(col_x[i+1]-col_x[i]), 0.5*(col_x[i+1]+col_x[i])) for i in range(len(col_x)-1)]
    col_x_pairs.sort(key=lambda t: t[0])
    if not col_x_pairs:
        return None
    if col_x_pairs[0][0] > pair_dx_frac * w:
        return None
    return col_x_pairs[0][1]


def _find_seam_x_by_gu_cluster(boxes, classes, w):
    """Fallback: split gu_cor center-x by largest gap into two clusters; seam at midpoint.
    Returns None if <2 gu detected or if clusters collapse.
    """
    xs = [box_center(b)[0] for b, c in zip(boxes, classes) if c == 'gu_cor']
    if len(xs) < 2:
        return None
    xs.sort()
    # find largest gap
    gaps = [(xs[i+1]-xs[i], i) for i in range(len(xs)-1)]
    gaps.sort(key=lambda t: t[0], reverse=True)
    if not gaps or gaps[0][0] <= 0:
        return None
    i = gaps[0][1]
    left = xs[:i+1]; right = xs[i+1:]
    if not left or not right:
        return None
    return 0.5 * (left[-1] + right[0])


def _gu_same_row_too_close(boxes, classes, w, h, min_dx_frac=0.12):
    gu = [box_center(b) for b, c in zip(boxes, classes) if c == 'gu_cor']
    if len(gu) < 2:
        return False
    min_dx = float('inf')
    for i in range(len(gu)):
        for j in range(i+1, len(gu)):
            (x1,y1),(x2,y2) = gu[i], gu[j]
            same_row = (y1 < h*0.5 and y2 < h*0.5) or (y1 >= h*0.5 and y2 >= h*0.5)
            if same_row:
                min_dx = min(min_dx, abs(x1-x2))
    return min_dx < (min_dx_frac * w)


def _validate_bottom_alignment(corners_dict, max_align_diff_px=40):
    """Check if BL and BR corners are horizontally aligned (same row).
    
    Returns:
        - 'valid': both corners aligned within threshold
        - 'remove_BL': BL corner should be removed (y too far from BR)
        - 'remove_BR': BR corner should be removed (y too far from BL)
    """
    if 'BL' not in corners_dict or 'BR' not in corners_dict:
        return 'valid'  # can't validate if corners missing
    
    bl_x, bl_y = corners_dict['BL']
    br_x, br_y = corners_dict['BR']
    
    diff_y = abs(bl_y - br_y)
    if diff_y <= max_align_diff_px:
        return 'valid'
    
    # Determine which corner is more deviated
    # Use TL/TR as reference if available to decide which bottom corner is wrong
    if 'TL' in corners_dict and 'TR' in corners_dict:
        tl_y = corners_dict['TL'][1]
        tr_y = corners_dict['TR'][1]
        h_left = abs(bl_y - tl_y)
        h_right = abs(br_y - tr_y)
        
        # The column with more extreme height deviation is likely wrong
        # But we also check: if one bottom corner is much lower than the other, remove the lower one
        if bl_y > br_y:
            # BL is lower than BR
            return 'remove_BL'
        else:
            # BR is lower than BL
            return 'remove_BR'
    else:
        # Fallback: remove the corner that's further down (higher y value)
        if bl_y > br_y:
            return 'remove_BL'
        else:
            return 'remove_BR'


def _validate_vertical_alignment(corners_dict, max_align_diff_px=80):
    """Check if corners in the same column are vertically aligned (same x).
    
    Validates:
    - Left column: TL.x vs BL.x
    - Right column: TR.x vs BR.x
    
    Returns list of corners to remove (e.g., ['TL', 'BR'])
    """
    to_remove = []
    
    # Check left column (TL vs BL)
    if 'TL' in corners_dict and 'BL' in corners_dict:
        tl_x, tl_y = corners_dict['TL']
        bl_x, bl_y = corners_dict['BL']
        diff_x = abs(tl_x - bl_x)
        if diff_x > max_align_diff_px:
            # Remove the corner with lower confidence or the one further from expected position
            # For now, remove the one with larger x deviation from the smaller x value
            # (assuming left edge should be closer to x=0)
            if tl_x > bl_x:
                to_remove.append('TL')
            else:
                to_remove.append('BL')
    
    # Check right column (TR vs BR)
    if 'TR' in corners_dict and 'BR' in corners_dict:
        tr_x, tr_y = corners_dict['TR']
        br_x, br_y = corners_dict['BR']
        diff_x = abs(tr_x - br_x)
        if diff_x > max_align_diff_px:
            # Remove the corner further from expected position
            # (assuming right edge should be closer to image width)
            if tr_x < br_x:
                to_remove.append('TR')
            else:
                to_remove.append('BR')
    
    return to_remove


def _select_by_side(boxes, scores, classes, w, h, side: str, x_split: Optional[float]=None):
    """Return boxes on left (side='L') or right (side='R') of seam_x.
    side is handled by caller; this helper only filters by x relative to x_mid.
    """
    if side not in ('L','R'):
        side = 'L'
    x_mid = x_split if x_split is not None else (0.5 * w)
    out_b, out_s, out_c = [], [], []
    for b, s, c in zip(boxes, scores, classes):
        cx, cy = box_center(b)
        if (side == 'L' and cx < x_mid) or (side == 'R' and cx >= x_mid):
            out_b.append(b)
            out_s.append(s)
            out_c.append(c)
    return out_b, out_s, out_c


def _keep_two_gu_two_edge(boxes, scores, classes, w, h, side: str, x_split: Optional[float]=None):
    # First filter to side
    b_side, s_side, c_side = _select_by_side(boxes, scores, classes, w, h, side, x_split=x_split)
    # Split gu vs edge and top vs bottom
    top_gu = []; bot_gu = []; top_ed = []; bot_ed = []
    for b, s, c in zip(b_side, s_side, c_side):
        cx, cy = box_center(b)
        is_top = cy < h*0.5
        if c == 'gu_cor':
            # ignore very weak gu detections so edge_cor is preferred
            if s is None or s < globals().get('MIN_GU_CONF', 0.0):
                # treat as non-gu (skip adding to gu lists)
                continue
            (top_gu if is_top else bot_gu).append((b,s,c))
        elif c == 'edge_cor':
            (top_ed if is_top else bot_ed).append((b,s,c))
    # sort by score desc
    for arr in (top_gu, bot_gu, top_ed, bot_ed):
        arr.sort(key=lambda t: t[1] if t[1] is not None else -1, reverse=True)
    chosen = []
    # choose one per row per type if available
    if top_gu: chosen.append(top_gu[0])
    if bot_gu: chosen.append(bot_gu[0])
    if top_ed: chosen.append(top_ed[0])
    if bot_ed: chosen.append(bot_ed[0])
    # if not enough, fill from remaining side boxes, preferring missing types
    if len(chosen) < 4:
        remaining = [(b,s,c) for (b,s,c) in zip(b_side, s_side, c_side) if (b,s,c) not in chosen]
        remaining.sort(key=lambda t: t[1] if t[1] is not None else -1, reverse=True)
        for item in remaining:
            chosen.append(item)
            if len(chosen) >= 4:
                break
    out_b = [b for (b,_,_) in chosen[:4]]
    out_s = [s for (_,s,_) in chosen[:4]]
    out_c = [c for (_,_,c) in chosen[:4]]
    return out_b, out_s, out_c


def _keep_two_gu_two_edge_leftmost(boxes, scores, classes, w, h):
    # Sort by center x ascending (leftmost first)
    trip = list(zip(boxes, scores, classes))
    trip.sort(key=lambda t: box_center(t[0])[0])
    top_gu = []; bot_gu = []; top_ed = []; bot_ed = []
    for b, s, c in trip:
        cx, cy = box_center(b)
        is_top = cy < h*0.5
        if c == 'gu_cor':
            if s is None or s < globals().get('MIN_GU_CONF', 0.0):
                continue
            (top_gu if is_top else bot_gu).append((b,s,c))
        elif c == 'edge_cor':
            (top_ed if is_top else bot_ed).append((b,s,c))
    chosen = []
    if top_gu: chosen.append(top_gu[0])
    if bot_gu: chosen.append(bot_gu[0])
    if top_ed: chosen.append(top_ed[0])
    if bot_ed: chosen.append(bot_ed[0])
    if len(chosen) < 4:
        remaining = [(b,s,c) for (b,s,c) in trip if (b,s,c) not in chosen]
        for item in remaining:
            chosen.append(item)
            if len(chosen) >= 4:
                break
    out_b = [b for (b,_,_) in chosen[:4]]
    out_s = [s for (_,s,_) in chosen[:4]]
    out_c = [c for (_,_,c) in chosen[:4]]
    return out_b, out_s, out_c


def _keep_two_gu_two_edge_rightmost(boxes, scores, classes, w, h):
    # Sort by center x descending (rightmost first)
    trip = list(zip(boxes, scores, classes))
    trip.sort(key=lambda t: box_center(t[0])[0], reverse=True)
    top_gu = []; bot_gu = []; top_ed = []; bot_ed = []
    for b, s, c in trip:
        cx, cy = box_center(b)
        is_top = cy < h*0.5
        if c == 'gu_cor':
            if s is None or s < globals().get('MIN_GU_CONF', 0.0):
                continue
            (top_gu if is_top else bot_gu).append((b,s,c))
        elif c == 'edge_cor':
            (top_ed if is_top else bot_ed).append((b,s,c))
    chosen = []
    if top_gu: chosen.append(top_gu[0])
    if bot_gu: chosen.append(bot_gu[0])
    if top_ed: chosen.append(top_ed[0])
    if bot_ed: chosen.append(bot_ed[0])
    if len(chosen) < 4:
        remaining = [(b,s,c) for (b,s,c) in trip if (b,s,c) not in chosen]
        for item in remaining:
            chosen.append(item)
            if len(chosen) >= 4:
                break
    out_b = [b for (b,_,_) in chosen[:4]]
    out_s = [s for (_,s,_) in chosen[:4]]
    out_c = [c for (_,_,c) in chosen[:4]]
    return out_b, out_s, out_c


def _keep_two_gu_two_edge_global(boxes, scores, classes, w, h):
    """Pick up to 4 with priority: top gu, bottom gu, top edge, bottom edge.
    If still fewer than 4, fill by remaining highest scores.
    This is used pre-boundary to ensure gu (if detected) are not dropped by quadrant tie-breaks.
    """
    top_gu = []; bot_gu = []; top_ed = []; bot_ed = []; rest = []
    for b, s, c in zip(boxes, scores, classes):
        cx, cy = box_center(b)
        is_top = cy < h * 0.5
        if c == 'gu_cor':
            # treat very weak gu as low-priority: move into rest so edges are preferred
            if s is None or s < globals().get('MIN_GU_CONF', 0.0):
                rest.append((b, s, c))
                continue
            (top_gu if is_top else bot_gu).append((b, s, c))
        elif c == 'edge_cor':
            (top_ed if is_top else bot_ed).append((b, s, c))
        else:
            rest.append((b, s, c))

    for arr in (top_gu, bot_gu, top_ed, bot_ed, rest):
        arr.sort(key=lambda t: t[1] if t[1] is not None else -1, reverse=True)

    chosen = []
    if top_gu: chosen.append(top_gu[0])
    if bot_gu: chosen.append(bot_gu[0])
    if top_ed: chosen.append(top_ed[0])
    if bot_ed: chosen.append(bot_ed[0])

    if len(chosen) < 4:
        # Combine remaining pools preserving order of preference
        remaining = []
        remaining.extend(top_gu[1:])
        remaining.extend(bot_gu[1:])
        remaining.extend(top_ed[1:])
        remaining.extend(bot_ed[1:])
        remaining.extend(rest)
        remaining.sort(key=lambda t: t[1] if t[1] is not None else -1, reverse=True)
        for item in remaining:
            if item not in chosen:
                chosen.append(item)
                if len(chosen) >= 4:
                    break

    out_b = [b for (b,_,_) in chosen[:4]]
    out_s = [s for (_,s,_) in chosen[:4]]
    out_c = [c for (_,_,c) in chosen[:4]]
    return out_b, out_s, out_c


def _suppress_cross_class_overlaps(boxes, scores, classes, iou_thresh=0.45):
    """Post-process to remove duplicate detections with high overlap.
    
    Handles two types of overlaps:
    1. Cross-class overlaps (gu_cor vs edge_cor): prefer edge_cor
    2. Same-class overlaps (edge_cor vs edge_cor, gu_cor vs gu_cor): prefer higher confidence
    
    This prevents false seam splits caused by duplicate detections at the same corner.
    Returns new (boxes, scores, classes) lists.
    """
    if not boxes:
        return boxes, scores, classes
    # compute IoU matrix
    def iou(a,b):
        x1,y1,x2,y2 = a
        x1b,y1b,x2b,y2b = b
        xa1 = max(x1,x1b); ya1 = max(y1,y1b)
        xa2 = min(x2,x2b); ya2 = min(y2,y2b)
        if xa2<=xa1 or ya2<=ya1:
            return 0.0
        inter = (xa2-xa1)*(ya2-ya1)
        area_a = max(0,(x2-x1)) * max(0,(y2-y1))
        area_b = max(0,(x2b-x1b)) * max(0,(y2b-y1b))
        union = area_a + area_b - inter
        return inter/union if union>0 else 0.0

    keep = [True] * len(boxes)
    # iterate over all pairs
    for i, (bi, si, ci) in enumerate(zip(boxes, scores, classes)):
        for j, (bj, sj, cj) in enumerate(zip(boxes, scores, classes)):
            if i == j or not keep[i] or not keep[j]:
                continue
            _iou = iou(bi, bj)
            if _iou > iou_thresh:
                # Handle cross-class overlaps (gu_cor vs edge_cor)
                if {ci, cj} == {'gu_cor', 'edge_cor'}:
                    # prefer edge_cor
                    if ci == 'edge_cor' and cj == 'gu_cor':
                        keep[j] = False
                    elif cj == 'edge_cor' and ci == 'gu_cor':
                        keep[i] = False
                # Handle same-class overlaps: prefer higher confidence
                elif ci == cj:
                    if si >= sj:
                        keep[j] = False
                    else:
                        keep[i] = False
    new_boxes = [b for k,b in zip(keep, boxes) if k]
    new_scores = [s for k,s in zip(keep, scores) if k]
    new_classes = [c for k,c in zip(keep, classes) if k]
    return new_boxes, new_scores, new_classes


def _resolve_selected_overlaps(boxes, scores, classes, iou_thresh=0.45):
    """Ensure the small set of selected boxes (e.g. up to 4) have no
    large overlaps. If two selected boxes overlap above iou_thresh, drop the
    lower-priority one (prefer 'edge_cor' over 'gu_cor', otherwise higher score).
    Returns filtered (boxes, scores, classes) preserving order as much as
    possible but removing conflicting boxes.
    """
    if not boxes:
        return boxes, scores, classes
    n = len(boxes)
    keep = [True] * n
    def iou(a,b):
        x1,y1,x2,y2 = a
        x1b,y1b,x2b,y2b = b
        xa1 = max(x1,x1b); ya1 = max(y1,y1b)
        xa2 = min(x2,x2b); ya2 = min(y2,y2b)
        if xa2<=xa1 or ya2<=ya1:
            return 0.0
        inter = (xa2-xa1)*(ya2-ya1)
        area_a = max(0,(x2-x1)) * max(0,(y2-y1))
        area_b = max(0,(x2b-x1b)) * max(0,(y2b-y1b))
        union = area_a + area_b - inter
        return inter/union if union>0 else 0.0

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i+1, n):
            if not keep[j]:
                continue
            _iou = iou(boxes[i], boxes[j])
            if _iou > iou_thresh:
                ci = classes[i]; cj = classes[j]
                si = scores[i] if scores[i] is not None else -1
                sj = scores[j] if scores[j] is not None else -1
                # prefer edge_cor
                if ci == 'edge_cor' and cj == 'gu_cor':
                    keep[j] = False
                elif cj == 'edge_cor' and ci == 'gu_cor':
                    keep[i] = False
                else:
                    # keep higher score
                    if si >= sj:
                        keep[j] = False
                    else:
                        keep[i] = False

    new_boxes = [b for k,b in zip(keep, boxes) if k]
    new_scores = [s for k,s in zip(keep, scores) if k]
    new_classes = [c for k,c in zip(keep, classes) if k]
    return new_boxes, new_scores, new_classes


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='last11scor_3class_31_07.pt')
    ap.add_argument('--dir', type=str, required=True)
    ap.add_argument('--indices', type=str, nargs='+', help='numbers or ranges like 54 60-65; if omitted, process all files matching suffix')
    ap.add_argument('--suffix', type=str, default='_cropped_padded.jpg')
    ap.add_argument('--out1', type=str, required=True, help='Output folder for container 1 segment')
    ap.add_argument('--out2', type=str, required=True, help='Output folder for container 2 segment')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--max-per-quad', type=int, default=1)
    ap.add_argument('--seam-consec', type=int, default=1, help='Number of consecutive frames that must trigger seam before committing split')

    # corner refine params
    ap.add_argument('--no-corners', action='store_true', help='Skip corner refinement outputs')
    ap.add_argument('--corner-side-margin-frac', type=float, default=0.12)
    ap.add_argument('--corner-center-window-frac', type=float, default=0.45)
    ap.add_argument('--corner-center-near-px', type=int, default=4)
    ap.add_argument('--corner-center-thr-alpha', type=float, default=0.5)
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
    ap.add_argument('--corner-head-area-thresh', type=float, default=2000.0)
    ap.add_argument('--corner-head-offset', type=int, default=20)
    ap.add_argument('--min-gu-conf', type=float, default=0.20, help='Minimum confidence to treat a gu_cor as strong (below this prefer edge_cor)')
    args = ap.parse_args()

    # global threshold used by helpers to treat weak gu_cor as low-priority
    global MIN_GU_CONF
    MIN_GU_CONF = float(args.min_gu_conf)

    out1 = Path(args.out1); out1.mkdir(parents=True, exist_ok=True)
    out2 = Path(args.out2); out2.mkdir(parents=True, exist_ok=True)
    model = load_model(args.model)

    # expand indices or glob all files
    idxs: List[int] = []
    files: List[Path] = []
    if args.indices:
        for s in args.indices:
            if '-' in s:
                a,b = s.split('-',1)
                if a.isdigit() and b.isdigit():
                    idxs.extend(range(int(a), int(b)+1))
            elif s.isdigit():
                idxs.append(int(s))
        files = [Path(args.dir) / f"{i}{args.suffix}" for i in idxs]
    else:
        # glob all matching files but skip stitched composites (stitch_img_* or stitch_*)
        all_files = list(Path(args.dir).glob(f"*{args.suffix}"))
        files = []
        for p in all_files:
            name = p.name
            # skip typical stitched/composite filenames
            if name.startswith('stitch_img_') or name.startswith('stitch_'):
                continue
            files.append(p)
        # try to derive numeric order when possible
        def _key(p: Path):
            # Try to extract trailing integer index from filename stem (e.g. 'img_12' -> 12)
            stem = p.name[:-len(args.suffix)] if p.name.endswith(args.suffix) else p.stem
            m = re.search(r"(\d+)$", stem)
            if m:
                try:
                    return (0, int(m.group(1)))
                except Exception:
                    pass
            # Fallback: non-numeric stems sort after numeric ones and use lexical order
            return (1, stem)
        files.sort(key=_key)
        # Skip stitched composite images (they often are named stitch_img_*) so
        # the detector doesn't process previously-created panoramas.
        files = [p for p in files if not p.name.startswith('stitch_img_')]

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

    seam_found: bool = False
    # deque to track recent seam triggers; size determined by --seam-consec
    seam_window = deque(maxlen=max(1, args.seam_consec))

    total_files = len(files)

    for pos, p in enumerate(files):
        # derive printable index label
        stem = p.name[:-len(args.suffix)] if p.name.endswith(args.suffix) else p.stem
        idx_label = stem
        if not p.exists():
            print(f"Missing: {p}"); continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Read fail: {p}"); continue
        h, w = img.shape[:2]

        boxes, scores, classes = run_detect(model, img, conf=args.conf, iou=args.iou)
        # Post-process cross-class overlaps (gu_cor vs edge_cor)
        boxes, scores, classes = _suppress_cross_class_overlaps(boxes, scores, classes, iou_thresh=0.45)

        # Decide which folder
        if not seam_found:
            # test seam condition
            # NOTE: Ignore seam triggers for the first 10 and last 10 frames of the
            # sequence as requested (these boundary frames should not cause a
            # split even if they contain >4 boxes or many gu_cor detections).
            if pos < 10 or pos >= (total_files - 10):
                trigger = False
            else:
                # Primary seam condition: presence of multiple gu_cor (container heads)
                # indicating 2 containers in view. We require >= 4 gu_cor (2 per container)
                # OR the secondary condition of gu_cor pairs too close horizontally.
                # Total box count alone (>4) is NOT sufficient since noise detections
                # (low-conf edge_cor) can cause false splits in single-container sequences.
                num_gu = sum(1 for c in classes if c == 'gu_cor')
                trigger = (num_gu >= 4) or _gu_same_row_too_close(boxes, classes, w, h)

            # push into rolling window and decide only when enough consecutive
            # triggers have been observed (helps ignore transient false positives)
            seam_window.append(bool(trigger))
            if sum(1 for v in seam_window if v) >= args.seam_consec:
                # commit seam
                seam_found = True
                print(f"[{idx_label}] seam trigger committed (window={list(seam_window)})")
                # estimate seam x from gu_cor pairs to split accurately
                seam_x = _find_seam_x_from_gu(boxes, classes, h, w)
                if seam_x is None:
                    seam_x = _find_seam_x_by_gu_cluster(boxes, classes, w)
                # Save only container-1 (left) 2 gu + 2 edge if possible
                sel_b, sel_s, sel_c = _keep_two_gu_two_edge(boxes, scores, classes, w, h, side='L', x_split=seam_x)
                if sum(1 for c in sel_c if c == 'gu_cor') == 0:
                    # fallback: pick leftmost-priority to try keep gu if any exist on the left half
                    sel_b, sel_s, sel_c = _keep_two_gu_two_edge_leftmost(boxes, scores, classes, w, h)
                    if sum(1 for c in sel_c if c == 'gu_cor') == 0:
                        # keep C1-only selection even if it has no gu; do not pull from right side
                        msg = "no gu on left-side selection"
                        if seam_x is None:
                            msg += ", seam_x unknown"
                        print(f"[{idx_label}] {msg}; keeping left-side boxes only for C1 boundary")
                out_dir = out1
            else:
                # not yet committed; normal behavior for C1 segment
                sel_b, sel_s, sel_c = _keep_two_gu_two_edge_global(boxes, scores, classes, w, h)
                out_dir = out1
            
        else:
            # We are in C2 segment
            # If early frames still contain both containers (e.g., multiple gu), enforce right-side 2 gu + 2 edge
            num_gu = sum(1 for c in classes if c == 'gu_cor')
            if num_gu >= 4 or len(boxes) > 4:
                seam_x = _find_seam_x_from_gu(boxes, classes, h, w)
                if seam_x is None:
                    seam_x = _find_seam_x_by_gu_cluster(boxes, classes, w)
                sel_b, sel_s, sel_c = _keep_two_gu_two_edge(boxes, scores, classes, w, h, side='R', x_split=seam_x)
                if sum(1 for c in sel_c if c == 'gu_cor') == 0:
                    # fallback: rightmost-priority to try keep gu on the right
                    sel_b, sel_s, sel_c = _keep_two_gu_two_edge_rightmost(boxes, scores, classes, w, h)
                    if sum(1 for c in sel_c if c == 'gu_cor') == 0:
                        # keep C2-only selection even if it has no gu; do not pull from left side
                        msg = "no gu on right-side selection"
                        if seam_x is None:
                            msg += ", seam_x unknown"
                        print(f"[{idx_label}] {msg}; keeping right-side boxes only for early C2")
            else:
                # Prefer gu-first selection when available
                sel_b, sel_s, sel_c = _keep_two_gu_two_edge_global(boxes, scores, classes, w, h)
            out_dir = out2

        # Resolve overlaps among selected boxes (ensure final selection has no large overlaps)
        sel_b, sel_s, sel_c = _resolve_selected_overlaps(sel_b, sel_s, sel_c, iou_thresh=0.45)

        # Save annotations
        labels = [c if c is not None else 'det' for c in sel_c]
        annotated = draw_boxes(img, sel_b, sel_c, sel_s)
        out_img = out_dir / f"{idx_label}_annot{args.suffix}"
        cv2.imwrite(str(out_img), annotated)
        safe_suffix = args.suffix.replace('.', '_')
        out_txt = out_dir / f"{idx_label}_annot{safe_suffix}.txt"
        boxes_int = [[int(round(v)) for v in b] for b in sel_b]
        with open(out_txt, 'w') as f:
            for b_int, lab, sc in zip(boxes_int, labels, sel_s):
                x1,y1,x2,y2 = b_int
                if sc is None:
                    f.write(f"{lab} {x1} {y1} {x2} {y2}\n")
                else:
                    f.write(f"{lab} {x1} {y1} {x2} {y2} {sc:.4f}\n")

        print(f"[{idx_label}] wrote: {out_img}")

        if args.no_corners:
            continue

        # refine corners
        force_head = [(c == 'gu_cor') for c in sel_c]
        corner_candidates = refine_corners(img, boxes_int, corner_params, force_head=force_head)
        
        # Create mapping: quadrant -> (candidate, bbox)
        quad_to_data = {}
        for cand, b_int in zip(corner_candidates, boxes_int):
            quad_to_data[cand.quadrant] = (cand, b_int)
        
        # Validate bottom alignment and remove misaligned corner if needed
        corners_dict = {cand.quadrant: cand.point for cand in corner_candidates}
        validation_result = _validate_bottom_alignment(corners_dict, max_align_diff_px=40)
        
        # Validate bottom alignment (horizontal: BL.y vs BR.y)
        if validation_result != 'valid':
            # Remove the offending corner
            if validation_result == 'remove_BL' and 'BL' in quad_to_data:
                del quad_to_data['BL']
                print(f"[{idx_label}] Removed BL corner due to horizontal misalignment (|BL.y - BR.y| > 40px)")
            elif validation_result == 'remove_BR' and 'BR' in quad_to_data:
                del quad_to_data['BR']
                print(f"[{idx_label}] Removed BR corner due to horizontal misalignment (|BL.y - BR.y| > 40px)")
        
        # Validate vertical alignment (same column: TL.x vs BL.x, TR.x vs BR.x)
        corners_dict_updated = {q: quad_to_data[q][0].point for q in quad_to_data}
        corners_to_remove = _validate_vertical_alignment(corners_dict_updated, max_align_diff_px=80)
        for q in corners_to_remove:
            if q in quad_to_data:
                tl_x = quad_to_data.get('TL', [None, None])[0].point[0] if 'TL' in quad_to_data else None
                bl_x = quad_to_data.get('BL', [None, None])[0].point[0] if 'BL' in quad_to_data else None
                tr_x = quad_to_data.get('TR', [None, None])[0].point[0] if 'TR' in quad_to_data else None
                br_x = quad_to_data.get('BR', [None, None])[0].point[0] if 'BR' in quad_to_data else None
                
                if q in ('TL', 'BL') and tl_x is not None and bl_x is not None:
                    diff_x = abs(tl_x - bl_x)
                    del quad_to_data[q]
                    print(f"[{idx_label}] Removed {q} corner due to vertical misalignment (|TL.x - BL.x| = {diff_x:.0f}px > 80px)")
                elif q in ('TR', 'BR') and tr_x is not None and br_x is not None:
                    diff_x = abs(tr_x - br_x)
                    del quad_to_data[q]
                    print(f"[{idx_label}] Removed {q} corner due to vertical misalignment (|TR.x - BR.x| = {diff_x:.0f}px > 80px)")
        
        # Write corners (may be 2, 3 or 4 corners now)
        corner_txt = out_dir / f"{idx_label}_corners.txt"
        with open(corner_txt, 'w') as f:
            for quadrant in ['TL', 'TR', 'BR', 'BL']:  # consistent order
                if quadrant in quad_to_data:
                    cand, b_int = quad_to_data[quadrant]
                    x1, y1, x2, y2 = b_int
                    cx, cy = cand.point
                    f.write(f"{cand.quadrant} {cx} {cy} {cand.confidence:.4f} {cand.method} {x1} {y1} {x2} {y2}\n")

        corners_vis = img.copy()
        for quadrant, (cand, _) in quad_to_data.items():
            cx, cy = cand.point
            cv2.circle(corners_vis, (cx, cy), 6, (0,0,255), -1)
            txt = f"{cand.quadrant} {cand.confidence:.2f} {cand.method}"
            cv2.putText(corners_vis, txt, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
        corner_img = out_dir / f"{idx_label}_corners.jpg"
        cv2.imwrite(str(corner_img), corners_vis)
        print(f"[{idx_label}] corners: {corner_txt} and {corner_img}")

    print('Done.')


if __name__ == '__main__':
    main()
