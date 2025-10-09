#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Container corner completion - ONLY the 3-corner case.
- Detect corners with YOLO.
- Keep top-1 per quadrant (TL/TR/BR/BL).
- If and only if we have exactly 3 corners, infer the 4th corner purely by
  vector arithmetic on the box centers (no mirroring, no container bounds),
  then create a box for it using neighbor sizes.
- If not 3 detections, we just save the detections and warn.

Quadrants (image coordinates): A=TL, B=TR, C=BR, D=BL (clockwise).
Vector completion:
  missing A: A = B + D - C
  missing B: B = A + C - D
  missing C: C = B + D - A
  missing D: D = A + C - B
"""

import argparse
from pathlib import Path
import cv2
import numpy as np

from corner_completion import box_center, assign_quadrant, VectorCornerCompleter
from corner_refinement import CornerRefinerParams, refine_corners

# ---------- drawing ----------
def draw_boxes(img, boxes, labels, scores=None):
    out = img.copy()
    color = {
        "det":    (0,255,0),      # green
        "pred_3": (255,128,0),    # orange
    }
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        lab = labels[i]
        col = color.get(lab, (0,255,0))
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        txt = lab if scores is None or scores[i] is None else f"{lab} {scores[i]:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), col, -1)
        cv2.putText(out, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out

# ---------- YOLO ----------
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
    boxes, scores = [], []
    try:
        b = r.boxes.xyxy.cpu().numpy()
        s = r.boxes.conf.cpu().numpy()
        for bb, sc in zip(b, s):
            boxes.append(bb.tolist())
            scores.append(float(sc))
    except Exception:
        pass
    return boxes, scores

def keep_best_per_quadrant(boxes, scores, w, h, max_per_quad=1):
    quads = {"TL":[], "TR":[], "BR":[], "BL":[]}
    for b, s in zip(boxes, scores):
        cx, cy = box_center(b)
        q = assign_quadrant(cx, cy, w, h)
        quads[q].append((b, s))
    kept_boxes, kept_scores = [], []
    for q in quads:
        quads[q].sort(key=lambda x: x[1], reverse=True)
        for b,s in quads[q][:max_per_quad]:
            kept_boxes.append(b); kept_scores.append(s)
    return kept_boxes, kept_scores


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='last11s_cortop_10_08.pt')
    ap.add_argument('--dir', type=str, required=True)
    ap.add_argument('--indices', type=str, nargs='+', required=True,
                    help="numbers or ranges like 54 60-65")
    ap.add_argument('--suffix', type=str, default='_cropped_padded.jpg')
    ap.add_argument('--out', type=str, required=True, default='out_annot')
    ap.add_argument('--conf', type=float, default=0.05)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--max-per-quad', type=int, default=1)
    ap.add_argument('--debug', action='store_true')
    ap.add_argument('--fix-mode', choices=['none','snap','temporal'], default='snap',
                    help='Automatic correction mode for mixed-container detections')

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
    ap.add_argument('--corner-head-offset', type=int, default=10,
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

        # 1) detect
        boxes_all, scores_all = run_detect(model, img, conf=args.conf, iou=args.iou)

        # keep a mapping of all candidates per quadrant so we can try
        # alternative picks if the top-per-quadrant combination looks skewed
        quads_all = {"TL": [], "TR": [], "BR": [], "BL": []}
        for b, s in zip(boxes_all, scores_all):
            cx, cy = box_center(b)
            q = assign_quadrant(cx, cy, w, h)
            quads_all[q].append((b, s))

        boxes, scores = keep_best_per_quadrant(boxes_all, scores_all, w, h, args.max_per_quad)
        labels = ["det"] * len(boxes)

        # track whether this frame was auto-corrected
        corrected = False

        # Post-filter: ensure left column corners (TL vs BL) and right column (TR vs BR)
        # come from the same container / are horizontally consistent. If not, try
        # alternatives from the original detections to reduce the mismatch.
        def _box_x_center(box):
            return 0.5 * (box[0] + box[2])

        # build quick lookup for kept boxes by quadrant
        kept_map = {}
        for b, s in zip(boxes, scores):
            cx, cy = box_center(b)
            kept_map[assign_quadrant(cx, cy, w, h)] = (b, s)

        # More aggressive fix: consider re-pairing top/bottom from ALL candidates
        # within the left half and right half respectively. This helps when the
        # top/bottom detections were assigned to different quadrants but belong
        # to the same container column (mixed-container case).
        def _repair_column(is_left: bool):
            half_x = 0.5 * float(w)
            # collect candidates in the requested half
            cand = []
            for b, s in zip(boxes_all, scores_all):
                cx = 0.5 * (b[0] + b[2])
                if (is_left and cx <= half_x) or (not is_left and cx > half_x):
                    cand.append((b, s))
            if len(cand) < 2:
                return False
            # choose top-most and bottom-most candidates by center y
            cand_sorted = sorted(cand, key=lambda bs: 0.5 * (bs[0][1] + bs[0][3]))
            top_cand, bot_cand = cand_sorted[0], cand_sorted[-1]
            top_b, top_s = top_cand; bot_b, bot_s = bot_cand
            top_x = 0.5 * (top_b[0] + top_b[2]); bot_x = 0.5 * (bot_b[0] + bot_b[2])

            # compute current chosen pair (if any) for this half
            cur_top = None; cur_bot = None
            for bb in boxes:
                cx = 0.5 * (bb[0] + bb[2]); cy = 0.5 * (bb[1] + bb[3])
                if (is_left and cx <= half_x) or (not is_left and cx > half_x):
                    # top if y is above median of this half
                    pass
            # find current top/bot by y among kept boxes in this half
            kept_in_half = []
            for bb, ss in zip(boxes, scores):
                cx = 0.5 * (bb[0] + bb[2]); cy = 0.5 * (bb[1] + bb[3])
                if (is_left and cx <= half_x) or (not is_left and cx > half_x):
                    kept_in_half.append((bb, ss))
            if len(kept_in_half) < 2:
                # nothing to compare/replace
                return False
            kept_sorted = sorted(kept_in_half, key=lambda bs: 0.5 * (bs[0][1] + bs[0][3]))
            cur_top_b, cur_top_s = kept_sorted[0]
            cur_bot_b, cur_bot_s = kept_sorted[-1]
            cur_dx = abs(0.5*(cur_top_b[0]+cur_top_b[2]) - 0.5*(cur_bot_b[0]+cur_bot_b[2]))
            cand_dx = abs(top_x - bot_x)
            thr = max(8.0, 0.06 * float(w))
            # accept new pairing if it improves dx by 10% or goes below threshold
            if cand_dx < 0.9 * cur_dx or cand_dx <= thr:
                # replace kept boxes for top and bottom in this half
                # find indices
                def _replace_kept(old_bb, new_bb, new_s):
                    for idx_bb, bb in enumerate(boxes):
                        if bb == old_bb:
                            boxes[idx_bb] = new_bb
                            scores[idx_bb] = new_s
                            return True
                    return False

                replaced_top = _replace_kept(cur_top_b, top_b, top_s)
                replaced_bot = _replace_kept(cur_bot_b, bot_b, bot_s)
                if replaced_top or replaced_bot:
                    side = 'left' if is_left else 'right'
                    print(f"[repair] re-paired {side} column at frame {i}: dx {cur_dx:.1f} -> {cand_dx:.1f}")
                    return True
            return False

        _repair_column(True)
        _repair_column(False)

        # Strong override: if both left and right halves each contain at least
        # two detection candidates, pick the top-most and bottom-most in each
        # half (by center-y) and use those as TL/BL (left) and TR/BR (right).
        left_half = []
        right_half = []
        half_x = 0.5 * float(w)
        for b, s in zip(boxes_all, scores_all):
            cx = 0.5 * (b[0] + b[2])
            cy = 0.5 * (b[1] + b[3])
            if cx <= half_x:
                left_half.append((b, s, cy))
            else:
                right_half.append((b, s, cy))

        if len(left_half) >= 2 and len(right_half) >= 2:
            left_sorted = sorted(left_half, key=lambda t: t[2])
            right_sorted = sorted(right_half, key=lambda t: t[2])
            lt_b, lt_s, _ = left_sorted[0]
            lb_b, lb_s, _ = left_sorted[-1]
            rt_b, rt_s, _ = right_sorted[0]
            rb_b, rb_s, _ = right_sorted[-1]
            # assign boxes in order TL, TR, BR, BL
            boxes = [lt_b, rt_b, rb_b, lb_b]
            scores = [lt_s, rt_s, rb_s, lb_s]
            labels = ["det"] * len(boxes)
            print(f"[override] used per-half top/bottom pairing for frame {i}")

        # If top/bottom in a column are far apart horizontally relative to their
        # own box widths, treat this as a mixed-container detection and snap
        # the lower-confidence box horizontally to the higher-confidence one.
        def _snap_column(is_left: bool):
            half_x = 0.5 * float(w)
            # collect kept boxes in this half
            kept_in_half = []
            for idx_bb, (bb, ss, lab) in enumerate(zip(boxes, scores, labels)):
                # do not touch boxes that are synthetic (pred_3)
                if lab != 'det':
                    continue
                cx = 0.5 * (bb[0] + bb[2]); cy = 0.5 * (bb[1] + bb[3])
                if (is_left and cx <= half_x) or (not is_left and cx > half_x):
                    kept_in_half.append((idx_bb, bb, ss, cx, cy))
            if len(kept_in_half) < 2:
                return False
            # decide top vs bottom by center y
            kept_sorted = sorted(kept_in_half, key=lambda t: t[4])
            top_idx, top_b, top_s, top_cx, top_cy = kept_sorted[0]
            bot_idx, bot_b, bot_s, bot_cx, bot_cy = kept_sorted[-1]
            dx = abs(top_cx - bot_cx)
            top_w = abs(top_b[2] - top_b[0])
            bot_w = abs(bot_b[2] - bot_b[0])
            max_box_w = max(1.0, top_w, bot_w)
            thr = max(12.0, 0.25 * max_box_w)
            if dx <= thr:
                return False
            # choose which to move: move the one with lower confidence (None treated as low)
            def _conf_val(s):
                return -1.0 if s is None else float(s)
            if _conf_val(top_s) >= _conf_val(bot_s):
                # move bottom to top x
                target_cx = top_cx
                src_idx = bot_idx
                src_box = bot_b
            else:
                target_cx = bot_cx
                src_idx = top_idx
                src_box = top_b

            bw = src_box[2] - src_box[0]
            half = bw / 2.0
            new_x1 = target_cx - half
            new_x2 = target_cx + half
            # clamp
            new_x1 = max(0.0, new_x1); new_x2 = min(float(w - 1), new_x2)
            new_box = [new_x1, src_box[1], new_x2, src_box[3]]
            # write back
            boxes[src_idx] = new_box
            print(f"[snap] frame {i} {'left' if is_left else 'right'} column: moved box idx {src_idx} to cx={target_cx:.1f} (dx was {dx:.1f}, thr {thr:.1f})")
            return True

        # apply correction depending on mode
        if args.fix_mode == 'snap':
            if _snap_column(True):
                corrected = True
            if _snap_column(False):
                corrected = True
        elif args.fix_mode == 'temporal':
            # temporal mode not implemented here yet; placeholder
            pass

        if corrected:
            print(f"[auto-correct] frame {i} corrected ({args.fix_mode})")

        # 2) only handle 3-corner case
        if len(boxes) == 3:
            completion = VectorCornerCompleter.infer_missing_three(boxes, (w, h))
            if completion is not None:
                boxes.append(completion.box)
                scores.append(None)
                labels.append("pred_3")
        else:
            print(f"[{i}] detections = {len(boxes)} (skip vector-complete; need exactly 3)")

        # 3) save
        annotated = draw_boxes(img, boxes, labels, scores)
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

        corner_candidates = refine_corners(img, boxes_int, corner_params)
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
