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
    args = ap.parse_args()

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

        # Decide which folder
        if not seam_found:
            # test seam condition
            # NOTE: Ignore seam triggers for the first 6 and last 6 frames of the
            # sequence as requested (these boundary frames should not cause a
            # split even if they contain >4 boxes or many gu_cor detections).
            if pos < 6 or pos >= (total_files - 6):
                trigger = False
            else:
                trigger = (len(boxes) > 4) or _gu_same_row_too_close(boxes, classes, w, h)
            if trigger:
                seam_found = True
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
                # normal behavior for C1 segment
                # Prefer gu-first to avoid losing gu near pre-boundary frames
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
        corner_txt = out_dir / f"{idx_label}_corners.txt"
        with open(corner_txt, 'w') as f:
            for cand, b_int in zip(corner_candidates, boxes_int):
                x1,y1,x2,y2 = b_int
                cx, cy = cand.point
                f.write(f"{cand.quadrant} {cx} {cy} {cand.confidence:.4f} {cand.method} {x1} {y1} {x2} {y2}\n")

        corners_vis = img.copy()
        for cand in corner_candidates:
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
