import argparse, sys
from pathlib import Path
import cv2
import numpy as np

# ========== Drawing ==========
def draw_boxes(img, boxes, labels, scores=None):
    out = img.copy()
    color_map = {
        "det":     (0, 255,   0),   # green
        "pred_3":  (255,128,  0),   # orange
        "pred_2":  (0, 128, 255),   # blue
        "pred_1":  (180,  0, 180),  # purple
    }
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        lab = labels[i]
        col = color_map.get(lab, (0,255,0))
        cv2.rectangle(out, (x1,y1), (x2,y2), col, 2)
        txt = lab if scores is None or scores[i] is None else f"{lab} {scores[i]:.2f}"
        (tw,th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), col, -1)
        cv2.putText(out, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    return out

# ========== Quadrant helpers ==========
def box_center(b):
    x1,y1,x2,y2 = b
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def assign_quadrant(cx, cy, w, h):
    lr = 'L' if cx < w/2.0 else 'R'
    tb = 'T' if cy < h/2.0 else 'B'
    return tb + lr  # "TL","TR","BL","BR"

def clamp_box(x1,y1,x2,y2,w,h):
    x1,y1 = max(0,int(round(x1))), max(0,int(round(y1)))
    x2,y2 = min(w-1,int(round(x2))), min(h-1,int(round(y2)))
    if x2 <= x1: x2 = min(w-1, x1+1)
    if y2 <= y1: y2 = min(h-1, y1+1)
    return [x1,y1,x2,y2]

def make_box_from_center(cx, cy, bw, bh, w, h):
    return clamp_box(cx-bw/2.0, cy-bh/2.0, cx+bw/2.0, cy+bh/2.0, w, h)

# ========== Estimate container bounds ==========
def estimate_lr_bounds(img, thr_nonblack=12, row_top=0.18, row_bot=0.88,
                       col_frac_thr=0.98, pad_left=0, pad_right=0):
    """
    Left/Right via fraction of non-black pixels across columns in a vertical ROI.
    Inspired by compute_horizontal_crop.
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    non_black = gray > thr_nonblack

    r0, r1 = int(h*row_top), int(h*row_bot)
    if r1 <= r0:
        r0, r1 = int(h*0.2), int(h*0.8)
    roi = non_black[r0:r1, :]

    col_frac = roi.mean(axis=0)  # [0..1]
    # Optional smoothing to reduce jitter
    if col_frac.size > 9:
        col_frac = cv2.GaussianBlur(col_frac.reshape(1,-1).astype(np.float32), (1,9), 0).ravel()

    # left
    x_left = 0
    for i in range(w):
        if col_frac[i] >= col_frac_thr:
            x_left = max(0, i - pad_left)
            break
    # right
    x_right = w-1
    for i in range(w-1, -1, -1):
        if col_frac[i] >= col_frac_thr:
            x_right = min(w-1, i + pad_right)
            break

    if x_right - x_left < max(32, int(0.2*w)):
        x_left, x_right = int(0.1*w), int(0.9*w)
    return x_left, x_right

def estimate_tb_bounds(img, central_width_frac=0.6):
    """
    Top/Bottom via per-row edge energy inside central columns.
    Inspired by find_horizontal_rims.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 40, 120)

    h, w = edges.shape[:2]
    cx0 = int(w*(0.5 - central_width_frac/2.0))
    cx1 = int(w*(0.5 + central_width_frac/2.0))
    cx0, cx1 = max(0, cx0), min(w, cx1) if cx1 > cx0 else (0, w)
    roi = edges[:, cx0:cx1]
    row_sum = roi.sum(axis=1).astype(np.float32)
    row_smooth = cv2.GaussianBlur(row_sum.reshape(-1,1), (9,1), 0).ravel()

    if row_smooth.max() < 2.0:
        return int(0.18*h), int(0.88*h)
    c = h // 2
    y_top    = int(np.argmax(row_smooth[:c]))
    y_bottom = int(np.argmax(row_smooth[c:])) + c
    if y_bottom - y_top < int(0.12*h):
        y_top, y_bottom = int(0.18*h), int(0.88*h)
    return y_top, y_bottom

# ========== Inference (completion) ==========
def infer_3_free(boxes, w, h):
    """Parallelogram rule (free point, before snapping)."""
    pts = [np.array(box_center(b)) for b in boxes]
    whs = [(abs(b[2]-b[0]), abs(b[3]-b[1])) for b in boxes]
    med_w = int(np.median([a for a,_ in whs] or [0.05*w]))
    med_h = int(np.median([b for _,b in whs] or [0.05*h]))
    import math
    d = [
        (math.hypot(*(pts[0]-pts[1])), (0,1,2)),
        (math.hypot(*(pts[0]-pts[2])), (0,2,1)),
        (math.hypot(*(pts[1]-pts[2])), (1,2,0)),
    ]
    _, (i,j,k) = max(d, key=lambda x:x[0])
    pm = pts[i] + pts[j] - pts[k]  # free center
    return make_box_from_center(pm[0], pm[1], med_w, med_h, w, h)

def infer_3_snapped(boxes, w, h, bounds):
    """Infer 4th, then SNAP to estimated bounds corner."""
    pbox = infer_3_free(boxes, w, h)
    all_q = {"TL","TR","BL","BR"}
    quads = set(assign_quadrant(*box_center(b), w=w, h=h) for b in boxes)
    missing = list(all_q - quads)

    xl, xr, yt, yb = bounds
    bw = abs(pbox[2]-pbox[0]); bh = abs(pbox[3]-pbox[1])

    if missing:
        miss = missing[0]
        if   miss == "TL": cx, cy = xl + bw/2.0, yt + bh/2.0
        elif miss == "TR": cx, cy = xr - bw/2.0, yt + bh/2.0
        elif miss == "BL": cx, cy = xl + bw/2.0, yb - bh/2.0
        else:              cx, cy = xr - bw/2.0, yb - bh/2.0
    else:
        # clamp to inner rectangle if quadrant ambiguous
        cx, cy = box_center(pbox)
        cx = min(max(cx, xl + bw/2.0), xr - bw/2.0)
        cy = min(max(cy, yt + bh/2.0), yb - bh/2.0)
    return make_box_from_center(cx, cy, bw, bh, w, h)

def infer_2_snapped(
    img, boxes, w, h, bounds,
    head_span_frac=0.75,       # cần phủ >= 75% chiều cao ảnh -> coi là đầu container
    head_side_margin_frac=0.18,# và 2 box đang ở rất sát mép ảnh hiện có (<=18% bề rộng)
    inset_px=6,                # chừa mép khi đẩy sang ảnh đối diện
):
    (xl, xr, yt, yb) = bounds
    xmid = 0.5*(xl+xr); ymid = 0.5*(yt+yb)

    # centers & sizes
    cs  = [box_center(b) for b in boxes]
    whs = [(abs(b[2]-b[0]), abs(b[3]-b[1])) for b in boxes]
    bw  = int(np.median([a for a,_ in whs] or [0.05*w]))
    bh  = int(np.median([b for _,b in whs] or [0.05*h]))

    def clamp_img_x(cx): return float(np.clip(cx, bw/2.0, w - bw/2.0))
    def clamp_img_y(cy): return float(np.clip(cy, bh/2.0, h - bh/2.0))

    # helper: box edges
    x1s = [b[0] for b in boxes]; x2s = [b[2] for b in boxes]
    y1s = [b[1] for b in boxes]; y2s = [b[3] for b in boxes]

    (x0,y0),(x1,y1) = cs
    same_side_vertical   = (x0 < xmid) == (x1 < xmid)        # cùng L/R
    same_side_horizontal = (y0 < ymid) == (y1 < ymid)        # cùng T/B  (đã sửa bug)

    preds = []
    if same_side_vertical:
        # ---- Auto detect "head" dựa trên span & sát mép ảnh hiện có ----
        span_y = max(y2s) - min(y1s)                         # dùng cạnh box, không dùng tâm
        near_top_bottom = span_y >= head_span_frac * h

        side_is_right = (x0 >= xmid) and (x1 >= xmid)
        if side_is_right:
            # khoảng cách cạnh phải của box tới mép PHẢI ảnh
            near_current_side = min(w - max(x2s), w - max(x1s)) <= head_side_margin_frac * w
        else:
            # khoảng cách cạnh trái của box tới mép TRÁI ảnh
            near_current_side = min(min(x1s), min(x2s)) <= head_side_margin_frac * w

        is_head = near_top_bottom and near_current_side

        if is_head:
            # --- ĐẨY sang mép ẢNH đối diện, GIỮ nguyên y ---
            if side_is_right:
                x_target = inset_px + bw/2.0                  # đang ở PHẢI -> đẩy sang TRÁI ảnh
            else:
                x_target = w - inset_px - bw/2.0              # đang ở TRÁI -> đẩy sang PHẢI ảnh

            for (cx, cy) in cs:
                preds.append(make_box_from_center(clamp_img_x(x_target),
                                                  clamp_img_y(cy), bw, bh, w, h))
        else:
            # --- Không phải đầu container: mirror qua trục dọc của CONTAINER, giữ y ---
            for (cx, cy) in cs:
                cx_m = xl + xr - cx
                preds.append(make_box_from_center(clamp_img_x(cx_m),
                                                  clamp_img_y(cy), bw, bh, w, h))

    elif same_side_horizontal:
        # Mirror qua trục NGANG của container (ít gặp trong ví dụ của bạn)
        for (cx, cy) in cs:
            cy_m = yt + yb - cy
            preds.append(make_box_from_center(clamp_img_x(cx),
                                              clamp_img_y(cy_m), bw, bh, w, h))
    else:
        # Hai điểm nằm chéo: tạo (x0,y1) và (x1,y0), rồi kẹp trong ảnh
        for (cx, cy) in [(x0,y1), (x1,y0)]:
            preds.append(make_box_from_center(clamp_img_x(cx),
                                              clamp_img_y(cy), bw, bh, w, h))
    return preds


def infer_1_snapped(box, w, h, bounds):
    """Mirror a single detection across both axes and SNAP to nearest bound corners."""
    cx, cy = box_center(box)
    bw, bh = abs(box[2]-box[0]), abs(box[3]-box[1])
    xl, xr, yt, yb = bounds

    # target corners centers
    cand = [
        (xl + bw/2.0, yt + bh/2.0),  # TL
        (xr - bw/2.0, yt + bh/2.0),  # TR
        (xl + bw/2.0, yb - bh/2.0),  # BL
        (xr - bw/2.0, yb - bh/2.0),  # BR
    ]
    # choose three farthest from (cx,cy) so they are other corners
    d = [ ( (cx-x)**2 + (cy-y)**2, (x,y) ) for (x,y) in cand ]
    d.sort(reverse=True)
    out=[]
    for _, (x,y) in d[:3]:
        out.append(make_box_from_center(x, y, bw, bh, w, h))
    return out

# ========== Detection ==========
def load_model(model_path):
    try:
        from ultralytics import YOLO
        return YOLO(str(model_path))
    except Exception as e:
        print("Ultralytics not available:", e)
        sys.exit(1)

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
    quads = {"TL":[], "TR":[], "BL":[], "BR":[]}
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

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, default='last11s_cortop_10_08.pt')
    ap.add_argument('--dir', type=str, default='mnt/2025-09-04_14-35-53')
    ap.add_argument('--indices', type=int, nargs='+', default=[42,50,60,70,74])
    ap.add_argument('--suffix', type=str, default='_cropped.jpg')
    ap.add_argument('--out', type=str, default='out_annot')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.6)
    ap.add_argument('--max-per-quad', type=int, default=1)
    # bounds tunables
    ap.add_argument('--col-frac-thr', type=float, default=0.98)
    ap.add_argument('--row-top', type=float, default=0.18)
    ap.add_argument('--row-bot', type=float, default=0.88)
    args = ap.parse_args()

    model = load_model(args.model)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    for i in args.indices:
        p = Path(args.dir) / f"{i}{args.suffix}"
        if not p.exists():
            print(f"Missing: {p}"); continue
        img = cv2.imread(str(p))
        if img is None:
            print(f"Fail read: {p}"); continue
        h, w = img.shape[:2]

        # --- 1) Detect
        boxes, scores = run_detect(model, img, conf=args.conf, iou=args.iou)
        boxes, scores = keep_best_per_quadrant(boxes, scores, w, h, args.max_per_quad)

        labels = ["det"] * len(boxes)

        # --- 2) Estimate bounds once per image
        x_left, x_right = estimate_lr_bounds(
            img, col_frac_thr=args.col_frac_thr,
            row_top=args.row_top, row_bot=args.row_bot
        )
        y_top, y_bottom = estimate_tb_bounds(img)
        bounds = (x_left, x_right, y_top, y_bottom)

        # --- 3) Complete missing corners with SNAPPING / AUTO-HEAD
        if len(boxes) == 3:
            boxes.append(infer_3_snapped(boxes, w, h, bounds))
            scores.append(None); labels.append("pred_3")
        elif len(boxes) == 2:
            preds = infer_2_snapped(img, boxes, w, h, bounds)
            boxes.extend(preds); scores.extend([None]*len(preds)); labels.extend(["pred_2"]*len(preds))
        elif len(boxes) == 1:
            preds = infer_1_snapped(boxes[0], w, h, bounds)
            boxes.extend(preds); scores.extend([None]*len(preds)); labels.extend(["pred_1"]*len(preds))
        elif len(boxes) == 0:
            print(f"No detections: {p.name}")
            cv2.imwrite(str(out_dir / f"{i}_annot_cropped.jpg"), img)
            continue

        # --- 4) Draw & Save
        annotated = draw_boxes(img, boxes, labels, scores)
        outp = out_dir / f"{i}_annot_cropped.jpg"
        cv2.imwrite(str(outp), annotated)
        print(f"Wrote: {outp}")

    print("Done.")

if __name__ == "__main__":
    main()
