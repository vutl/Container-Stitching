def find_corners_horizontal_only(img, min_line_len_ratio=0.7, max_line_gap=40, debug_img=None):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 80, 200)
    min_line_len = int(w * min_line_len_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_line_len, maxLineGap=max_line_gap)
    y_vals = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            angle = np.degrees(np.arctan2(y2-y1, x2-x1))
            if abs(angle) < 15 or abs(angle) > 165:
                y_vals.extend([y1, y2])
    if len(y_vals) < 2:
        return None, debug_img
    y_top = min(y_vals)
    y_bot = max(y_vals)
    corners = np.array([
        [0, y_top],
        [w-1, y_top],
        [w-1, y_bot],
        [0, y_bot]
    ], dtype=np.float32)
    if debug_img is not None:
        dbg = debug_img.copy()
        cv2.line(dbg, (0, y_top), (w-1, y_top), (0,255,0), 2)
        cv2.line(dbg, (0, y_bot), (w-1, y_bot), (0,0,255), 2)
        for i, pt in enumerate(corners):
            cv2.circle(dbg, tuple(np.round(pt).astype(int)), 8, (255,0,0), -1)
            cv2.putText(dbg, str(i), tuple(np.round(pt).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        return corners, dbg
    return corners, debug_img
def is_valid_container_quad(quad, img_shape, aspect_range=(1.2,3.2), area_frac=(0.12,0.98)):
    h, w = img_shape[:2]
    arr = np.array(quad)
    side1 = np.linalg.norm(arr[0] - arr[1])
    side2 = np.linalg.norm(arr[1] - arr[2])
    side3 = np.linalg.norm(arr[2] - arr[3])
    side4 = np.linalg.norm(arr[3] - arr[0])
    aspect = max(side1, side3) / max(1, max(side2, side4))
    area = 0.5 * abs(
        arr[0,0]*arr[1,1] + arr[1,0]*arr[2,1] + arr[2,0]*arr[3,1] + arr[3,0]*arr[0,1]
        - arr[1,0]*arr[0,1] - arr[2,0]*arr[1,1] - arr[3,0]*arr[2,1] - arr[0,0]*arr[3,1]
    )
    area_frac_val = area / (h*w)
    cx, cy = arr[:,0].mean(), arr[:,1].mean()
    if not (aspect_range[0] < aspect < aspect_range[1]):
        return False
    if not (area_frac[0] < area_frac_val < area_frac[1]):
        return False
    if not (0.08*w < cx < 0.92*w and 0.08*h < cy < 0.92*h):
        return False
    return True
def find_corners_by_hough_pure(img, debug_img=None, min_line_len_ratio=0.3, max_line_gap=60, edge_density_win=15):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE tăng cường biên
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray_eq = clahe.apply(gray)
    # blur = cv2.GaussianBlur(gray_eq, (5,5), 0)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 80, 200)
    # Morphological close để nối đoạn đứt
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    min_line_len_x = int(w * min_line_len_ratio)
    min_line_len_y = int(h * min_line_len_ratio)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min(min_line_len_x, min_line_len_y), maxLineGap=max_line_gap)
    if lines is None or len(lines) < 4:
        return None, debug_img
    horizontals, verticals = [], []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = x2 - x1, y2 - y1
        angle = np.degrees(np.arctan2(dy, dx))
        length = np.hypot(dx, dy)
        # Lọc line dài, gần song song trục X hoặc Y, nằm trong vùng giữa ảnh
        if abs(angle) < 15 or abs(angle) > 165:
            if length > min_line_len_x and min(y1, y2) > 0.05*h and max(y1, y2) < 0.95*h:
                horizontals.append((x1, y1, x2, y2, length))
        elif 75 < abs(angle) < 105:
            if length > min_line_len_y and min(x1, x2) > 0.05*w and max(x1, x2) < 0.95*w:
                verticals.append((x1, y1, x2, y2, length))
    if len(horizontals) < 2 or len(verticals) < 2:
        return None, debug_img
    # Chọn 2 line ngang trên/dưới (theo y), 2 line dọc trái/phải (theo x)
    horizontals = sorted(horizontals, key=lambda l: min(l[1], l[3]))
    top_line = horizontals[0][:4]
    bot_line = horizontals[-1][:4]
    verticals = sorted(verticals, key=lambda l: min(l[0], l[2]))
    left_line = verticals[0][:4]
    right_line = verticals[-1][:4]
    # Tính giao điểm
    def line_to_abcd(x1, y1, x2, y2):
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    def intersect(l1, l2):
        a1, b1, c1 = line_to_abcd(*l1)
        a2, b2, c2 = line_to_abcd(*l2)
        d = a1*b2 - a2*b1
        if d == 0:
            return None
        x = (b1*c2 - b2*c1) / d
        y = (a2*c1 - a1*c2) / d
        return [x, y]
    corners = [
        intersect(top_line, left_line),
        intersect(top_line, right_line),
        intersect(bot_line, right_line),
        intersect(bot_line, left_line)
    ]
    corners = [np.array(pt) for pt in corners if pt is not None]
    # Hậu kiểm hình học
    if len(corners) == 4:
        # Aspect ratio, diện tích, vị trí
        arr = np.array(corners)
        side1 = np.linalg.norm(arr[0] - arr[1])
        side2 = np.linalg.norm(arr[1] - arr[2])
        side3 = np.linalg.norm(arr[2] - arr[3])
        side4 = np.linalg.norm(arr[3] - arr[0])
        aspect = max(side1, side3) / max(1, max(side2, side4))
        area = 0.5 * abs(
            arr[0,0]*arr[1,1] + arr[1,0]*arr[2,1] + arr[2,0]*arr[3,1] + arr[3,0]*arr[0,1]
            - arr[1,0]*arr[0,1] - arr[2,0]*arr[1,1] - arr[3,0]*arr[2,1] - arr[0,0]*arr[3,1]
        )
        if not (0.5 < aspect < 3.0 and 0.1*h*w < area < 0.95*h*w):
            return None, debug_img
    if debug_img is not None:
        dbg = debug_img.copy()
        color_map = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for l in [top_line, bot_line, left_line, right_line]:
            cv2.line(dbg, (int(l[0]),int(l[1])), (int(l[2]),int(l[3])), (0,255,0), 2)
        for i, pt in enumerate(corners):
            cv2.circle(dbg, tuple(np.round(pt).astype(int)), 8, color_map[i%4], -1)
        return np.array(corners, dtype=np.float32), dbg
    return np.array(corners, dtype=np.float32) if len(corners)==4 else None, debug_img
def find_corners_by_hough_box(img, mask, debug_img=None, min_line_len=120, max_line_gap=30, dist_thresh=25):
    # Lấy contour lớn nhất
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, debug_img
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect).astype(np.int32)
    # Chạy Canny + HoughLinesP
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None or len(lines) < 4:
        return None, debug_img
    # 4 cạnh box
    box_lines = [(box[i], box[(i+1)%4]) for i in range(4)]
    # Tìm 4 line Hough gần nhất với 4 cạnh box
    def line_dist(line1, line2):
        # Khoảng cách trung bình giữa 2 đoạn thẳng (4 điểm)
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        return (np.hypot(x1-x3, y1-y3) + np.hypot(x2-x4, y2-y4)) / 2
    hough_lines = [((l[0][0], l[0][1]), (l[0][2], l[0][3])) for l in lines]
    selected = []
    for box_line in box_lines:
        dists = [line_dist(box_line, hl) for hl in hough_lines]
        min_idx = np.argmin(dists)
        if dists[min_idx] < dist_thresh:
            selected.append(hough_lines[min_idx])
        else:
            selected.append(box_line)  # fallback: dùng cạnh box
    # Tính giao điểm 4 line
    def line_to_abcd(p1, p2):
        a = p2[1] - p1[1]
        b = p1[0] - p2[0]
        c = p2[0]*p1[1] - p1[0]*p2[1]
        return a, b, c
    def intersect(l1, l2):
        a1, b1, c1 = line_to_abcd(*l1)
        a2, b2, c2 = line_to_abcd(*l2)
        d = a1*b2 - a2*b1
        if d == 0:
            return None
        x = (b1*c2 - b2*c1) / d
        y = (c1*a2 - c2*a1) / d
        return [x, y]
    # Thứ tự: TL = 0,1; TR = 1,2; BR = 2,3; BL = 3,0
    corners = [
        intersect(selected[0], selected[3]),
        intersect(selected[0], selected[1]),
        intersect(selected[1], selected[2]),
        intersect(selected[2], selected[3])
    ]
    corners = [np.array(pt) for pt in corners if pt is not None]
    if debug_img is not None:
        dbg = debug_img.copy()
        color_map = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for i, l in enumerate(selected):
            cv2.line(dbg, tuple(np.round(l[0]).astype(int)), tuple(np.round(l[1]).astype(int)), color_map[i%4], 2)
        for i, pt in enumerate(corners):
            cv2.circle(dbg, tuple(np.round(pt).astype(int)), 8, color_map[i%4], -1)
        return np.array(corners, dtype=np.float32), dbg
    return np.array(corners, dtype=np.float32) if len(corners)==4 else None, debug_img
import cv2
import numpy as np
from pathlib import Path
import argparse

def find_corners_by_edge_hough_mask(img, mask=None, debug_img=None):
    # Nếu có mask, chỉ lấy vùng mask > 0
    if mask is not None:
        img_masked = img.copy()
        img_masked[mask == 0] = 0
    else:
        img_masked = img
    gray = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200, apertureSize=3)
    if mask is not None:
        edges = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=120, maxLineGap=30)
    h, w = img.shape[:2]
    horizontals, verticals = [], []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            dx, dy = x2 - x1, y2 - y1
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) < 15 or abs(angle) > 165:
                horizontals.append((x1, y1, x2, y2))
            elif 75 < abs(angle) < 105:
                verticals.append((x1, y1, x2, y2))
    # Chọn 2 đường ngang trên/dưới và 2 đường dọc trái/phải (theo vị trí trung bình)
    def select_extreme(lines, axis=1, top=True):
        if not lines:
            return None
        vals = [(l, (l[1]+l[3])/2 if axis==1 else (l[0]+l[2])/2) for l in lines]
        key = min if top else max
        l1 = key(vals, key=lambda x: x[1])[0]
        lines2 = [l for l in lines if l != l1]
        if not lines2:
            return l1, l1
        l2 = key(lines2, key=lambda l: l[1] if axis==1 else l[0])
        return l1, l2
    top_line, bot_line = select_extreme(horizontals, axis=1, top=True), select_extreme(horizontals, axis=1, top=False)
    left_line, right_line = select_extreme(verticals, axis=0, top=True), select_extreme(verticals, axis=0, top=False)
    # Tìm giao điểm
    def line_to_abcd(l):
        x1, y1, x2, y2 = l
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    def intersect(l1, l2):
        a1, b1, c1 = line_to_abcd(l1)
        a2, b2, c2 = line_to_abcd(l2)
        d = a1*b2 - a2*b1
        if d == 0:
            return None
        x = (b1*c2 - b2*c1) / d
        y = (c1*a2 - c2*a1) / d
        return [x, y]
    corners = []
    if top_line and left_line:
        corners.append(intersect(top_line[0], left_line[0]))
    if top_line and right_line:
        corners.append(intersect(top_line[0], right_line[0]))
    if bot_line and right_line:
        corners.append(intersect(bot_line[0], right_line[0]))
    if bot_line and left_line:
        corners.append(intersect(bot_line[0], left_line[0]))
    corners = [np.array(pt) for pt in corners if pt is not None]
    if debug_img is not None:
        dbg = debug_img.copy()
        color_map = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        for l in horizontals:
            cv2.line(dbg, (l[0],l[1]), (l[2],l[3]), (0,255,0), 2)
        for l in verticals:
            cv2.line(dbg, (l[0],l[1]), (l[2],l[3]), (255,0,0), 2)
        for i, pt in enumerate(corners):
            cv2.circle(dbg, tuple(np.round(pt).astype(int)), 8, color_map[i%4], -1)
        return np.array(corners, dtype=np.float32), dbg
    return np.array(corners, dtype=np.float32) if len(corners)==4 else None

def color_mask_from_bounds(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lh, ls, lv = lower
    uh, us, uv = upper
    if lh <= uh:
        mask = cv2.inRange(hsv, lower, upper)
    else:
        lower1 = np.array([0, ls, lv], dtype=np.uint8)
        upper1 = np.array([uh, us, uv], dtype=np.uint8)
        lower2 = np.array([lh, ls, lv], dtype=np.uint8)
        upper2 = np.array([179, us, uv], dtype=np.uint8)
        m1 = cv2.inRange(hsv, lower1, upper1)
        m2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Giới hạn mask trong vùng container ROI để loại bỏ background
    mask = cv2.bitwise_and(mask, container_roi(img))
    comp_mask = mask.copy()
    contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        idx = int(np.argmax(areas))
        big = np.zeros_like(mask)
        cv2.drawContours(big, contours, idx, 255, cv2.FILLED)
        mask = big
    return mask
def container_roi(img, top=0.18, bottom=0.88, left=0.08, right=0.92):
    h, w = img.shape[:2]
    r0, r1 = int(h * top), int(h * bottom)
    c0, c1 = int(w * left), int(w * right)
    mask = np.zeros((h, w), np.uint8)
    mask[r0:r1, c0:c1] = 255
    return mask

def find_container_corners(mask):
    # Tìm tất cả contour, chọn contour hợp lý nhất theo hình học
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best_quad = None
    best_score = -np.inf
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 4:
            corners = approx[:, 0, :]
        else:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            corners = box.astype(int)
        # Sắp xếp lại thứ tự: [top-left, top-right, bottom-right, bottom-left]
        s = corners.sum(axis=1)
        diff = np.diff(corners, axis=1)
        ordered = np.zeros((4,2), dtype=np.float32)
        ordered[0] = corners[np.argmin(s)]
        ordered[2] = corners[np.argmax(s)]
        ordered[1] = corners[np.argmin(diff)]
        ordered[3] = corners[np.argmax(diff)]
        # Kiểm tra hình học
        if is_valid_container_quad(ordered, mask.shape):
            # Ưu tiên diện tích lớn nhất
            area = cv2.contourArea(cnt)
            if area > best_score:
                best_score = area
                best_quad = ordered
    return best_quad if best_quad is not None else None

def sample_hsv_range(img, roi_frac=0.28, tol_h=15, tol_s=60, tol_v=60):
    h, w = img.shape[:2]
    cx0 = int(w * (0.5 - roi_frac / 2.0))
    cy0 = int(h * (0.5 - roi_frac / 2.0))
    cx1 = int(w * (0.5 + roi_frac / 2.0))
    cy1 = int(h * (0.5 + roi_frac / 2.0))
    roi = img[cy0:cy1, cx0:cx1]
    hsv_center = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    med_center = np.median(hsv_center.reshape(-1, 3), axis=0).astype(int)
    h_c, s_c, v_c = int(med_center[0]), int(med_center[1]), int(med_center[2])
    return np.array([h_c-15, max(0, s_c-60), max(0, v_c-60)], dtype=np.uint8), np.array([(h_c+15)%180, min(255, s_c+60), min(255, v_c+60)], dtype=np.uint8)

# --- New: Edge + HoughLinesP based corner detection ---
def find_corners_by_edge_hough(img, roi_frac=0.92, canny1=80, canny2=180, min_line_len=200, max_line_gap=30, debug_path=None):
    h, w = img.shape[:2]
    # Lấy ROI lớn (mặc định 92% ảnh)
    r0, r1 = int(h * (1-roi_frac)/2), int(h * (1+roi_frac)/2)
    c0, c1 = int(w * (1-roi_frac)/2), int(w * (1+roi_frac)/2)
    roi = img[r0:r1, c0:c1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, canny1, canny2)
    # HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None or len(lines) < 4:
        return None
    # Phân loại: các line gần ngang (|dy| < |dx|) là cạnh trên/dưới, gần dọc là trái/phải
    horizontals = []
    verticals = []
    for l in lines:
        x1, y1, x2, y2 = l[0]
        dx, dy = abs(x2-x1), abs(y2-y1)
        if dx > 1.5*dy:
            horizontals.append((x1, y1, x2, y2))
        elif dy > 1.5*dx:
            verticals.append((x1, y1, x2, y2))
    if len(horizontals) < 2 or len(verticals) < 2:
        return None
    # Chọn 2 line trên cùng (y nhỏ nhất) và 2 line dưới cùng (y lớn nhất)
    horizontals = sorted(horizontals, key=lambda l: min(l[1], l[3]))
    top_line = horizontals[0]
    bot_line = horizontals[-1]
    # Chọn 2 line trái/phải (x nhỏ nhất/lớn nhất)
    verticals = sorted(verticals, key=lambda l: min(l[0], l[2]))
    left_line = verticals[0]
    right_line = verticals[-1]
    # Tính giao điểm
    def line_to_abcd(x1, y1, x2, y2):
        # ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2*y1 - x1*y2
        return a, b, c
    def intersect(l1, l2):
        a1, b1, c1 = line_to_abcd(*l1)
        a2, b2, c2 = line_to_abcd(*l2)
        d = a1*b2 - a2*b1
        if d == 0:
            return None
        x = (b1*c2 - b2*c1) / d
        y = (a2*c1 - a1*c2) / d
        return np.array([x, y])
    # Giao điểm: TL = top & left, TR = top & right, BR = bot & right, BL = bot & left
    corners = [
        intersect(top_line, left_line),
        intersect(top_line, right_line),
        intersect(bot_line, right_line),
        intersect(bot_line, left_line)
    ]
    # Dịch lại về tọa độ gốc ảnh
    corners = [c + np.array([c0, r0]) if c is not None else None for c in corners]
    if any([c is None for c in corners]):
        return None
    corners = np.array(corners, dtype=np.float32)
    if debug_path is not None:
        dbg = img.copy()
        for l in [top_line, bot_line]:
            cv2.line(dbg, (l[0]+c0, l[1]+r0), (l[2]+c0, l[3]+r0), (0,255,0), 2)
        for l in [left_line, right_line]:
            cv2.line(dbg, (l[0]+c0, l[1]+r0), (l[2]+c0, l[3]+r0), (255,0,0), 2)
        for i, pt in enumerate(corners):
            cv2.circle(dbg, tuple(pt.astype(int)), 8, (0,0,255), -1)
            cv2.putText(dbg, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imwrite(str(debug_path), dbg)
    return corners

def main():
    parser = argparse.ArgumentParser(description='Find 4 corners of container in each image')
    parser.add_argument('dir', help='directory with images')
    parser.add_argument('start', type=int)
    parser.add_argument('end', type=int)
    parser.add_argument('--suffix', default='_cropped.jpg')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--edge-mask', action='store_true', help='Use edge+Hough in mask color')
    parser.add_argument('--hough-box', action='store_true', help='Use HoughLinesP + minAreaRect box')
    parser.add_argument('--hough-pure', action='store_true', help='Use pure Canny+HoughLinesP, no mask')
    parser.add_argument('--horizontal-only', action='store_true', help='Detect only top/bottom edge, full width')
    args = parser.parse_args()

    base = Path(args.dir)
    paths = [str(base / f"{i}{args.suffix}") for i in range(args.start, args.end + 1)]
    imgs = [cv2.imread(p) for p in paths]
    if any([im is None for im in imgs]):
        print('Missing image!')
        return
    lower, upper = sample_hsv_range(imgs[0])
    debug_dir = base / 'debug_corners'
    if args.debug:
        debug_dir.mkdir(exist_ok=True)
    for idx, img in enumerate(imgs):
        mask = color_mask_from_bounds(img, lower, upper)
        if args.horizontal_only:
            corners, dbg = find_corners_horizontal_only(img, debug_img=img)
        elif args.hough_pure:
            corners, dbg = find_corners_by_hough_pure(img, debug_img=img)
        elif args.hough_box:
            corners, dbg = find_corners_by_hough_box(img, mask, debug_img=img)
        elif args.edge_mask:
            corners, dbg = find_corners_by_edge_hough_mask(img, mask, debug_img=img)
        else:
            corners = find_container_corners(mask)
            dbg = img.copy()
            if corners is not None:
                for i, pt in enumerate(corners):
                    cv2.circle(dbg, tuple(pt.astype(int)), 8, (0,0,255), -1)
                    cv2.putText(dbg, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if corners is not None and len(corners)==4:
            print(f'Image {idx}: corners = {np.round(corners,1).tolist()}')
            if args.debug:
                cv2.imwrite(str(debug_dir / f'corners_{idx:03d}.png'), dbg)
        else:
            print(f'Image {idx}: No corners found!')

if __name__ == '__main__':
    import sys
    if '--edge' in sys.argv:
        # Test edge-based detection
        parser = argparse.ArgumentParser(description='Find 4 corners of container using edge/Hough')
        parser.add_argument('dir', help='directory with images')
        parser.add_argument('start', type=int)
        parser.add_argument('end', type=int)
        parser.add_argument('--suffix', default='_cropped.jpg')
        parser.add_argument('--debug', action='store_true')
        args = parser.parse_args([a for a in sys.argv[1:] if a != '--edge'])
        base = Path(args.dir)
        paths = [str(base / f"{i}{args.suffix}") for i in range(args.start, args.end + 1)]
        imgs = [cv2.imread(p) for p in paths]
        if any([im is None for im in imgs]):
            print('Missing image!')
            sys.exit(1)
        debug_dir = base / 'debug_corners_edge'
        if args.debug:
            debug_dir.mkdir(exist_ok=True)
        for idx, img in enumerate(imgs):
            dbg_path = debug_dir / f'edge_{idx:03d}.png' if args.debug else None
            corners = find_corners_by_edge_hough(img, debug_path=dbg_path)
            if corners is not None:
                print(f'Image {idx}: corners = {corners.tolist()}')
            else:
                print(f'Image {idx}: No corners found!')
    else:
        main()
