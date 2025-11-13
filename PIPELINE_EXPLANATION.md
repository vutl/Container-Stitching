# GIẢI THÍCH CHI TIẾT PIPELINE GHÉP CONTAINER

## TÓM TẮT NHANH

Pipeline này ghép các ảnh chụp mặt bên container thành panorama hoàn chỉnh bằng 5 bước chính:

1. **Detect & Split**: Tìm 4 góc container bằng YOLO → Tách thành 1-2 containers riêng
2. **Trim Borders**: Cắt bỏ viền đen, cập nhật tọa độ góc
3. **Align Frames**: Warp tất cả frames vào canvas chung dựa trên median container height
4. **Rectify**: Perspective transform để container thành hình chữ nhật hoàn hảo
5. **Stitch**: Ghép từng frame theo thứ tự bằng feature matching (SIFT/LoFTR) + seam blending

---

## BƯỚC 1: CORNER DETECTION & SPLIT

### Mục đích
- Tìm 4 góc container: TL (top-left), TR (top-right), BR (bottom-right), BL (bottom-left)
- Phát hiện nếu có 2 containers trong 1 frame → Tách thành 2 sequences riêng

### Quy trình chi tiết

#### 1.1. YOLO Detection
```python
# File: test_sides_split.py
model.predict(img, conf=0.005, iou=0.15)
# → Trả về: boxes, scores, classes
# Classes: 'gu_cor' (container head corner), 'edge_cor' (container edge corner)
```

**Output YOLO**: Mỗi box có format `[x1, y1, x2, y2, score, class]`

#### 1.2. Quadrant Assignment
```python
# Chia ảnh thành 4 góc phần tư (TL/TR/BR/BL)
cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
quadrant = "TL" if (cx < w/2 and cy < h/2) else ...
```

#### 1.3. Corner Refinement
**Với `gu_cor`** (container head): Dùng **head-offset rule**
```python
# Đặt corner ở offset cố định từ tâm box
# TL: (x1 + offset, y1 + offset)
# TR: (x2 - offset, y1 + offset)
# BR: (x2 - offset, y2 - offset)
# BL: (x1 + offset, y2 - offset)
```

**Với `edge_cor`**: Dùng **gradient-based refinement**
```python
# File: corner_refinement_sides.py
# 1. Tính Sobel gradients (gx, gy)
gx = cv2.Sobel(gray, CV2_64F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, CV2_64F, 0, 1, ksize=3)

# 2. Tìm peak gradient trong margin area của box
# Ví dụ TL corner:
#   - Tìm peak gx trong left margin (cột gần x1)
#   - Tìm peak gy trong top margin (hàng gần y1)
#   - Iteratively refine: col → row → col...

# 3. Return refined corner position
```

#### 1.4. Split Logic (Phát hiện 2 containers)
```python
# Kiểm tra mỗi frame xem có seam (ranh giới giữa 2 containers) không
if total_detections > 4:  # Có >4 boxes → 2 containers
    # Tìm seam_x dựa trên vị trí gu_cor clusters
    # Split boxes thành:
    #   - C1 (bên trái seam): 2 edge_cor + 2 gu_cor leftmost
    #   - C2 (bên phải seam): 2 edge_cor + 2 gu_cor rightmost
```

**Key insight**: Khi frame có 2 containers, có thể thấy 5-6 boxes (4 của C1 + 2-3 của C2 visible)

### Output Bước 1
```
split_c1/
  ├─ img_0_annot.jpg       # Visualization: boxes + class labels
  ├─ img_0_annot.jpg.txt   # Raw YOLO boxes (x1 y1 x2 y2 score)
  ├─ img_0_corners.jpg     # Visualization: refined corners
  └─ img_0_corners.txt     # TL x y\nTR x y\nBR x y\nBL x y
split_c2/  (nếu có 2 containers)
```

---

## BƯỚC 2: TRIM BLACK BORDERS

### Mục đích
Cắt bỏ viền đen xung quanh container để tiết kiệm memory và tăng độ chính xác khi matching

### Quy trình chi tiết

#### 2.1. Detect Non-Black Pixels
```python
# File: trim_black_update_corners.py
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Đếm số pixels > thresh (default 8) mỗi cột/hàng
col_nonblack = (gray > 8).sum(axis=0)  # Sum theo chiều dọc
row_nonblack = (gray > 8).sum(axis=1)  # Sum theo chiều ngang

# Tìm bounds: cột/hàng đầu tiên có đủ non-black pixels
min_rows = int(0.02 * h)  # Yêu cầu ít nhất 2% rows không đen
min_cols = int(0.02 * w)

left = first column where col_nonblack[left] >= min_rows
right = last column where col_nonblack[right] >= min_rows
top = first row where row_nonblack[top] >= min_cols
bottom = last row where row_nonblack[bottom] >= min_cols
```

#### 2.2. Crop Image
```python
trimmed_img = img[top:bottom+1, left:right+1]
cv2.imwrite('img_0_trim.jpg', trimmed_img)
```

#### 2.3. Update Corner Coordinates
```python
# Corners file cũ: TL (100, 50), TR (900, 52), ...
# Sau khi trim: left=20, top=10
# → Corners file mới:
TL_new = (100 - 20, 50 - 10) = (80, 40)
TR_new = (900 - 20, 52 - 10) = (880, 42)
# ...

# Clamp về trong ảnh mới
TL_x = clamp(TL_x, 0, new_width - 1)
TL_y = clamp(TL_y, 0, new_height - 1)
```

### Output Bước 2
```
trimmed_c1/
  └─ img_0_trim.jpg
trimmed_corners_c1/
  └─ img_0_corners.txt  # Tọa độ đã cập nhật theo ảnh trimmed
```

---

## BƯỚC 3: ALIGN FRAMES

### Mục đích
Warp tất cả frames vào **uniform canvas** để:
- Container có **cùng chiều cao** (median của tất cả frames)
- Container được **vertically centered**
- **Left/right padding** để bảo toàn toàn bộ content

### Quy trình chi tiết

#### 3.1. Tính Median Container Height
```python
# File: warp_align_sides.py
# Từ mỗi frame, tính container height từ corners
cont_h = max(
    norm(TL - BL),  # Left edge height
    norm(TR - BR)   # Right edge height
)

# Lấy median của tất cả frames
cont_h_med = median([cont_h_0, cont_h_1, ..., cont_h_73])
# Ví dụ: cont_h_med = 1011 pixels
```

#### 3.2. Tạo Uniform Canvas
```python
target_w = max(frame_widths)  # Ví dụ: 961 pixels
target_h = cont_h_med + top_margin + bottom_margin
# Ví dụ: 1011 + 20 + 20 = 1051 pixels

canvas = np.zeros((target_h, target_w, 3), dtype=uint8)
```

#### 3.3. Perspective Warp Mỗi Frame
```python
# Source points: 4 góc container trong ảnh gốc
src_pts = np.float32([TL, TR, BR, BL])

# Destination points: 4 góc trên canvas uniform
# Container được center vertically
y_top = top_margin  # 20
y_bot = top_margin + cont_h_med  # 20 + 1011 = 1031

# Left/right padding để giữ nguyên width
left_pad = max(0, floor(min(TL_x, BL_x)))
right_pad = max(0, ceil((w - 1) - max(TR_x, BR_x)))

x_left = left_pad
x_right = target_w - right_pad

dst_pts = np.float32([
    [x_left, y_top],     # TL
    [x_right, y_top],    # TR
    [x_right, y_bot],    # BR
    [x_left, y_bot]      # BL
])

# Tính homography
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Warp
aligned_img = cv2.warpPerspective(img, H, (target_w, target_h))
```

**Visual Example**:
```
Before align:           After align (uniform canvas):
┌─────────────┐        ┌────────────────────────┐
│ ┌──────┐    │        │                        │ ← top margin (20px)
│ │      │    │   →    │ ┌──────────────────┐  │
│ │ CONT │    │        │ │   CONTAINER      │  │ ← cont_h_med (1011px)
│ │      │    │        │ └──────────────────┘  │
│ └──────┘    │        │                        │ ← bottom margin (20px)
│             │        └────────────────────────┘
└─────────────┘         ↑                     ↑
                     left_pad            right_pad
```

### Output Bước 3
```
aligned_c1/
  ├─ 0_aligned.jpg
  └─ 0_aligned_corners.txt  # TL, TR, BR, BL trên canvas aligned
```

---

## BƯỚC 4: RECTIFY TO RECTANGLES

### Mục đích
Transform container thành **perfect rectangle** (không skew, không perspective distortion)

### Quy trình chi tiết

#### 4.1. Tính Target Rectangle Dimensions
```python
# File: rectify_to_rectangle_sides.py
# Từ aligned corners
width_top = norm(TR - TL)
width_bottom = norm(BR - BL)
height_left = norm(BL - TL)
height_right = norm(BR - TR)

target_w = max(width_top, width_bottom)  # Ví dụ: 960
target_h_core = max(height_left, height_right)  # Ví dụ: 1010

# Thêm vertical margin
target_h = target_h_core + 2 * vertical_margin  # 1010 + 40 = 1050
```

#### 4.2. Perspective Transform
```python
src_pts = np.float32([TL, TR, BR, BL])

# Output rectangle (perfect 90° corners)
dst_pts = np.float32([
    [0, vertical_margin],              # TL
    [target_w, vertical_margin],       # TR
    [target_w, target_h - vertical_margin],  # BR
    [0, target_h - vertical_margin]    # BL
])

H = cv2.getPerspectiveTransform(src_pts, dst_pts)
rectified_img = cv2.warpPerspective(aligned_img, H, (target_w, target_h))
```

**Visual**:
```
Aligned (có skew):       Rectified (perfect rectangle):
┌──────────────┐        ┌──────────────────┐
│╱          ╲  │        │┌────────────────┐│ ← margin
││  CONTAINER│ │   →    ││   CONTAINER    ││
│╲          ╱  │        ││                ││
└──────────────┘        │└────────────────┘│ ← margin
                        └──────────────────┘
```

### Output Bước 4
```
rectified_c1/
  ├─ 0_aligned.jpg   # 960x1050 perfect rectangles
  ├─ 1_aligned.jpg
  └─ ...
```

**CHÚ Ý**: Tất cả rectified images có **cùng kích thước** (960x1050)

---

## BƯỚC 5: STITCH PANORAMAS ⭐ (QUAN TRỌNG NHẤT)

### Mục đích
Ghép 74 frames thành 1 panorama dài bằng **incremental stitching**

### 5.1. KHỞI TẠO

```python
# File: test3_sides.py → stitch_incremental_with_corners()

# Đọc frame đầu tiên làm mosaic ban đầu
mosaic = cv2.imread('0_aligned.jpg')  # 960x1050
mosaic_corners = read_corners('0_corners.txt')

# Canvas transform: identity (frame 0 is reference)
H_canvas = np.eye(3)
```

### 5.2. LOOP QUA TỪNG FRAME

```python
for idx in [1, 2, 3, ..., 73]:  # stride=1
    # === ĐỌC FRAME HIỆN TẠI ===
    cur_img = cv2.imread(f'{idx}_aligned.jpg')
    cur_corners = read_corners(f'{idx}_corners.txt')
    
    # === TẠO MASK ===
    # QUAN TRỌNG: Mask chỉ cho phép detect keypoints TRONG container
    cur_mask = mask_from_corners(cur_img.shape, cur_corners, pad=0)
    # → cur_mask[y, x] = 255 nếu (x,y) trong bbox của 4 corners
    # → cur_mask[y, x] = 0 nếu ngoài (background)
    
    mosaic_gray = cv2.cvtColor(mosaic, cv2.COLOR_BGR2GRAY)
    mosaic_mask = (mosaic_gray > 0).astype(np.uint8) * 255
    # → mosaic_mask = vùng đã có content (pixels không đen)
    
    # === ESTIMATE TRANSFORM ===
    H = estimate_translation_masked(
        cur_img, mosaic, 
        cur_mask, mosaic_mask
    )
    # → Trả về translation matrix hoặc None nếu fail
    
    if H is None:
        # Fallback: ECC (optical flow)
        try:
            H = cv2.findTransformECC(...)
        except:
            print("ECC failed; skipping frame")
            continue
    
    # === GHÉP VÀO MOSAIC ===
    mosaic, _, _ = expand_canvas(
        mosaic, H, cur_img,
        blend_mode='seam',
        seam_width=3
    )
```

---

### 5.3. ESTIMATE TRANSLATION (MATCHING CORE)

Đây là **TIM MẠ CỦA PIPELINE** - nơi tìm transform giữa 2 frames

```python
def estimate_translation_masked(src, tar, mask_src, mask_tar):
    """
    Args:
        src: Current frame (960x1050)
        tar: Mosaic (current panorama, ví dụ 3500x1072)
        mask_src: Container mask của src
        mask_tar: Content mask của mosaic
    
    Returns:
        H: Translation matrix [[1,0,dx],[0,1,dy],[0,0,1]] hoặc None
    """
    
    detector = get_detector()  # 'SIFT' hoặc 'LOFTR'
    
    # ==========================================
    # BRANCH 1: SIFT/ORB (classical features)
    # ==========================================
    if detector in ['SIFT', 'ORB', 'AKAZE', 'KAZE']:
        gray_src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray_tar = cv2.cvtColor(tar, cv2.COLOR_BGR2GRAY)
        
        # 1. DETECT KEYPOINTS (CHỈ TRONG MASK!)
        kp_src, des_src = detector.detectAndCompute(gray_src, mask_src)
        kp_tar, des_tar = detector.detectAndCompute(gray_tar, mask_tar)
        #        ↑ mask_src chỉ cho phép detect trong container bbox
        #        ↑ mask_tar chỉ cho phép detect trong vùng có content
        
        if des_src is None or des_tar is None or len(kp_src) < 3:
            return None  # Not enough keypoints
        
        # 2. MATCH DESCRIPTORS
        bf = cv2.BFMatcher(cv2.NORM_L2)  # Euclidean distance
        raw_matches = bf.knnMatch(des_src, des_tar, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 3:
            return None
        
        # 3. EXTRACT POINT PAIRS
        pts_src = np.float32([kp_src[m.queryIdx].pt for m in good_matches])
        pts_tar = np.float32([kp_tar[m.trainIdx].pt for m in good_matches])
        
        # 4. COMPUTE DELTAS
        deltas = pts_tar - pts_src
        # deltas[i] = (dx_i, dy_i) cho match thứ i
        
        # 5. MEDIAN ESTIMATOR
        dx_median = float(np.median(deltas[:, 0]))
        dy_median = float(np.median(deltas[:, 1]))
        
        # 6. INLIER FILTERING (loại outliers)
        residuals = deltas - np.array([dx_median, dy_median])
        dist = np.linalg.norm(residuals, axis=1)
        inliers = dist <= 12.0  # threshold: 12 pixels
        
        if np.count_nonzero(inliers) < 3:
            return None
        
        # 7. RECOMPUTE từ inliers
        dx = float(np.median(deltas[inliers, 0]))
        dy = float(np.median(deltas[inliers, 1]))
        
        # 8. BUILD TRANSLATION MATRIX
        H = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        return H
    
    # ==========================================
    # BRANCH 2: LoFTR (learned matcher)
    # ==========================================
    elif detector == 'LOFTR':
        # 1. CROP IMAGES TO MASK BBOX (NEW!)
        # Tránh match background
        if mask_src is not None and mask_src.sum() > 0:
            ys, xs = np.where(mask_src > 0)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            src_crop = src[y0:y1, x0:x1]
            x_src_offset, y_src_offset = x0, y0
        else:
            src_crop = src
            x_src_offset, y_src_offset = 0, 0
        
        # Tương tự với mosaic
        if mask_tar is not None and mask_tar.sum() > 0:
            ys, xs = np.where(mask_tar > 0)
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            tar_crop = tar[y0:y1, x0:x1]
            x_tar_offset, y_tar_offset = x0, y0
        else:
            tar_crop = tar
            x_tar_offset, y_tar_offset = 0, 0
        
        # 2. RUN LoFTR ON CROPPED IMAGES
        mkpts_src, mkpts_tar, mconf = run_loftr_match(src_crop, tar_crop)
        #                                   ↑ Chạy trên ảnh đã crop
        
        # 3. TRANSLATE POINTS BACK TO ORIGINAL COORDINATES
        mkpts_src[:, 0] += x_src_offset
        mkpts_src[:, 1] += y_src_offset
        mkpts_tar[:, 0] += x_tar_offset
        mkpts_tar[:, 1] += y_tar_offset
        
        if len(mkpts_src) < 3:
            return None
        
        # 4-8: SAME AS SIFT (median + inlier filtering)
        deltas = mkpts_tar - mkpts_src
        dx_median = float(np.median(deltas[:, 0]))
        dy_median = float(np.median(deltas[:, 1]))
        
        residuals = deltas - np.array([dx_median, dy_median])
        dist = np.linalg.norm(residuals, axis=1)
        inliers = dist <= 12.0
        
        if np.count_nonzero(inliers) < 3:
            return None
        
        dx = float(np.median(deltas[inliers, 0]))
        dy = float(np.median(deltas[inliers, 1]))
        
        H = np.array([
            [1.0, 0.0, dx],
            [0.0, 1.0, dy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        return H
```

**KEY POINTS**:

1. **Mask được dùng ĐỂ:**
   - **SIFT/ORB**: Giới hạn nơi detect keypoints (chỉ trong container bbox)
   - **LoFTR (NEW)**: Crop ảnh trước khi inference → loại background hoàn toàn

2. **Median thay vì Mean**:
   - Median **robust** hơn với outliers
   - Nhưng không đủ khi có **periodic aliasing** (corrugation patterns)
   - → **CẦN RANSAC** để filter geometric inconsistencies

3. **Inlier filtering**:
   - Loại matches có residual > 12px so với median
   - Yêu cầu ít nhất 3 inliers còn lại

4. **Translation-only**:
   - Chỉ dùng `dx, dy` (2 parameters)
   - KHÔNG dùng rotation, scale, shear
   - Lý do: Container đã rectified → chỉ cần dịch ngang

---

### 5.4. EXPAND CANVAS (WARPING & BLENDING)

```python
def expand_canvas(mosaic, h_to_canvas, new_img, blend_mode='seam'):
    """
    Args:
        mosaic: Current panorama (ví dụ 3500x1072)
        h_to_canvas: Transform from new_img → mosaic space
        new_img: Frame mới (960x1050)
        blend_mode: 'seam', 'feather', hoặc 'none'
    
    Returns:
        new_mosaic, new_h_to_canvas, mask
    """
    
    # === 1. TÍNH CANVAS MỚI (to fit both old and new) ===
    h_cur, w_cur = mosaic.shape[:2]
    h_new, w_new = new_img.shape[:2]
    
    # Transform 4 góc của new_img vào mosaic space
    corners_new = np.float32([
        [0, 0],
        [w_new, 0],
        [w_new, h_new],
        [0, h_new]
    ])
    corners_warped = cv2.perspectiveTransform(
        corners_new[None, :, :], h_to_canvas
    )[0]
    
    # Bounding box của warped new_img
    x_min_new = min(corners_warped[:, 0])
    x_max_new = max(corners_warped[:, 0])
    y_min_new = min(corners_warped[:, 1])
    y_max_new = max(corners_warped[:, 1])
    
    # Canvas mới phải chứa: mosaic cũ (0→w_cur, 0→h_cur) + warped new
    x_min = min(0, x_min_new)
    x_max = max(w_cur, x_max_new)
    y_min = min(0, y_min_new)
    y_max = max(h_cur, y_max_new)
    
    canvas_w = int(np.ceil(x_max - x_min))
    canvas_h = int(np.ceil(y_max - y_min))
    
    # === 2. TRANSLATION TO CANVAS ===
    # Nếu x_min < 0 hoặc y_min < 0, cần dịch để fit
    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0
    
    T_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # === 3. PLACE OLD MOSAIC ===
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    ox, oy = int(offset_x), int(offset_y)
    canvas[oy:oy+h_cur, ox:ox+w_cur] = mosaic
    
    # === 4. WARP NEW IMAGE INTO CANVAS ===
    H_new_to_canvas = T_offset @ h_to_canvas
    warped_new = cv2.warpPerspective(
        new_img, H_new_to_canvas, (canvas_w, canvas_h)
    )
    
    # === 5. CREATE MASKS ===
    # Mask của old mosaic content
    mask_old = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8) * 255
    
    # Mask của warped new image
    full_mask = np.ones((h_new, w_new), dtype=np.uint8) * 255
    mask_new = cv2.warpPerspective(
        full_mask, H_new_to_canvas, (canvas_w, canvas_h)
    )
    
    # === 6. FIND OVERLAP ===
    overlap = cv2.bitwise_and(mask_new, mask_old)
    #  ↑ QUAN TRỌNG: Overlap KHÔNG tìm từ keypoints!
    #                 Mà từ intersection của 2 masks sau khi warp
    
    # Bounding box của overlap region
    ys, xs = np.where(overlap > 0)
    if len(ys) == 0:
        # No overlap → hard paste
        result = cv2.bitwise_or(canvas, warped_new)
        return result, H_new_to_canvas, mask_new
    
    oy0, oy1 = int(ys.min()), int(ys.max()) + 1
    ox0, ox1 = int(xs.min()), int(xs.max()) + 1
    
    # === 7. BLENDING ===
    if blend_mode == 'seam':
        # Extract overlap regions
        old_ov = canvas[oy0:oy1, ox0:ox1]
        new_ov = warped_new[oy0:oy1, ox0:ox1]
        overlap_crop = overlap[oy0:oy1, ox0:ox1]
        
        # Compute seam using Dynamic Programming
        gray_old = cv2.cvtColor(old_ov, cv2.COLOR_BGR2GRAY)
        gray_new = cv2.cvtColor(new_ov, cv2.COLOR_BGR2GRAY)
        
        # Cost matrix: absolute difference
        cost = np.abs(gray_old.astype(float) - gray_new.astype(float))
        cost[overlap_crop == 0] = 1e9  # Penalty outside overlap
        
        # DP: find min-error vertical seam
        h_ov, w_ov = cost.shape
        dp = np.zeros_like(cost)
        dp[0, :] = cost[0, :]
        
        for y in range(1, h_ov):
            for x in range(w_ov):
                candidates = [dp[y-1, x]]
                if x > 0:
                    candidates.append(dp[y-1, x-1])
                if x < w_ov - 1:
                    candidates.append(dp[y-1, x+1])
                dp[y, x] = cost[y, x] + min(candidates)
        
        # Backtrack to find seam path
        seam_x = np.zeros(h_ov, dtype=int)
        seam_x[-1] = int(np.argmin(dp[-1, :]))
        
        for y in range(h_ov - 2, -1, -1):
            prev_x = seam_x[y+1]
            candidates = [dp[y, prev_x]]
            choices = [prev_x]
            if prev_x > 0:
                candidates.append(dp[y, prev_x-1])
                choices.append(prev_x-1)
            if prev_x < w_ov - 1:
                candidates.append(dp[y, prev_x+1])
                choices.append(prev_x+1)
            seam_x[y] = choices[np.argmin(candidates)]
        
        # Create blend mask: left of seam = old, right = new
        blend_mask = np.zeros((h_ov, w_ov), dtype=np.float32)
        for y in range(h_ov):
            blend_mask[y, :seam_x[y]] = 0.0  # old
            blend_mask[y, seam_x[y]:] = 1.0  # new
        
        # Soften seam with Gaussian blur
        blend_mask = cv2.GaussianBlur(blend_mask, (seam_width*2+1, seam_width*2+1), 0)
        
        # Alpha blending
        blended = (
            old_ov * (1 - blend_mask[:, :, None]) +
            new_ov * blend_mask[:, :, None]
        ).astype(np.uint8)
        
        # Place blended overlap back
        canvas[oy0:oy1, ox0:ox1] = blended
        
        # Paste non-overlap regions of new image
        non_overlap_new = (mask_new > 0) & (mask_old == 0)
        canvas[non_overlap_new] = warped_new[non_overlap_new]
    
    elif blend_mode == 'feather':
        # Distance-transform alpha blending
        # (similar logic, using cv2.distanceTransform for smooth gradients)
        ...
    
    elif blend_mode == 'none':
        # Hard cut: prefer new image in overlap
        canvas[mask_new > 0] = warped_new[mask_new > 0]
    
    return canvas, H_new_to_canvas, mask_new
```

**KEY POINTS**:

1. **Overlap KHÔNG tìm từ keypoints**:
   - Overlap = `mask_new & mask_old` (intersection sau khi warp)
   - Nếu H (transform) sai → overlap region sai → ghép lệch!

2. **Seam finding**:
   - DP tìm đường min-cost vertical seam
   - Cost = grayscale absolute difference
   - **VẤN ĐỀ**: Với uniform texture (container), absdiff rất nhỏ → seam không ổn định

3. **Blending chỉ trong overlap**:
   - Vùng non-overlap: paste trực tiếp
   - Vùng overlap: alpha blend theo seam mask

---

### 5.5. KẾT QUẢ CHẠY GẦN NHẤT (LoFTR + Mask)

```
Processing 1_aligned.jpg -> mosaic
  -> Stitched; canvas 988x1054
Processing 2_aligned.jpg -> mosaic
  -> Stitched; canvas 1034x1059
...
Processing 54_aligned.jpg -> mosaic
  -> Stitched; canvas 3646x1090   ← Canvas KHÔNG tăng
Processing 55_aligned.jpg -> mosaic
  -> Stitched; canvas 3675x1090   ← Tăng 29px
Processing 56_aligned.jpg -> mosaic
  -> Stitched; canvas 3675x1090   ← STUCK (dx=0)
Processing 57_aligned.jpg -> mosaic
  -> ECC failed; skipping frame    ← LoFTR fail, ECC cũng fail
Processing 58_aligned.jpg -> mosaic
  -> ECC failed; skipping frame
Processing 59_aligned.jpg -> mosaic
  -> Stitched; canvas 3675x1090   ← Recovered nhưng dx=0
...
Processing 64_aligned.jpg -> mosaic
  -> Stitched; canvas 3675x1090   ← Vẫn stuck
Processing 65_aligned.jpg -> mosaic
  -> Stitched; canvas 3675x1092   ← Height tăng (dy≠0)
...
Processing 70_aligned.jpg -> mosaic
  -> Stitched; canvas 3737x1094   ← Width resumed growth
...
Processing 73_aligned.jpg -> mosaic
  -> Stitched; canvas 3908x1094

Final panorama: 3908x1094 pixels
```

**PHÂN TÍCH**:
- Frames 56→64: **dx ≈ 0** → không tăng width
- Frames 57-58: **LoFTR fail hoàn toàn** (< 3 inliers)
- Có thể do:
  1. Periodic pattern gây **ambiguous matches** → median bị bias
  2. Low-texture region → LoFTR cũng không tìm đủ confident matches
  3. **Thiếu RANSAC** → geometric inconsistent matches không bị loại

---

## TỔNG KẾT: TẠI SAO GHÉP TỆ?

### 1. **Median Estimator Vulnerable**
```
LoFTR tìm 1000 matches, nhưng corrugation lặp lại mỗi 5cm:
- 400 matches: dx = -50 (lock đúng stripe)
- 400 matches: dx = -45 (lock stripe bên cạnh)
- 200 matches: dx = -40 (lock 2 stripes xa hơn)

→ Median([-50, -50, ..., -45, -45, ..., -40, ...]) = -45 (SAI!)
→ Đáng lẽ phải là -50

→ Cần RANSAC để loại 600 matches sai trước khi tính median
```

### 2. **Seam Cost Fragile**
```python
cost = np.abs(gray_old - gray_new)
# Với uniform yellow container: gray_old ≈ gray_new ≈ 180
# → cost ≈ 0 everywhere
# → Seam path random, không có ý nghĩa
```

### 3. **Không có Temporal Smoothing**
```
Frame 55: dx = -29
Frame 56: dx = 0    ← Jump 29 pixels!
Frame 57: FAIL

→ Đáng lẽ phải reject jump quá lớn hoặc smooth bằng EMA
```

### 4. **Overlap Verification Missing**
```
Sau khi warp bằng H, KHÔNG kiểm tra:
- Overlap region có chứa keypoints matched không?
- Keypoints trong overlap có thẳng hàng không?

→ Có thể H sai nhưng vẫn ghép → lệch
```

---

## GIẢI PHÁP ĐỀ XUẤT

### Fix 1: RANSAC Geometric Filtering (URGENT!)
```python
# Trong estimate_translation_masked(), sau khi có matches:
if len(good_matches) >= 4:
    pts_src = ...
    pts_tar = ...
    
    # RANSAC estimateAffinePartial2D
    M, inlier_mask = cv2.estimateAffinePartial2D(
        pts_src, pts_tar,
        method=cv2.RANSAC,
        ransacReprojThreshold=12.0,  # Same as current threshold
        confidence=0.99,
        maxIters=2000
    )
    
    if M is not None and inlier_mask.sum() >= 3:
        # Extract RANSAC inliers
        inliers = inlier_mask.ravel().astype(bool)
        deltas_inliers = pts_tar[inliers] - pts_src[inliers]
        
        # Recompute median from RANSAC inliers
        dx = float(np.median(deltas_inliers[:, 0]))
        dy = float(np.median(deltas_inliers[:, 1]))
        
        H = np.array([[1,0,dx],[0,1,dy],[0,0,1]])
        return H
```

### Fix 2: Temporal Gating
```python
# Trong stitch_incremental_with_corners():
prev_dx = 0
max_dx_jump = 50  # pixels

for frame in frames:
    dx, dy = estimate_translation(...)
    
    if abs(dx - prev_dx) > max_dx_jump:
        print(f"Warning: dx jump {prev_dx:.1f} → {dx:.1f}")
        # Clip to max jump
        dx = prev_dx + np.sign(dx - prev_dx) * max_dx_jump
    
    prev_dx = dx
```

### Fix 3: Overlap Keypoint Verification
```python
# Sau khi tính overlap region trong expand_canvas():
# Kiểm tra: keypoints matched có nằm trong overlap không?

# Transform keypoints từ src → tar
kp_src_in_tar = cv2.perspectiveTransform(pts_src, H)

# Kiểm tra bao nhiêu % keypoints nằm trong overlap bbox
in_overlap = (
    (kp_src_in_tar[:, 0] >= ox0) & (kp_src_in_tar[:, 0] < ox1) &
    (kp_src_in_tar[:, 1] >= oy0) & (kp_src_in_tar[:, 1] < oy1)
)

overlap_ratio = in_overlap.sum() / len(kp_src_in_tar)
if overlap_ratio < 0.5:
    print(f"Warning: Only {overlap_ratio:.1%} keypoints in overlap")
    # Có thể reject H này
```

### Fix 4: Better Seam Cost
```python
# Thay vì grayscale absdiff:
# Dùng gradient magnitude difference
gx_old = cv2.Sobel(gray_old, CV2_64F, 1, 0)
gy_old = cv2.Sobel(gray_old, CV2_64F, 0, 1)
grad_mag_old = np.sqrt(gx_old**2 + gy_old**2)

gx_new = cv2.Sobel(gray_new, CV2_64F, 1, 0)
gy_new = cv2.Sobel(gray_new, CV2_64F, 0, 1)
grad_mag_new = np.sqrt(gx_new**2 + gy_new**2)

cost = np.abs(grad_mag_old - grad_mag_new)
```

---

## KẾT LUẬN

**Pipeline đã làm ĐÚNG**:
✓ Mask giới hạn keypoints trong container (SIFT)
✓ LoFTR crop ảnh theo mask (NEW)
✓ Overlap tìm từ warped masks (không phải keypoints)
✓ Seam blending trong overlap

**Pipeline còn THIẾU**:
✗ RANSAC để loại geometric outliers
✗ Temporal smoothing để tránh dx jumps
✗ Overlap verification (keypoints có nằm trong overlap?)
✗ Robust seam cost (gradient thay vì intensity)

**KẾT QUẢ HIỆN TẠI**:
- LoFTR + mask: 3908x1094 panorama
- Stuck ở frames 56-64 (dx≈0)
- 2 frames fail hoàn toàn (57-58)
- Có seams visible do:
  - Median bị bias bởi periodic matches
  - Seam cost không ổn định trên uniform texture

**BƯỚC TIẾP THEO**:
1. Implement RANSAC filtering (FIX #1)
2. Test lại trên MSNU dataset
3. So sánh before/after
4. Nếu cải thiện → apply Fix #2, #3, #4
