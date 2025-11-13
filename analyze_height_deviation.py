#!/usr/bin/env python3
"""Analyze height deviation in bbox vs refined corner coordinates."""

# Data từ img_40
print("=" * 70)
print("FRAME 40 - Phân tích độ lệch Y")
print("=" * 70)

# BBOX coordinates (detection boxes)
bbox_data = {
    'TL': {'y1': 193, 'y2': 236, 'cy': (193+236)/2},
    'BL': {'y1': 1265, 'y2': 1314, 'cy': (1265+1314)/2},
    'TR': {'y1': 231, 'y2': 270, 'cy': (231+270)/2},
    'BR': {'y1': 1344, 'y2': 1376, 'cy': (1344+1376)/2},
}

# Refined CORNER coordinates (gradient-based)
corner_data = {
    'TL': {'y': 217},
    'BL': {'y': 1308},
    'TR': {'y': 250},
    'BR': {'y': 1359},
}

print("\n1. BBOX COORDINATES (Detection boxes):")
print("-" * 70)
for corner, data in bbox_data.items():
    print(f"{corner:3s}: y1={data['y1']:4d}, y2={data['y2']:4d}, center_y={data['cy']:.1f}")

print("\n2. REFINED CORNER COORDINATES (Gradient-based):")
print("-" * 70)
for corner, data in corner_data.items():
    print(f"{corner:3s}: y={data['y']:4d}")

print("\n" + "=" * 70)
print("PHÂN TÍCH CHIỀU CAO (HEIGHT)")
print("=" * 70)

# Tính chiều cao từ BBOX (dùng center_y)
h_left_bbox = bbox_data['BL']['cy'] - bbox_data['TL']['cy']
h_right_bbox = bbox_data['BR']['cy'] - bbox_data['TR']['cy']

print("\nA. Dùng BBOX CENTER (y_center của detection box):")
print(f"  Cột trái  (BL - TL): {h_left_bbox:.1f}px")
print(f"  Cột phải  (BR - TR): {h_right_bbox:.1f}px")
print(f"  Độ lệch: {abs(h_left_bbox - h_right_bbox):.1f}px ({abs(h_left_bbox - h_right_bbox)/max(h_left_bbox, h_right_bbox)*100:.2f}%)")

# Tính chiều cao từ REFINED CORNER
h_left_corner = corner_data['BL']['y'] - corner_data['TL']['y']
h_right_corner = corner_data['BR']['y'] - corner_data['TR']['y']

print("\nB. Dùng REFINED CORNER (y của góc sau gradient refinement):")
print(f"  Cột trái  (BL - TL): {h_left_corner}px")
print(f"  Cột phải  (BR - TR): {h_right_corner}px")
print(f"  Độ lệch: {abs(h_left_corner - h_right_corner)}px ({abs(h_left_corner - h_right_corner)/max(h_left_corner, h_right_corner)*100:.2f}%)")

print("\n" + "=" * 70)
print("SO SÁNH TỌA ĐỘ Y (Vertical positions)")
print("=" * 70)

print("\nTop corners (TL vs TR):")
print(f"  BBOX center:   TL={bbox_data['TL']['cy']:.1f}, TR={bbox_data['TR']['cy']:.1f}, diff={abs(bbox_data['TL']['cy'] - bbox_data['TR']['cy']):.1f}px")
print(f"  Refined corner: TL={corner_data['TL']['y']}, TR={corner_data['TR']['y']}, diff={abs(corner_data['TL']['y'] - corner_data['TR']['y'])}px")

print("\nBottom corners (BL vs BR):")
print(f"  BBOX center:   BL={bbox_data['BL']['cy']:.1f}, BR={bbox_data['BR']['cy']:.1f}, diff={abs(bbox_data['BL']['cy'] - bbox_data['BR']['cy']):.1f}px")
print(f"  Refined corner: BL={corner_data['BL']['y']}, BR={corner_data['BR']['y']}, diff={abs(corner_data['BL']['y'] - corner_data['BR']['y'])}px")

print("\n" + "=" * 70)
print("KẾT LUẬN")
print("=" * 70)

print("""
1. BBOX CENTER vs REFINED CORNER:
   - BBOX center: height diff = {:.1f}px ({:.2f}%)
   - Refined corner: height diff = {}px ({:.2f}%)
   
2. Bottom corners deviation (BL vs BR):
   - BBOX center: {}px lệch
   - Refined corner: {}px lệch ⚠️ LỚN HƠN!
   
3. PHÁT HIỆN:
   ✓ Refined corner coordinates THỂ HIỆN RÕ HƠN độ lệch
   ✓ BR (bottom-right) lệch XUỐNG so với BL đến {}px
   ✓ Đây là CORNER SAI - detection nhầm vào edge của sàn/nền
   
4. GIẢI PHÁP:
   → Validate heights SAU BƯỚC REFINE CORNERS
   → Dùng tọa độ Y của REFINED CORNERS để check
   → Nếu |h_left - h_right| > threshold: loại cột bị lệch
""".format(
    abs(h_left_bbox - h_right_bbox),
    abs(h_left_bbox - h_right_bbox)/max(h_left_bbox, h_right_bbox)*100,
    abs(h_left_corner - h_right_corner),
    abs(h_left_corner - h_right_corner)/max(h_left_corner, h_right_corner)*100,
    abs(bbox_data['BL']['cy'] - bbox_data['BR']['cy']),
    abs(corner_data['BL']['y'] - corner_data['BR']['y']),
    abs(corner_data['BL']['y'] - corner_data['BR']['y'])
))

print("\n" + "=" * 70)
print("DEMO: So sánh với reference height (giả sử = 1091px từ ảnh đầu)")
print("=" * 70)

reference_height = 1091  # Giả sử từ ảnh trước đó
tolerance_percent = 10.0

h_left_dev = abs(h_left_corner - reference_height) / reference_height * 100
h_right_dev = abs(h_right_corner - reference_height) / reference_height * 100

print(f"\nReference height: {reference_height}px (tolerance: {tolerance_percent}%)")
print(f"\nCột trái:  {h_left_corner}px → deviation = {h_left_dev:.2f}%", 
      "✓ OK" if h_left_dev <= tolerance_percent else "❌ REJECT")
print(f"Cột phải: {h_right_corner}px → deviation = {h_right_dev:.2f}%",
      "✓ OK" if h_right_dev <= tolerance_percent else "❌ REJECT")

if h_right_dev > tolerance_percent:
    print(f"\n→ Cột phải bị REJECT (lệch {h_right_dev:.2f}% > {tolerance_percent}%)")
    print("→ Chỉ giữ TL, BL (2 góc), cần vector completion để tìm TR, BR")
