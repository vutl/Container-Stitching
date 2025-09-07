import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_img(img, size=(15, 8), title=""):
    plt.figure(figsize=size)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def crop_container_region(img):
    """Cắt vùng container để tránh lấy keypoint từ background"""
    h, w = img.shape[:2]
    # Cắt bỏ viền để chỉ focus vào container
    crop_top = int(h * 0.2)
    crop_bottom = int(h * 0.8)  
    crop_left = int(w * 0.15)
    crop_right = int(w * 0.85)
    
    return img[crop_top:crop_bottom, crop_left:crop_right], (crop_left, crop_top)

def stitch_container_images():
    # Đường dẫn ảnh
    base_dir = Path("D:/Documents/Container Snitching/mnt/2025-09-04_14-35-53")
    
    # Load 3 ảnh đầu tiên
    img1 = cv2.imread(str(base_dir / "43_cropped.jpg"))
    img2 = cv2.imread(str(base_dir / "44_cropped.jpg"))
    img3 = cv2.imread(str(base_dir / "45_cropped.jpg"))
    
    print("Loaded images 43, 44, 45")
    
    # Ghép img1 và img2 trước
    print("Stitching 43 + 44...")
    result_12 = stitch_two_images(img1, img2)
    
    if result_12 is not None:
        print("Success! Now stitching with 45...")
        # Ghép kết quả với img3
        final_result = stitch_two_images(result_12, img3)
        
        if final_result is not None:
            # Lưu kết quả
            cv2.imwrite("D:/Documents/Container Snitching/container_stitched_43_44_45.jpg", final_result)
            print("Saved final result!")
            
            # Hiển thị
            plot_img(final_result, title="Container Stitched: 43 + 44 + 45")
            return final_result
        else:
            print("Failed to stitch with image 45")
            # Ít nhất hiển thị kết quả 43+44
            plot_img(result_12, title="Container Stitched: 43 + 44")
            return result_12
    else:
        print("Failed to stitch images 43 and 44")
        return None

def stitch_two_images(src_img, tar_img):
    """Ghép 2 ảnh theo phương pháp trong cv.py"""
    
    # Crop vùng container để tránh noise từ background
    src_crop, src_offset = crop_container_region(src_img)
    tar_crop, tar_offset = crop_container_region(tar_img)
    
    # Convert to grayscale
    src_gray = cv2.cvtColor(src_crop, cv2.COLOR_BGR2GRAY)
    tar_gray = cv2.cvtColor(tar_crop, cv2.COLOR_BGR2GRAY)
    
    # SIFT detector
    try:
        sift_detector = cv2.SIFT_create()
    except AttributeError:
        print("SIFT not available, trying ORB...")
        sift_detector = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints
    kp1, des1 = sift_detector.detectAndCompute(src_gray, None)
    kp2, des2 = sift_detector.detectAndCompute(tar_gray, None)
    
    if des1 is None or des2 is None:
        print("No features detected!")
        return None
    
    print(f"Detected {len(kp1)} and {len(kp2)} keypoints")
    
    # Match keypoints
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(des1, des2, 2)
    
    matches = []
    ratio = 0.75
    
    for match_pair in raw_matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < n.distance * ratio:
                matches.append(m)
    
    if len(matches) < 10:
        print(f"Not enough matches: {len(matches)}")
        return None
        
    print(f"Found {len(matches)} good matches")
    
    # Chọn top matches
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:min(200, len(matches))]
    
    # Điều chỉnh keypoints về tọa độ ảnh gốc
    kp1_adjusted = []
    kp2_adjusted = []
    
    for kp in kp1:
        new_kp = cv2.KeyPoint(kp.pt[0] + src_offset[0], kp.pt[1] + src_offset[1], kp.size)
        kp1_adjusted.append(new_kp)
    
    for kp in kp2:
        new_kp = cv2.KeyPoint(kp.pt[0] + tar_offset[0], kp.pt[1] + tar_offset[1], kp.size)
        kp2_adjusted.append(new_kp)
    
    # Extract matching points - theo cách cv.py
    kp1_pts = np.float32([kp.pt for kp in kp1_adjusted])
    kp2_pts = np.float32([kp.pt for kp in kp2_adjusted])
    pts1 = np.float32([kp1_pts[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2_pts[m.trainIdx] for m in matches])
    
    # Estimate homography
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    
    if H is None:
        print("Could not compute homography")
        return None
    
    print("Homography matrix:")
    print(H)
    
    # Stitch images - theo cách cv.py nhưng cải tiến
    h1, w1 = src_img.shape[:2]
    h2, w2 = tar_img.shape[:2]
    
    # Tính toán canvas size thông minh hơn
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners1_transformed = cv2.perspectiveTransform(corners1, H)
    
    # Tìm bounding box
    all_corners = np.concatenate((
        np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2),
        corners1_transformed
    ), axis=0)
    
    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())
    
    # Điều chỉnh homography để có tọa độ dương
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    h_adjusted = translation @ H
    
    # Warp source image
    result_width = x_max - x_min
    result_height = y_max - y_min
    
    result = cv2.warpPerspective(src_img, h_adjusted, (result_width, result_height))
    
    # Paste target image
    y_offset = -y_min
    x_offset = -x_min
    
    if (y_offset >= 0 and x_offset >= 0 and 
        y_offset + h2 <= result_height and x_offset + w2 <= result_width):
        result[y_offset:y_offset+h2, x_offset:x_offset+w2] = tar_img
    
    return result

if __name__ == "__main__":
    print("Starting container image stitching...")
    result = stitch_container_images()
    if result is not None:
        print("Stitching completed successfully!")
    else:
        print("Stitching failed!")
