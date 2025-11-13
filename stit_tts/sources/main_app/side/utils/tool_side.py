import numpy as np
import cv2
import math
from sources.utils.tools import nms


def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def crop_frame(img, corners):
    corners = adjust_corners(corners, img)
    
    h, w, _ = img.shape
    
    pts = np.array(corners, dtype=np.float32)
    
    # Expand corners
    # pts[0][1], pts[1][1] = max(pts[0][1] - 5, 0), max(pts[1][1] - 5, 0)  # expand 5px
    # pts[0][0], pts[1][0] = pts[0][0] - 45, pts[1][0] + 60
    
    # pts[3][0], pts[2][0] = pts[3][0] - 45, pts[2][0] + 60
    pts[2][1], pts[3][1] = min(pts[2][1] + 5, h), min(pts[3][1] + 5, h) # expand 10px

    # Define the width and height of the new image
    width = max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3]))
    height = max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0]))

    # Define the destination points
    dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(pts, dst)

    # Apply the perspective transformation
    warped_image = cv2.warpPerspective(img, M, (int(width), int(height)))
    h,w = warped_image.shape[:2]
    
    warped_image = warped_image[:, 20:w-20]
    
    return warped_image


def find_fourth_point(corners):
    d1 = distance(corners[0], corners[1])
    d2 = distance(corners[1], corners[2])
    d3 = distance(corners[2], corners[0])

    if d1 > d2 and d1 > d3:
        x_cen, y_cen = get_center(
            corners[0][0], corners[0][1], corners[1][0], corners[1][1])
        return 2*x_cen - corners[2][0], 2*y_cen - corners[2][1]

    elif d2 > d1 and d2 > d3:
        x_cen, y_cen = get_center(
            corners[1][0], corners[1][1], corners[2][0], corners[2][1])
        return 2*x_cen - corners[0][0], 2*y_cen - corners[0][1]

    else:
        x_cen, y_cen = get_center(
            corners[2][0], corners[2][1], corners[0][0], corners[0][1])
        return 2*x_cen - corners[1][0], 2*y_cen - corners[1][1]


def arrange_points(points):
    points = np.array(points)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)

    arranged_points = np.zeros((4, 2), dtype="float32")

    arranged_points[0] = points[np.argmin(s)]
    arranged_points[2] = points[np.argmax(s)]

    arranged_points[1] = points[np.argmin(diff)]
    arranged_points[3] = points[np.argmax(diff)]

    return list(arranged_points)


def get_results(img, results):
    corners = []
    for r in results[0].boxes:
        box = r.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        conf = r.conf[0]
        corners.append([x1, y1, x2, y2, conf])
  
    corners = nms(corners, 0.3)
    # get center
    corners = [get_center(x1, y1, x2, y2) for x1, y1, x2, y2, _ in corners]
    # num_corner = len(corners)
    corners = process_corners(corners)

    return corners, img


def process_corners(corners):

    if len(corners) == 3:
        x3, y3 = find_fourth_point(corners)
        corners.append([x3, y3])
        corners = arrange_points(corners)

    if len(corners) >= 4:
        corners = arrange_points(corners)

    return corners


def adjust_corners(corners, img):
    pts = np.array(corners, dtype=np.float32)

    h, w = img.shape[:2]

    height01, width01 = abs(pts[0][1]-pts[1][1]), pts[1][0]-pts[0][0]
    height32, width32 = abs(pts[3][1]-pts[2][1]), pts[2][0]-pts[3][0]
    height03, height12 = pts[3][1]-pts[0][1], pts[2][1]-pts[1][1]

    c23 = math.sqrt((pts[2][0]-pts[3][0])**2+(pts[2][1]-pts[3][1])**2)
    c01 = math.sqrt((pts[0][0]-pts[1][0])**2+(pts[0][1]-pts[1][1])**2)
    
    if height01 == 0: 
        height01 = 0.01
    if height32 == 0:
        height32 = 0.01

    if pts[0][1] > pts[1][1]:
        # calculate for pts[3]
        if pts[3][0] < 60:
            c_tar3 = height32*height03/width32
            k_scale3 = c_tar3/c23*height01/height32

            pts[3][0] += k_scale3*width32
            pts[3][1] -= k_scale3*height32
    else:
        # calculate for pts[2]
        if pts[2][0] > w-60:
            c_tar2 = height32*height12/width32
            k_scale2 = c_tar2/c23*height01/height32

            pts[2][0] -= k_scale2*width32
            pts[2][1] -= k_scale2*height32

    if pts[3][1] > pts[2][1]:
        # calculate for pts[1]
        if pts[1][0] > w-60:
            c_tar1 = height01*height12/width01
            k_scale1 = c_tar1/c01*height32/height01

            pts[1][0] -= k_scale1*width01
            pts[1][1] += k_scale1*height01
    else:
        # calculate for pts[0]
        if pts[0][0] < 60:
            c_tar0 = height01*height03/width01
            k_scale0 = c_tar0/c01*height32/height01

            pts[0][0] += k_scale0*width01
            pts[0][1] += k_scale0*height01
            
            
    #expand first and last point
    if pts[0][0] > 80 and pts[3][0]>80:
        pts[0][0]-=15
        pts[3][0]-=15
        if pts[0][1] > pts[1][1]:
            pts[0][1] += height01*50/width01
        else:
            pts[0][1] -= height01*50/width01

        if pts[3][1] > pts[2][1]:
            pts[3][1] += height32*50/width32
        else:
            pts[3][1] -= height32*50/width32
            
    if pts[1][0] < w-80 and pts[2][0]<w-80:
        pts[1][0]+=15
        pts[2][0]+=15
        if pts[0][1] > pts[1][1]:
            pts[1][1] -= height01*50/width01
        else:
            pts[1][1] += height01*50/width01

        if pts[3][1] > pts[2][1]:
            pts[2][1] -= height32*50/width32
        else:
            pts[2][1] += height32*50/width32

    return pts


def undistort_image(image, mtx, dist):
    return cv2.undistort(image, mtx, dist, None, mtx)


def rotateimg(img, angle, scale, center):
    rotation_mat = cv2.getRotationMatrix2D(center, angle, scale)
    rotate_img = cv2.warpAffine(img, rotation_mat, center)

    return rotate_img
