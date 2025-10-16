import numpy as np
import cv2


def get_center(x1, y1, x2, y2):
    return (x1 + x2) // 2, (y1 + y2) // 2


def get_center_top(x1, y1, x2, y2, imgsize):
    x_c, y_c = imgsize[1]/2, imgsize[0]/2
    w,h = x2-x1, y2-y1
    if x2-x1 < 50 and y2-y1 < 50:
        return (x1 + x2) / 2, (y1 + y2) / 2
    else:
        if x1<x_c and y1<y_c:
            return x1+w//8, y1+h//8
        elif x1>x_c and y1<y_c:
            return x2-w//8, y1+h//8
        elif x1<x_c and y1>y_c:
            return x1+w//8, y2-h//8
        else:
            return x2-w//8, y2-h//8


def distance(point1, point2):
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def xyxy_to_axb(x1, y1, x2, y2): # convert to ax+b
    if x1 == x2:
        x1 += 1e-5
        
    a = (y2-y1)/(x2-x1)
    b = y1 - a*x1
    return a, b

def line_intersection(A, B, C, D):
    r"""
    A, B, C, D: tuple (x, y) - coordinates of points
    Return intersection coordinates (x, y) n
    """
    x1, y1 = A[:2]
    x2, y2 = B[:2]
    x3, y3 = C[:2]
    x4, y4 = D[:2]

    a1, b1 = xyxy_to_axb(x1, y1, x2, y2)
    a2, b2 = xyxy_to_axb(x3, y3, x4, y4)
    
    if a1 == a2:
        a1+= 1e-5
        
    x_inter = (b2 - b1) / (a1 - a2)
    y_inter = a1*x_inter + b1
    return (int(x_inter), int(y_inter))

def cal_angle_2line(line1, line2):
    A, B = line1
    C, D = line2
    # cal angle between two lines :  AB and CD
    a1, b1 = xyxy_to_axb(A[0], A[1], B[0], B[1])
    a2, b2 = xyxy_to_axb(C[0], C[1], D[0], D[1])

    if 1+a1*a2 == 0:
        angle = 0
    else:
        angle = np.arctan(abs((a2-a1)/(1+a1*a2)))

    return angle * 180 / np.pi 


def convert_cortop(pts, img):
    pts = np.array(pts, dtype=np.float32)
    height, width = img.shape[:2]
    w01 = abs(pts[0][0]-pts[1][0])
    h01 = abs(pts[0][1]-pts[1][1])

    w32 = abs(pts[3][0]-pts[2][0])
    h32 = abs(pts[3][1]-pts[2][1])

    if pts[0][1]>pts[1][1]:
        if pts[0][0]<100:
            pts[0][1] += h01*(100-pts[0][0])/w01
            pts[0][0] = 100
        if pts[1][0]>width-100:
            pts[1][1] -= h01*(pts[1][0]-width+100)/w01
            pts[1][0] = width-100
    else:
        if pts[0][0]<100:
            pts[0][1] -= h01*(100-pts[0][0])/w01
            pts[0][0] = 100
        if pts[1][0]>width-100:
            pts[1][1] += h01*(pts[1][0]-width+100)/w01
            pts[1][0] = width-100

    if pts[3][1]>pts[2][1]:
        if pts[3][0]<100:
            pts[3][1] += h32*(100-pts[3][0])/w32
            pts[3][0] = 100
        if pts[2][0]>width-100:
            pts[2][1] -= h32*(pts[2][0]-width+100)/w32
            pts[2][0] = width-100
    else:
        if pts[3][0]<100:
            pts[3][1] -= h32*(100-pts[3][0])/w32
            pts[3][0] = 100
        if pts[2][0]>width-100:
            pts[2][1] += h32*(pts[2][0]-width+100)/w32
            pts[2][0] = width-100


    img = img[:, 100:width-100]
   
    return pts, img


def iou(box1, box2):
    # box format [x1, y1, x2, y2, confidence]
    x1_box1, y1_box1, x2_box1, y2_box1 = box1[:4]
    x1_box2, y1_box2, x2_box2, y2_box2 = box2[:4]
    
    # calculate th region between box1 and box2
    x1 = max(x1_box1, x1_box2)
    y1 = max(y1_box1, y1_box2)
    x2 = min(x2_box1, x2_box2)
    y2 = min(y2_box1, y2_box2)
    
    # calculate the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # calculate the area of union rectangle
    union_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1) + (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1) - inter_area
    
    return inter_area / union_area
  
    
def nms(boxes, iou_threshold):
    '''
    boxe: [x1, y1, x2, y2, confidence]
    iou_threshold: threshold of iou to remove the box and keep higher confidence box
    return boxes after nms
    '''
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep_boxes = []
    
    for box in boxes:
        if len(keep_boxes) == 0:
            keep_boxes.append(box)
        else:
            for keep_box in keep_boxes:
                if iou(box, keep_box) > iou_threshold:
                    break
            else:
                keep_boxes.append(box)
        
    return keep_boxes


def stit_2img(img1, img2, left_pts, right_pts):
    """
    Nối hai ảnh dựa trên các cặp điểm src_pts và dst_pts.
    """
    width, height = img1.shape[1], img1.shape[0]
    
    for left_pt, right_pt in zip(left_pts, right_pts):

        img1_cropped = img1[0:height, int(left_pt[0]):width]
        img2_cropped = img2[0:height, 0:int(right_pt[0])]
        
        # Nối hai ảnh lại
        return cv2.hconcat((img2_cropped, img1_cropped))


def concate_2image(img1, img2, point1, point2):
    # Nối hai ảnh dựa trên điểm tốt nhất
    return stit_2img(img1, img2, [point1], [point2])


def stitched_status(len_ls_0, len_ls_1, cont_size=42):
    if cont_size == 22:
        if len_ls_0 == 5 and len_ls_1 == 0:
            return True
        return False
    elif cont_size == 12:
        if len_ls_0 == 5 and len_ls_1 == 5:
            return True
        return False
    elif cont_size == 45:
        if len_ls_0 == 13 and len_ls_1 == 0:
            return True
        return False
    else:
        if len_ls_0 == 11 and len_ls_1 == 0:
            return True
        return False
        
    