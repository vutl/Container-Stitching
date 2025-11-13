import numpy as np
import cv2
import math
import random
from sources.main_app.top.utils.tool_top import arrange_points
from sources.main_app.top.utils.helpper import line_intersection, nms,\
    get_center, distance, cal_angle_2line

ADD_COR = 20
ADD_BIGCOR = 10


def crop_frame(img, corners):
    h, w, _ = img.shape
    
    pts = np.array(corners, dtype=np.float32)

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
    
    warped_image = warped_image[0:h, 10:-10]
    
    return warped_image


def find_fourth_point(corners):
    p1, p2, p3 = corners
    h_center = (p1[1] + p2[1] + p3[1]) // 3
    
    if abs(p1[0] - p2[0]) < 10:
        if p3[1] > h_center:
            return (p3[0], p3[1] - abs(p1[1] - p2[1]))
        else:
            return (p3[0], p3[1] + abs(p1[1] - p2[1]))
        
    elif abs(p1[0]-p3[0]) < 10:
        if p2[1] > h_center:
            return (p2[0], p2[1] - abs(p1[1] - p3[1]))
        else:
            return (p2[0], p2[1] + abs(p1[1] - p3[1]))
    else:
        if p1[1] > h_center:
            return (p1[0], p1[1] - abs(p2[1] - p3[1]))
        else:
            return (p1[0], p1[1] + abs(p2[1] - p3[1]))


def box_to_point(edge_cors, gu_cors, h, w):
    w_center, h_center = w//2, h//2
    edge_cors_output = []
    gu_cors_output = []
    # point for gu container
    for box in gu_cors:
        x1, y1, x2, y2 = box[:4]
        if x1<w_center and y1<h_center:
            gu_cors_output.append([x1-ADD_BIGCOR-10, y1-ADD_BIGCOR])
        elif x1>w_center and y1<h_center:
            gu_cors_output.append([x2+ADD_BIGCOR+10, y1-ADD_BIGCOR])
        elif x1>w_center and y1>h_center:
            gu_cors_output.append([x2+ADD_BIGCOR+10, y2+ADD_BIGCOR])
        else:   
            gu_cors_output.append([x1-ADD_BIGCOR-10, y2+ADD_BIGCOR])
            
    # point for edge container
    for box in edge_cors:
        x1, y1, x2, y2 = box[:4]
        cx, cy = get_center(x1, y1, x2, y2)
        if cy < h_center:
            edge_cors_output.append([cx, cy-ADD_COR+5])
        else:
            edge_cors_output.append([cx, cy+ADD_COR])
    
    return edge_cors_output, gu_cors_output

def filter_edge(edge_cors, w, h):
    right_ecors = [cor for cor in edge_cors if cor[0] > w//2]
    left_ecors = [cor for cor in edge_cors if cor[0] < w//2]
    
    def del_noise_cors(cors):
        r'''
        Activate func when number of edge corners is odd
        delete noise corners that lower confidence than other corners
        '''
        if len(cors) % 2 == 1:
            cors = sorted(cors, key=lambda x: x[4])
            return cors[1:]
        else:
            return cors
        
    right_ecors = del_noise_cors(right_ecors)
    left_ecors = del_noise_cors(left_ecors)    
        
    return right_ecors+left_ecors

def filter_gucors(gu_cors, w, h):
    gu_cors_output = []
    max_x_gucor = max(gu_cors, key=lambda x: x[0])[0]
    min_x_gucor = min(gu_cors, key=lambda x: x[0])[0]
    
    top_gucors = [cor for cor in gu_cors if cor[1] < h//2]
    bot_gucors = [cor for cor in gu_cors if cor[1] > h//2]
    top_gucors = sorted(top_gucors, key=lambda x: x[0])
    bot_gucors = sorted(bot_gucors, key=lambda x: x[0])
    
    if len(top_gucors) == 2:
        if max_x_gucor < w//2:
            gu_cors_output.append(top_gucors[1])
        else:
            gu_cors_output.append(top_gucors[0])
    else:
        gu_cors_output.append(top_gucors[0])
    
    if len(bot_gucors) == 2:
        if min_x_gucor < w//2:
            gu_cors_output.append(bot_gucors[1])
        else:
            gu_cors_output.append(bot_gucors[0])
    else:
        gu_cors_output.append(bot_gucors[0])
    
    return gu_cors_output


def final_4cors(edge_pts, gu_pts, h, w):
    # Return 4 corners of container that is nearest to edge of image from 3 list points
    final_cors = []
    ls_all_pts = edge_pts + gu_pts
    if len(ls_all_pts) == 3:
        ls_all_pts.append(find_fourth_point(ls_all_pts))
    
    ls_top_pts = [cor for cor in ls_all_pts if cor[1] < h//2]
    ls_bot_pts = [cor for cor in ls_all_pts if cor[1] > h//2]
    
    if len(ls_top_pts) == 0 or len(ls_bot_pts) == 0:
        return []
    
    point0 = min(ls_top_pts, key=lambda x: x[0])
    point1 = max(ls_top_pts, key=lambda x: x[0])
    point2 = max(ls_bot_pts, key=lambda x: x[0])
    point3 = min(ls_bot_pts, key=lambda x: x[0])
    
    final_cors.append(point0)
    final_cors.append(point1)
    final_cors.append(point2)
    final_cors.append(point3)
    
    return final_cors


def check_2cont(gu_pts):
    r'''
    This function is to check if the container exists 2 containers:
    - return True if: + distance between 2 gu points is near enough
                      + more than 2 gu points
    - else return False
    '''
    if len(gu_pts) > 2:
        return True
    elif len(gu_pts) == 2:
        return distance(gu_pts[0], gu_pts[1]) < 300 
    else:
        return False
    
    
def pass_angle_condition(top_pts, bot_pts, pt1, pt2, angle_limit):
    r'''
    Find angle between line and 4 corner points
    - return True if angle between line and 4 corner points is higher than angle_limit
    - else return False
    '''
    angle1 = cal_angle_2line((pt1, pt2), (top_pts[0], top_pts[1]))
    # print('pt1, pt2', pt1, pt2)
    # print('top_pts', top_pts[0], top_pts[1])
    # print('angle1', angle1)
    angle2 = cal_angle_2line((pt1, pt2), (bot_pts[0], bot_pts[1]))
    
    # print('angle2', angle2)
    
    return abs(angle1) > angle_limit and abs(angle2) > angle_limit   

def cal_angle(top_pts, bot_pts, pt1, pt2):
    angle1 = cal_angle_2line((pt1, pt2), (top_pts[0], top_pts[1]))
    angle2 = cal_angle_2line((pt1, pt2), (bot_pts[0], bot_pts[1]))
    
    return (angle1 + angle2)/2

def filter_angle(lines, angles):
    """
    This function filters lines based on their angles.
    Find densest range of angles and keep lines within that range.
    """
    counts, bin_edges = np.histogram(angles, bins=20)
    max_count_index = np.argmax(counts)
    densest_angle_range = (bin_edges[max_count_index], bin_edges[max_count_index + 1])
    # print('densest_angle_range', densest_angle_range)
    return [line for line, angle in zip(lines, angles) if angle >= densest_angle_range[0]]   
    
    
def correct_corners(img, corners, gu_cors, h, w):
    r'''
    Corect the angle coordinates to avoid the vehicle going diagonally
    Use image processing to find the line near the corner
    and correct the corner position to the line
        AB : Line on top that contain 2 point corner
        CD : Line on bottom that contain 2 point corner
    '''
    final_cors = []
    corners = sorted(corners, key=lambda x: x[0])
    gu_cors = sorted(gu_cors, key=lambda x: x[0])
    top_pts = [cor for cor in corners if cor[1] < h//2]
    bot_pts = [cor for cor in corners if cor[1] > h//2]
    # calculate diagonal length
    diag_lenght = max(abs(top_pts[1][1]-top_pts[0][1]), abs(bot_pts[1][1]-bot_pts[0][1]))
    
    # get limit coordinates 
    right_limit = top_pts[-1][0]
    left_limit = top_pts[0][0]
    top_limit = top_pts[0][1]
    bot_limit = bot_pts[0][1]
    
    # init flag left and right gu_cors
    flag_left_gu, flag_right_gu = False, False
    if len(gu_cors) == 2:
        if gu_cors[0][0] < w//3:
            flag_left_gu = True
        if gu_cors[-1][2] > w//3*2:
            flag_right_gu = True
    
    # get limmit coordinates from gu points
    if len(gu_cors):
        left_gu_limit = max(gu_cors[0][0]-50, 50)
        right_gu_limit = min(gu_cors[-1][2]+50, w-50)
    else:
        left_gu_limit, right_gu_limit = 0, 0
    
    img_copy = img.copy()
    img_half = cv2.resize(img_copy, (img.shape[1]//2, img.shape[0]//2)) # reduce time for houghlinesp
    gray = cv2.cvtColor(img_half, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 55, minLineLength=80, maxLineGap=20)
    
    ls_lines, ls_angles = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x1, y1, x2, y2 = int(x1*2), int(y1*2), int(x2*2), int(y2*2)
        if abs(x2-x1) < 50 + diag_lenght//2 : # get vertical line
            if ((x1>max(55, 20+diag_lenght) and x1<200+diag_lenght) or (x2<w-diag_lenght-20 and x2>w-200-diag_lenght)) \
                and (y1>top_limit+50 and y1<bot_limit-50) \
                    and (y2>top_limit+50 and y2<bot_limit-50) \
                        and (x1>left_limit and x2<right_limit)\
                            and not (x1>left_gu_limit and x2<right_gu_limit):
                      
                angle = cal_angle(top_pts, bot_pts, (x1, y1), (x2, y2))
                if angle < 85:
                    continue       
                ls_lines.append((x1, y1, x2, y2))
                ls_angles.append(angle)
                         
    if not len(ls_lines) : # return if not found line
        corners = arrange_points(corners)
        return adjust_corners(corners, h, w)
    
    ls_lines = filter_angle(ls_lines, ls_angles)       
    ls_lines.sort(key=lambda x: x[0])
    #------------------------------------------------------------------#
    # Let's correct corners
    # correct left top(+bot) corner
    if ls_lines[0][0] < w//2 and top_pts[0][0] < 60 and not flag_left_gu:
        final_cors.append(line_intersection(
            A=(ls_lines[0][0], ls_lines[0][1]),
            B=(ls_lines[0][2], ls_lines[0][3]),
            C=(top_pts[0][0], top_pts[0][1]),
            D=(top_pts[1][0], top_pts[1][1])
        ))
        
        final_cors.append(line_intersection(
            A=(ls_lines[0][0], ls_lines[0][1]),
            B=(ls_lines[0][2], ls_lines[0][3]),
            C=(bot_pts[0][0], bot_pts[0][1]),
            D=(bot_pts[1][0], bot_pts[1][1])
        ))
    else:
        final_cors.append(top_pts[0])
        final_cors.append(bot_pts[0])
        
    # correct right top(+bot) corner
    if ls_lines[-1][0] > w//2 and not flag_right_gu:
        final_cors.append(line_intersection(
            A=(ls_lines[-1][0], ls_lines[-1][1]),
            B=(ls_lines[-1][2], ls_lines[-1][3]),
            C=(top_pts[0][0], top_pts[0][1]),
            D=(top_pts[1][0], top_pts[1][1])
        ))
        
        final_cors.append(line_intersection(
            A=(ls_lines[-1][0], ls_lines[-1][1]),
            B=(ls_lines[-1][2], ls_lines[-1][3]),
            C=(bot_pts[0][0], bot_pts[0][1]),
            D=(bot_pts[1][0], bot_pts[1][1])
        ))
    else:
        final_cors.append(top_pts[1])
        final_cors.append(bot_pts[1])

    return arrange_points(final_cors)
    
    
def get_results(img, results):
    h, w = img.shape[:2]
    
    check_pass = False
    edge_cors = []
    gu_cors = []
    for r in results[0].boxes:
        box = r.xyxy[0]
        x1, y1, x2, y2 = map(int, box)
        conf = r.conf[0]
        
        if max(x2-x1, y2-y1) < 48:
            edge_cors.append([x1, y1, x2, y2, conf])
        else:       
            gu_cors.append([x1, y1, x2, y2, conf]) if conf > 0.5 else None
            # pass this frame cause gu_cors is too near to edge_cors
            if x1<55 or x2>w-55:
                check_pass = True

    edge_cors = nms(edge_cors, 0.3)
    edge_cors = filter_edge(edge_cors, h, w)
    gu_cors = nms(gu_cors, 0.3)
    
    edge_pts, gu_pts = box_to_point(edge_cors, gu_cors, h, w)
    # filter corners for >2 gu corners
    if len(gu_cors) > 2:
        gu_cors = filter_gucors(gu_cors, w, h)
        
    # find 4 points from 3 list points that nearest to edge of image
    corners = final_4cors(edge_pts, gu_pts, h, w)
    
    corners = correct_corners(img, corners, gu_cors, h, w) if len(corners) == 4 else corners
    
    return corners, check_pass


def adjust_corners(corners, h, w):
    pts = np.array(corners, dtype=np.float32)

    height01, width01 = abs(pts[0][1]-pts[1][1]), pts[1][0]-pts[0][0]
    height32, width32 = abs(pts[3][1]-pts[2][1]), pts[2][0]-pts[3][0]
    height03, height12 = pts[3][1]-pts[0][1], pts[2][1]-pts[1][1]

    c23 = math.sqrt((pts[2][0]-pts[3][0])**2+(pts[2][1]-pts[3][1])**2)
    c01 = math.sqrt((pts[0][0]-pts[1][0])**2+(pts[0][1]-pts[1][1])**2)
    
    height32 = max(height32, 0.01)
    height01 = max(height01, 0.01)

    width32 = max(width32, 0.01)
    width01 = max(width01, 0.01)

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
            
            
    # expand first and last point
    if pts[0][0] > 80 and pts[3][0]>80:
        pts[0][0]-=25
        pts[3][0]-=25
        if pts[0][1] > pts[1][1]:
            pts[0][1] += height01*50/width01
        else:
            pts[0][1] -= height01*50/width01

        if pts[3][1] > pts[2][1]:
            pts[3][1] += height32*50/width32
        else:
            pts[3][1] -= height32*50/width32
            
    if pts[1][0] < w-80 and pts[2][0]<w-80:
        pts[1][0]+=25
        pts[2][0]+=25
        if pts[0][1] > pts[1][1]:
            pts[1][1] -= height01*50/width01
        else:
            pts[1][1] += height01*50/width01

        if pts[3][1] > pts[2][1]:
            pts[2][1] -= height32*50/width32
        else:
            pts[2][1] += height32*50/width32

    return pts
