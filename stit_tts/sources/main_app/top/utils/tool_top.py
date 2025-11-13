import numpy as np
import cv2
from typing import List


# expand to increase view of crop image
ADD_COR = 20 # 15 pixel
ADD_BIGCOR = 10 # 10 pixel

def update_height_crop(height, big_corners: List, longlines: List) -> int:
    # determine min y of big corners
    y_top_min = min(big_corners[0][1], big_corners[1][1], longlines[0][1]) 
    # determine max y of big corners
    y_bottom_max = max(big_corners[0][3], big_corners[1][3], longlines[0][3]) 
    
    k_scale = 0.6 # percentage of height crop
    h_above = y_top_min*k_scale 
    h_under = (height-y_bottom_max)*k_scale
    
    # ⚠️check if h_above or h_under is too small => don't crop 
    if h_above < 100:
        h_above = 0
    if h_under < 100:
        h_under = 0
    
    return int(h_above), int(h_under)

def update_width_crop(width, big_corners: List, longlines: List) -> int:
    # target width crop will be 65% width of image
    x_right_max = max(big_corners[0][0], big_corners[1][2]) # determine max x of big corners
    x_right_min = min(longlines[0][0], longlines[0][2])
    w_container = x_right_max - x_right_min
    k_scale = 0.8 # percentage of width after crop
    return int(max(1, width-w_container/k_scale))
        

def clean_noise_lines(big_corners, longlines):
    # this function clean noise lines that are too close to each other
    ll_offical = []
    pixel_limit = 300
    for ll in longlines:
        delete = False
        for bcor in big_corners:
            if abs(ll[0]-bcor[0]) < pixel_limit:
                print('delete')
                delete = True
                break
        
        if not delete:
            ll_offical.append(ll)
                
    return ll_offical


def process2bigcors(big_corners, img_shape, longlines):
    r'''
    This function is used to process big corners when number of big corners is more than 2
    Return 2 big corners that are closest to longline
    '''
    h, w, _ = img_shape
    if len(big_corners) <3 or len(longlines) == 0:
        return big_corners
    
    top_bigcor = [cor for cor in big_corners if cor[1] < h//2]
    bottom_bigcor = [cor for cor in big_corners if cor[3] > h//2]
    top_bigcor = sorted(top_bigcor, key=lambda x: x[0])
    bottom_bigcor = sorted(bottom_bigcor, key=lambda x: x[0])
    
    if longlines[0][0] < w//2:
        big_corners = [top_bigcor[0],bottom_bigcor[0]]
        return big_corners
    else:
        big_corners = [top_bigcor[-1], bottom_bigcor[-1]]
        return big_corners


def check_pass_3lines(corners, big_corners, longlines, last_lls, imgshape):
    h, w, _ = imgshape
    pixel_limit = 30 # corner must be at least 10 pixels from the edge of the image

    # 📌 if corners is too small or longlines is too small => pass this frame
    if len(corners) < len(longlines) or len(big_corners)==1:
        return True
        
    for ll in longlines:
        if ll[0]<pixel_limit or ll[2]>w-pixel_limit:
            return True
        
        check_near = False # check exist cor(3lines) near this longline
        for cor in corners:
            if abs(ll[0]-cor[0])<30 or abs(ll[2]-cor[2])<30:
                # 0->30 usually is the distance between corners and ll
                check_near = True
                break
                
        if not check_near:
            return True
    
    for bigcor in big_corners:
        if bigcor[0]<5 or bigcor[2]>w-5:
            return True
        
        for ll in longlines:
            if abs(ll[0]-bigcor[0])<100 or ll[0]<10 or ll[2]>w-10:
                return True
            
    r'''
    📌 Check condition for longlines(current) and last_lls(last longlines):
    - continue if number of longlines is same as last frame
    - check if number of longlines is different with last frame (model detect wrong)
    '''
    if len(longlines)==1 and len(last_lls)==2:
        # arrange last_lls from left to right
        last_lls = sorted(last_lls, key=lambda x: x[0])
        if longlines[0][0] > last_lls[1][0]:
            return True
        
        if w-last_lls[1][2] > 100: # longline not yet out of the image if it > 100
            return True
                   
    return False


def filter_corner(corners, longlines):
    r'''
    Filter out the corners that aren't near longlines and remove them
    If not enought corners, generate new corners from longlines
    '''
    filter_cors = []
    for i, cor in enumerate(corners):
        for ll in longlines:
            if (abs(cor[0]-ll[0])<30 or abs(cor[2]-ll[2])<30):
                filter_cors.append(cor)
                break
            
    if len(filter_cors) < 4:
        # generate new corner
        for ll in longlines:
            count = 0
            for cor in filter_cors:
                if abs(cor[0]-ll[0])<30 or abs(cor[2]-ll[2])<30:
                    count += 1
                    sample_cor = cor # this corner will be combined with longlines to generate new corner
                    
            if count == 1:
                x1 = (ll[2]+ll[0])/2 - ((sample_cor[0]+sample_cor[2])/2-(ll[2]+ll[0])/2)
                if sample_cor[1] < (ll[1]+ll[3])/2:
                    y1 = ll[3]
                else:
                    y1 = ll[1]
                new_cor = [x1, y1, x1+1, y1+1, 0.5]
                filter_cors.append(new_cor)
                
    return filter_cors, longlines
                 

def get_point_of_cors(cor, height, width):
    r'''
    🛠️🛠️🛠️ 
    This function to calculate the point that represent the box of the corner
    Each point will be added some pixel to increase the view of the corner
    '''
    x_c, y_c = width//2, height//2 # ratio center of image
    x1, y1, x2, y2, _ = cor
    w, h = x2-x1, y2-y1
    add_cor = ADD_COR #add pixel for corner(3lines)
    add_bigcor = ADD_BIGCOR # add pixel for big corner
    
    if x2-x1 < 48 and y2-y1 < 48: # big corner usually
       if y1 < y_c:
           return ((x1+x2)/2, y1-add_cor)
       else:
           return ((x1+x2)/2, y2+add_cor)
    else:
        if x1<x_c and y1<y_c:
            return (x1-add_bigcor, y1-add_bigcor)
        elif x1>x_c and y1<y_c:
            return (x2+add_bigcor, y1-add_bigcor)
        elif x1<x_c and y1>y_c:
            return (x1-add_bigcor, y2+add_bigcor)
        else:
            return (x2+add_bigcor, y2+add_bigcor)
        

def crop_frame(img, corners):   
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
    
    return warped_image


def crop_3lines_img(img, big_corners, longlines, corners):
    h, w, _ = img.shape
    corners, longlines = filter_corner(corners=corners, longlines=longlines)
    main_cors = []
    
    for cor in corners:
        main_cors.append(get_point_of_cors(cor, h, w))
    
    for cor in big_corners:
        main_cors.append(get_point_of_cors(cor, h, w))
                                                                                                                                            
    main_cors = arrange_points(main_cors)
    
    return crop_frame(img, main_cors)


def crop_3lines_L(img, big_corners: List, type: str):
    add_bigcor = ADD_BIGCOR
    main_cors = []
    # get point of 4 big corners
    y_center = sum([bcor[1] for bcor in big_corners])//len(big_corners)

    # get point of 4 corners
    for cor in big_corners:
        x1, y1, x2, y2, _ = cor
        if type == 'first':
            if y1 < y_center:
                main_cors.append((x2 + add_bigcor, y1-add_bigcor))
            else:
                main_cors.append((x2 + add_bigcor, y2+add_bigcor))
        else: # type is 'last'
            if y1 < y_center:
                main_cors.append((x1 - add_bigcor, y1-add_bigcor))
            else:
                main_cors.append((x1 - add_bigcor, y2+add_bigcor))                                                                                                                    
    main_cors = arrange_points(main_cors)
    
    return crop_frame(img, main_cors)


def crop_special_img(data_bigcor):
    r'''
    This function to crop the special image that
    has only 2 big corners and 1 longline but don't have 3lines(corners)
    => Try generate 3lines from longline
    Return the cropped image 
    '''
    main_cors = []
    img, big_corners, longlines, _, _ = data_bigcor
    h, w, _ = img.shape
    cor_top = ((longlines[0][0]+longlines[0][2])/2, longlines[0][1]-ADD_COR)
    cor_bottom = ((longlines[0][0]+longlines[0][2])/2, longlines[0][3]+ADD_COR)
    main_cors.append(cor_top)
    main_cors.append(cor_bottom)
    
    for cor in big_corners:
        main_cors.append(get_point_of_cors(cor, h, w))
        
    main_cors = arrange_points(main_cors)

    return crop_frame(img, main_cors)


def is_first_or_last_cont(big_corners, imgshape):
    _, weight = imgshape[:2]
    r'''
    return False if:
        - number of big corners list other than 4
        - coordinates of big corners are too close with edge of image
    return True for the remaining cases
    '''
    
    if len(big_corners) != 4:
        return False
    
    limit_pixels = 30 # limit pixel between big corners and edge of image
    for bcor in big_corners:
        if bcor[0] < limit_pixels or bcor[2] > weight-limit_pixels:
            return False
        
    return True


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
    