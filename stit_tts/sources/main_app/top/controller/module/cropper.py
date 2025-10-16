import torch
import cv2
from ultralytics import YOLO
from sources.main_app.top.controller.thread.thread_detect_cor import DetectCor
from ...utils.tool_top import get_point_of_cors, crop_frame, arrange_points
from ...utils.helpper import get_center, line_intersection, nms
from sources.main_app.top.utils.cor_util import get_results

ADD_COR = 20
ADD_BIGCOR = 10

CROP_LEFT = 130
CROP_RIGHT = 130
     
     
class Cropper():
    def __init__(self, model_detect_corners: DetectCor):
        
        self.detect_corners = model_detect_corners
        
        self.max_pixel_distance_2points_corner = 200
        self.ratio_size_cont_dewarp = 0.65
        self.diff_limit = 70
    
    def filter_corner(self, corners, longlines):
        r'''
        Filter out the corners that aren't near longlines and remove them
        If not enought corners, generate new corners from longlines
        '''
        dis_limit = 50 # distance limit to filter corners
        filter_cors = []
        for i, cor in enumerate(corners):
            for ll in longlines:
                if (abs(cor[0]-ll[0])<dis_limit or abs(cor[2]-ll[2])<dis_limit):
                    filter_cors.append(cor)
                    break
                
        return filter_cors
    
    def get_sp_corners(self, img) -> list:
        r'''
        get far corners of container image after padding border around the image
        return coordinates of corners to support for cropping
        '''
        img_pad = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        results = self.detect_corners.detect_cor(img_pad)
        
        sp_corners = [] # list support for cropping
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            sp_corners.append([x1, y1, x2, y2, r.conf[0]]) if y2-y1 < 50 else None
            
        sp_corners = nms(sp_corners, 0.2)

        sp_pt_cors = []
        for cor in sp_corners:
            x1, y1, x2, y2 = cor[:4]
            center_x, center_y = get_center(x1, y1, x2, y2)
            center_x, center_y = max(0, center_x - 50), max(0, center_y - 50)
            center_x, center_y = min(img.shape[1], center_x), min(img.shape[0], center_y)
            sp_pt_cors.append([center_x, center_y])
            
        return sp_pt_cors

    def generate_corners(self, corners, longlines, sp_cors, imgshape):
        gen_cors = corners
        center_h, center_w = (longlines[0][1]+longlines[0][3])//2, (longlines[0][0]+longlines[1][0])//2
        sp_top_cors = [cor for cor in sp_cors if cor[1] < center_h]
        sp_bot_cors = [cor for cor in sp_cors if cor[1] > center_h]
        
        for ll in longlines:
            count = 0
            for cor in corners:
                if abs(cor[0]-ll[0])<30 or abs(cor[2]-ll[2])<30:
                    count += 1
                    sample_cor = cor # this corner will be combined with longlines to generate new corner
            
            # find temperary corner that depends on longlines       
            if count == 1:
                x_temp = (ll[2]+ll[0])/2 - ((sample_cor[0]+sample_cor[2])/2-(ll[2]+ll[0])/2) 
               
                if sample_cor[1] < center_h:
                    y_temp = ll[3]
                    
                    if len(sp_bot_cors) == 2:
                        # get x, y official corner
                        x_official, y_official = line_intersection(
                            sp_bot_cors[0], sp_bot_cors[1], (x_temp, y_temp), sample_cor)
                    else:
                        x_official, y_official = x_temp, y_temp      
                else:
                    y_temp = ll[1]
                    
                    if len(sp_top_cors) == 2:
                        # get x, y official corner
                        x_official, y_official = line_intersection(
                            sp_top_cors[0], sp_top_cors[1], (x_temp, y_temp), sample_cor)
                    else:
                        x_official, y_official = x_temp, y_temp
                    
                cor_gen = [x_official, y_official, x_official+1, y_official+1, 0.5]
                gen_cors.append(cor_gen)
    
        return gen_cors

    def get_temp_cors(self, longlines, big_pts, top_cor_3l, bot_cor_3l, center_h, center_w):
        r'''
        This function to get temporary corners that depend on:
        longlines, corners and big corners
        return temporary corners (bot and top)
        '''
        # get top temporary corner
        if len(top_cor_3l):
            temp_cor_top = top_cor_3l[0]
        elif len(bot_cor_3l):
            centerX_ll = (longlines[0][0]+longlines[0][2])/2
            x_temp = centerX_ll - ((bot_cor_3l[0][0]+bot_cor_3l[0][2])/2-centerX_ll)
            y_temp = longlines[0][1] - ADD_COR
            temp_cor_top = (x_temp, y_temp)
        else:
            y_temp = longlines[0][1] - ADD_COR
            if big_pts[0][0] > center_w:
                x_temp = longlines[0][2] if y_temp>big_pts[0][1] else longlines[0][0]
            else:
                x_temp = longlines[0][0] if y_temp>big_pts[0][1] else longlines[0][2]
            temp_cor_top = (x_temp, y_temp)
            
        # get bottom temporary corner
        if len(bot_cor_3l):
            temp_cor_bot = bot_cor_3l[0]
        elif len(top_cor_3l):
            centerX_ll = (longlines[0][0]+longlines[0][2])/2
            x_temp = centerX_ll - ((top_cor_3l[0][0]+top_cor_3l[0][2])/2-centerX_ll)
            y_temp = longlines[0][3] + ADD_COR
            temp_cor_bot = (x_temp, y_temp)
        else:
            y_temp = longlines[0][3] + ADD_COR
            if big_pts[0][0] > center_w:
                x_temp = longlines[0][2] if y_temp>big_pts[1][1] else longlines[0][0]
            else:
                x_temp = longlines[0][0] if y_temp>big_pts[1][1] else longlines[0][2]
            temp_cor_bot = (x_temp, y_temp)  
        
        return temp_cor_top, temp_cor_bot
    
    def cropping(self, img, big_corners, longlines, corners):
        r'''
        This function to generate main corners(if lack corners)
        Return the cropped image
        '''
        h, w = img.shape[:2]
        corners = self.filter_corner(corners, longlines)
        
        if len(big_corners):
            data_bigcor = (img, big_corners, longlines, corners, 0)
            return self.special_cropping(data_bigcor)
        
        if len(corners) < len(longlines)*2:
            sp_cors = self.get_sp_corners(img)
            corners = self.generate_corners(corners, longlines, sp_cors, img.shape)
        
        main_cors = []   
        for cor in corners:
            main_cors.append(get_point_of_cors(cor, h, w))
        
        for cor in big_corners:
            main_cors.append(get_point_of_cors(cor, h, w))
            
        main_cors = arrange_points(main_cors)
    
        return crop_frame(img, main_cors)  
    
    
    def special_cropping(self, data_bigcor):
        r'''
        This function to crop the special image that
        has only 2 big corners and 1 longline but don't have 3lines(corners)
        => Try generate 3lines from longline
        Return the cropped image 
        '''
        
        img, big_corners, longlines, corners, i = data_bigcor
        center_h = (big_corners[0][1]+big_corners[1][1])//2
        center_w = (longlines[0][0]+big_corners[0][0])//2
        
        big_pts = [get_point_of_cors(cor, img.shape[0], img.shape[1])\
            for cor in big_corners]
        big_pts = sorted(big_pts, key=lambda x: x[1])
        
        top_cor_3l, bot_cor_3l = [], []
        for cor in corners:
            if cor[1] < center_h:
                top_cor_3l.append(cor)
            else:
                bot_cor_3l.append(cor)
        
        # get temporary corner that depends on longlines
        temp_cor_top, temp_cor_bot = self.get_temp_cors(
            longlines, big_pts, top_cor_3l, bot_cor_3l, center_h, center_w
        )
        
        sp_cors = self.get_sp_corners(img)
        sp_cors = sorted(sp_cors, key=lambda x: x[0])
        sp_top_cors = [cor for cor in sp_cors if cor[1] < center_h]
        sp_bot_cors = [cor for cor in sp_cors if cor[1] > center_h]
        
        # corection for sp_cors ( add 15 pixels to the y-axis)
        if len(sp_top_cors):
            sp_top_cors[0][1] -= 15
        if len(sp_bot_cors):
            sp_bot_cors[0][1] += 15
        
        # Let's find the official corner
        main_cors = []
        if len(top_cor_3l):
            x_official = (top_cor_3l[0][0]+top_cor_3l[0][2])/2
            y_official = top_cor_3l[0][1] - ADD_COR
        elif len(sp_top_cors)==2:
            print('sp_top_cors', sp_top_cors)
            
            if big_pts[0][0] > center_w:
                del sp_top_cors[1]
            else:
                del sp_top_cors[0]
            
            sp_top_cors[0][1] -= ADD_BIGCOR
            if sp_top_cors[0][0] < center_w:
                sp_top_cors[0][0] -= ADD_BIGCOR
            else:
                sp_top_cors[0][0] += ADD_BIGCOR
            x_official, y_official = line_intersection(
                sp_top_cors[0], big_pts[0], temp_cor_top, temp_cor_bot)
        else:
            x_official, y_official = temp_cor_top[0], temp_cor_top[1]   
        main_cors.append((x_official, y_official))
        
        if len(bot_cor_3l):
            x_official = (bot_cor_3l[0][0]+bot_cor_3l[0][2])/2
            y_official = bot_cor_3l[0][3] + ADD_COR    
        elif len(sp_bot_cors)==2:
            if big_pts[0][0] > center_w:
                del sp_bot_cors[1]
            else:
                del sp_bot_cors[0]
            
            sp_bot_cors[0][1] += ADD_BIGCOR
            if sp_bot_cors[0][0] < center_w:
                sp_bot_cors[0][0] -= ADD_BIGCOR
            else:
                sp_bot_cors[0][0] += ADD_BIGCOR
            x_official, y_official = line_intersection(
                sp_bot_cors[0], big_pts[1], temp_cor_top, temp_cor_bot)
        else:
            x_official, y_official = temp_cor_bot[0], temp_cor_bot[1]
        main_cors.append((x_official, y_official))
            
        # Get the last corner
        main_cors = main_cors + big_pts
        main_cors = arrange_points(main_cors)
        
        return crop_frame(img, main_cors)
    
    def is_corners(self, corners, w, h):
        xy1, xy2, xy3, xy4 = corners
        w, h = w-100, h-100
        if abs(xy4[0] - xy1[0]) < self.max_pixel_distance_2points_corner and abs(xy4[1] - xy1[1]) > h // 3 \
            and abs(xy3[0] - xy2[0]) < self.max_pixel_distance_2points_corner and abs(xy3[1] - xy2[1]) > h // 3 \
                and abs(xy2[0] - xy1[0]) > w * self.ratio_size_cont_dewarp :
                    return True
        return False

    def corners_process(self, list_imgs, step=5): 
        list_process_imgs = []
        ls_corners = []
        flag_split_2cont = False
        
        step = max(1, (len(list_imgs)-20) // 30)
        list_imgs = list_imgs[::step]
        print('len after divide', len(list_imgs)) 
        
        for i, img in enumerate(list_imgs): 
            img = img[100:-200, CROP_LEFT:-CROP_RIGHT] 
            img = cv2.copyMakeBorder(img, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            h, w, _ = img.shape
            results = self.detect_corners.detect_cor(img)
            
            if len(list_process_imgs)==0 and len(results[0].boxes) > 1:
                step = 1
            
            if len(results[0].boxes) == 0 and len(list_process_imgs) > 3 :
                break
            
            if len(results[0].boxes) < 3:
                continue

            corners, check_pass = get_results(img, results)
            
            if check_pass:
                continue
        
            if len(corners) >= 4:
                if self.is_corners(corners, w, h):
                    img_crop = crop_frame(img, corners)
                    list_process_imgs.append(img_crop)
                    ls_corners.append(corners[0])
        print('len list_process_imgs', len(list_process_imgs))
        if not len(list_process_imgs):
            return [], False
        
        # ------------------------resize to same height------------------------#
        height_scale = min(img.shape[0] for img in list_process_imgs)

        processed_frames = [cv2.resize(img,\
                            (int(img.shape[1]*height_scale/img.shape[0]), height_scale))\
                            for img in list_process_imgs]

        return processed_frames, ls_corners


        
            