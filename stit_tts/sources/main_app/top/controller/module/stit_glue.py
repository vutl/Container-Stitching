import cv2
import numpy as np
import time
from sources.utils.matching import *
from sources.utils.tools import nms
from sources.main_app.top.utils.helpper import concate_2image
from sources.main_app.top.controller.thread.thread_detect_lines import DetectLines

HEIGHT_LIMIT = 70


class StitGlue():
    def __init__(self, glue_matching_default, glue_matching_low_accuracy, model_detect_lines: DetectLines):
        self.max_weight_match_superglue = 1980
        
        self.superglue_matcher_default = glue_matching_default
        self.superglue_matcher_low_accuracy = glue_matching_low_accuracy
        self.detect_line = model_detect_lines
    
    def glue_matching(self, left_kpts, right_kpts, matches, imgshape1, imgshape2):
        matches_points = []
        for m in matches:
            left_point = left_kpts[m.queryIdx].pt
            right_point = right_kpts[m.trainIdx].pt
            
            if left_point[1] < HEIGHT_LIMIT or right_point[1] < HEIGHT_LIMIT\
                or left_point[1] > imgshape1[0] - HEIGHT_LIMIT \
                    or right_point[1] > imgshape2[0] - HEIGHT_LIMIT\
                        or abs(left_point[1] - right_point[1]) > 10:
                continue
            
            matches_points.append((left_point, right_point))
        return matches_points
    
    def match_points(self, img1, img2, cor1, cor2):
        #####
        w1 = img1.shape[1]
        w2 = img2.shape[1]
        if w1 > self.max_weight_match_superglue:
            crop1 = img1[:, w1-self.max_weight_match_superglue:].copy()
        else:
            crop1 = img1.copy()
        if w2 > self.max_weight_match_superglue:
            crop2 = img2[:, :self.max_weight_match_superglue].copy()
        else:
            crop2 = img2.copy()
        

        img1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        
        left_kpts, right_kpts, _, _, matches = self.superglue_matcher_default.match(
            img1_gray, img2_gray)

        matcher_points = glue_matching(
            left_kpts, right_kpts, matches, \
                img1_gray.shape, img2_gray.shape, \
                    cor1, cor2)

        if len(matcher_points) < 16:
            left_kpts, right_kpts, _, _, matches = self.superglue_matcher_low_accuracy.match(
                img1_gray, img2_gray)
            matcher_points = glue_matching(
            left_kpts, right_kpts, matches, \
                img1_gray.shape, img2_gray.shape, \
                    cor1, cor2)
            # print("**** matching < 16")
            if len(matcher_points) < 3:
                # print("Số lượng matches vẫn ít hơn 3, chọn điểm có độ lệch y nhỏ nhất.")
                matcher_points.sort(key=lambda x: abs(x[0][1] - x[1][1]))
                if len(matcher_points) < 2:
                    return [], []
                # Trả về cặp điểm có độ lệch y nhỏ nhất
                return np.asarray(matcher_points[1][0]), np.asarray(matcher_points[1][1])
        ####
        # Phân cụm các điểm bằng KMeans
        clustered_points = cluster_keypoints_kmeans(matcher_points, image_height=crop1.shape[0])

        # Tính độ tự tin cho các cụm
        best_cluster_label = calculate_confidence(clustered_points)

        # Lấy điểm có tọa độ y gần với giá trị trung bình của cụm nhất
        point1, point2 = get_point_near_cluster_avg(
            clustered_points, best_cluster_label)

        return np.asarray(point1), np.asarray(point2)

    def find_3lines_pts(self, img1_shape, img2_shape, result1, result2, cor1=None, cor2=None):
        r'''
        this function find 2 point matching between two images
        with detection of 3 lines
        - return: pt1, pt2 that is the center of long lines
        - if not find or not satisfy condition between 2 points return ((False, False))
        '''
        h1, w1 = img1_shape[:2]
        h2, w2 = img2_shape[:2]
        diff_limit = 260 + max(cor2[0]-cor1[0], 0) # limit of distance between 2 points
        
        longlines1, longlines2 = [], []
        big_cors1, big_cors2 = [], [] 
        for i, r in enumerate([result1, result2]):
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                if y2-y1 > 50 and y2-y1 < 200:
                    big_cors1.append([x1,y1,x2,y2, box.conf[0]]) if i == 0 \
                        else big_cors2.append([x1,y1,x2,y2, box.conf[0]])
                
                if y2-y1 < 200:
                    continue
                if i == 0:
                    longlines1.append([x1,y1,x2,y2, box.conf[0]])
                else:
                    longlines2.append([x1,y1,x2,y2, box.conf[0]])

        longlines1 = nms(longlines1, 0.3)
        longlines2 = nms(longlines2, 0.3)     
        big_cors1 = nms(big_cors1, 0.1)
        big_cors2 = nms(big_cors2, 0.1)       
        longlines1 = sorted(longlines1, key=lambda x: x[0])
        longlines2 = sorted(longlines2, key=lambda x: x[0])
        big_cors1 = sorted(big_cors1, key=lambda x: x[0])
        big_cors2 = sorted(big_cors2, key=lambda x: x[0])
        
        if len(big_cors1) == 2 and len(big_cors2) == 2:
            # print('matching with big cors')
            if big_cors1[0][0] > w1//2:
                pt1, pt2 = (w1-5, h1//2), (w2-5, h2//2)
            else:
                pt1, pt2 = (5, h1//2), (5, h2//2)
            return ((pt1, pt2))
        elif len(big_cors1) == 4 and len(big_cors2) == 4:
            pt1 = (big_cors1[1][0], h1//2)
            pt2 = (big_cors2[1][0], h2//2)
            return ((pt1, pt2))        
        
        if len(longlines1)==0 or len(longlines2)==0:
            return ((False, False))
        
        for ll1 in longlines1[::-1]:
            pt1 = ((ll1[0] + ll1[2]) // 2, (ll1[1] + ll1[3]) // 2)
            pt2 = None
            for ll2 in longlines2:
                if ll2[0]+cor2[0] > ll1[0]+cor1[0]+10 and abs(ll2[0] - ll1[0]) < diff_limit:
                    pt2 = ((ll2[0] + ll2[2]) // 2, (ll2[1] + ll2[3]) // 2)
                    break
            
            if pt2 != None:
                break
            else:
                pt2 = ((longlines2[-1][0] + longlines2[-1][2]) // 2, (longlines2[-1][1] + longlines2[-1][3]) // 2)   
        
        
        if pt2[0]+cor2[0] < pt1[0]+cor1[0]+10 or abs(pt2[0] - pt1[0]) >= diff_limit:
            # print('check point1', pt1, 'point2', pt2, 'diff', abs(pt1[0] - pt2[0]), 'diff_limit', diff_limit)
            return ((False, False))
        
        return ((pt1, pt2))
    
    def stit_glue(self, processed_imgs, ls_corners, step=1):
        print(len(processed_imgs), "**********")
        final_concate_img = None
        time_start = time.time()
        # find 3 lines for all images
        results = self.detect_line.detect_line(processed_imgs)
        
        count_match_img = 0
        count_next_img = 0
        for i, img in enumerate(processed_imgs):
            if i < 1:
                final_concate_img = img
                continue
            if count_next_img == i:
                continue

            # Try to match with find 3 lines
            point1, point2 = self.find_3lines_pts(
                processed_imgs[i-1].shape, img.shape, results[i-1], results[i], ls_corners[i-1], ls_corners[i]
            )
            if not point1:
                point1, point2 = self.match_points(
                    processed_imgs[i-1], img, ls_corners[i-1], ls_corners[i]
                    )
            if not len(point1):
                if i == len(processed_imgs) - 1:
                    continue
                point1, point2 = self.match_points(
                    processed_imgs[count_match_img], processed_imgs[i+1],
                    ls_corners[count_match_img], ls_corners[i+1]
                    )
                count_next_img = i + 1
                if not len(point1):
                    continue
                final_concate_img = concate_2image(
                    final_concate_img, processed_imgs[i+1], point1, point2)
                count_match_img = i
                continue

            count_match_img = i
            final_concate_img = concate_2image(
                final_concate_img, img, point1, point2)
        
        print("Time stit glue:", time.time() - time_start)
        return final_concate_img

