import cv2
import numpy as np
from sources.utils.matching import *
from sources.main_app.side.utils.util import concate_2image, concate_2image_fix_dent, fix_point


class StitGlue():
    def __init__(self, glue_matching_default, glue_matching_low_accuracy):
        self.max_weight_match_superglue = 1980
        
        self.superglue_matcher_default = glue_matching_default
        self.superglue_matcher_low_accuracy = glue_matching_low_accuracy
    
    def match_points(self, img1, img2):
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
        #####

        img1_gray = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)
        # print(img1_gray.shape, img2_gray.shape, "shape")
        left_kpts, right_kpts, _, _, matches = self.superglue_matcher_default.match(
            img1_gray, img2_gray)
        print(len(matches), "###################")
        matcher_points = glue_matching(left_kpts, right_kpts, matches, imgshape1=img1_gray.shape, imgshape2=img2_gray.shape)

        if len(matcher_points) < 16:
            left_kpts, right_kpts, _, _, matches = self.superglue_matcher_low_accuracy.match(
                img1_gray, img2_gray)
            matcher_points = glue_matching(left_kpts, right_kpts, matches, imgshape1=img1_gray.shape, imgshape2=img2_gray.shape)
            print("**** matching < 16")
            if len(matcher_points) < 3:
                print("Số lượng matches vẫn ít hơn 3, chọn điểm có độ lệch y nhỏ nhất.")
                matcher_points.sort(key=lambda x: abs(x[0][1] - x[1][1]))
                if len(matcher_points) < 2:
                    return [], []
                # Trả về cặp điểm có độ lệch y nhỏ nhất
                return np.asarray(matcher_points[1][0]), np.asarray(matcher_points[1][1])
        ####
        # Phân cụm các điểm bằng KMeans
        clustered_points = cluster_keypoints_kmeans(matcher_points)

        # Tính độ tự tin cho các cụm
        best_cluster_label = calculate_confidence(clustered_points)

        # Lấy điểm có tọa độ y gần với giá trị trung bình của cụm nhất
        point1, point2 = get_point_near_cluster_avg(
            clustered_points, best_cluster_label)
        
        p1, p2 = np.asarray(point1), np.asarray(point2)
        p1, p2 = fix_point(p1, p2, w1, w2)
        
        return p1, p2
        # return np.asarray(point1), np.asarray(point2)
    
    def stit_glue(self, processed_imgs):
        print(len(processed_imgs), "**********")
        final_concate_img = None
        count_match_img = 0
        count_next_img = 0
        for i, img in enumerate(processed_imgs):
            if i < 1:
                final_concate_img = img
                continue
            if count_next_img == i:
                continue
            point1, point2 = self.match_points(processed_imgs[i-1], img)
            if not len(point1):
                if i == len(processed_imgs) - 1:
                    continue
                point1, point2 = self.match_points(
                    processed_imgs[count_match_img], processed_imgs[i+1])
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
        return final_concate_img

    def stit_glue_fix_dent(self, processed_imgs):
        len_imgs = len(processed_imgs)
        print("**********", len_imgs)
        final_concate_img = None
        count_match_img = 0
        count_next_img = 0
        for i, img in enumerate(processed_imgs):
            if i < 1:
                final_concate_img = img
                continue
            if count_next_img == i:
                continue
            point1, point2 = self.match_points(processed_imgs[i-1], img)
            if not len(point1):
                if i == len_imgs - 1:
                    continue
                point1, point2 = self.match_points(
                    processed_imgs[count_match_img], processed_imgs[i+1])
                count_next_img = i + 1
                if not len(point1):
                    continue

                final_concate_img = concate_2image_fix_dent(
                    final_concate_img, processed_imgs[i+1], point1, point2, i, len_imgs)
                count_match_img = i
                continue

            count_match_img = i
            final_concate_img = concate_2image_fix_dent(
                final_concate_img, img, point1, point2, i, len_imgs)
        return final_concate_img

