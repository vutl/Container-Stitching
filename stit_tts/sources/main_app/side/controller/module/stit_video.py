import os
import cv2
import time
import numpy as np
from sources.utils.matching import *
from sources.main_app.side.utils.util import concate_2image, concate_2image_fix_dent, fix_point
from sources.main_app.side.controller.module.image_cropper import ImageCropper
from sources.utils.util import sort_name_in_folder
from sources.config import ROOT_STORAGE


class StitVideoGlue():
    def __init__(self, glue_matching_default, glue_matching_low_accuracy):
        self.max_weight_match_superglue = 1980
        
        self.image_cropper = ImageCropper()
        
        self.superglue_matcher_default = glue_matching_default
        self.superglue_matcher_low_accuracy = glue_matching_low_accuracy
        
    def draw_matcher_points(self, img1, img2, matcher_points):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # ghép ảnh cạnh nhau
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = img1
        canvas[:h2, w1:w1+w2] = img2

        # vẽ điểm match và line nối
        for (xy1, xy2) in matcher_points:
            x1, y1 = xy1
            x2, y2 = xy2
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # điểm bên trái (ảnh 1)
            cv2.circle(canvas, (int(x1), int(y1)), 4, color, -1)
            # điểm bên phải (ảnh 2) -> dịch x theo w1
            cv2.circle(canvas, (int(x2) + w1, int(y2)), 4, color, -1)
            # nối line
            cv2.line(canvas, (int(x1), int(y1)), (int(x2) + w1, int(y2)), color, 1)

        return canvas
    
    def match_points(self, image1, image2, i):
        img1 = image1.copy()
        img2 = image2.copy()
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
        
        left_kpts, right_kpts, _, _, matches = self.superglue_matcher_default.match(
            img1_gray, img2_gray)
        print(len(matches), "###################")
        matcher_points = glue_matching(left_kpts, right_kpts, matches, imgshape1=img1_gray.shape, imgshape2=img2_gray.shape)
        # img = self.draw_matcher_points(img1, img2, matcher_points)
        
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
        
        # points = [p[:2] for p in clustered_points if p[2] == best_cluster_label]
        # img2 = self.perspec_img(img2, points)

        # Lấy điểm có tọa độ y gần với giá trị trung bình của cụm nhất
        point1, point2 = get_point_near_cluster_avg(
            clustered_points, best_cluster_label)

        p1, p2 = np.asarray(point1), np.asarray(point2)
        p1, p2 = fix_point(p1, p2, w1, w2)
        
        return p1, p2

    def perspec_img(self, img2, matcher_points):

        pts1 = [p[0] for p in matcher_points]
        pts1 = np.float32(pts1)
        pts2 = [p[1] for p in matcher_points]
        pts2 = np.float32(pts2)

        # Tính homography
        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        # Warp ảnh 2 cho khớp ảnh 1
        h, w, _ = img2.shape
        aligned_img2 = cv2.warpPerspective(img2, H, (w, h))
        # img = cv2.hconcat((img2, aligned_img2))
        return aligned_img2

    def stit_glue(self, processed_imgs):
        print(len(processed_imgs), "**********")
        final_concate_img = None
        count_match_img = 0
        count_next_img = 0
        for i, img in enumerate(processed_imgs):
            if i < 1 and final_concate_img is None:
                final_concate_img = processed_imgs[0]
                continue
            if count_next_img == i:
                continue
            point1, point2 = self.match_points(processed_imgs[i-1], img, i)
            if not len(point1):
                if i == len(processed_imgs) - 1:
                    continue
                point1, point2 = self.match_points(
                    processed_imgs[count_match_img], processed_imgs[i+1], i)
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
        return final_concate_img[:-110,:]

    def drop_imgs_with_min_len(self, ls_imgs, min_len=4):
        ls_drop_imgs = []
        ls_idx = []
        if len(ls_imgs) < min_len:
            return ls_imgs, []
        for i in range(min_len):
            idx = round(i * (len(ls_imgs) - 1) / (min_len - 1))
            ls_drop_imgs.append(ls_imgs[idx])
            ls_idx.append(idx)
        return ls_drop_imgs, ls_idx
    
    def test_imgs(self, path_folder):
        ls_imgs = []
        ls_fns = sort_name_in_folder(os.path.join(ROOT_STORAGE, path_folder))
        for fn in ls_fns:
            img = cv2.imread(os.path.join(path_folder, fn))
            ls_imgs.append(img)
        return ls_imgs

    def stit_with_video(self, video_path, cont_size=42, side='left'):
        t = time.time()
        process_imgs = self.image_cropper.read_video(video_path)
        print("len process imgs:", len(process_imgs))
        ls_drop_imgs, ls_idx = self.drop_imgs_with_min_len(process_imgs, 60)
        print("time process video:", time.time() - t)
        if side == 'left':
            ls_drop_imgs = ls_drop_imgs[::-1]
        return self.stit_glue(ls_drop_imgs)


