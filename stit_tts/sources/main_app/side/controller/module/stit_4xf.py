import cv2
import numpy as np
from sources.main_app.side.controller.thread.thread_detect_dents import DetectDent
from sources.main_app.side.controller.module.stit_glue import StitGlue
from sources.main_app.side.utils.util import crop_stit_imgs
from sources.utils.matching import *


class Stit4xf():
    def __init__(self, glue_matching_default, glue_matching_low_accuracy):

        self.max_weight_match_superglue = 1980
        self.superglue_matcher_default = glue_matching_default
        self.superglue_matcher_low_accuracy = glue_matching_low_accuracy

        self.detect_dent = DetectDent()
        self.stit_glue = StitGlue(
            glue_matching_default=self.superglue_matcher_default,
            glue_matching_low_accuracy=self.superglue_matcher_low_accuracy
        )

    def setup_model(self):
        self.detect_dent.setup_model()

    def match_points_with_filter(self, img1, img2):
        """
        So khớp điểm với việc điều chỉnh kích thước ảnh.
        """
        w1, w2 = img1.shape[1], img2.shape[1]

        crop_offset_img1 = 0
        crop_offset_img2 = 0

        if w1 > self.max_weight_match_superglue:
            crop_offset_img1 = w1 - self.max_weight_match_superglue
            img1 = img1[:, crop_offset_img1:w1]
        if w2 > self.max_weight_match_superglue:
            crop_offset_img2 = 0
            img2 = img2[:, :self.max_weight_match_superglue]

        min_width = min(img1.shape[1], img2.shape[1])
        img1, img2 = img1[:, :min_width], img2[:, -min_width:]

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        left_kpts, right_kpts, _, _, matches = self.superglue_matcher_default.match(
            img1_gray, img2_gray)
        matcher_points = glue_matching(
            left_kpts, right_kpts, matches, imgshape1=img1_gray.shape, imgshape2=img2_gray.shape)

        if len(matcher_points) < 20:
            left_kpts, right_kpts, _, _, matches = self.superglue_matcher_low_accuracy.match(
                img1_gray, img2_gray)
            matcher_points = glue_matching(
                left_kpts, right_kpts, matches, imgshape1=img1_gray.shape, imgshape2=img2_gray.shape)

            if len(matcher_points) < 3:
                matcher_points.sort(key=lambda x: abs(x[0][1] - x[1][1]))
                return matcher_points[0][0], matcher_points[0][1] if matcher_points else ([], [])

        matcher_points = [x for x in matcher_points if (
            (x[0][0] + 3) <= (x[1][0]))]

        clustered_points = cluster_keypoints_kmeans(matcher_points)
        best_cluster_label = determine_best_cluster(clustered_points)

        point1, point2 = get_point_near_cluster_avg(
            clustered_points, best_cluster_label)

        if point1 and point2:
            # Điều chỉnh tọa độ trở lại ảnh gốc
            actual_x1 = point1 + crop_offset_img1
            actual_x2 = point2 + crop_offset_img2
            distance_x = abs(actual_x2 - actual_x1)
        else:
            distance_x = None

        return distance_x

    def cal_distance(self, img1, img2):
        pts1, pts2 = self.stit_glue.match_points(img1, img2)
        return np.linalg.norm(pts1 - pts2) if pts1 is not None and pts2 is not None else None

    def choose_frames(self, imgs, list_first, ndent_perimg_ls, l_dent, step_dent):
        stit_imgs = []
        count_dent = 0

        for i, first in enumerate(list_first):
            if i == 0:
                print(i)
                stit_imgs.append(imgs[i])
                print('pick =============> ', i, count_dent)
                continue

            print('step____', i, imgs[i].shape[1]/l_dent,
                  'from', list_first[i-1], 'to', list_first[i])

            if list_first[i] > list_first[i-1]:

                if list_first[i]-list_first[i-1] > max(step_dent*0.65, 0.1):
                    print('pass----check',
                          list_first[i], 'and', list_first[i-1])
                    add_dent = list_first[i]-list_first[i-1]

                elif list_first[i]-list_first[i-1] < 0.2*step_dent:
                    add_dent = list_first[i]-list_first[i-1]+1

                elif list_first[i]-list_first[i-1]+1 > 2.2*step_dent:
                    add_dent = list_first[i]-list_first[i-1]

                else:
                    print('check with Stitching image', i,
                          list_first[i], 'and', i-1, list_first[i-1])

                    try:
                        # check with Stitching image
                        # distance = self.match_points_with_filter(imgs[i-1],imgs[i])
                        distance = self.cal_distance(imgs[i-1], imgs[i])

                        k_dis = distance / \
                            ((list_first[i]-list_first[i-1])*l_dent)
                        print('*****distance', distance,
                              (list_first[i]-list_first[i-1])*l_dent, 'k_dis', k_dis)
                    except:
                        print('error')
                        k_dis = 1
                        distance = 0

                    if (k_dis > 0.6 and k_dis < 1.6) or distance < l_dent*0.7:
                        add_dent = list_first[i]-list_first[i-1]
                    else:
                        add_dent = list_first[i]-list_first[i-1]+1
            else:
                if 1-list_first[i-1]+list_first[i] > max(step_dent*0.7, 0.1):
                    add_dent = 1-list_first[i-1]+list_first[i]
                    print('pass----check',
                          list_first[i], 'and', list_first[i-1])
                else:
                    try:
                        # check with Stitching image
                        # distance = self.match_points_with_filter(imgs[i-1],imgs[i])
                        distance = self.cal_distance(imgs[i-1], imgs[i])

                        k_dis = (1-list_first[i-1] +
                                 list_first[i])*l_dent/distance
                        print('*****distance', distance,
                              (1-list_first[i-1]+list_first[i])*l_dent, 'k_dis', k_dis)
                    except:
                        k_dis = 1
                        distance = 0

                    if (k_dis > 0.6 and k_dis < 1.6) or distance < l_dent*0.7:
                        add_dent = 1-list_first[i-1]+list_first[i]

                    else:
                        add_dent = 1-list_first[i-1]+list_first[i]+1

            count_dent += add_dent

            print(
                f'check step {i} with value {count_dent} --- {ndent_perimg_ls[i]}')

            if count_dent+0.4*step_dent > ndent_perimg_ls[i]:

                print('pick =============> ', i-1, count_dent)
                # print('_______________________________________')
                count_dent = add_dent
                x_crop = int(0.2*l_dent*step_dent)
                stit_imgs.append(imgs[i-1][:, :-x_crop])

            if i == len(list_first)-1:
                print('pick =============> ', i, count_dent)
                stit_imgs.append(imgs[i])

        return stit_imgs, count_dent

    def stit_dents(self, imgs, cont_size):
        if cont_size == 42:
            cont_size = 44

        imgs = self.detect_dent.get_nessessary_imgs(imgs)
        print('__________len_imgs___________', len(imgs))

        # get nessessary params to prepare for stitching
        l_dent, stepdent = self.detect_dent.get_params(imgs, cont_size)
        print('__________prepared_params___________', l_dent, stepdent)

        # get list of first boxes in order to estimate distance between 2 frames
        first_boxes = []
        ndent_perimg_ls = []
        for i, img in enumerate(imgs):
            dents = self.detect_dent.detect_dents(img)
            cv2.imwrite(f'bug.jpg', img)
            # average length of dents in each frame
            dent_average = sum(x2-x1 for x1, y1, x2, y2,
                               _ in dents[1:-1])/len(dents[1:-1])
            # ratio of first dent in each frame
            first = (dents[0][2]-dents[0][0])/dent_average
            # number of dents in each frame
            ndent_perimg = (img.shape[1]/(dents[3][0]-dents[2][0]))
            first_boxes.append(first)
            ndent_perimg_ls.append(ndent_perimg)

        print('__________len_first_boxes___________', len(first_boxes))

        # Choose frames to stitch
        stit_imgs, last_count_dent = self.choose_frames(
            imgs, first_boxes, ndent_perimg_ls, l_dent, stepdent)

        # Get list of dents from stit_imgs
        list_stit_dents = self.detect_dent.get_stit_dents(stit_imgs)

        # crop stit_imgs before stitching ...
        stit_imgs = crop_stit_imgs(
            stit_imgs, list_stit_dents, l_dent, last_count_dent)

        r"""
        Try to stitching 2 last images with stitch glue
        Except continue stitching with counting dents
        """
        last_2imgs = self.stit_glue.stit_glue(
            processed_imgs=[stit_imgs[-2], imgs[-1]])
        if last_2imgs is not None:
            stit_imgs = stit_imgs[:-2] + [last_2imgs]

        # Let Stitching images !!!!!!!!!!!!!!!!!
        lsimgs = stit_imgs[::-1]
        sitched_imgs = np.concatenate(lsimgs, axis=1)

        return sitched_imgs
