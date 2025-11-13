from itertools import count
import cv2
import numpy as np
from sources.config import ROOT_STORAGE, CORNERS_CONFIG
from sources.main_app.side.utils.tool_side import crop_frame, get_results, adjust_corners, get_center, nms
from sources.main_app.side.controller.thread.thread_detect_corner import DetectCorners
from typing import List, Tuple, Union, Any
 

class ImageCropper:
    def __init__(self):
        self.detect_corners = DetectCorners()
        
        self.max_pixel_distance_2points_corner = 100
        
        self.count = 0
        
        self.setup_defisheye()
    
    def setup_defisheye(self):
        h, w = 1600, 1200
        K = np.array([[1134, 0, w//2], [0, 1173, h//2],
                      [0, 0, 1]], dtype=np.float32)  # 1200x1600
        D = np.array([-0.248, 0.1, 0, 0], dtype=np.float32)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=1)
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
        
        self.mapx, self.mapy = mapx, mapy
    
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
    
    def defish_image(self, image):
        undistorted = cv2.remap(image, self.mapx, self.mapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        return undistorted
    
    def is_choose_corners(self, corners) -> bool:
        if len(corners) != 4:
            return False
        xy1, xy2, xy3, xy4 = corners
        dis_x1 = abs(xy4[0] - xy1[0])
        dis_x2 = abs(xy3[0] - xy2[0])
        
        dis_y1 = abs(xy1[1] - xy3[1])
        dis_y2 = max(1, abs(xy2[1] - xy4[1]))
        if abs(dis_x1 - dis_x2) < 22:
            if abs((dis_y1 / dis_y2) - 1) < 0.08:
                return True
        print("Not choose corners:", abs(dis_x1 - dis_x2), abs((dis_y1 / dis_y2) - 1))
        return False
    
    def is_choosen_image(self, corners, w, h) -> bool:
        xy1, xy2, xy3, xy4 = corners
        dis_x1 = abs(xy4[0] - xy1[0])
        dis_x2 = abs(xy3[0] - xy2[0])
        if abs(dis_x1 - dis_x2) > 25:
            return False
        if abs(xy4[0] - xy1[0]) < self.max_pixel_distance_2points_corner and abs(xy4[1] - xy1[1]) > h // 2 \
            and abs(xy3[0] - xy2[0]) < self.max_pixel_distance_2points_corner and abs(xy3[1] - xy2[1]) > h // 2 \
                and abs(xy2[0] - xy1[0]) > w * 0.5:
            return True
        return False

    def sort_results(self, results) -> List[Any]:
        boxes = []
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2, conf))
        boxes = nms(boxes, 0.3)
        boxes = [b[:4] for b in boxes]
        return boxes
    
    def re_box_corners(self, corners, w, h):
        const_pixel = 51
        xy1, xy2, xy3, xy4 = corners
        if xy1[0] > const_pixel:
            xy1 = np.array([const_pixel, xy1[1]], dtype=np.float32)
        if xy4[0] > const_pixel:
            xy4 = np.array([const_pixel, xy4[1]], dtype=np.float32)
        if xy2[0] < w - const_pixel:
            xy2 = np.array([w - const_pixel, xy2[1]], dtype=np.float32)
        if xy3[0] < w - const_pixel:
            xy3 = np.array([w - const_pixel, xy3[1]], dtype=np.float32)
        return [np.array(p, dtype=np.float32) for p in [xy1, xy2, xy3, xy4]]

    def rescale_images(self, cropped_images: List[np.ndarray], rotate=None) -> List[np.ndarray]:
        
        if not cropped_images:
            return []
        
        if rotate is not None:
            cropped_images = [cv2.flip(img, -1) for img in cropped_images]

        # Get minimum height
        height_scale = sum(img.shape[0] for img in cropped_images) // len(cropped_images)

        # Resize all images to match minimum height while preserving aspect ratio
        processed_frames = [cv2.resize(img, (img.shape[1], height_scale))
                            for img in cropped_images]
        if rotate is not None:
            processed_frames = [cv2.flip(img, -1) for img in processed_frames]

        return processed_frames
    
    def fix_point_corners(self, cur_corner, back_corner, w, h):
        c1, c2, c3, c4 = cur_corner
        b1, b2, b3, b4 = back_corner
        idx = self.find_outlier(back_corner, cur_corner)
        if idx < 3:
            if abs((c2[1] - c1[1]) - (b2[1] - b1[1])) > 150:
                cur_corner[idx][1] = back_corner[idx][1] 
        else:
            if abs((c3[1] - c4[1]) - (b3[1] - b4[1])) > 150:
                cur_corner[idx][1] = back_corner[idx][1]
        
    def read_video_not_process(self, video_path):
        cropped_imgs = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.count += 1
            frame = self.defish_image(frame)
            frame = frame[50:-100, 300:-300]
            img = frame.copy()
            frame = cv2.copyMakeBorder(frame,
                                   50, 50, 50, 50,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        
            h, w = frame.shape[:2]
            results = self.detect_corners.detect_corner_full_result(frame)
            corners = get_results(results, w, h)
            if len(corners) < 4:
                continue
            
            if self.is_choosen_image(results, corners, w, h):
                # corners = adjust_corners(corners, w)
                # h_min = min(c[1] for c in corners)
                cropped_imgs.append(img)
        
        cap.release()
        # cropped_imgs = self.rescale_images(new_cropped_imgs, rotate=None)
        drop_imgs, ls_idx = self.drop_imgs_with_min_len(cropped_imgs, 80)
        drop_imgs = self.rescale_images(drop_imgs, rotate=None)
        return drop_imgs
    
    def check_corners(self, last_y, y):
        if last_y is None:
            return y
        new_y = []
        for i in range(len(y)):
            if abs(y[i] - last_y[i]) > 15:
                new_y.append(last_y[i])
            else:
                new_y.append(y[i])
        return new_y

    def read_video(self, video_path):
        cropped_imgs = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        last_y_corners = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = frame.copy()
            frame = self.defish_image(frame)
            frame = frame[:, 120:-120]
            frame = cv2.copyMakeBorder(frame,
                                   50, 50, 50, 50,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
            
            h, w = frame.shape[:2]
            results = self.detect_corners.detect_corner_full_result(frame)
            corners = get_results(results, w, h)
            if len(corners) < 4:
                continue
            
            if self.is_choosen_image(corners, w, h):
                self.count += 1
                    
                y_corners = self.check_corners(last_y_corners, [c[1] for c in corners])
                last_y_corners = y_corners
                for i in range(len(y_corners)):
                    corners[i][1] = y_corners[i]
                
                # cv2.imwrite(f"/home/atin/Hoai/api_stit/img2/{self.count}_no_warp.jpg", frame)
                
                corners_adjusted = adjust_corners(corners, w)
                # if not self.is_choose_corners(corners_adjusted):
                #     continue

                frame = crop_frame(frame, corners_adjusted, 120)
                cropped_imgs.append(frame)
    
                # cv2.imwrite(f"/home/atin/Hoai/api_stit/img1/{self.count}.jpg", frame)
        
        cap.release()
        rescaled_imgs = self.rescale_images(cropped_imgs, rotate=None)
        # for i, im in enumerate(drop_imgs):
        #     cv2.imwrite(f"/mnt/atin/Hoai/api_stit/img/{i}.jpg", im)
        return rescaled_imgs

    def process_points(self, points1, points2, h, w1, w2):
        p_top_p1 = [p for p in points1 if p[1] < h//2]
        p_bottom_p1 = [p for p in points1 if p[1] >= h//2]
        p_top_p2 = [p for p in points2 if p[1] < h//2]
        p_bottom_p2 = [p for p in points2 if p[1] >= h//2]
        p_top_p1, p_top_p2 = self.filter_points(p_top_p1, p_top_p2, h, w1, w2)
        p_bottom_p1, p_bottom_p2 = self.filter_points(p_bottom_p1, p_bottom_p2, h, w1, w2)
        if not len(p_top_p1) or not len(p_top_p2) or not len(p_bottom_p1) or not len(p_bottom_p2):
            return [], []
        return [p_top_p1, p_bottom_p1], [p_top_p2, p_bottom_p2]

    def filter_points(self, p1, p2, h, w1, w2):
        if len(p1) > 1 and len(p2) > 1:
            min_h_p1 = min(p[1] for p in p1)
            max_h_p1 = max(p[1] for p in p1)
            min_h_p2 = min(p[1] for p in p2)
            max_h_p2 = max(p[1] for p in p2)
            if abs(min_h_p1 - min_h_p2) < abs(max_h_p1 - max_h_p2):
                return [p1[0][0], min_h_p1], [p2[0][0], min_h_p2]
            else:
                return [p1[0][0], max_h_p1], [p2[0][0], max_h_p2]
        elif len(p1) > 1 and len(p2) == 1:
            min_h_p1 = min(p[1] for p in p1)
            max_h_p1 = max(p[1] for p in p1)
            if abs(min_h_p1 - p2[0][1]) < abs(max_h_p1 - p2[0][1]):
                return [p1[0][0], min_h_p1], p2[0]
            else:
                return [p1[0][0], max_h_p1], p2[0]
        elif len(p1) == 1 and len(p2) > 1:
            min_h_p2 = min(p[1] for p in p2)
            max_h_p2 = max(p[1] for p in p2)
            if abs(p1[0][1] - min_h_p2) < abs(p1[0][1] - max_h_p2):
                return p1[0], [p2[0][0], min_h_p2]
            else:
                return p1[0], [p2[0][0], max_h_p2]
        elif len(p1) == 1 and len(p2) == 0:
            return p1[0], [0, p1[0][1]]
        elif len(p1) == 0 and len(p2) == 1:
            return [w1, p2[0][1]], p2[0]
        elif len(p1) > 1 and len(p2) == 0:
            min_h_p1 = min(p[1] for p in p1)
            return [p1[0][0], min_h_p1], [0, min_h_p1]
        elif len(p1) == 0 and len(p2) > 1:
            min_h_p2 = min(p[1] for p in p2)
            return [w1, min_h_p2], [p2[0][0], min_h_p2]
        elif len(p1) == 0 and len(p2) == 0:
            return [], []
        return p1[0], p2[0]

    def transform_2_imgs(self, w1, img2, pts1, pts2):
        h2, w2 = img2.shape[:2]
        pts1 = [p for p in pts1 if p[0] < 5]
        pts2 = [p for p in pts2 if p[0] > w2 - 5]

        pts1, pts2 = self.process_points(pts1, pts2, h2, w1, w2)
        if len(pts1) < 2 or len(pts2) < 2:
            return img2
        
        pts1 = np.float32([pts1[0], pts1[1]])
        pts2 = np.float32([pts2[0], pts2[1]])
        
        dy1 = max(0, pts1[0][1] - pts2[0][1])
        dy2 = h2 - (pts1[1][1] - pts2[1][1])
        if dy2 > 15:
            return img2
        
        # cv2.circle(img2, (5, dy2), 3, (0, 255, 0), -1)
        # cv2.circle(img2, (5, dy1), 3, (0, 255, 0), -1)
        # cv2.imwrite(f"/mnt/atin/Hoai/api_stit/img_sobel/dy2.jpg", img2)
        
        old_points2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
        new_points2 =  np.array([[0, dy1], [w2, dy1], [w2, dy2], [0, dy2]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(old_points2, new_points2)

        # Apply the perspective transformation
        warped_img2 = cv2.warpPerspective(img2, M, (w2, h2))
        
        # warped_img2 = cv2.resize(warped_img2, (w2, h2))
        # cv2.imwrite(f"/mnt/atin/Hoai/api_stit/img_sobel/warp.jpg", warped_img2)
        return warped_img2
        
    def process_img_stit(self, img2, img1):
        img_2 = cv2.copyMakeBorder(img2,
                                   0, 0, 0, 50,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        img_1 = cv2.copyMakeBorder(img1,
                                   0, 0, 50, 0,
                                   borderType=cv2.BORDER_CONSTANT,
                                   value=(0, 0, 0))
        
        re2 = self.detect_corners.detect_corner_full_result(img_2)
        re1 = self.detect_corners.detect_corner_full_result(img_1)
        re2 = self.sort_results(re2)
        re1 = self.sort_results(re1)

        box2, box1 = [get_center(x1, y1, x2, y2) for [x1, y1, x2, y2] in re2], [get_center(x1, y1, x2, y2) for [x1, y1, x2, y2] in re1]
        
        h1, w1 = img_1.shape[:2]
        h2, w2 = img_2.shape[:2]
        
        box1 = [[0, cy] for [cx, cy] in box1]
        box2 = [[w2-50, cy] for [cx, cy] in box2]

        img = self.transform_2_imgs(w1-50, img_2[:, :w2-50], box1, box2)
        print("done transform")
        print("img1", img1.shape, "img2", img.shape)

        return img
        