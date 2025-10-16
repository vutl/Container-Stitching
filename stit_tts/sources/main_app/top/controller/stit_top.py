import time
import numpy as np
from sources.superglue.matcher import Matcher
from sources.main_app.top.controller.thread.thread_detect_lines import DetectLines
from sources.main_app.top.controller.thread.thread_detect_cor import DetectCor
from sources.main_app.top.controller.module.stit_glue import StitGlue
from sources.main_app.top.controller.module.cropper import Cropper
from sources.main_app.top.controller.module.split_cont import SplitContainer
from sources.main_app.top.controller.module.process_img import ProcessImg
from sources.main_app.top.utils.helpper import stitched_status
from sources.config import DEVICE


class StitTop():
    def __init__(self):
        
        self.model_detect_lines = DetectLines()
        self.model_detect_cor = DetectCor()
        self.load_super_glue_default()
        self.load_super_glue_low_accuracy()
        self.model_detect_lines.setup_models()
        self.model_detect_cor.setup_model()
    
    def load_super_glue_default(self, keypoint_threshold=0.1, match_threshold=0.81):
        self.superglue_matcher_default = Matcher(
            {
                "superpoint": {
                    "descriptor_dim": 256,
                    "nms_radius": 4,
                    "keypoint_threshold": keypoint_threshold,
                    "max_keypoints": 512,
                    "remove_borders": 5,
                    "input_shape": (-1, -1),
                },
                "superglue": {
                    "descriptor_dim": 256,
                    "sinkhorn_iterations": 120,
                    "match_threshold": match_threshold
                },
                "device": DEVICE,
            }
        )
        
    def load_super_glue_low_accuracy(self, keypoint_threshold=0.02, match_threshold=0.15):
        self.superglue_matcher_low_accuracy = Matcher(
            {
                "superpoint": {
                    "descriptor_dim": 256,
                    "nms_radius": 4,
                    "keypoint_threshold": keypoint_threshold,
                    "max_keypoints": 512,
                    "remove_borders": 5,
                    "input_shape": (-1, -1),
                },
                "superglue": {
                    "descriptor_dim": 256,
                    "sinkhorn_iterations": 75,
                    "match_threshold": match_threshold
                },
                "device": DEVICE,
            }
        )
        
    def setup_module(self):
        self.stit_glue = StitGlue(self.superglue_matcher_default, self.superglue_matcher_low_accuracy, self.model_detect_lines)
        self.cropper = Cropper(self.model_detect_cor)
        self.split_container = SplitContainer(self.model_detect_cor)
        self.process_img = ProcessImg(self.cropper, self.model_detect_lines)
    
    def drop_imgs(self, ls_imgs, cont_size):
        if cont_size == 22:
            min_len = 22
        else:
            min_len = 46
        ls_drop_imgs = []
        if len(ls_imgs) < min_len:
            return ls_imgs
        for i in range(min_len):
            idx = round(i * (len(ls_imgs) - 1) / (min_len - 1))
            ls_drop_imgs.append(ls_imgs[idx])
        return ls_drop_imgs
            
    def stit_image(self, ls_imgs, cont_size):
        ls_stit_imgs = []
        t = time.time()
        if cont_size == 45:
            ls_process_imgs, first_id, last_id = self.process_img.process_lines_L(ls_imgs=ls_imgs)
        else:
            ls_process_imgs, first_id, last_id = self.process_img.process_lines(ls_imgs=ls_imgs)
        print("time detect lines:", time.time() - t)
        if stitched_status(len_ls_0=len(ls_process_imgs[0]), len_ls_1=len(ls_process_imgs[1]), cont_size=cont_size):
            for ls_imgs in ls_process_imgs:
                if not len(ls_imgs):
                    continue
                imgs = ls_imgs[::-1]
                ls_stit_imgs.append(np.concatenate(imgs, axis=1))
            return ls_stit_imgs
            
        t = time.time()
        ls_imgs_not_process = ls_imgs[first_id:last_id]
        print(len(ls_imgs_not_process), "imgs not process")
        ls_pro_imgs, ls_corners = self.cropper.corners_process(list_imgs=ls_imgs_not_process)
        print("time cropper corners:", time.time() - t)
        ls_drop_imgs = self.drop_imgs(ls_pro_imgs, cont_size)
        img_stit_glue = self.stit_glue.stit_glue(processed_imgs=ls_drop_imgs, ls_corners=ls_corners)
        if cont_size == 12:
            return self.split_container.cut_2cont(img_stit_glue)
        return [img_stit_glue]
        
    def stitching(self, ls_imgs, cont_size):
        if len(ls_imgs): 
            return self.stit_image(ls_imgs, cont_size)
        else:
            return None
