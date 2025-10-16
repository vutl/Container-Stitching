from sources.superglue.matcher import Matcher
from sources.utils.matching import *
from sources.main_app.side.controller.module.stit_22f import Stit22f
from sources.main_app.side.controller.module.stit_4xf import Stit4xf
from sources.main_app.side.controller.module.stit_glue import StitGlue
from sources.main_app.side.controller.module.split_cont import SplitContainer
from sources.main_app.side.controller.module.stit_video import StitVideoGlue
from sources.utils.tools import get_ratio_img
from sources.config import DROP, DEVICE
import time


class StitSide():
    def __init__(self):

        self.max_weight_match_superglue = 1980

        self.ratio_22f = 1.7  # tỉ lệ chuẩn khi ghép container 22 feet 2.4
        self.ratio_42f = 3.0  # tỉ lệ chuẩn khi ghép container 40, 42 feet 4.2
        self.ratio_45f = 3.0  # tỉ lệ chuẩn khi ghép container 45 feet 4.4
        self.ratio_12f = 2.0  # tỉ lệ chuẩn khi ghép container 12 feet 4.4

        self.load_super_glue_default()
        self.load_super_glue_low_accuracy()

        self.default_ratio = 1.5
        self.max_default_ratio = 6.0

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
                    # "keypoint_encoder": [32, 64, 128, 256],
                    # "GNN_layers": ["self", "cross"] * 9,
                    "sinkhorn_iterations": 120,
                    "match_threshold": match_threshold
                },
                "device": DEVICE,
            }
        )

    def load_super_glue_low_accuracy(self, keypoint_threshold=0.08, match_threshold=0.72):
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
                    # "keypoint_encoder": [32, 64, 128, 256],
                    # "GNN_layers": ["self", "cross"] * 9,
                    "sinkhorn_iterations": 75,
                    "match_threshold": match_threshold
                },
                "device": DEVICE,
            }
        )

    def setup_module(self):
        # self.stitcher_4xf = Stit4xf(
        #     self.superglue_matcher_default, self.superglue_matcher_low_accuracy)
        self.stitcher_glue = StitGlue(
            self.superglue_matcher_default, self.superglue_matcher_low_accuracy)
        self.stitcher_22f = Stit22f()
        self.split_container = SplitContainer()
        self.stit_video_glue = StitVideoGlue(
            self.superglue_matcher_default, self.superglue_matcher_low_accuracy)

    def check_ratio_img_stit(self, ratio, cont_size) -> bool:
        if cont_size == 22:
            if self.ratio_22f < ratio < (self.ratio_22f * 2.5):
                return True
        if cont_size == 42:
            if self.ratio_42f < ratio < (self.ratio_42f * 2.0):
                return True
        if cont_size == 45:
            if self.ratio_45f < ratio < (self.ratio_45f * 2.0):
                return True
        if cont_size == 12:
            if self.ratio_12f < ratio < (self.ratio_12f * 3.0):
                return True
        return False

    def check_default_ratio(self, w, h):
        return self.default_ratio < get_ratio_img(w, h) < self.max_default_ratio

    def drop_imgs(self, ls_imgs, min_len):
        ls_drop_imgs = []
        if len(ls_imgs) < min_len:
            return ls_imgs
        for i in range(min_len):
            idx = round(i * (len(ls_imgs) - 1) / (min_len - 1))
            ls_drop_imgs.append(ls_imgs[idx])
        return ls_drop_imgs

    def stitching(self, ls_drop_imgs, ls_full_imgs, cont_size=22):

        if len(ls_drop_imgs):
            print(len(ls_drop_imgs), "len drop imgs")
            img_glue = self.stitcher_glue.stit_glue(ls_drop_imgs)
            img_glue = img_glue[:-110,:]
            h, w = img_glue.shape[:2]
            ratio = get_ratio_img(w, h)

            print(ratio, "ratio")
            if self.check_ratio_img_stit(ratio, cont_size):
                if cont_size == 12:
                    return self.split_container.cut_2cont(img_glue)
                return [img_glue]

            if cont_size == 22:
                img22 = self.stitcher_22f.stit_blackbox_22f(ls_full_imgs)
                h1, w1 = img22.shape[:2]
                if self.check_default_ratio(w1, h1):
                    return [img22]
        return [img_glue]
    
    def stitching_with_video(self, video_path, cont_size=42, side='left'):
        image_stitched = self.stit_video_glue.stit_with_video(video_path, cont_size=cont_size, side=side)
        if image_stitched is None:
            return None
        if cont_size == 12:
            return self.split_container.cut_2cont(image_stitched)
        print(image_stitched.shape)
        return [image_stitched]
    
