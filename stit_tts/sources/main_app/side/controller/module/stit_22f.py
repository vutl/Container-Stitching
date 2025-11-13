import cv2
import numpy as np
from sources.main_app.side.controller.thread.thread_detect_blackbox import DetectBlackBox


class Stit22f():
    def __init__(self):
        self.detect_blackbox = DetectBlackBox()

    def setup_model(self):
        self.detect_blackbox.setup_model()

    def stit_blackbox_22f(self, ls_imgs):
        # get necessary images
        imgs = self.detect_blackbox.get_nessesary_imgs(ls_imgs)

        # get blackbox
        img0, img4 = imgs[0], imgs[-1]
        img1 = imgs[len(imgs)//6-1]
        img2 = imgs[len(imgs)//2-1]
        img3 = imgs[-len(imgs)//6]

        bbox0 = self.detect_blackbox.detect_blackbox(img0)
        bbox1 = self.detect_blackbox.detect_blackbox(img1)
        bbox2 = self.detect_blackbox.detect_blackbox(img2)
        bbox3 = self.detect_blackbox.detect_blackbox(img3)
        bbox4 = self.detect_blackbox.detect_blackbox(img4)

        # fix bbox2 if nessessary
        if len(bbox2) == 1:
            if bbox2[0][0] < img2.shape[1]/2:
                img2 = imgs[len(imgs)//2-2]
            else:
                img2 = imgs[len(imgs)//2]
            bbox2 = self.detect_blackbox.detect_blackbox(img2)

        extra_pixel = 5
        img0 = img0[:, bbox0[0][2]-extra_pixel:]
        img1 = img1[:, bbox1[0][0]-extra_pixel:bbox1[0][2]+extra_pixel]
        img2 = img2[:, bbox2[0][2]-extra_pixel:bbox2[1][0]+extra_pixel]
        img3 = img3[:, bbox3[0][0]-extra_pixel:bbox3[0][2]+extra_pixel]
        img4 = img4[:, :bbox4[0][0]+extra_pixel]

        return np.concatenate([img4, img3, img2, img1, img0], axis=1)
