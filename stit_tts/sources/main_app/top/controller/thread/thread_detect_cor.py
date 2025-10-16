import torch
import cv2
from ultralytics import YOLO
from sources.config import CORNERS_TOP_CONFIG, ROOT
from PyQt5.QtCore import QThread
from sources.utils.split_2cont import x_split_2cont, get_gucors_top


class DetectCor(QThread):
    def __init__(self):
        super().__init__()

        self.load_config(CORNERS_TOP_CONFIG)
       
    def setup_model(self):
        self.model_detect_cor = YOLO(model=self.weight, verbose=False).to(device=self.device) 
        print("Done load", self.weight)
            
    def load_config(self, config):
        w = config["weight"]
        self.weight = f"{ROOT}/{w}"
        self.imgsz = config["imgsz"]
        self.conf = config["conf"]
        self.device = config["device"]
    
    def detect_cor(self, img):
        return self.model_detect_cor(img, conf=self.conf, device=self.device, verbose=False)
    
    def coordinate_split_2cont(self, image):
        height, width, _ = image.shape

        img_mid = image[:, int(width*0.35):int(width*0.7)]
        results = self.detect_cor(img_mid)
        gu_cors = get_gucors_top(results)
        if len(gu_cors) > 1:
            h1 = img_mid.shape[0]
            x_split = x_split_2cont(gu_cors, h1) 
            if x_split:
                x_split += int(width*0.35)
                return x_split

        img_left = image[:, 100:int(width*0.6)]
        results = self.detect_cor(img_left)
        gu_cors = get_gucors_top(results)
        if len(gu_cors) > 1:
            h2 = img_left.shape[0]
            x_split = x_split_2cont(gu_cors, h2)
            if x_split:
                x_split += 100
                return x_split  

        img_right = image[:, int(width*0.4):width-100]
        results = self.detect_cor(img_right)
        gu_cors = get_gucors_top(results)
        if len(gu_cors) > 1:
            h3 = img_right.shape[0]
            x_split = x_split_2cont(gu_cors, h3) 
            if x_split:
                x_split += int(width*0.4)
                return x_split
        
        return 0
    
    
        
        