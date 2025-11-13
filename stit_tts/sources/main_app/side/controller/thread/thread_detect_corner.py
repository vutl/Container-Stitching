from ultralytics import YOLO
from PyQt5.QtCore import QThread
from sources.config import ROOT, CORNERS_CONFIG
from sources.utils.split_2cont import x_split_2cont, get_gucors_side


class DetectCorners(QThread):
    def __init__(self):
        super().__init__()
        
        self.add_pixel = 10
        
        self.setup_model()
    
    def load_config(self, config):
        w = config["weight"]
        self.weight = f"{ROOT}/{w}"
        self.imgsz = config["imgsz"]
        self.conf = config["conf"]
        self.device = config["device"]
    
    def setup_model(self):
        self.load_config(CORNERS_CONFIG)
        print("Done load", self.weight)
        self.model_corner = YOLO(model=self.weight, verbose=False)
        self.model_corner.to(device=self.device)
    
    def detect_corner(self, img):
        results = self.model_corner.predict(img, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False)
        return results[0].boxes
    
    def coordinate_split_2cont(self, image):
        height, width, _ = image.shape

        img_mid = image[:, int(width*0.35):int(width*0.7)]
        results = self.detect_corner(img_mid)
        gu_cors = get_gucors_side(results)
        if len(gu_cors) > 1:
            h1 = img_mid.shape[0]
            x_split = x_split_2cont(gu_cors, h1) 
            if x_split:
                x_split += int(width*0.35)
                return x_split

        img_left = image[:, 100:int(width*0.6)]
        results = self.detect_corner(img_left)
        gu_cors = get_gucors_side(results)
        if len(gu_cors) > 1:
            h2 = img_left.shape[0]
            x_split = x_split_2cont(gu_cors, h2)
            if x_split:
                x_split += 100
                return x_split  

        img_right = image[:, int(width*0.4):width-100]
        results = self.detect_corner(img_right)
        gu_cors = get_gucors_side(results)
        if len(gu_cors) > 1:
            h3 = img_right.shape[0]
            x_split = x_split_2cont(gu_cors, h3) 
            if x_split:
                x_split += int(width*0.4)
                return x_split
        
        return 0
        
        