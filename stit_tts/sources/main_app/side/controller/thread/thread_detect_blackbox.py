from ultralytics import YOLO
from PyQt5.QtCore import QThread
from sources.config import ROOT, BLACKBOX_CONFIG
from sources.main_app.side.utils.tool_side import nms


class DetectBlackBox(QThread):
    def __init__(self):
        super().__init__()
        
        self.setup_model()
    
    def load_config(self, config):
        w = config["weight"]
        self.weight = f"{ROOT}/{w}"
        self.imgsz = config["imgsz"]
        self.conf = config["conf"]
        self.device = config["device"]

    def setup_model(self):
        self.load_config(BLACKBOX_CONFIG)
        print("Done load", self.weight)
        self.model_blackbox = YOLO(model=self.weight, verbose=False)
        self.model_blackbox.to(device=self.device)
         
    def detect_blackbox(self, img):
        bboxs = []
        results = self.model_blackbox.predict(img, conf=self.conf, imgsz=self.imgsz, verbose=False)
        
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])            
            bboxs.append((x1, y1, x2, y2, r.conf[0]))    
                
        bboxs = nms(bboxs, 0.2)
        return sorted(bboxs, key=lambda x: x[0])
    
    def get_nessesary_imgs(self, imgs):
        num_limit = len(imgs)//5
        first_id = 0
        last_id = -1
        
        for i,img in enumerate(imgs[:num_limit]):
            bbox = self.detect_blackbox(img)
            if len(bbox) == 0:
                continue
            print('bbox', bbox[0][2], img.shape[1])
            
            if bbox[0][0] > 5:
                first_id = i-1
                break
            else:
                first_id = 1
        
        print('--------------------------')
                
        for i,img in enumerate(imgs[-num_limit:]):
            bbox = self.detect_blackbox(img)
            if len(bbox) == 0:
                continue
            print('bbox', bbox[0][0])
            if bbox[0][2] < img.shape[1]-5:
                last_id = min(-i, -1)
                break
            else:
                last_id = -2
        
        print('first_id', first_id, 'last_id', last_id)
        return imgs[first_id:last_id]    
    
