from ultralytics import YOLO
from sources.config import LINE_CONFIG, ROOT
from PyQt5.QtCore import QThread


class DetectLines(QThread):
    def __init__(self):
        super().__init__()
        
        self.load_config(LINE_CONFIG)
    
    def load_config(self, config):
        w = config["weight"]
        self.weight = f"{ROOT}/{w}"
        self.imgsz = config["imgsz"]
        self.conf = config["conf"]
        self.device = config["device"]
        
    def setup_models(self):
        print("Done load", self.weight)
        self.model_lines = YOLO(model=self.weight)
        self.model_lines.to(device=self.device) 

    def detect_line(self, img):
        return self.model_lines.predict(img, imgsz=self.imgsz, conf=self.conf, verbose=False)  
    
    def get_necessary_imgs(self, list_imgs: list):
        r'''
            Get necessary images because exist very much background 
            in first and last list frame
            return list_imgs[first_start:first_end]
        '''
        first_id, last_id = 0, 0
        num_standard = 50 # number of frame to get step
        
        # find first id
        for i, img in enumerate(list_imgs):
            if i % 3 != 0:
                continue
            img = img[200:-200, 100:-100]
            # detect gu container to get first_id
            results = self.detect_line(img)
            if len(results[0].boxes)>0:
                first_id = i
                print(f'first_id: {first_id}')
                break
                
        # find last id
        for i, img in enumerate(list_imgs[::-1]):
            if i % 3 != 0:
                continue
            img = img[200:-200, 100:-100]
            # detect gu container to get last_id
            results = self.detect_line(img)
            if len(results[0].boxes)>0:
                last_id = len(list_imgs) - i
                print(f'last_id: {last_id}')
                break
        
        # get step for list frame
        step = max(1,(last_id - first_id) // num_standard)
        list_imgs = list_imgs[first_id:last_id:step]
        print(f'first_id: {first_id}, last_id: {last_id}, step: {step}', 'len list_imgs', len(list_imgs))
        return list_imgs, first_id, last_id

    