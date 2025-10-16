from ultralytics import YOLO
from PyQt5.QtCore import QThread
from sources.config import ROOT, DENT_CONFIG, MAX_PIXEL_DISTANCE
from sources.utils.tools import nms


class DetectDent(QThread):
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
        self.load_config(DENT_CONFIG)
        print("Done load", self.weight)
        self.model_dent = YOLO(model=self.weight, verbose=False)
        self.model_dent.to(device=self.device)

    def detect_dents(self, img):
        dents = []
        results = self.model_dent.predict(
            img, imgsz=self.imgsz, conf=self.conf, verbose=False)

        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            if x2-x1 < 10:
                continue
            if y1 < img.shape[0]//2:
                dents.append((x1, y1, x2, y2, r.conf[0]))

        dents = nms(dents, 0.2)
        dents = sorted(dents, key=lambda x: x[0])

        return dents

    def get_gucors(self, img):
        # results = self.model_detect_cor(img, conf=CONF_CORNERS, verbose=False)
        results = self.model_dent(img, conf=self.conf, verbose=False)

        gu_cors = []
        for r in results[0].boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0])
            class_id = int(r.cls[0])
            if class_id == 1:
                gu_cors.append((x1, y1, x2, y2, r.conf[0]))

        gu_cors = nms(gu_cors, 0.2)
        gu_cors = sorted(gu_cors, key=lambda x: x[0])

        return gu_cors

    def get_nessessary_imgs(self, imgs):
        num_limit = int(len(imgs)*0.1+4)

        first_id = 1
        for i, img in enumerate(imgs[0:num_limit]):
            gu_cors = self.get_gucors(imgs[i])
            if len(gu_cors) == 0:
                last_id = min(-i+1, -1)
                break

            print('step', i, img.shape[1]-gu_cors[0][2])
            if gu_cors[0][2] > img.shape[1]-7:
                first_id = max(i-1, 1)
                break

        last_id = len(imgs)-2
        for i, img in enumerate(imgs[-num_limit:-1]):
            gu_cors = self.get_gucors(imgs[-i-1])
            if len(gu_cors) == 0:
                last_id = min(-i+1, -1)
                break
            print('step', i, gu_cors[0][0])
            if gu_cors[0][0] < 7:
                last_id = min(-i+1, -1)
                break

        print('choose frame---->', first_id, last_id)
        return imgs[first_id:last_id]

    def get_params(self, imgs, cont_size):
        # get dents from first frame
        dents = self.detect_dents(imgs[0])

        l_dent = sum(x2-x1 for x1, y1, x2, y2, _ in dents)/len(dents)
        # stepdent = (cont_size-1-(len(dents)-1+first_0))/(len(imgs)-1)
        stepdent = (cont_size-1-(imgs[0].shape[1]/l_dent))/(len(imgs)-1)

        # get dents from middle frame
        dents = self.detect_dents(imgs[7])
        l_dent = sum(x2-x1 for x1, y1, x2, y2,
                     _ in dents[1:-1])/len(dents[1:-1])

        return l_dent, stepdent

    def get_stit_dents(self, stit_imgs):
        list_stit_dents = []
        for i, img in enumerate(stit_imgs):
            dents = self.detect_dents(img)

            if i == len(stit_imgs)-1:
                list_stit_dents.append(dents)
            else:
                list_stit_dents.append((dents[0], dents[-1]))

        return list_stit_dents
