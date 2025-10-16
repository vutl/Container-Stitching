import os
import cv2
from numpy import ndarray
from datetime import datetime


class Top:
    image : ndarray = None
    status : int = None
        
    def create_time(self):
        self.date_created = datetime.now().strftime("%Y-%m-%d")
        self.time_created = datetime.now().strftime("%H-%M-%S")
    
    def get_save_path(self, root, stt=1):
        # self.create_time()
        # save_root = os.path.join(root, "side", str(self.date_created), "images")
        # try:
        #     os.makedirs(save_root, exist_ok=True)
        # except Exception as e:
        #     print(e)
        
        save_path = os.path.join(root, f"stitch_img_{stt}.jpg")
        
        return save_path

    def save_image(self, image, save_path):
        cv2.imwrite(save_path, image)
        print("Saved image to: ", save_path)
    
    def format_top(self, save_path1, save_path2, status):
        return {
            "imagePath1" : str(save_path1),
            "imagePath2" : str(save_path2),
            "status" : int(status)
        }
            