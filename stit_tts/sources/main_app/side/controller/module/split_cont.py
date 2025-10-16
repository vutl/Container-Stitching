import time
from sources.main_app.side.controller.thread.thread_detect_corner import DetectCorners

class SplitContainer:
    def __init__(self):
        self.add_pixel = 10
        self.detect_corner = DetectCorners()

    def cut_2cont(self, img):
        h, w = img.shape[:2]
        t = time.time()
        coord_x_xplit = self.detect_corner.coordinate_split_2cont(img)
        print("Time Split Container:", time.time() - t)
        if coord_x_xplit:
            img1 = img[:, :coord_x_xplit+self.add_pixel]
            img2 = img[:, coord_x_xplit-self.add_pixel:]
            print("Done cut 2 container")
            return [img1, img2]
        print("Fail cut 2 container")
        return [img[:, 10:w-10]]  # crop 20 pixel from left and right of image
