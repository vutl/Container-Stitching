from pydantic import BaseModel
# from sources.main_app.main_controller import Controller
from sources.main_app.side.controller.stit_side import StitSide
from sources.main_app.top.controller.stit_top import StitTop
from sources.utils.util import read_imgs, sort_name_in_folder, read_imgs_top
from sources.main_app.models.side import Side
from sources.main_app.models.top import Top
from sources.config import SAVE_ROOT, ROOT_STORAGE
from sources.utils.tools import get_ratio_img
import time
import traceback
import cv2


class APIData(BaseModel):
    folderPath: str  # folder images
    contSize: int  #
    

class SplitData(BaseModel):
    imagePath: str  # image path
    

class APIVideoData(BaseModel):
    videoPath: str  # video path
    contSize: int  # container size
    side: str # mặt container, 'left' or 'right' or 'top


class APIController:
    def __init__(self):
        self.stitch_side = StitSide()
        self.stitch_top = StitTop()
        self.save_side = Side()
        self.save_top = Top()

    def setup_controller(self):
        self.stitch_side.setup_module()
        self.stitch_top.setup_module()

    async def stitching_side(self, data: APIData):
        folder_name = data.folderPath
        cont_size = data.contSize
        print(folder_name, cont_size)
        try:
            t = time.time()
            ls_drop_imgs, ls_full_imgs = read_imgs(
                folder_images=folder_name, cont_size=cont_size)

            print("len imgs:", len(ls_drop_imgs), len(ls_full_imgs))
            print(time.time() - t)
            imgs_stit = self.stitch_side.stitching(
                ls_drop_imgs=ls_drop_imgs, ls_full_imgs=ls_full_imgs, cont_size=cont_size)
            ls_drop_imgs, ls_full_imgs = None, None
            if len(imgs_stit) == 1:
                save_path1 = self.save_side.get_save_path(
                    root=f"{ROOT_STORAGE}/{folder_name}", stt=1)
                self.save_side.save_image(imgs_stit[0], save_path1)
                return self.save_side.format_side(save_path1=save_path1, save_path2="", status=200)
            if len(imgs_stit) == 2:
                save_path1 = self.save_side.get_save_path(
                    root=f"{ROOT_STORAGE}/{folder_name}", stt=1)
                save_path2 = self.save_side.get_save_path(
                    root=f"{ROOT_STORAGE}/{folder_name}", stt=2)
                self.save_side.save_image(imgs_stit[0], save_path1)
                self.save_side.save_image(imgs_stit[1], save_path2)
                return self.save_side.format_side(save_path1=save_path1, save_path2=save_path2, status=200)
            return self.save_side.format_side(save_path1="", save_path2="", status=400)
        except Exception as e:
            print(e, "***stitching_side***")
            print(traceback.format_exc())
            return self.save_side.format_side(save_path1="", save_path2="", status=400)

    async def stitching_top(self, data: APIData):
        folder_name = data.folderPath
        cont_size = data.contSize
        print(folder_name, cont_size)
        save_path1 = ""
        save_path2 = ""
        try:
            t = time.time()
            ls_imgs = read_imgs_top(folder_images=folder_name)
            print("time read imgs top:", time.time() - t)
            imgs_stit = self.stitch_top.stitching(
                ls_imgs=ls_imgs, cont_size=cont_size)
            ls_imgs = None
            if imgs_stit is not None and len(imgs_stit):
                h1, w1 = imgs_stit[0].shape[:2]
                print("len imgs stit:", len(imgs_stit), "h1:", h1, "w1:", w1)
                if get_ratio_img(w1, h1) > 1.5:
                    save_path1 = self.save_top.get_save_path(
                        root=f"{ROOT_STORAGE}/{folder_name}", stt=1)
                    self.save_top.save_image(imgs_stit[0], save_path1)

                if len(imgs_stit) == 2:
                    h2, w2 = imgs_stit[1].shape[:2]
                    if get_ratio_img(w2, h2) > 1.5:
                        save_path2 = self.save_top.get_save_path(
                            root=f"{ROOT_STORAGE}/{folder_name}", stt=2)
                        self.save_top.save_image(imgs_stit[1], save_path2)
                        return self.save_top.format_top(save_path1=save_path1, save_path2=save_path2, status=200)

                if len(save_path1):
                    return self.save_top.format_top(save_path1=save_path1, save_path2="", status=200)
            return self.save_top.format_top(save_path1="", save_path2="", status=400)

        except Exception as e:
            print(e, "***stitching_top***")
            print(traceback.format_exc())
            return self.save_top.format_top(save_path1="", save_path2="", status=400)
        
    async def stitching_side_with_video(self, data: APIVideoData):
        video_path = f"{ROOT_STORAGE}/{data.videoPath}"
        cont_size = data.contSize
        side = data.side
        __path = data.videoPath.split('.')[0]
        print(video_path, cont_size, side)
        try:
            t = time.time()
            imgs_stit = self.stitch_side.stitching_with_video(video_path, cont_size=cont_size, side=side)
            if imgs_stit is not None and len(imgs_stit):
                save_path1 = self.save_side.get_save_path(
                    root=f"{SAVE_ROOT}/{__path}", stt=1)
                self.save_side.save_image(imgs_stit[0], save_path1)
                if len(imgs_stit) == 2:
                    save_path2 = self.save_side.get_save_path(
                        root=f"{SAVE_ROOT}/{__path}", stt=2)
                    self.save_side.save_image(imgs_stit[1], save_path2)
                    return self.save_side.format_side(save_path1=save_path1, save_path2=save_path2, status=200)
                return self.save_side.format_side(save_path1=save_path1, save_path2="", status=200)
            return self.save_side.format_side(save_path1="", save_path2="", status=400)
        except Exception as e:
            print(e, "***stitching_side_with_video***")
            return self.save_side.format_side(save_path1="", save_path2="", status=400)

    async def image_split_container_side(self, data: SplitData):
        __path = data.imagePath
        image_path = f"{ROOT_STORAGE}/{__path}"
        print(image_path)
        try:
            t = time.time()
            img = cv2.imread(image_path)
            imgs_split = self.stitch_side.split_container.cut_2cont(img)
            print("time split container side:", time.time() - t)
            if len(imgs_split) == 2:
                save_path1 = image_path
                self.save_side.save_image(imgs_split[0], save_path1)
                save_path2 = save_path1.replace("_1.jpg", "_2.jpg")
                self.save_side.save_image(imgs_split[1], save_path2)
                return {"status": 200}
            return {"status": 400}
        except Exception as e:
            print(e, "***split_container***")
            print(traceback.format_exc())
            return {"status": 400}
    
    
    async def image_split_container_top(self, data: SplitData):
        __path = data.imagePath
        image_path = f"{ROOT_STORAGE}/{__path}"
        print(image_path)
        try:
            t = time.time()
            img = cv2.imread(image_path)
            imgs_split = self.stitch_top.split_container.cut_2cont(img)
            print("time split container top:", time.time() - t)
            if len(imgs_split) == 2:
                save_path1 = image_path
                self.save_top.save_image(imgs_split[0], save_path1)
                save_path2 = save_path1.replace("_1.jpg", "_2.jpg")
                self.save_top.save_image(imgs_split[1], save_path2)
                return {"status": 200}
            return {"status": 400}
        except Exception as e:
            print(e, "***split_container***")
            print(traceback.format_exc())
            return {"status": 400}
