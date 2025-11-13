import cv2
from typing import List
from sources.main_app.top.utils.tool_top import *
from sources.main_app.top.utils.helpper import nms
from sources.main_app.top.controller.module.cropper import Cropper
from sources.main_app.top.controller.thread.thread_detect_lines import DetectLines


class ProcessImg():
    def __init__(self, cropper: Cropper, model_detect_lines: DetectLines):
        self.cropper = cropper
        self.detect_line = model_detect_lines
        
        self.crop_left = 80
        self.crop_right = 80
    
    def mean_ls_id(self, ls_id: list) -> float:
        ls_diff_2id = [ls_id[i+1]-ls_id[i] for i in range(len(ls_id)-1)]
        return sum(ls_diff_2id)/len(ls_diff_2id) 

    def process_lines(self, ls_imgs: List) -> List:
        ls_imgs, first_id, last_id = self.detect_line.get_necessary_imgs(ls_imgs)
        
        list_process_imgs = []
        data_bigcor = [] # save temporary big corners and will be used if necessary
        num_last_3lines = 0 # number of 3 lines in last frame
        num_split = 0 # number of 2 container
        h_above = 0 # auto scale height crop abvove image
        h_under = 0 # auto scale height crop under image
        w_right = 0 # auto scale width crop right image
        last_lls = [] # this list is used to store last longlines. It helps to analyze box between 2 frames
        list_id_pick = [] # this list is used to store id of pick container
        flag = False # A flag to check if there is a container truck apear in this frame
        
        id_start = 0 # id of first frame
        
        for i, im in enumerate(ls_imgs):
            '''
            📌📌📌
            The first is the conditions for ignoring unnecessary images, including: 
            - Do not take 2 images in a row
            - Some passing if not appear head of container
            - Return early if there is a container truck tail
            => This is help to reduce time process and inference
            '''
            if not flag and i % 3:
                id_start = i
                continue

            # 🔧🔧🔧 Preprocesing image before predict🔧🔧🔧 
            # img = cv2.imread(img_path)
            img = im[h_above:im.shape[0]-h_under, self.crop_left:im.shape[1]-self.crop_right-w_right]
            height, width, _ = img.shape
            
            # 🔥 🔥 🔥 Let predict and process results 🚀🚀🚀
            results = self.detect_line.detect_line(img)
            corners=[]
            longlines=[]
            big_corners=[]
            
            for r in results[0].boxes:
                box = r.xyxy[0]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                if y2-y1 > 700: # height of longline usually > 700
                    longlines.append([x1, y1, x2, y2, r.conf[0]])
                elif max(x2-x1, y2-y1) > 50: # height(width) of big corner usually > 50
                    if r.conf[0] > 0.5: # only take big corner with high confidence
                        big_corners.append([x1, y1, x2, y2, r.conf[0]])
                        flag = True
                        if x1<5: # big corner near left of image that is last num line = 1
                            num_last_3lines = 1
                else:
                    corners.append([x1, y1, x2, y2, r.conf[0]])
                    
            corners = nms(corners, 0.3)
            # print('step', i, 'len corners', corners)
            longlines = nms(longlines, 0.1)
            longlines = sorted(longlines, key=lambda x: x[0])
            big_corners = nms(big_corners, 0.3)
        
            # 🏃‍♂️‍➡️ Early skip if appear tail container 🏃‍♂️‍➡️
            if len(big_corners)==2 and flag:
                if big_corners[0][0] > width//3 and big_corners[0][0] < width//4*3 and i-id_start>20:
                    break

            if len(big_corners) > 1 and len(longlines) > 0:
                longlines = clean_noise_lines(big_corners, longlines)

            big_corners = process2bigcors(big_corners, img.shape, longlines)
            
            # initializing crop height variable
            if h_above == 0 and h_under == 0 and len(big_corners) > 1 and len(longlines) > 0:
                h_above, h_under = update_height_crop(height, big_corners, longlines)
            
            # initializing crop width variable    
            if w_right == 0 and len(big_corners) > 1 and len(longlines) > 0 and longlines[0][0] > 10:
                w_right = update_width_crop(width, big_corners, longlines)  
            
            num_current_3lines = 0    
            num_current_3lines += len(longlines) + min((len(big_corners)+1)//2,1)
            
            # check if number of longlines > 1 and so near edge,it will be passed
            if len(longlines):
                if len(longlines) > 1 and longlines[0][0] < 20:
                    num_last_3lines = 1

            # check condition to pass this frame that is unnecessary
            if check_pass_3lines(corners, big_corners, longlines, last_lls, img.shape):
                last_lls = longlines
                # save to temporary data
                if len(big_corners) > 1 and len(longlines) > 0 and len(data_bigcor)==0:
                    data_bigcor = [img, big_corners, longlines, corners, i]
                
                continue
            
            last_lls = longlines
            # ==============👉👉👉Check condition to pick frame👉👉👉=============
            if num_current_3lines != num_last_3lines and num_current_3lines == 2 and len(big_corners) < 3:
                if len(longlines) > 1 and len(list_process_imgs) == 0:
                    if len(data_bigcor) != 0:
                        list_process_imgs.append(self.cropper.special_cropping(data_bigcor))
                        list_id_pick.append(data_bigcor[-1])
                    else:
                        list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))
                        list_id_pick.append(i)

                longlines = sorted(longlines, key=lambda x: x[0])
                # check scammed if model didn't detect enough longline before
                if longlines[0][0]>200 and len(longlines)>1 and i-list_id_pick[-1]<7:
                    pass
                
                elif len(big_corners)>0 and len(list_id_pick)>1 and i-list_id_pick[-1] < 0.7*self.mean_ls_id(list_id_pick):
                    if i-list_id_pick[-1]>2:
                        del list_process_imgs[-1]
                        del list_id_pick[-1]
                        list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))  
                        list_id_pick.append(i)
                else:
                    list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))
                    list_id_pick.append(i)
            
            # update data for next frame
            num_last_3lines = num_current_3lines    
            
            if len(big_corners)==4 and big_corners[0][0]>width//2:
                num_split = len(list_process_imgs)   
        
        print('len list_process_imgs', len(list_process_imgs)) 
        if not len(list_process_imgs):
            return [[],[]], first_id, last_id
        height_scale = min(img.shape[0] for img in list_process_imgs)
        processed_frames = [cv2.resize(img, (img.shape[1], height_scale)) for img in list_process_imgs]
        
        print('list id pick', list_id_pick)
        if num_split != 0:
            print(num_split)
            return [processed_frames[:num_split], processed_frames[num_split:]], first_id, last_id
        else:
            return [processed_frames, []], first_id, last_id
          
    def process_lines_L(self, ls_imgs: List) -> List:
        ls_imgs, first_id, last_id = self.detect_line.get_necessary_imgs(ls_imgs)
        
        list_process_imgs = []
        data_bigcor = [] # save temporary big corners and will be used if necessary
        num_last_3lines = 0 # number of 3 lines in last frame
        h_above = 0 # auto scale height crop abvove image
        h_under = 0 # auto scale height crop under image
        w_right = 0 # auto scale width crop right image
        last_lls = [] # this list is used to store last longlines. It helps to analyze box between 2 frames
        list_id_pick = [] # this list is used to store id of pick container
        flag = False # A flag to check if there is a container truck apear in this frame
        
        # Initialize first and last crop image of container (modified container type L)
        first_cont_crop = None
        last_cont_crop = None
        
        for i,im in enumerate(ls_imgs):
            '''
            📌📌📌
            The first is the conditions for ignoring unnecessary images, including: 
            - Do not take 2 images in a row
            - Some passing if not appear head of container
            - Return early if there is a container truck tail
            => This is help to reduce time process and inference
            '''
            if not flag and i % 3:
                continue

            # 🔧🔧🔧 Preprocesing image before predict🔧🔧🔧 
            img = im[h_above:im.shape[0]-h_under, self.crop_left:im.shape[1]-self.crop_right-w_right]
            height, width, _ = img.shape
            
            # 🔥 🔥 🔥 Let predict and process results 🚀🚀🚀
            results = self.detect_line.detect_line(img)
            corners=[]
            longlines=[]
            big_corners=[]
            
            for r in results[0].boxes:
                box = r.xyxy[0]
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                if y2-y1 > 300: # height of longline usually > 300
                    longlines.append([x1, y1, x2, y2, r.conf[0]])
                elif max(x2-x1, y2-y1) > 50: # height(width) of big corner usually > 60
                    big_corners.append([x1, y1, x2, y2, r.conf[0]])
                    flag = True
                    if x1<5: # big corner near left of image that is last num line = 1
                        num_last_3lines = 1
                else:
                    corners.append([x1, y1, x2, y2, r.conf[0]])
                    
            corners = nms(corners, 0.3)
            longlines = nms(longlines, 0.1)
            longlines = sorted(longlines, key=lambda x: x[0])
            big_corners = nms(big_corners, 0.3)
            
            if len(list_process_imgs)==0 or len(list_process_imgs)>2:
                if is_first_or_last_cont(big_corners, img.shape):
                    if first_cont_crop is None:
                        first_cont_crop = crop_3lines_L(img, big_corners, type='first')
                        continue
                    
                    if last_cont_crop is None and len(list_process_imgs)>2:
                        last_cont_crop = crop_3lines_L(img, big_corners, type='last')
                        break
                        
            if len(big_corners) > 1 and len(longlines) > 0:
                longlines = clean_noise_lines(big_corners, longlines)
            
            # initializing crop height variable
            if h_above == 0 and h_under == 0 and len(big_corners) > 1 and len(longlines) > 0:
                h_above, h_under = update_height_crop(height, big_corners, longlines)
            
            # initializing crop width variable    
            if w_right == 0 and len(big_corners) > 1 and len(longlines) > 0 and longlines[0][0] > 10:
                w_right = update_width_crop(width, big_corners, longlines)  
            
            num_current_3lines = 0    
            num_current_3lines += len(longlines) + min((len(big_corners)+1)//2,1)
            
            # check if number of longlines > 1 and so near edge,it will be passed
            if len(longlines):
                if len(longlines) > 1 and longlines[0][0] < 20:
                    num_last_3lines = 1

            # check condition to pass this frame that is unnecessary
            if check_pass_3lines(corners, big_corners, longlines, last_lls, img.shape):
                last_lls = longlines
                # save to temporary data
                if len(big_corners) > 1 and len(longlines) > 0 and len(data_bigcor)==0:
                    data_bigcor = [img, big_corners, longlines, corners, i]
                
                continue
            
            last_lls = longlines

            # ==============👉👉👉Check condition to pick frame👉👉👉=============
            if num_current_3lines != num_last_3lines and num_current_3lines == 2 and len(big_corners) < 3:
                if len(longlines) > 1 and len(list_process_imgs) == 0:
                    if len(data_bigcor) != 0:
                        list_process_imgs.append(self.cropper.special_cropping(data_bigcor))
                        list_id_pick.append(data_bigcor[-1])
                    else:
                        list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))
                        list_id_pick.append(i)

                longlines = sorted(longlines, key=lambda x: x[0])
                # check scammed if model didn't detect enough longline before
                if longlines[0][0]>200 and len(longlines)>1 and i-list_id_pick[-1]<7:
                    pass
                
                elif len(big_corners)>0 and len(list_id_pick)>1 and i-list_id_pick[-1] < 0.7*self.mean_ls_id(list_id_pick):
                    if i-list_id_pick[-1]>2:
                        del list_process_imgs[-1]
                        del list_id_pick[-1]
                        list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))  
                        list_id_pick.append(i)
                else:
                    list_process_imgs.append(self.cropper.cropping(img, big_corners, longlines, corners))
                    list_id_pick.append(i)
            
            # update data for next frame
            num_last_3lines = num_current_3lines   
        
        list_process_imgs = [first_cont_crop] + list_process_imgs + [last_cont_crop]
        print('len list_process_imgs', len(list_process_imgs)) 
        if not len(list_process_imgs):
            return [[], []], _, _
        height_scale = min(img.shape[0] for img in list_process_imgs)
        processed_frames = [cv2.resize(img, (img.shape[1], height_scale)) for img in list_process_imgs]

        return [processed_frames, []], first_id, last_id
