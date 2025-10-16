import os
import cv2
from ..config import ROOT_STORAGE
import time


def sort_name_in_folder(folders, split_param='.') -> list:
    ls_stt = []
    for fl in os.listdir(folders):
        stt, sf = fl.split(split_param)
        if stt.isdigit():
            ls_stt.append([int(stt), os.path.join(folders, fl)])
    
    new_sort = sorted(ls_stt, key=lambda x: x[0])
    
    return [x[1] for x in new_sort]


def drop_imgs(ls_imgs, min_len=4):
    ls_drop_imgs = []
    if len(ls_imgs) < min_len:
        return ls_imgs
    for i in range(min_len):
        idx = round(i * (len(ls_imgs) - 1) / (min_len - 1))
        ls_drop_imgs.append(ls_imgs[idx])
    return ls_drop_imgs


def drop(ls_imgs, ls_ws):
    imgs_tractor = []
    imgs_trailer = []
    imgs_centers = []
    ls_index = []
    w_max = max(w for w in ls_ws)
    w_min = min(w for w in ls_ws)
    w_avg = w_max - int((w_max - w_min) * 0.32)
    for i in range(len(ls_ws)):
        if ls_ws[i] < w_avg:
            imgs_tractor.append(ls_imgs[i])
            ls_index.append(i)
            continue
        break
    
    for i in range(len(ls_ws)-1, -1, -1):
        if ls_ws[i] < w_avg:
            imgs_trailer.append(ls_imgs[i])
            ls_index.append(i)
            continue
        break
    
    for i in range(len(ls_ws)):
        if i    not in ls_index:
            imgs_centers.append(ls_imgs[i])
    return imgs_tractor, imgs_trailer[::-1], imgs_centers       


def read_imgs(folder_images, cont_size=42):
    
    if cont_size == 22:
        min_len_imgs = 13
    elif cont_size == 45 or cont_size == 12:
        min_len_imgs = 62
    else:
        min_len_imgs = 60
    
    t = time.time()
    ls_imgs = []
    ls_ws = []
    ls_drop_imgs = []
    new_folders = sort_name_in_folder(folders=f"{ROOT_STORAGE}/{folder_images}")
    for fl in new_folders:
        img = cv2.imread(fl)
        w = img.shape[1]
        ls_ws.append(w)
        ls_imgs.append(img)
    
    imgs_tractor, imgs_trailer, imgs_centers = drop(ls_imgs, ls_ws)
    
    imgs_tractor = drop_imgs(imgs_tractor, min_len=4)
    imgs_trailer = drop_imgs(imgs_trailer, min_len=4)
    imgs_centers = drop_imgs(imgs_centers, min_len=min_len_imgs)
    if cont_size == 22:
        ls_drop_imgs = imgs_tractor + imgs_centers + imgs_trailer
    else:
        ls_drop_imgs = imgs_tractor[:3] + imgs_centers[1:-2] + imgs_trailer[-3:]
        if len(ls_drop_imgs) < min_len_imgs:
            ls_drop_imgs = imgs_tractor + imgs_centers + imgs_trailer
    print("Time read imgs:", time.time()-t)
    return ls_drop_imgs, ls_imgs


def read_imgs_top(folder_images):
    ls_imgs = []
    new_folders = sort_name_in_folder(folders=f"{ROOT_STORAGE}/{folder_images}")
    for fl in new_folders:
        img = cv2.imread(fl)
        ls_imgs.append(img)
    return ls_imgs


if __name__ == '__main__':
    folders = r"/mnt/atin/Hoai/Stit_Image/resources/img/20241121_145623"
    new = sort_name_in_folder(folders)
    for n in new:
        print(n)
    