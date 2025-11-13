import cv2

min_size_img = 50

def check_min_size_img(w):
    """
    Kiểm tra kích thước ảnh có nhỏ hơn kích thước tối thiểu không
    """
    if w < min_size_img:
        return False
    return True


def fix_point(pts1, pts2, w1, w2):
    new_pts1, new_pts2 = [], []
    x1l, y1l = pts1
    x2l, y2l = pts2
    d1 = x1l - w1 // 2
    d2 = x2l - w2 // 2
    
    avr = abs(d1 + d2) // 2
    if w1 < w2:
        if x1l > w1 //2:
            x1l -= avr
            x2l -= avr
        else:
            x1l += avr
            x2l += avr
    else:
        if x2l > w2 //2:
            x1l -= avr
            x2l -= avr
        else:
            x1l += avr
            x2l += avr
    
    new_pts1 = [x1l, y1l]
    new_pts2 = [x2l, y2l]
    return new_pts1, new_pts2


def stit_2img(img1, img2, left_pts, right_pts):
    """
    Nối hai ảnh dựa trên các cặp điểm src_pts và dst_pts.
    """
    width, height = img1.shape[1], img1.shape[0]
    for left_pt, right_pt in zip(left_pts, right_pts):

        img1_cropped = img1[0:height, int(left_pt[0]):width]
        img2_cropped = img2[0:height, 0:int(right_pt[0])]
        # Nối hai ảnh lại
        return cv2.hconcat((img2_cropped, img1_cropped))


def concate_2image(img1, img2, point1, point2):
    # Nối hai ảnh dựa trên điểm tốt nhất
    return stit_2img(img1, img2, [point1], [point2])

def stit_2img_fix_dent(img1, img2, left_pts, right_pts, i_cur, len_imgs):
    """
    Nối hai ảnh dựa trên các cặp điểm src_pts và dst_pts.
    """
    width, height = img1.shape[1], img1.shape[0]
    add_count = int(0.1 * len(left_pts))+4
    
    for left_pt, right_pt, i in zip(left_pts, right_pts, range(len_imgs)):
        if i_cur > len_imgs - add_count:
            left_pt, right_pt = min(width-50, left_pt[0]+170), min(width-50, right_pt[0]+170)
        elif i_cur < add_count:
            left_pt, right_pt = max(10, left_pt[0]-100), max(10, right_pt[0]-100)
        else:
            left_pt, right_pt = left_pt[0], right_pt[0]
        # left_pt, right_pt = left_pt[0], right_pt[0]
        img1_cropped = img1[0:height, int(left_pt):width]
        img2_cropped = img2[0:height, 0:int(right_pt)]
        
        # Nối hai ảnh lại
        return cv2.hconcat((img2_cropped, img1_cropped))


def concate_2image_fix_dent(img1, img2, point1, point2, i_cur, len_imgs):
    # Nối hai ảnh dựa trên điểm tốt nhất
    return stit_2img_fix_dent(img1, img2, [point1], [point2], i_cur, len_imgs)


def crop_stit_imgs(stit_imgs, list_stit_dents, l_dent, last_count_dent):
    for i, img, dents in zip(range(1,len(stit_imgs)), stit_imgs, list_stit_dents):
        
        if i == len(stit_imgs)-1:
            last_dent = int(last_count_dent)
            if list_stit_dents[-2][0][2]-list_stit_dents[-2][0][0]>l_dent-5:
                # print('check', list_stit_dents[-1], '---', last_dent)
                stit_imgs[-1] = stit_imgs[-1][:,:list_stit_dents[-1][last_dent][0]]
                if list_stit_dents[-2][0][0]>5:
                    x_crop = list_stit_dents[-2][0][0]
                    stit_imgs[-2] = stit_imgs[-2][:,x_crop:]

            else:
                x_add = int(l_dent-(list_stit_dents[-2][0][2]-list_stit_dents[-2][0][0]))          
                stit_imgs[-1] = stit_imgs[-1][:,:list_stit_dents[-1][last_dent][0]+x_add]       
            
            return stit_imgs  
        
        if list_stit_dents[i-1][0][0]<10:
            
            if list_stit_dents[i][1][2]-list_stit_dents[i][1][0]+list_stit_dents[i-1][0][2]-list_stit_dents[i-1][0][0]>l_dent-5:
                x_crop = list_stit_dents[i][1][2]-list_stit_dents[i][1][0]+list_stit_dents[i-1][0][2]-list_stit_dents[i-1][0][0]-l_dent+5
                stit_imgs[i] = stit_imgs[i][:,:list_stit_dents[i][1][2]-int(x_crop)]
                
            else:
                stit_imgs[i] = stit_imgs[i][:,:list_stit_dents[i][1][0]]
                stit_imgs[i-1] = stit_imgs[i-1][:,list_stit_dents[i-1][0][2]:]
                
        else:
            x_crop = list_stit_dents[i-1][0][0]
            stit_imgs[i-1] = stit_imgs[i-1][:,x_crop:]
            
            if list_stit_dents[i][1][2]<stit_imgs[i].shape[1]-5:
                x_crop = list_stit_dents[i][1][2]
                stit_imgs[i] = stit_imgs[i][:,:x_crop]
            else:
                x_crop = list_stit_dents[i][1][0]
                stit_imgs[i] = stit_imgs[i][:,:x_crop]  

    return
