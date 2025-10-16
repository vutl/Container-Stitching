def get_gucors_side(results):
    gu_cors = []
    for r in results:
        box = r.xyxy[0]
        class_id = r.cls[0]
        x1, y1, x2, y2 = map(int, box)
        x_c,yc = (x1+x2)//2, (y1+y2)//2
        gu_cors.append([x_c, yc]) if class_id == 1 else None
    return gu_cors


def get_gucors_top(results):
    gu_cors = []
    for r in results[0].boxes:
        box = r.xyxy[0]
        class_id = r.cls[0]
        x1, y1, x2, y2 = map(int, box)
        x_c,yc = (x1+x2)//2, (y1+y2)//2
        gu_cors.append([x_c, yc]) if abs(y2-y1)>50 else None
    return gu_cors


def x_split_2cont(gu_cors, h):
    top_gucor = [cor for cor in gu_cors if cor[1] < h//2]
    bot_gucor = [cor for cor in gu_cors if cor[1] > h//2]

    if len(top_gucor) == 2 and abs(top_gucor[0][0]-top_gucor[1][0])<350: # 350 pixels is the center distance of 2 containers
        return (top_gucor[0][0]+top_gucor[1][0])//2

    if len(bot_gucor) == 2 and abs(bot_gucor[0][0]-bot_gucor[1][0])<350: # 350 pixels is the center distance of 2 containers
        return (bot_gucor[0][0]+bot_gucor[1][0])//2
  
    return 0
