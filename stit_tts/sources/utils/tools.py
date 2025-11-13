

def iou(box1, box2):
    # box format [x1, y1, x2, y2, confidence]
    x1_box1, y1_box1, x2_box1, y2_box1 = box1[:4]
    x1_box2, y1_box2, x2_box2, y2_box2 = box2[:4]
    
    # calculate th region between box1 and box2
    x1 = max(x1_box1, x1_box2)
    y1 = max(y1_box1, y1_box2)
    x2 = min(x2_box1, x2_box2)
    y2 = min(y2_box1, y2_box2)
    
    # calculate the area of intersection rectangle
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # calculate the area of union rectangle
    union_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1) + (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1) - inter_area
    
    return inter_area / union_area
    
    
def nms(boxes, iou_threshold):
    '''
    boxe: [x1, y1, x2, y2, confidence]
    iou_threshold: threshold of iou to remove the box and keep higher confidence box
    return boxes after nms
    '''
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    
    keep_boxes = []
    
    for box in boxes:
        if len(keep_boxes) == 0:
            keep_boxes.append(box)
        else:
            for keep_box in keep_boxes:
                if iou(box, keep_box) > iou_threshold:
                    break
            else:
                keep_boxes.append(box)
        
    return keep_boxes


def get_ratio_img(width, height) -> float:
    return round(width / height, 2)