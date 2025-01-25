import numpy as np

# IoU Calculation Function
def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    xi1, yi1 = max(x1, x1_p), max(y1, y1_p)
    xi2, yi2 = min(x2, x2_p), min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Merge overlapping boxes
def merge_boxes_iteratively(boxes, iou_threshold=0.1):
    merged = False
    while True:
        new_boxes = []
        used = set()
        merged = False

        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            x1, y1, x2, y2 = box1
            for j, box2 in enumerate(boxes):
                if i != j and j not in used and iou(box1, box2) > iou_threshold:
                    x1 = min(x1, box2[0])
                    y1 = min(y1, box2[1])
                    x2 = max(x2, box2[2])
                    y2 = max(y2, box2[3])
                    used.add(j)
                    merged = True
            used.add(i)
            new_boxes.append([x1, y1, x2, y2])

        boxes = np.array(new_boxes)
        if not merged:
            break

    return boxes
