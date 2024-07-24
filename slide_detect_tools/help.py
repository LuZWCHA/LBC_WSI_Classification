import numpy as np
import torchvision.ops
import torch

ID2LABEL = {
    "0": "10",
    "1": "8",
    "2": "11",
    "3": "9",
    "4": "2",
    "5": "1010",
    "6": "7",
    "7": "3",
    "8": "4",
    "9": "5",
    "10": "6",
}

def nms_boxes(boxes, box_confidences, nms_threshold=0.2, use_torchvision=False):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
    confidence scores and return an array with the indexes of the bounding boxes we want to
    keep (and display later).

    Keyword arguments:
    boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
    with shape (N,4); 4 for x,y,height,width coordinates of the boxes
    box_confidences -- a Numpy array containing the corresponding confidences with shape N
    """
    
    if use_torchvision:
        # The convert between tensor and numpy array is very fast, dont care about the cost of the conversion.
        if isinstance(boxes, np.ndarray):
            boxes = torch.from_numpy(boxes).cuda(non_blocking=True)
        if isinstance(box_confidences, np.ndarray):
            box_confidences = torch.from_numpy(box_confidences).cuda(non_blocking=True)

        kept = torchvision.ops.nms(boxes, box_confidences, nms_threshold)
        kept = kept.cpu().numpy()
        return kept

    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[ordered[1:]])
        yy1 = np.maximum(y1[i], y1[ordered[1:]])
        xx2 = np.minimum(x2[i], x2[ordered[1:]])
        yy2 = np.minimum(y2[i], y2[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
        # candidates to a minimum. In this step, we keep only those elements whose overlap
        # with the current bounding box is lower than the threshold:
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep

def remove_overlap(bboxes, scores, labels, threshold=0.75):
    areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
    order = areas.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(bboxes[i, 1], bboxes[order[1:], 1])
        yy1 = np.maximum(bboxes[i, 0], bboxes[order[1:], 0])
        xx2 = np.minimum(bboxes[i, 3], bboxes[order[1:], 3])
        yy2 = np.minimum(bboxes[i, 2], bboxes[order[1:], 2])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ratio = inter / areas[order[1:]]
        # assign new probs class
        overlap_inds = order[np.where(ratio >= threshold)[0]+1]
        if len(overlap_inds) > 0:
            overlap_scores = scores[overlap_inds]
            overlap_labels = labels[overlap_inds]
            max_score_id = np.argmax(overlap_scores)
            max_label = overlap_labels[max_score_id]
            max_score = overlap_scores[max_score_id]
            if max_score > scores[i]:
                scores[i] = max_score
                labels[i] = max_label
        inds = np.where(ratio < threshold)[0]
        order = order[inds + 1]

    return keep