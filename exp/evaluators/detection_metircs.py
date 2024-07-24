import math
import os
from pathlib import Path
import time
from typing import List
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import logging
import pickle


LOGGER = logging.getLogger(__name__)

def plot_many(image_list, titles=None, rows=None, cols=None, show=True, default_type="image", **kwargs):
    assert isinstance(image_list, np.ndarray) or isinstance(image_list, List)
    if isinstance(image_list, np.ndarray):
        image_list = [image_list]
        
    image_num = len(image_list)
    layout_flag = False
    if rows is not None:
        cols = image_num // rows
    elif cols is not None:
        rows = image_num // cols
    elif rows is None and cols is None:
    
        if image_num == 1:
            cols = rows = 1
        else:
            layout_flag = True
            raise RuntimeWarning("Rows/columns should be assigned.")

        # TODO
    
    cache = []
    for i, image in enumerate(image_list):
        assert isinstance(image, np.ndarray) and 3 >= image.ndim >= 1
        
        type_ = None
        
        if image.ndim == 2:
            # expand one channel
            image = image[..., None]
        
        if image.ndim == 3:
            if default_type == "image":
                channels = image.shape[-1]
                assert channels in [1, 2, 3, 4]
                type_ = "image"
            elif default_type == "points":
                # TODO
                pass
        if default_type=="hist" or (image.ndim == 1 and default_type=="image"):
            default_type = "hist"
            plt.hist(image, **kwargs)
            
        if image.ndim == 1 and default_type=="bar":
            # type_ = "hist"
            plt.bar(range(image.shape[0]), image, **kwargs)
            plt.xticks(range(image.shape[0]))
            # TODO
            
        if not layout_flag:
            plt.subplot(rows, cols, i + 1)

        if default_type == "image":
            plt.imshow(image)
            plt.axis("off")
            
        if titles is not None and len(titles) > i:
            title = titles[i]
            plt.title(title)
        plt.tight_layout()
    
    if show:    
        plt.show()
    return plt

    """
    We use multi-framework to train our detection models, and use the default evaluation in the frameworks,
    but the metrics between different frameworks has slight difference, so we extract the yolo-v8's evaluation as the 
    uniform standards.
    
    The below codes is from yolo-v8, and removed the dependencies at the yolo libs
    """

def box_iou(box1, box2, eps=1e-7):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        eps

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(
        2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        # convex (smallest enclosing box) width
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 +
                    b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        # GIoU https://arxiv.org/pdf/1902.09630.pdf
        return iou - (c_area - union) / c_area
    return iou  # IoU


def mask_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [M, n] m2 means number of gt objects
    Note: n means image_w x image_h
    Returns: masks iou, [N, M]
    """
    intersection = torch.matmul(mask1, mask2.t()).clamp(0)
    union = (mask1.sum(1)[:, None] + mask2.sum(1)[None]) - \
        intersection  # (area1 + area2) - intersection
    return intersection / (union + eps)


def masks_iou(mask1, mask2, eps=1e-7):
    """
    mask1: [N, n] m1 means number of predicted objects
    mask2: [N, n] m2 means number of gt objects
    Note: n means image_w x image_h
    Returns: masks iou, (N, )
    """
    intersection = (mask1 * mask2).sum(1).clamp(0)  # (N, )
    # (area1 + area2) - intersection
    union = (mask1.sum(1) + mask2.sum(1))[None] - intersection
    return intersection / (union + eps)


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


def smooth(y, f=0.05):
    # Box filter of fraction f
    # number of filter elements (must be odd)
    nf = round(len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            # plot(recall, precision)
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue',
            label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title('Precision-Recall Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            # plot(confidence, metric)
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')
    else:
        # plot(confidence, metric)
        ax.plot(px, py.T, linewidth=1, color='grey')

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue',
            label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    ax.set_title(f'{ylabel}-Confidence Curve')
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    Arguments:
        recall:    The recall curve (list)
        precision: The precision curve (list)
    Returns:
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x-axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class SimpleClass:
    """
    Ultralytics SimpleClass is a base class providing helpful string representation, error reporting, and attribute
    access methods for easier debugging and usage.
    """

    def __str__(self):
        """Return a human-readable string representation of the object."""
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('__'):
                if isinstance(v, SimpleClass):
                    # Display only the module and class name for subclasses
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        """Return a machine-readable string representation of the object."""
        return self.__str__()

    def __getattr__(self, attr):
        """Custom attribute access error message with helpful information."""
        name = self.__class__.__name__
        raise AttributeError(
            f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names=(), eps=1e-16, prefix=''):
    """
    Computes the average precision per class for object detection evaluation.

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct (True) or not (False).
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        plot (bool, optional): Whether to plot PR curves or not. Defaults to False.
        save_dir (Path, optional): Directory to save the PR curves. Defaults to an empty path.
        names (tuple, optional): Tuple of class names to plot PR curves. Defaults to an empty tuple.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.
        prefix (str, optional): A prefix string for saving the plot files. Defaults to an empty string.

    Returns:
        (tuple): A tuple of six arrays and one array of unique classes, where:
            tp (np.ndarray): True positive counts for each class.
            fp (np.ndarray): False positive counts for each class.
            p (np.ndarray): Precision values at each confidence threshold.
            r (np.ndarray): Recall values at each confidence threshold.
            f1 (np.ndarray): F1-score values at each confidence threshold.
            ap (np.ndarray): Average precision for each class at different IoU thresholds.
            unique_classes (np.ndarray): An array of unique classes that have data.

    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros(
        (nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                          left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    # list: only classes that have data
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))  # to dict
    if plot:
        plot_pr_curve(px, py, ap, save_dir / f'{prefix}PR_curve.png', names)
        plot_mc_curve(px, f1, save_dir /
                      f'{prefix}F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, save_dir /
                      f'{prefix}P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, save_dir /
                      f'{prefix}R_curve.png', names, ylabel='Recall')

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


class Metric(SimpleClass):
    """
        Class for computing evaluation metrics for YOLOv8 model.

        Attributes:
            p (list): Precision for each class. Shape: (nc,).
            r (list): Recall for each class. Shape: (nc,).
            f1 (list): F1 score for each class. Shape: (nc,).
            all_ap (list): AP scores for all classes and all IoU thresholds. Shape: (nc, 10).
            ap_class_index (list): Index of class for each AP score. Shape: (nc,).
            nc (int): Number of classes.

        Methods:
            ap50(): AP at IoU threshold of 0.5 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            ap(): AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: List of AP scores. Shape: (nc,) or [].
            mp(): Mean precision of all classes. Returns: Float.
            mr(): Mean recall of all classes. Returns: Float.
            map50(): Mean AP at IoU threshold of 0.5 for all classes. Returns: Float.
            map75(): Mean AP at IoU threshold of 0.75 for all classes. Returns: Float.
            map(): Mean AP at IoU thresholds from 0.5 to 0.95 for all classes. Returns: Float.
            mean_results(): Mean of results, returns mp, mr, map50, map.
            class_result(i): Class-aware result, returns p[i], r[i], ap50[i], ap[i].
            maps(): mAP of each class. Returns: Array of mAP scores, shape: (nc,).
            fitness(): Model fitness as a weighted combination of metrics. Returns: Float.
            update(results): Update metric attributes with new evaluation results.

        """

    def __init__(self) -> None:
        self.p = []  # (nc, )
        self.r = []  # (nc, )
        self.f1 = []  # (nc, )
        self.all_ap = []  # (nc, 10)
        self.ap_class_index = []  # (nc, )
        self.nc = 0

    @property
    def ap50(self):
        """AP@0.5 of all classes.
        Returns:
            (nc, ) or [].
        """
        return self.all_ap[:, 0] if len(self.all_ap) else []

    @property
    def ap(self):
        """AP@0.5:0.95
        Returns:
            (nc, ) or [].
        """
        return self.all_ap.mean(1) if len(self.all_ap) else []

    @property
    def mp(self):
        """mean precision of all classes.
        Returns:
            float.
        """
        return self.p.mean() if len(self.p) else 0.0

    @property
    def mr(self):
        """mean recall of all classes.
        Returns:
            float.
        """
        return self.r.mean() if len(self.r) else 0.0

    @property
    def map50(self):
        """Mean AP@0.5 of all classes.
        Returns:
            float.
        """
        return self.all_ap[:, 0].mean() if len(self.all_ap) else 0.0

    @property
    def map75(self):
        """Mean AP@0.75 of all classes.
        Returns:
            float.
        """
        return self.all_ap[:, 5].mean() if len(self.all_ap) else 0.0

    @property
    def map(self):
        """Mean AP@0.5:0.95 of all classes.
        Returns:
            float.
        """
        return self.all_ap.mean() if len(self.all_ap) else 0.0

    def mean_results(self):
        """Mean of results, return mp, mr, map50, map"""
        return [self.mp, self.mr, self.map50, self.map]

    def class_result(self, i):
        """class-aware result, return p[i], r[i], ap50[i], ap[i]"""
        return self.p[i], self.r[i], self.ap50[i], self.ap[i]

    @property
    def maps(self):
        """mAP of each class"""
        maps = np.zeros(self.nc) + self.map
        for i, c in enumerate(self.ap_class_index):
            maps[c] = self.ap[i]
        return maps

    def fitness(self):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
        return (np.array(self.mean_results()) * w).sum()

    def update(self, results):
        """
        Args:
            results: tuple(p, r, ap, f1, ap_class)
        """
        self.p, self.r, self.f1, self.all_ap, self.ap_class_index = results


class DetMetrics(SimpleClass):
    """
    This class is a utility class for computing detection metrics such as precision, recall, and mean average precision
    (mAP) of an object detection model.

    Args:
        save_dir (Path): A path to the directory where the output plots will be saved. Defaults to current directory.
        plot (bool): A flag that indicates whether to plot precision-recall curves for each class. Defaults to False.
        names (tuple of str): A tuple of strings that represents the names of the classes. Defaults to an empty tuple.

    Attributes:
        save_dir (Path): A path to the directory where the output plots will be saved.
        plot (bool): A flag that indicates whether to plot the precision-recall curves for each class.
        names (tuple of str): A tuple of strings that represents the names of the classes.
        box (Metric): An instance of the Metric class for storing the results of the detection metrics.
        speed (dict): A dictionary for storing the execution time of different parts of the detection process.

    Methods:
        process(tp, conf, pred_cls, target_cls): Updates the metric results with the latest batch of predictions.
        keys: Returns a list of keys for accessing the computed detection metrics.
        mean_results: Returns a list of mean values for the computed detection metrics.
        class_result(i): Returns a list of values for the computed detection metrics for a specific class.
        maps: Returns a dictionary of mean average precision (mAP) values for different IoU thresholds.
        fitness: Computes the fitness score based on the computed detection metrics.
        ap_class_index: Returns a list of class indices sorted by their average precision (AP) values.
        results_dict: Returns a dictionary that maps detection metric keys to their computed values.
    """

    def __init__(self, save_dir=Path('.'), plot=False, names=()) -> None:
        self.save_dir = save_dir
        self.plot = plot
        self.names = names
        self.box = Metric()
        self.speed = {'preprocess': 0.0, 'inference': 0.0,
                      'loss': 0.0, 'postprocess': 0.0}

    def process(self, tp, conf, pred_cls, target_cls):
        results = ap_per_class(tp, conf, pred_cls, target_cls, plot=self.plot, save_dir=self.save_dir,
                               names=self.names)[2:]
        self.box.nc = len(self.names)
        self.box.update(results)

    @property
    def keys(self):
        return ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

    def mean_results(self):
        return self.box.mean_results()

    def class_result(self, i):
        return self.box.class_result(i)

    @property
    def maps(self):
        return self.box.maps

    @property
    def fitness(self):
        return self.box.fitness()

    @property
    def ap_class_index(self):
        return self.box.ap_class_index

    @property
    def results_dict(self):
        return dict(zip(self.keys + ['fitness'], self.mean_results() + [self.fitness]))


def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        if detections is None:
            gt_classes = labels.int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1  # background FN
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat(
                (torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # true background

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        import seaborn as sn

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-9)
                               if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        sn.set(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (names + ['background']) if labels else 'auto'
        with warnings.catch_warnings():
            # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            warnings.simplefilter('ignore')
            sn.heatmap(array,
                       ax=ax,
                       annot=nc < 30,
                       annot_kws={
                           'size': 8},
                       cmap='Blues',
                       fmt='.2f',
                       square=True,
                       vmin=0.0,
                       xticklabels=ticklabels,
                       yticklabels=ticklabels).set_facecolor((1, 1, 1))
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title('Confusion Matrix')
        fig.savefig(Path(save_dir) / 'confusion_matrix.png', dpi=250)
        plt.close(fig)
        

    def print(self):
        for i in range(self.nc + 1):
            LOGGER.info(' '.join(map(str, self.matrix[i])))


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


    """
    Merge the yolo codes into our work flow.
    """

def get_currect_box_matrix(detections, labels, iouv=torch.linspace(0.5, 0.95, 10)):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iouv[i]) & correct_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(
                    matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)


def update_metrics(preds, batch, stats, confusion_matrix: ConfusionMatrix, device="cpu", single_cls=False, box_convert=xywh2xyxy, iouv=torch.linspace(0.5, 0.95, 10)):
    niou = iouv.numel()
    # Metrics
    for si, pred in enumerate(preds):
        idx = si == batch['batch_idx']
        cls = batch['cls'][idx]
        bbox = batch['bboxes'][idx]
        nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
        shape = batch['ori_shape'][si]
        correct_bboxes = torch.zeros(
            npr, niou, dtype=torch.bool, device=device)  # init

        if npr == 0:
            if nl:
                stats.append(
                    (correct_bboxes, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                confusion_matrix.process_batch(
                    detections=None, labels=cls.squeeze(-1))
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                    ratio_pad=batch['ratio_pad'][si])  # native-space pred

        # Evaluate
        if nl:
            height, width = batch['img'][si].shape[1:]
            tbox = box_convert(bbox) * torch.tensor(
                (width, height, width, height), device=device)  # target boxes

            scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                        ratio_pad=batch['ratio_pad'][si])  # native-space labels
            labelsn = torch.cat((cls, tbox), 1)  # native-space labels

            correct_bboxes = get_currect_box_matrix(predn, labelsn, iouv=iouv)
            confusion_matrix.process_batch(predn, labelsn)
        # (conf, pcls, tcls)
        stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))


def get_stats(metrics: DetMetrics, stats, nc):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        metrics.process(*stats)
    # number of targets per class
    nt_per_class = np.bincount(stats[-1].astype(int), minlength=nc)
    return metrics, nt_per_class


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nc=0,  # number of classes (optional)
        max_time_img=0.05,
        max_nms=30000,
        max_wh=7680,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    # YOLOv8 model in validation model, output = (inference_out, loss_out)
    if isinstance(prediction, (list, tuple)):
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 0.5 + max_time_img * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        # center_x, center_y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(box)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None],
                          j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # sort by confidence and remove excess boxes
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            LOGGER.warning(
                f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output


def postprocess(preds, conf=0.25, iou=0.75):
    preds = non_max_suppression(preds,
                                conf,
                                iou,
                                multi_label=True)
    return preds

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
    Returns:
       y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y

class DetectMetircs():

    def __init__(self, names, output_dir, cm_conf=0.45, nms_thresold=0.75, plot=True) -> None:
        self._output_dir = output_dir
        self.nc = len(names)
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(self.nc, conf=cm_conf)
        self.names = names
        self.metrics = DetMetrics(save_dir=Path(
            self._output_dir), names=self.names_list2dict(), plot=True)
        
        self.nms_thresold = nms_thresold
        self.plot = plot

    def names_list2dict(self):
        return {i: name for i, name in enumerate(self.names)}


    def _update(self, preds_instances, gt_instances, image_name, image_shape, ori_shape=None):
        """Update the metircs one by one

        Args:
            preds_instances (ndarray): numpy array, shape (N, 6), order should be [x,y,x,y,score,class_id]
            gt_instances (ndarray): numpy array, shape (N, 5), order should be [x,y,x,y,class_id]
            image_shape (tuple): (H, W, ...)
            org_shape (tuple): (H, W)
            image_name (str): image id
        """
        
        assert ori_shape or image_shape
        
        if ori_shape is None:
            ori_shape = image_shape[:2]
        if image_shape is None:
            image_shape = list(ori_shape) + [3]
            
            
        preds_instances = torch.tensor(preds_instances)
        gt_instances = torch.tensor(gt_instances)
        
        gt_classes = gt_instances[:, -1:]
        gt_boxes = gt_instances[:, :4]
        # height, width = image_shape[:2]
        # bn = gt_classes.shape[0]
        
        num_instance = len(preds_instances)
        
        # Addtional nms
        if num_instance > 0:
            # do NMS
            boxes = preds_instances[:, :4]
            scores = preds_instances[:, 4]
            classes = preds_instances[:, 5]
            kept = torchvision.ops.nms(boxes, scores, self.nms_thresold)
            boxes = boxes[kept]
            scores = scores[kept]
            classes = classes[kept]
        
            preds_instances = torch.concat([boxes, scores[..., None], classes[..., None]], dim=1).unsqueeze(0)
        
        # Convert to yolo acceptable input format
        # Yolo stack the different batch's instance on the first dimension (batch)
        # the batch_idx flag the batch index of the instances
        # for example :
        # [
        #   [batch_idx0, ...],
        #   [batch_idx0, ...]
        #   [batch_idx0, ...],
        #   ...
        #   [batch_idx1, ...],
        #   [batch_idx1, ...],
        #   [batch_idx1, ...],
        #   ...
        #   [batch_idxN, ...],
        #   [batch_idxN, ...],
        #   [batch_idxN, ...],
        #   ...
        # ]
        # we process only one batch, so the instances will all flag as 0
        # the prediction of the model may be scaled (ori_shape-> model_input_shape-> predic box coord), so the "ori_shape" and "img" will help the convert the box coord into the 
        # the origin image' scale (predic box coord on the model_input_shape -> predic box coord on the ori_shape)
        # NOTE: label's box coord should on the ori_shape !
        one_batch = {"batch_idx": torch.tensor([0] * num_instance), "ori_shape": [ori_shape], "cls": gt_classes, "bboxes": gt_boxes, "ratio_pad": [None], "img": np.empty((1, *image_shape)).transpose(0, 3, 1, 2), "im_file": [image_name]}
        if len(gt_boxes) > 0:
            one_batch['bboxes'] =  xyxy2xywh(gt_boxes)
            
        update_metrics(preds_instances, one_batch, self.stats, self.confusion_matrix)
        
    def evaluate(self):
        results, nt_per_class = get_stats(self.metrics, self.stats, self.nc)#.results_dict
        cm_np = self.confusion_matrix.matrix

        if self.plot:
            # plot confusion matix
            self.confusion_matrix.plot(save_dir = self._output_dir, names=self.names)
            print(nt_per_class)
            # plot class number
            plot_many(nt_per_class, titles=["class count"], default_type="bar", show=False).savefig(os.path.join(self._output_dir, "class_his.jpg"))
        # save the instances
        with open(os.path.join(self._output_dir, "instances_results.pth"), "wb") as f:
            pickle.dump(results, f)
        # save the confusion matrix
        np.save(os.path.join(self._output_dir, "confusion_matrix"), cm_np)
        return results
 
# TODO   
#     #   [
#     #       {   
#     #           version: "1.0",
#     #           image_name: "xxx.png", 
#     #           instances:
#     #           [
#     #               {
#     #                   bbox:   [x, y, x, y], 
#     #                   category_id: 0
#     #                }, ...
#     #           ],
#     #           image_shape:    [height, width]
#     #       },
#     #       ...
#     #   ]
# def parse_label_json(label_data):
#     version = label_data["version"]
#     if version == "1.0":
#         for image in label_data:
#             image_name = image["image_name"]
#             instances_raw =  image["instances"]
#             instances = []
#             for instance in instances_raw:
#                 instances.append(instance)
        
#     else:
#         raise RuntimeError(f"Unknow version, not support version: {version}")
    
#     pass

#     #   [
#     #       {   
#     #           version: "1.0",
#     #           image_name: "xxx.png", 
#     #           instances:
#     #           [
#     #               {
#     #                   bbox:   [x, y, x, y],  # absolut coord in the image_shape
#     #                   category_id: 0,
#     #                   confidence: 0.7832
#     #                }, ...
#     #           ],
#     #           image_shape:    [height, width], # image shape (may be resized for the model acceptable input shape), padding shape not support
#     #           org_shape:  [height, width] # optional, if image_shape is as same as org_shape, this shape will be ignore
#     #       },
#     #       ...
#     #   ]
# def parse_prediction_json(pred_data):
#     pass


def get_metrics_batch(detect_batch_iter, names, output_dir, cm_conf=0.45, nms_thresold=0.75, plot=True):
    """Detect metrics

    Args:
        detect_batch_iter (genrator or list): the list or genrator of the prediction and ground truth.
        names (List[str]): label names.
        output_dir (str): output folder path.
        cm_conf (float, optional): confusion matrix confidence. Defaults to 0.45.
        nms_thresold (float, optional): NMS thresold. Defaults to 0.75.

    Returns:
        DetMetric: get the metric of the input data
    """    
    dm = DetectMetircs(names, output_dir=output_dir, cm_conf=cm_conf, nms_thresold=nms_thresold, plot=plot)
    for preds_instances, gt_instances, image_name, image_shape, ori_shape in detect_batch_iter:
        # normalize the bbox xxyy if it is abs size
        if np.all(gt_instances[:, :4] > 1):
            gt_instances[:, 0::2] /= ori_shape[0]
            gt_instances[:, 1::2] /= ori_shape[1]

        dm._update(preds_instances=preds_instances, gt_instances=gt_instances, image_shape=image_shape, image_name=image_name, ori_shape=ori_shape)
    
    return dm.evaluate()

    
if __name__ == "__main__":
    from tools.class_map import CLASS_TEN_NAMES
    # Test one by one
    dm = DetectMetircs(CLASS_TEN_NAMES, output_dir="outputs/test_detect_metrics")
    os.makedirs("outputs/test_detect_metrics", exist_ok=True)
    
    predictions = np.array([[1023, 1111, 3131, 3121, 0.8, 0], [4023, 4111, 4131, 5121, 0.6, 1]], dtype=np.float32)
    gts = np.array([[1023, 1111, 3131, 3121, 0], [4023, 4111, 4131, 5121, 1]], dtype=np.float32)

    gts[:, :4] /= 10000
    dm._update(preds_instances=predictions, gt_instances=gts, image_shape=(10000, 10000, 3), image_name="0001")
    predictions = np.array([[1023, 1111, 3131, 3101, 0.8, 0],  [4023, 4111, 4131, 5121, 0.6, 1]], dtype=np.float32)
    gts = np.array([[1023, 1111, 3131, 3121, 0], [4023, 4111, 4131, 5121, 1]], dtype=np.float32)
    gts[:, :4] /= 10000
    dm._update(preds_instances=predictions, gt_instances=gts, image_shape=(10000, 10000, 3), image_name="0002")

    
    # Test iteration
    # every item format:
    # (preds_instances, gt_instances, image_name, image_shape, ori_shape)
    # (ndarray: (N, 6), ndarray: (N, 5), str: "xxx.png",  tuple: (height, width, ...), tuple|None: (height, width) or None)
    
    test_iter = [(predictions, gts, "001", (10000, 10000, 3), None), (predictions, gts, "002", (10000, 10000, 3), None)]
    os.makedirs("outputs/test_detect_metrics2", exist_ok=True)

    print(get_metrics_batch(test_iter, CLASS_TEN_NAMES, "outputs/test_detect_metrics2"))
    
    
# end main