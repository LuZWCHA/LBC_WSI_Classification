import random
from albumentations.core.bbox_utils import denormalize_bbox, normalize_bbox

from albumentations.core.transforms_interface import (
    DualTransform,
    ImageOnlyTransform,
    NoOp,
    to_tuple,
)
import numpy as np

# detectron2 input format
class BBoxNoiseDT2(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, offset_mean=(0, 0), offset_std=(3, 3), scale_mean=1, scale_std=0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.offset_mean = offset_mean
        self.offset_std = offset_std
        self.scale_mean = scale_mean
        self.scale_std = scale_std

    def apply(self, img, **params):
        return img

    def _clip(self, a, min_, max_):
        return max(min(a, max_), min_)

    def apply_to_bbox(self, bbox, **params):
        # deteron2 box input is XYXY_ABS format
        x_min, y_min, x_max, y_max = bbox[0]
        
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        
        center_x += random.gauss(self.offset_mean[0], self.offset_std[0])
        center_y += random.gauss(self.offset_mean[1], self.offset_std[1])
        
        scale = random.gauss(self.scale_mean, self.scale_std)
        w *= scale
        h *= scale
        
        # detetron2 will clip the coords, so we donot need to clip to (0, height) and (0, width)
        x_min, y_min, x_max, y_max = center_x - w / 2, center_x + w / 2, center_y - h / 2, center_y + h / 2
        
        return np.array([[x_min, y_min, x_max, y_max]])


# Yolo augment input format
class BBoxNoise(DualTransform):
    """Flip the input vertically around the x-axis.

    Args:
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(self, offset_mean=(0, 0), offset_std=(3, 3), scale_mean=1, scale_std=0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
        self.offset_mean = offset_mean
        self.offset_std = offset_std
        self.scale_mean = scale_mean
        self.scale_std = scale_std

    def apply(self, img, **params):
        return img

    def _clip(self, a, min_, max_):
        return max(min(a, max_), min_)

    def apply_to_bbox(self, bbox, **params):
        rows = params["rows"]
        cols = params["cols"]
        x_min, y_min, x_max, y_max = denormalize_bbox(bbox, rows, cols)
        
        center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
        w, h = x_max - x_min, y_max - y_min
        
        center_x += random.gauss(self.offset_mean[0], self.offset_std[0])
        center_y += random.gauss(self.offset_mean[1], self.offset_std[1])
        
        scale = random.gauss(self.scale_mean, self.scale_std)
        w *= scale
        h *= scale
        
        x_min, y_min, x_max, y_max = center_x - w / 2, center_x + w / 2, center_y - h / 2, center_y + h / 2
        
        bbox = self._clip(x_min, 0, cols), self._clip(y_min, 0, rows), self._clip(x_max, 0, cols), self._clip(y_max, 0, rows)
        return normalize_bbox(bbox, rows, cols)
