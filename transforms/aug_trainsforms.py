

import random, numpy as np
import logging



LOGGER = logging.Logger("Albumentations")

class Albumentations:

    def __init__(self, p=1.0):
        self.p = p
        self.transform = None
        self.ids = [0, 2, 4]  # IDs to apply augmentation to
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),]  # transforms
            self.transform = A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: add supports of segments and keypoints
            
            # Check if any of the label IDs are in the list of IDs to apply augmentation to
            if any(i in self.ids for i in cls):
                if self.transform and random.random() < self.p:
                    new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    
            labels["instances"].update(bboxes=bboxes)
        return labels


