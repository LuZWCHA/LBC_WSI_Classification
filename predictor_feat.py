import os
import logging
import argparse
import tqdm
import torch

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

import datasets


logger = logging.getLogger("detectron2")


class Predictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
            self.model = instantiate(cfg.model)
            logger.info("Model:\n{}".format(self.model))
            self.model.to(cfg.train.device)
            self.model.eval()
            
            if isinstance(cfg.dataloader.test.dataset.names, str):
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
            else:
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names[0])
      
            DetectionCheckpointer(self.model).load(cfg.train.init_checkpoint)
            self.aug = T.AugmentationList(
                [instantiate(aug) for aug in cfg.dataloader.test.mapper.augmentations]
            )
            self.input_format = cfg.model.input_format
            assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]

            """
            aug_input = T.AugInput(original_image)
            transforms = self.aug(aug_input)
            image = aug_input.image
            """
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            predictions = predictions["instances"].to("cpu")
            
            scores = predictions.scores.numpy()
            bboxes = predictions.pred_boxes.tensor.numpy()
            classes = predictions.pred_classes.numpy()
            cls_features = predictions.cls_features.numpy()
            
            return classes, scores, bboxes, cls_features
            

class BatchedPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.
    Compared to using the model directly, this class does the following additions:
    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.
    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.
    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.
    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
            self.model = instantiate(cfg.model)
            logger.info("Model:\n{}".format(self.model))
            self.model.to(cfg.train.device)
            self.model.eval()
            
            if isinstance(cfg.dataloader.test.dataset.names, str):
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
            else:
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names[0])
      
            DetectionCheckpointer(self.model).load(cfg.train.init_checkpoint)
            self.aug = T.AugmentationList(
                [instantiate(aug) for aug in cfg.dataloader.test.mapper.augmentations]
            )
            self.input_format = cfg.model.input_format
            assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_images = [i[:, :, ::-1] for i in original_images]
            height, width = original_images[0].shape[:2]

            """
            aug_input = T.AugInput(original_image)
            transforms = self.aug(aug_input)
            image = aug_input.image
            """
            # image = original_image
            # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            inputs =[ {"image": torch.as_tensor(image.astype("float32").transpose(2, 0, 1)), "height": height, "width": width} for image in original_images]
            predictions = self.model(inputs)
            res = []
            for i in range(len(inputs)):
                instances = predictions[i]["instances"]
                instances = instances.to("cpu")
                
                scores = instances.scores.numpy()
                bboxes = instances.pred_boxes.tensor.numpy()
                classes = instances.pred_classes.numpy()
                cls_features = instances.cls_features.numpy()
            
                del instances
                res.append((classes, scores, bboxes, cls_features))
            for i in inputs:
                del i
            # del inputs
            # del original_images

        return res

def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = LazyConfig.load(config_file)
    #cfg = LazyConfig.apply_overrides(cfg, args.opts)
    # Set score_threshold for builtin models
    #cfg.model.test_score_thresh = args.confidence_threshold
    return cfg


if __name__ == '__main__':
    import argparse
    import pandas as pd
    import json
    import os
    import cv2
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config-file", default="", type=str, help="image root path")
    parser.add_argument('--input', default="/mnt/group-ai-medical-sz/shared/TCT_datasets/wsi_classification/shishi_patches/shishi_6_classes.csv", type=str)
    parser.add_argument("--output-path", default="/mnt/group-ai-medical-sz/shared/TCT_datasets/wsi_classification/shishi_patches/shishi_6_classes_results", type=str, help="trainset label path")
    parser.add_argument(
        "--weights", default=None, type=str, help="the model weights, path of checkpoint"
    )
    args = parser.parse_args()

    det_cfg = setup_cfg(args.config_file)
    det_cfg.train.init_checkpoint = args.weights
    det_cfg.train.device = "cuda:7"
:
    predictor = Predictor(det_cfg)

    data = pd.read_csv(args.input)
    outputs = {}
    image_files = data["image_path"]
    idx = 0

    for image_file in tqdm(image_files):
        image = cv2.imread(image_file)
        image = cv2.resize(image, (1280, 720))
        classes, scores, bboxes, feature = predictor(image)
        image_id = os.path.basename(image_file).split('.')[0]
        outputs[image_id] = {
            "class": classes.tolist(),
            "score": scores.tolist(),
            "box": bboxes.tolist(),
            "feature": feature.tolist(),
        }

        if len(outputs) == 10000:
            with open(os.path.join(args.output_path, f"result_{idx}.json"), "w") as f:
                f.write(json.dumps(outputs))
            outputs = {}
            idx += 1
            
    with open(os.path.join(args.output_path, "result_end.json"), "w") as f:
        f.write(json.dumps(outputs))
        
