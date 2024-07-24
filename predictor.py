import os
import logging
import argparse
from pathlib import Path
import time
from numpy import imag
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
import numpy as np

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

            # don't use aug
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
        
        return classes, scores, bboxes


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
                # cls_features = instances.cls_features.numpy()
            
                del instances
                res.append((classes, scores, bboxes))
            for i in inputs:
                del i

        return res


def adj_brt_const(image_path):
    if isinstance(image_path, str):
        bgr = cv2.imread(image_path)
    else:
        bgr = image_path
        
    alpha = 0.7  # 对比度调整参
    beta = 30    # 亮度调整参数
    new_img = cv2.addWeighted(bgr, alpha, np.zeros(bgr.shape, dtype=bgr.dtype), 0, beta)
    return new_img

def apply_CLAHE(image_path, gridsize=8):
    if isinstance(image_path, str):
        bgr = cv2.imread(image_path)
    else:
        bgr = image_path

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = list(cv2.split(lab))
    clahe = cv2.createCLAHE(clipLimit=2,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr


class PredictorWithPreprocess:
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

    def __init__(self, cfg, preprocess=None):
            self.model = instantiate(cfg.model)
            logger.info("Model:\n{}".format(self.model))
            self.model.to(cfg.train.device)
            self.model.eval()
            self.preprocess = preprocess
            
            if isinstance(cfg.dataloader.test.dataset.names, str):
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
            else:
                self.metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names[0])
            print(cfg.train.init_checkpoint)
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
            if self.preprocess:
                original_image = self.preprocess(original_image)
            height, width = original_image.shape[:2]

            # don't use aug
            """
            aug_input = T.AugInput(original_image)
            transforms = self.aug(aug_input)
            image = aug_input.image
            """
            image = original_image
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            # print(image.max(), image.min())

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]

            predictions = predictions["instances"].to("cpu")
            
            scores = predictions.scores.numpy()
            bboxes = predictions.pred_boxes.tensor.numpy()
            classes = predictions.pred_classes.numpy()
        
        return classes, scores, bboxes



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
    parser.add_argument('--input', default="", type=str)
    parser.add_argument("--output-path", default="", type=str, help="trainset label path")
    parser.add_argument("--preprocess", default="", type=str, help="trainset label path")
    parser.add_argument("--output-format", default="dict", type=str, help="trainset label path")
    parser.add_argument(
        "--weights", default="/nasdata/private/zwlu/Now/ai_trainer/output/r50_wsi_patch_nc20/model_final.pth", type=str, help="the model weights, path of checkpoint"
    )
    args = parser.parse_args()

    det_cfg = setup_cfg(args.config_file)
    if args.weights is not None:
        det_cfg.train.init_checkpoint = args.weights
    det_cfg.train.device = "cuda:0"
    preprocess = None
    
    os.makedirs(args.output_path, exist_ok=True)
    
    if args.preprocess == "CLAHE":
        preprocess = apply_CLAHE
    elif args.preprocess == "BRT_CONT":
        preprocess = adj_brt_const
    
    predictor = PredictorWithPreprocess(det_cfg, preprocess=preprocess)

    data = pd.read_csv(args.input)
    outputs = {}
    id_ = 0
    for image_file in tqdm(data["path"]):
        if args.output_format != "dict":
            subdir = os.path.join(args.output_path, Path(image_file).parent.name)
            names = Path(image_file).name.split("_")
            idx = names[0]
            save_path = os.path.join(subdir, idx + ".json")
            if os.path.exists(save_path):
                continue
        
        image = cv2.imread(image_file)
        classes, scores, bboxes = predictor(image)
        # if len(classes) > 0:
        #     print("not empty")
        image_id = id_
        classes, scores, bboxes = predictor(image)
        # image_id = image_file.split('/')[-3] + '_' + Path(image_file).stem
        if args.output_format == "dict":
            outputs[image_id] = {
                "class": classes.tolist(),
                "score": scores.tolist(),
                "bboxes": bboxes.tolist(),
                "image_file": image_file
            }
        else:
            
            os.makedirs(subdir, exist_ok=True)
            names = Path(image_file).name.split("_")
            idx = names[0]
            cid = names[1]
            conf = names[2].replace(".jpg", "")
            
            with open(save_path, "w") as f:
                f.write(json.dumps({
                    "class": classes.tolist(),
                    "score": scores.tolist(),
                    "bboxes": bboxes.tolist(),
                    "v3_detection_result": [int(idx), int(cid), float(conf)],
                    "image_file": image_file
                }))
        # image_id += 1
        id_ += 1
    
    with open(os.path.join(args.output_path, "result.json"), "w") as f:
        f.write(json.dumps(outputs))
        
