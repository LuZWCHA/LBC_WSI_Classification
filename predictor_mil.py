import os
import logging
import argparse
from numpy import imag
import tqdm
import torch
from glob import glob

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

    def __call__(self, wsi_data):
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
            pred = self.model(None, wsi_data)
        pred = pred[0].to("cpu").numpy()
        pred_score = [float(pred.max())] + pred.tolist()
        return pred_score


def setup_cfg(config_file):
    # load config from file and command-line arguments
    cfg = LazyConfig.load(config_file)
    #cfg = LazyConfig.apply_overrides(cfg, args.opts)
    # Set score_threshold for builtin models
    #cfg.model.test_score_thresh = args.confidence_threshold
    return cfg

class ClsBranchPredictor:
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

    def __call__(self, wsi_data):
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
            pred = self.model(wsi_data)
        pred = pred[0].to("cpu").numpy()
        pred_score = [float(pred.max())] + pred.tolist()
        return pred_score


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
    parser.add_argument(
        "--weights", default=None, type=str, help="the model weights, path of checkpoint"
    )
    args = parser.parse_args()

    det_cfg = setup_cfg(args.config_file)
    det_cfg.train.init_checkpoint = args.weights
    det_cfg.train.device = "cuda:7"

    predictor = Predictor(det_cfg)
    
    npy_files = glob("/mnt/group-ai-medical-abp/private/huye/datasets/cro2_lsil_fn/*.npy")
    outputs = {}
    for npy_file in tqdm(npy_files):
        image_data = np.load(npy_file)
        images = []
        for image in image_data:
            images.append(cv2.cvtColor(image[64:64+384, 64:64+640], cv2.COLOR_RGB2BGR))

        pred = predictor({"images": images, "label": 0})
        image_id = os.path.basename(npy_file).split('.')[0]
        outputs[image_id] = {
        }

        print(pred)
    
    with open(os.path.join(args.output_path, "result.json"), "w") as f:
        f.write(json.dumps(outputs))
        
