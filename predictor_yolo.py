import os
import logging
from pathlib import Path
from typing import List
from numpy import imag
import torch

# from detectron2.data.detection_utils import read_image
# from detectron2.utils.logger import setup_logger
# from detectron2.data import MetadataCatalog
# from detectron2.utils.visualizer import ColorMode, Visualizer
# from detectron2.config import LazyConfig, instantiate
# from detectron2.checkpoint import DetectionCheckpointer
# import detectron2.data.transforms as T

from ultralytics import YOLO
from matplotlib import pyplot as plt
import numpy as np
from ultralytics.engine.results import Results

logger = logging.getLogger("yolo")
import cv2

class YoloPredictor:
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

    def __init__(self, yolo_model, device="cuda:0", params={}):
        self.model = YOLO(yolo_model)
        logger.info("Model:\n{}".format(self.model))
        self.model.to(device)
        #             "crop_size_h": 1280,
        # "crop_size_w": 1280,
        # self.h, self.w = params["crop_size_h"], params["crop_size_w"]
        self.input_format = "RGB"
            
    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        # with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        #     # Apply pre-processing to image.
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
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        # TODO
        results: Results = self.model.predict(image, conf=0.01, verbose=False)

        # return the last one result
        for result in results:
            result = result.cpu().numpy()
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmenation masks outputs
            # probs = result.probs  # Class probabilities for classification outputs
            bboxes = boxes.xyxy
            classes = boxes.cls
            scores = boxes.conf
        
        del image
        return classes, scores, bboxes

    def predict_and_show(self, original_image, save_path="", yolo_label_path=None):
        # if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            # original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]

        # don't use aug
        """
        aug_input = T.AugInput(original_image)
        transforms = self.aug(aug_input)
        image = aug_input.image
        """
        image = original_image
        # image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        res: List[Results] = self.model.predict(image, conf=0.20)
        
        
        if yolo_label_path:
            # boxes=None, masks=None, probs=None
            # xyxy, (track_id), conf, cls
            
            if os.path.exists(yolo_label_path):
                with open(yolo_label_path, "r") as f:
                    boxes_lines = f.readlines()
                box_list = []
                for line in boxes_lines:
                    class_id, cx, cy, ow, oh = [float(i.strip()) for i in line.split(" ")]
                    # class_id = int(class_id)
                    box_list.append([(cx - ow / 2) * width, (cy - oh / 2) * height, (cx + ow / 2) * width, (cy + oh / 2) * height, 1, class_id])
                    
                num_boxes = len(box_list)
                print(f"Number of boxes: {num_boxes}")
                if num_boxes > 0:
                    label_boxes = np.array(box_list)
                    old_boxes = res[0].boxes.boxes
                    # old_probs = res[0].boxes.conf
                    
                    print("Positive sample.")
                    
                    res[0].update(boxes=np.concatenate((old_boxes, label_boxes)))
            else:
                print(f"Warning: label not find: {yolo_label_path}")

        for result in res:
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmenation masks outputs
            # probs = result.probs  # Class probabilities for classification outputs
            
            res  = result.plot()
            
            bboxes = boxes.xyxy
            classes = boxes.cls
            scores = boxes.conf
            # image = image.transpose(2, 0, 1)
            # image = image[None, ...]
            print(bboxes, classes, scores)
            # plt.imshow(res)
            if len(bboxes) > 0:
                plt.imsave(save_path, res[..., ::-1])
        
        del image
        return classes, scores, bboxes
    
    
# TODO
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

    def __init__(self, yolo_model, device="cuda:0", params={}):
            self.model = YOLO(yolo_model)
            logger.info("Model:\n{}".format(self.model))
            self.model.to(device)
            #             "crop_size_h": 1280,
            # "crop_size_w": 1280,
            # self.h, self.w = params["crop_size_h"], params["crop_size_w"]
            self.input_format = "RGB"

    def __call__(self, original_images):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).
        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        
        # Apply pre-processing to image.
        if self.input_format == "RGB":
            # whether the model expects BGR inputs or RGB
            original_images = [cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) for original_image in original_images]
        # height, width = original_images[0].shape[:2]

        # don't use aug
        """
        aug_input = T.AugInput(original_image)
        transforms = self.aug(aug_input)
        image = aug_input.image
        """
        images = original_images
        # print("2")
        results: Results = self.model.predict(images, conf=0.01, verbose=False)
        # print("3")
        res = []
        for result in results:
            result = result.cpu().numpy()
            boxes = result.boxes  # Boxes object for bbox outputs
            # masks = result.masks  # Masks object for segmenation masks outputs
            # probs = result.probs  # Class probabilities for classification outputs
            bboxes = boxes.xyxy
            classes = boxes.cls
            scores = boxes.conf
            res.append((classes, scores, bboxes))
        for i in images:
            del i

        return res

# def setup_cfg(config_file):
#     # load config from file and command-line arguments
#     cfg = LazyConfig.load(config_file)
#     #cfg = LazyConfig.apply_overrides(cfg, args.opts)
#     # Set score_threshold for builtin models
#     #cfg.model.test_score_thresh = args.confidence_threshold
#     return cfg



if __name__ == '__main__':
    import argparse
    import pandas as pd
    import json
    import os
    import cv2
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("--config-file", default="", type=str, help="image root path")
    parser.add_argument('--input', default="/nasdata/dataset/wsi_patch_data/train_yolo.txt", type=str)
    parser.add_argument("--output-path", default="", type=str, help="trainset label path")
    parser.add_argument("--output-format", default="dict", type=str, help="trainset label path")
    parser.add_argument("--visual", action="store_true")
    parser.add_argument(
        "--weights", default=None, type=str, help="the model weights, path of checkpoint"
    )
    parser.add_argument(
        "--batch-size", default=1, type=int, help="the model weights, path of checkpoint"
    )
    args = parser.parse_args()

    # det_cfg = setup_cfg(args.config_file)
    # det_cfg.train.init_checkpoint = args.weights
    # det_cfg.train.device = "cuda:7"

    if args.input.endswith(".csv"):
        data = pd.read_csv(args.input)
    elif args.input.endswith(".txt"):
        with open(args.input, mode="r") as f:
            a = f.readlines()
        data = {"path": a}
    else:
        data = {"path": [args.input]}
        
    image_id = 0
    os.makedirs(args.output_path, exist_ok=True)
    
    batch_size = args.batch_size
    
    if batch_size > 1 and not args.visual:
        predictor = BatchedPredictor(args.weights)
    else:
        predictor = YoloPredictor(args.weights)
    
    if args.visual:
        os.makedirs(os.path.join(args.output_path, "visual"), exist_ok=True)
    
    outputs = {}
    batch = []
    batch_info = []
    total_size = len(data["path"])
    cnt = 0
    for image_file in tqdm(data["path"]):
        cnt += 1
        image_file = image_file.strip()
        # print(image_file)
        image = cv2.imread(image_file.strip())
        
        if batch_size > 1:
            batch.append(image)
            batch_info.append(image_file)
            if len(batch) >= batch_size or cnt >= total_size:
                # classes, scores, bboxes = predictor(image)
                # yolo_label_path = image_file.replace("images", "labels").replace(".jpg", ".txt")
                # print(yolo_label_path)
                if not args.visual:
                    res = predictor(image)
                else:
                    raise RuntimeError("Not support!")
                
                for image_file, (classes, scores, bboxes) in zip(batch_info, res):
                    # image_id = image_file.split('/')[-3] + '_' + Path(image_file).stem
                    if args.output_format == "dict":
                        outputs[image_id] = {
                            "class": classes.tolist(),
                            "score": scores.tolist(),
                            "bboxes": bboxes.tolist(),
                            "image_file": image_file
                        }
                    else:
                        subdir = os.path.join(args.output_path, Path(image_file).parent.name)
                        os.makedirs(subdir, exist_ok=True)
                        names = Path(image_file).name.split("_")
                        idx = names[0]
                        cid = names[1]
                        conf = names[2].replace(".jpg", "")
                        save_path = os.path.join(subdir, idx + ".json")
                        with open(save_path, "w") as f:
                            f.write(json.dumps({
                                "class": classes.tolist(),
                                "score": scores.tolist(),
                                "bboxes": bboxes.tolist(),
                                "v3_dectection_result": [int(idx), int(cid), float(conf)],
                                "image_file": image_file
                            }))
                    image_id += 1
                batch = []
                batch_info = []
        else:
            if not args.visual:
                classes, scores, bboxes = predictor(image)
            else:
                image_pred_save_path = os.path.join(args.output_path, "visual", Path(image_file).name)
                print(image_pred_save_path)
                classes, scores, bboxes = predictor.predict_and_show(image, image_pred_save_path)
            # image_id = image_file.split('/')[-3] + '_' + Path(image_file).stem
            if args.output_format == "dict":
                outputs[image_id] = {
                    "class": classes.tolist(),
                    "score": scores.tolist(),
                    "bboxes": bboxes.tolist(),
                    "image_file": image_file
                }
            else:
                subdir = os.path.join(args.output_path, Path(image_file).parent.name)
                os.makedirs(subdir, exist_ok=True)
                names = Path(image_file).name.split("_")
                idx = names[0]
                cid = names[1]
                conf = names[2].replace(".jpg", "")
                save_path = os.path.join(subdir, idx + ".json")
                with open(save_path, "w") as f:
                    f.write(json.dumps({
                        "class": classes.tolist(),
                        "score": scores.tolist(),
                        "bboxes": bboxes.tolist(),
                        "v3_dectection_result": [int(idx), int(cid), float(conf)],
                        "image_file": image_file
                    }))
            image_id += 1
    
    
    if args.output_format == "dict":
        if args.output_path is not None or args.output_path != "":
            with open(os.path.join(args.output_path, "result.json"), "w") as f:
                f.write(json.dumps(outputs))
        
