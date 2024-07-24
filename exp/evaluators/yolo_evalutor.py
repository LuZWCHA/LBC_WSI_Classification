import math
from pathlib import Path
import cv2
from sympy import threaded
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.ops import xyxy2xywh, xyn2xy, xywh2xyxy
from detectron2.evaluation import DatasetEvaluator
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager

from tools.inference_help import get_currect_box_matrix, postprocess, update_metrics, get_stats
from ultralytics.utils.plotting import output_to_target, plot_images, plot_labels
from torchvision.ops import nms

try:
    from detectron2.evaluation.fast_eval_api import COCOeval_opt
except ImportError:
    COCOeval_opt = COCOeval


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


class YoloEvalutor(DatasetEvaluator):
    
    def __init__(self, dataset_name, plot_interval=200, distributed=True, output_dir=None, names=[], device=torch.device("cpu")) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self._distributed = distributed
        self._logger = logging.getLogger(__name__)
        self._output_dir = output_dir
        self._predictions = []
        self._device = device
        self.nc = len(names)
        self.stats = []
        self.confusion_matrix = ConfusionMatrix(self.nc, conf=0.45)
        self.names = names
        self.metrics = DetMetrics(save_dir=Path(self._output_dir), names=self.names_list2dict(), plot = True)
        self.polt_flag = True
        self.seen = 0
        self.plot_interval = plot_interval
        
        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path, allow_cached=True)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self.plot_labels = True
        # if self._do_evaluation:
        #     self._kpt_oks_sigmas = kpt_oks_sigmas
        # if comm.is_main_process():
        #     annos = self._coco_api.anns
        
        #     gt_boxes = []
        #     cls_ids = []

        #     for key, anno in annos.items():
        #         bbox = anno["bbox"]
        #         cls_id = anno["category_id"]
        #         t = self._coco_api.imgs[anno['image_id']]
        #         h, w = t['height'], t['width']
        #         gt_boxes.append(bbox)
        #         cls_ids.append(cls_id)
                
        #     gt_boxes = np.array(gt_boxes)
        #     gt_classes = np.array(cls_ids)
                    
        #     gt_boxes[:, 0] = gt_boxes[:, 0] + gt_boxes[:, 2] // 2
        #     gt_boxes[:, 1] = gt_boxes[:, 1] + gt_boxes[:, 3] // 2
            
        #     gt_boxes[:, 0::2] /=  w
        #     gt_boxes[:, 1::2] /=  h
            
        #     print(gt_boxes.shape)
            
        #     plot_labels(gt_boxes, gt_classes, names=self.names_list2dict(), save_dir=Path(self._output_dir))

        
    def reset(self):
        self._predictions = []
        self.stats = []
        self.confusion_matrix.matrix *= 0

        self.polt_flag = True
        self.seen = 0
        self.metrics.save_dir = Path(self._output_dir)
        if self.plot_labels:
            
            self.plot_labels = False
        
        return super().reset()
    
    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

            pred, batch = self._to_yolo_format(input, output, device=self._device)
            # preds.append(pred)
            # batches.append(batch)
            self.seen += 1
            if self.plot_interval > 0 and self.seen % self.plot_interval == 0 and comm.is_main_process():
                self.plot_val_samples(batch, self.seen)
                self.plot_predictions(batch, pred, self.seen)
            
            update_metrics(pred, batch, self.stats, self.confusion_matrix, self._device)
        
        return super().process(inputs, outputs)
    
    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))
            stats  = comm.gather(self.stats, dst=0)
            stats = list(itertools.chain(*stats))
            confusion_matrix_list  = comm.gather(self.confusion_matrix, dst=0)

            if not comm.is_main_process():
                return {}
            else:
                cm = ConfusionMatrix(self.nc)
                for confusion_matrix in confusion_matrix_list:
                    cm.matrix += confusion_matrix.matrix
                confusion_matrix = cm
        else:
            predictions = self._predictions
        
        
        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}    
        
        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)
        
        results, nt_per_class = get_stats(self.metrics, stats, self.nc)#.results_dict

        # results.maps
        
        from tools.visual.plot_utils import plot_many
        # print(results_dict)
        confusion_matrix.plot(save_dir = self._output_dir, names=self.names)
        plot_many(nt_per_class, titles=["number of classes"], default_type="bar", show=False).savefig(os.path.join(self._output_dir, "classes_number.jpg"))
        import pickle
        pickle.dump(results, open(os.path.join(self._output_dir, "instances_results.pth"), "wb"))
        return results.results_dict
    
    # transform coco format to yolo format
    def _to_yolo_format(self, inputs, output, device):
        
        img = inputs["image"]
        file_name = inputs["file_name"]
        image_id = inputs["image_id"]
        width = inputs["width"]
        height = inputs["height"]
        # cls_label = inputs["image_label"]

        annos = self._coco_api.imgToAnns[image_id]
        
        gt_boxes = []
        cls_ids = []
        for anno in annos:
            bbox = anno["bbox"]
            cls_id = anno["category_id"]
            gt_boxes.append(bbox)
            cls_ids.append(cls_id)
            
        gt_boxes = torch.tensor(gt_boxes)
        gt_classes = torch.tensor(cls_ids)

        instances = output["instances"]
        num_instance = len(instances)

        boxes = instances.pred_boxes.tensor

        img_shape = img.shape[1:]
        scale = height / img_shape[0]

        if scale != 1 and len(boxes) > 0:
            boxes /= scale

        if len(gt_boxes) > 0:
            # coco xywh to xyxy
            gt_boxes[:, 2] = gt_boxes[:, 2] + gt_boxes[:, 0]
            gt_boxes[:, 3] = gt_boxes[:, 3] + gt_boxes[:, 1]
            gt_boxes[:, 0::2] /= width
            gt_boxes[:, 1::2] /= height

        scores = instances.scores
        classes = instances.pred_classes
        
        bn = gt_classes.shape[0]
        
        if num_instance > 0:
            kept = nms(boxes, scores, 0.75)
            boxes = boxes[kept]
            scores = scores[kept]
            classes = classes[kept]
        
        
        # boxes + scores + classes
        # nd array shape: (N, 6)
        preds = torch.concat([boxes, scores[..., None], classes[..., None]], dim=1).unsqueeze(0).to(device)

        # gt_boxes: ndarray shape: (N, 4) xywh
        # ori_shape: (h, w)
        # gt_classes: ndarray shape: (N, )
        
        batch = {"batch_idx": torch.tensor([0] * bn), "ori_shape": [(height, width)], "cls": gt_classes.unsqueeze(-1).to(device), "bboxes": gt_boxes.to(device), "ratio_pad": [None], "img": img.unsqueeze(0).to(device), "im_file": [file_name]}
        bboxes =  batch['bboxes']
        
        # coco xyxy to yolo xywh
        if len(batch['cls']) > 0:
            batch['bboxes'] =  xyxy2xywh(bboxes)
        return preds, batch

    def names_list2dict(self):
        return {i: name for i, name in enumerate(self.names)}

    def plot_val_samples(self, batch, ni):
        save_path = Path(self._output_dir).joinpath("val_images")
        save_path.mkdir(parents=True, exist_ok=True)
        
        plot_images(batch['img'],
                    batch['batch_idx'],
                    batch['cls'].squeeze(-1),
                    batch['bboxes'],
                    paths=batch['im_file'],
                    fname=save_path / f'val_batch{ni}_labels.jpg',
                    names=self.names_list2dict())

    def plot_predictions(self, batch, preds, ni):
        save_path = Path(self._output_dir).joinpath("val_images")
        save_path.mkdir(parents=True, exist_ok=True)
        
        plot_images(batch['img'],
                    *output_to_target(preds, max_det=15),
                    paths=batch['im_file'],
                    fname=save_path / f'val_batch{ni}_pred.jpg',
                    names=self.names_list2dict())  # pred
