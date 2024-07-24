# -*- coding: UTF-8 -*-
# **********************************************************
# * Copyright (c) 2021 Tencent
# * Author        : huye
# * Create time   : 2021-10-26 14:44
# * Last modified : 2021-10-26 14:44
# **********************************************************

import copy
import logging
import numpy as np
from typing import List, Optional, Union
import torch
import albumentations as A
import torchvision.transforms as TT
from PIL import Image

from detectron2.config import configurable

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperContrast"]


class DatasetMapperContrast:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.
    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.
    The callable currently does the following:
    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        albu_augmentations = None,
        pil_augmentations = None,
        random_erase_prob = -1,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        use_contrast: bool = False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            albu_augmentations: a list of albumentation augs, only support image only aug.
            pil_augmentations: a list of torchvision (pil) augs, only support image only aug.
            random_erase_prob: float, RandomErasing
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.albu_augmentations     = albu_augmentations
        self.pil_augmentations      = pil_augmentations
        self.random_erase_prob      = random_erase_prob
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes
        self.use_contrast           = use_contrast
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")
        if self.albu_augmentations:
            self.albu_augmentations = A.Compose(self.albu_augmentations)
            logger.info(f"[DatasetMapper] albu Augmentations used in {mode}: {self.albu_augmentations}")
        if self.pil_augmentations:
            assert self.image_format == "RGB"
            self.pil_augmentations = TT.Compose(self.pil_augmentations)
            logger.info(f"[DatasetMapper] pil Augmentations used in {mode}: {self.pil_augmentations}")
        if self.random_erase_prob > 0:
            self.random_erasing = TT.RandomErasing(self.random_erase_prob, scale=(0.02, 0.2))

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input1 = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms1 = self.augmentations(aug_input1)
        image1, sem_seg_gt1 = aug_input1.image, aug_input1.sem_seg

        aug_input2 = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms2 = self.augmentations(aug_input2)
        image2, sem_seg_gt1 = aug_input2.image, aug_input2.sem_seg

        # albu aug
        if self.is_train and self.albu_augmentations:
            image1 = self.albu_augmentations(image=image1)["image"]
            image2 = self.albu_augmentations(image=image2)["image"]

        image_shape1 = image1.shape[:2]  # h, w
        image_shape2 = image2.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.

        image_tensor1 = torch.as_tensor(np.ascontiguousarray(image1.transpose(2, 0, 1)))
        image_tensor2 = torch.as_tensor(np.ascontiguousarray(image2.transpose(2, 0, 1)))
        if self.random_erase_prob > 0:
            image_tensor1 = self.random_erasing(image_tensor1)
            image_tensor2 = self.random_erasing(image_tensor2)

        dataset_dict["image1"] = image_tensor1
        dataset_dict["image2"] = image_tensor2
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape1, transforms1, proposal_topk=self.proposal_topk
            )
            utils.transform_proposals(
                dataset_dict, image_shape2, transforms2, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict
        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms1, image_shape1)
            # self._transform_annotations(dataset_dict, transforms2, image_shape2)

        if 'image_label' in dataset_dict:
            dataset_dict["image_label"] = torch.as_tensor(dataset_dict['image_label'], dtype=torch.float32)

        if 'image_grade' in dataset_dict:
            dataset_dict["image_grade"] = torch.as_tensor(dataset_dict['image_grade'], dtype=torch.float32)

        return dataset_dict