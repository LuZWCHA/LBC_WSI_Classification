import torch
from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from omegaconf import OmegaConf
import detectron2.data.transforms as T
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.backbone.fpn import LastLevelP6P7
from detectron2.modeling.backbone import BasicStem, FPN, ResNet
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
import albumentations as A

from txdet.data.dataset_mapper import DatasetMapper
from ..common.data.coco import dataloader
from exp.retinanet_mil import RetinaNetM, RetinaNetHead, ImageClsHead
from exp.mil import MILClassifier
from ..common.models.retinanet import model
from txdet.modeling.backbone import ConvNeXt


########## model ############
model.backbone = L(FPN)(
    bottom_up=L(ConvNeXt)(
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.2,
        layer_scale_init_value=1.0,
        out_features=["stage2", "stage3", "stage4"],
        ),
    in_features=["stage2", "stage3", "stage4"],
    out_channels=1,
    top_block=L(LastLevelP6P7)(in_channels="${..out_channels}", out_channels="${..out_channels}", in_feature="p5")
)

######### optimizer #########
lr_multiplier = L(WarmupParamScheduler)(
        scheduler=L(MultiStepParamScheduler)(
            values=[1.0, 0.1, 0.01],
            # note that scheduler is scale-invariant. This is equivalent to
            # milestones=[22, 28, 30]
            milestones=[80000, 100000, 120000],
            #milestones=[15000, 20000, 25000]
        ),
        warmup_length=500 / 120000,
        warmup_method="linear",
        warmup_factor=0.001,
    )

optimizer = L(torch.optim.SGD)(
    params=L(get_default_optimizer_params)(
        # params.model is meant to be set to the model object, before instantiating
        # the optimizer.
        weight_decay_norm=0.0
    ),
    lr=0.005,
    momentum=0.9,
    weight_decay=1e-4,
)

########## data #############
dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="abp_tctc3_train", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(288, 320, 352, 416, 384, 416, 432),
                sample_style="choice",
                max_size=768,
            ),
            L(T.RandomFlip)(horizontal=True, vertical=False),
            L(T.RandomFlip)(horizontal=False, vertical=True),
            L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.5),
            L(T.RandomSaturation)(intensity_min=0.8, intensity_max=1.5),
        ],
        image_format="BGR",
        albu_augmentations=[
            A.CLAHE(p=0.65),
            A.OneOf([
                A.GaussNoise(p=0.8),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8) 
            ], p=0.65),
            A.Blur(blur_limit=[3, 5], p=0.3), # small
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.8),
                A.ToGray(p=0.7),
            ], p=0.65),
            A.CoarseDropout(max_holes=64, max_height=8, max_width=16, min_height=8, min_width=16, p=0.5),
            A.ImageCompression(quality_lower=30, quality_upper=100, p=0.4),
        ],
        use_instance_mask=False,
    ),
    total_batch_size=8,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="abp_tctc3_test", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=360, max_size=640),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=8,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


########## train ############
train = dict(
    output_dir="./output",
    # init_checkpoint="detectron2://ImageNetPretrained/MSRA/R-50.pkl",
    init_checkpoint="/nasdata/private/jli/WSI_framework/detectron/output-contrast-2view-samplemax/model_0022999.pth",
    max_iter=120000,
    amp=dict(enabled=True),  # options for Automatic Mixed Precision
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    checkpointer=dict(period=10000, max_to_keep=5),  # options for PeriodicCheckpointer
    eval_period=10000,
    log_period=20,
    device="cuda"
    # ...
)

