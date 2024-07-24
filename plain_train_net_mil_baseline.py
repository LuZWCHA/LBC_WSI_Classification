
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
import time
import datetime
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils.events import EventStorage
from detectron2.config import LazyConfig, instantiate
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine.hooks import HookBase
from detectron2.solver import LRMultiplier
from fvcore.common.timer import Timer
import albumentations as albu
from sklearn.metrics import roc_auc_score
import numpy as np

from txdet.utils.logger import setup_logger

import datasets

# test the wsidataset
# modified by nowandfuture
from exp.wsi_dataset_now import WSIDataset, build_wsi_test_dataloader, build_wsi_train_dataloader


logger = logging.getLogger("detectron2")

class IterationTimer(HookBase):
    """
    Track the time spent for each iteration (each run_step call in the trainer).
    Print a summary in the end of training.
    This hook uses the time between the call to its :meth:`before_step`
    and :meth:`after_step` methods.
    Under the convention that :meth:`before_step` of all hooks should only
    take negligible amount of time, the :class:`IterationTimer` hook should be
    placed at the beginning of the list of hooks to obtain accurate timing.
    """

    def __init__(self, storage, start_iter, warmup_iter=3):
        """
        Args:
            warmup_iter (int): the number of iterations at the beginning to exclude
                from timing.
        """
        self._warmup_iter = warmup_iter
        self._step_timer = Timer()
        self._start_time = time.perf_counter()
        self._total_timer = Timer()

        self.storage = storage
        self.start_iter = start_iter

    def before_train(self):
        self._start_time = time.perf_counter()
        self._total_timer.reset()
        self._total_timer.pause()

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        total_time_minus_hooks = self._total_timer.seconds()
        hook_time = total_time - total_time_minus_hooks

        num_iter = self.storage.iter + 1 - self.start_iter - self._warmup_iter

        if num_iter > 0 and total_time_minus_hooks > 0:
            # Speed is meaningful only after warmup
            # NOTE this format is parsed by grep in some scripts
            logger.info(
                "Overall training speed: {} iterations in {} ({:.4f} s / it)".format(
                    num_iter,
                    str(datetime.timedelta(seconds=int(total_time_minus_hooks))),
                    total_time_minus_hooks / num_iter,
                )
            )

        logger.info(
            "Total training time: {} ({} on hooks)".format(
                str(datetime.timedelta(seconds=int(total_time))),
                str(datetime.timedelta(seconds=int(hook_time))),
            )
        )

    def before_step(self):
        self._step_timer.reset()
        self._total_timer.resume()

    def after_step(self):
        # +1 because we're in after_step, the current step is done
        # but not yet counted
        iter_done = self.storage.iter - self.start_iter + 1
        if iter_done >= self._warmup_iter:
            sec = self._step_timer.seconds()
            self.storage.put_scalars(time=sec)
        else:
            self._start_time = time.perf_counter()
            self._total_timer.reset()

        self._total_timer.pause()

def do_test(cfg, args, model):
    # wsi dataset
    test_file = args.test_file
    albu_transform = albu.Compose([
        albu.CenterCrop(384, 640, always_apply=True, p=1.0),
        
        #albu.CLAHE(p=0.5),
        #albu.OneOf([
        #    albu.GaussNoise(p=0.8),
        #    albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8) 
        #], p=0.7),
        #albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5)
        #albu.Blur(blur_limit=[3, 3], p=0.3),
    ])
    dataset = WSIDataset(test_file, size=(640, 384), train=False, 
        albu_transform=albu_transform)
    wsi_dt = build_wsi_test_dataloader(dataset, num_workers=8)

    model.eval()
    pred_scores = []
    gt_labels = []
    img_ids = []
    for wsi_data in wsi_dt:
        with torch.no_grad():
            pred = model(wsi_data[0])

        pred = pred[0].to('cpu').numpy().max()
        gt = wsi_data[0]['label'].to('cpu').numpy().max()
        pred_scores.append(pred)
        gt_labels.append(gt)
        img_ids.append(wsi_data[0]['id'])
        
    outputs = comm.all_gather([pred_scores, gt_labels, img_ids])
    pred = np.concatenate([x[0] for x in outputs])
    gt = np.concatenate([x[1] for x in outputs])
    names = []
    for x in outputs:
        names += x[2]
    auc_score = roc_auc_score(gt, pred)
    if comm.is_main_process():
        df = pd.DataFrame({"wsi_score": pred, "slide_name": names})
        df.to_csv(os.path.join(cfg.train.output_dir, os.path.basename(test_file).split('.')[0] + ".csv"), index=False)
    model.train()
    logger.info('The auc of validation data is: %.4f' % (auc_score,))
    return {'auc': auc_score}
    
def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `common_train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optimizer = instantiate(cfg.optimizer)
    scheduler = LRMultiplier(optimizer, instantiate(cfg.lr_multiplier), cfg.train.max_iter)

    # train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    checkpointer = DetectionCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume).get("iteration", -1) + 1
    )
    if not args.resume:
        start_iter = 0
    
    max_iter = cfg.train.max_iter

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, **cfg.train.checkpointer
    )

    writers = default_writers(cfg.train.output_dir, max_iter) if comm.is_main_process() else []

    train_file = args.train_file
    albu_transform = albu.Compose([
        #albu.CenterCrop(384, 640, always_apply=True, p=1.0),
        albu.RandomCrop(384, 640, always_apply=True, p=1.0),
        albu.CLAHE(p=0.5),
        albu.OneOf([
            albu.GaussNoise(p=0.8),
            albu.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8) 
        ], p=0.7),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
        albu.Blur(blur_limit=[3, 3], p=0.3),
    ])
    dataset = WSIDataset(train_file, size=(640, 384), train=True, 
        albu_transform=albu_transform)
    
    # args.num_gpus,
    wsi_dt = build_wsi_train_dataloader(dataset, args.num_gpus, num_workers=4)

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        iter_timer = IterationTimer(storage, start_iter)
        grad_scaler = GradScaler() if cfg.train.amp.enabled else None
        iter_timer.before_train()
        # data_loader_iter = iter(train_loader)
        wsi_dt_iter = iter(wsi_dt)
        for iteration in range(start_iter, max_iter):
                storage.iter = iteration
                iter_timer.before_step()
                start = time.perf_counter()
                # data = next(data_loader_iter)
                wsi_data = next(wsi_dt_iter)
                data_time = time.perf_counter() - start
                with autocast(enabled=cfg.train.amp.enabled):
                    loss_dict = model(wsi_data[0])
                    losses = sum(loss_dict.values())
                    assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    storage.put_scalars(data_time=data_time, total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                if cfg.train.amp.enabled:
                    grad_scaler.scale(losses).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    losses.backward()
                    optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()
                iter_timer.after_step()

                if (
                    cfg.train.eval_period > 0
                    and (iteration + 1) % cfg.train.eval_period == 0
                    and iteration != max_iter - 1
                ):
                    logger.info('Start evaluation')
                    results = do_test(cfg, args, model)
                    storage.put_scalars(**results, smoothing_hint=False)
                    # Compared to "train_net.py", the test results are not dumped to EventStorage
                    comm.synchronize()

                if iteration - start_iter > 5 and (
                    (iteration + 1) % cfg.train.log_period == 0 or iteration == max_iter - 1
                ):
                    for writer in writers:
                        writer.write()
                periodic_checkpointer.step(iteration)
        iter_timer.after_train()
        do_test(cfg, args, model)

def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    setup_logger(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model, cfg.train.output_dir).resume_or_load(cfg.train.init_checkpoint,
            resume=args.resume)
        logger.info(do_test(cfg, args, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args_parser = default_argument_parser()
    args_parser.add_argument("--train-file", default="", type=str)
    args_parser.add_argument("--test-file", default="", type=str)
    args = args_parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
