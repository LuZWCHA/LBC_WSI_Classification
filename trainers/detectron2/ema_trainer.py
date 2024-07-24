import os
import logging
import time

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
import detectron2.data.common

from txdet.utils.logger import setup_logger

import torch, torch
from copy import deepcopy
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as torchdata


class EMASimpleTrainer(SimpleTrainer):
    def __init__(self, model, data_loader, optimizer, ema_alpha=0.999):
        super().__init__(model, data_loader, optimizer)
        self.ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            ema_alpha * averaged_model_parameter + (1 - ema_alpha) * model_parameter
        # self.swa_model: AveragedModel = AveragedModel(model, device=model.device, avg_fn=self.ema_avg, use_buffers=True)
        # self.iter: int = 0
        # self.start_iter: int = 0
        # self.max_iter: int
        # ema_model = deepcopy.copy(self.model)
        self.epoch_len = 0
        dataset = data_loader.dataset.dataset
        batch_size = data_loader.batch_size
        if isinstance(dataset, torchdata.IterableDataset):
            if isinstance(dataset, detectron2.data.common.ToIterableDataset):
                dataset = dataset.dataset

        self.epoch_len = len(dataset) // batch_size
        
        assert self.epoch_len > 0
        
    def update_swa_model(self):
        if self.epoch_len > 0 and self.iter % self.epoch_len == 0: 
            self.swa_model.update_parameters(model=self.model)
            
    def init_swa_model(self):
        self.swa_model = AveragedModel(self.model, device=self.model.device, avg_fn=self.ema_avg, use_buffers=True)
        
    def run_step(self):
        res =  super().run_step()
        self.update_swa_model()
        return res

class EMAAMPTrainer(AMPTrainer):
    
    def __init__(self, model, data_loader, optimizer, grad_scaler=None, ema_alpha=0.99):
        super().__init__(model, data_loader, optimizer, grad_scaler)
        self.ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged:\
                            ema_alpha * averaged_model_parameter + (1 - ema_alpha) * model_parameter
        # self.swa_model: AveragedModel = AveragedModel(model, device=model.device, avg_fn=self.ema_avg, use_buffers=True)
        # self.iter: int = 0
        # self.start_iter: int = 0
        # self.max_iter: int
        # ema_model = deepcopy.copy(self.model)
        self.epoch_len = 0
        dataset = data_loader.dataset.dataset
        batch_size = data_loader.batch_size
        if isinstance(dataset, torchdata.IterableDataset):
            if isinstance(dataset, detectron2.data.common.ToIterableDataset):
                dataset = dataset.dataset

        self.epoch_len = len(dataset) // batch_size

        assert self.epoch_len > 0
    
    def init_swa_model(self):
        self.swa_model = AveragedModel(self.model, device=self.model.device, avg_fn=self.ema_avg, use_buffers=True)
    
    def run_step(self):
        res = super().run_step()
        self.update_swa_model()
        return res
    
    def update_swa_model(self):
        if self.epoch_len > 0 and self.iter % self.epoch_len == 0: 
            self.swa_model.update_parameters(model=self.model)
        
        
class MeanTeacherDetectModelTrainer(EMASimpleTrainer):
    
    def __init__(self, model, data_loader, unlabeled_data_loader, optimizer, ema_alpha=0.99):
        super().__init__(model, data_loader, optimizer, ema_alpha)
        self.unlabeled_loader = unlabeled_data_loader
        self.ul_loader_iter = iter(unlabeled_data_loader)
    
    def before_train(self):
        self.swa_model.eval()
        self.swa_model.module.eval()
        self.swa_model.module.training = False
        return super().before_train()
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        ul_data = next(self.ul_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        with torch.no_grad():
            pesudo_label = self.swa_model(ul_data, return_pl=True)

        loss_dict = self.model(data)
        loss_dict_ul = self.model(ul_data, pl=pesudo_label)
        
        loss_dict.update(loss_dict_ul)
        
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()
        
        self.update_swa_model()

class MeanTeacherDetectModelAMPTrainer(EMAAMPTrainer):
    
    def __init__(self, model, data_loader, unlabeled_data_loader, optimizer, grad_scaler=None, ema_alpha=0.999):
        super().__init__(model, data_loader, optimizer, grad_scaler, ema_alpha)
        self.unlabeled_loader = unlabeled_data_loader
        self.ul_loader_iter = iter(unlabeled_data_loader)
    
    def before_train(self):
        self.swa_model.eval()
        self.swa_model.module.eval()
        self.swa_model.module.training = False
        return super().before_train()
    
    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        ul_data = next(self.ul_loader_iter)

        data_time = time.perf_counter() - start

        with autocast():
            with torch.no_grad():
                pesudo_label = self.swa_model(ul_data, return_pl=True)

            loss_dict = self.model(data)
            loss_dict_ul = self.model(ul_data, pl=pesudo_label)
            
        loss_dict.update(loss_dict_ul)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()        
        self.update_swa_model()