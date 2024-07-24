import os
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import Detect
from ultralytics.nn.tasks import DetectionModel

from trainers.yolo.custom_validator import MyDetectionValidator
from trainers.yolo.mt_trainer import MeanTeacherTrainer

def fix_ddp_timeout():
        def _setup_ddp(self, world_size):
            torch.cuda.set_device(RANK)
            self.device = torch.device('cuda', RANK)
            LOGGER.info(f'DDP settings: RANK {RANK}, WORLD_SIZE {world_size}, DEVICE {self.device}')
            dist.init_process_group('nccl' if dist.is_nccl_available() else 'gloo', rank=RANK, world_size=world_size)


def train_yolos(data="yolo_config/wsi_patch_10.yaml"):
    model = YOLO(data)

    model.train(data=data, epochs=64, device=[0, 1, 2, 3, 4, 5, 6, 7], batch=64, imgsz=1536, workers=16, scale=0, mosaic=0, plots=True, show=True, save=True, name="wsi_patch_8gpu", lr0=0.01/2)


def train2():
    model = YOLO("runs/detect/wsi_patch3/weights/last.pt")

    model.train(data="yolo_config/wsi_patch_10.yaml", epochs=3, device=[0, 1, 2, 3, 4, 5], batch=48, imgsz=1536, workers=16, scale=0, mosaic=0, plots=True, show=True, save=True, name="wsi_patch_8gpu", lr0=0.01/2)


def train_yolom(data="yolo_config/wsi_patch_10.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=12 * 8):
    from trainers.yolo.custom_trainer import CustomTrainer
    
    CustomTrainer.free_normalize = False
    model.train(data=data, trainer=CustomTrainer, epochs=50, device=device, batch=batch_size, imgsz=3840//4, workers=2, mosaic=1, scale=0.1, plots=True, show=True, save=True, name="yolom_nc20_finetune_wsi_shishi_v3#", label_smoothing=0.1, save_period=2, resume=True, lr0=1e-3)

def train_yolox(data="yolo_config/shishi_wsi_v2_20_mini.yaml", device=[0, 1, 2, 3, 4, 5, 6, 7], batch_size=12 * 8):
    # model = YOLO("/nasdata/share/abp_trainer/pretrained_models/yolov8m_backbone_nc_20.pt")
    model = YOLO(data)
    # model = YOLO("runs/detect/yolom_nc20_finetune_wsi_shishi_v22/weights/best.pt")
    from trainers.yolo.custom_trainer import CustomTrainer
    
    CustomTrainer.free_normalize = False
    model.train(data=data, trainer=CustomTrainer, epochs=100, device=device, batch=batch_size, imgsz=3840//4, workers=2, mosaic=1, scale=0.4, plots=True, show=True, save=True, name="yolox_nc20", label_smoothing=0.1, save_period=2, resume=False, lr0=1e-2)

def train_yolom_weighted_loss(data="yolo_config/wsi_patch_10.yaml"):
    model = YOLO(data)

    from trainers.yolo.other_normal_trainers import BCEWeightTrainer
    # BCEWeightTrainer.bce_pos_weights = 
    trainer = BCEWeightTrainer
    model.train(trainer=trainer, data=data, epochs=100, device=[0, 1, 6, 7], batch=96, imgsz=3840//4, workers=16, scale=0.2, plots=True, show=True, save=True, name="yolom_nc20_bce_weighted", label_smoothing=0.1, save_period=2, resume=False)

def train_yolom_fn(data="yolo_config/wsi_patch_10.yaml"):
    model = YOLO(data)

    from trainers.yolo.custom_trainer import CustomTrainer
    
    CustomTrainer.free_normalize = True
    model.train(trainer=CustomTrainer, data=data, epochs=100, device=[0,4], batch=12, imgsz=3840//4, workers=16, scale=0.2, plots=True, show=True, save=True, name="yolom_nc20_finetune_wsi_shishi_v2", label_smoothing=0.1, save_period=2, amp=False)

def export_yolom():
    model = YOLO("runs/detect/yolom34/weights/best.pt")
    model.export(format='onnx', imgsz=960, simplify=True, dynamic=False, opset=12, batch=1)

def export_yolonn():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
    # model = YOLO("runs/detect/yolom_nc20_finetune_wsi_shishi_v22/weights/best.pt")
    model = YOLO("/nasdata/private/zwlu/Now/runs/detect/yolonn_cn_3_v2_7364/weights/best.pt")
    model.export(format='onnx', imgsz=736, simplify=True, dynamic=False, opset=12, batch=1)

def export_yolos():
    # model = YOLO("runs/detect/yolom_nc20_finetune_wsi_shishi_v22/weights/best.pt")
    model = YOLO("/nasdata/private/zwlu/Now/runs/detect/yolos_cn_3_finetune_v23/weights/best.pt")
    model.export(format='onnx', imgsz=960, simplify=True, dynamic=False, opset=12, batch=1)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    train_yolom(data="yolo_config/shishi_wsi_v2_20.yaml", device=[4, 5, 6, 7], batch_size=72)



# end main
