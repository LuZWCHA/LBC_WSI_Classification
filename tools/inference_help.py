# This file is to generate the test file names of the predictors

import os, glob, pandas as pd
import random

import os
from pathlib import Path

import numpy as np
import torch

# from ultralytics.data import build_dataloader
# from ultralytics.data.dataloaders import create_dataloader
# from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import ops
# from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
# from ultralytics.utils.plotting import output_to_target, plot_images
# from ultralytics.utils.torch_utils import de_parallel

def gen_data_file_names_from(file_dir, output="inference.csv", random_sample=False, sample_count=0):
    files = glob.glob(os.path.join(file_dir, "**/*"), recursive=True)
    name = "slide_path"
    files = filter(lambda x: os.path.isfile(x), files)

    slide_paths = {
        name: []
    }
    for f in files:
        slide_paths[name].append(f)

    if random_sample:
        random_sampled = random.sample(slide_paths[name], sample_count)
        slide_paths[name] = random_sampled

    pd.DataFrame(slide_paths).to_csv(output, index=False)
    print("Export successful.")


def onehot(value: int, class_num=3):
    if value <= 0:
        return [0] * class_num
    res = []
    for _ in range(class_num):
        res.append(value & 1)
        value >>= 1
    return res
    
    
def gen_name2label_dict(dataset_label_file, class_num=3):
    csv_data = pd.read_csv(dataset_label_file)
    print(csv_data)
    res = {}
    name2path = {}
    
    for idx in csv_data.index:
        item = csv_data.iloc[idx]
        slide_path = item["slide_path"] if "slide_path" in item else item["slide_path_sdpc"]
        slide_id = os.path.basename(slide_path).split('.')[0]
        grade_name = item["grade"]
        slide_label = onehot(int(item["grade"]), class_num=class_num)
        res[slide_id] = slide_label
        name2path[slide_id] = slide_path
    
    return res, name2path

def gen_name2label_dict_name(dataset_label_file):
    csv_data = pd.read_csv(dataset_label_file)
    print(csv_data)
    res = {}
    name2path = {}
    
    for idx in csv_data.index:
        item = csv_data.iloc[idx]
        slide_path = item["slide_path"] if "slide_path" in item else item["slide_path_sdpc"]
        slide_id = os.path.basename(slide_path).split('.')[0]
        # grade_name = item["grade"]
        slide_label = item["grade_6"]
        res[slide_id] = slide_label
        name2path[slide_id] = slide_path
    
    return res, name2path

def gen_path2label_dict(dataset_label_file, class_num=3):
    csv_data = pd.read_csv(dataset_label_file)
    print(csv_data)
    res = {}
    
    for idx in csv_data.index:
        item = csv_data.iloc[idx]
        slide_path = item["slide_path"] if "slide_path" in item else item["slide_path_sdpc"]
        slide_label = onehot(int(item["grade"]), class_num=class_num)
        res[slide_path] = slide_label
    
    return res

import torch, numpy as np

def get_currect_box_matrix(detections, labels, iouv=torch.linspace(0.5, 0.95, 10)):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
                                1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=detections.device)

def update_metrics(preds, batch, stats, confusion_matrix: ConfusionMatrix, device, single_cls=False, box_convert=ops.xywh2xyxy, iouv=torch.linspace(0.5, 0.95, 10)):
    niou = iouv.numel()
    # Metrics
    for si, pred in enumerate(preds):
        idx = si == batch['batch_idx']
        cls = batch['cls'][idx]
        bbox = batch['bboxes'][idx]
        nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
        shape = batch['ori_shape'][si]
        correct_bboxes = torch.zeros(npr, niou, dtype=torch.bool, device=device)  # init

        if npr == 0:
            if nl:
                stats.append((correct_bboxes, *torch.zeros((2, 0), device=device), cls.squeeze(-1)))
                confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
            continue

        # Predictions
        if single_cls:
            pred[:, 5] = 0
        predn = pred.clone()
        ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                        ratio_pad=batch['ratio_pad'][si])  # native-space pred

        # Evaluate
        if nl:
            height, width = batch['img'][si].shape[1:]
            tbox = box_convert(bbox) * torch.tensor(
                (width, height, width, height), device=device)  # target boxes
            
            ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
            labelsn = torch.cat((cls, tbox), 1)  # native-space labels
            
            correct_bboxes = get_currect_box_matrix(predn, labelsn, iouv=iouv)
            confusion_matrix.process_batch(predn, labelsn)
        stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)


def get_stats(metrics: DetMetrics, stats, nc):
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        metrics.process(*stats)
    nt_per_class = np.bincount(stats[-1].astype(int), minlength=nc)  # number of targets per class
    return metrics, nt_per_class


def postprocess(preds, conf=0.25, iou=0.75):
    preds = ops.non_max_suppression(preds,
                                    conf,
                                    iou,
                                    multi_label=True)
    return preds

# ############################   from yolo v8  #########################
# # Ultralytics YOLO 🚀, GPL-3.0 license

# class DetectionValidator(BaseValidator):

#     def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None):
#         super().__init__(dataloader, save_dir, pbar, args)
#         self.args = args
#         self.is_coco = False
#         self.class_map = None
#         self.metrics = DetMetrics(save_dir=self.save_dir)
#         self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
#         self.niou = self.iouv.numel()

#     # preprocess
#     # normalized
#     # input batch should has: img, batch_idx, cls, bboxes
#     def preprocess(self, batch):
#         batch['img'] = batch['img'].to(self.device, non_blocking=True)
#         batch['img'] = (batch['img'].half() if self.args.half else batch['img'].float()) / 255
#         for k in ['batch_idx', 'cls', 'bboxes']:
#             batch[k] = batch[k].to(self.device)

#         nb = len(batch['img'])
#         self.lb = [torch.cat([batch['cls'], batch['bboxes']], dim=-1)[batch['batch_idx'] == i]
#                    for i in range(nb)] if self.args.save_hybrid else []  # for autolabelling

#         return batch

#     def init_metrics(self, names):
#         self.is_coco = False
#         self.class_map = ops.coco80_to_coco91_class() if self.is_coco else list(range(1000))
#         self.args.save_json |= self.is_coco and not self.training  # run on final val if training COCO
#         self.names = names
#         self.nc = len(names)
#         self.metrics.names = self.names
#         self.metrics.plot = self.args.plots
#         self.confusion_matrix = ConfusionMatrix(nc=self.nc)
#         self.seen = 0
#         self.jdict = []
#         self.stats = []

#     def get_desc(self):
#         return ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)')

#     def postprocess(self, preds):
#         preds = ops.non_max_suppression(preds,
#                                         self.args.conf,
#                                         self.args.iou,
#                                         labels=self.lb,
#                                         multi_label=True,
#                                         agnostic=self.args.single_cls,
#                                         max_det=self.args.max_det)
#         return preds

#     #  input batch should has: ori_shape, batch_idx, cls, bboxes, ratio_pad, img
#     # preds should like: b (x1 x2 y1 y2 conf class)
#     def update_metrics(self, preds, batch):
#         # Metrics
#         for si, pred in enumerate(preds):
#             idx = batch['batch_idx'] == si
#             cls = batch['cls'][idx]
#             bbox = batch['bboxes'][idx]
#             nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
#             shape = batch['ori_shape'][si]
#             correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
#             self.seen += 1

#             if npr == 0:
#                 if nl:
#                     self.stats.append((correct_bboxes, *torch.zeros((2, 0), device=self.device), cls.squeeze(-1)))
#                     if self.args.plots:
#                         self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
#                 continue

#             # Predictions
#             if self.args.single_cls:
#                 pred[:, 5] = 0
#             predn = pred.clone()
#             ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
#                             ratio_pad=batch['ratio_pad'][si])  # native-space pred

#             # Evaluate
#             if nl:
#                 height, width = batch['img'].shape[2:]
#                 tbox = ops.xywh2xyxy(bbox) * torch.tensor(
#                     (width, height, width, height), device=self.device)  # target boxes
#                 ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
#                                 ratio_pad=batch['ratio_pad'][si])  # native-space labels
#                 labelsn = torch.cat((cls, tbox), 1)  # native-space labels
#                 correct_bboxes = self._process_batch(predn, labelsn)
#                 # TODO: maybe remove these `self.` arguments as they already are member variable
#                 if self.args.plots:
#                     self.confusion_matrix.process_batch(predn, labelsn)
#             self.stats.append((correct_bboxes, pred[:, 4], pred[:, 5], cls.squeeze(-1)))  # (conf, pcls, tcls)

#             # Save
#             if self.args.save_json:
#                 self.pred_to_json(predn, batch['im_file'][si])
#             # if self.args.save_txt:
#             #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

#     def finalize_metrics(self, *args, **kwargs):
#         self.metrics.speed = self.speed

#     def get_stats(self):
#         stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy
#         if len(stats) and stats[0].any():
#             self.metrics.process(*stats)
#         self.nt_per_class = np.bincount(stats[-1].astype(int), minlength=self.nc)  # number of targets per class
#         return self.metrics.results_dict

#     def print_results(self):
#         pf = '%22s' + '%11i' * 2 + '%11.3g' * len(self.metrics.keys)  # print format
#         LOGGER.info(pf % ('all', self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
#         if self.nt_per_class.sum() == 0:
#             LOGGER.warning(
#                 f'WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels')

#         # Print results per class
#         if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
#             for i, c in enumerate(self.metrics.ap_class_index):
#                 LOGGER.info(pf % (self.names[c], self.seen, self.nt_per_class[c], *self.metrics.class_result(i)))

#         if self.args.plots:
#             self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))

#     def _process_batch(self, detections, labels):
#         """
#         Return correct prediction matrix
#         Arguments:
#             detections (array[N, 6]), x1, y1, x2, y2, conf, class
#             labels (array[M, 5]), class, x1, y1, x2, y2
#         Returns:
#             correct (array[N, 10]), for 10 IoU levels
#         """
#         iou = box_iou(labels[:, 1:], detections[:, :4])
#         correct = np.zeros((detections.shape[0], self.iouv.shape[0])).astype(bool)
#         correct_class = labels[:, 0:1] == detections[:, 5]
#         for i in range(len(self.iouv)):
#             x = torch.where((iou >= self.iouv[i]) & correct_class)  # IoU > threshold and classes match
#             if x[0].shape[0]:
#                 matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]),
#                                     1).cpu().numpy()  # [label, detect, iou]
#                 if x[0].shape[0] > 1:
#                     matches = matches[matches[:, 2].argsort()[::-1]]
#                     matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
#                     # matches = matches[matches[:, 2].argsort()[::-1]]
#                     matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
#                 correct[matches[:, 1].astype(int), i] = True
#         return torch.tensor(correct, dtype=torch.bool, device=detections.device)

#     def get_dataloader(self, dataset_path, batch_size):
#         # TODO: manage splits differently
#         # calculate stride - check if model is initialized
#         gs = max(int(de_parallel(self.model).stride if self.model else 0), 32)
#         return create_dataloader(path=dataset_path,
#                                  imgsz=self.args.imgsz,
#                                  batch_size=batch_size,
#                                  stride=gs,
#                                  hyp=vars(self.args),
#                                  cache=False,
#                                  pad=0.5,
#                                  rect=self.args.rect,
#                                  workers=self.args.workers,
#                                  prefix=colorstr(f'{self.args.mode}: '),
#                                  shuffle=False,
#                                  seed=self.args.seed)[0] if self.args.v5loader else \
#             build_dataloader(self.args, batch_size, img_path=dataset_path, stride=gs, names=self.data['names'],
#                              mode='val')[0]

#     def plot_val_samples(self, batch, ni):
#         plot_images(batch['img'],
#                     batch['batch_idx'],
#                     batch['cls'].squeeze(-1),
#                     batch['bboxes'],
#                     paths=batch['im_file'],
#                     fname=self.save_dir / f'val_batch{ni}_labels.jpg',
#                     names=self.names)

#     def plot_predictions(self, batch, preds, ni):
#         plot_images(batch['img'],
#                     *output_to_target(preds, max_det=15),
#                     paths=batch['im_file'],
#                     fname=self.save_dir / f'val_batch{ni}_pred.jpg',
#                     names=self.names)  # pred

#     def pred_to_json(self, predn, filename):
#         stem = Path(filename).stem
#         image_id = int(stem) if stem.isnumeric() else stem
#         box = ops.xyxy2xywh(predn[:, :4])  # xywh
#         box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
#         for p, b in zip(predn.tolist(), box.tolist()):
#             self.jdict.append({
#                 'image_id': image_id,
#                 'category_id': self.class_map[int(p[5])],
#                 'bbox': [round(x, 3) for x in b],
#                 'score': round(p[4], 5)})

#     def eval_json(self, stats):
#         if self.args.save_json and self.is_coco and len(self.jdict):
#             anno_json = self.data['path'] / 'annotations/instances_val2017.json'  # annotations
#             pred_json = self.save_dir / 'predictions.json'  # predictions
#             LOGGER.info(f'\nEvaluating pycocotools mAP using {pred_json} and {anno_json}...')
#             try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
#                 check_requirements('pycocotools>=2.0.6')
#                 from pycocotools.coco import COCO  # noqa
#                 from pycocotools.cocoeval import COCOeval  # noqa

#                 for x in anno_json, pred_json:
#                     assert x.is_file(), f'{x} file not found'
#                 anno = COCO(str(anno_json))  # init annotations api
#                 pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
#                 eval = COCOeval(anno, pred, 'bbox')
#                 if self.is_coco:
#                     eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
#                 eval.evaluate()
#                 eval.accumulate()
#                 eval.summarize()
#                 stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = eval.stats[:2]  # update mAP50-95 and mAP50
#             except Exception as e:
#                 LOGGER.warning(f'pycocotools unable to run: {e}')
#         return stats


# def val(cfg=DEFAULT_CFG, use_python=False):
#     model = cfg.model or 'yolov8n.pt'
#     data = cfg.data or 'coco128.yaml'

#     args = dict(model=model, data=data)
#     if use_python:
#         from ultralytics import YOLO
#         YOLO(model).val(**args)
#     else:
#         validator = DetectionValidator(args=args)
#         validator(model=args['model'])


if __name__ == "__main__":
    # gen_data_file_names_from("/nasdata/ai_data/dt3", output="dt3_sample_100.csv", random_sample=True, sample_count=100)
    gen_name2label_dict("/nasdata/ai_data/labels/dt3.csv")





