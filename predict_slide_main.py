# Copyright (c) Tencent Inc. All Rights Reserved
from dataclasses import dataclass
import os
import argparse
import json
import multiprocessing as mp
import logging
from glob import glob
from pathlib import Path
import time

import torch
import pandas as pd
import numpy as np

from predictor import Predictor, setup_cfg
from slide_detect_tools.slide_grid_patch import SdpcReader, OpenReader, IblReader
from slide_detect_tools.help import remove_overlap, nms_boxes
import warnings

warnings.filterwarnings("ignore")

import logging
from tools.logging_helper import logger_init, get_default_logger


# worker_logger = logging.Logger(__name__)

class WSIAsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    """
    class _StopToken:
        pass
    
    class _EndToken:
        """WSI crop end"""
        def __init__(self, slide_id, ratio, width, height, num_patchs):
            self.slide_id = slide_id
            self.ratio = ratio
            self.width = width
            self.height = height
            self.num_patchs = num_patchs

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, patch_queue, result_queue):
            super().__init__()
            self.cfg = cfg
            self.task_queue = patch_queue
            self.result_queue = result_queue

        def run(self):
            predictor = Predictor(self.cfg)
            print('start predictor', os.getpid())
            while True:
                task = self.task_queue.get()
                if isinstance(task, WSIAsyncPredictor._StopToken):
                    break
                if isinstance(task, WSIAsyncPredictor._EndToken):
                    self.result_queue.put(task)
                    continue
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))
                
    class _SlideReader(mp.Process):
        def __init__(self, params, slide_queue, patch_queue, preseg=False, rank=0):
            super().__init__()
            self.task_queue = slide_queue
            self.result_queue = patch_queue
            self.params = params
            self.rank = rank
            self.pre_seg = preseg
            
        def run(self):
            reader_worker_logger = get_default_logger(f"{self.name}_{self.rank}", "logs")

            # get slide file and crop to patch
            reader_worker_logger.info(f'start slide reader {os.getpid()}')
            from slide_detect_tools.detect_performance import (
                check_inside,
                get_cir,
                draw_regions
            )
            reader_start = time.time()
            reader_worker_logger.info("reader start ...")
            while True:
                task = self.task_queue.get() # slide file
                
                reader_worker_logger.info("Slide: %s" % (task,))
                if isinstance(task, WSIAsyncPredictor._StopToken):
                    break
                try:
                    if task.split('.')[-1] == "sdpc":
                        crop_reader = SdpcReader(task, self.params)
                    elif task.split('.')[-1] == "ibl":
                        crop_reader = IblReader(task, self.params)
                    else:
                        crop_reader = OpenReader(task, self.params)
                    regions = crop_reader.get_crop_region()
                    
                    if self.pre_seg:
                        scale = 64
                        reader_worker_logger.info("start find circle")
                        start_time = time.time_ns()
                        res = get_cir(task, scale=scale, plot=True, r_scale=0.85)
                        cost_time = (time.time_ns() - start_time) / 1e6
                        reader_worker_logger.info(f"end find circle, cost: {cost_time} ms")
                    backgroud_cnt = 0
                    
                except Exception as e:
                    reader_worker_logger.error("Slide %s read error: %s" % (task, str(e)))
                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, 0, 0, 0, 0))
                else:
                   
                    num_patchs = 0
                    start_time = time.time_ns()
                    
                    outside_regions = []
                    
                    for x, y in regions:
                        try:
                            if self.pre_seg:
                                is_inside = check_inside((x / scale, y / scale), crop_reader.crop_size_w_ / scale, crop_reader.crop_size_h_ / scale, res)
                                if not is_inside:
                                    outside_regions.append((x, y, crop_reader.crop_size_w_, crop_reader.crop_size_h_))
                                    backgroud_cnt += 1
                                    continue

                            patchs = crop_reader.crop_patch(x, y)
                            for patch_id, patch_image in patchs:
                                if patch_id is not None:
                                    self.result_queue.put((patch_id, patch_image))
                                    num_patchs += 1
                        except Exception as e:
                            reader_worker_logger.error(f"crop error {e}")
                            
                    cost_time = (time.time_ns() - start_time) / 1e6
                    reader_worker_logger.info(f"{backgroud_cnt / len(regions)}, {backgroud_cnt}/{len(regions)}, cost: {cost_time} ms")
                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, crop_reader.ratio,
                                crop_reader.width, crop_reader.height, num_patchs))
                    
                    # view the regions
                    if len(outside_regions) > 0:
                        draw_regions(task, scale, outside_regions)
                    
                finally:
                    if crop_reader:
                        crop_reader.close()
                    cost_all = time.time() - reader_start
                    reader_worker_logger.info(f"reader end, cost: {cost_all} s")
    
    class _PostprocessWorker(mp.Process):
        def __init__(self, output_path, result_queue, total, nms_boxes_threshold=10):
            super().__init__()
            self.result_queue = result_queue
            self.output_path = output_path
            self.results = {}
            self.end_tokens = {}
            self.count = 0
            self.total = total
            self.nms_boxes_threshold = nms_boxes_threshold
            self.bar = None
            
        
        @staticmethod
        def get_max(scores, idxs=[0, 1, 2, 4]):
            #  idx2name = {0: 'ASCUS', 1: 'LSIL', 2: 'HSIL', 4: 'AGC'}
            data = np.array(scores, dtype=np.float32)
            mask = np.zeros((len(data),), dtype=np.bool_)
            for idx in idxs:
                mask = mask | (data[:, 0] == idx)
            data = data[mask]
            return max(data)

        def run(self):
            start = time.time()
            nms_time = 0
            import tqdm
            self.bar = tqdm.tqdm(range(self.total), total=self.total, desc="Post")

            print('start post', os.getpid())
            while True:
                task = self.result_queue.get()

                if isinstance(task, WSIAsyncPredictor._StopToken):
                    break

                slide_end_flag = False
                if isinstance(task, WSIAsyncPredictor._EndToken):
                    # bad slide
                    if task.num_patchs == 0:
                        self.total += 1
                        continue

                    slide_id = task.slide_id
                    if slide_id not in self.results:
                        cur_slide_patchs = 0
                    else:
                        cur_slide_patchs = len(self.results[slide_id])
                    self.end_tokens[slide_id] = task
                    slide_end_flag = cur_slide_patchs >= task.num_patchs
                    if not slide_end_flag:
                        print('Warning slide %s, %d, %d' % (slide_id, cur_slide_patchs, task.num_patchs))
                else:
                    patch_id, (classes, scores, bboxes) = task
                    slide_id = "_".join(patch_id.split('_')[:-2])
                    if slide_id not in self.results:
                        self.results[slide_id] = []
                    if len(bboxes) > 0:
                        #keep = remove_overlap(bboxes, scores, classes)
                        start_time = time.time_ns()
                        keep = nms_boxes(bboxes, scores, nms_threshold=0.2, use_torchvision=False)
                        t = (time.time_ns() - start_time) / 1000
                        nms_time += t

                        classes = classes[keep]
                        scores = scores[keep]
                        bboxes = bboxes[keep]
                        self.results[slide_id].append((patch_id, classes, scores, bboxes))
                    else:
                        self.results[slide_id].append(None)
                    if slide_id in self.end_tokens:
                        slide_end_flag = len(self.results[slide_id]) >= self.end_tokens[slide_id].num_patchs

                if slide_end_flag:
                    task = self.end_tokens[slide_id]
                    ratio = task.ratio
                    slide_results = []
                    max_score = 0
                    for patch_result in self.results[slide_id]:
                        if patch_result is None:
                            continue
                        patch_id, classes, scores, bboxes = patch_result
                        names = patch_id.split('_')
                        start_x, start_y = int(names[-2]), int(names[-1].split('.')[0])
                        bboxes *= ratio
                        bboxes[:, 0::2] += start_x
                        bboxes[:, 1::2] += start_y
                        for b, s, c in zip(bboxes, scores, classes):
                            slide_results.append((
                                int(c), round(float(s), 3), int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                            if int(c) in [0, 1, 2, 4] and float(s) > max_score:
                                max_score = float(s)
                    with open(os.path.join(self.output_path, str(slide_id)+".json"), "w") as f:
                        ser_data = json.dumps({
                                    "max_score": max_score,
                                    "bboxes": slide_results,
                                    "slide_id": slide_id,
                                    "width": task.width,
                                    "height": task.height,
                                })
                        f.write(ser_data)

                    self.count += 1
                    self.bar.update(1)
                    self.bar.set_description_str(slide_id)


                    if self.count == self.total:
                        break
                    # delete slide results
                    del self.results[slide_id]
                    print("-" * 80)
                    print("Slide %s is done" % (slide_id,))
                    print("-" * 80, flush=True)
                    
            self.bar.close()
            print('end post', os.getpid(), time.time() - start, nms_time / 1000)



    def __init__(self, slide_files, num_gpus, num_slide_workers, params, output_path, det_cfg, bboxes_threshold=1e9, preseg=False):
        """
        Args:
            wsi_files (list): wsi path list
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
            num_wsi_workers (int): num workers to read WSI file
        """
        
        # init queues
        num_gpus = max(num_gpus, 1)
        self.slide_queue = mp.Queue(maxsize=len(slide_files)+num_slide_workers+2)
        self.patch_queue = mp.Queue(maxsize=num_gpus * 3)
        self.result_queue = mp.Queue(maxsize=num_gpus * 3)
        
        self.num_predictor_workers = num_gpus
        self.num_slide_workers = num_slide_workers
        
        self.model_procs = []
        # model predictor workers
        for gpuid in range(num_gpus):
            cfg = det_cfg.copy()
            cfg.train.device = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.model_procs.append(
                WSIAsyncPredictor._PredictWorker(cfg, self.patch_queue, self.result_queue)
            )
       
        self.slide_procs = []
        # slide reader workers
        slide_reader_params = {
            "crop_pixel_size": params["crop_pixel_size"] ,
            "crop_size_h": params["crop_size_h"],
            "crop_size_w": params["crop_size_w"],
            "crop_overlap": params["crop_overlap"],
            "crop_level": params["crop_level"]
        }
        for idx in range(num_slide_workers):
            self.slide_procs.append(
                WSIAsyncPredictor._SlideReader(slide_reader_params, self.slide_queue, self.patch_queue, rank=idx, preseg=preseg)
            )
            
        for p in self.model_procs:
            p.start()
        for p in self.slide_procs:
            p.start()
        
        import tqdm
        # add slide
        for slide_file in slide_files:
            self.slide_queue.put(slide_file)
        # add end token
        for _ in range(self.num_slide_workers):
            self.slide_queue.put(WSIAsyncPredictor._StopToken())
        
        # postprocess worker
        self.post_proc = WSIAsyncPredictor._PostprocessWorker(output_path, self.result_queue, len(slide_files), nms_boxes_threshold=bboxes_threshold)
        self.post_proc.start()
        self.post_proc.join()
        
        # shut model worker
        for _ in range(self.num_predictor_workers):
            self.patch_queue.put(WSIAsyncPredictor._StopToken())
        

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default='.',
        help="config file"
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        help="wsi slide path csvs"
    )
    parser.add_argument(
        "--weights",
        help="wsi slide path csv"
    )
    parser.add_argument(
        "--output",
        default='.',
        help="A directory to save output"
    )
    parser.add_argument(
        "--num-gpus",
        default=-1,
        type=int,
        help="GPU number"
    )
    parser.add_argument(
        "--num-workers",
        default=64,
        type=int,
        help="worker number"
    )
    parser.add_argument(
        "--other-args",
        type=str,
        required=False,
        help="string parameters for debug"
    )
    parser.add_argument(
        "--pre-segment",
        action="store_true",
        help="segment the image to reduce the crop number,"
    )
    return parser


def get_filename(path):
    return Path(path).stem

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = get_default_logger(os.path.basename(__file__), "logs")

    params = {
        "crop_pixel_size": 0.31,
        "crop_size_h": 1280,
        "crop_size_w": 1280,
        "crop_overlap": 64,
        "crop_level": 0,
    }
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    logger.info(args.inputs)
    logger.info(args.output)
    logger.info("start ...")
    start_time = time.time()
    for input_csv_file in args.inputs:
        dataset_id =    get_filename(input_csv_file)
        output_dir = os.path.join(args.output, dataset_id)
        os.makedirs(output_dir, exist_ok=True)
        finished_slides = [get_filename(f) for f \
             in glob(os.path.join(output_dir, "*.json"))]

        # white csv
        df = pd.read_csv(input_csv_file)
        print("input files:")
        print(df)

        wsi_files_ = list(df["slide_path"])
        
        wsi_files = []
        for wsi_file in wsi_files_:
            if str(wsi_file) == 'nan':
                continue
            
            if get_filename(wsi_file) not in finished_slides:
                wsi_files.append(wsi_file)
                
        if len(wsi_files) == 0:
            continue

        det_cfg = setup_cfg(args.config_file)
        det_cfg.train.init_checkpoint = args.weights
        
        num_gpus = torch.cuda.device_count()
        if args.num_gpus != -1:
            num_gpus = args.num_gpus

        WSIAsyncPredictor(wsi_files, num_gpus, args.num_workers, params, output_dir, det_cfg, bboxes_threshold=int(args.other_args), preseg=args.pre_segment)
    
    cost_time = time.time() - start_time
    logger.info(f"end, cost: {cost_time} s")
