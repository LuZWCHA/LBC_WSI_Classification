# Copyright (c) Tencent Inc. All Rights Reserved
import os
import argparse
import json
import multiprocessing as mp
import logging
from glob import glob

import torch
import pandas as pd
import numpy as np

from predictor_feat import Predictor, setup_cfg
from slide_detect_tools.slide_grid_patch import SdpcReader, OpenReader
from slide_detect_tools.help import remove_overlap, nms_boxes


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
        def __init__(self, params, slide_queue, patch_queue):
            super().__init__()
            self.task_queue = slide_queue
            self.result_queue = patch_queue
            self.params = params

        def run(self):
            # get slide file and crop to patch
            print('start slide reader', os.getpid())
            while True:
                task = self.task_queue.get() # slide file
                print("Slide: %s" % (task,))
                if isinstance(task, WSIAsyncPredictor._StopToken):
                    break
                try:
                    if task.split('.')[-1] == "sdpc":
                        crop_reader = SdpcReader(task, self.params)
                    else:
                        crop_reader = OpenReader(task, self.params)
                    regions = crop_reader.get_crop_region()
                except Exception as e:
                    print("Slide %s read error: %s" % (task, str(e)))
                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, 0, 0, 0, 0))
                else:
                    num_patchs = 0
                    for x, y in regions:
                        try:
                            patchs = crop_reader.crop_patch(x, y)
                            for patch_id, patch_image in patchs:
                                if patch_id is not None:
                                    self.result_queue.put((patch_id, patch_image))
                                    num_patchs += 1
                        except Exception as e:
                            print("crop error {}".format(str(e)))

                    crop_reader.close()
                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, crop_reader.ratio,
                                crop_reader.width, crop_reader.height, num_patchs))
                
    class _PostprocessWorker(mp.Process):
        def __init__(self, output_path, result_queue, total):
            super().__init__()
            self.result_queue = result_queue
            self.output_path = output_path
            self.results = {}
            self.end_tokens = {}
            self.count = 0
            self.total = total

        def run(self):

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
                    patch_id, (classes, scores, bboxes, features) = task
                    slide_id = "_".join(patch_id.split('_')[:-2])
                    if slide_id not in self.results:
                        self.results[slide_id] = []
                    if len(bboxes) > 0:
                        #keep = remove_overlap(bboxes, scores, classes)
                        keep = nms_boxes(bboxes, scores, nms_threshold=0.2)
                        classes = classes[keep]
                        scores = scores[keep]
                        bboxes = bboxes[keep]
                        features = features[keep]
                        self.results[slide_id].append((patch_id, classes, scores, bboxes, features))
                    else:
                        self.results[slide_id].append(None)
                    if slide_id in self.end_tokens:
                        slide_end_flag = len(self.results[slide_id]) >= self.end_tokens[slide_id].num_patchs
                    
                if slide_end_flag:
                    task = self.end_tokens[slide_id]
                    ratio = task.ratio
                    slide_results = []
                    box_features = []
                    for patch_result in self.results[slide_id]:
                        if patch_result is None:
                            continue
                        patch_id, classes, scores, bboxes, features = patch_result
                        names = patch_id.split('_')
                        start_x, start_y = int(names[-2]), int(names[-1].split('.')[0])
                        bboxes *= ratio
                        bboxes[:, 0::2] += start_x
                        bboxes[:, 1::2] += start_y
                        for b, s, c, f in zip(bboxes, scores, classes, features):
                            slide_results.append((
                                int(c), "%.3f" % (float(s)), int(b[0]), int(b[1]), int(b[2]), int(b[3])))
                            box_features.append(f.tolist()) 
                    with open(os.path.join(self.output_path, str(slide_id)+".json"), "w") as f:
                        f.write(json.dumps({
                                    "bboxes": slide_results,
                                    "features": box_features,
                                    "slide_id": slide_id,
                                    "width": task.width,
                                    "height": task.height,
                                }))
                    self.count += 1
                    if self.count == self.total:
                        break
                    # delete slide results
                    del self.results[slide_id]
                    print("slide %s is done" % (slide_id,))


    def __init__(self, slide_files, num_gpus, num_slide_workers, params, output_path, det_cfg):
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
        for _ in range(num_slide_workers):
            self.slide_procs.append(
                WSIAsyncPredictor._SlideReader(slide_reader_params, self.slide_queue, self.patch_queue)
            )
            
        for p in self.model_procs:
            p.start()
        for p in self.slide_procs:
            p.start()
            
        # add slide
        for slide_file in slide_files:
            self.slide_queue.put(slide_file)
        # add end token
        for _ in range(self.num_slide_workers):
            self.slide_queue.put(WSIAsyncPredictor._StopToken())
        
        # postprocess worker
        self.post_proc = WSIAsyncPredictor._PostprocessWorker(output_path, self.result_queue, len(slide_files))
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
        help="A directory to save output"
    )
    parser.add_argument(
        "--num-workers",
        default=64,
        type=int,
        help="A directory to save output"
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()

    params = {
        "crop_pixel_size": 0.31,
        "crop_size_h": 1280,
        "crop_size_w": 1280,
        "crop_overlap": 64,
        "crop_level": 0,
    }
    
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    print(args.inputs, args.output)
    for input_csv_file in args.inputs:
        dataset_id = os.path.basename(input_csv_file).split('.')[0]
        output_dir = os.path.join(args.output, dataset_id)
        os.makedirs(output_dir, exist_ok=True)
        finished_slides = [os.path.basename(f).split('.')[0] for f \
             in glob(os.path.join(output_dir, "*.json"))]

        # white csv
        df = pd.read_csv(input_csv_file)
        wsi_files_ = list(df["slide_path"])
        wsi_files = []
        for wsi_file in wsi_files_:
            if str(wsi_file) == 'nan':
                continue
            if os.path.basename(wsi_file).split('.')[0] not in finished_slides:
                wsi_files.append(wsi_file)
                
        if len(wsi_files) == 0:
            continue

        det_cfg = setup_cfg(args.config_file)
        det_cfg.train.init_checkpoint = args.weights
        
        num_gpus = torch.cuda.device_count()
        if args.num_gpus != -1:
            num_gpus = args.num_gpus
        WSIAsyncPredictor(wsi_files, num_gpus, args.num_workers, params, output_dir, det_cfg)
