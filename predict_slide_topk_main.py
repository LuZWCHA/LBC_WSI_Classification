# Copyright (c) Tencent Inc. All Rights Reserved
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
from slide_detect_tools.slide_crop_patch import SdpcReader, OpenSlideReader, IblReader
from slide_detect_tools.help import remove_overlap, nms_boxes
import warnings

warnings.filterwarnings("ignore")

class WSIScore(object):
        
    def __init__(self):
        self.idx2name = {0: 'ASCUS', 1: 'LSIL', 2: 'HSIL', 4: 'AGC'}
        self.idxs = [0, 1, 2, 4]
            
    def get_topk(self, json_file, k=32):
        json_data = json.load(open(json_file))
        data = np.array(json_data["bboxes"], dtype=np.float32)
        mask = np.zeros((len(data),), dtype=np.bool_)
        for idx in self.idxs:
            mask = mask | (data[:, 0] == idx)
        data = data[mask]
        ind = np.argsort(-data[:, 1])
        outputs = data[ind[:k]]

        return outputs


class WSIAsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    """
    class _StopToken:
        pass
    
    class _EndToken:
        """WSI crop end"""
        def __init__(self, slide_id, width, height, num_patchs):
            self.slide_id = slide_id
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

            print('end predictor', os.getpid())
                
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
                task = self.task_queue.get()
                if isinstance(task, WSIAsyncPredictor._StopToken):
                    break
                task = json.loads(task)
                print("Slide: %s" % (task['slide_path'], ))
                try:
                    if task['slide_path'].split('.')[-1] == "sdpc":
                        crop_reader = SdpcReader(task['slide_path'])
                    elif task['slide_path'].split('.')[-1] == 'ibl':
                        crop_reader = IblReader(task['slide_path'])
                    else:
                        crop_reader = OpenSlideReader(task['slide_path'])

                    wsi_scorer = WSIScore()
                    topk_boxes = wsi_scorer.get_topk(task['json_path'], k=32)
                except Exception as e:
                    print("Slide %s read error: %s" % (task['slide_path'], str(e)))
                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, 0, 0, 0))
                else:
                    num_patchs = 0
                    for idx, box in enumerate(topk_boxes):
                        label_id = int(box[0])
                        score = float(box[1])
                        x1, y1, x2, y2 = [int(x) for x in box[2:]]
                        x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
                        ratio = self.params["crop_pixel_size"] / crop_reader.slide_pixel_size
                        x = x_center - int(ratio * self.params["crop_patch_width"] / 2)
                        y = y_center - int(ratio * self.params["crop_patch_height"] / 2)
                        if x < 0 or y < 0:
                            continue
        
                        try:
                            patch_image = crop_reader.crop_patch(x, y, self.params["crop_patch_width"],
                                    self.params["crop_patch_height"], self.params["crop_pixel_size"], crop_level=0)
                        except Exception as e:
                            print("Crop failed: ", e)
                            continue
                        patch_id = f"{crop_reader.slide_id}_{x}_{y}.png"
                        self.result_queue.put((patch_id, patch_image))
                        num_patchs += 1

                    self.result_queue.put(WSIAsyncPredictor._EndToken(crop_reader.slide_id, crop_reader.width,
                        crop_reader.height, num_patchs))
                    crop_reader.close()

            print('end slide reader', os.getpid())

    class _PostprocessWorker(mp.Process):
        def __init__(self, output_path, result_queue, total):
            super().__init__()
            self.result_queue = result_queue
            self.output_path = output_path
            self.results = {}
            self.end_tokens = {}
            self.count = 0
            self.total = total

            self.label2name = { 
                0: "ASCUS_score",
                1: "LSIL_score",
                2: "HSIL_score",
                4: 'AGC_score'
            }

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
                        self.count += 1
                        if self.count == self.total:
                            break
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
                        keep = nms_boxes(bboxes, scores, nms_threshold=0.2)
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
                    slide_results = {
                        'ASCUS_score': 0,
                        'LSIL_score': 0,
                        'HSIL_score': 0,
                        'AGC_score': 0,
                        'max_score': 0,
                    }
                    for patch_result in self.results[slide_id]:
                        if patch_result is None:
                            continue

                        patch_id, classes, scores, bboxes = patch_result
                        for b, s, c in zip(bboxes, scores, classes):
                            label = int(c)
                            if label in self.label2name:
                                slide_results[self.label2name[label]] = max(slide_results[self.label2name[label]], float(s))
                                slide_results['max_score'] = max(slide_results['max_score'], float(s))
                            
                    with open(os.path.join(self.output_path, str(slide_id)+".json"), "w") as f:
                        f.write(json.dumps(slide_results))
                    self.count += 1
                    # delete slide results
                    with open("timming_old.log", "a") as f:
                        f.write(f"{time.time()},{slide_id}\n")
                    del self.results[slide_id]
                    if self.count == self.total:
                        break
                    print("slide %s is done" % (slide_id,))
            print('end post', os.getpid())


    def __init__(self, slide_files, json_files, num_gpus, num_slide_workers, params, output_path, det_cfg):
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
            "crop_patch_width": params["crop_patch_width"],
            "crop_patch_height": params["crop_patch_height"],
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
        for slide_file, json_file in zip(slide_files, json_files):
            self.slide_queue.put(json.dumps({'slide_path': slide_file, 'json_path': json_file}))
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
        "crop_patch_width": 1280,
        "crop_patch_height": 1280
    }
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args.inputs, args.output)

    for input_csv_file in args.inputs:
        dataset_id = Path(input_csv_file).stem
        output_dir = os.path.join(args.output, dataset_id)
        os.makedirs(output_dir, exist_ok=True)
        finished_slides = [Path(f).stem for f \
             in glob(os.path.join(output_dir, "*.json"))]

        # white csv
        df = pd.read_csv(input_csv_file)
        slide_files = []
        json_files = []
        for n in range(len(df)):
            row = df.iloc[n]
            slide_path = row["slide_path"]
            json_path = row["json_path"]
            if Path(slide_path).stem not in finished_slides:
                slide_files.append(slide_path)
                json_files.append(json_path)
        
        if len(slide_files) == 0:
            continue

        det_cfg = setup_cfg(args.config_file)
        det_cfg.train.init_checkpoint = args.weights
        
        num_gpus = torch.cuda.device_count()
        if args.num_gpus != -1:
            num_gpus = args.num_gpus
        WSIAsyncPredictor(slide_files, json_files, num_gpus, args.num_workers, params, output_dir, det_cfg)




