import os
import argparse
import json
import multiprocessing as mp
import logging
from glob import glob
from pathlib import Path
import time
import cv2

import torch
import pandas as pd
import numpy as np
from datasets.our_datasets import WSI_RAW

from predictor_mil import Predictor, setup_cfg
from slide_detect_tools.slide_crop_patch import AutoReader, SdpcReader, OpenSlideReader
from slide_detect_tools.help import remove_overlap, nms_boxes
from slide_detect_tools.model_view_sampler import TopKSampler

# class WSIScore(object):
        
#     def __init__(self):
#         self.idx2name = {0: 'ASCUS', 1: 'LSIL', 2: 'HSIL', 4: 'AGC'}
#         self.idxs = [0, 1, 2, 4]

#     def get_topk(self, json_file, k=32):
#         json_data = json.load(open(json_file))
#         data = np.array(json_data["bboxes"], dtype=np.float32)
#         mask = np.zeros((len(data),), dtype=np.bool_)
#         for idx in self.idxs:
#             mask = mask | (data[:, 0] == idx)
#         data = data[mask]
#         ind = np.argsort(-data[:, 1])
#         outputs = data[ind[:k]]                                                                
#         return outputs


class StopToken:
    pass


class EndToken:
    pass


class PredictWorker(mp.Process):
    """predict worker"""

    def __init__(self, cfg, input_queue, output_queue, params):
        super().__init__()
        self.cfg = cfg
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.params = params

    def run(self):
        predictor = Predictor(self.cfg)
        print('start predictor', os.getpid())
        from convert_util.label_convert import deserialize_output_json
        while True:
            task = self.input_queue.get()
            if isinstance(task, StopToken):
                break

            task = json.loads(task)
            print("Slide: %s" % (task['slide_path'], ))
            try:
                json_data = deserialize_output_json(task['json_path'])
                # fix the path
                json_data[0] = task['slide_path']
                # 4 classes and take top 6 for every one
                # totally 24 views
                topk_sampler = TopKSampler(json_data, k=6)
                xy_dict, _ = topk_sampler.sample()
                crop_reader = AutoReader(task['slide_path'])
            except Exception as e:
                print("Slide %s read error: %s" % (task['slide_path'], str(e)))
                self.output_queue.put(EndToken())
            else:
                images = []
                for v in xy_dict.values():
                    for x, y in v:
                        try:
                            patch_image = crop_reader.crop_patch(x, y, self.params["crop_patch_width"],
                                    self.params["crop_patch_height"], self.params["crop_pixel_size"], crop_level=0)
                                    
                            # np.save(f"{slide}")
                            # slide_id = task['json_path']
                            # slide_id = f"{Path(task['json_path']).stem}_{x}_{y}.jpg"
                            # save_path = os.path.join("/nasdata/private/zwlu/Now/ai_trainer/outputs/test_inference_debug_svs", "topk_images")
                            # if not os.path.exists(save_path):
                            #     os.makedirs(save_path)
                            # cv2.imwrite(os.path.join(save_path, slide_id), patch_image)
                            
                        except Exception as e:
                            continue
                        images.append(torch.as_tensor(np.ascontiguousarray(patch_image.transpose(2, 0, 1)))) 
                
                if len(images) == 0:
                    self.output_queue.put(EndToken())
                    continue

                output = predictor({"images": images, "label": 0})
                output_data = json.dumps({
                    "slide_name": crop_reader.slide_id,
                    "max_score": output[0],
                    #"L_score": output[1],
                    #"H_score": output[2],
                    #"AGC_score": output[3]
                })
                # print(output_data)
                self.output_queue.put(output_data)
            finally:
                crop_reader.close()

class PostprocessWorker(mp.Process):
    def __init__(self, output_path, result_queue, total):
        super().__init__()
        self.result_queue = result_queue
        self.output_path = output_path
        self.results = {}
        self.count = 0
        self.total = total

    def run(self):
        print('start post', os.getpid())
        while True:
            task = self.result_queue.get()
            self.count += 1
            if isinstance(task, EndToken):
                continue

            output = json.loads(task)
            self.results[output['slide_name']] = output
            with open("timming_old.log", "a") as f:
                f.write(f"{time.time()},{output['slide_name']}\n")
            print(self.count, self.total)
            if self.count == self.total:
                out_df = pd.DataFrame(self.results).T
                out_df.to_csv(self.output_path, index=False)
                break

class WSIAsyncPredictor:
    def __init__(self, slide_files, json_files, num_gpus, num_workers, params, output_path, det_cfg):
        """
        Args:
            wsi_files (list): wsi path list
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
            num_wsi_workers (int): num workers to read WSI file
        """
        
        # init queues
        num_gpus = max(num_gpus, 1)
        self.input_queue = mp.Queue(maxsize=len(slide_files) + num_gpus + 1)
        self.result_queue = mp.Queue(maxsize=num_gpus * 3)
        
        self.num_predictor_workers = num_gpus
        
        self.model_procs = []
        # model predictor workers
        slide_reader_params = {
            "crop_pixel_size": params["crop_pixel_size"] ,
            "crop_patch_width": params["crop_patch_width"],
            "crop_patch_height": params["crop_patch_height"],
        }
        for gpuid in range(num_gpus):
            cfg = det_cfg.copy()
            cfg.train.device = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.model_procs.append(
                PredictWorker(cfg, self.input_queue, self.result_queue, slide_reader_params)
            )
       
        for p in self.model_procs:
            p.start()
            
        # add slide
        for slide_file, json_file in zip(slide_files, json_files):
            self.input_queue.put(json.dumps({'slide_path': slide_file, 'json_path': json_file}))
        # add end token
        for _ in range(self.num_predictor_workers):
            self.input_queue.put(StopToken())
        
        # postprocess worker
        self.post_proc = PostprocessWorker(output_path, self.result_queue, len(slide_files))
        self.post_proc.start()
        self.post_proc.join()
        
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
        "crop_pixel_size": 0.31 * 2,
        "crop_patch_width": 640,
        "crop_patch_height": 384
    }
    
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    print(args.inputs, args.output)
    
    for input_csv_file in args.inputs:
        dataset_id = Path(input_csv_file).stem
        output_dir = os.path.join(args.output, dataset_id+".csv")
        
        finished_slides = []
        if os.path.exists(output_dir):
            df = pd.read_csv(output_dir)
            finished_slides = [df.loc[row]['slide_name'] for row in df.index]

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
            print("-" * 120)
            print("Nothing to do... skip all slides.")
            continue

        det_cfg = setup_cfg(args.config_file)
        det_cfg.train.init_checkpoint = args.weights
        
        num_gpus = torch.cuda.device_count()
        if args.num_gpus != -1:
            num_gpus = args.num_gpus    
        

        WSIAsyncPredictor(slide_files, json_files, num_gpus, args.num_workers, params, output_dir, det_cfg)    

            

