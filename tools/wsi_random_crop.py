import glob
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
sys.path.append("/nasdata/private/zwlu/trainer/miis-algorithm/")
from slide_detect_tools.slide_crop_patch import AutoReader



def random_crop_25(slide_path, save_path, crop_params):
    reader = AutoReader(slide_path)
    
    w, h = reader.width, reader.height
    
    boxes = []
    for _ in range(25):
        boxes.append([random.randint(0, w - 1), random.randint(0, h - 1)])
    
    for idx, box in enumerate(boxes):
        
        x1, y1 = [int(x) for x in box]
        x_center, y_center = x1, y1
        ratio = crop_params["crop_pixel_size"] / reader.slide_pixel_size
        x = x_center - int(ratio * crop_params["crop_patch_width"] / 2)
        y = y_center - int(ratio * crop_params["crop_patch_height"] / 2)
        if x < 0 or y < 0 :
            continue
        patch: np.ndarray = reader.crop_patch(x, y,crop_params["crop_patch_width"],crop_params["crop_patch_height"], crop_params["crop_pixel_size"], 0)
        if patch.std() < 5:
            continue
        slide_id = Path(slide_path).stem
        file_name = f"{slide_id}_{idx + 1:02}.jpg"
        cv2.imwrite(os.path.join(save_path, file_name), patch)
    reader.close()
    
    
if __name__ == "__main__":
    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description="Crop Program")
    parser.add_argument("--slide-path", default="/nasdata/dataset/moshi_data/tiff/dt1/", type=str, help="image root path")
    parser.add_argument("--output-path", default="/nasdata/private/zwlu/trainer/miis-algorithm/output/test", type=str, help="trainset label path")

    args = parser.parse_args()
    params = {
        "crop_pixel_size": 0.1,
        "crop_patch_width": 3840,
        "crop_patch_height": 2160
    }
    
    for i in tqdm.tqdm(glob.glob(os.path.join(args.slide_path, "*"))):
        random_crop_25(i, args.output_path, params)