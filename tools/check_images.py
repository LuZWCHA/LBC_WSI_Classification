import glob
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
sys.path.append("/nasdata/private/zwlu/trainer/miis-algorithm/")
from slide_detect_tools.slide_crop_patch import AutoReader

import hashlib
def get_md5(file_path):
    with open(file_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(1024*16)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(1024*16)

    return file_hash.hexdigest()

def check_image(slide):
    try:
        reader = AutoReader(slide_path)
        reader.close()
    except:
        return False
    return True
    
    
if __name__ == "__main__":
    import argparse
    import tqdm
    import pandas as pd

    parser = argparse.ArgumentParser(description="Crop Program")
    parser.add_argument("--slide-path", default="/media/now/st200010/Downloads/原始数据/ahslcro_1208", type=str, help="image root path")

    args = parser.parse_args()
    
    all_md5 = set()
    res = {
        "file_name": [],
        "is_ok": [],
        "is_repeat": [],
    }
    for i in tqdm.tqdm(glob.glob(os.path.join(args.slide_path, "*"))[:10]):
        is_ok = check_image(i)
        is_repeat = False
        md5 = get_md5(i)
        if md5 in all_md5:
            is_repeat = True
        res["file_name"].append(Path(i).stem)
        res["is_ok"].append(is_ok)
        res["is_repeat"].append(is_repeat)
    pd.DataFrame(res).to_csv("output/check.csv", index=False)
    
        