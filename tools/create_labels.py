import glob
import json
from class_map import *
from datasets.our_datasets import WSI_RAW, WSI_RAW_LABEL, SHISHI_RAW
from write_readme import create_dt_readme
import os.path as osp
import numpy as np

wsi_raw_c3_desc = f"""
This is WSI RAW Dataset's 3 classes label dictory: {CLASS_THREE_NAMES} are provided.
The labels' generation script is following:\n
"""

@create_dt_readme(WSI_RAW.root_path+"./label_c3", dataset_title="WSI RAW Dataset labels", dataset_desc=wsi_raw_c3_desc, verbose=True)
def create_wsi_raw_label_c3():
    csv_list = glob.glob(osp.join(WSI_RAW_LABEL.root_path, "*.csv"))
    gen_valid_labels_csv(csv_list, "label_c3", root_dir= WSI_RAW.root_path, label_map=name_to_label_3class)


wsi_raw_c6_desc = f"""
This is WSI RAW Dataset's 6 classes label dictory: {CLASS_SIX_NAMES} are provided.
The labels' generation script is following:\n
"""

@create_dt_readme(WSI_RAW.root_path+"./label_c6", dataset_title="WSI RAW Dataset labels", dataset_desc=wsi_raw_c6_desc, verbose=True)
def create_wsi_raw_label_c6():
    csv_list = glob.glob(osp.join(WSI_RAW_LABEL.root_path, "*.csv"))
    gen_valid_labels_csv(csv_list, "label_c6", root_dir= WSI_RAW.root_path, label_map=name_to_label_6class)



def create_mask_m(path):
    
    labels=list(range(10))
    label_num = len(labels)
    image_size=(3840, 2160)
    with open(path, "r") as f:
        lines = f.readlines()
    W, H = image_size
    parent_dir = os.path.dirname(path)
    
    name = Path(path).stem
    
    if os.path.exists(os.path.join(parent_dir, name+".npy")):
        return path

    parent_dir = parent_dir.replace("labels", "masks")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    mask = np.zeros((label_num, H, W), dtype=np.uint8)
    for line in lines:
        label_id, x, y, w, h = [float(i) for i in line.split()]
        x_a = int(x * W)
        y_a = int(y * H)
        w_a = int(w * W)
        h_a = int(h * H)
        label_id = int(label_id)
        mask[label_id, y_a - h_a // 2: y_a + h_a // 2, x_a - w_a // 2: x_a + w_a // 2] = label_id + 1
    
    # plot_many([mask[label_id] for label_id in labels], cols=1)
        
    np.save(os.path.join(parent_dir, name), mask)
    del mask
    return path

def create_shishi_segment_label_c10(image_size=(3840, 2160), labels=list(range(10))):
    from datasets.our_datasets import SHISHI_RAW
    from tools.visual.plot_utils import plot_many
    # SHISHI_RAW.at("dt1")
    label_paths = SHISHI_RAW.get_target_files("**/labels/*.txt")
    
    # label_paths = glob.iglob(SHISHI_RAW.at("**/labels/*.txt"), recursive=True)
    
    from multiprocessing import Pool
    
    pool = Pool(16)
    iter_ = pool.imap_unordered(create_mask_m, label_paths, chunksize=4)
    import tqdm
    for i in tqdm.tqdm(iter_):
        pass


if __name__ == "__main__":
    # create_wsi_raw_label_c3()
    
    create_shishi_segment_label_c10()
