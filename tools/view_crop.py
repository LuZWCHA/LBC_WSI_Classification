
import collections
import os, json, numpy as np
import cv2, random
from pathlib import Path

import tqdm
from datasets.our_datasets import TCT_WSI_RAW_LABEL
# from libiblsdk.ibl_py_sdk import IblWsi
from pipeline import create, PSegment, IPSWorkerGroup, IdTransfrom, DataPacket, DATA_PACKET, Worker, ProgressObserver
from slide_detect_tools.slide_crop_patch import AutoReader
import pandas as pd
import multiprocessing as mp
from tools.logging_helper import logger_init_by_name, default_logger_init, get_default_logger

import logging
logger = get_default_logger(__file__, output="logs")

class CropWorker(Worker):

    def __init__(self, ibl_open=None) -> None:
        super().__init__()
        self.ibl_open = ibl_open

    def process(self, p: DataPacket) -> DATA_PACKET:
        # DataPacket({"view":view, "slide_path": slide_path, "max_scores": p["max_scores"]})
        pre_ret = p.obj
        slide_path = pre_ret['slide_path']
        # with suppress_stdout_stderr(lock=Lock()):
        
        reader = AutoReader(slide_path)
        patch = reader.crop_patch(*pre_ret["crop_params"])
        reader.close()

            
        return DataPacket({"patch": patch, "slide_path": slide_path, "class_id": pre_ret["class_id"], "score": pre_ret["score"], "position":pre_ret["position"] })

class RegionConvert(IdTransfrom):
    
    def __init__(self, params, path) -> None:
        super().__init__()
        self.path = path
        self.params = params

    def transform(self, p: DataPacket) -> DATA_PACKET:
        boxes = p.obj
        reader = AutoReader(self.path)
        res = []
        for box in boxes:
            label_id = int(box[0])
            score = float(box[1])
            x1, y1, x2, y2 = [int(x) for x in box[2:]]
            x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
            ratio = self.params["crop_pixel_size"] / reader.slide_pixel_size
            x = x_center - int(ratio * self.params["crop_patch_width"] / 2)
            y = y_center - int(ratio * self.params["crop_patch_height"] / 2)
            if x < 0 or y < 0 :
                continue
            
            crop_params = (x, y, self.params["crop_patch_width"],self.params["crop_patch_height"], self.params["crop_pixel_size"], 0 )
    
            dp = DataPacket({"slide_path": self.path, "slide_id": Path(self.path).stem, "crop_params": crop_params, "class_id": label_id, "score": score, "position": (x, y)})
            res.append(dp)
        reader.close()
        return res
        

from convert_util.class_map import CLASS_TEN_NAMES
class Save(IdTransfrom):
    
    def __init__(self, save_path) -> None:
        super().__init__()
        self.save_path = save_path

    def transform(self, p: DataPacket) -> DATA_PACKET:
        res = p.obj
        # {"patch": patch, "slide_path": slide_path, "class_id": pre_ret["class_id"], "score": pre_ret["score"], "position": (x, y)}
        patch = res["patch"]
        slide_path = res["slide_path"]
        class_id = res["class_id"]
        score = res["score"]
        x, y = res["position"]
        
        slide_id = Path(slide_path).stem
        file_name = f"{slide_id}_{CLASS_TEN_NAMES[class_id]}_{score}_{x}_{y}.jpg"
        if score > 0.4:
            cv2.imwrite(os.path.join(self.save_path, file_name), patch)
        
        return DataPacket(slide_path)
        

def check_path(path):
    return os.path.exists(path)


def crop_views(path, predict_view_json, crop_params, save_path, idxs=[0, 1, 2, 4], topk=32, threshold=0.4, label=None, mode="gt"):

    assert check_path(path) and check_path(predict_view_json)

    with open(predict_view_json, "rb") as f:
        res   = json.load(f)
        
    res["bboxes"]  = [[cl, float(s), a, b ,c ,d] for cl, s, a, b , c, d in res["bboxes"] ]
        
    data = np.array(res["bboxes"], dtype=np.float32)
    
    mask = np.zeros((len(data),), dtype=np.bool_)
    for idx in idxs:
        mask = mask | (data[:, 0] == idx)
    data = data[mask]
    ind = np.argsort(-data[:, 1])
    topk_boxes = data[ind[:topk]]
    
    boxes = topk_boxes
    reader = AutoReader(path)
    res = []
    for box in boxes:
        label_id = int(box[0])
        score = float(box[1])
        x1, y1, x2, y2 = [int(x) for x in box[2:]]
        x_center, y_center = (x1 + x2) // 2, (y1 + y2) // 2
        ratio = crop_params["crop_pixel_size"] / reader.slide_pixel_size
        x = x_center - int(ratio * crop_params["crop_patch_width"] / 2)
        y = y_center - int(ratio * crop_params["crop_patch_height"] / 2)
        if x < 0 or y < 0 :
            continue
        patch = reader.crop_patch(x, y,crop_params["crop_patch_width"],crop_params["crop_patch_height"], crop_params["crop_pixel_size"], 0)
        
        slide_id = Path(path).stem
        
        label_name = CLASS_TEN_NAMES[label_id]
        
        if (label is not None  and label_name == label) or label is None:
            if (mode=="gt" and score > threshold) or (mode=="lt" and score < threshold):
                file_name = f"{slide_id}_{label_name}_{score}_{x}_{y}.jpg"
                cv2.imwrite(os.path.join(save_path, file_name), patch)
    reader.close()

    return path

def crop_views_m(p):
    return crop_views(*p)


def crop_views_for_neg(predict_res_dir= "/nasdata/private/gfhuang/00_data/cell_AI/02_output/2023-01-28_test_wsi_mil/model1/m2_ibl_356_IblReader_5not_seg_recall/dt2_new", output_dir="outputs/topk_views/neg", process_num=16):
    import pandas as pd
    from datasets.our_datasets import TCT_WSI_RAW_LABEL
    from multiprocessing import Pool
    # dt1_new.csv  dt2_new.csv
    
    # label_csv1 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt1_new.csv"))
    label_csv = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt2_new.csv"))
    
    # label_csv = pd.concat((label_csv1, label_csv2), axis=0)
    
    neg_sample = label_csv[label_csv["grade_6"] == "NILM"]

    slide_paths = neg_sample["slide_path"].to_list()
    
    params = {
    "crop_pixel_size": 0.1,
    "crop_patch_width": 3840,
    "crop_patch_height": 2160
    }
    params_ = ((p, os.path.join(predict_res_dir, Path(p).stem + ".json"), params, output_dir) for p in slide_paths)
    
    pool = Pool(process_num)
    
    iter_ = pool.imap_unordered(crop_views_m, params_)
    bar = tqdm.tqdm(iter_, total=len(slide_paths))
    for p in   bar:
        bar.set_postfix_str(Path(p).stem)
    
    pool.close()
    pool.join()
    
def crop_views_for_neg_low_score(predict_res_dir= "/nasdata/private/gfhuang/00_data/cell_AI/02_output/2023-01-28_test_wsi_mil/model1/m2_ibl_356_IblReader_5not_seg_recall/dt2_new", output_dir="outputs/topk_views/neg", process_num=16):
    import pandas as pd
    from datasets.our_datasets import TCT_WSI_RAW_LABEL
    from multiprocessing import Pool
    # dt1_new.csv  dt2_new.csv
    
    # label_csv1 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt1_new.csv"))
    label_csv = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt2_new.csv"))
    
    # label_csv = pd.concat((label_csv1, label_csv2), axis=0)
    
    neg_sample = label_csv[label_csv["grade_6"] == "NILM"]

    slide_paths = neg_sample["slide_path"].to_list()
    
    params = {
    "crop_pixel_size": 0.1,
    "crop_patch_width": 3840,
    "crop_patch_height": 2160
    }
    params_ = ((p, os.path.join(predict_res_dir, Path(p).stem + ".json"), params, output_dir, list(range(10)), 32, 0.4, None, "lt") for p in slide_paths)
    
    pool = Pool(process_num)
    
    iter_ = pool.imap_unordered(crop_views_m, params_)
    bar = tqdm.tqdm(iter_, total=len(slide_paths))
    for p in   bar:
        bar.set_postfix_str(Path(p).stem)
    
    pool.close()
    pool.join()
    
def crop_views_for_pos(predict_res_dir="/nasdata/private/gfhuang/00_data/cell_AI/02_output/2023-01-28_test_wsi_mil/model1/m2_ibl_356_IblReader_5not_seg_recall/dt2_new", output_dir="outputs/topk_views/pos", process_num=16):
    import pandas as pd
    from datasets.our_datasets import TCT_WSI_RAW_LABEL
    from multiprocessing import Pool
    # dt1_new.csv  dt2_new.csv
    
    # label_csv1 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt1_new.csv"))
    label_csv2 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt2_new.csv"))
    
    # label_csv = pd.concat((label_csv1, label_csv2), axis=0)
    
    pos_sample = label_csv2[label_csv2["grade_6"] != "NILM"]
    
    slide_paths = pos_sample["slide_path"].to_list()
    # predict_res_dir = "/nasdata/private/gfhuang/00_data/cell_AI/02_output/2023-01-28_test_wsi_mil/model1/m2_ibl_356_IblReader_5not_seg_recall/dt2_new"
    params = {
    "crop_pixel_size": 0.1,
    "crop_patch_width": 3840,
    "crop_patch_height": 2160
    }
    
    params_ = ((p, os.path.join(predict_res_dir, Path(p).stem + ".json"), params, output_dir, [0, 1, 2, 4], 32, 0.5, pos_sample.loc[pos_sample["slide_path"] == p, "grade_6"].item()) for p in slide_paths)
    
    pool = Pool(process_num)
    
    iter_ = pool.imap_unordered(crop_views_m, params_)
    bar = tqdm.tqdm(iter_, total=len(slide_paths))
    for p in   bar:
        bar.set_postfix_str(Path(p).stem)
    
    pool.close()
    pool.join()


def get_view(slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), max_score_thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4], sort_view=False):
    with open(json_path, "rb") as f:
        json_data = json.load(f)

    bboxes = json_data["bboxes"]
    
    if "max_score" in json_data:
        max_score = json_data["max_score"]
    else:
        max_score = 0

    if "max_score" in json_data and not (max_score_thredshold_range[0] <= max_score <= max_score_thredshold_range[1]):
        return slide_path
    
    try:
        reader = AutoReader(slide_path)
    except Exception as e:
        logger.error("skip", json_path)
        return slide_path
    ps = reader.slide_pixel_size
    crop_w, crop_h, crop_ps, crop_overlap = grid_params
    name = Path(slide_path).stem
    
    patch2boxes = dict()
    patch2strs = dict()
    name = Path(slide_path).stem
    for box in bboxes:
        c, s, x1, y1, x2, y2 = box
        s = float(s)
        
        offset_x, offset_y, local_x1, local_y1, local_x2, local_y2, ratio = cal_grid_x_y((x1, y1, x2, y2), crop_w, crop_h, crop_ps, crop_overlap, ps)
        anno_box = (int(local_x1 / ratio), int(local_y1 / ratio), int(local_x2 / ratio), int(local_y2 / ratio), int(c), s)
        if anno_box[2] >= crop_w or anno_box[3] >= crop_h:
            continue
            
        if (offset_x, offset_y) not in patch2boxes:
            patch2boxes[(offset_x, offset_y)] = []
            patch2strs[(offset_x, offset_y)] = []
        patch2boxes[(offset_x, offset_y)].append(np.array(anno_box))
    
    logger.debug(f"{slide_path}: {len(patch2boxes)}")
    patch_list = []
    for (offset_x, offset_y), anno_boxes in patch2boxes.items():
        annos = np.stack(anno_boxes, 0)
        mask = np.zeros((len(anno_boxes),), dtype=np.bool_)
        for i in class_ids:
            mask |= (annos[:, -2] == i) & (annos[:, -1] > thredshold_range[0]) & (annos[:, -1] < thredshold_range[1])
        annos = annos[mask]
        if len(annos) > 0:
            patch_list.append( ((offset_x, offset_y), annos))
    
    
    if not sort_view:
        random.shuffle(patch_list)
    else:
        patch_list = sorted(patch_list, key=lambda x: np.max(x[1][:, 5]), reverse=True)
        ml = []
        for patch in patch_list:
            ml.append(np.max(patch[1][:, -1]))
    
    label_str_list = []
    score_list = []
    for (offset_x, offset_y), anno_boxes in patch_list:
        outputs = anno_boxes
        offset_x, offset_y = int(offset_x), int(offset_y)
        ll = len(outputs)
        boxes_label_str = []
        if ll > 0:
            max_score = 0
            for i in range(ll):
                x1, y1, x2, y2, c, s = outputs[i].tolist()
                str_list = [str(i) for i in [int(y1), int(x1), int(y2), int(x2), int(c), float(s)]]
                label = ",".join(str_list)
                boxes_label_str.append(label)
                max_score = max(s, max_score)
                
            file_name = f"{name}_{offset_x}_{offset_y}.jpg"
            file_name2 = f"{name}_{offset_x}_{offset_y}_label.jpg"
            path = os.path.join(output_path, "images",  file_name)
            path2 = os.path.join(output_path, "image_with_annos", file_name2)
            label_path = os.path.join(output_path, "pesudo_labels", "labels.txt")
            
            for p in [path, path2, label_path]:
                par_dir = Path(p).parent
                if not par_dir.exists():
                    par_dir.mkdir(exist_ok=True)
            
            label_str_list_list = [x.split(",")[-2:] for x in boxes_label_str]
            label_str_list.append(boxes_label_str)
            try:
                patch = reader.crop_patch(offset_x, offset_y, crop_w, crop_h, crop_ps)
            except Exception as e:
                logger.error(e)
                continue
            cv2.imwrite(path, patch)
            draw_bounding_boxes_on_image_array(patch, outputs[:, :4], display_str_list_list=label_str_list_list, box_mode="xyxy")
            cv2.imwrite(path2, patch)
            
            with open(label_path, "a") as lo:
                contents = [file_name] + boxes_label_str
                lo.write(" ".join(contents))
                lo.write("\n")
            
        score_list.append(max_score)
        if len(label_str_list) >= k:
            break
    # print(score_list)
    reader.close()
    return slide_path
    
    
def get_view_only_pos_high_score(slide_path, json_path, grid_params, output_path="", k=8, low_conf_range=(0.3, 0.5), max_score_thredshold_range=(0.62, 1), class_ids=[0, 1, 2, 4], mask=True):
    with open(json_path, "rb") as f:
        json_data = json.load(f)

    bboxes = json_data["bboxes"]
    if "max_score" in json_data:
        max_score = json_data["max_score"]
    else:
        max_score = 0
    if "max_score" in json_data and not (max_score_thredshold_range[0] < max_score < max_score_thredshold_range[1]):
        return
    try:
        reader = AutoReader(slide_path)
    except Exception as e:
        logger.error("skip", json_path)
        return
    ps = reader.slide_pixel_size
    crop_w, crop_h, crop_ps, crop_overlap = grid_params
    name = Path(slide_path).stem
    
    patch2boxes = dict()
    patch2strs = dict()
    name = Path(slide_path).stem
    for box in bboxes:
        c, s, x1, y1, x2, y2 = box
        s = float(s)
            
        offset_x, offset_y, local_x1, local_y1, local_x2, local_y2, ratio = cal_grid_x_y((x1, y1, x2, y2), crop_w, crop_h, crop_ps, crop_overlap, ps)
        anno_box = (int(local_x1 / ratio), int(local_y1 / ratio), int(local_x2 / ratio), int(local_y2 / ratio), c, s)
        if anno_box[2] >= crop_w or anno_box[3] >= crop_h:
            continue
            
        if (offset_x, offset_y) not in patch2boxes:
            patch2boxes[(offset_x, offset_y)] = []
            patch2strs[(offset_x, offset_y)] = []
        patch2boxes[(offset_x, offset_y)].append(np.array(anno_box))
        # patch2strs[(offset_x, offset_y)].append("score:")

    patch_list = []
    for (offset_x, offset_y), anno_boxes in patch2boxes.items():
        score_list = []
        
        for box in anno_boxes:
            _, _, _, _, c, s = box
            
            min_socre = 1
            if c in class_ids:
                if s > low_conf_range[1]:
                    score_list.append(s)
                    patch_list.append( ((offset_x, offset_y), anno_boxes))
                    break
                min_socre = min(min_socre, s)
 
    random.shuffle(patch_list)
    
    print(len(patch_list))
    label_str_list = []
    for (offset_x, offset_y), anno_boxes in patch_list:
            # print(offset_x, offset_y, crop_w, crop_h, crop_ps, reader.width, reader.height)
            # strs = patch2strs[(offset_x, offset_y)]

            # draw_bounding_boxes_on_image_array(patch, np.stack(anno_boxes, 0)[:, :4], display_str_list_list=strs)
            
            # anno_boxes.sort(key=lambda x: -x[-1])
            anno_boxes = np.stack(anno_boxes, 0)

            # index = list(range(min(k, len(anno_boxes))))
            # np.random.shuffle(anno_boxes)
            outputs = anno_boxes
            
            ll = len(outputs)

            if ll > 0:
                boxes_label_str = []
                mask_boxes = []
                for i in range(ll):
                    x1, y1, x2, y2, c, s = outputs[i].tolist()
                    if s > low_conf_range[1]:
                        str_list = [str(i) for i in [int(y1), int(x1), int(y2), int(x2), int(c)]]
                        label = ",".join(str_list)
                        boxes_label_str.append(label)
                    elif low_conf_range[0] < s <= low_conf_range[1]:
                        print("low conf find")
                        mask_boxes.append((x1, y1, x2, y2))
                file_name = f"{name}_{offset_x}_{offset_y}.jpg"
                path = os.path.join(output_path, file_name)
                label_str = " ".join([path] + boxes_label_str)
                label_str_list.append(label_str)
                try:
                    patch = reader.crop_patch(offset_x, offset_y, crop_w, crop_h, crop_ps)
                    # try to mask the low confidence areas
                    if mask:
                        for mask_box in mask_boxes:
                            x1, y1, x2, y2 = [int(i) for i in mask_box]
                            try:
                                patch[x1:x2, y1:y2, 0] = 255
                                patch[x1:x2, y1:y2, 1] = 255
                                patch[x1:x2, y1:y2, 2] = 255

                            except Exception as e:
                                print(e)
                    
                except Exception as e:
                    # reader.close()
                    continue
                cv2.imwrite(path, patch)
                with open(os.path.join(output_path, "pos_annos.txt"), "a") as f:
                    f.write(label_str)
                    f.write("\n")

                
            if len(label_str_list) > k:
                break
    reader.close()


def get_veiw_m(p):
    return get_view(*p)
    
def get_high_conf_pos_view_m(p):
    return get_view_only_pos_high_score(*p)

""" reverse to calculate the grid location of the box by the crop parameters and the box's glob location (at level 0) """
def cal_grid_x_y(bbox, crop_w, crop_h, crop_ps, crop_overlap, ps):
    ratio =  crop_ps / ps
    w_in_ori = int((crop_w - crop_overlap) * ratio)
    h_in_ori = int((crop_h - crop_overlap) * ratio)
    
    x_min, y_min, x_max, y_max = bbox
    
    offset_x = x_min // w_in_ori
    offset_y = y_min // h_in_ori
    
    offset_x *= w_in_ori
    offset_y *= h_in_ori
    
    local_x = x_min % w_in_ori
    local_y = y_min % h_in_ori
    
    return offset_x, offset_y, local_x, local_y, local_x + x_max - x_min, local_y + y_max - y_min, ratio


def box_local_xy2slide_xy(box, crop_w, crop_h, crop_ps, crop_overlap, ps):
    c, s, x1, y1, x2, y2 = box
        
    offset_x, offset_y, local_x1, local_y1, local_x2, local_y2, ratio = cal_grid_x_y((x1, y1, x2, y2), crop_w, crop_h, crop_ps, crop_overlap, ps)
    anno_box = (int(local_x1 / ratio), int(local_y1 / ratio), int(local_x2 / ratio), int(local_y2 / ratio), c, s)
    return anno_box, offset_x, offset_y


from tools.visual.box_visualizer import draw_bounding_boxes_on_image_array
""" generate the boxes annotations and silde views from boxes  prediction results, only take high score positive samples"""
def gen_bboxes_labels_from_predict(slide_path, json_path, grid_params, output_path="", k=16, score_thredshold=0.6, topk_classes=[0, 1, 2, 4]):
    
    with open(json_path, "rb") as f:
        json_data = json.load(f)
    reader = AutoReader(slide_path)
    ps = reader.slide_pixel_size
    bboxes = json_data["bboxes"]
    crop_w, crop_h, crop_ps, crop_overlap = grid_params
    
    patch2boxes = dict()
    patch2strs = dict()
    patch2scores = dict()
    name = Path(slide_path).stem
    for box in bboxes:
        c, s, x1, y1, x2, y2 = box
        
        anno_box, offset_x, offset_y= box_local_xy2slide_xy(box, crop_w, crop_h, crop_ps, crop_overlap, ps)
        if anno_box[2] >= crop_w or anno_box[3] >= crop_h:
            continue
            
        if (offset_x, offset_y) not in patch2boxes:
            patch2boxes[(offset_x, offset_y)] = []
            patch2strs[(offset_x, offset_y)] = []
            patch2scores[(offset_x, offset_y)] = []
            
        patch2boxes[(offset_x, offset_y)].append(np.array(anno_box))
        patch2strs[(offset_x, offset_y)].append("score:")
        patch2scores[(offset_x, offset_y)].append((c, s))
        
        
    label_str_list = []
    for (offset_x, offset_y), anno_boxes in patch2boxes.items():
            # print(offset_x, offset_y, crop_w, crop_h, crop_ps, reader.width, reader.height)
            strs = patch2strs[(offset_x, offset_y)]
            try:
                patch = reader.crop_patch(offset_x, offset_y, crop_w, crop_h, crop_ps)
            except Exception as e:
                continue
            # draw_bounding_boxes_on_image_array(patch, np.stack(anno_boxes, 0)[:, :4], display_str_list_list=strs)
            
            # anno_boxes.sort(key=lambda x: -x[-1])
            anno_boxes = np.stack(anno_boxes, 0)
            mask = np.zeros((len(anno_boxes),), dtype=np.bool_)
            for i in topk_classes:
                mask |= (anno_boxes[:, -2] == i) & (anno_boxes[:, -1] > score_thredshold)
            
            idx = np.argsort(-anno_boxes[mask][:, -1])
            outputs = anno_boxes[idx[:k]]
            ll = len(outputs)
            
            if ll > 0:
                boxes_label_str = []
                for i in range(ll):
                    x1, y1, x2, y2, c, s = outputs[i].tolist()
                    str_list = [str(i) for i in [int(x1), int(y1), int(x2), int(y2), int(c)]]
                    label = ",".join(str_list)
                    boxes_label_str.append(label)
                file_name = f"{name}_{offset_x}_{offset_y}.jpg"
                path = os.path.join(output_path, file_name)
                label_str = " ".join([path] + boxes_label_str)
                label_str_list.append(label_str)
                cv2.imwrite(path, patch)

    reader.close()
    return label_str_list

""" This function is only for tct dt1/dt2 """
def get_pos_slides():
    label_csv1 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt1_new.csv"))
    label_csv2 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt2_new.csv"))
    label_csv = pd.concat([label_csv1, label_csv2], axis=0)
        
    pos_sample = label_csv[label_csv["grade_6"] != "NILM"]
    
    slide_paths = pos_sample["slide_path"].to_list()
    
    return set(slide_paths)
    
""" This function is only for tct dt1/dt2 """
def get_neg_slides():
    label_csv1 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt1_new.csv"))
    label_csv2 = pd.read_csv(TCT_WSI_RAW_LABEL.at("dt2_new.csv"))
    label_csv = pd.concat([label_csv1, label_csv2], axis=0)
        
    pos_sample = label_csv[label_csv["grade_6"] == "NILM"]
    
    slide_paths = pos_sample["slide_path"].to_list()
    
    return set(slide_paths)
    
def gen_bboxes_labels_m(p):
    return gen_bboxes_labels_from_predict(*p)

""" This function is only for tct dt1/dt2 """
def get_pos_low_socre_views(
    output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/views/pos_low_score",   
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat",
    sample_rate=0.3,
    sample_limit=100,
    random_state = 0,
):
    if not check_path(output_path):
        os.makedirs(output_path)
                
    tct_all_files = get_pos_slides()
    print(len(tct_all_files))
    tct_all_files_sampled = random.Random(random_state).sample(tct_all_files, int(min(len(tct_all_files) * sample_rate, sample_limit)))
    # tct_all_files_sampled =  tct_all_files.sample()
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    grid_params = [3840, 2160, 0.1, 0]
    topk_limit = 24
    
    pool = mp.Pool(16)
    params_list = []
    for file in tqdm.tqdm(tct_all_files_sampled):
        file_name = Path(file).with_suffix(".json").name
        if "dt1" in file:
            threshold = (0.1, 0.50)
            max_sc_range = (0, 0.60)
        else:
             threshold = (0.1, 0.55)
             max_sc_range = (0, 0.65)
             
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4]
        p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold, max_sc_range)
        params_list.append(p)
        
    iter_ = pool.imap_unordered(get_veiw_m, params_list)
    for _ in tqdm.tqdm(iter_, total=len(params_list)):
        pass
        # with open(os.path.join(output_path, "anno.txt"), "a") as f:
        #     for label_str in item:
        #         f.write(label_str)
        #         f.write("\n")
        
""" This function is only for tct dt1/dt2 """
def get_pos_high_socre_views(
    output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/views/pos_high_score",   
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat",
    sample_rate=0.2,
    sample_limit=100,
    random_state = 0,
):
    if not check_path(output_path):
        os.makedirs(output_path)
            
    tct_all_files = get_pos_slides()
    print(len(tct_all_files))
    tct_all_files_sampled = random.Random(random_state).sample(tct_all_files, int(min(len(tct_all_files) * sample_rate, sample_limit)))
    # tct_all_files_sampled =  tct_all_files.sample()
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    grid_params = [3840, 2160, 0.1, 0]
    topk_limit = 12
    
    pool = mp.Pool(16)
    params_list = []
    for file in tqdm.tqdm(tct_all_files_sampled):
        file_name = Path(file).with_suffix(".json").name
        if "dt1" in file:
            threshold = (0.65, 1)
            max_sc_range = (0.55, 1)
        else:
             threshold = (0.7, 1)
             max_sc_range = (0.60, 1)
             
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4]
        p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold, max_sc_range)
        params_list.append(p)
        
    iter_ = pool.imap_unordered(get_veiw_m, params_list)
    for _ in tqdm.tqdm(iter_, total=len(params_list)):
        pass        

""" This function is only for tct dt1/dt2 """
def get_neg_high_socre_views(
    output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/views/neg_high_score",   
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat",
    sample_rate=0.5,
    sample_limit=100,
    random_state = 0,
):
    if not check_path(output_path):
        os.makedirs(output_path)
    tct_all_files = get_neg_slides()
    print(len(tct_all_files))
    tct_all_files_sampled = random.Random(random_state).sample(tct_all_files, int(min(len(tct_all_files) * sample_rate, sample_limit)))
    # tct_all_files_sampled =  tct_all_files.sample()
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    grid_params = [3840, 2160, 0.1, 0]
    topk_limit = 8
    
    pool = mp.Pool(16)
    params_list = []
    for file in tqdm.tqdm(tct_all_files_sampled):
        file_name = Path(file).with_suffix(".json").name
        if "dt1" in file:
            threshold = (0.40, 1)
            max_sc_range = (0.45, 1)
        else:
             threshold = (0.485, 1)
             max_sc_range = (0.50, 1)
             
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4]
        p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold, max_sc_range)
        params_list.append(p)
        
    iter_ = pool.imap_unordered(get_veiw_m, params_list)
    for _ in tqdm.tqdm(iter_, total=len(params_list)):
        pass

""" This function is only for tct dt1/dt2 """
def get_neg_low_socre_views(
    output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/views/neg_low_score",   
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat",
    sample_rate=0.5,
    sample_limit=160,
    random_state = 0,
):
    if not check_path(output_path):
        os.makedirs(output_path)
    tct_all_files = get_neg_slides()
    print(len(tct_all_files))
    tct_all_files_sampled = random.Random(random_state).sample(tct_all_files, int(min(len(tct_all_files) * sample_rate, sample_limit)))
    # tct_all_files_sampled =  tct_all_files.sample()
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    grid_params = [3840, 2160, 0.1, 0]
    topk_limit = 16
    
    pool = mp.Pool(16)
    params_list = []
    for file in tqdm.tqdm(tct_all_files_sampled):
        file_name = Path(file).with_suffix(".json").name
        if "dt1" in file:
            threshold = (0, 0.45)
            max_sc_range = (0, 0.5)
        else:
             threshold = (0, 0.45)
             max_sc_range = (0, 0.5)
             
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4]
        p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold, max_sc_range)
        params_list.append(p)
        
    iter_ = pool.imap_unordered(get_veiw_m, params_list)
    for _ in tqdm.tqdm(iter_, total=len(params_list)):
        pass

""" This function is only for tct dt1/dt2 """
def get_high_conf_pesudo_label_views(
    output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label/high_conf_pos",   
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat",
    sample_rate=1,
    sample_limit=200,
    random_state = 0,
    mask=False):
    if not check_path(output_path):
        os.makedirs(output_path)
    tct_all_files = get_pos_slides()
    print(len(tct_all_files))
    tct_all_files_sampled = random.Random(random_state).sample(tct_all_files, int(min(len(tct_all_files) * sample_rate, sample_limit)))
    # tct_all_files_sampled =  tct_all_files.sample()
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    grid_params = [3840, 2160, 0.1, 0]
    topk_limit = 24
    
    pool = mp.Pool(16)
    params_list = []
    for file in tqdm.tqdm(tct_all_files_sampled):
        file_name = Path(file).with_suffix(".json").name
        if "dt1" in file:
            threshold = (0.60, 1)
            max_sc_range = (0.65, 1)
        else:
             threshold = (0.65, 1)
             max_sc_range = (0.7, 1)
             
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4]
        p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold, max_sc_range, mask)
        params_list.append(p)
        
    iter_ = pool.imap_unordered(get_high_conf_pos_view_m, params_list)
    
    
    for path in tqdm.tqdm(iter_, total=len(params_list)):
        print(path)
        

def get_topk_views(files, k=30, output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_topk_views/",   topk_classes=(0, 1, 2, 4),
    detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat", grid_params = [3840, 2160, 0.1, 0], worker_num=16):
    if not check_path(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    if isinstance(k, int):
        k = [k] * len(files)
    
    pool = mp.Pool(worker_num)
    params_list = []
    for file, k_ in tqdm.tqdm(zip(files, k)):
        file_name = Path(file).with_suffix(".json").name
        json_path = os.path.join(detect_result_dir, file_name)
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), max_score_thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4], sort_view=False
        p = (file, json_path,  grid_params, output_path , k_, (0, 1), (0, 1), topk_classes, True)
        params_list.append(p)
    iter_ = pool.imap_unordered(get_veiw_m, params_list)

    for path in tqdm.tqdm(iter_, total=len(params_list)):
        logger.info(path)

def get_topk_views_by_pairs(files, k=30, output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_topk_views/",   topk_classes=(0, 1, 2, 4),grid_params = [3840, 2160, 0.1, 0], worker_num=16):
    if not check_path(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    pool = mp.Pool(worker_num)
    params_list = []
    for file_pair in tqdm.tqdm(files):
        file, json_file = file_pair
        # slide_path, json_path, grid_params, output_path="", k=8, thredshold_range=(0, 0.5), max_score_thredshold_range=(0, 0.5), class_ids=[0, 1, 2, 4], sort_view=False
        p = (file, json_file,  grid_params, output_path , k, (0, 1), (0, 1), topk_classes, True)
        params_list.append(p)
    iter_ = pool.imap_unordered(get_veiw_m, params_list)

    for path in tqdm.tqdm(iter_, total=len(params_list)):
        logger.info(path)


if __name__ == "__main__":

    """gen views"""
    # print("get_pos_low_socre_views")
    # get_pos_low_socre_views()
    # print("get_neg_heigh_socre_views")
    # get_neg_high_socre_views()
    # print("get_pos_heigh_socre_views")
    # get_pos_heigh_socre_views()
    # print("get_neg_low_socre_views")
    # get_neg_low_socre_views(output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/views/neg_mid_score_ext")
    
    # """gen pesudo labels' views"""
    # get_high_conf_pesudo_label_views()
    
    # get_topk_views(k=32, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    
    
    # six cro
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    # get_topk_views(k=25, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all_hq.csv")["slide_path"].tolist())
    
    get_topk_views(k=32, files=pd.read_csv("/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_all.csv")["slide_path"].tolist(),
                   detect_result_dir="/nasdata/private/jli/cervicalspace/ai_trainer/outputs/mmdet_tct/retinanet_R_50_FPN_mil/inference_tct_all",
                   output_path="/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_new_model_views/topk_viwes", topk_classes=[3, 4, 5, 6, 7])

    # from datasets.our_datasets import WSI_PATCH
    # import glob, random, multiprocessing as mp
    # tct_all_files = get_pos_slides()
    # print(len(tct_all_files))
    # # tct_all_files_sampled = random.sample(tct_all_files, 160)
    # tct_all_files_sampled =  tct_all_files
    # output_path = "/nasdata/private/zwlu/Now/ai_trainer/outputs/pesudo_label"
    # detect_result_dir = "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat"
    # grid_params = [3840, 2160, 0.1, 0]
    # topk_limit = 16
    
    # pool = mp.Pool(32)
    # params_list = []
    # for file in tqdm.tqdm(tct_all_files_sampled):
    #     file_name = Path(file).with_suffix(".json").name
    #     if "dt1" in file:
    #         threshold = 0.6
    #     else:
    #          threshold = 0.7
    #     p = (file, os.path.join(detect_result_dir, file_name),  grid_params, output_path , topk_limit, threshold)
    #     params_list.append(p)
        
    # iter_ = pool.imap_unordered(gen_bboxes_labels_m, params_list)
    # for item in tqdm.tqdm(iter_, total=len(params_list)):
    #     with open(os.path.join(output_path, "anno.txt"), "a") as f:
    #         for label_str in item:
    #             f.write(label_str)
    #             f.write("\n")
    
    # gen_bboxes_labels()
    # gen_bboxes_labels("/nasdata/dataset/moshi_data/dt1/2022-12-26/AIMS-002.ibl", "/nasdata/private/zwlu/Now/ai_trainer/outputs/tct_all_feat_pipeline/detect_model_feat/AIMS-002.json", [1280, 1280, 0.31, 64])
    # crop_views_for_pos()
    # crop_views_for_neg()
    # img = cv2.imread("/nasdata/dataset/wsi_patch_data/dt4/images/1_50388_39240_969_545.jpg")
    # print(img.shape)

    
    # for slide_path in slide_paths:
    #     predict_json = os.path.join(predict_res_dir, Path(slide_path).stem + ".json")
        
    #     print(predict_json)
    #     # 3840x2160
    #     params = {
    #         "crop_pixel_size": 0.1,
    #         "crop_patch_width": 3840,
    #         "crop_patch_height": 2160
    #     }
        
    #     # Path("outputs/topk_views/neg").mkdir(exist_ok=False)
    #     crop_views(slide_path, predict_json, crop_params=params, save_path="outputs/topk_views/neg")
    pass

# end main