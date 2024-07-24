from functools import partial
import glob

import os
from pathlib import Path

import imageio
from convert_util.class_map import CLASS_TWELVE_NAMES, RAW_ID_TO_CLASS_NAME

import tqdm
try:
    import orjson as orjson
    orjson_exist = True
except:
    orjson_exist = False
    pass

import json

def deserialize_multilabel(json_file):
    with open(json_file, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())
    path = js["image_path"]
    annotations = js["annotations"]
    
    res = []
    for anno in annotations:
        cid_name = "tbs_20"
        for i in anno.keys():
            if "tbs_" in i:
                cid_name = i
        cid = anno[cid_name]
        box = anno["box"]
        instance = box + [cid]
        res.append(instance)
        
    
    return [path, res]
    

def deserialize_coco(coco_json):
    with open(coco_json, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())
        
    info = js["info"]
    licenses = js["licenses"]
    images = js["images"]
    annotations = js["annotations"]
    boxes = dict()
    
    for image in images:
        #{"id": 1, "width": 3840, "height": 2160, "file_name": "/nasdata/dataset/wsi_patch_data/dt1/images/T2019_884_8320_32080_8.jpg", "license": 0}
        width = image["width"]
        height = image["height"]
        file_name = image["file_name"]
        image_id = image["id"]
        boxes[image_id] = [file_name, []]
    
    for anno in annotations:
        # anno = {"category_id": 4, "bbox": [799, 1603, 728, 517], "id": 27, "image_id": 1, "area": 77106, "iscrowd": 0}
        category_id = anno["category_id"]
        xmin, ymin, w, h = anno["bbox"]
        image_id = anno["image_id"]
        label, score, xmin, ymin, xmax, ymax =category_id, 1, xmin, ymin, w + xmin, h + ymin
        boxes[image_id][1].append([xmin, ymin, xmax, ymax, label])

    return list(boxes.values())


def deserialize_output_json(train_output_json):
    with open(train_output_json, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())
        
    
    boxes = []
    bboxes = js["bboxes"]
    w, h = js["width"], js["height"]
    path = js["slide_id"]
    if "image_path" in js:
        path = js["image_path"]
    if "slide_path" in js:
        path = js["slide_path"]
    for i in bboxes:
        c_id, score, xmin, ymin, xmax, ymax = [float(j) for j in i]
        c_id = int(c_id)
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        
        boxes.append([c_id, score, xmin, ymin, xmax, ymax])
        
    return [path, w, h, boxes]


def deserialize_output_json_ai3(testset_output_json="/nasdata/private/zwlu/Now/ai_trainer/.data/slides/AHSLAI-001.sdpc-det_cls.json", return_meta_data=False):
    # /nasdata/private/zwlu/Now/ai_trainer/.data/slides/AHSLAI-001.sdpc-det_cls.json
    import gc
    gc.disable()
    with open(testset_output_json, "r") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())
    gc.enable()
    
    """
    CLASS_20_NAMES_V3: dict = {
    -1: "NILM",
    0: "ASCUS_S",
    1: "LSIL_S",
    2: "ASCH_S",
    3: "HSIL_S",
    4: "ASCUS_M",
    5: "LSIL_M",
    6: "ASCH_M",
    7: "HSIL_M",
    8: "TRI",
    9: "AGC",
    10: "EC",
    11: "FUNGI",
    12: "CC",
    13: "ACTINO",
    14: "HSV",
    
    15: "MP_RC",
    16: "ECC",
    17: "SCC",
    18: "AGC_NOS",
    19: "AGC_FN"
}
    """
    
    
    bboxes = js["bboxes"]
    slide_path = js["slide_path"]
    ratio = js["ratio"]
    width = js["width"]
    height = js["height"]
    
    num_patchs = js["num_patchs"]
    
    boxes = []
    pos_boxes = []
    for bbox in bboxes:
        #[0, "0.13", 50260, 44210, 50422, 43369, -1.0, -1]
        cls_id = bbox[0]
        score = float(bbox[1])
        xmin, ymin, xmax, ymax = list(map(int, bbox[2:6]))
        
        refine_score = float(bbox[6])
        refine_cls_id = int(bbox[7])
        
        boxes.append([cls_id, score, xmin, ymin, xmax, ymax, refine_score, refine_cls_id])
        if cls_id in [0, 1, 2, 3, 4, 5, 6, 7, 9, 17, 18, 19]:
            pos_boxes.append([cls_id, score, xmin, ymin, xmax, ymax, refine_score, refine_cls_id])
    
    
    max_score = -1
    max_score_cls_id = -1
    # sort by score
    if len(pos_boxes) > 0:
        pos_boxes.sort(key=lambda x: -x[1])
        
        max_score = pos_boxes[0][1]
        max_score_cls_id = pos_boxes[0][0]
    
    if not return_meta_data:
        return boxes, pos_boxes, max_score, max_score_cls_id
    else:
        return boxes, pos_boxes, max_score, max_score_cls_id, width, height, slide_path
    

def deserialize_miis(miis_json_path):
    with open(miis_json_path, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())

    boxes = []
    for i in js:
        image_id = 0
        image_url = filename = ""
        if "Id" in i:
            image_id = i["Id"]
        if "ImgUrl" in i:
            image_url = i["ImgUrl"]
        if "Filename" in i:
            filename = i["Filename"]
        
        normal_name = filename if filename != "" else image_url
        
        boxes_raw = i["Mark"]
        instances = []
        for b in boxes_raw:
            # {"StartX":0.0357000008225441,"StartY":0.7634000182151794,"EndX":0.0812000036239624,"EndY":0.8755000233650208,"Tag":"8","TagName":"LSIL","Score":0}
            start_x = float(b["StartX"])
            start_y = float(b["StartY"])
            end_x = float(b["EndX"])
            end_y = float(b["EndY"])
            c_id = int(b["Tag"])
            c_name = b["TagName"]
            score = float(b["Score"])
            instances.append((start_x, start_y, end_x, end_y, c_id, c_name, score))
        boxes.append((image_id, normal_name, instances))

    return boxes


def deserialize_raw_dataset_export(dataset_export_json_path):
    with open(dataset_export_json_path, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())

    boxes = []
    for i in js:
        image_id = 0
        image_url = filename = ""
        if "id" in i:
            image_id = i["id"]
        if "imgUrl" in i:
            image_url = i["imgUrl"]
        if "filename" in i:
            filename = i["filename"]
        
        normal_name = filename if filename != "" else image_url
        
        boxes_raw = i["markData"]
        instances = []
        for b in boxes_raw:
            # {"id":679067,"start_x":0.3099,"start_y":0.0815,"end_x":0.4094,"end_y":0.2292,"tag_option_id":11,"tag_option_name":"ASC-H"}]}
            start_x = float(b["start_x"])
            start_y = float(b["start_y"])
            end_x = float(b["end_x"])
            end_y = float(b["end_y"])
            c_id = int(b["tag_option_id"])
            # c_name = b["tag_option_name"]
            # score = float(b["Score"])
            instances.append((int(start_x * 3840), int(start_y * 2160), int(end_x * 3840), int(end_y * 2160), c_id))
        boxes.append((image_id, normal_name, instances))

    return boxes


def deserialize_miis_with_auditor(miis_json_path):
    with open(miis_json_path, "rb") as f:
        if not orjson_exist:
            js = json.load(f)
        else:
            js = orjson.loads(f.read())

    boxes = []
    for i in js:
        image_id = 0
        image_url = filename = ""
        if "Id" in i:
            image_id = i["Id"]
        if "ImgUrl" in i:
            image_url = i["ImgUrl"]
        if "Filename" in i:
            filename = i["Filename"]
        auditor = i["Auditor"]
        
        normal_name = filename if filename != "" else image_url
        
        boxes_raw = i["Mark"]
        instances = []
        for b in boxes_raw:
            # {"StartX":0.0357000008225441,"StartY":0.7634000182151794,"EndX":0.0812000036239624,"EndY":0.8755000233650208,"Tag":"8","TagName":"LSIL","Score":0}
            start_x = float(b["StartX"])
            start_y = float(b["StartY"])
            end_x = float(b["EndX"])
            end_y = float(b["EndY"])
            c_id = int(b["Tag"])
            c_name = b["TagName"]
            score = float(b["Score"])
            instances.append((start_x, start_y, end_x, end_y, c_id, c_name, score))
        boxes.append((image_id, normal_name, auditor, instances))

    return boxes

def deserialize_txt(txt_path):
    """deserialize train txt file

    Args:
        txt_path (str): txt file path

    Returns:
        List[Tuple]: (pic_path, instances)
        instance: xmin, ymin, xmax, ymax, label
    """
    with open(txt_path, encoding="utf8") as f:
        lines = f.readlines()

    all_lines = [i.strip() for i in lines]
    boxes = []
    for items in all_lines:
        data = items.split(" ")
        pic_path = data[0].strip()
        # label_path = img2label_paths([pic_path])[0]
        # if not os.path.exists(pic_path):
        #     print(pic_path)
        #     continue
        instances = []
        if len(data) > 1:
            for obj in data[1:]:
                ymin, xmin, ymax, xmax, label_id = [int(v) for v in obj.split(',')]
                label = label_id
                instances.append((xmin, ymin, xmax, ymax, label))

        boxes.append([pic_path, instances])
                # center_x = (xmin + xmax) / pic_size[0] / 2
                # center_y = (ymin + ymax) / pic_size[1] / 2
                # w = (xmax - xmin) / pic_size[0] 
                # h = (ymax - ymin) / pic_size[1] 
                
                # str_list = [str(j) for j in (ymin, xmin, ymax, xmax, label_id)]
                
                # instances.append(" ".join(str_list).strip() + "\n")
    return boxes


def deserize_pred_res_txt(txt_path):
    with open(txt_path, "r") as f:
        lines = f.readlines()

    all_lines = [i.strip() for i in lines]
    boxes = []
    for items in all_lines:
        data = items.split(" ")
        pic_path = data[0].strip()
        # label_path = img2label_paths([pic_path])[0]
        # if not os.path.exists(pic_path):
        #     print(pic_path)
        #     continue
        instances = []
        if len(data) > 1:
            for obj in data[1:]:
                ins = [float(v) for v in obj.split(',')]
                while len(ins) >= 6:
                    y_min, x_min,y_max, x_max, class_id,score = ins[:6]
                    ins = ins[6:]
                label = class_id
                instances.append((int(x_min), int(y_min), int(x_max), int(y_max), score, int(label)))

        boxes.append((pic_path, instances))
                # center_x = (xmin + xmax) / pic_size[0] / 2
                # center_y = (ymin + ymax) / pic_size[1] / 2
                # w = (xmax - xmin) / pic_size[0] 
                # h = (ymax - ymin) / pic_size[1] 
                
                # str_list = [str(j) for j in (ymin, xmin, ymax, xmax, label_id)]
                
                # instances.append(" ".join(str_list).strip() + "\n")
    return boxes


def miis_classid2fourteen_classes(raw_id):
    if raw_id == 34:
        return 11
    if raw_id == 35:
        return 12
    if raw_id == 38:
        return 13

    if raw_id not in RAW_ID_TO_CLASS_NAME:
        name = "NONE"
    else:
        name = RAW_ID_TO_CLASS_NAME[raw_id]
    if name == "NONE":
        name = "NILM"
    if name == "SCC":
        name = "HSIL"
        
    if name == "TRICH":
        name = "TRI"
    
    # merge NONE to NILM, SCC to HSIL
    for k, v in CLASS_TWELVE_NAMES.items():
        if v == name:
            return k
    return -1


def miis_classid_to_CLASS_TWELVE_NAMES_EXT_V3(raw_id):
    if raw_id in [34, 35]:
        return 15 - 4
    if raw_id in [38]:
        return 16 - 4
    if raw_id in [15, 16, 1010, 71, 73, 74]:
        return 9 - 4
    if raw_id in [12]:
        return 17 - 4 
    if raw_id in [7]:
        return 10 - 4
    if raw_id in [13]:
        return 18 - 4
    if raw_id in [14]:
        return 19 - 4
    if raw_id in [6]:
        return 14 - 4
    if raw_id in [2]:
        return 8 - 4
    if raw_id in [5]:
        return 13 - 4
    if raw_id in [4]:
        return 12 - 4
    if raw_id in [3]:
        return 11 - 4
    if raw_id in [10]:
        return 0
    if raw_id in [8]:
        return 1
    if raw_id in [9]:
        return 3
    if raw_id in [11]:
        return 2
    
    return -1
    

def miis_classid2twelve_classes(raw_id):
    if raw_id not in RAW_ID_TO_CLASS_NAME:
        name = "NONE"
    else:
        name = RAW_ID_TO_CLASS_NAME[raw_id]
    if name == "NONE":
        name = "NILM"
    if name == "SCC":
        name = "HSIL"
        
    if name == "TRICH":
        name = "TRI"
        
    # merge NONE to NILM, SCC to HSIL
    for k, v in CLASS_TWELVE_NAMES.items():
        if v == name:
            return k
    return -1

def twelve_classes2ten_classes(label_id):
    if label_id == -1:
        return -1
    elif label_id == 3:
        label_id = 2
    elif label_id > 3:
        label_id = label_id - 1
    return label_id
    
def miispath2localpath(miis_path, name2path):
    file_name = miis_path.split("/")[-1]
    return name2path[file_name]

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def miis_json2train_txt(miis_json_path, txt_output_path, miispath2localpath, id_trans=miis_classid2twelve_classes):
    train_file_items = dict()
    boxes = deserialize_miis(miis_json_path)
    with open(txt_output_path, "w") as f:
        for b in tqdm.tqdm(boxes):
            image_id, image_url, instances = b
            local_path = miispath2localpath(image_url)
            H, W = imageio.v2.imread(local_path).shape[:2]
            instances_trans = [(int(start_y * H), int(start_x * W), int(end_y * H), int(end_x * W),  id_trans(c_id)) for start_x, start_y, end_x, end_y, c_id, c_name, score in instances]
            if local_path not in train_file_items:
                train_file_items[local_path] = []
            train_file_items[local_path].append(instances_trans)
            f.write(local_path)
            if len(instances_trans) > 0:
                for j in instances_trans:
                    label_str = ",".join([str(i) for i in j])
                    f.write(" " + label_str)
            f.write("\n")
    return train_file_items

def anno_text2pred_json(txt_path, json_dir_path, id_trans=twelve_classes2ten_classes, max_score_classes=[0, 1, 2, 4]):
    
    boxes = deserialize_txt(txt_path)
    
    for b in tqdm.tqdm(boxes):
        pic_path, instances = b
        save_path = os.path.join(json_dir_path, Path(pic_path).stem + ".json")
        new_instance = []
        max_score = 0
        for i in instances:
            xmin, ymin, xmax, ymax, label = i
            score = 1
            ten_class_id = id_trans(label)
            new_instance.append([ten_class_id, score, xmin, ymin, xmax, ymax])
            if ten_class_id in max_score_classes:
                max_score = max(max_score, ten_class_id)
        
        H, W = imageio.imread(pic_path).shape[:2]
        
        json_dict = dict()
        json_dict["max_score"] = max_score
        json_dict["bboxes"] = new_instance
        json_dict["slide_id"] = Path(pic_path).stem
        json_dict["slide_path"] = pic_path
        json_dict["width"] = H
        json_dict["height"] = W
        
        with open(save_path, "w") as f:
            if not orjson_exist:
                json.dump(json_dict, f)
            else:
                f.write(orjson.dumps(json_dict))
  

def pred_res_txt2json(pred_res_txt_path, json_dir_path, id_trans=twelve_classes2ten_classes, max_score_classes=[0, 1, 2, 4], image_path="/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_mmd_model_views/topk_viwes/sampled_data/images"):
    boxes = deserize_pred_res_txt(pred_res_txt_path)
    
    for b in tqdm.tqdm(boxes):
        pic_path, instances = b
        save_path = os.path.join(json_dir_path, Path(pic_path).stem + ".json")
        new_instance = []
        max_score = 0
        for i in instances:
            x_min, y_min, x_max, y_max, score, label = i
            ten_class_id = id_trans(label)
            new_instance.append([ten_class_id, score, x_min, y_min, x_max, y_max])
            if ten_class_id in max_score_classes:
                max_score = max(max_score, score)
        
        if image_path is not None:
            pic_path = os.path.join(image_path, pic_path)
        H, W = imageio.imread(pic_path).shape[:2]
        
        json_dict = dict()
        json_dict["max_score"] = max_score
        json_dict["bboxes"] = new_instance
        json_dict["slide_id"] = Path(pic_path).stem
        json_dict["slide_path"] = pic_path
        json_dict["width"] = H
        json_dict["height"] = W
        
        with open(save_path, "w") as f:
            if not orjson_exist:
                json.dump(json_dict, f)
            else:
                f.write(orjson.dumps(json_dict))

    
if __name__ == "__main__":
    print(deserialize_output_json_ai3())
    
    # files =  glob.glob(os.path.join("/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_mmd_model_views_15000/topk_viwes/**", "*.jpg"), recursive=True)
    # name2path = {os.path.basename(i): i for i in files}
    
    # miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230615.json", 
    #                     "/nasdata/private/zwlu/Now/ai_trainer/.data/20230615.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid2twelve_classes)
    
    # miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230607.json", 
    #                     "/nasdata/private/zwlu/Now/ai_trainer/.data/20230607.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid2twelve_classes)
    
    # miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230522.json", 
    #                     "/nasdata/private/zwlu/Now/ai_trainer/.data/20230522.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid2twelve_classes)
    
    # miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230515.json", 
    #                     "/nasdata/private/zwlu/Now/ai_trainer/.data/20230515.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid2twelve_classes)



    files =  glob.glob(os.path.join("/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_mmd_model_views_15000/topk_viwes/**", "*.jpg"), recursive=True)
    name2path = {os.path.basename(i): i for i in files}
    
    miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230615.json", 
                        "/nasdata/private/zwlu/Now/ai_trainer/.data/20230615_v3.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid_to_CLASS_TWELVE_NAMES_EXT_V3)
    
    miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230607.json", 
                        "/nasdata/private/zwlu/Now/ai_trainer/.data/20230607_v3.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid_to_CLASS_TWELVE_NAMES_EXT_V3)
    
    miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230522.json", 
                        "/nasdata/private/zwlu/Now/ai_trainer/.data/20230522_v3.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid_to_CLASS_TWELVE_NAMES_EXT_V3)
    
    miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/20230515.json", 
                        "/nasdata/private/zwlu/Now/ai_trainer/.data/20230515_v3.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid_to_CLASS_TWELVE_NAMES_EXT_V3)




    # anno_text2pred_json("/nasdata/private/zwlu/Now/ai_trainer/.data/20230522.txt", "outputs/view_annos4view/mmd_results_15000_0522")
    
    # os.makedirs("outputs/moshi_2000/mmd_NMS_results", exist_ok=True)
    
    # files =  glob.glob(os.path.join("/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_mmd_model_views/topk_viwes/sampled_data/images", "*.jpg"))
    # name2path = {os.path.basename(i): i for i in files}
    
    # miis_json2train_txt("/nasdata/private/zwlu/Now/ai_trainer/.data/moshi_2000.json", 
    #                     "/nasdata/private/zwlu/Now/ai_trainer/.data/moshi_2000.txt", miispath2localpath=partial(miispath2localpath, name2path=name2path), id_trans=miis_classid2twelve_classes)

    # anno_text2pred_json("/nasdata/private/zwlu/Now/ai_trainer/.data/crop_views/moshi_2000.txt", "outputs/crop_views/moshi_2000/mmd_NMS_results")
    
    # pred_res_txt2json("/nasdata/private/zwlu/Now/ai_trainer/outputs/crop_views/tct_mmd_model_views/topk_viwes/sampled_data/pesudo_labels/labels.txt", "outputs/crop_views/moshi_2000/mmd_NMS_results", lambda x: x, [3, 4, 5, 6, 7])
    
    
# end main