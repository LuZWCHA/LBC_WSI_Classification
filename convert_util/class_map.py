
from pathlib import Path
import pandas as pd
import queue
from os import PathLike
import os
from typing import Sequence, Union

try:
    from datasets.our_datasets import WSI_RAW
except:
    pass

__parent_path =  Path(__file__).parent
DEFAULT_RAW_ID_MAP_CSV = __parent_path.joinpath("data/raw_id_mapping.tsv")

class TSV_PROTECT_ATTR:
    CHINESE_NAME = 'chi_name'
    ENGLISH_NAME = 'eng_name'
    RAW_ID = 'raw_id'
    ID = 'id'

def create_name2raw_ids_map(spilt="\t"):
    """example input csv
    chi_name	raw_id	id	eng_name
    滴虫	2	8	TRICH
    念珠菌	3	9	FUNGI
    细菌性阴道病	4	10	CC
    放线菌	5	11	ACTINO
    ... ...
    """
    
    df = pd.read_csv(DEFAULT_RAW_ID_MAP_CSV, sep=spilt)
    eng_names = set(list(df[TSV_PROTECT_ATTR.ENGLISH_NAME]))
    
    ret0 = dict()
    ret1 = dict()
    ret2 = dict()
    for name in eng_names:
        ddf = df[df[TSV_PROTECT_ATTR.ENGLISH_NAME] == name]
        raw_ids = list(ddf[TSV_PROTECT_ATTR.RAW_ID])
        # chinese_names = list(ddf["chi_name"])
        id_ = list(ddf[TSV_PROTECT_ATTR.ID])[0]
        if id_ == "-":
            continue
        ret0[name] = raw_ids
        for i in raw_ids:
            ret2[i] = name
        ret1[name] = int(id_)
    return ret0, ret1, ret2


""" name to raw ids
ACTINO [5]
NILM [34, 35, 36, 37, 38, 39, 40]
HSIL [9]
CC [4]
LSIL [8]
NONE [29, 42, 43, 47, 49, 50, 69]
AGC [13, 14, 15, 16, 1010, 71, 73, 74]
ASCUS [10]
EC [7]
HSV [6]
ASCH [11]
SCC [12]
TRICH [2]
FUNGI [3]"""

""" name to id
ACTINO 11
NILM 0
HSIL 4
CC 10
LSIL 2
NONE -1
AGC 6
ASCUS 1
EC 7
HSV 12
ASCH 3
SCC 5
TRICH 8
FUNGI 9"""

CLASS_NAME_TO_RAW_IDS, CLASS_NAME_TO_ID, RAW_ID_TO_CLASS_NAME = create_name2raw_ids_map()

CLASS_FOURTEEN_NAMES: dict ={
    -1: "NILM",
    0: "ASCUS",
    1: "LSIL",
    2: "ASCH",
    3: "HSIL",
    4: "TRI",
    5: "AGC",
    6: "EC", #子宫内膜细胞
    7: "FUNGI", # 念珠菌
    8: "CC", #细菌性阴道病
    9: "ACTINO", # 放线菌
    10: "HSV", # 疱疹
    
    11: "RC", # 修复细胞
    12: "MP", # 化生细胞
    13: "ECC" # 宫颈管细胞
}

CLASS_18_NAMES: dict ={
    -1: "NILM",
    0: "ASCUS",
    1: "LSIL",
    2: "ASCH",
    3: "HSIL",
    4: "TRI",
    5: "AGC",
    6: "EC", #子宫内膜细胞
    7: "FUNGI", # 念珠菌
    8: "CC", #细菌性阴道病
    9: "ACTINO", # 放线菌
    10: "HSV", # 疱疹
    11: "RC", # 修复细胞
    12: "MP", # 化生细胞
    13: "ECC", # 宫颈管细胞
    
    14: "AGC-NOS",
    15: "AGC-FN",
    16: "AIS",
    17: "ADC",
}

#-1	0	1	2	3	4	5	6	7	8	9	10
#NILM	ASCUS	LSIL	ASCH	HSIL	TRI	AGC	EC	FUNGI	CC	ACTINO	HSV
# This class id map is for wsi patch annotation file.
CLASS_TWELVE_NAMES: dict ={
    -1: "NILM",
    0: "ASCUS",
    1: "LSIL",
    2: "ASCH",
    3: "HSIL",
    4: "TRI",
    5: "AGC",
    6: "EC", #子宫内膜细胞
    7: "FUNGI", # 念珠菌
    8: "CC", #细菌性阴道病
    9: "ACTINO", # 放线菌
    10: "HSV" #疱疹
}

"""HSIL merged SCC used for shishi detect model"""
"""Add the NILM to the TEN CLASS NAMES"""
CLASS_TEN_NAMES_EXT: list = [  # discard other class: SCC merged to HSIL
    "ASCUS", #1
    "LSIL", #2
    "HSIL", # 4  merge SCC #5
    "TRICH", # 8
    "AGC", #6
    "EC", # 7
    "FUNGI", # 9
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
    
    # this is a extension for the ten classes classify, if the score is small enough, the cell class id will set 10
    "NILM",
]

"""HSIL merged SCC used for shishi detect model"""
CLASS_TEN_NAMES: list = [  # discard other class: SCC merged to HSIL
    "ASCUS", #1
    "LSIL", #2
    "HSIL", # 4  merge SCC #5
    "TRICH", # 8
    "AGC", #6
    "EC", # 7
    "FUNGI", # 9
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
]

# AI 2.0 yunkang plus categories
CLASS_TWELVE_NAMES_YUNKANG_PLUS: list = [  # discard other class: SCC merged to HSIL
    "ASCUS", #1
    "LSIL", #2
    "HSIL", # 4  merge SCC #5
    "TRICH", # 8
    "AGC", #6
    "EC", # 7
    "FUNGI", # 9
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
    "MP_RC",
    "ECC"
]

"""The below classes discard, only focus on positive classes.     
    "EC", # 7 
    "TRICH", # 8  
    "FUNGI", # 9                        
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
"""
CLASS_SIX_NAMES = [  # discard other class
    "ASCUS",
    "LSIL",
    "HSIL",
    "ASCH",
    "AGC",
    "SCC"
]

"""The below classes discard, only focus on positive classes.     
    "EC", # 7 
    "TRICH", # 8  
    "FUNGI", # 9                        
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
    
    and HSIL represents HSIL, SCC and ASCH
"""
CLASS_FOUR_NAMES = [  # discard other class
    "ASCUS",
    "LSIL",
    "HSIL",  # merge ASCH
    "AGC",
]


"""The below classes discard, only focus on positive classes.     
    "EC", # 7 
    "TRICH", # 8  
    "FUNGI", # 9                        
    "CC", # 10
    "ACTINO", #11
    "HSV", # 12
    
    and HSIL represents HSIL, SCC and ASCH
    and ASCUS represents ASCUS and LSIL
"""
# ascus < lsil < asch < hsil < agc
CLASS_THREE_NAMES = [  # discard other class
    "ASCUS",  # LSIL
    "HSIL",  # ASCH #SCC
    "AGC",
]

# onehot encoder mapping
name_to_label_3class = {
    "ASCUS": 1, "LSIL": 1,
    "HSIL": 1 << 1,  "ASCH": 1 << 1,
    "AGC": 1 << 2,
}

label_to_name_3class = {
    1: "ASCUS_LSIL",
    2: "HSIL_ASCH_SCC",
    4: "AGC"
}

name_to_label_6class = {
    "ASCUS": 1 << 0,
    "LSIL": 1 << 1,
    "ASCH": 1 << 2,
    "HSIL": 1 << 3,
    "AGC": 1 << 4,
    "SCC": 1 << 5
}

label_to_name_6class = {
    1: "ASCUS",
    2: "LSIL",
    4: "ASCH",
    8: "HSIL",
    16: "AGC",
    32: "SCC"
}

class Grade(object):

    D_PRIORITY = {
        "ASCUS": 1 << 0,
        "LSIL": 1 << 1,
        "ASCH": 1 << 2,
        "HSIL": 1 << 3,
        "AGC": 1 << 4,
        "SCC": 1 << 5
    }

    def __init__(self, grad) -> None:
        self.grad = grad
    
    @staticmethod
    def get_priority(class_name):
        if class_name in Grade.D_PRIORITY:
            return Grade.D_PRIORITY[class_name]
        return 0

    def __lt__(self, other):
        return Grade.get_priority(self.grad) > Grade.get_priority(other.grad)


def get_most_serious_class(grade_str: str, sp="_"):

    if sp not in grade_str:
        return grade_str

    q = queue.PriorityQueue(maxsize=10086) # lol
    grads = grade_str.split(sp)

    for i in grads:
        q.put(Grade(i))

    return q.get().grad


def get_label_id(label_name, not_in_map_set, label_map=name_to_label_3class):

    if label_name not in label_map:
        not_in_map_set.add(label_name)
        return -1

    return label_map[label_name]


log_file = "not_found.log"

def convert_root_path(root, old_paths, new_path):
    '''
    The root path is not same because of the data tansferring
    '''
    for old_path in old_paths.split(';'):
        root = os.path.normpath(root.replace(old_path, new_path))

    if not check_path(root):
        # print(root, "Not Found")
        with open(log_file, 'a') as f:
            f.write(root + "\n")
        # root = "Not Found"
    return root

# the convert method is only for tencent's old path
def default_convert_root_path(root, new_path):
    return convert_root_path(root, "/mnt/group-ai-medical-abp/shared;/cfsdata", new_path)

def check_path(path):
    return os.path.exists(path)


# clean the dataset
def gen_valid_labels_csv(input_csvs: Union[Sequence[PathLike], PathLike], output_dir_name: Union[Sequence[PathLike], PathLike] = "labels_c3", root_dir="/nasdata/ai_data", label_map=name_to_label_3class, file_path_convert=default_convert_root_path):

    if isinstance(input_csvs, PathLike):
        input_csvs = [input_csvs]

    if len(input_csvs) == 0:
        raise RuntimeError("Not label file find.")

    unknow_sets = dict()
        
    for input_csv in input_csvs:
        print(input_csv)
        df = pd.read_csv(input_csv)
        print(df)
        dataset_dir_name = Path(input_csv).stem

        path = os.path.join(root_dir, dataset_dir_name)
        if not check_path(path) or dataset_dir_name not in WSI_RAW.get_subset_names():
            print(f"Skip {path} ...")
            continue

        # check the main columes
        # grade
        # slide path / slide_path_sdpc, two type of grade are available.
        columns = df.columns.array.to_numpy().tolist()
        if "slide_path_sdpc" in columns:
            slide_path = 'slide_path_sdpc'
        else:
            slide_path = 'slide_path'

        slide_paths = df[slide_path]
        grades = df['grade']
        unknow_set = set()
        grades = grades.apply(
            lambda x: get_label_id(get_most_serious_class(x), unknow_set, label_map=label_map)
        )
        unknow_sets[input_csv] = unknow_set

        slide_paths = slide_paths.apply(
            lambda x: file_path_convert(x, path)
        )

        cleaned_csv = pd.concat([slide_paths, grades], axis=1)

        label_save_dir = os.path.join(root_dir, output_dir_name)
        if not check_path(label_save_dir):
            os.makedirs(label_save_dir)

        cleaned_csv.to_csv(os.path.join(
            label_save_dir, dataset_dir_name + ".csv"), index=False)
    
    return unknow_sets
    
    
def convert2name_to_id(names):
    name2id = dict()
    
    for i, name  in enumerate(names):
        name2id[name] = i
        
    return name2id