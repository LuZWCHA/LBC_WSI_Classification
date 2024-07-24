# This file contains all dataset/sub_dataset we have generated
import glob
import os
from pathlib import Path
from slide_detect_tools.slide_supports import *
from os.path import join
import yaml

from tools import classproperty

"""by nowandfuture"""

CONFIG_FORMAT = ".yaml"
DEFAULT_CONFIG_PATH = "dataset_configs/"

config_dir_path = join(Path(__file__).parent, DEFAULT_CONFIG_PATH)

def name2path(file_list):
    name2path_dict = dict()
    for file_path in file_list:
        name2path_dict[Path(file_path).stem] = file_path
    
    return name2path_dict

class LazyLoadValue():
    
    def __init__(self, key, config_path: str) -> None:
        self.config_path = config_path
        self.key = key
        
        self.search_files = None
        self.cache = None
        self.has_loaded = False
        
    def __load(self, key):

        candaniates = []
        
        if self.search_files is None:
            if Path(self.config_path).is_dir():
                candaniates = glob.glob(join(self.config_path, f"**/*{CONFIG_FORMAT}"), recursive=True)
            elif self.config_path.endswith(CONFIG_FORMAT):
                candaniates.append(self.config_path)
            self.search_files = candaniates
        else:
            candaniates = self.search_files
        
        for path in candaniates:
            with open(path) as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
            if key in data:
                return data[key]
        
        return None
            
    def load(self):
        if not self.has_loaded:
            self.cache = self.__load(self.key)
            self.has_loaded = True
            
        return self.cache

class LazyConfig():
    
    def __init__(self, config_path) -> None:
        self.dict = dict()
        self.config_path = config_path
        
    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = LazyLoadValue(key, self.config_path)
        
        return self.dict[key]

__CONFIG__ = LazyConfig(config_dir_path)

#################################################################################################################################################
#                                                                              Datasets-Hardcode to get the global dataset path easily
#                                                                              NOTE: IF you want to add a new dataset
#                                                                              please follow the example dataset class bellow
#################################################################################################################################################

# ALL SHISHI DATASET
def default_fliter(path: Path):
    return path.is_file()

def get_file_size(path: Path):
    fsize = path.stat().st_size
    fsize = fsize / float(1024 * 1024)
    return round(fsize,5)

def print_dataset_info(dataset_path, data_format="**/*", fliter=default_fliter, print_all=False, _print=True):
    cnt = 0
    data_size_max = 0
    data_size_min = 1e9
    all_info = []
    sum_size = 0
    for i in Path(dataset_path).glob(data_format):
        i: Path
        if(fliter(i)):
            cnt += 1
            size_ = get_file_size(i)
            sum_size += size_
            data_size_max = max(data_size_max, size_)
            data_size_min = min(data_size_min, size_)
            if print_all:
                all_info.append((i.name, size_))
                if _print:
                    print(i.name, f"size: {size_}MB")
    if _print:       
        print(f"file number: {cnt}, file total size: {sum_size}MB, max size: {data_size_max}MB, min size: {data_size_min}MB, avg size: {round(sum_size / max(cnt, 1e-9), 5)}MB")
        
    return cnt, sum_size, data_size_min, data_size_max, all_info

class BaseDatasetConfig:

    @classproperty
    def root_path(cls):
        return __CONFIG__[cls.__name__].load()
    
    @classmethod
    def at(cls, sub_path):
        return os.path.normpath(os.path.join(cls.root_path, sub_path))

    @classmethod
    def print_info(cls, print_all=False, fliter=default_fliter, do_print=True):
        return print_dataset_info(cls.root_path, print_all=print_all, fliter=fliter, do_print=do_print)

    @classmethod
    def get_target_files(cls, search_subdirs="**/*.*"):
        return glob.iglob(os.path.join(cls.root_path, search_subdirs), recursive=True)

###########################################   The datasets will be move to other script files   ###############################################

## TODO
    
'''
This is shishi raw dataset without any preprocess proagrams
'''
class SHISHI_RAW(BaseDatasetConfig):
    
    default_train_file_path = "annotations_shishi_train.txt"
    default_test_file_path = "annotations_shishi_test.txt"
    default_anno_file_name = 'annotations.txt'
    
    #dt1  dt2  dt3  dt4  dt5  dt6    
    def __init__(self) -> None:
        pass
     
    @staticmethod
    def get_subset_name():
    
        res = []

        for subfile in Path(SHISHI_RAW.root_path).glob("*"):
            if not subfile.is_file():
                res.append(subfile.name)
    
        return res

    @staticmethod
    def get_default_train_datasets():
        return ['dt3', 'dt6']
        
    @staticmethod
    def get_default_test_datasets():
        return  ['dt1', 'dt2', 'dt4', 'dt5']

        

class WSI_RAW_LABEL(BaseDatasetConfig):
    pass

'''
This is wsi raw data, no preprocess.
'''
class WSI_RAW(BaseDatasetConfig):

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_subset_names():
        res = []
        label_paths = WSI_RAW.get_all_label_paths()
        
        for subfile in Path(WSI_RAW.root_path).glob("*"):
            if not subfile.is_file() and subfile.name not in label_paths:
                res.append(subfile.name)
    
        return res
        
    @staticmethod
    def get_label_path(class_num=3):
        return Path(WSI_RAW.root_path).joinpath(f"label_c{class_num}").name
        
    @staticmethod
    def get_all_label_paths():
        return [WSI_RAW.get_label_path(i) for i in [3, 6]]

    @staticmethod
    def get_default_train_datasets():
        return ['dt1']
        
    @staticmethod
    def get_default_test_datasets():
        return [ 'dt2',
                         'agc-in-abp-storage',
                         'dt3',
                         'jys_agc_storage',
                         'jys_agc_0907',
                         'dt5',
                         'dt4']


class WSI_PATCH(BaseDatasetConfig):
    
    default_train_file_path = 'annotations_train.txt'
    default_test_file_path = 'annotations_test.txt'
    
    default_anno_file_name =  'annotations.txt'
    
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def get_subset_name():
        res = []
        # label_paths = WSI_RAW.get_all_label_paths()
        for subfile in Path(WSI_RAW.root_path).glob("*"):
            if not subfile.is_file():
                res.append(subfile.name)
    
        return res
    
    @staticmethod
    def get_label_path(subset_name):
        return os.path.join(WSI_PATCH.root_path, subset_name, WSI_PATCH.default_anno_file_name)
    
    @staticmethod
    def get_default_train_datasets():
        return ['dt2', 'dt4', 'dt5', 'dt7', 'dt10', 'dt11', 'dt12', 'dt13']
        
    @staticmethod
    def get_default_test_datasets():
        return ['dt1', 'dt3', 'dt6', 'dt8', 'dt9']

class MOSHI_DATASET(BaseDatasetConfig):

    pass# This file contains all dataset/sub_dataset we have generated
import glob
import os
from pathlib import Path
from slide_detect_tools.slide_supports import *
from os.path import join
import yaml

from tools import classproperty

"""by nowandfuture"""

CONFIG_FORMAT = ".yaml"
DEFAULT_CONFIG_PATH = "dataset_configs/"

config_dir_path = join(Path(__file__).parent, DEFAULT_CONFIG_PATH)

class LazyLoadValue():
    
    def __init__(self, key, config_path: str) -> None:
        self.config_path = config_path
        self.key = key
        
        self.search_files = None
        self.cache = None
        self.has_loaded = False
        
    def __load(self, key):

        candaniates = []
        
        if self.search_files is None:
            if Path(self.config_path).is_dir():
                candaniates = glob.glob(join(self.config_path, f"**/*{CONFIG_FORMAT}"), recursive=True)
            elif self.config_path.endswith(CONFIG_FORMAT):
                candaniates.append(self.config_path)
            self.search_files = candaniates
        else:
            candaniates = self.search_files
        
        for path in candaniates:
            with open(path) as f:
                data = yaml.load(f, Loader=yaml.SafeLoader)
            if key in data:
                return data[key]
        
        return None
            
    def load(self):
        if not self.has_loaded:
            self.cache = self.__load(self.key)
            self.has_loaded = True
            
        return self.cache

class LazyConfig():
    
    def __init__(self, config_path) -> None:
        self.dict = dict()
        self.config_path = config_path
        
    def __getitem__(self, key):
        if key not in self.dict:
            self.dict[key] = LazyLoadValue(key, self.config_path)
        
        return self.dict[key]

__CONFIG__ = LazyConfig(config_dir_path)

#################################################################################################################################################
#                                                                              Datasets-Hardcode to get the global dataset path easily
#                                                                              NOTE: IF you want to add a new dataset
#                                                                              please follow the example dataset class bellow
#################################################################################################################################################

# ALL SHISHI DATASET
def default_fliter(path: Path):
    return path.is_file()

def get_file_size(path: Path):
    fsize = path.stat().st_size
    fsize = fsize / float(1024 * 1024)
    return round(fsize,5)

def print_dataset_info(dataset_path, data_format="**/*", fliter=default_fliter, print_all=False, _print=True):
    cnt = 0
    data_size_max = 0
    data_size_min = 1e9
    all_info = []
    sum_size = 0
    for i in Path(dataset_path).glob(data_format):
        i: Path
        if(fliter(i)):
            cnt += 1
            size_ = get_file_size(i)
            sum_size += size_
            data_size_max = max(data_size_max, size_)
            data_size_min = min(data_size_min, size_)
            if print_all:
                all_info.append((i.name, size_))
                if _print:
                    print(i.name, f"size: {size_}MB")
    if _print:       
        print(f"file number: {cnt}, file total size: {sum_size}MB, max size: {data_size_max}MB, min size: {data_size_min}MB, avg size: {round(sum_size / max(cnt, 1e-9), 5)}MB")
        
    return cnt, sum_size, data_size_min, data_size_max, all_info

# class BaseDatasetConfig:

#     @classproperty
#     def root_path(cls):
#         return __CONFIG__[cls.__name__].load()
    
#     @classmethod
#     def at(cls, sub_path):
#         return os.path.normpath(os.path.join(cls.root_path, sub_path))

#     @classmethod
#     def print_info(cls, print_all=False, fliter=default_fliter, do_print=True):
#         return print_dataset_info(cls.root_path, print_all=print_all, fliter=fliter, do_print=do_print)
    
#     @classmethod
#     def get_target_files(cls, search_subdirs="**/*.*"):
#         return glob.glob(os.path.join(cls.root_path, search_subdirs), recursive=True)

'''
This is shishi raw dataset without any preprocess proagrams
'''
class SHISHI_RAW(BaseDatasetConfig):
    
    default_train_file_path = "annotations_train_new.txt"
    default_test_file_path = "annotations_test_new.txt"
    default_anno_file_name = 'annotations.txt'
    
    #dt1  dt2  dt3  dt4  dt5  dt6    
    def __init__(self) -> None:
        pass
     
    @staticmethod
    def get_subset_name():
    
        res = []

        for subfile in Path(SHISHI_RAW.root_path).glob("*"):
            if not subfile.is_file():
                res.append(subfile.name)
    
        return res

    @staticmethod
    def get_default_train_datasets():
        return ['dt3', 'dt6']
        
    @staticmethod
    def get_default_test_datasets():
        return  ['dt1', 'dt2', 'dt4', 'dt5']

        

class WSI_RAW_LABEL(BaseDatasetConfig):
    pass

'''
wsi raw data, no preprocess.
'''
class WSI_RAW(BaseDatasetConfig):

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_subset_names():
        res = []
        label_paths = WSI_RAW.get_all_label_paths()
        
        for subfile in Path(WSI_RAW.root_path).glob("*"):
            if not subfile.is_file() and subfile.name not in label_paths:
                res.append(subfile.name)
    
        return res
        
    @staticmethod
    def get_label_path(class_num=3):
        return Path(WSI_RAW.root_path).joinpath(f"label_c{class_num}").name
        
    @staticmethod
    def get_all_label_paths():
        return [WSI_RAW.get_label_path(i) for i in [3, 6]]

    @staticmethod
    def get_default_train_datasets():
        return ['dt1']
        
    @staticmethod
    def get_default_test_datasets():
        return [ 'dt2',
                         'agc-in-abp-storage',
                         'dt3',
                         'jys_agc_storage',
                         'jys_agc_0907',
                         'dt5',
                         'dt4']

class WSI_PATCH(BaseDatasetConfig):
    
    default_train_file_path = 'annotations_train_new.txt'
    default_test_file_path = 'annotations_test_new.txt'
    
    lq_train_file_path = 'annotations_train_lq.txt'
    hq_train_file_path = 'annotations_train_hq.txt'
    
    default_anno_file_name =  'annotations.txt'
    
    def __init__(self) -> None:
        pass
        
    @staticmethod
    def get_subset_names():
        res = []
        # label_paths = WSI_RAW.get_all_label_paths()
        for subfile in Path(WSI_RAW.root_path).glob("*"):
            if not subfile.is_file():
                res.append(subfile.name)
    
        return res
    
    @staticmethod
    def get_label_path(subset_name):
        return os.path.join(WSI_PATCH.root_path, subset_name, WSI_PATCH.default_anno_file_name)
    
    @staticmethod
    def get_default_train_datasets():
        return ['dt2', 'dt4', 'dt5', 'dt7', 'dt10', 'dt11', 'dt12', 'dt13']
        
    @staticmethod
    def get_default_test_datasets():
        return ['dt1', 'dt3', 'dt6', 'dt8', 'dt9']
    
    @staticmethod
    def test_mixed_dataset_2k():
        return ['dt2', 'dt3']
        
    @staticmethod
    def test_mixed_dataset_3k():
        return ['dt4', 'dt3']

class TCT_WSI_RAW(BaseDatasetConfig):

    default_train_file_path = 'annotations_train.txt'
    yolo_train_file_path = 'annotations_train_yolo.txt'
    default_test_file_path = 'annotations_test.txt'
    yolo_test_file_path = 'annotations_test_yolo.txt'

    @staticmethod
    def get_subset_names():
        res = []
        # label_paths = WSI_RAW.get_all_label_paths()
        for subfile in Path(TCT_WSI_RAW.root_path).glob("dt*"):
            if not subfile.is_file():
                res.append(subfile.name)
    
        return res
        
class TCT_WSI_RAW_LABEL(BaseDatasetConfig):
    pass

class Cx22_SEG_RAW(BaseDatasetConfig):
    pass

class WSI_SHISHI_MIX_RAW(BaseDatasetConfig):
    default_train_file_path = 'annotations_train.txt'
    yolo_train_file_path = 'annotations_train_yolo.txt'
    default_test_file_path = 'annotations_test.txt'
    yolo_test_file_path = 'annotations_test_yolo.txt'
    pass

# mix shishi train and test set
class WSI_SHISHI_MIX_V2_RAW(BaseDatasetConfig):
    default_train_file_path = 'annotations_train.txt'
    yolo_train_file_path = 'annotations_train_yolo.txt'
    default_test_file_path = 'annotations_test.txt'
    yolo_test_file_path = 'annotations_test_yolo.txt'
    pass

class LR_SR_COMPARE_TEST_RAW(BaseDatasetConfig):
    pass