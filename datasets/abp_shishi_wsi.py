from fvcore.common.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from datasets.our_datasets import SHISHI_RAW


from tools.class_map import CLASS_TEN_NAMES


CLASS_NAMES= CLASS_TEN_NAMES


def get_tct_dicts(anno_file):
    dicts = []
    idx = 0
    for l in PathManager.open(anno_file).readlines():
        data = l.split()
        r = {
                "file_name": data[0],
                "image_id": idx,
                "height": 2160,
                "width": 3840,
                "image_label": [0] * len(CLASS_NAMES)
        }

        instances = []
        for obj in data[1:]:
            ymin, xmin, ymax, xmax, label_id = [int(v) for v in obj.split(',')]
            label = label_id
            # L
            if label_id == -1:
                continue
            elif label_id == 3:
                label = 2
            elif label_id > 3:
                label = label_id - 1

            r['image_label'][label] = 1

            bbox = [xmin, ymin, xmax, ymax]
            instances.append(
                    {"category_id": label, "bbox": bbox, "bbox_mode": BoxMode.XYXY_ABS}
            )

        r["annotations"] = instances
        dicts.append(r)
        idx += 1

    return dicts


def register_tct(name, anno_file):
    DatasetCatalog.register(name, lambda: get_tct_dicts(anno_file))
    MetadataCatalog.get(name).set(thing_classes=CLASS_NAMES)
    MetadataCatalog.get(name).evaluator_type = "coco"


def register_all_tct(name, train_anno_file, test_anno_file):
    SPLITS = [
        (name+"_train", train_anno_file),
        (name+"_test", test_anno_file),
    ]
    for n, a in SPLITS:
        register_tct(n, a)


# train_txt = "/mnt/group-ai-medical-abp/shared/shishi_data/datasetv2/trainset/shishi_train_set_v1.txt"
# test_txt = "/mnt/group-ai-medical-abp/shared/shishi_data/testset/abp_test_v6.txt"
# register_all_tct('abp_shishi', train_txt, test_txt)


train_txt =SHISHI_RAW.at(SHISHI_RAW.default_train_file_path)
test_txt = SHISHI_RAW.at(SHISHI_RAW.default_test_file_path)

register_all_tct('abp_shishi', train_txt, test_txt)