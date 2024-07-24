from fvcore.common.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

from datasets.our_datasets import TCT_WSI_RAW

###################################################################
#
# TODO: this file is used to predict tct data, no annotation now
#
###################################################################

CLASS_NAMES= [
        "ASCUS",
        "LSIL",
        "HSIL",
        "TRICH",
        "AGC",
        "EC",
        'FUNGI',
        'CC',
        'ACTINO',
        'HSV',
]


def get_tct_dicts(anno_file):
    dicts = []
    idx = 0
    for l in PathManager.open(anno_file).readlines():
        data = l.split()
#         print('\ndata: ', l)
#         print('\ndata: ', data[0])
#         print('\ndata: ', data[1:])
        if '220_029875_30258_18910_3362_1891.jpg' in data[0]:
            print(l)
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

# TODO
train_txt = "outputs/pesudo_label/annos/mix_annos_13/train.txt"
test_txt = "outputs/pesudo_label/annos/mix_annos_13/test.txt"

register_all_tct('abp_tct_wsi_patch', train_txt, test_txt)
