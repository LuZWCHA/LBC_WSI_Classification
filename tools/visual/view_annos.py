import glob
import numpy as np
from convert_util import label_convert
import os
from tools.visual.box_visualizer import draw_bounding_boxes_on_image_array
import imageio.v2 as imageio
from tools.visual.plot_utils import plot_many



def view_anno_view_level_multilabel_v2(json_path, id2name=None, respath2abspath=None):
    from convert_util.label_convert import deserialize_txt, deserialize_coco, deserialize_multilabel
    path, instances = deserialize_multilabel(json_path)
    
    print(len(instances))

    if respath2abspath is not None:
        path = respath2abspath(path)
    
    # if "AIMS-526_19096.0_34515.0.jpg" not in path:
    #     continue
    # instance = i[1]
    # print(len(instance), instance)
    if len(instances) > 0:
        # print(instance)
        b = np.array(instances)

        image_array  = imageio.imread(path)
        # print(image_array)
        display_str_list_list = [[str(j) if id2name is None else id2name(j)] for j in b[:, -1]]
        # display_str_list_list = [[str(j) if id2name is None else id2name(j)] for j in b[:, -1]]
        draw_bounding_boxes_on_image_array(image_array, b[:,:-1], display_str_list_list=display_str_list_list, box_mode="xyxy")
        print(path)
        plot_many(image_array)


def view_anno_view_level_coco(json_path, id2name=None, respath2abspath=None):
    from convert_util.label_convert import deserialize_txt, deserialize_coco
    instances = deserialize_coco(json_path)
    cnt = 24
    print(len(instances))
    for i in instances:
        path = i[0]
        if respath2abspath is not None:
            path = respath2abspath(path)
        
        # if "AIMS-526_19096.0_34515.0.jpg" not in path:
        #     continue
        instance = i[1]
        # print(len(instance), instance)
        if len(instance) > 0:
            # print(instance)
            b = np.array(instance)
            if (b[:, -1] >= 12).sum() == 0:
                print("skip")
                continue
            print(path)
            image_array  = imageio.imread(path)
            # print(image_array)
            display_str_list_list = [[str(j) if id2name is None else id2name(j)] for j in b[:, -1]]
            draw_bounding_boxes_on_image_array(image_array, b[:,:-1], display_str_list_list=display_str_list_list, box_mode="xyxy")
            plot_many(image_array)
            cnt -= 1
            if cnt == 0:
                return


def view_anno_view_level(txt_path, id2name=None, respath2abspath=None):
    from convert_util.label_convert import deserialize_txt
    instances = deserialize_txt(txt_path)
    cnt = 32
    for i in instances:
        path = i[0]
        if respath2abspath is not None:
            path = respath2abspath(path)
        # if "AIMS-526_19096.0_34515.0.jpg" not in path:
        #     continue
        instance = i[1]
        if len(instance) > 0:
            print(path)
            b = np.array(instance)
            if (b[:, -1] == -1).sum() == 0:
                continue
            
            image_array  = imageio.imread(path)

            display_str_list_list = [[str(j) if id2name is None else id2name(j)] for j in b[:, -1]]
            draw_bounding_boxes_on_image_array(image_array, b[:,:-1], display_str_list_list=display_str_list_list, box_mode="xyxy")
            plot_many(image_array)
            cnt -= 1
            if cnt == 0:
                return

def view_image_anno(json_path, id2name=lambda x: str(x), font_path=None):
    path, w, h, boxes = label_convert.deserialize_output_json(json_path)
    if os.path.exists(path):
        image_array = imageio.imread(path)
    bboxes = np.array(boxes)
    bboxes = bboxes[bboxes[:, 1] > 0.4]
    old = bboxes[bboxes[:, 0] < 100]
    new = bboxes[bboxes[:, 0] >= 100]
    new[:, 2:] *= 3.333
    bboxes = np.concatenate([old, new], axis=0)
    print(old.shape[0])
    print(bboxes[:, 1].max(), w, h)
    display_str_list_list = []
    for c_id, score in bboxes[:, :2]:
        display_str_list_list.append(id2name(c_id), str(score))
    
    draw_bounding_boxes_on_image_array(image_array, bboxes[:,2:], display_str_list_list=display_str_list_list, box_mode="xyxy", font_path=font_path)
    plot_many(image_array)


    
        
            
