import os
from datasets.our_datasets import WSI_PATCH, SHISHI_RAW

def check_and_create(path):
    parent_path = os.path.dirname(path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    

# Yolov5 image 2 label path function
# from path/to/images/img.xxx -> path/to/labels/img.txt
# for example:
# img1.png and its label: img1.txt
# They will under the 'path/to/' root path
# 
# path/to/images/img1.png
# path/to/labels/img1.txt
#
def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]
    

def create_labels(dataset_root,pic_size = (3840, 2160)):
    train_txt, test_txt = os.path.join(dataset_root, "annotations_train_new.txt"), os.path.join(dataset_root, "annotations_test_new.txt")
    
    with open(train_txt, "r") as i, open(test_txt, "r") as i2:
        all_lines = i.readlines() + i2.readlines()

    for items in all_lines:
        data = items.split(" ")
        pic_path = data[0].strip()
        label_path = img2label_paths([pic_path])[0]
        if not os.path.exists(pic_path):
            print(pic_path)
            continue
        instances = []
        if len(data) > 1:
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
                    
                center_x = (xmin + xmax) / pic_size[0] / 2
                center_y = (ymin + ymax) / pic_size[1] / 2
                w = (xmax - xmin) / pic_size[0] 
                h = (ymax - ymin) / pic_size[1] 
                
                str_list = [str(j) for j in (label, center_x, center_y, abs(w), abs(h))]
                
                instances.append(" ".join(str_list).strip() + "\n")
                
        check_and_create(label_path)        
        
        print(label_path, instances)
        
        with open(label_path, "w") as f:
            f.writelines(instances)
        
            # for it in data[1:]:
            #     x_min, y_min, x_max, y_max, class_id = [float(j) for j in it.split(",")]
            #     class_id = int(class_id)

def create_train_sets(dataset_root, train_set_txt, val_set_txt):
    train_txt, test_txt = os.path.join(dataset_root, "annotations_train_new.txt"), os.path.join(dataset_root, "annotations_test_new.txt")
    
    with open(train_txt, "r") as  train:
        all_lines = train.readlines()
    
    image_paths = []
    for items in all_lines:
        data = items.split(" ")
        pic_path = data[0].strip()
        
        image_paths.append(pic_path + "\n")
        
    check_and_create(train_set_txt)        
        
    with open(train_set_txt, "w") as f:
        f.writelines(image_paths)
    
    with  open(test_txt, "r") as val:
        all_lines = val.readlines()
    
    image_paths = []
    for items in all_lines:
        data = items.split(" ")
        pic_path = data[0].strip()
        if not os.path.exists(pic_path):
            print(pic_path)
            continue
        image_paths.append(pic_path + "\n")
        
    check_and_create(train_set_txt)        
        
    with open(val_set_txt, "w") as f:
        f.writelines(image_paths)


if __name__ == "__main__":
    # create_labels(WSI_PATCH.root_path)
    # create_labels(SHISHI_RAW.root_path)
    # create_train_sets(WSI_PATCH.root_path, WSI_PATCH.at("train_yolo.txt"), WSI_PATCH.at("test_yolo.txt"))
    create_train_sets(SHISHI_RAW.root_path, SHISHI_RAW.at("train_yolo.txt"), SHISHI_RAW.at("test_yolo.txt"))
# end main
