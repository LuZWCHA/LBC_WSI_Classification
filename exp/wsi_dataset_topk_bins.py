import os
import random

import torch
import cv2
import numpy as np
import albumentations as albu

from detectron2.data.common import ToIterableDataset
from detectron2.data.samplers import TrainingSampler, InferenceSampler
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng

from tools.inference_help import onehot
# from convert_util.class_map import CLASS_SIX_NAMES

name_to_label_3class = {
    "ASCUS": 0, "LSIL": 0,
    "HSIL": 1,  "ASCH": 1,
    "AGC": 2, "SCC": 2
}

# TODO merge to new files
class WSIDataset(torch.utils.data.Dataset):
    """WSI dataset"""

    def cname2cid(self, name):
        # six calss name to three class id
        return name_to_label_3class[name] if name in name_to_label_3class else -1

    def __init__(self, data_file, size, max_num_patchs=24, train=False, albu_transform=None,
        weak_albu_transform=None):
        self.data_file = data_file
        self.train = train
        self.albu_transform = albu_transform
        self.size = size
        self.max_num_patchs = max_num_patchs
        self.weak_albu_transform = weak_albu_transform

        self.min_num_patchs = 24

        self.data = []
        for l in open(data_file).readlines():
            d = l.split()
            img_path = d[0]
            cid = self.cname2cid(d[1])
            target = onehot(cid, class_num=3) if cid >= 0 else [0, 0, 0]
            self.data.append((img_path, target))

        if self.train:
           random.shuffle(self.data) 

    def __getitem__(self, index):
        img_path, target = self.data[index]
        # read image
        # modified by nowandfuture, the data is stored at key of 'patchs' in a NPZ file but not NPY file
        # print(img_path)
        imgs_file = np.load(img_path, allow_pickle=True)
        imgs = imgs_file["patchs"]
        imgs_file.close()
        if self.train:
            num_patchs = random.randint(self.min_num_patchs, self.max_num_patchs)
        else:
            num_patchs = self.max_num_patchs
        images = []
        for img in imgs[:num_patchs]:
            if self.albu_transform is not None:
                img = self.albu_transform(image=img)["image"]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            images.append(img)
        
        target = torch.as_tensor(target, dtype=torch.float32)

        idx = os.path.basename(img_path).split('.')[0]

        if self.weak_albu_transform:
            weak_images = []
            for img in imgs[:self.max_num_patchs]:
                img = self.weak_albu_transform(image=img)["image"]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
                weak_images.append(img)
            return {'images': images, 'label': target, 'id': idx, 'weak_images': weak_images}
        else:
            return {'images': images, 'label': target, 'id': idx}

    def __len__(self):
        return len(self.data)



def trivial_batch_collator(batch):
    return batch

def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all_rng(initial_seed + worker_id)

def build_wsi_train_dataloader(dataset, total_batch_size, num_workers=8,
    sampler=None):

    if sampler is None:
        sampler = TrainingSampler(len(dataset))
    
    dataset = ToIterableDataset(dataset, sampler)
    
    world_size = get_world_size()
    batch_size = total_batch_size // world_size

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=worker_init_reset_seed,
    )    

def build_wsi_test_dataloader(dataset, num_workers=8,
    sampler=None):

    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    
    #batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
    )    
    
