import glob
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch, matplotlib
from bertviz import head_view, model_view
from convert_util.class_map import CLASS_TEN_NAMES, CLASS_TWELVE_NAMES

all_attns = ""
inputs =  ""

all_attns = glob.glob(all_attns + "/*.pth", recursive=False)
all_attns.sort()
# print(all_attns)

with open(inputs, mode="r") as f:
    lines = f.readlines()

i = 0
cls_token_activate = 0
fc = "LSIL"
for line, attn in zip(lines, all_attns):
    path, slide_cls = line.strip().split(" ")
    if slide_cls != fc:
        continue
    stem = Path(path).stem
    npz_data = np.load(path)
    cls_id_score_list = npz_data["infos"]
    # print(cls_id_score_list)
    npz_data.close()
    
    tokens = [f"CLS_{slide_cls}"]
    
    for cid, s in cls_id_score_list:
        # cname = "NILM"
        # if cid >= 0 and s > 0.2:
        cname = CLASS_TWELVE_NAMES[int(cid)]
                
        dis_str = f"{cname}_{s:.2f}"
        tokens.append(dis_str)
        
    attn = torch.load(attn).cpu()[None, ]

    html_str = head_view(attn, tokens=tokens, html_action="return")
    with open(f"head_view_{fc}_{i:03}.html", "w") as f:
        f.write(html_str.data)
    head_view(attn, tokens=tokens)
    cls_token_activate += attn[0, 0, :, 0, 1:]

    i += 1
    if i > 8:
        break
cls_token_activate # (8, 24)
import torch.nn as nn

from tools.visual.plot_utils import plot_many
plot_many(cls_token_activate.numpy().mean(0) / 24, default_type="bar")




