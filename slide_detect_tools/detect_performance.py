from copy import deepcopy
import glob
import math
import os
import time
import cv2, numpy as np
import imageio

plot_many_exists = False
try:
    from tools.visual.plot_utils import plot_many
    plot_many_exists = True
except Exception as e:
    pass
from slide_detect_tools import compatible

def find_circle(img, polt=False, r_scale=1):
    # cv2.imwrite(".data/thumb_cirs/" + " test"+ ".png", img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # img[img < 80] = 230
    h, w = img.shape[:2]

    short_edge = min(w, h)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=8, minDist=24,minRadius=max(short_edge // 8, 24), maxRadius=short_edge // 2)
    
    if circles is not None:
        circles[0] = list(circles[0])
        offset = 0
        # for a in circles[0]:
        #     if a[2] >= short_edge / 4:
        #             a[2] = 0
                    # offset += 1
        cir_img = np.copy(img)
        cir_mask = np.zeros_like(img)
        
        center_x = 0
        center_y = 0
        
        topk_cirs = circles[0][:64 + offset]
        
        k = len(topk_cirs)
        for i in topk_cirs:
            center_x += i[0] / k
            center_y += i[1] / k
            cv2.circle(cir_mask, (int(i[0]), int(i[1])), int( i[2]), (255,255, 255), -1)
        
        cir_mask = cv2.Canny(cir_mask, 24, 90)
        idxs = np.argwhere(cir_mask > 0)
        
        center, radius = cv2.minEnclosingCircle(idxs)
        center = (int(center[1]), int(center[0]))
        radius = int(radius * r_scale)
        
        # too small radius
        if radius < short_edge / 2 / 4:
            return None
        
        cv2.circle(cir_img, center, radius, (0, 0, 0), 10)

        if polt and plot_many_exists:
            plot_many([img, cir_img, cir_mask] ,rows=1, show=True).savefig(".data/thumb_cirs/" + str(time.time_ns()) + ".png")
        return (center, radius)
    else:
        if polt:
            plot_many(img)
        return None

def gen_thumbnail(slide_path, scale=64):
    slide = compatible.SildeFactory().of(slide_path)
    w, h = slide.dimensions
    
    thmn_w = w // scale
    thmn_h = h // scale
    
    thmn_w = thmn_h = max(thmn_h, thmn_w)
    
    assert thmn_w > 0 and thmn_h > 0
    
    thum = slide.get_thumbnail((thmn_w, thmn_h))
    thum_array = np.array(thum)
    thum.close()
    slide.close()
    return thum_array[..., :3]


def get_cir(slide_path, scale=64, plot=False, r_scale=1):
    thmn = gen_thumbnail(slide_path, scale)
    return find_circle(thmn, plot, r_scale=r_scale)

def check_inside(index, patch_w, patch_h, cir_res, rtol=1):
    if cir_res is None:
        return True
    (cx, cy), r = cir_res
    return check(index[0], index[1], index[0] + patch_w, index[1] + patch_h, cx, cy, r * rtol)

def get_inside_patchs_mask(slide_path, scale=64, polt=False):

    thmn = gen_thumbnail(slide_path, scale)
    res = find_circle(thmn, polt=polt)
    cir_mask = np.zeros_like(thmn)

    if res is not None:
        center, r = res
        cv2.circle(cir_mask, (int(center[0]), int(center[1])), int(r), (255,255, 255), -1)
        cir_mask = cir_mask[..., 0]
    
    return cir_mask

def check(x1,y1, x2, y2, cx, cy, r):
    minx = min(abs(x1 - cx), abs(x2 - cx))
    miny = min(abs(y1 - cy), abs(y2 - cy))
    if (minx * minx + miny * miny) < r * r: 
        return True

    x0 = (x1+x2) / 2
    y0 = (y1+y2) / 2
    if (abs(x0 - cx) < abs(x2 - x1) / 2 + r) and abs(cy - y0) < abs(y2 - y1) / 2: return True
    if (abs(y0 - cy) < abs(y2 - y1) / 2 + r) and abs(cx - x0) < abs(x2 - x1) / 2: return True
 
    return False

def draw_regions(slide_path, scale, regions):
    thmn = gen_thumbnail(slide_path, scale)
    masked_thmn = deepcopy(thmn)
    mask = np.zeros_like(thmn)
    for x, y, w, h in regions:
        x /= scale
        y /= scale
        w /= scale
        h /= scale
        cv2.rectangle(mask, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), thickness=1)
        cv2.rectangle(masked_thmn, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), thickness=1)
        
    file_name = os.path.basename(slide_path)

    if plot_many_exists:
        plot_many([thmn, mask, masked_thmn], rows=1, show=False).savefig(f".data/regions_show/{file_name}.png")
        

if __name__ == "__main__":

    # for i in glob.iglob("/nasdata/private/zwlu/Now/ai_trainer/.data/thumbnail/*"):
    #     img = cv2.imread(i)
    #     find_circle(img)
    # for i in glob.iglob("/nasdata/private/zwlu/Now/ai_trainer/.data/slides/outline/*"):
    #     print(get_cir(i, plot=True, r_scale=0.9))
    # draw_regions()
    
    # img = imageio.v3.imread("/nasdata/private/zwlu/Now/ai_trainer/.data/20230608-173557.png")
   
    # print(img.shape)
    # find_circle(img, polt=True)
    
    # get_cir("/nasdata/private/zwlu/Now/ai_trainer/.data/slides/﻿052901-042-20230529-235739.ibl.tiff", plot=True)
    get_cir("/nasdata/private/zwlu/Now/ai_trainer/.data/slides/0530-027-20230530-234239.ibl.tiff", plot=True)
    
# end main