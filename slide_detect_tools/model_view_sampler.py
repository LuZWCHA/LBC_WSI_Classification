import os
import time
from convert_util.label_convert import deserialize_output_json
from slide_detect_tools.compatible import *



class BaseSampler:
    
    def __init__(self, boxes, heatmap_cell_size=(1280 * 0.1, 1280 * 0.1), intreseted_cid=[0, 1, 2, 4]) -> None:
        self.sample_records = None
        self.boxes_info = boxes
        self.heat_map_cell_size = heatmap_cell_size
        self.heatmap = None
        self.sample_mask = None
        self.ori2heatmap_ratio = -1
        self.intreseted_cid = intreseted_cid
    
    def transform_coords(self, coords):
        coords[:, 0] *= self.ori2heatmap_ratio[0]
        coords[:, 1] *= self.ori2heatmap_ratio[1]
        return coords
    
    def transform_coords_back(self, coords):
        coords[:, 0] = coords[:, 0] / self.ori2heatmap_ratio[0]
        coords[:, 1] = coords[:, 1] / self.ori2heatmap_ratio[1]
        return coords
    
    def create_heatmap(self, class_ids=[0, 1, 2, 4]):
        path, w, h, boxes  = self.boxes_info
        # print(w, h)
        id2index = {id_: idx for idx, id_ in enumerate(class_ids)}
        
        if not os.path.exists(path):
            raise RuntimeError(f"{path} not found.")
        
        slide = SildeFactory.of(path)
        pixel_size_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        pixel_size_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        slide.close()
        
        r_x = int(pixel_size_x * w / self.heat_map_cell_size[0])
        r_y = int(pixel_size_y * h / self.heat_map_cell_size[1])
        
        class_num = len(class_ids)
        
        self.heatmap = np.zeros((r_y, r_x, class_num))
        self.sample_mask = np.ones((r_y, r_x, class_num))
        self.ori2heatmap_ratio = [pixel_size_x / self.heat_map_cell_size[0], pixel_size_y / self.heat_map_cell_size[1]]
        
        # for b in boxes:
        #     c_id, score, xmin, ymin, xmax, ymax = b
        boxes = np.array(boxes)
        boxes_ids = boxes[:, 0]
        boxes_scores = boxes[:, 1]
        boxes_centers = np.stack([boxes[:, 2] / 2 + boxes[:, 4] / 2, boxes[:, 3] / 2 + boxes[:, 5] / 2], axis=1)
        # transform to heatmap coords
        boxes_centers = self.transform_coords(boxes_centers)
        
        x_valid_indices = (boxes_centers[:, 0] < r_x)
        y_valid_indices = (boxes_centers[:, 1] < r_y)
        id_valide_indoces = np.zeros_like(y_valid_indices)
        for i in class_ids:
            id_valide_indoces |= (boxes_ids == i)
        xy_valid_indices = x_valid_indices & y_valid_indices & id_valide_indoces
        # xy_valid_indices
        boxes_ids = boxes_ids[xy_valid_indices]
        boxes_scores = boxes_scores[xy_valid_indices]
        boxes_centers = boxes_centers[xy_valid_indices]
        boxes_centers = boxes_centers.astype(np.int32)
        for c, s, i in zip(boxes_centers, boxes_scores, boxes_ids):
            idx = id2index[i]
            self.heatmap[c[1], c[0], idx] = max(s, self.heatmap[c[1], c[0], idx])
        

    def _sample(self, x, y, channel_num=0):
        h, w = self.heatmap.shape[:2]
        if y >= h or x >= w:
            return None
        self.sample_records.append((x, y, channel_num))
        self.sample_mask[y, x, channel_num] = 0
        return self.heatmap[y, x, channel_num], x, y
    
    def show_heatmap(self):
        
        assert self.heatmap is not None
        
        import numpy as np
        import seaborn as sns
        import matplotlib.pylab as plt
        cnt = self.heatmap.shape[-1]
        max_value = self.heatmap.max()
        for i in range(cnt):
            uniform_data = self.heatmap[..., i]
            plt.subplot(2, 2, i + 1)
            ax = sns.heatmap(uniform_data, linewidth=0.0, vmin=0, vmax=max_value)
        plt.show()
        
        pass
    
    def sample(self, sample_size=[640, 384], pixel_size=[0.31*2, 0.31*2]):
        # image grids
        if self.heatmap is None:
            self.create_heatmap(self.intreseted_cid)
        ps = [i * j for i, j in zip(sample_size, pixel_size)]
        if self.heat_map_cell_size != ps:
            self.heat_map_cell_size = ps
            self.create_heatmap(self.intreseted_cid)
        # TODO
        pass
    


class TopKSampler(BaseSampler):
    """generate the heatmap of the scores by different classes.
            Take TOPK scores' regions for every classes.
    """
    def __init__(self, boxes, heatmap_cell_size=(1280 * 0.1, 1280 * 0.1), k=8, interseted_cid=[0, 1, 2, 4]) -> None:
        super().__init__(boxes, heatmap_cell_size, interseted_cid)
        self.k = k
        self.cache = None
    
    def sample(self, sample_size=[640, 384], pixel_size=[0.31*2, 0.31*2]):
        super().sample(sample_size, pixel_size)
        class_num = self.heatmap.shape[-1]
        # print(self.heatmap.shape, self.ori2heatmap_ratio)
        indices = []
        for i in range(class_num):
            indices_sorted = np.argsort(-self.heatmap[..., i], axis=None)            
            indices.append(indices_sorted)
        
        topk_dict = dict()
        topk_mask = np.zeros_like(self.heatmap)
        for idx, sorted_idx in enumerate(indices):
            y, x = np.unravel_index(sorted_idx[:self.k], self.heatmap.shape[:2])
            topk_mask[y,x,idx] = 1
            i = self.heatmap[y,x,idx]
            topk_dict[idx] =(self.transform_coords_back(np.stack([x, y], 1)).astype(np.int32), i)
        self.sample_records = (topk_dict, )
        # self.heatmap *= topk_mask
        
        return topk_dict, topk_mask
        
if __name__ == "__main__":
    data = deserialize_output_json("")
    
    data[0] = ""
    s = TopKSampler(data, (640 * 0.62, 384 * 0.62))
    start = time.time()
    s.create_heatmap()
    print(f"Cost time: {(time.time() - start) * 1e3} ms")
    s.show_heatmap()
    topk_dict, topk_mask = s.sample()
    s.show_heatmap()
    from tools.visual.plot_utils import plot_many
    # plot_many([1 - s.heatmap, 1- topk_mask], rows=1)
    from slide_detect_tools.slide_crop_patch import AutoReader
    slide = AutoReader(data[0])
    # for k, v in topk_dict.items():
    #     print(k)
    #     for x, y in v:
    #         print(x, y, slide.width, slide.height)
    #         patch = slide.crop_patch(x, y, 640, 384, 0.62)
    #         plot_many(patch[..., ::-1])
    for v in topk_dict.values():
        # print(k)
        for x, y in v:
            print(x, y, slide.width, slide.height)
            patch = slide.crop_patch(x, y, 640, 384, 0.62)
            plot_many(patch[..., ::-1])
# end main
