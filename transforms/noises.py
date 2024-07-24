# this file contains the adventive

from collections import OrderedDict
import glob
import itertools
import math
import random
import time
from typing import Any
import cv2
import imageio
import numpy as np
from scipy import signal


def get_noise_distribution():
    bg_ = "/nasdata/dataset/wsi_patch_data/dt6/images/FQxrU1611580494840.jpg"
    bg_ = cv2.imread(bg_)
    bg_ = cv2.cvtColor(bg_, cv2.COLOR_BGR2GRAY)
    # bg_ = bg_ @ np.array([0.2126, 0.7152, 0.0722])

    bg_ = bg_[:500, :][..., None]
    # bg_[bg_ > 0.4] = 1
    # print(bg_.shape)
    # plt.imshow(bg_.astype(np.uint8), cmap="gray")
    pixels = bg_.astype(np.uint8).flatten()
    # plot_many(pixels, default_type="hist", hist_params={"bins": 60, "range": (0, 255)})

    t, bg_ = cv2.threshold(bg_,220, 255, cv2.THRESH_BINARY)
    # plt.imshow(bg_.astype(np.uint8), cmap="gray")
    # plt.show()
    # print(t, np.unique(bg_))

    from skimage.measure import label
    bg_ = 255 - bg_
    bg_, num = label(bg_, connectivity=2, return_num=True)
    # print(bg_.max(), num)
    # print(num / bg_.size)

    # plt.imshow(bg_.astype(np.uint8))
    # plt.show()
    w_list, h_list = [], dict()
    for i in range(num + 1):
        item = (bg_ == i).astype(np.uint8)
        ct = cv2.findContours(item, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        # print(ct.shape)
        x,y,w,h = cv2.boundingRect(ct[0])
        w_list.append(min(w, h))
        if int(min(w, h)) not in h_list:
            h_list[int(min(w, h))] = []
        h_list[int(min(w, h))].append(max(w, h) / min(w, h)) 
    
    display_h = [v for k, v in h_list.items()]
    # plot_many([np.array(w_list)],default_type="hist", show=True)
    # plot_many([np.array(display_h)],default_type="hist", show=True)
    # # plt.hist(w_list,)
    # plt.show()
    # indices = np.where(pixels < 220, pixels, 0)
    return w_list, h_list, pixels[pixels < 225], num / bg_.size

def sample_shape(w_list, extent_list, ins):
    w = random.sample(w_list, 1)[0]
    
    while w not in extent_list:
        w = random.sample(w_list, 1)[0]
        
    h_w = random.sample(extent_list[w], 1)[0]
    h = w * h_w
    
    
    i = random.sample(ins.tolist(), 1)[0]
    return w, h,  int(i * 0.8)

def random_num(p):
    return random.gauss(0.6, 0.2) * p
    

def sample_data(img, w_list, extent_list, ins, p):
    
    num = int(random_num(p) * img.size)
    
    for i in range(num):
        w, h, s = sample_shape(w_list, extent_list, ins)
        if w*h > 512:
            continue
        x, y = random.randint(0, img.shape[0] - 1), random.randint(0, img.shape[1] - 1)

        gen_strip(img, (int(y), int(x)), (int(w * 0.6), int(h * 2)), color=(s, s, s))
    
    # img = elastic_transform_random_one_channel([img], interpolations=["linear"], kernel_size=3, scale=8)[0][0]
    img = elastic(img)
    img = cv2.GaussianBlur(img, ksize=(3, 3),  sigmaX=1, sigmaY=1)
    return img
    
def gen_strip(img, position, shape, color):
    
    cv2.ellipse(img, position, shape, random.uniform(0, 360), 0, 360, color, -1) #画椭圆
    
    pass

def elastic(img):
    sigma = 20
    alpha = 10

    # 生成随机位移场
    random_state = np.random.RandomState(None)
    displacement = np.float32(
        random_state.randn(img.shape[0], img.shape[1], 2) * sigma
    )
    displacement = cv2.GaussianBlur(displacement, (0, 0), alpha)

    # 计算目标坐标
    map_x, map_y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    map_x = np.float32(map_x + displacement[..., 0])
    map_y = np.float32(map_y + displacement[..., 1])

    # 进行弹性形变
    img_distorted = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return img_distorted

def gauss_kernel(kernel_size, channels, sigma=(1, 1, 1)):
    kernel: np.ndarray = 1
    meshgrids = np.meshgrid(
        *[
            np.arange(size, dtype=np.float32)
            for size in kernel_size
        ]
    )

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * np.exp(-((mgrid - mean) / std) ** 2 / 2)

    kernel /= np.sum(kernel)

    kernel = kernel.reshape(*kernel.shape)
    # kernel = kernel.repeat(channels)
    return kernel

def _grid2(shape):
    ndims = len(shape)
    shape = np.array(shape, dtype='uint')
    axis_list = [[i for i in range(shape[dims])] for dims in range(ndims)]
    grid = np.meshgrid(*axis_list, indexing='ij')
    return grid

def _interpn(loc, org_size, vox, new_shape, interpolation):
    """
    Interpolation (by linear or nearest method) by numpy, the codes stucture is similar with voxelmorph.
    Some codes is obtained from voxelmorph, and fixed the CPU floating point precision error.
    :param loc: the relative position of the voxel's pixels
    :param org_size: the orginal size of voxel.
    :param vox: the reshaped voxel.
    :param new_shape: the voxel to be resized.
    :param interpolation: interpolation.
    :return: resized or distort voxel.
    """

    def prod_n(lst):
        """
        Alternative to tf.stacking and prod, since tf.stacking can be slow
        """
        prod = lst[0]
        for p in lst[1:]:
            prod *= p
        return prod

    def sub2ind(siz, subs, **kwargs):
        """
        assumes column-order major
        """
        # subs is a list
        assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

        k = np.cumprod(siz[::-1])

        ndx = subs[-1]
        for i, v in enumerate(subs[:-1][::-1]):
            ndx = ndx + v * k[i]

        return ndx

    interp_vol = 0
    loc = loc.astype(np.float32)
    # second order or first order
    if interpolation == 'linear':
        nb_dims = len(org_size)
        loc0 = np.floor(loc)

        # clip values
        max_loc = [d - 1 for d in list(org_size)]
        loc0lst = [np.clip(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        del loc0
        clipped_loc = [np.clip(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [np.clip(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        del max_loc
        locs = [[f.astype(np.int32) for f in loc0lst], [f.astype(np.int32) for f in loc1]]
        del loc0lst
        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        del loc1
        del clipped_loc
        diff_loc0 = [1 - d for d in diff_loc1]

        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.
        del diff_loc0
        del diff_loc1
        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        vox_reshaped = vox.view()
        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(org_size, subs)
            vol_val = np.take(vox_reshaped, idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0

            # fixed CPU float precision error by simply np.abs(weights_loc)
            wts_lst = [np.abs(weights_loc[c[d]][d]) for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            # wt = np.expand_dims(wt, -1)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val
        interp_vol: np.ndarray
        interp_vol = interp_vol.reshape(new_shape)

    elif interpolation == 'nearest':
        loc = np.round(loc).astype(np.int32)

        shape = loc.shape
        a = np.empty_like(loc)
        for i in range(shape[1]):
            np.clip(loc[..., i], 0, org_size[i] - 1, a[..., i])

        a = tuple(a.transpose())
        interp_vol = vox[a].reshape(new_shape)

    return interp_vol


def resample_nd_by_transform_field(np_voxel: np.ndarray, transformed_loc, interpolation):
    """
    Resample N-D data by linear or nearest resample method qucikly use the numpy array without python loop
    :param np_voxel: the data to resample
    :param transformed_loc: the pxiel-level displacement to warp the data.
    :param interpolation: the method of interpolation to use, now support nearest and linear.
    :return: resampled voxel/image/N-D data
    """
    assert np.prod(np_voxel.shape) == transformed_loc.shape[:-1]
    return _interpn(transformed_loc, np_voxel.shape, np_voxel, np_voxel.shape, interpolation)

def elastic_transform_random_one_channel(datas: list, interpolations=None, kernel_size=9, scale=1, sigma=1, filter=None):
    def random_displacement(shape_, a):
        grids = _grid2(shape_)
        f = filter if filter is not None else default_filter

        dsm = [f((np.random.rand(*list(shape_)) - 0.5) * 2 * a, kernel_size=kernel_size, sigma=sigma) for g in grids]
        grids_ = [g + d for g, d in zip(grids, dsm)]

        grid = np.array(grids_)
        dsm = np.array(dsm)
        shape = grid.shape
        location_list = grid.reshape(shape[0], -1).transpose()
        return location_list, dsm.reshape(shape[0], -1).transpose()

    def default_filter(voxel, sigma=3, kernel_size=9):
        dim_ = voxel.ndim
        kernel = gauss_kernel(kernel_size=(kernel_size,) * dim_, channels=1, sigma=(sigma,) * dim_)
        # kernel = np.ones((kernel_size,) * dim_) / kernel_size ** dim_
        res = signal.convolve(voxel, kernel, mode='same')
        return res

    def smooth_displacement(shape_, a):
        random_dsp = random_displacement(*list(shape_), a)
        smooth_dsp = []
        for dsp in random_dsp:
            smooth_dsp.append(filter(dsp))

        return smooth_dsp

    assert len(datas) > 0 and kernel_size > 0

    if interpolations is None:
        interpolations = ['linear'] + (['nearest'] * (len(datas) - 1))
    else:
        if isinstance(interpolations, list):
            assert len(interpolations) == len(datas)
        elif isinstance(interpolations, str) and interpolations in ['nearest', 'linear']:
            interpolations = [interpolations] * len(datas)

    results = []
    dsm = smdsp = None
    for idx, data in enumerate(datas):
        if dsm is None or smdsp is None:
            smdsp, dsm = random_displacement(data.shape, scale)

        data = resample_nd_by_transform_field(data, smdsp, interpolation=interpolations[idx])
        results.append(data)

    return tuple(results), dsm.reshape(*results[0].shape, results[0].ndim)

def bloom(img):
    # image_blur = cv2.GaussianBlur(img, ksize=(3, 3),  sigmaX=1, sigmaY=1)
    image_blur = cv2.GaussianBlur(img, ksize=(5, 5),  sigmaX=0.5, sigmaY=0.5)
    
    image_blur[img < 230] = 0
    image_blur[img == 255] = 0
    # image_blur /= 30
    return img + image_blur


def fusion(img, noise, scale):
    center = (img.shape[1] // 2, img.shape[0] // 2)
    noise = cv2.resize(noise, dsize=(int(img.shape[1]), int(img.shape[0])))
    noise = noise[:img.shape[0], :img.shape[1]]
    img = cv2.seamlessClone(noise.astype(np.uint8), img, noise.astype(np.uint8),center, flags=cv2.MIXED_CLONE)

    return img


class RandomNoise:
    
    def __init__(self, scale=1, noise_tempates_dir="/nasdata/private/zwlu/Now/ai_trainer/.data/noises/", cache_size=128, resize_mode="shortest_edge") -> None:
        self.path = noise_tempates_dir
        self.template = glob.glob(self.path + "*.jpg")
        self.rng = random.Random(0)
        self.noise_cache = OrderedDict()
        self.cache_size = cache_size
        self.scale = scale
        self.load_times = []
        self.fusion_times = []
        self.resize_mode = resize_mode
    
    def __call__(self, data, *args: Any, **kwds: Any) -> Any:
        t_id = self.rng.randint(0, len(self.template) - 1)
        if t_id in self.noise_cache:
            noise = self.noise_cache[t_id]
            
        else:
            t = self.template[t_id]
            start = time.time_ns()
            noise = imageio.imread(t)

            end = time.time_ns()
            self.load_times.append((end - start) / 1e6)
            # print(self.load_times[-1])
            if len(self.noise_cache) >= self.cache_size:
                self.noise_cache.popitem(True)
            self.noise_cache[t_id] = noise
        
        noise_shape = noise.shape
        
        if self.scale <= 0:
            if self.resize_mode == "shortest_edge":
                if data.shape[0] > data.shape[1]:
                    self.scale =data.shape[1] / noise_shape[1]
                else:
                    self.scale =data.shape[0] / noise_shape[0]
            else:
                if data.shape[1] > data.shape[0]:
                    self.scale =data.shape[1] / noise_shape[1]
                else:
                    self.scale =data.shape[0] / noise_shape[0]
                
        
        if noise_shape[0] * self.scale > data.shape[0] or noise_shape[1] * self.scale > data.shape[1]:
            # random crop
            H, W = data.shape[0], data.shape[1]
            x, y = self.rng.randint(0, noise_shape[1] - W), self.rng.randint(0, noise_shape[0] - H)
            x,y = max(0, x), max(0, y)
            noise = noise[y: int(data.shape[0] / self.scale) + y, x: int(data.shape[1] / self.scale) + x]
        start = time.time_ns()
        res = fusion(data, noise, self.scale)
        end = time.time_ns()
        self.fusion_times.append((end - start) / 1e6)
        return res
        
def test_random_noise():
    random_noise = RandomNoise(scale=1)
    img = "/nasdata/dataset/wsi_patch_data/dt6/images/Ru4Gq1611580727916.jpg"
    img = imageio.v3.imread(img)[:640, :640]
    start = time.time()
    for _ in range(1000):
        img = random_noise(img)
    print(f"cost: {(time.time() - start) / 1000 * 1000} ms")
    print(f"numpy load cost, sum:{sum(random_noise.load_times)}ms, avg: {sum(random_noise.load_times) / len(random_noise.load_times)}ms")
    print(f"fusion cost, sum:{sum(random_noise.fusion_times)}ms, avg: {sum(random_noise.fusion_times) / len(random_noise.fusion_times)}ms")

    
