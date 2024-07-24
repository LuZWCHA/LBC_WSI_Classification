import os
import bisect
from abc import ABCMeta, abstractmethod
from pathlib import Path
import time

import cv2
import numpy as np
import openslide

from slide_detect_tools.slide_crop_patch import AutoReader
try:
    from SdpcDecoder import SdpcDecoder
except:
    SdpcDecoder = None
    print('SdpcDecoder module is not install')

    
from libiblsdk.ibl_py_sdk import IblWsi

def getFgROI(colorimg):
    rgb_max = np.max(colorimg, axis=2)
    rgb_min = np.min(colorimg, axis=2)
    rgb_diff = cv2.GaussianBlur(rgb_max - rgb_min, (5, 5), 0)
    thresh_bin = rgb_diff.max() / 5
    mask_bin = np.where(rgb_diff > 10, 1, 0)
    return mask_bin

# crop at level 0
class BaseReader(metaclass=ABCMeta):
    """Slide Reader base"""

    def __init__(self, slide_file, params):
        self.slide_file = slide_file
        self.params = params
        self.slide_id = Path(self.slide_file).stem

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def crop_patch(self, x, y):
        pass

    @property
    def ratio(self):
        return self._ratio
    
    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    @property
    def slide_pixel_size(self):
        return self._slide_pixel_size
    
    def get_crop_region(self):
        crop_size_h = self.params["crop_size_h"]
        crop_size_w = self.params["crop_size_w"]
        crop_overlap = self.params["crop_overlap"]

        x_min, x_max = 0, self.width
        y_min, y_max = 0, self.height

        crop_size_w_ = int(crop_size_w * self.ratio)
        crop_size_h_ = int(crop_size_h * self.ratio)
        crop_overlap_ = int(crop_overlap * self.ratio)

        crop_step_x = (crop_size_w_ - crop_overlap_)
        crop_step_y = (crop_size_h_ - crop_overlap_)
            
        xs = np.arange(x_min, x_max - crop_size_w_, crop_step_x)
        ys = np.arange(y_min, y_max - crop_size_h_, crop_step_y)
        
        # crop patch uses these property
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.crop_size_w_ = crop_size_w_
        self.crop_size_h_ = crop_size_h_

        region_x, region_y = np.meshgrid(xs, ys)
        region = np.stack([region_x.reshape(-1), region_y.reshape(-1)], 1)
        return region

class SdpcReader(BaseReader):
    """Sdpc slide reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)

        self.slide = SdpcDecoder(self.slide_file)

        _, self._slide_pixel_size = self.slide.get_pixel_size()
        self._width, self._height = self.slide.get_dimensions()
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        return

    def crop_patch(self, x, y):
        img = self.slide.read_region((x, y), 0, (self.crop_size_w_,
                    self.crop_size_h_))

        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))
        outputs = []
        for sl, shift in zip(self.slices, self.shifts):
            sub_img = img[sl[0]:sl[2], sl[1]:sl[3]]
            ## hard code to remove background image
            if np.mean(getFgROI(sub_img)) < 0.01:
                outputs.append((None, None))
            else:
                patch_id = "{}_{}_{}.png".format(self.slide_id, str(x+shift[1]), str(y+shift[0]))
                outputs.append((patch_id, sub_img))
        return outputs

    def get_crop_region(self):
        crop_size_h = self.params["crop_size_h"]
        crop_size_w = self.params["crop_size_w"]
        crop_overlap = self.params["crop_overlap"]

        x_min, x_max = 0, self.width
        y_min, y_max = 0, self.height

        crop_size_w_ = int(crop_size_w * self.ratio)
        crop_size_h_ = int(crop_size_h * self.ratio)
        crop_overlap_ = int(crop_overlap * self.ratio)

        crop_step_x = (crop_size_w_ - crop_overlap_)
        crop_step_y = (crop_size_h_ - crop_overlap_)

        crop_size_w_ += crop_step_x
        crop_size_h_ += crop_step_y
        crop_size_w += crop_size_w - crop_overlap
        crop_size_h += crop_size_h - crop_overlap

        xs = np.arange(x_min, x_max - crop_size_w_, 2*crop_step_x)
        ys = np.arange(y_min, y_max - crop_size_h_, 2*crop_step_y)

        # crop path uses these property
        self.crop_size_w = crop_size_w
        self.crop_size_h = crop_size_h
        self.crop_size_w_ = crop_size_w_
        self.crop_size_h_ = crop_size_h_
        self.slices = [
            (0, 0, self.params["crop_size_h"], self.params["crop_size_w"]),
            (0, -self.params["crop_size_w"], self.params["crop_size_h"], self.crop_size_w),
            (-self.params["crop_size_h"], 0, self.crop_size_h, self.params["crop_size_w"]),
            (-self.params["crop_size_h"], -self.params["crop_size_w"], self.crop_size_h, self.crop_size_w)
        ]
        # y s
        self.shifts = [
            (0, 0),
            (0, crop_step_x),
            (crop_step_y, 0),
            (crop_step_y, crop_step_x)
        ]

        region_x, region_y = np.meshgrid(xs, ys)
        region = np.stack([region_x.reshape(-1), region_y.reshape(-1)], 1)
        return region


class SdpcReaderO(BaseReader):
    """Sdpc slide reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)

        self.slide = SdpcDecoder(self.slide_file)

        _, self._slide_pixel_size = self.slide.get_pixel_size()
        self._width, self._height = self.slide.get_dimensions()
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        self.slide.close()

    def crop_patch(self, x, y):
        img = self.slide.read_region((x, y), 0, 
        (self.crop_size_w_, self.crop_size_h_))
        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))

        if np.mean(getFgROI(img)) < 0.01:
            return [(None, None)]
        
        patch_id = "{}_{}_{}.png".format(self.slide_id, str(x), str(y))
        return [(patch_id, img),]

from slide_detect_tools.sdpc_encode_fixed import SdpcDecoder as FixedDecoder
class SdpcReaderX(BaseReader):
    """Sdpc slide reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)

        self.slide = FixedDecoder(self.slide_file)

        self._slide_pixel_size = self.slide.get_pixel_size()
        self._width, self._height = self.slide.get_level_dimensions()[0]
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        self.slide.close()

    def crop_patch(self, x, y):
        img = self.slide.read_region_bgr((x, y), 0, 
        (self.crop_size_w_, self.crop_size_h_))
        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))

        if np.mean(getFgROI(img)) < 0.01:
            return [(None, None)]
        
        patch_id = "{}_{}_{}.png".format(self.slide_id, str(x), str(y))
        return [(patch_id, img),]

class OpenReader(BaseReader):
    """Openslide Reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)
        
        self.slide_id = Path(self.slide_file).stem
        self.slide = openslide.OpenSlide(self.slide_file)
        self._slide_pixel_size, _ = self.get_slide_pixel_size(self.slide)
        
        self._width, self._height = self.slide.dimensions
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        self.slide.close()

    def crop_patch(self, x, y):
        img = self.slide.read_region(level=0, location=(x, y),
                            size=(self.crop_size_w_, self.crop_size_h_))
        img = np.array(img, dtype=np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                
        ## hard code to remove background image
        if np.mean(getFgROI(img)) < 0.01:
            return [(None, None)]
        
        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))
        patch_id = "{}_{}_{}.png".format(self.slide_id, str(x), str(y))
        return [(patch_id, img),]

    # def get_slide_pixel_size(self, slide):
    #     """Get the pixel size of the slide"""
    #     try:
    #         pixel_size_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    #         pixel_size_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
    #     except Exception as e:
    #         pixel_size_x = 0.26
    #         pixel_size_y = 0.26

    #     return pixel_size_x, pixel_size_y
    def get_slide_pixel_size(self, slide):
        """Get the pixel size (µm) of the slide"""
        try:
            pixel_size_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            pixel_size_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
                
        except Exception as e:
            
            try: # for sdpc->tiff
                pixel_size_x = 10000 / float(slide.properties["tiff.XResolution"])
                pixel_size_y = 10000 / float(slide.properties["tiff.YResolution"])
                
            except Exception as e: # default value
                
                try: # for ibl
                    pixel_size_x = float(slide.fPixelSize.value)*1000
                    pixel_size_y = float(slide.fPixelSize.value)*1000
                    
                except Exception as e:
                    pixel_size_x = 0.26
                    pixel_size_y = 0.26
    
        return pixel_size_x, pixel_size_y

class IblReader(BaseReader):
    """Iblslide Reader"""

    def __init__(self, slide_file, params):
        super().__init__(slide_file, params)
        ret = 0
        retry_time = 5

        # while ret < 0 and retry_time > 0:
        self.slide = IblWsi(bytes(self.slide_file, encoding = "utf-8"))
        ret = self.slide.ret
            # if ret < 0:
            #     retry_time -= 1
            #     self.slide.CloseIBL()
        if ret != 0:
            raise RuntimeError(f"Ibl Open Error: {slide_file} can open failed.")
        self._slide_pixel_size, _ = self.get_slide_pixel_size()

        self._width, self._height = int(self.slide.width.value), int(self.slide.height.value)
        self._ratio = self.params["crop_pixel_size"] / self._slide_pixel_size
        
    def close(self):
        return self.slide.CloseIBL()
    
    @staticmethod
    def close_file(file_path):
        IblWsi.Close(bytes(file_path, encoding = "utf-8"))

    def crop_patch(self, x, y):
        patch = self.slide.GetRoiData(float(self.slide.scanScale.value), int(x), int(y), self.crop_size_w_, self.crop_size_h_)
        decode_img = cv2.imdecode(np.frombuffer(patch, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decode_img is None:
            print("Decode failed", float(self.slide.scanScale.value), int(x), int(y), self.crop_size_w_, self.crop_size_h_)
        img = decode_img.reshape((int(self.crop_size_h_), int(self.crop_size_w_), 3))
            
        ## hard code to remove background image
        if np.mean(getFgROI(img)) < 0.01:
            return [(None, None)]
        
        img = cv2.resize(img, (self.crop_size_w, self.crop_size_h))
        patch_id = "{}_{}_{}.png".format(self.slide_id, str(x), str(y))
        return [(patch_id, img),]

    def get_slide_pixel_size(self):
        """Get the pixel size of the slide"""
        
        try:
            pixel_size_x = float(self.slide.fPixelSize.value)*1000
            pixel_size_y = pixel_size_x
        except Exception as e:            
            pixel_size_x = 0.26
            pixel_size_y = 0.26

        return pixel_size_x, pixel_size_y


if __name__ == "__main__":
    params = {
        "crop_pixel_size": 0.31,
        "crop_size_h": 1280,
        "crop_size_w": 1280,
        "crop_overlap": 64,
    }

    import pandas as pd
    slide_file = ""
    slide_files = pd.read_csv("")["slide_path"].tolist()
    svs_slide_file = ""
    # reader = OpenReader(svs_slide_file, params)
    #reader = SdpcReader(slide_file, params)
    for i , slide_file in enumerate(slide_files):
        print(i)
        # reader1 = IblReader(slide_file, params)
        # reader2 = IblReader(slide_file, params)
        # # reader = IblReader(slide_file, params)
        ## region = reader1.get_crop_region()
        ## region = reader2.get_crop_region()
        
        # reader1.crop_patch(0, 0)
        # reader2.crop_patch(0, 0)
        # reader2.close()
        # reader1.crop_patch(0, 0)
        
        # with AutoReader(slide_file) as crop_reader:
        #     crop_reader.crop_patch(0,0,100,100,0.31,0)
        
        for j in range(16):
            IblReader(slide_file, params)
        
        IblReader(slide_file, params).close()

        # reader.crop_patch(0, 0)
        # reader.close()
    
    # for x, y in region:
    #     imgs = reader.crop_patch(x, y)
    #     if imgs[0][1] is not None:
    #         print(imgs[0][1].shape)
    #         cv2.imwrite("test.png", imgs[0][1])
    #     else:
    #         print(x,y)
    # reader.close()


