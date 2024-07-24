from abc import ABC, abstractmethod
import os
import bisect
from pathlib import Path
import time
from types import TracebackType
from typing import Dict

import cv2 
import numpy as np
import openslide
from SdpcDecoder import SdpcDecoder
from .sdpc_encode_fixed import SdpcDecoder

import sys
from libiblsdk.ibl_py_sdk import IblWsi



def get_slide_pixel_size(slide):
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

class Closeable(object):

    def close(self):
        raise NotImplementedError()

class PathologySliceReader(Closeable):

    def __enter__(self):
        return self

    def read(self):
        raise NotImplementedError()

    def __init__(self, slide_file):
        self.slide_id = Path(slide_file).stem

    def __exit__(self, type, value, trace: TracebackType):
        
        try:
            exp = None
            self.close()
        except Exception as e:
            exp = e
        finally:
            if trace:
                if exp:
                    raise exp.with_traceback(trace)
                else:
                    raise value

        

class CropAtImage(ABC):

    @property
    def width(self):
        return self._width
    
    @property
    def height(self):
        return self._height

    @property
    def slide_pixel_size(self):
        return self._slide_pixel_size

    def crop_patch(self, x, y, width, height, crop_pixel_size, crop_level=None):
        raise NotImplementedError()

class AbstractPSReader(CropAtImage, PathologySliceReader):
    def read(self, pixel_size=None):
        if pixel_size is None:
            pixel_size = self.slide_pixel_size
        
        ratio = pixel_size / self.slide_pixel_size
        return self.crop_patch(0, 0, self.width * ratio, self.height * ratio, crop_pixel_size=pixel_size)


# for format svs and so on
class OpenSlideReader(AbstractPSReader):
    """Read WSI and Crop"""

    def __init__(self, slide_file):
        super().__init__(slide_file)
        self.slide_file = slide_file
        self.slide = openslide.OpenSlide(slide_file)
        slide_pixel_size, _ = get_slide_pixel_size(self.slide)
        self._slide_pixel_size = slide_pixel_size
        
        slide_width, slide_height = self.slide.dimensions
        self._width = slide_width
        self._height = slide_height

        self.slide_level_pixel_size = [self.slide_pixel_size *(int(n)) for n in self.slide.level_downsamples]
        
    def close(self):
        self.slide.close()

    def get_best_crop_level(self, crop_pixel_size):
        # find best crop level
        crop_level = bisect.bisect_left(self.slide_level_pixel_size, crop_pixel_size) - 1
        if crop_level < 0:
            crop_level = 0
        return crop_level, self.slide_level_pixel_size[crop_level]

    def crop_patch(self, x, y, width, height, crop_pixel_size, crop_level=None):
        """Crop patch"""
        if crop_level is None:
            crop_level, slide_pixel_size = self.get_best_crop_level(crop_pixel_size)
        else:
            slide_pixel_size = self.slide_level_pixel_size[crop_level]
        crop_width = int(width * crop_pixel_size / slide_pixel_size)
        crop_height = int(height * crop_pixel_size / slide_pixel_size)
        img = self.slide.read_region(level=crop_level, location=(x, y), size=(crop_width, crop_height))
        img = np.array(img, dtype=np.uint8)[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img


# for sdpc
class SdpcReader(AbstractPSReader):
    """Read WSI and Crop"""

    def __init__(self, slide_file):
        super().__init__(slide_file)
        self.slide_file = slide_file
        # self.slide = SdpcDecoder(slide_file)
        self.slide = SdpcDecoder(slide_file)
        self.slide.get_level_count()
        self._slide_pixel_size = self.slide.get_pixel_size()
        
        slide_width, slide_height = self.slide.get_level_dimensions()[0]
        self._width = slide_width
        self._height = slide_height

        self.slide_level_pixel_size = [self.slide_pixel_size, ]
        for level in range(0, self.slide.get_level_count()):
            width = self.slide.get_level_dimensions()[level][0]
            self.slide_level_pixel_size.append(
                self.slide_pixel_size * (self.width / width)
            )
        
    def close(self):
        pass

    def get_best_crop_level(self, crop_pixel_size):
        # find best crop level
        crop_level = bisect.bisect_left(self.slide_level_pixel_size, crop_pixel_size) - 1
        if crop_level < 0:
            crop_level = 0
        # use level 0, maybe get error if use other level
        crop_level = 0
        return crop_level, self.slide_level_pixel_size[crop_level]

    def crop_patch(self, x, y, width, height, crop_pixel_size, crop_level=None):
        """crop patch"""
        if crop_level is None:
            crop_level, slide_pixel_size = self.get_best_crop_level(crop_pixel_size)
        else:
            slide_pixel_size = self.slide_level_pixel_size[crop_level]
        crop_width = int(width * crop_pixel_size / slide_pixel_size)
        crop_height = int(height * crop_pixel_size / slide_pixel_size)
        # print(x, y, crop_width, crop_height, self.width, self.height)

        img = self.slide.read_region_bgr((x, y), crop_level, (crop_width, crop_height))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        return img


# modified by nowandfuture add crop for “ibl” format image
class IblReader(AbstractPSReader):
    """Iblslide Reader"""

    def __init__(self, slide_file):        
        super().__init__(slide_file)
        ret = -1
        retry_time = 5
        while ret < 0 and retry_time > 0:
            self.slide = IblWsi(bytes(slide_file, encoding = "utf-8"))
            ret = self.slide.ret
            if ret < 0:
                retry_time -= 1
                self.slide.CloseIBL()
            time.sleep(0.05)
        if ret != 0:
            raise RuntimeError(f"File Open Error: {slide_file} can't open successful.") 
        self._slide_pixel_size = self.get_slide_pixel_size(self.slide)
        
        self._width, self._height = self.slide.width.value, self.slide.height.value
        
    def close(self):
        self.slide.CloseIBL()

    def crop_patch(self, x, y, w, h, crop_pixel_size, crop_level=None):
        if crop_level is None:
            crop_level = 0
    
        ratio = crop_pixel_size / self._slide_pixel_size

        crop_width = int(w * ratio)
        crop_height = int(h * ratio)

        patch = self.slide.GetRoiData(float(self.slide.scanScale.value) / (2 ** crop_level), int(x), int(y), crop_width, crop_height)
        
        buff = np.frombuffer(patch, dtype=np.uint8)
        
        img = cv2.imdecode(buff, cv2.IMREAD_COLOR)
        if ratio != 1.0:
            img = cv2.resize(img, (w, h))

        return img

    def get_slide_pixel_size(self, slide):
        """Get the pixel size of the slide"""
        pixel_size = float(slide.fPixelSize.value)*1000
        return pixel_size


import cv2
class OpenCVReader(AbstractPSReader):
    def __init__(self, slide_file):        
        super().__init__(slide_file)
        self.slide = cv2.imread(slide_file, cv2.IMREAD_UNCHANGED)
        
        self._slide_pixel_size = self.get_slide_pixel_size(self.slide)
        
        # 3 dimensions support, 2D image only
        assert self.slide.ndim in [2, 3]

        if self.slide.ndim == 2:
            self.slide = self.slide[..., None]

        self._width, self._height = self.slide.shape[1], self.slide.shape[0]

    def close(self):
        pass

    def crop_patch(self, x, y, w, h, crop_pixel_size, crop_level=None):
        ratio = crop_pixel_size / self._slide_pixel_size

        crop_width = int(w * ratio)
        crop_height = int(h * ratio)

        patch = self.slide[y: y + crop_height, x:x + crop_width, :]
    
        img = cv2.resize(patch, (w, h))

        return img

    def get_slide_pixel_size(self, slide):
        """Get the pixel size of the slide"""
        return -1


class AutoReader(AbstractPSReader):

    DEFAULT_MAP: Dict[str, AbstractPSReader] = {
        ".ibl": IblReader,
        ".sdpc": SdpcReader,
        ".svs": OpenSlideReader,
        ".png": OpenCVReader,
        ".jpg": OpenCVReader,
        ".jpeg": OpenCVReader,
        ".tiff": OpenSlideReader,
        ".tif": OpenSlideReader,
        ".scn": OpenSlideReader,
        ".mrxs": OpenSlideReader,
        ".svslide": OpenSlideReader,
        ".vms": OpenSlideReader,
        ".vmu": OpenSlideReader,
        ".ndpi": OpenSlideReader,
        ".bif": OpenSlideReader
    }

    def __init__(self, slide_file, router=DEFAULT_MAP) -> None:
        super().__init__(slide_file)
        self.real: AbstractPSReader = router[Path(slide_file).suffix](slide_file)
        assert self.real is not None
        self._with = self.real.width
        self._height = self.real.height
        self._slide_pixel_size = self.real.slide_pixel_size

    @property
    def width(self):
        return self.real.width
    
    @property
    def height(self):
        return self.real.height

    @property
    def slide_pixel_size(self):
        return self.real.slide_pixel_size

    def crop_patch(self, x, y, width, height, crop_pixel_size, crop_level=None):
        return self.real.crop_patch(x, y, width, height, crop_pixel_size, crop_level)

    def read(self, pixel_size=None):
        return self.real.read(pixel_size)

    def close(self):
        self.real.close()

