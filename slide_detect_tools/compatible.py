# This file provide the compatible of the reader of different formats (.sdpc, .ibl .eg.) with OpenSlide
# Also provide the interface what is compatible with DeepZoom server
import bisect
from pathlib import Path
from openslide import AbstractSlide
from PIL import Image
from .sdpc_encode_fixed import SdpcDecoder
import numpy as np
from typing import Mapping
import imageio 

import cv2
try:
    from libiblsdk.ibl_py_sdk import IblWsi
except:
    pass
from typing import Mapping
import openslide
from .slide_supports import sdpc_support_formats, ibl_support_formats, openslide_support_formats, kfb_support_formats
from openslide import OpenSlide
try:
    from kfb import KfbSlide
except:
    pass

# from turbojpeg import TurboJPEG,TJPF_GRAY,TJSAMP_GRAY,TJFLAG_PROGRESSIVE,TJPF_BGR

"""
    Sdpc decoder compatible with DeepZoom
    by nowandfuture
"""
class  SdpcSlide(AbstractSlide):
    
    def __init__(self, filename) -> None:
        AbstractSlide.__init__(self)
        self._filename = filename
        self._osr = SdpcDecoder(filename)
        
    @classmethod
    def detect_format(cls, filename):
        """Return a string describing the format vendor of the specified file.

        If the file format is not recognized, return None."""
        ret = SdpcDecoder.check_header(filename)
        return "sdpc" if ret == 0 else None

    def close(self):
        """Close the OpenSlide object."""
        self._osr.close()
        
    @property
    def level_count(self):
        """The number of levels in the image."""
        return self._osr.get_level_count()

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.

        level_dimensions[n] contains the dimensions of level n."""
        return self._osr.get_level_dimensions()
        
    @property
    def level_downsamples(self):
        """A list of downsampling factors for each level of the image.

        level_downsample[n] contains the downsample factor of level n."""

        return self._osr.get_level_downsamples()
        
    @property
    def properties(self):
        """Metadata about the image.

        This is a map: property name -> property value."""
        return _SdpcPropertyMap(self._osr)

    @property
    def associated_images(self):
        """Images associated with this whole-slide image.

        This is a map: image name -> PIL.Image.

        Unlike in the C interface, the images accessible via this property
        are not premultiplied."""
        return None

    def get_best_level_for_downsample(self, downsample):
        """Return the best level for displaying the given downsample."""
        return bisect.bisect(self.level_downsamples, int(downsample)) - 1


    def read_region(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        
        data = self._osr.read_region(
            location, level, size
        )
        return Image.fromarray(data, mode="RGB").convert("RGBA")

    def get_best_level_for_downsample(self, downsample):
        """Return the best level for displaying the given downsample."""
        return bisect.bisect(self.level_downsamples, int(downsample)) - 1


    def read_region_native(self, location, level, size):
        """Return a numpy ndarray containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        
        data = self._osr.read_region(
            location, level, size
        )
        return data


    def set_cache(self, cache):
        """Use the specified cache to store recently decoded slide tiles.

        By default, the object has a private cache with a default size.

        cache: an OpenSlideCache object."""
        pass
        
class _SdpcMap(Mapping):
    def __init__(self, osr):
        self._osr = osr

    def __repr__(self):
        return f'<{self.__class__.__name__} {dict(self)!r}>'

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return iter(self._keys())

    def _keys(self):
        # Private method; always returns list.
        raise NotImplementedError()

import openslide
class _SdpcPropertyMap(_SdpcMap):
    def _keys(self):
        return {"level_count", "level_dimensions", "level_downsamples", "pixel_size", openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y}

    @staticmethod
    def get_pixel_size(sdpc: SdpcDecoder):
        return (sdpc.sdpc.contents.picHead.contents.scale, sdpc.sdpc.contents.picHead.contents.ruler)

    def __getitem__(self, key):
    
        if key == openslide.PROPERTY_NAME_MPP_X or key == openslide.PROPERTY_NAME_MPP_Y:
            return self.get_pixel_size(self._osr)[1]
        
        v = None
        if key in self._keys():
            method_ = getattr(self._osr, f"get_{key}")
            v = method_()
        if v is None:
            raise KeyError()
        return v
    
"""
    IBL SDK compatible with DeepZoom
    by nowandfuture
    TODO finishe the ibl sdk
"""
class  IblSlide(AbstractSlide):
    
    def __init__(self, filename) -> None:
        AbstractSlide.__init__(self)
        self._filename = filename
        self._osr = IblWsi(bytes(filename, "utf-8"))
        if self._osr.ret != 0:
            self.close()
            raise RuntimeError("Open IBL slide failed.")
        
        # precompute the downsamples
        self.max_scale = self.properties["scan_scale"]

        __downsamples = [1, 2, 4, 8, 16, 32, 64]
        
        # while self.max_scale / __downsamples[-1] < 1:
        #     __downsamples.append(__downsamples[-1] * 2)
            
            
        # if __downsamples[-1] > self.max_scale:
        #     __downsamples[-1] = self.max_scale
        
        self._downsamples = __downsamples
        
    @classmethod
    def detect_format(cls, filename):
        """Return a string describing the format vendor of the specified file.
        If the file format is not recognized, return None."""
        ret, _, _, _, _, _, _, _, _ = IblWsi.GetHeaderInfo(bytes(filename, "utf-8"))
        return "ibl" if ret == 0 else None

    def close(self):
        """Close the OpenSlide object."""
        self._osr.CloseIBL()
        
    @property
    def level_count(self):
        """The number of levels in the image."""
        return len(self.level_downsamples)

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.

        level_dimensions[n] contains the dimensions of level n."""
        level_0_dimension = (int(self._osr.width.value), int(self._osr.height.value))
        return [(int(level_0_dimension[0] / s), int(level_0_dimension[1] / s)) for s in self._downsamples]
    
    # @staticmethod
    # def high_bit(x):
    #     x = x | (x >> 1)
    #     x = x | (x >> 2)
    #     x = x | (x >> 4)
    #     x = x | (x >> 8)
    #     x = x | (x >> 16)
    #     x = x | (x >> 32)
    #     return x + 1
    
    @property
    def level_downsamples(self):
        """A list of downsampling factors for each level of the image.

        level_downsample[n] contains the downsample factor of level n."""
        return self._downsamples
        
    @property
    def properties(self):
        """Metadata about the image.

        This is a map: property name -> property value."""
        return _IblPropertyMap(self._osr)

    @property
    def associated_images(self):
        """Images associated with this whole-slide image.

        This is a map: image name -> PIL.Image.

        Unlike in the C interface, the images accessible via this property
        are not premultiplied."""
        return None

    def get_best_level_for_downsample(self, downsample):
        """Return the best level for displaying the given downsample."""
        return max(bisect.bisect(self.level_downsamples, int(downsample)) - 1, 0)


    def read_region(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        # crop_level, int(x), int(y), crop_width, crop_height
        img = self.read_region_native(location, level, size)

        return Image.fromarray(img, "RGB").convert("RGBA")

    def read_region_native(self, location, level, size):
        """Return a numpy array containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        downsample = self.level_downsamples[level]
        scan_scale = self.max_scale / downsample
        data = self._osr.GetRoiData(
           scan_scale,  int(location[0] / downsample),int(location[1] / downsample), int(size[0]), int(size[1])
        )
        buff = np.frombuffer(data, dtype=np.uint8)
        try:
            decode_img = cv2.imdecode(buff, 1)
            reshaped = decode_img.reshape((int(size[1]), int(size[0]), 3))
            img = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise RuntimeError(f"{self.dimensions}, {self.max_scale}, {self.level_dimensions},{location}, {level}, {size}")

        return img
        
    def read_region_raw(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size.

        Unlike in the C interface, the image data returned by this
        function is not premultiplied."""
        downsample = self.level_downsamples[level]
        scan_scale = self.max_scale / downsample
        data = self._osr.GetRoiData(
           scan_scale,  int(location[0] / downsample),int(location[1] / downsample), int(size[0]), int(size[1])
        )
    
        return data

    def set_cache(self, cache):
        """Use the specified cache to store recently decoded slide tiles.

        By default, the object has a private cache with a default size.

        cache: an OpenSlideCache object."""
        pass
        
class _IblMap(Mapping):
    def __init__(self, osr):
        self._osr = osr

    def __repr__(self):
        return f'<{self.__class__.__name__} {dict(self)!r}>'

    def __len__(self):
        return len(self._keys())

    def __iter__(self):
        return iter(self._keys())

    def _keys(self):
        # Private method; always returns list.
        raise NotImplementedError()

class _IblPropertyMap(_IblMap):
    def _keys(self):
        return {"jpeg_quality", "scan_scale", "depth", "focus_number", "pixel_size",  "backgroud_color", openslide.PROPERTY_NAME_MPP_X, openslide.PROPERTY_NAME_MPP_Y, openslide.PROPERTY_NAME_BACKGROUND_COLOR}

    @staticmethod
    def get_pixel_size(osr: IblWsi):
        return osr.fPixelSize.value * 1000

    def __getitem__(self, key):
    
        if key == openslide.PROPERTY_NAME_MPP_X:
            return self.get_pixel_size(self._osr)
        if key == openslide.PROPERTY_NAME_MPP_Y:
            return self.get_pixel_size(self._osr)
        if key == openslide.PROPERTY_NAME_BACKGROUND_COLOR:
            return  hex(self._osr.bkColor.value)[2:] * 3
        
        v = None
        if key in self._keys():
            if key == "jpeg_quality":
                v = self._osr.jpegQuality.value
            elif key == "scan_scale":
                v = float(self._osr.scanScale.value)
            elif key == "focus_number":
                v = self._osr.focusNumber.value
            elif key == "depth":
                v = self._osr.depth.value
            elif key == "pixel_size":
                v = self._osr.fPixelSize.value
                
        if v is None:
            raise KeyError()
        return v

class ImageSlide2(openslide.ImageSlide):
    """A wrapper for a PIL.Image that provides the OpenSlide interface."""

    def __init__(self, file):
        """Open an image file.

        file can be a filename or a PIL.Image."""
        AbstractSlide.__init__(self)
        self._file_arg = file
        if isinstance(file, np.ndarray):
            self._image = file
        else:
            self._image = imageio.imread(file)
            
            # normalized to 0-255 if the image is out of [0,255] or all the pixel values are  less than 1
            max_ = self._image.max()
            min_ = self._image.min()
            if max_ > 255 or min_< 0 or max_ <= 1.01:
                self._image = (self._image - min_) / (max_ - min_) * 255
 
    def __repr__(self):
        return f'{self.__class__.__name__}({self._file_arg!r})'

    @classmethod
    def detect_format(cls, filename):
        """Return a string describing the format of the specified file.

        If the file format is not recognized, return None."""
        image = imageio.imread(filename)
            
        return "imageio support" if image is not None else None


    def close(self):
        """Close the slide object."""
        del self._image
        self._image = None

    @property
    def level_dimensions(self):
        """A list of (width, height) tuples, one for each level of the image.

        level_dimensions[n] contains the dimensions of level n."""
        return (self._image.shape[:2][::-1],)

    def read_region(self, location, level, size):
        """Return a PIL.Image containing the contents of the region.

        location: (x, y) tuple giving the top left pixel in the level 0
                  reference frame.
        level:    the level number.
        size:     (width, height) tuple giving the region size."""
        if level != 0:
            raise openslide.OpenSlideError("Invalid level")
        if ['fail' for s in size if s < 0]:
            raise openslide.OpenSlideError(f"Size {size} must be non-negative")
        # Any corner of the requested region may be outside the bounds of
        # the image.  Create a transparent tile of the correct size and
        # paste the valid part of the region into the correct location.
        image_topleft = [
            max(0, min(l, limit - 1)) for l, limit in zip(location, self.level_dimensions[0])
        ]
        image_bottomright = [
            max(0, min(l + s - 1, limit - 1))
            for l, s, limit in zip(location, size, self.level_dimensions[0])
        ]
        tile = Image.new("RGBA", size, (0,) * 4)
        if not [
            'fail' for tl, br in zip(image_topleft, image_bottomright) if br - tl < 0
        ]:  # "< 0" not a typo
            # Crop size is greater than zero in both dimensions.
            # PIL thinks the bottom right is the first *excluded* pixel
            crop = self._image[image_topleft[1]:image_bottomright[1] + 1, image_topleft[0]:image_bottomright[0] + 1]
            crop = Image.fromarray(crop).convert("RGBA")
            tile_offset = tuple(il - l for il, l in zip(image_topleft, location))
            tile.paste(crop, tile_offset)
        return tile

class FormatError(Exception):

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
        
class SildeFactory:
    
    @staticmethod
    def of(path: str) -> AbstractSlide:
        path: Path  = Path(path)
        if path.suffix in sdpc_support_formats:
            actual_format = SdpcSlide.detect_format(str(path))
            if actual_format is not None:
                return SdpcSlide(str(path))
        elif path.suffix in ibl_support_formats:
            # read the header of the file
            actual_format = IblSlide.detect_format(str(path))
            if actual_format is not None:
                return IblSlide(str(path))
        elif path.suffix in kfb_support_formats:
            actual_format = "kfb"
            if actual_format is not None:
                return KfbSlide(str(path))
        elif path.suffix in openslide_support_formats:
            # read the header of the file
            # return openslide.open_slide(str(path))
            try:
                return OpenSlide(str(path))
            except openslide.OpenSlideUnsupportedFormatError:
                return ImageSlide2(str(path))
        else:
             # read the header of the file
            actual_format = ImageSlide2.detect_format(str(path))
            if actual_format is not None:
                return ImageSlide2(str(path))
        raise FormatError("Unrecognized format.")