from enum import Enum
import numpy.ctypeslib as npCtypes
from ctypes import *
import os

# import the sdpc to load the .so library
from sdpc.Sdpc import so

"""by nowandfuture"""

# rewrite the sdpc decoder by nowandfuture
# The poor code style of the origin one :(
# And missing essentail of the ERROR of the class Slide

# To use this module, please install the thirdpart library: sdpc for linux, it provide the .so dynamic library of C.
# Full inforamation please see: https://github.com/lzx325/sdpc2svs_repo/blob/dae5cba212e5f45dafa7ea0cbfa0d9457cfa30f4/include/param.h
# The 'param.h' file provide (nearly) compeleted information of the SDPC LIBRARY's variables
# so.macrograph

# so.SqGetRoiRgbOfSpecifyLayer.argtypes = [POINTER(SqSdpcInfo), POINTER(POINTER(c_uint8)),
#                                          c_int, c_int, c_uint, c_uint, c_int]
# so.SqGetRoiRgbOfSpecifyLayer.restype = c_int

class SqCode(Enum):
    SqSuccess = 0, "Success"
    SqFileFormatError = -1,	"File Format Error"  # 文件格式错误
    SqOpenFileError = -2, "Open File Error"  # 打开文件错误
    SqReadFileError = -3, "Read File Error"  # 读取文件错误
    SqWriteFileError = -4, "Write File Error"  # 写入文件错误
    SqJpegFormatError = -5, "Jpeg Format Error"  # Jpeg格式错误
    SqEncodeJpegError = -6, "Encode Jpeg Error"  # 压缩jpeg格式错误
    SqDecodeJpegError = -7, "Decode Jpeg Error"  # 解压jpeg格式错误
    SqSliceNumError = -8, "Slice Num Error"  # 切片数量错误
    SqGetSliceRgbError = -9, "Get Slice Rgb Error"  # 获取rgb小图错误
    SqPicInfoError = -10, "Picture Information Error"  # picInfo信息错误
    SqGetThumbnailError = -11, "Get Thumbnail Error"  # 读取缩略图错误
    SqPicHeadError = -12, "Picture Head Error"  # 包头信息错误
    SqPathError = -13, "Path Error"  # 路径错误
    SqDataNullError = -14, "Data Null Error"  # 数据为空
    SqPersonInfoError = -15, "Person Information Error"  # 病理信息错误
    SqMacrographInfoError = -16, "Macrograph Information Error"  # 宏观图信息错误
    SqNotExist = -17, "Not Exist"  # 不存在（假如病理信息与宏观图出现这个，不是错误，不影响后面信息）
    SqLayerIndexesError = -18, "Layer Indexes Error"  # 层级索引错误
    SqSliceIndexesError = -19, "Slice Indexes Error"  # 指定小图索引错误
    SqROIRange = -20, "ROI Range Error"  # 取值范围错误
    SqBlockJpeg = -21, "Block Jpeg Error"  # 自定义将Sdpc切块成Jpeg错误
    SqExtraInfoError = -22, "ExtraInformation Error"  # 额外信息错误
    SqTileImageHeadError = -23, "Tile Image Head Error"  # 白细胞信息头错误
    SqTileImageConfigCheckError = -24, "Tile Image Config Check Error"  # 血液配置文件校验失败
    SqTileImageConfig2JsonError = - \
        25, "Tile Image Config To Json Error"  # 血液配置文件转换Json失败
    SqTileImageConfigNodeError = -26, "Tile Image Config Node Error"  # 血液配置文件获取节点失败
    SqTileImageConfigHeadError = -27, "Tile Image Config Head Error"  # 血液配置文件头信息错误
    SqDecodeHevcError = -28, "Decode Hevc Error"      # hevc解码错误

    def __init__(self, code, info) -> None:
        super().__init__()
        self._code = code
        self._info = info

    @property
    def code(self):
        return self._code

    @property
    def info(self):
        return self._info


class SdpcDecodeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


def raise_exception_if_has_error(error_code: int):
    if error_code != 0:
        for sq_code in SqCode:
            if sq_code.code == error_code:
                raise SdpcDecodeError(sq_code.info)


class SdpcDecodeHeadError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class SdpcDecoder:

    def __init__(self, sdpc_path):
        self.sdpc = SdpcDecoder.read(sdpc_path)
        self.__realse = False

    def get_bgr(self, bgr_pos, width, height):
        # FIX ME: not all sdcp using the bit depth of 3 (2^3 bits), check the head information to make the depth
        intValue = npCtypes.as_array(bgr_pos, (height, width, 3))
        return intValue

    HEADER_FLAG_HEX = "5351"

    @staticmethod
    def check_header(file_path):
        # "0x5153" should at the header (be)
        try:
            with open(file_path, "rb") as f:
                header_flag = f.read1(2).hex()
        except Exception as e:
            return -2
        # 0x5153 -> 0x5351 （le)
        if header_flag != SdpcDecoder.HEADER_FLAG_HEX:
            return -1
        return 0

    @staticmethod
    def read(file_name):
        sdpc = None
        # FIX ME
        # Default library didn't raise the exception, and I have to catch the expception by myself
        # SqOpenSdpc does not return the error code, it confused me.
        if SdpcDecoder.check_header(file_name) == 0:
            if os.path.exists(file_name):

                bytes_ = bytes(file_name, encoding='utf-8')

                sdpc = so.SqOpenSdpc(c_char_p(bytes_))
                try:
                    contents = getattr(sdpc, "contents",  None)
                    contents.fileName = bytes(file_name, encoding='utf-8')
                except Exception as e:
                    raise_exception_if_has_error(SqCode.SqOpenFileError)
            else:
                raise FileNotFoundError()
        else:
            raise RuntimeError("Unsupported slide.")
        return sdpc

    def get_level_count(self):
        return self.sdpc.contents.picHead.contents.hierarchy

    def get_pixel_size(self):
        # ruler -> physic size in real world per pixel (unit: um)
        pixel_p_size = self.sdpc.contents.picHead.contents.ruler
        return pixel_p_size

    def get_level_downsamples(self):
        level_count = self.get_level_count()
        # scale -> scan rate
        rate = self.sdpc.contents.picHead.contents.scale

        _list = []
        for i in range(level_count):
            _list.append(1 / rate ** i)
        return tuple(_list)

    def read_region(self, location, level, size):
        start_x, start_y = location
        ds: tuple = self.get_level_downsamples()

        if level >= len(ds):
            raise_exception_if_has_error(SqCode.SqLayerIndexesError.code)

        scale = ds[level]
        level_n_ds = self.get_level_dimensions()[level]
        
        start_x = int(start_x / scale)
        start_y = int(start_y / scale)

        width, height = size

        if (width + start_x) > level_n_ds[0] or (height + start_y) > level_n_ds[1]:
            raise_exception_if_has_error(SqCode.SqROIRange.code)

        rgb_pos = POINTER(c_uint8)()
        rgb_pos_pointer = byref(rgb_pos)

        ret = so.SqGetRoiRgbOfSpecifyLayer(
            self.sdpc, rgb_pos_pointer, width, height, start_x, start_y, level)
        raise_exception_if_has_error(ret)

        # The document decribe that the SqGetRoiRgbOfSpecifyLayer return RGB pointer, however it return BGR instead :(
        rgb = self.get_bgr(rgb_pos, width, height)[..., ::-1]
        rgb_copy = rgb.copy()

        so.Dispose(rgb_pos)

        del rgb_pos
        del rgb_pos_pointer

        return rgb_copy

    def read_region_bgr(self, location, level, size):
        start_x, start_y = location
        ds: tuple = self.get_level_downsamples()

        if level >= len(ds):
            raise_exception_if_has_error(SqCode.SqLayerIndexesError.code)

        scale = ds[level]
        level_n_ds = self.get_level_dimensions()[level]

        start_x = int(start_x / scale)
        start_y = int(start_y / scale)

        width, height = size
        if (width + start_x) > level_n_ds[0] or (height + start_y) > level_n_ds[1]:
            raise_exception_if_has_error(SqCode.SqROIRange.code)

        rgb_pos = POINTER(c_uint8)()
        rgb_pos_pointer = byref(rgb_pos)
        ret = so.SqGetRoiRgbOfSpecifyLayer(
            self.sdpc, rgb_pos_pointer, width, height, start_x, start_y, level)
        raise_exception_if_has_error(ret)

        rgb = self.get_bgr(rgb_pos, width, height)
        rgb_copy = rgb.copy()

        so.Dispose(rgb_pos)

        del rgb_pos
        del rgb_pos_pointer

        return rgb_copy

    def get_level_dimensions(self):

        def find_str_index(subStr, str):
            index1 = str.find(subStr)
            index2 = str.find(subStr, index1 + 1)
            index3 = str.find(subStr, index2 + 1)
            index4 = str.find(subStr, index3 + 1)
            return index1, index2, index3, index4

        level_count = self.get_level_count()
        level_dimensions = []
        for level in range(level_count):
            layerInfo: c_char = so.GetLayerInfo(self.sdpc, level)

            cnt = 0
            while layerInfo[cnt] != b'\0':
                cnt += 1
            str_info = str(bytes(layerInfo[:cnt]), encoding="utf-8")

            equal1, equal2, equal3, equal4 = find_str_index("=", str_info)
            line1, line2, line3, line4 = find_str_index("|", str_info)

            raw_width = int(str_info[equal1 + 1:line1])
            raw_height = int(str_info[equal2 + 1:line2])
            bound_width = int(str_info[equal3 + 1:line3])
            bound_height = int(str_info[equal4 + 1:line4])
            w, h = raw_width - bound_width, raw_height - bound_height
            level_dimensions.append((w, h))

        return tuple(level_dimensions)

    def close(self):
        if self.sdpc:
            so.SqCloseSdpc(self.sdpc)
            del self.sdpc
            self.sdpc = None
            self.__realse = True
        else:
            if self.__realse:
                raise SdpcDecodeError("Sdpc close twice.")
