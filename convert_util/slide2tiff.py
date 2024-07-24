
import glob
import os
import time

import imageio.v2 as imageio
import pandas as pd
try:
    import pyvips
except Exception as e:
    pass
from slide_detect_tools import slide_supports, compatible
from pathlib import Path
from PIL import Image
import numpy as np
import openslide
from pipeline import *
from tifffile import TiffWriter
# vips im_vips2tiff <source_image> <output_image.tif>:<compression>,tile:<size>,pyramid
# vips tiffsave source_image output_image.tif --tile --pyramid --compression deflate --tile-width 256 --tile-height 256

def vips_convert(img, output_path, tile_width=512, tile_height=512, xr=10, yr=10, compression="jpeg"):
    start = time.time_ns()
    img.tiffsave(output_path,  tile=True, pyramid=True, tile_width=tile_width,
                 tile_height=tile_height, compression=compression, xres=xr, yres=yr)
    del img
    print("save time", (time.time_ns() - start) / 1e6)

def image_convert2ometiff(path, output_path, tile=(256, 256), pixelsize = 0.1, subresolutions = 0, downsample=32,compress="jpeg"):
    data = imageio.imread(path)[None, None, ...]
    with TiffWriter(output_path, bigtiff=True) as tif:
        metadata={
             'axes': 'TSCYX',
             'SignificantBits': 10,
             'TimeIncrement': 0.1,
             'TimeIncrementUnit': 's',
             'PhysicalSizeX': pixelsize,
             'PhysicalSizeXUnit': 'Âµm',
             'PhysicalSizeY': pixelsize,
             'PhysicalSizeYUnit': 'Âµm',
             'Channel': {'Name': ['Channel 0']},
             'Plane': {'PositionX': [0.0] * 16, 'PositionXUnit': ['Âµm'] * 16},
            'unit': 'um'
         }
         
        options = dict(
             photometric='rgb',
             tile=tile,
             compression=compress
         )
         
        tif.write(
             data,
             subifds=subresolutions,
             resolution=(1e4 / pixelsize, 1e4 / pixelsize),
             metadata=metadata,
             **options
         )
         
         # write pyramid levels to the two subifds
         # in production use resampling to generate sub-resolution images
        for level in range(subresolutions):
             mag = 2**(level + 1)
             tif.write(
                 data[..., ::mag, ::mag, :],
                 subfiletype=1,
                 resolution=(1e4 / mag / pixelsize, 1e4 / mag / pixelsize),
                 **options
             )
         # add a thumbnail image as a separate series
         # it is recognized by QuPath as an associated image
        thumbnail = (data[0, 0, ::downsample, ::downsample] >> 2).astype('uint8')
        tif.write(thumbnail, metadata={'Name': 'thumbnail'})
         

def slide_convert2tiff(path, output_path, tile_width=512, tile_height=512, max_size_for_ibl=1 << 14, max_size_for_kfb=1 << 12):
    if Path(path).suffix in slide_supports.ibl_support_formats + slide_supports.sdpc_support_formats + slide_supports.kfb_support_formats:

        slide = compatible.SildeFactory().of(path)
        # print( slide.properties)
        xr = slide.properties[openslide.PROPERTY_NAME_MPP_X]
        yr = slide.properties[openslide.PROPERTY_NAME_MPP_Y]
        xr = 1 / xr * 1e4
        yr = 1 / yr * 1e4

        tile_size = max_size_for_ibl if Path(
            path).suffix in slide_supports.ibl_support_formats else min(slide.dimensions)
        tile_size = max_size_for_kfb if Path(
            path).suffix in slide_supports.kfb_support_formats else tile_size
        
        w_cnt = slide.dimensions[0] // tile_size
        w_rest = slide.dimensions[0] % tile_size
        w_cnt += 1 if w_rest > 0 else 0
        h_cnt = slide.dimensions[1] // tile_size
        h_rest = slide.dimensions[1] % tile_size
        h_cnt += 1 if h_rest > 0 else 0
        
        if w_cnt == 1:
            w_rest = tile_size
        if h_cnt == 1:
            h_rest = tile_size
        row_datas = []
        
        start = time.time_ns()
        for w_c in range(w_cnt):
            col_datas = []
            for h_c in range(h_cnt):
                cur_tile_w = tile_size if w_c != w_cnt - 1 else w_rest
                cur_tile_h = tile_size if h_c != h_cnt - 1 else h_rest
                start_x = max(w_c, 0) * tile_size
                start_y = max(h_c, 0) * tile_size
                # the method read_region_native only exsits in IBL and SDPC slide structure
                
                tile = slide.read_region_native(
                    (start_x, start_y), 0, (cur_tile_w, cur_tile_h))
                
                col_datas.append(tile)
                
            row_datas.append(np.concatenate(col_datas, axis=0))
            col_datas.clear()
            
        del col_datas
        img_array = np.concatenate(row_datas, axis=1)
        del row_datas
        
        slide.close()
        print("copy time", (time.time_ns() - start) / 1e6)
        vips_img = pyvips.Image.new_from_array(img_array)
    else:
        vips_img = pyvips.Image.new_from_file(path)
        xr = vips_img.xres
        yr = vips_img.yres
    
    # xr, yr: pixel / mm
    return vips_convert(vips_img, output_path, tile_width=tile_width, tile_height=tile_height, xr=xr, yr=yr)

class ConvertWorker(Worker):

    def __init__(self, save_dir) -> None:
        super().__init__()
        self.save_dir = save_dir

    def process(self, p: DataPacket) -> DATA_PACKET:
        path = p.obj
        save_file = Path(path).stem + ".tif"
        save_file = os.path.join(self.save_dir, save_file)
        if os.path.exists(save_file):
            print("skip", save_file)
            return p
        slide_convert2tiff(path, save_file)
        return p
        
def conver2tiff_batch(paths, save_dir, worker_num=16):
    create(paths)\
    .connect(PSegment(ConvertWorker(save_dir), worker_num))\
    .subscribe(ProgressObserver(lambda x: print(x.obj), total_size=len(paths)))


# def tif_tag_modify(tiff_file):
#     import tifffile
#     import imageio as io
#     import numpy as np
#     # image = np.zeros(shape=(1000, 1000), dtype=np.uint8)
#     # io.imwrite(tiff_file, image)
#     tiff = tifffile.TiffFile(tiff_file, 'r+b')
#     try:
#         old_x_value = tiff.pages[0].tags['XResolution'].value
#         old_y_value = tiff.pages[0].tags['YResolution'].value

        
#         if old_x_value[0] / old_x_value[1] < 10000:
#             print(f"modify {tiff_file}")
#             tiff.pages[0].tags['XResolution'].overwrite(tiff, (old_x_value[0] * 10, old_x_value[1]))
#             tiff.pages[0].tags['YResolution'].overwrite(tiff, (old_y_value[0] * 10, old_y_value[1]))
#     except Exception as e:
#         print(tiff_file, e)
#     tiff.close()

if __name__ == "__main__":
    # https://www.libvips.org/API/current/VipsForeignSave.html#vips-tiffsave
    # img = pyvips.Image.new_from_file("/nasdata/private/zwlu/Now/ai_trainer/.data/slides/20210526_125715_1.svs")
    # # print(img)
    # img.tiffsave(".data/slides/20210526_125715_1.tif",  tile=True, pyramid=True, bigtiff=True, )
    # print("finshed write")

    # conver2tiff("/workspace/AIMS-702.ibl", ".data/slides/AIMS-702.tif")

    # from openslide import OpenSlide
    # import pandas as pd
    # slide_convert2tiff("/nasdata/private/zwlu/Now/ai_trainer/.data/slides/21-32224+21-38210HE.kfb", ".data/slides/21-32224+21-38210HE.tif")
    
    
    # test_tiff = ".data/slides/AIMS-702.tif"
    # test_tiff = "/nasdata/dataset/moshi_data/dt3/2023-03-15/-20230315-225229.ibl.tiff"
    # slide = OpenSlide(test_tiff)
    # old_tiff_files = glob.glob("/nasdata/dataset/moshi_data/tiff/dt2/*.tif")
    # print(len(old_tiff_files))
    # for  test_tiff in old_tiff_files:
    
    #     tif_tag_modify(test_tiff)
    #     try:
    #         slide = OpenSlide(test_tiff)
    #         # print(slide.detect_format(test_tiff))
    #         # print(slide.level_count)
    #         # print(slide.dimensions)
    #         # print(slide.level_downsamples)
    #         # print(test_tiff, slide.properties["tiff.XResolution"], slide.properties["tiff.YResolution"])
    #         # slide.get_thumbnail((512, 512)).show()
    #         slide.close()
    #     except Exception as e:
    #         print(test_tiff, e)
    

    
    # tct_path = "/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_dt1.csv"
    output_path = "/nasdata/dataset/moshi_data/tiff/dt4_part_0"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # import pandas as pd
    # ibl_paths = pd.read_csv(tct_path)["slide_path"].tolist()
    
    data = pd.read_csv("/nasdata/private/jli/dataset/baseline/baseline_fy.csv")
    print(data)
    ibl_paths = data["slide_path"].tolist()
    # ibl_paths = glob.glob("/nasdata/dataset/moshi_data/北京安必平AI扫描/北京协和临床试验/2023-06-15/*.ibl")
    conver2tiff_batch(ibl_paths, output_path, worker_num=10)
    
    
    # tct_path = "/nasdata/private/zwlu/Now/ai_trainer/scripts/inference_scripts/inference_tct_dt2.csv"
    # output_path = "/nasdata/dataset/moshi_data/tiff/dt2"
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    
    # ibl_paths = pd.read_csv(tct_path)["slide_path"].tolist()
    # conver2tiff_batch(ibl_paths, output_path)
# end main
