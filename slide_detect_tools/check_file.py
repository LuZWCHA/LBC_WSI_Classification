from dataclasses import dataclass
import logging
from os import O_APPEND
import subprocess
import datasets.our_datasets as dt
import pipeline
from pipeline import *


import glob, tqdm, random
from pathlib import Path
from slide_crop_patch import AutoReader, OpenSlideReader
from datasets.our_datasets import WSI_RAW
import openslide

"""by nowandfuture"""

""" This file provide the function to  check donwload files by opreation: random crop and file open, and if the file opened failed or crop failed it will automatically download/sync file using tencent's cos service """

class CropWork(Worker):
    def process(self, p: DataPacket) -> DATA_PACKET:
        slide_path = p.obj

        with pipeline.core.suppress_stdout_stderr(out=False):
            with AutoReader(slide_path) as reader:
                point_x_list = [random.randrange(0, reader.width - 1280) for _ in range(2)]
                point_y_list = [random.randrange(0, reader.height - 1280) for _ in range(2)]
                max_x = max(point_x_list)
                min_x = min(point_x_list)
                max_y = max(point_y_list)
                min_y = min(point_y_list)
                if min_x < 0 or min_y < 0:
                    raise RuntimeError("Abort crop ...")
                # min_x = max(min_x, reader.width -1 - 1280)
                # min_y = max(min_x, reader.height -1 - 1280)
                reader.crop_patch(min_x, min_y, 1280, 1280, crop_pixel_size=0.31)

        return p

@dataclass
class Result:
    return_code: int
    stdout: str
    stderr: str
    
    def __str__(self):
        return f"return_code: {self.return_code}, stdout: {self.stdout}, stderr: {self.stderr}"
    
    @staticmethod
    def create(res: subprocess.CompletedProcess) -> Self:
        return Result(int(res.returncode), str(res.stdout, encoding='utf-8'), str(res.stderr, encoding='utf-8'))

class SyncFile(Worker):
    def __init__(self, coscli_path) -> None:
        super().__init__()
        self.coscli_path = coscli_path

    def process(self, p: DataPacket) -> DATA_PACKET:
    
        file_path = p.obj
        file_path = Path(file_path)
        path = file_path.relative_to(dt.WSI_RAW.root_path)
        
        # example:
        # ./coscli sync cos://ai-data-1253492636/ai_lab_sync/slides/jys_agc_storage/wsi_datasets/abp_agc jys_agc_storage/wsi_datasets/abp_agc

        if openslide.OpenSlide.detect_format(str(file_path)) == ".mrxs":
            # this kind of file depend on other files
            res1 = subprocess.run([self.coscli_path, "sync", f"cos://ai-data-1253492636/ai_lab_sync/slides/{path}", file_path], capture_output=True)
            res2 = subprocess.run([self.coscli_path, "sync", f"cos://ai-data-1253492636/ai_lab_sync/slides/{path.stem}.zip", file_path.stem + ".zip"], capture_output=True)
            return_code = 0 if res1.returncode == res2.returncode == 0 else 1
            res = Result(return_code, res1.stdout + ";" + res2.stdout, res1.stderr + ";" + res2.stderr)        
        else:
            res = subprocess.run([self.coscli_path, "sync", f"cos://ai-data-1253492636/ai_lab_sync/slides/{path}", file_path], capture_output=True)
        
        res: Result = Result.create(res)
        if res.return_code != 0:
            print(res)
            raise RuntimeError(f"Sync failed: {res.stderr}")
        return DataPacket(res.stdout)


def sync_file(file_list, coscli_path):
    pipeline.create(file_list) \
                .connect(PSegment(SyncFile(coscli_path), 1)) \
                .subscribe(ProgressObserver(recv= lambda p: p, do_print=True))

def check_images(all_file_path, checkpoint, log_output_path="logs/failed.log"):
    
    failed_file_path = []
    failed_log = log_output_path
    
    def error(p: ErrorPacket):
        failed_file_path.append(p.failed_datapack_content())
        with open(failed_log, "a") as f:
            f.write(str(p.failed_datapack_content()) + "," + str(p.format_error()) + "\n")
                
    def end(bar: tqdm.tqdm):
        bar.total = bar.n
        with open(failed_log, "a") as f:
            f.write("\n")
            f.writelines(failed_file_path)
            
    import pickle
    files =  all_file_path if checkpoint is None else pickle.load(open(checkpoint))
    pipeline.create(files) \
                    .connect(Filter(lambda path: Path(path.obj).is_file() and Path(path.obj).suffix in dt.get_default_support_slide_formats()[0])) \
                    .connect(PSegment(CropWork())) \
                    .subscribe(ProgressObserver(recv= lambda p: p,error=error, end=end, total_size=len(files)))
    
    return failed_file_path


if __name__ == "__main__":

    # to change the sync method, pelease modify the class SyncFile.

    # check whole dataset example
    failed_paths = check_images(glob.glob(WSI_RAW.at("**/*"), recursive=True), None)
    
    # sync all failed slide example
    sync_file(failed_paths, "/nasdata/ai_data/coscli")