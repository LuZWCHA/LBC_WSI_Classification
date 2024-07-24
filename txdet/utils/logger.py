from detectron2.utils import comm, logger
from detectron2.engine.defaults import _try_get_key

def setup_logger(cfg, args):
    output_dir = _try_get_key(cfg, "OUTPUT_DIR", "output_dir", "train.output_dir")
    rank = comm.get_rank()
    logger.setup_logger(output_dir, distributed_rank=rank, name="txdet")