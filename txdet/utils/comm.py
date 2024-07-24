import torch
import torch.distributed as dist

from detectron2.utils.comm import get_world_size

DIST_OPS = {
    "SUM": dist.ReduceOp.SUM,
    "MAX": dist.ReduceOp.MAX,
    "MIN": dist.ReduceOp.MIN,
    "BAND": dist.ReduceOp.BAND,
    "BOR": dist.ReduceOp.BOR,
    "BXOR": dist.ReduceOp.BXOR,
    "PRODUCT": dist.ReduceOp.PRODUCT,
}

def all_reduce(data, op="sum"):
    world_size = get_world_size()
    if world_size > 1:
        reduced_data = data.clone()
        dist.all_reduce(reduced_data, op=DIST_OPS[op.upper()])
        return reduced_data
    return data
