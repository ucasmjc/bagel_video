"""
Distributed basic functions.
"""

import os
import torch
import torch.distributed as dist

from typing import Optional
from torch.nn.parallel import DistributedDataParallel

_DATA_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_CPU_GROUP = None


def get_global_rank() -> int:
    """
    Get the global rank, the global index of the GPU.
    """
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    """
    Get the local rank, the local index of the GPU.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    """
    Get the world size, the total amount of GPUs.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    """
    Get current rank device.
    """
    return torch.device("cuda", get_local_rank())


def barrier_if_distributed(*args, **kwargs):
    """
    Synchronizes all processes if under distributed context.
    """
    if dist.is_initialized():
        return dist.barrier(*args, **kwargs)


def init_torch(cudnn_benchmark=True):
    """
    Common PyTorch initialization configuration.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.cuda.set_device(get_local_rank())
    dist.init_process_group(
        backend="nccl",
        rank=get_global_rank(),
        world_size=get_world_size(),
    )


def convert_to_ddp(module: torch.nn.Module, **kwargs) -> DistributedDataParallel:
    return DistributedDataParallel(
        module=module,
        device_ids=[get_local_rank()],
        output_device=get_local_rank(),
        **kwargs,
    )


def get_data_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get data parallel process group.
    """
    return _DATA_PARALLEL_GROUP


def get_sequence_parallel_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel process group.
    """
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_cpu_group() -> Optional[dist.ProcessGroup]:
    """
    Get sequence parallel CPU process group.
    """
    return _SEQUENCE_PARALLEL_CPU_GROUP


def get_data_parallel_rank() -> int:
    """
    Get data parallel rank.
    """
    group = get_data_parallel_group()
    return dist.get_rank(group) if group else get_global_rank()


def get_data_parallel_world_size() -> int:
    """
    Get data parallel world size.
    """
    group = get_data_parallel_group()
    return dist.get_world_size(group) if group else get_world_size()


def get_sequence_parallel_rank() -> int:
    """
    Get sequence parallel rank.
    """
    group = get_sequence_parallel_group()
    return dist.get_rank(group) if group else 0


def get_sequence_parallel_world_size() -> int:
    """
    Get sequence parallel world size.
    """
    group = get_sequence_parallel_group()
    return dist.get_world_size(group) if group else 1


def init_sequence_parallel(sequence_parallel_size: int):
    """
    Initialize sequence parallel.
    """
    global _DATA_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_CPU_GROUP
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    data_parallel_size = world_size // sequence_parallel_size
    for i in range(data_parallel_size):
        start_rank = i * sequence_parallel_size
        end_rank = (i + 1) * sequence_parallel_size
        ranks = range(start_rank, end_rank)
        group = dist.new_group(ranks)
        cpu_group = dist.new_group(ranks, backend="gloo")
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_CPU_GROUP = cpu_group
