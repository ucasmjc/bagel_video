import torch
import random
import importlib
import contextlib
import numpy as np

from typing import Any, Dict, List, Optional
from torch.utils.data import get_worker_info
from omegaconf import DictConfig, ListConfig

from .utils.partition_utils import partition_by_groups
from .utils.distributed_utils import get_data_parallel_rank, get_data_parallel_world_size


def get_worker_id() -> int:
    """
    Get the current dataloader worker id.
    """
    return get_worker_info().id if get_worker_info() is not None else 0


def get_worker_count() -> int:
    """
    Get the total dataloader worker count.
    """
    return get_worker_info().num_workers if get_worker_info() is not None else 1


def get_seed_for_rank_and_worker(seed: Optional[int]) -> Optional[int]:
    """
    Get seed for current rank and worker.
    """
    if seed is None:
        return None
    return seed + get_data_parallel_rank() * get_worker_count() + get_worker_id()


def get_random_for_rank_and_worker(seed: Optional[int]) -> random.Random:
    """
    Get random.Random for the current rank and worker.
    """
    return random.Random(get_seed_for_rank_and_worker(seed))


def get_random_for_all_ranks(seed: Optional[int]) -> random.Random:
    """
    Get random.Random that is the same for all ranks.
    """
    return random.Random(seed or 0)


def get_portion_for_rank_and_worker(items: List[Any], force: bool = False, allow_empty: bool = False) -> List[Any]:
    """
    Get the portion of items for current rank and worker.
    """
    rank = get_data_parallel_rank()
    world_size = get_data_parallel_world_size()
    worker_id = get_worker_id()
    worker_count = get_worker_count()

    if world_size * worker_count <= len(items):
        # If there are enough items to be divided, we divide the items
        items = partition_by_groups(items, world_size)[rank]
        items = partition_by_groups(items, worker_count)[worker_id]
    elif allow_empty:
        if rank * worker_count + worker_id < len(items):
            items = [items[rank * worker_count + worker_id]]
        else:
            items = []
    elif not force:
        # If not enough items to be divided, all ranks and workers shuffle it
        # with different seed.
        items = list(items)
        get_random_for_rank_and_worker(0).shuffle(items)
    else:
        raise ValueError("Items not divisible by world_size * worker_count")
    return items


def get_portion_for_worker_only(items: List[Any]) -> List[Any]:
    """
    Get the portion of items for current worker.
    """
    worker_id = get_worker_id()
    worker_count = get_worker_count()

    items = partition_by_groups(items, worker_count)[worker_id]
    return items


@contextlib.contextmanager
def local_seed(seed: Optional[int]):
    """
    Create a local context with seed is set, but exit back to the original random state.
    If seed is None, do nothing.
    """
    if seed is not None:
        random_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(random_state)
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)
    else:
        yield


def _as_list(datasets):
    if isinstance(datasets, list):
        return datasets
    if isinstance(datasets, dict):
        return [d for d in datasets.values() if d is not None]
    raise ValueError


def import_item(path: str, name: str) -> Any:
    """
    Import a python item. Example: import_item("path.to.file", "MyClass") -> MyClass
    """
    return getattr(importlib.import_module(path), name)


def create_dataset(path: str, *args, **kwargs) -> Any:
    """
    Create a dataset. Requires the file to contain a "create_dataset" function.
    """
    return import_item(path, "create_dataset")(*args, **kwargs)


def shift_seed(seed: Optional[int], shift: int) -> Optional[int]:
    """
    Shift the seed by a given amount. Or return None if seed is None.
    """
    return (seed + shift) if seed is not None else None
