from itertools import chain
from multiprocessing import Pool
from pyarrow.parquet import ParquetFile
from torch.utils.data import IterableDataset
from typing import List, Literal, Optional, Union
from pyarrow.fs import HadoopFileSystem, LocalFileSystem

from .utils.hdfs_utils import listdir_with_metafile, exists
from .parquet_utils import (
    get_portion_for_worker_only,
    get_random_for_rank_and_worker,
    get_portion_for_rank_and_worker,
    get_worker_id
)

def hack_s_data(filepath):
    if "vae-1011" in filepath:
        return filepath.replace("byte_data_tt_m/VGFM/data/packed/vae-1011", "byte_icvg_aigc_cp/user/video/temp/19900101/packed/vae-1011")
    elif "dit-1126" in filepath:
        return filepath.replace("byte_data_tt_m/user/sheng.bi/vgfm/packed/dit-1126", "byte_icvg_aigc_cp/user/video/temp/19900101/dit-1126")
    else:
        return filepath

def get_filesystem(path: str) -> Union[LocalFileSystem, HadoopFileSystem]:
    """
    Get filesystem based on the path.
    """
    if path.startswith("hdfs://"):
        return HadoopFileSystem.from_uri(path)
    else:
        return LocalFileSystem()


def read_metadata(
    path: str,
):
    fs = get_filesystem(path)
    with ParquetFile(path, filesystem=fs) as file:
        metadata = file.metadata
    return metadata


class ParquetDataset(IterableDataset):
    """
    Parquet dataset.

    Arguments:
        path: a directory path that contains *.parquet files.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy. Split by *.parquet file or by row groups in each file.
        force_partition: if True, raise error if partition is indivisible.
        num_parallel_files: number of parallel files to read.
        infinite: If True, data will be returned infinitely.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        seed: Optional[int],
        partition: Literal["file", "group", "dump"] = "file",
        force_partition: bool = False,
        num_parallel_files: int = 8,
        infinite: bool = True,
        path_mode: Literal["dir", "file"] = "dir",
        shuffle: bool = True,
        columns: Optional[List[str]] = None,
        plugin_caption_path="",
        dump_path = "",
    ):
        assert partition in ["file", "group", "dump"]
        assert path_mode in ["dir", "file"]

        # Save settings.
        self.seed = seed
        self.infinite = infinite
        self.partition = partition
        self.force_partition = force_partition
        self.num_parallel_files = num_parallel_files
        self.shuffle = shuffle
        self.columns = columns

        # List file paths.
        filepaths = path if isinstance(path, list) else [path]
        if path_mode == "dir":
            filepaths = map(listdir_with_metafile, filepaths)
            filepaths = chain(*filepaths)
        filepaths = filter(lambda path: path.endswith(".parquet"), filepaths)
        
        filepaths = [hack_s_data(path) for path in filepaths]
        # if len(filepaths)<10:
        #     filepaths=filepaths*70
        # else:
        #     filepaths=filepaths
        filepaths = sorted(filepaths)
        assert len(filepaths) > 0

        # Create file readers.
        self.filereaders = []
        for path in filepaths:
            test_plugin_caption_path = plugin_caption_path.rstrip('/')+"/"+path.split('/')[-1] if plugin_caption_path != "" else ""
            test_dump_path = dump_path.rstrip('/')+"/"+path.split('/')[-1] if dump_path != "" else "",
        self.filereaders = [
            ParquetFileReader(
                path=path,
                seed=seed,
                partition=partition,
                force_partition=force_partition,
                shuffle=shuffle,
                columns=columns,
                plugin_caption_path=plugin_caption_path.rstrip(
                    '/')+"/"+path.split('/')[-1] if plugin_caption_path != "" else "",
                dump_path = dump_path.rstrip(
                    '/')+"/"+path.split('/')[-1] if dump_path != "" else "",
            )
            for path in filepaths
        ] 

    # Please don't use a fake __len__(self)! Try making other functions e.g. get_size() instead.
    def __len__(self):
        if not hasattr(self, "count"):
            # Calculate an approximate dataset item count.
            # We open 5 files and compute the average items per file.
            # Then we use this to approximate total dataset item count.

            with Pool(1) as pool:
                counts = pool.map(len, self.filereaders[:5])
            self.count = int(sum(counts) / len(counts) * len(self.filereaders))
        return self.count

    def __iter__(self):
        epoch = 0
        filereaders = self.filereaders
        random = get_random_for_rank_and_worker(self.seed)

        # Partition by files if needed.
        if self.partition == "file":
            filereaders = get_portion_for_rank_and_worker(
                filereaders, self.force_partition)
       # print(get_worker_id(),len(filereaders),len(self.filereaders))

        while True:
            # Initialize filereaders iterators.
            iterators = [reader.__iter__(epoch=epoch)
                         for reader in filereaders]
            if self.shuffle:
                random.shuffle(iterators)

            # Yield samples.
            bad_file_count = 0
            max_bad_file_count = len(iterators)
            while any(iterators):
                if self.shuffle:
                    iterator = random.choice(
                        iterators[: self.num_parallel_files])
                else:
                    iterator = iterators[0]
                try:
                    result = next(iterator)
                    if result == "invalid parquet file!":
                        print("encounter data-caption file problem, removing iterator")
                        iterators.remove(iterator)
                        bad_file_count += 1
                        if bad_file_count >= max_bad_file_count:
                            bad_file_count = 0
                            yield "max_bad_file_count_reached"
                        continue
                    else:
                        yield result
                except StopIteration:
                    iterators.remove(iterator)

            # Break after the first epoch if not infinite.
            if not self.infinite:
                break

            # Increment epoch.
            epoch += 1


class ParquetFileReader:
    """
    Read a single *.parquet file.

    Arguments:
        path: a *.parquet file path.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy.
        force_partition: if True, raise error if partition is indivisible.
    """

    def __init__(
        self,
        path: str,
        seed: Optional[int],
        partition: bool,
        force_partition: bool,
        shuffle: bool,
        columns: Optional[List[str]],
        plugin_caption_path: str,
        dump_path: str,
    ):
        self.path = path
        self.seed = seed
        self.partition = partition
        self.force_partition = force_partition
        self.shuffle = shuffle
        self.columns = columns
        self.plugin_caption_path = plugin_caption_path
        self.dump_path = dump_path

    def __len__(self):
        fs = get_filesystem(self.path)

        with ParquetFile(self.path, filesystem=fs) as file:
            return file.metadata.num_rows

    def __iter_parallel(self, epoch):
        fs = get_filesystem(self.path)
        if not exists(self.path) or not exists(self.plugin_caption_path) or not exists(self.dump_path):
            # return and make the iter empty
            print(f"parallel loading warning: {self.path} or {self.plugin_caption_path} not exists, return empty iter")
            yield "invalid parquet file!"
        
        try:
            
            with ParquetFile(self.path, filesystem=fs) as file, \
                    ParquetFile(self.plugin_caption_path, filesystem=fs) as plugin_caption, \
                        ParquetFile(self.dump_path, filesystem=fs) as dump_file:
                # List all groups.
                groups = list(range(file.num_row_groups))

                # Partition groups if needed.
                if self.partition == "group":
                    groups = get_portion_for_rank_and_worker(
                        groups, self.force_partition)
                elif self.partition == "dump":
                    groups = get_portion_for_worker_only(groups)

                if self.shuffle:
                    # Shuffle groups
                    seed = (self.seed + epoch) if self.seed is not None else None
                    get_random_for_rank_and_worker(seed).shuffle(groups)

                # Iteration over all samples from all row groups.
                for group in groups:
                    iter_main = file.iter_batches(
                        batch_size=1, row_groups=[group], columns=self.columns,
                        use_threads=False,)
                    iter_plugin_caption = plugin_caption.iter_batches(
                        batch_size=1, row_groups=[group], columns=None,
                        use_threads=False,)
                    iter_dump = dump_file.iter_batches(
                        batch_size=1, row_groups=[group], columns=None,
                        use_threads=False,)

                    # Zip the two iterators to read rows "in parallel"
                    for main_batch, caption_batch, dump_batch in zip(iter_main, iter_plugin_caption, iter_dump):
                        # Convert each single-row batch to a dict
                        main_batch_dict = main_batch.to_pandas().iloc[0].to_dict()
                        caption_batch_dict = caption_batch.to_pandas(
                        ).iloc[0].to_dict()
                        dump_batch_dict = dump_batch.to_pandas().iloc[0].to_dict()
                        assert caption_batch_dict['uttid'] == main_batch_dict[
                            'uttid'] and caption_batch_dict['uttid'] == dump_batch_dict['uttid'], f"uttid not match {caption_batch_dict['uttid']} vs {main_batch_dict['uttid']}"
                        main_batch_dict.update(caption_batch_dict)
                        main_batch_dict.update(dump_batch_dict)
                        yield main_batch_dict
        except Exception as e:
            print(f"parallel loading error: {e}")
            return

    def __iter_normal(self, epoch):
        fs = get_filesystem(self.path)

        with ParquetFile(self.path, filesystem=fs) as file:
            # List all groups.
            groups = list(range(file.num_row_groups))

            # Partition groups if needed. "file"
            if self.partition == "group":
                groups = get_portion_for_rank_and_worker(
                    groups, self.force_partition)
            elif self.partition == "dump":
                groups = get_portion_for_worker_only(groups)

            if self.shuffle:
                # Shuffle groups
                seed = (self.seed + epoch) if self.seed is not None else None
                get_random_for_rank_and_worker(seed).shuffle(groups)

            # Iteration over all samples from all row groups.
            for group in groups:
                for sample in file.iter_batches(
                    batch_size=1, row_groups=[group], columns=self.columns,
                    use_threads=False,
                ):
                    yield sample.to_pandas().iloc[0].to_dict()

    def __iter__(self, epoch=0):
        
        if self.plugin_caption_path != "":
            return self.__iter_parallel(epoch)
        else:
            return self.__iter_normal(epoch)