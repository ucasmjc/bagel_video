import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_tensor
# from nebudata import refds
from .parquet_dataset.utils import hdfs_utils
from .parquet_dataset.parquet_utils import get_random_for_rank_and_worker, get_portion_for_rank_and_worker, get_worker_id, get_worker_count
from .parquet_dataset.utils.distributed_utils import get_data_parallel_rank, get_data_parallel_world_size
import random
from PIL import Image
import copy
import numpy as np
import json
from multiprocessing import Pool
import traceback


try:
    import av
    import io
    pyav_enabled = True
except:
    pyav_enabled = False

try:
    import imageio.v3 as iio
    imageio_enabled = True
except:
    imageio_enabled = False


def get_length(path, ignore_prefixes):
    dataset = refds.RefDataset(
        path, ignore_prefixes=ignore_prefixes)
    return dataset.rank_total


def get_length_subprocess(path, ignore_prefixes):
    with Pool(1) as pool:
        counts = pool.apply(
            get_length, args=(path, ignore_prefixes, ))
    return counts


def sampling(video_length, sample_n_frames, sample_stride, skip_start_end=10):
    # Jacob Sep 17th: If sample frames > video frames, we drop this video
    if (sample_n_frames - 1) * sample_stride + 1 > (video_length - skip_start_end * 2):
        return None
    clip_length = min(
        video_length, (sample_n_frames - 1) * sample_stride + 1)
    start_idx = random.randint(
        skip_start_end, video_length - clip_length - skip_start_end)
    batch_index = np.linspace(
        start_idx, start_idx + clip_length - 1, sample_n_frames, dtype=int)
    return batch_index


class AIPVideoDataset(Dataset):
    def __init__(self,
                 path,
                 sample_size=256,
                 sample_stride=4,
                 sample_n_frames=16,
                 caption_key='caption',
                 caption_path="",
                 fps=24,
                 shuffle=True,
                 infinite=True,
                 parquet_batch=128,
                 video_toskey='clip_toskey',
                 bytes_key='bytes',
                 ignore_prefixes=None,
                 decode_backend='pyav',
                 force_partition=False,
                 data_world_size=10000, # TODO: can be dynamic
                 local_cache_prefix='',
                 ):
        self.sample_size = sample_size
        assert self.sample_size == -1, \
            "only support original size, consider using sample_size==-1 for bucketing"
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.shuffle = shuffle
        self.infinite = infinite # this doesn't work, the dataset is always infinite
        self.fps = fps
        self.force_partition = force_partition
        self.data_world_size = data_world_size
        self.state_dict = {'data_world_size': self.data_world_size, 'seen_times': [0 for _ in range(self.data_world_size)]}
        self.remaining_ranks = []
        self.local_cache_prefix = local_cache_prefix

        self.path = path
        self.parquet_batch = parquet_batch
        self.video_toskey = video_toskey
        self.caption_key = caption_key  # the key used to store caption
        self.bytes_key = bytes_key  # the key used to store real bytes
        self.ignore_prefixes = ignore_prefixes
        self.decode_backend = decode_backend

        self.total_length = None
        # read caption json file from caption_path seperately, for Seed V2 dataset
        self.caption_data = None
        if caption_path != "":
            # with open(caption_path, 'r') as f:
            if caption_path.startswith("hdfs"):
                caption_path = hdfs_utils.download(caption_path, './')
            with open(caption_path, 'r') as f:
                caption_data = json.load(f)
            caption_data = json.loads(hdfs_utils.read(caption_path))
            self.total_length = len(caption_data)
            self.caption_data = {item['uttid']: item[self.caption_key]
                                 for item in caption_data}

        if self.decode_backend == 'imageio':
            assert imageio_enabled, 'failed to install imageio'
        elif self.decode_backend == 'pyav':
            assert pyav_enabled, 'failed to install pyav'

    def __iter__(self):
        rank = get_data_parallel_rank()
        world_size = get_data_parallel_world_size()
        worker_id = get_worker_id()
        worker_count = get_worker_count()
        overall_workers = world_size * worker_count

        self.local_cache_path = f'{self.local_cache_prefix}_{rank}_{worker_id}.txt'
        refs = [(self.video_toskey, self.bytes_key)
                ] if self.video_toskey != '' else []

        worker_ranks = get_portion_for_rank_and_worker(self.remaining_ranks, allow_empty=True)

        while True:
            if self.shuffle:
                get_random_for_rank_and_worker(None).shuffle(worker_ranks)

            for rank in worker_ranks:
                with open(self.local_cache_path, 'a') as f:
                    f.write(f'{rank}\n')
                filereader = refds.RefDataset(self.path, ignore_prefixes=self.ignore_prefixes, world_size=self.data_world_size, rank=rank)
                for batch in filereader.iter_batches(batch_size=self.parquet_batch, refs=refs):
                    actual_size = len(batch[self.bytes_key])
                    columns = [col for col in batch.column_names]
                    for i in range(actual_size):
                        params_dict = {col: batch[col]
                                       [i].as_py() for col in columns}
                        if self.caption_data is not None:
                            # if we have caption_data, use it to replace caption
                            uttid = params_dict['uttid']
                            if uttid not in self.caption_data:
                                continue
                            params_dict[self.caption_key] = self.caption_data[uttid]
                        frames, metadata = self._data_process(params_dict)
                        if frames is None:
                            continue
                        yield self._pack_frames(frames, metadata)

            overall_ranks = []
            while len(overall_ranks) < overall_workers:
                overall_ranks += list(range(self.data_world_size))
            worker_ranks = get_portion_for_rank_and_worker(overall_ranks, force=True)

    def _pack_frames(self, frames, metadata):
        tensor_frames = []
        for frame in frames:
            frame = to_tensor(frame)
            tensor_frames.append(frame)
        tensor_frames = torch.stack(tensor_frames)
        # make value from -1.0 to 1.0
        pixel_values = tensor_frames * 2.0 - 1.0
        item = dict(
            mp4=pixel_values,
            txt=metadata[self.caption_key],
            num_frames=self.sample_n_frames,
            fps=metadata.get('fps', self.fps),
        )
        return item

    def _data_process(self, params):
        tosbytes = params[self.bytes_key]
        del params[self.bytes_key]  # remove the bytes key
        metadata = copy.deepcopy(params)
        try:
            frames = self._bytes_to_PILs(tosbytes)
        except:
            print("data error: ", metadata)
            traceback.print_exc()
            return None, None
        if frames is None:
            return None, None
        return frames, metadata

    def _bytes_to_PILs(self, video_bytes):
        if self.decode_backend == 'imageio':
            raw_frames = iio.imread(
                video_bytes, index=None, format_hint=".mp4")
            video_length = raw_frames.shape[0]
            video_idxs = sampling(
                video_length, self.sample_n_frames, self.sample_stride)
            if video_idxs is None:
                return None
            frames = []
            for i in video_idxs:
                frames.append(Image.fromarray(raw_frames[i], 'RGB'))

        elif self.decode_backend[:4] == 'pyav':
            file_io = io.BytesIO(video_bytes)
            container = av.open(file_io)
            stream = container.streams.video[0]
            video_length = container.streams.video[0].frames
            video_idxs = sampling(
                video_length, self.sample_n_frames, self.sample_stride)
            if video_idxs is None:
                return None
            frames_sorted = []
            key_frame_idxs = []

            # Get keyframe without decoding
            stream.codec_context.skip_frame = "NONKEY"
            for packet in container.demux(stream):
                if packet.is_keyframe:
                    frame_idx = int(
                        packet.pts * stream.time_base * stream.average_rate + 1e-6)
                    key_frame_idxs.append(frame_idx)

            # Reset for decode any frames
            stream.codec_context.skip_frame = "DEFAULT"

            # Sort the frames under the cases that frames are unsorted
            video_idxs_sort_idx = np.argsort(np.array(video_idxs))
            video_idxs_sorted = np.array(video_idxs)[video_idxs_sort_idx]

            # The keyframe assignment for each frame
            keyframe_assignment = np.clip(((np.array(video_idxs_sorted)[
                                          None] - np.array(key_frame_idxs)[:, None]) > 0).sum(0) - 1, 0, None)

            time_base = container.streams.video[0].time_base
            framerate = container.streams.video[0].average_rate

            previous_keyframe_assigment = -1
            for ii, frame_num in enumerate(video_idxs_sorted):
                this_assignment = keyframe_assignment[ii]

                # Reseek only if when the keyframe are changed, avoid redecode frames
                if this_assignment != previous_keyframe_assigment:
                    # Calculate the timestamp for the desired frame
                    frame_container_pts = int(
                        ((key_frame_idxs[this_assignment] + 1) / framerate) / time_base)

                    # Seek to the closest keyframe before the desired timestamp
                    container.seek(frame_container_pts, backward=True,
                                   stream=container.streams.video[0])
                    previous_keyframe_assigment = this_assignment

                    # Record where we start, for debug only
                    # start_idx = key_frame_idxs[this_assignment]

                previous_frame_idx = -1
                while previous_frame_idx < frame_num:
                    frame = next(container.decode(video=0))
                    previous_frame_idx = int(
                        frame.pts * stream.time_base * stream.average_rate + 1e-6)
                # Debug code to check if always get the desired frame
                # print(f"start={start_idx}, source={previous_frame_idx}, target={frame_num}, ")
                frames_sorted.append(frame.to_image())

            # Recollect to the original sorts => inverse sort
            frames = [None for _ in range(len(video_idxs))]
            for i, idx in enumerate(video_idxs_sort_idx):
                frames[idx] = frames_sorted[i]
        elif self.decode_backend == 'image_bytes':
            video_length = len(video_bytes)
            video_idxs = sampling(
                video_length, self.sample_n_frames, self.sample_stride)
            if video_idxs is None:
                return None
            frames = []
            for idx in video_idxs:
                frame_byte = video_bytes[idx]
                with Image.open(io.BytesIO(frame_byte)) as frame:
                    frame = frame.convert("RGB")
                frames.append(frame)

        return frames

    def load_state_dict(self, state_dict):
        # get remaining ranks
        if 'data_world_size' not in self.state_dict:
            print('[AIP_dataset] no state_dict; init data loading')
        elif self.data_world_size != self.state_dict['data_world_size']:
            print('[AIP_dataset] inconsistent data_world_size, init data loading')
        elif self.state_dict['data_world_size'] != len(self.state_dict.get('seen_times', [])):
            print('[AIP_dataset] corrupted state_dict; init data loading')
        else:
            #this has to be the same across all workers
            self.state_dict = state_dict
            print('[AIP_dataset] resume data loading from state_dict')
            max_times = max(self.state_dict['seen_times'])
            for rank, times in enumerate(self.state_dict['seen_times']):
                for _ in range(max_times-times):
                    self.remaining_ranks.append(rank)

    def __len__(self):
        if self.total_length is None:
            counts = get_length_subprocess(self.path, self.ignore_prefixes)
            self.total_length = counts
        return self.total_length

    @ classmethod
    def create_dataset_function(cls, data_path, args, **kwargs):
        return cls(path=data_path, **kwargs)
