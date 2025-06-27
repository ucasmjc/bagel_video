# -------------------------------------
# Modified by: Jacob Zhiyuan Fang
# Date: 2024/09/10
# Email: jacob.fang@bytedance.com
# Author: Xun Guo
# Email: guoxun.99@bytedance.com
# Date: 2024/05/29
# -------------------------------------

import os
import json
import time
import random
import subprocess

import torch
import numpy as np
# import tensorflow as tf
import multiprocessing as mp
import torchvision.transforms as transforms

from .parquet_dataset.parquet_utils import get_random_for_rank_and_worker, get_portion_for_rank_and_worker
from typing import List, Tuple
# from dataloader import KVReader
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image


class T2VHDFSDataset(Dataset):
    def __init__(self,
                 json_path,
                 sample_size=256,
                 sample_stride=4,
                 sample_n_frames=16,
                 is_image=False,
                 pick=False,
                 fps=24,
                 shuffle=True,
                 infinite=True,
                 ):
        super().__init__()

        with open(json_path, 'r') as jsonfile:
            self.dataset = json.load(jsonfile)
        assert type(
            self.dataset) == list, "The annotation file should contain a list !!!"

        # IMPORTANT: Prevent tf load tensor to GPU.
        tf.config.set_visible_devices([], 'GPU')
        self._context_features = {
            'title': tf.io.FixedLenFeature([], dtype=tf.string)}
        self._sequence_features = {
            'data': tf.io.FixedLenSequenceFeature([], dtype=tf.string)}

        self.length = len(self.dataset)
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.is_image = is_image
        self.pick = pick
        self.num_parallel_reader = 32
        self.shuffle = shuffle
        self.infinite = infinite
        if sample_size == -1:  # if sample_size is None, using Identity transformation
            self.pixel_transforms = transforms.Compose([
                transforms.Lambda(lambda x: x)
            ])
        else:
            sample_size = tuple(sample_size) if not isinstance(
                sample_size, int) else (sample_size, sample_size)
            self.pixel_transforms = transforms.Compose([
                transforms.Resize(sample_size[0]),
                transforms.CenterCrop(sample_size),
            ])
        self.fps = fps

    def __iter__(self):
        if self.shuffle:
            get_random_for_rank_and_worker(None).shuffle(self.dataset)
        part_dataset = get_portion_for_rank_and_worker(self.dataset)
        while True:
            if self.shuffle:
                get_random_for_rank_and_worker(None).shuffle(part_dataset)
            for idx in range(len(part_dataset)):
                try:
                    to_return = self.__getitem_impl__(idx)
                    yield to_return
                except (RuntimeError, ValueError):
                    print('Appearing HDFS iops error setting src img \n' * 5)
                    # idx = random.sample(range(self.length), 1)[0]
            if not self.infinite:
                break

    def __len__(self):
        return len(self.dataset)

    def decode_image(self, raw_data):
        return tf.image.decode_jpeg(raw_data, channels=3, dct_method='INTEGER_ACCURATE').numpy()

    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        video_name, index_file, caption = video_dict[
            'video_name'], video_dict['index_file'], video_dict['caption']
        reader = KVReader(index_file, self.num_parallel_reader)
        keys = reader.list_keys()
        assert video_name in keys, "video file not in this index file !!!"
        values = reader.read_many([video_name])[0]

        # Decode record
        contexts, sequences = tf.io.parse_single_sequence_example(
            serialized=values,
            context_features=self._context_features,
            sequence_features=self._sequence_features)

        # Raw frames data
        raw_frames = sequences['data']
        del reader
        video_length = len(raw_frames)

        # Sample frames
        if not self.is_image:

            # Jacob Sep 17th: If sample frames > video frames, we drop this video
            if (self.sample_n_frames - 1) * self.sample_stride + 1 > video_length:
                return None, None
            clip_length = min(
                video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(
                start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        # Decode frames
        pixel_values = []
        for idx in batch_index:
            frame = raw_frames[idx]
            frame = self.decode_image(frame)
            frame = torch.as_tensor(frame).float().permute(2, 0, 1)
            frame = (frame - 127.5) / 127.5
            pixel_values.append(frame)

        if self.is_image:
            pixel_values = pixel_values[0]

        pixel_values = torch.stack(pixel_values, dim=0)
        return pixel_values, caption

    def __getitem_impl__(self, idx, candidate=None):
        # To avoid bad videos, we retry if there is an Exception.
        # By default the size of videos are all 512, 910 so no need filter.
        if candidate is None:
            candidate = list(range(self.length))
        while True:
            try:
                pixel_values, caption = self.get_batch(idx)

                if pixel_values is None:
                    # restart
                    idx = random.sample(candidate, 1)[0]
                else:
                    # end the iteration
                    break
            except Exception as e:
                print(f"VideoTextPairDataset got unexpected exception: {e}")
                idx = random.sample(candidate, 1)[0]
        pixel_values = self.pixel_transforms(pixel_values)

        # pixel_values in shape of Frames x channel x H x W
        sample = dict(
            mp4=pixel_values,
            txt=caption,
            num_frames=self.sample_n_frames,
            fps=self.fps,
        )

        return sample

    @classmethod
    def create_dataset_function(cls, json_path, args, **kwargs):
        return cls(json_path=json_path, **kwargs)


# Dataset unit test checking how many videos are not preferred
if __name__ == "__main__":
    dataset = T2VHDFSDataset(
        json_path="/mnt/bn/icvg/video_gen/captions/pond5_res/pond5_data_res_human.json",
        sample_size=512,
        sample_stride=4,
        sample_n_frames=49,
        is_image=False,
        pick=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=1)
    for idx, batch in enumerate(dataloader):
        if idx % 100 == 0:
            breakpoint()
