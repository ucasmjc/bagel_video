# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import json
import pyarrow.parquet as pq
import random
from PIL import Image
import torch.distributed as dist
from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .parquet_utils import get_parquet_data_paths, init_arrow_hdfs_fs
import glob
Image.MAX_IMAGE_PIXELS = 20_000_000
from datasets import load_dataset

class WebIterableDataset(DistributedIterableDataset):
    def __init__(
        self, dataset_name, transform, tokenizer, tar_dir_list, 
        local_rank=0, world_size=1, num_workers=8, data_status=None,
    ):
        """
        data_dir_list: list of data directories contains parquet files
        num_used_data: list of number of sampled data paths for each data directory
        """
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        #self.data_paths = self.get_data_paths(data_dir_list, num_used_data)
        self.data_paths = self.get_tar_paths(tar_dir_list)
        self.set_epoch()

    def get_tar_paths(self,data_dir_list,rank=0, world_size=1):
        num_data_dirs = len(data_dir_list)
        if world_size > 1:
            chunk_size = (num_data_dirs + world_size - 1) // world_size
            start_idx = rank * chunk_size
            end_idx = min(start_idx + chunk_size, num_data_dirs)
            local_data_list = [data_dir_list[idx] for idx in range(start_idx,end_idx)]
        else:
            local_data_list = [data_dir_list[idx] for idx in range(num_data_dirs)]

        if world_size > 1:
            gather_list = [None] * world_size
            dist.all_gather_object(gather_list, local_data_list)

            combined_chunks = []
            for chunk_list in gather_list:
                if chunk_list is not None:
                    combined_chunks.extend(chunk_list)
        else:
            combined_chunks = local_data_list

        return combined_chunks
    def read_jsonfile(self, jsonfile: str) -> dict:
        with open(jsonfile, 'r', encoding='utf-8') as f:
            return json.load(f)

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            parquet_start_id = self.data_status[worker_id][0]
            row_group_start_id = self.data_status[worker_id][1]
            row_start_id = self.data_status[worker_id][2] + 1
        else:
            parquet_start_id = 0
            row_group_start_id = 0
            row_start_id = 0
        transform_stride = self.transform.stride

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming data at parquet#{parquet_start_id}, row#{row_start_id}"
        )
        #print(data_paths_per_worker)
        while True:
            data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            for idx, file_path in enumerate(data_paths_per_worker_, start=parquet_start_id):
                single_data = load_dataset("webdataset", data_files=file_path, cache_dir='/mnt/localdisk/hongwei/Bagel/.cache', split="train", num_proc=1)
                for row_id in range(row_start_id,len(single_data)):
                    
                    try:
                        current_row_id=row_id
                        row=single_data[current_row_id]
                        num_tokens = 0
                        image = pil_img2rgb(row["jpg"])
                    except Exception as e:
                        print(f'Error: {e} in {current_row_id},{row_start_id}')
                        continue
                    image_tensor = self.transform(image)
                    height, width = image_tensor.shape[1:]
                    num_tokens += width * height // transform_stride ** 2
                    prompt = row["txt"]

                    caption_token = self.tokenizer.encode(prompt)

                    sequence_plan, text_ids_list = [], []
                    text_ids = caption_token
                    num_tokens += len(caption_token)
                    text_ids_list.append(text_ids)
                    sequence_plan.append({
                        'type': 'text',
                        'enable_cfg': 1,
                        'loss': 0,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })
                
                    sequence_plan.append({
                        'type': 'vae_image',
                        'enable_cfg': 0,
                        'loss': 1,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })

                    sample = dict(
                        image_tensor_list=[image_tensor], #3*512*512
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": [idx,0,current_row_id],
                            "worker_id": worker_id,
                            "dataset_name": self.dataset_name,
                        }
                    )
                    yield sample
                row_start_id=0
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
