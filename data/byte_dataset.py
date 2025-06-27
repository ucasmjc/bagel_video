import os
import glob
from copy import deepcopy
from typing import Dict
import importlib
import random
from torch.utils.data import ChainDataset, IterableDataset, Dataset
import torchvision.transforms as transforms
from torch.utils.data._utils.collate import default_collate
from torchvision.transforms import functional as F
import concurrent.futures

from .byte_data import AIPVideoDataset
from .interleave_datasets import UnifiedEditIterableDataset
from .distributed_iterable_dataset import DistributedIterableDataset
def collate_fn_map(samples):
    """
    Custom collate function that processes a list of samples into a batch.
    """
    if type(samples) is list and type(samples[0]) is list:
        samples = samples[0]  # remove the first batch, as it is always 1
        if isinstance(samples[0], dict):
            return {key: default_collate([sample[key] for sample in samples]) for key in samples[0]}
        raise NotImplementedError
    else:
        return default_collate(samples)

from torch.utils.data import IterableDataset
class CollectionDataset(IterableDataset):
    def __init__(
        self,
        train_data: list[str],
        train_data_weights: list[int | float],
        dataset_collections: Dict[str, Dict],
        batch_size=1,
        image_batch_size=48,
        enable_bucket=False,
        infinite=True,
        shuffle=True,
        local_cache='', # this should be a ByteNAS path
        data_cache_prefix={'AIPVideoDataset': 'aip_dataset_cache'},
        **kwargs
    ):
        #super().__init__(kwargs.get("dataset_name",None),kwargs.get("local_rank",0),kwargs.get("world_size",8),kwargs.get("num_workers",1) )
        # prepare for bucketings
        self.data_status = kwargs.get("data_status",None)
        self.tokenizer=kwargs.get("tokenizer",None)
        self.enable_bucket = enable_bucket
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size #8

        self.buckets = {}
        self.buckets_transform = {}
        self.resolutions = set()
        if not self.enable_bucket:
            assert batch_size == 1, "if not enable_bucket, batch_size must be 1"

        self.train_data_weights = train_data_weights

        self.dataset_list = []
        self.dataset_names = []
        self.image_dataset_names = []
        self.dataset_collections = dataset_collections
        self.dataset_to_aspect_ratios = {}
        self.init_state_dict = {}
        self.local_cache_prefix_list = []
        for data_name in train_data:
            if data_name not in dataset_collections:
                print(f'{data_name} not in dataset collections')
                return
            self.dataset_config = dataset_collections[data_name]
            aspect_ratios = self.dataset_config['aspect_ratios']
            self.dataset_to_aspect_ratios[data_name] = aspect_ratios
            self.add_aspect_ratios(aspect_ratios)

            module, cls = self.dataset_config['target'].rsplit(".", 1)
            data_class = getattr(
                importlib.import_module(module, package=None), cls)
            if cls == 'T2IHDFSDataset':
                self.image_dataset_names.append(data_name)

            if cls in data_cache_prefix:
                data_cache = os.path.join(local_cache, data_cache_prefix[cls])
                os.makedirs(data_cache, exist_ok=True)
                local_cache_prefix = os.path.join(data_cache, data_name)
                self.clean_cache(local_cache_prefix)
                self.dataset_config['params']['local_cache_prefix'] = local_cache_prefix
                self.local_cache_prefix_list.append(local_cache_prefix)
            else:
                self.local_cache_prefix_list.append('')
            dataset = data_class.create_dataset_function(
                self.dataset_config['path'], None, **self.dataset_config['params'])
            if cls == 'AIPVideoDataset':
                self.init_state_dict[data_name] = dataset.state_dict
            self.dataset_list.append(dataset)
            self.dataset_names.append(data_name)
        self.length = sum([len(dataset) for dataset in self.dataset_list])
        self.dataset_iter_list = [iter(dataset) for dataset in self.dataset_list]
        #self.data_paths=self.dataset_list
        # if len(self.dataset_iter_list)<10:
        #     self.data_paths=self.dataset_iter_list*70
        # else:
        #     self.data_paths=self.dataset_iter_list
    #     self.set_epoch()
    #     # import pdb
    #     # pdb.set_trace()

    # def set_epoch(self, seed=42):
    #     if self.data_paths is None:
    #         return
    #     data_paths=self.data_paths
    #     self.rng.seed(seed)
    #     self.rng.shuffle(data_paths)

    #     num_files_per_rank = len(data_paths) // self.world_size
    #     local_start = self.local_rank * num_files_per_rank
    #     local_end = (self.local_rank + 1) * num_files_per_rank
    #     self.num_files_per_rank = num_files_per_rank
    #     self.data_paths_per_rank = data_paths[local_start:local_end]
    def add_aspect_ratios(self, aspect_ratios):
        for key in aspect_ratios.keys():
            self.buckets[key] = []

        for key, sample_size in aspect_ratios.items():
            sample_size = tuple(sample_size)
            self.buckets_transform[key] = transforms.Compose([
                transforms.Resize(min(sample_size[0], sample_size[1])), # fix when height > width
                transforms.CenterCrop(sample_size),
            ])
        for h, w in aspect_ratios.values():
            self.resolutions.add((49, h, w))

    def get_bucket_id(self, item, dataset_name):
        """
        for large resolution data, we may have multiple bucket ids
        """
        frames, c, H, W = item['mp4'].shape
        ratio = float(H) / float(W)

        ratio_strategy = self.dataset_collections[dataset_name]['ratio_strategy']
        ratios = self.dataset_to_aspect_ratios[dataset_name]
        if ratio_strategy == 'random':
            bucket_id = random.choice(list(ratios.keys()))
        elif ratio_strategy == 'closest':
            bucket_id = min(ratios.items(),
                            key=lambda r: abs(float(r[1][0]) / float(r[1][1]) - ratio))[0]
        else:
            raise f"ratio_strategy {ratio_strategy} not support ..."

        return bucket_id

    def __len__(self):
        return self.length

    def crop_and_resize(self, image, h_prime, w_prime):
        """
        Crop and resize a 4D tensor image.

        Args:
            image: The input 4D tensor image of shape (frame, channel, h, w).
            h_prime: Desired height of the cropped image.
            w_prime: Desired width of the cropped image.

        Returns:
            The cropped and resized 4D tensor image.
        """
        frames, channels, h, w = image.shape
        aspect_ratio_original = h / w
        aspect_ratio_target = h_prime / w_prime

        if aspect_ratio_original >= aspect_ratio_target:
            new_h = int(w * aspect_ratio_target)
            top = (h - new_h) // 2
            bottom = top + new_h
            left = 0
            right = w
        else:
            new_w = int(h / aspect_ratio_target)
            left = (w - new_w) // 2
            right = left + new_w
            top = 0
            bottom = h
        # print(f"left {left}, right {right}, top {top}, bottom {bottom}")
        # Crop the image
        cropped_image = image[:, :, top:bottom, left:right]
        # Resize the cropped image
        resized_image = F.resize(cropped_image, (h_prime, w_prime))
        return resized_image

    def put_to_bucket(self, item, dataset_name):
        bucket_id = self.get_bucket_id(item, dataset_name)
        ori_frams, ori_c, ori_H, ori_W = item['mp4'].shape
        ori_ratio = ori_H / ori_W
        bucket_h, bucket_w = self.dataset_to_aspect_ratios[dataset_name][bucket_id][0], self.dataset_to_aspect_ratios[dataset_name][bucket_id][1]
        bucket_ratio = bucket_h / bucket_w
        # print(f"ori_H {ori_H}, ori_W {ori_W}, ori_ratio {ori_ratio}. bucket_h {bucket_h}, bucket_w {bucket_w}, bucket_ratio {bucket_ratio}")
        item['mp4'] = self.crop_and_resize(item['mp4'], bucket_h, bucket_w)

        frames, c, H, W = item['mp4'].shape
        # rewrite item to the same format as the original dataset
        new_item = {}
        new_item['videos'] = item['mp4']
        new_item['prompts'] = item['txt'] if item['txt'] is not None else "" # check text
        new_item['video_metadata'] = {
            'num_frames': frames,
            'height': H,
            'width': W,
        }
        self.buckets[bucket_id].append(new_item)

        batch = None
        cur_batch_size = self.image_batch_size if bucket_id.startswith("i-") else self.batch_size
        if len(self.buckets[bucket_id]) >= cur_batch_size:
            batch = self.buckets[bucket_id]
            self.buckets[bucket_id] = []
        return batch

    def __iter__(self):
        # data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        # if self.data_status is not None:
        #     parquet_start_id = self.data_status[worker_id][0]
        #     #row_start_id = self.data_status[worker_id][2] + 1
        # else:
        #     parquet_start_id = 0
        #     #row_start_id = 0
        # print(
        #     f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
        #     f"resuming data at parquet#{parquet_start_id}, row# 0"
        # )
        def __native__iter():
            
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.dataset_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                yield next(dataset)

        def __bucket__iter():
            #data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            while True:
                idx=random.choice(range(len(self.dataset_iter_list)))
                single_data=self.dataset_iter_list[idx]
                item = next(single_data)
                #dict_keys(['mp4', 'txt', 'fps', 'num_frames']), 17*3*720*1280;
                batch_data = self.put_to_bucket(item, "seed_i")
                
                if batch_data is not None:
                    data_i=batch_data[0]
                    frames=data_i["videos"]
                    time,channel,height, width = frames.shape
                    frames=frames.permute(1, 0, 2, 3)
                    num_tokens = ((time-1)/4+1)*width * height // 16 ** 2 #vae stride
                    caption_token = self.tokenizer.encode(data_i["prompts"])
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
                        'type': 'vae_video',
                        'enable_cfg': 0,
                        'loss': 1,
                        'special_token_loss': 0,
                        'special_token_label': None,
                    })

                    sample = dict(
                        image_tensor_list=[frames], #3*512*512
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": [idx,0,0],
                            # "worker_id": worker_id,
                            # "dataset_name": self.dataset_name,
                             "worker_id": 0,
                            "dataset_name": 0,
                        }
                    )
                    yield sample
        
        if self.enable_bucket:
            return __bucket__iter()
        else:
            return __native__iter()

    def state_dict(self):
        output_state_dict = deepcopy(self.init_state_dict)
        for dataset_name, local_cache_prefix in zip(self.dataset_names, self.local_cache_prefix_list):
            if dataset_name not in self.init_state_dict:
                continue
            cache_list = glob.glob(f'{local_cache_prefix}*')
            for cache_path in cache_list:
                with open(cache_path, 'r') as f:
                    for l in f.readlines():
                        r = int(l.strip())
                        output_state_dict[dataset_name]['seen_times'][r] += 1
        return output_state_dict

    def load_state_dict(self, state_dict):
        for dataset_name, local_cache_prefix, dataset in zip(self.dataset_names, self.local_cache_prefix_list, self.dataset_list):
            if dataset_name not in state_dict:
                continue
            if dataset_name not in self.init_state_dict:
                continue
            self.clean_cache(local_cache_prefix)
            dataset.load_state_dict(state_dict[dataset_name])
            self.init_state_dict[dataset_name] = dataset.state_dict

    def clean_cache(self, local_cache_prefix):
        for fname in glob.glob(f'{local_cache_prefix}*'):
            try:
                os.remove(fname)
            except OSError:
                pass

    @classmethod
    def create_dataset_function(cls, data, data_weights, **kwargs):
        return cls(data, data_weights, **kwargs)

class CollectionDataset_frame(IterableDataset):
    def __init__(
        self,
        train_data: list[str],
        train_data_weights: list[int | float],
        dataset_collections: Dict[str, Dict],
        batch_size=1,
        image_batch_size=48,
        enable_bucket=False,
        infinite=True,
        shuffle=True,
        local_cache='', # this should be a ByteNAS path
        data_cache_prefix={'AIPVideoDataset': 'aip_dataset_cache'},
        **kwargs
    ):
        #super().__init__(kwargs.get("dataset_name",None),kwargs.get("local_rank",0),kwargs.get("world_size",8),kwargs.get("num_workers",1) )
        # prepare for bucketings
        self.data_status = kwargs.get("data_status",None)
        self.tokenizer=kwargs.get("tokenizer",None)
        self.enable_bucket = enable_bucket
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size #8

        self.buckets = {}
        self.buckets_transform = {}
        self.resolutions = set()
        if not self.enable_bucket:
            assert batch_size == 1, "if not enable_bucket, batch_size must be 1"

        self.train_data_weights = train_data_weights

        self.dataset_list = []
        self.dataset_names = []
        self.image_dataset_names = []
        self.dataset_collections = dataset_collections
        self.dataset_to_aspect_ratios = {}
        self.init_state_dict = {}
        self.local_cache_prefix_list = []
        for data_name in train_data:
            if data_name not in dataset_collections:
                print(f'{data_name} not in dataset collections')
                return
            self.dataset_config = dataset_collections[data_name].copy()
            #抽首帧
            self.dataset_config['params']["video_frame_sampler"]={"type":"first_frame"}
            aspect_ratios = self.dataset_config['aspect_ratios']
            self.dataset_to_aspect_ratios[data_name] = aspect_ratios
            self.add_aspect_ratios(aspect_ratios)

            module, cls = self.dataset_config['target'].rsplit(".", 1)
            data_class = getattr(
                importlib.import_module(module, package=None), cls)

            dataset = data_class.create_dataset_function(
                self.dataset_config['path'], None, **self.dataset_config['params'])
            self.dataset_list.append(dataset)
            self.dataset_names.append(data_name)

        self.length = sum([len(dataset) for dataset in self.dataset_list])
        self.dataset_iter_list = [iter(dataset) for dataset in self.dataset_list]
        #self.data_paths=self.dataset_list
    #     # if len(self.dataset_iter_list)<10:
    #     #     self.data_paths=self.dataset_iter_list*70
    #     # else:
    #     #     self.data_paths=self.dataset_iter_list
    #     self.data_paths=self.dataset_iter_list
    #     self.set_epoch()
    #     # import pdb
    #     # pdb.set_trace()

    # def set_epoch(self, seed=42):
    #     if self.data_paths is None:
    #         return
    #     data_paths=self.data_paths
    #     self.rng.seed(seed)
    #     self.rng.shuffle(data_paths)

    #     num_files_per_rank = len(data_paths) // self.world_size
    #     local_start = self.local_rank * num_files_per_rank
    #     local_end = (self.local_rank + 1) * num_files_per_rank
    #     self.num_files_per_rank = num_files_per_rank
    #     self.data_paths_per_rank = data_paths[local_start:local_end]
    def add_aspect_ratios(self, aspect_ratios):
        for key in aspect_ratios.keys():
            self.buckets[key] = []

        for key, sample_size in aspect_ratios.items():
            sample_size = tuple(sample_size)
            self.buckets_transform[key] = transforms.Compose([
                transforms.Resize(min(sample_size[0], sample_size[1])), # fix when height > width
                transforms.CenterCrop(sample_size),
            ])
        for h, w in aspect_ratios.values():
            self.resolutions.add((49, h, w))

    def get_bucket_id(self, item, dataset_name):
        """
        for large resolution data, we may have multiple bucket ids
        """
        frames, c, H, W = item['mp4'].shape
        ratio = float(H) / float(W)

        ratio_strategy = self.dataset_collections[dataset_name]['ratio_strategy']
        ratios = self.dataset_to_aspect_ratios[dataset_name]
        if ratio_strategy == 'random':
            bucket_id = random.choice(list(ratios.keys()))
        elif ratio_strategy == 'closest':
            bucket_id = min(ratios.items(),
                            key=lambda r: abs(float(r[1][0]) / float(r[1][1]) - ratio))[0]
        else:
            raise f"ratio_strategy {ratio_strategy} not support ..."

        return bucket_id

    def __len__(self):
        return self.length

    def crop_and_resize(self, image, h_prime, w_prime):
        """
        Crop and resize a 4D tensor image.

        Args:
            image: The input 4D tensor image of shape (frame, channel, h, w).
            h_prime: Desired height of the cropped image.
            w_prime: Desired width of the cropped image.

        Returns:
            The cropped and resized 4D tensor image.
        """
        frames, channels, h, w = image.shape
        aspect_ratio_original = h / w
        aspect_ratio_target = h_prime / w_prime

        if aspect_ratio_original >= aspect_ratio_target:
            new_h = int(w * aspect_ratio_target)
            top = (h - new_h) // 2
            bottom = top + new_h
            left = 0
            right = w
        else:
            new_w = int(h / aspect_ratio_target)
            left = (w - new_w) // 2
            right = left + new_w
            top = 0
            bottom = h
        # print(f"left {left}, right {right}, top {top}, bottom {bottom}")
        # Crop the image
        cropped_image = image[:, :, top:bottom, left:right]
        # Resize the cropped image
        resized_image = F.resize(cropped_image, (h_prime, w_prime))
        return resized_image

    def put_to_bucket(self, item, dataset_name):
        bucket_id = self.get_bucket_id(item, dataset_name)
        ori_frams, ori_c, ori_H, ori_W = item['mp4'].shape
        ori_ratio = ori_H / ori_W
        bucket_h, bucket_w = self.dataset_to_aspect_ratios[dataset_name][bucket_id][0], self.dataset_to_aspect_ratios[dataset_name][bucket_id][1]
        bucket_ratio = bucket_h / bucket_w
        # print(f"ori_H {ori_H}, ori_W {ori_W}, ori_ratio {ori_ratio}. bucket_h {bucket_h}, bucket_w {bucket_w}, bucket_ratio {bucket_ratio}")
        item['mp4'] = self.crop_and_resize(item['mp4'], bucket_h, bucket_w)

        frames, c, H, W = item['mp4'].shape
        # rewrite item to the same format as the original dataset
        new_item = {}
        new_item['videos'] = item['mp4']
        new_item['prompts'] = item['txt'] if item['txt'] is not None else "" # check text
        new_item['video_metadata'] = {
            'num_frames': frames,
            'height': H,
            'width': W,
        }
        self.buckets[bucket_id].append(new_item)

        batch = None
        cur_batch_size = self.image_batch_size if bucket_id.startswith("i-") else self.batch_size
        if len(self.buckets[bucket_id]) >= cur_batch_size:
            batch = self.buckets[bucket_id]
            self.buckets[bucket_id] = []
        return batch

    def __iter__(self):
        # data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        # if self.data_status is not None:
        #     parquet_start_id = self.data_status[worker_id][0]
        #     #row_start_id = self.data_status[worker_id][2] + 1
        # else:
        #     parquet_start_id = 0
        #     #row_start_id = 0
        # print(
        #     f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
        #     f"resuming data at parquet#{parquet_start_id}, row# 0"
        # )
        def __native__iter():
            
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.dataset_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                yield next(dataset)

        def __bucket__iter():
            #data_paths_per_worker_ = data_paths_per_worker[parquet_start_id:]
            while True:
                idx=random.choice(range(len(self.dataset_iter_list)))
                single_data=self.dataset_iter_list[idx]
                item = next(single_data)
                #dict_keys(['mp4', 'txt', 'fps', 'num_frames']), 17*3*720*1280;
                batch_data = self.put_to_bucket(item, self.dataset_names[idx])
                if batch_data is not None:
                    data_i=batch_data[0]
                    frames=data_i["videos"]
                    time,channel,height, width = frames.shape
                    frames=frames.permute(1, 0, 2, 3)
                    num_tokens = ((time-1)/4+1)*width * height // 16 ** 2 #vae stride
                    caption_token = self.tokenizer.encode(data_i["prompts"])
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
                        image_tensor_list=[frames], #3*512*512
                        text_ids_list=text_ids_list,
                        num_tokens=num_tokens,
                        sequence_plan=sequence_plan,
                        data_indexes={
                            "data_indexes": [idx,0,0],
                            "worker_id": 0,
                            "dataset_name":"btye",
                        }
                    )
                    yield sample
        
        if self.enable_bucket:
            return __bucket__iter()
        else:
            return __native__iter()

    def state_dict(self):
        output_state_dict = deepcopy(self.init_state_dict)
        for dataset_name, local_cache_prefix in zip(self.dataset_names, self.local_cache_prefix_list):
            if dataset_name not in self.init_state_dict:
                continue
            cache_list = glob.glob(f'{local_cache_prefix}*')
            for cache_path in cache_list:
                with open(cache_path, 'r') as f:
                    for l in f.readlines():
                        r = int(l.strip())
                        output_state_dict[dataset_name]['seen_times'][r] += 1
        return output_state_dict

    def load_state_dict(self, state_dict):
        for dataset_name, local_cache_prefix, dataset in zip(self.dataset_names, self.local_cache_prefix_list, self.dataset_list):
            if dataset_name not in state_dict:
                continue
            if dataset_name not in self.init_state_dict:
                continue
            self.clean_cache(local_cache_prefix)
            dataset.load_state_dict(state_dict[dataset_name])
            self.init_state_dict[dataset_name] = dataset.state_dict

    def clean_cache(self, local_cache_prefix):
        for fname in glob.glob(f'{local_cache_prefix}*'):
            try:
                os.remove(fname)
            except OSError:
                pass

    @classmethod
    def create_dataset_function(cls, data, data_weights, **kwargs):
        return cls(data, data_weights, **kwargs)
