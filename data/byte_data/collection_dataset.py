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

from .AIP_dataset import AIPVideoDataset


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


class CollectionDataset_dump(IterableDataset):
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
    ):
        # prepare for bucketings
        self.enable_bucket = enable_bucket
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size

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
            if cls == 'T2IHDFSDataset' or cls == 'T2IHDFSDataset_dump':
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
        _,_,_,H,W = item['mp4']['latent_256_size']
        H = H * 64
        W = W* 64
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
        if len(item['latent'].shape) == 5:
            _,_,_,H,W = item['latent'].shape
        else:
            _,_,H,W = item['latent'].shape
        bucket_id = []
        for key, value in self.dataset_to_aspect_ratios[dataset_name].items():
            if value == [H * 64, W* 64]:
                bucket_id = key
        ori_frams, ori_c, ori_H, ori_W = item['mp4'].shape
        ori_ratio = ori_H / ori_W
        bucket_h, bucket_w = self.dataset_to_aspect_ratios[dataset_name][bucket_id][0], self.dataset_to_aspect_ratios[dataset_name][bucket_id][1]
        bucket_ratio = bucket_h / bucket_w
        # print(f"ori_H {ori_H}, ori_W {ori_W}, ori_ratio {ori_ratio}. bucket_h {bucket_h}, bucket_w {bucket_w}, bucket_ratio {bucket_ratio}")
        item['mp4'] = self.crop_and_resize(item['mp4'], bucket_h, bucket_w)

        # ori_frams, ori_c, ori_H, ori_W = item['mp4'].shape
        # ori_ratio = ori_H / ori_W
        # bucket_h, bucket_w = self.dataset_to_aspect_ratios[dataset_name][bucket_id][0], self.dataset_to_aspect_ratios[dataset_name][bucket_id][1]
        # bucket_ratio = bucket_h / bucket_w
        # # print(f"ori_H {ori_H}, ori_W {ori_W}, ori_ratio {ori_ratio}. bucket_h {bucket_h}, bucket_w {bucket_w}, bucket_ratio {bucket_ratio}")
        # item['mp4'] = self.crop_and_resize(item['mp4'], bucket_h, bucket_w)

        # frames, c, H, W = item['mp4'].shape
        # # rewrite item to the same format as the original dataset
        new_item = {}
        new_item['videos'] = item['mp4']
        if len(item['latent'].shape) == 5:
            new_item['latent'] = item['latent'][0]
        else:
            new_item['latent'] = item['latent']
        new_item['prompts'] = item['txt'] if item['txt'] is not None else "" # check text
        latent_tail = item.get('latent_tail')
        if latent_tail is not None:
            new_item['latent_tail'] = item['latent_tail']
        # else:
        #     new_item['latent_tail'] = None
        # new_item['video_metadata'] = {
        #     'num_frames': frames,
        #     'height': H,
        #     'width': W,
        # }
        self.buckets[bucket_id].append(new_item)

        batch = None
        cur_batch_size = self.image_batch_size if bucket_id.startswith("i-") else self.batch_size
        if len(self.buckets[bucket_id]) >= cur_batch_size:
            batch = self.buckets[bucket_id]
            self.buckets[bucket_id] = []
        return batch

    def __iter__(self):
        def __native__iter():
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.dataset_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                yield next(dataset)

        def __bucket__iter():
            def get_next_item(dataset):
                return next(dataset)
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.train_data_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                dataset_name = self.dataset_names[dataset_idx]
                if dataset_name in self.image_dataset_names:
                    replicate_times = max(int(self.image_batch_size / self.batch_size), 1)
                    batch_data_list = []
                    while replicate_times > 0:
                        item = next(dataset)
                        batch_data = self.put_to_bucket(item, dataset_name)
                        if batch_data is not None:
                            batch_data_list.append(batch_data)
                        replicate_times -= 1
                    for batch_data in batch_data_list:
                        yield batch_data
                # else:
                #     item = next(dataset)
                #     if item == "wtf_is_abnormal":
                #         print(f"too much abnormal from {dataset_name}, continue")
                #         continue
                #     if item == "max_bad_file_count_reached":
                #         print(f"{dataset_name} for this worker is corrupted, continue")
                #         continue
                #     batch_data = self.put_to_bucket(item, dataset_name)
                #     if batch_data is not None:
                #         yield batch_data
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(get_next_item, dataset)
                        try:
                            item = future.result(timeout=10)
                        except concurrent.futures.TimeoutError:
                            print(f"timeout for get data from {dataset_name}")
                            continue
                        if item == "wtf_is_abnormal":
                            print(f"too much abnormal from {dataset_name}, continue")
                            continue
                        if item == "max_bad_file_count_reached":
                            print(f"{dataset_name} for this worker is corrupted, continue")
                            continue
                    batch_data = self.put_to_bucket( item, dataset_name)
                    if batch_data is not None:
                        yield batch_data

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
    ):
        # prepare for bucketings
        self.enable_bucket = enable_bucket
        self.batch_size = batch_size
        self.image_batch_size = image_batch_size

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
        def __native__iter():
            
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.dataset_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                yield next(dataset)

        def __bucket__iter():
            while True:
                dataset_idx = random.choices(
                    list(range(len(self.dataset_list))), weights=self.train_data_weights)[0]
                dataset = self.dataset_iter_list[dataset_idx]
                dataset_name = self.dataset_names[dataset_idx]
                if dataset_name in self.image_dataset_names:
                    replicate_times = max(int(self.image_batch_size / self.batch_size), 1)
                    batch_data_list = []
                    while replicate_times > 0:
                        item = next(dataset)
                        batch_data = self.put_to_bucket(item, dataset_name)
                        if batch_data is not None:
                            batch_data_list.append(batch_data)
                        replicate_times -= 1
                    for batch_data in batch_data_list:
                        yield batch_data
                else:
                    item = next(dataset)
                    batch_data = self.put_to_bucket(item, dataset_name)
                    if batch_data is not None:
                        yield batch_data
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
