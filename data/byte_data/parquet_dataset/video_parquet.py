import io
import json
import torch
import pickle


from torch import nn
from PIL import Image
from einops import rearrange
import torchvision.transforms as transforms




from torchvision.transforms.functional import to_tensor
from typing import Any, Callable, List, Literal, Optional, Union

from .base_parquet import ParquetDataset
from .parquet_utils import local_seed, get_seed_for_rank_and_worker
from .samplers.utils import create_text_sampler, create_frame_sampler
from .samplers.text_sampler import TextFrequencySampler, TextSampler, TextCleaner
from .samplers.frame_sampler import FrameSampler, AllFrameSampler, AdaptiveAdvancedFrameSampler
import numpy as np
from .tos_client import NebuTosClient
import decord

DATA_SOURCE_MAPPING = {
    # average quality data
    "panda70m": "panda_middle_quality",
    "hdvg": "hdvg_middle_quality",
}

def patch_seed_sample(sample, plugin_caption_key):
    # add new caption to text 
    new_caption = sample[plugin_caption_key]
    text = sample.get("text", json.dumps({}))
    text = json.loads(sample["text"])
    text[plugin_caption_key] = new_caption
    sample['text'] = json.dumps(text)
    # construct frames_meta
    sample['frames_meta'] = {
        'start_idxs': sample['start_idxs'],
        'end_idxs': sample['end_idxs'],
    }
    return sample
    
class SeedV1Dataset(ParquetDataset):
    """
    Video parquet dataset.

    Arguments:
        path: a directory path that contains *.parquet files.
        video_frame_sampler: a callable function to sample frames from video.
        video_transform: a callable function to perform transformation on video.
        text_transform: a callable function to perform transformation on text.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy. Split by *.parquet file or by row groups in each file.
        force_partition: if True, raise error if partition is indivisible.
        num_parallel_files: number of parallel files to read.
        infinite: If True, data will be returned infinitely.
        use_offline_emb: If True, load latent/text_emb from offline dataset.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        sample_size: int | List[int],
        video_frame_sampler: FrameSampler = AllFrameSampler(),
        video_transform: Callable[[torch.Tensor], Any] = nn.Identity(),
        text_sampler: TextSampler = TextFrequencySampler(),
        text_transform: Callable[[str], Any] = nn.Identity(),
        seed: Optional[int] = None,
        partition: Literal["file", "group"] = "file",
        force_partition: bool = False,
        path_mode: Literal["dir", "file"] = "dir",
        num_parallel_files: int = 8,
        infinite: bool = True,
        shuffle: bool = True,
        columns: Optional[List[str]] = None,
        use_offline_emb: bool = False,
        # todo: remove these once offline embs are ready
        latent_channels: int = 16,
        txt_in_dim: int = 4096,
        system_prompt_prob: float = 0.0,
        fps: int = 24,
        plugin_caption_path = "",
        plugin_caption_key = "",
    ):
        super().__init__(
            path=path,
            seed=seed,
            partition=partition,
            num_parallel_files=num_parallel_files,
            infinite=infinite,
            force_partition=force_partition,
            path_mode=path_mode,
            shuffle=shuffle,
            columns=columns,
            plugin_caption_path=plugin_caption_path,
        )
        self.video_frame_sampler = video_frame_sampler
        self.video_transform = video_transform
        self.text_sampler = text_sampler
        self.text_transform = text_transform
        self.use_offline_emb = use_offline_emb
        self.mock_offline_emb = False  # 遵循数据分布
        self.latent_channels = latent_channels
        self.txt_in_dim = txt_in_dim
        self.system_prompt_prob = system_prompt_prob
        self.plugin_caption_path = plugin_caption_path
        self.plugin_caption_key = plugin_caption_key

        if sample_size == -1: # if sample_size is None, using Identity transformation
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
        seed = get_seed_for_rank_and_worker(self.seed)
        
        for sample in super().__iter__():
            if seed is not None:
                seed += 1
            try:
                if self.plugin_caption_path != "":
                    sample = patch_seed_sample(sample, self.plugin_caption_key)
                if sample.get("is_abnormal_caption", False):
                    print("skip abnormal caption")
                    continue
                results = {}

                if self.use_offline_emb and "latent" not in sample:
                    self.use_offline_emb = False
                    self.mock_offline_emb = True
                    print(
                        "You are using MOCK offline_emb! Make sure it's your intention."
                    )

                # Return offline embedding if provided.
                if self.use_offline_emb:
                    results["latent"] = pickle.loads(sample["latent"]).cpu()
                    text_emb_dict = pickle.loads(sample["text_emb"])
                    with local_seed(seed):
                        sampled_text_keys = self.text_sampler(
                            text=text_emb_dict).keys
                    results["text_emb"] = text_emb_dict[sampled_text_keys].cpu()
                    results["text"] = (
                        json.loads(sample["text"])[
                            sampled_text_keys] if "text" in sample else ""
                    )
                    results["uttid"] = sample["uttid"]
                    results["meta"] = json.loads(
                        sample["meta"]) if "meta" in sample else dict()
                    yield results
                    continue

                # Compute frame indices.
                frame_raw = sample["image"]
                frame_num = len(frame_raw)

                # Frame sampling.   why直接start+80
                with local_seed(seed):
                    frame_idx, frame_info = self.video_frame_sampler(frame_num, sample.get('frames_meta', None))

                # Decode frames. scale values between -1 and 1.
                frames = self._decode_frames(frame_raw, frame_idx)
                with local_seed(seed):
                    frames = self.video_transform(frames)

                if self.mock_offline_emb:
                    c, f, h, w = frames.shape
                    f, h, w = f // 4 + 1, h // 8, w // 8
                    text_dict = json.loads(
                        sample["text"]) if "text" in sample else {}
                    with local_seed(seed):
                        sampled_text_keys = self.text_sampler(
                            text=text_dict).keys
                        results["latent"] = torch.randn(
                            (f, h, w, self.latent_channels))
                        results["text_emb"] = torch.randn(
                            (100, self.txt_in_dim), dtype=torch.bfloat16
                        )
                    results["text"] = self.text_transform(
                        text_dict[sampled_text_keys] if "text" in sample else ""
                    )
                    results["uttid"] = sample["uttid"]
                    results["meta"] = json.loads(
                        sample["meta"]) if "meta" in sample else dict()
                    yield results
                    continue

                frames = self.pixel_transforms(frames)
                results["video"] = frames
                results.update(frame_info)

                # Decode meta.
                meta = json.loads(sample["meta"])
                if sample.get("text"):
                    text = json.loads(sample["text"])
                else:
                    # Decode text.
                    if meta.get("title"):
                        text = {"text": meta["title"]}
                    else:
                        text = {"text": ""}

                # Sample text
                with local_seed(seed):
                    sampled_text_keys = self.text_sampler(text=text).keys

                # Preprare system prompt
                if self.system_prompt_prob > 0:
                    data_source = meta["original"]["dataset"]
                    data_rename = DATA_SOURCE_MAPPING[data_source]
                    sys_prompt = f" SEP {data_rename}"
                else:
                    sys_prompt = ""

                # Text transform and system appending
                with local_seed(seed):
                    if isinstance(sampled_text_keys, list):
                        results["text_dict"] = {}
                        for i in sampled_text_keys:
                            if torch.rand(1) < self.system_prompt_prob:
                                results["text_dict"].update(
                                    {i: self.text_transform(
                                        text[i]) + sys_prompt}
                                )
                            else:
                                results["text_dict"].update(
                                    {i: self.text_transform(text[i])})
                    else:
                        results["text"] = self.text_transform(
                            text[sampled_text_keys])
                        if torch.rand(1) < self.system_prompt_prob:
                            results["text"] = results["text"] + sys_prompt

                results["uttid"] = sample["uttid"]
                results["meta"] = meta

                outputs = {"mp4": results['video'], "txt": results['text'],
                           "fps": self.fps, "num_frames": results['video'].shape[0],
                           }
                del results
                # Yield.
                yield outputs
            except Exception as ex:
                print(f"SeedV1Dataset got unexpected expcetion: {ex}")
                continue

    @staticmethod
    def _decode_frames(frame_raw: List[bytes], frame_idx: List[int]):
        frames = []
        for idx in frame_idx:
            frame_byte = frame_raw[idx]
            with Image.open(io.BytesIO(frame_byte)) as frame:
                frame = frame.convert("RGB")
                frame = to_tensor(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        # make value from -1.0 to 1.0
        frames = frames * 2.0 - 1.0
        return frames

    @classmethod
    def create_dataset_function(cls, json_path, args, **kwargs):
        if 'video_frame_sampler' in kwargs:
            kwargs['video_frame_sampler'] = create_frame_sampler(
                kwargs['video_frame_sampler'])
        if 'text_sampler' in kwargs:
            kwargs['text_sampler'] = create_text_sampler(
                kwargs['text_sampler'])
        return cls(path=json_path, **kwargs)





class SeedV1Dataset_dump(ParquetDataset):
    """
    Video parquet dataset.

    Arguments:
        path: a directory path that contains *.parquet files.
        video_frame_sampler: a callable function to sample frames from video.
        video_transform: a callable function to perform transformation on video.
        text_transform: a callable function to perform transformation on text.
        seed: seed for deterministic sampling. If None, just random.
        partition: partition strategy. Split by *.parquet file or by row groups in each file.
        force_partition: if True, raise error if partition is indivisible.
        num_parallel_files: number of parallel files to read.
        infinite: If True, data will be returned infinitely.
        use_offline_emb: If True, load latent/text_emb from offline dataset.
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        *,
        sample_size: int | List[int],
        video_frame_sampler: FrameSampler = AllFrameSampler(),
        video_transform: Callable[[torch.Tensor], Any] = nn.Identity(),
        text_sampler: TextSampler = TextFrequencySampler(),
        text_transform: Callable[[str], Any] = nn.Identity(),
        seed: Optional[int] = None,
        partition: Literal["file", "group"] = "file",
        force_partition: bool = False,
        path_mode: Literal["dir", "file"] = "dir",
        num_parallel_files: int = 8,
        infinite: bool = True,
        shuffle: bool = True,
        columns: Optional[List[str]] = None,
        use_offline_emb: bool = False,
        # todo: remove these once offline embs are ready
        latent_channels: int = 16,
        txt_in_dim: int = 4096,
        system_prompt_prob: float = 0.0,
        fps: int = 24,
        plugin_caption_path = "",
        plugin_caption_key = "",
        dump_path = ""
    ):
        super().__init__(
            path=path,
            seed=seed,
            partition=partition,
            num_parallel_files=num_parallel_files,
            infinite=infinite,
            force_partition=force_partition,
            path_mode=path_mode,
            shuffle=shuffle,
            columns=columns,
            plugin_caption_path=plugin_caption_path,
            dump_path = dump_path
        )
        self.video_frame_sampler = video_frame_sampler
        self.video_transform = video_transform
        self.text_sampler = text_sampler
        self.text_transform = text_transform
        self.use_offline_emb = use_offline_emb
        self.mock_offline_emb = False  # 遵循数据分布
        self.latent_channels = latent_channels
        self.txt_in_dim = txt_in_dim
        self.system_prompt_prob = system_prompt_prob
        self.plugin_caption_path = plugin_caption_path
        self.plugin_caption_key = plugin_caption_key
        self.dump_path = dump_path
        self.path = path
        self.client = NebuTosClient(ref_tos_bucket="nebudata-sg", idc="my2")
        

        if sample_size == -1: # if sample_size is None, using Identity transformation
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
        seed = get_seed_for_rank_and_worker(self.seed)
        abnormal_count = 0
        for sample in super().__iter__():
            if sample == "max_bad_file_count_reached":
                yield sample
            if seed is not None:
                seed += 1
            try:
                if self.plugin_caption_path != "":
                    sample = patch_seed_sample(sample, self.plugin_caption_key)
                if sample.get("is_abnormal_caption", False):
                    print(f"skip abnormal caption, count: {abnormal_count}, path:{self.path}")
                    abnormal_count = abnormal_count + 1
                    if abnormal_count < 200:
                        continue
                    else:
                        abnormal_count = 0
                        yield "wtf_is_abnormal"
                results = {}
                abnormal_count = 0
                if self.use_offline_emb and "latent" not in sample:
                    self.use_offline_emb = False
                    self.mock_offline_emb = True
                    print(
                        "You are using MOCK offline_emb! Make sure it's your intention."
                    )

                # Return offline embedding if provided.
                if self.use_offline_emb:
                    results["latent"] = pickle.loads(sample["latent"]).cpu()
                    text_emb_dict = pickle.loads(sample["text_emb"])
                    with local_seed(seed):
                        sampled_text_keys = self.text_sampler(
                            text=text_emb_dict).keys
                    results["text_emb"] = text_emb_dict[sampled_text_keys].cpu()
                    results["text"] = (
                        json.loads(sample["text"])[
                            sampled_text_keys] if "text" in sample else ""
                    )
                    results["uttid"] = sample["uttid"]
                    results["meta"] = json.loads(
                        sample["meta"]) if "meta" in sample else dict()
                    yield results
                    continue

                if sample.get('video_tos_uri', None):
                    toskey = sample['video_tos_uri']
                    tos_results = [self.client(toskey, hashs=toskey.split('cas/')[-1])]
                    file_io = io.BytesIO(tos_results[0])
                    reader = decord.VideoReader(file_io, ctx=decord.cpu())
                    video_length = len(reader)
                    assert video_length == 121, f"video length error:', {video_length}" 
                    valid_indices = list(range(121))
                    frames_batch = reader.get_batch(valid_indices).asnumpy()
                    del reader
                    frames = torch.from_numpy(frames_batch).float()
                    frames = (frames / 127.5) - 1
                    frames = frames.permute(0, 3, 1, 2)
                    results["video"] = frames
                    latent = np.frombuffer(sample['latent_512'], dtype=np.float32)
                    latent = latent.reshape(sample['latent_512_size'])
                    latent = torch.from_numpy(latent).to(torch.bfloat16)
                    latent_tail = np.frombuffer(sample['latent_512_tail'], dtype=np.float32)
                    latent_tail = latent_tail.reshape(sample['latent_512_size'][1],1,sample['latent_512_size'][3],sample['latent_512_size'][4])
                    latent_tail = torch.from_numpy(latent_tail).to(torch.bfloat16)
                    results['latent'] = latent
                    results['latent_tail'] = latent_tail
                    assert sample.get('caption_qwen7b_align', None) is not None, "no caption_qwen7b_align"
                    results["text"] = sample.get('caption_qwen7b_align', None)

                    outputs = {"mp4": results['video'], "txt": results['text'],
                            "fps": self.fps, "num_frames": results['video'].shape[0], 
                            "latent": results['latent'], "latent_tail": results['latent_tail'],
                            }
                    del results
                    # Yield.
                    yield outputs

                else:
                    # Compute frame indices.
                    frame_raw = sample["image"]
                    frame_num = len(frame_raw)

                    # Frame sampling.   why直接start+80
                    with local_seed(seed):
                        frame_idx, frame_info = self.video_frame_sampler(frame_num, sample.get('frames_meta', None))

                    # Decode frames.
                    frames = self._decode_frames(frame_raw, frame_idx)
                    with local_seed(seed):
                        frames = self.video_transform(frames)

                    if self.mock_offline_emb:
                        c, f, h, w = frames.shape
                        f, h, w = f // 4 + 1, h // 8, w // 8
                        text_dict = json.loads(
                            sample["text"]) if "text" in sample else {}
                        with local_seed(seed):
                            sampled_text_keys = self.text_sampler(
                                text=text_dict).keys
                            results["latent"] = torch.randn(
                                (f, h, w, self.latent_channels))
                            results["text_emb"] = torch.randn(
                                (100, self.txt_in_dim), dtype=torch.bfloat16
                            )
                        results["text"] = self.text_transform(
                            text_dict[sampled_text_keys] if "text" in sample else ""
                        )
                        results["uttid"] = sample["uttid"]
                        results["meta"] = json.loads(
                            sample["meta"]) if "meta" in sample else dict()
                        yield results
                        continue
                    frames = self.pixel_transforms(frames)

                    results["video"] = frames
                    latent = np.frombuffer(sample['latent_256'], dtype=np.float32)
                    latent = latent.reshape(sample['latent_256_size'])
                    latent = torch.from_numpy(latent).to(torch.bfloat16)
                    latent_tail = np.frombuffer(sample['latent_256_tail'], dtype=np.float32)
                    latent_tail = latent_tail.reshape(sample['latent_256_size'][1],1,sample['latent_256_size'][3],sample['latent_256_size'][4])
                    latent_tail = torch.from_numpy(latent_tail).to(torch.bfloat16)

                    results['latent'] = latent
                    results['latent_tail'] = latent_tail
                    # results.update(frame_info)

                    # Decode meta.
                    meta = json.loads(sample["meta"])
                    if sample.get("text"):
                        text = json.loads(sample["text"])
                    else:
                        # Decode text.
                        if meta.get("title"):
                            text = {"text": meta["title"]}
                        else:
                            text = {"text": ""}

                    # Sample text
                    with local_seed(seed):
                        sampled_text_keys = self.text_sampler(text=text).keys

                    # Preprare system prompt
                    if self.system_prompt_prob > 0:
                        data_source = meta["original"]["dataset"]
                        data_rename = DATA_SOURCE_MAPPING[data_source]
                        sys_prompt = f" SEP {data_rename}"
                    else:
                        sys_prompt = ""

                    # Text transform and system appending
                    with local_seed(seed):
                        if isinstance(sampled_text_keys, list):
                            results["text_dict"] = {}
                            for i in sampled_text_keys:
                                if torch.rand(1) < self.system_prompt_prob:
                                    results["text_dict"].update(
                                        {i: self.text_transform(
                                            text[i]) + sys_prompt}
                                    )
                                else:
                                    results["text_dict"].update(
                                        {i: self.text_transform(text[i])})
                        else:
                            results["text"] = self.text_transform(
                                text[sampled_text_keys])
                            if torch.rand(1) < self.system_prompt_prob:
                                results["text"] = results["text"] + sys_prompt

                    results["uttid"] = sample["uttid"]
                    results["meta"] = meta

                    outputs = {"mp4": results['video'], "txt": results['text'],
                            "fps": self.fps, "num_frames": results['video'].shape[0], 
                            "latent": results['latent'], "latent_tail": results['latent_tail'],
                            }
                    del results
                    # Yield.
                    yield outputs
            except Exception as ex:
                print(f"SeedV1Dataset got unexpected expcetion: {ex}")
                continue

    @staticmethod
    def _decode_frames(frame_raw: List[bytes], frame_idx: List[int]):
        frames = []
        for idx in frame_idx:
            frame_byte = frame_raw[idx]
            with Image.open(io.BytesIO(frame_byte)) as frame:
                frame = frame.convert("RGB")
                frame = to_tensor(frame)
            frames.append(frame)
        frames = torch.stack(frames)
        # make value from -1.0 to 1.0
        frames = frames * 2.0 - 1.0
        return frames

    @classmethod
    def create_dataset_function(cls, json_path, args, **kwargs):
        if 'video_frame_sampler' in kwargs:
            kwargs['video_frame_sampler'] = create_frame_sampler(
                kwargs['video_frame_sampler'])
        if 'text_sampler' in kwargs:
            kwargs['text_sampler'] = create_text_sampler(
                kwargs['text_sampler'])
        return cls(path=json_path, **kwargs)