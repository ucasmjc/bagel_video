import os
import sys
import time
import torch
import random
import bson, json
# from dataloader import KVReader, FalconReader
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from torch.utils.data.dataset import Dataset
from torchvision.transforms import functional as TVF
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, RandomResizedCrop
from pyarrow import fs, Field
import pyarrow.parquet as  pq
import numpy as np
# import debugpy; debugpy.connect(('localhost', 5678))
########## Utils ##########
def hlist_files(folders, postfix=".index"):
    """
        罗列一些 hdfs 路径下的文件。
    """
    import subprocess
    import os
    if isinstance(folders, str):
        folders = [folders]
    files = []
    for folder in folders:
        if folder.startswith('hdfs'):
            pipe = subprocess.Popen("hdfs dfs -ls -R {}".format(folder), shell=True,
                                    stdout=subprocess.PIPE)
            # output, _ = pipe.communicate()
            for line in pipe.stdout:  # type: ignore
                line = line.strip()
                # drwxr-xr-x   - user group  4 file
                if len(line.split()) < 5:
                    continue
                filepath = line.split()[-1].decode("utf8")
                if filepath.endswith(postfix):
                    files.append(filepath)
            pipe.stdout.close()  # type: ignore
            pipe.wait()
        else:
            return []
    files = sorted(files)
    return files


def resize_crop(image, image_height, image_width, use_resize_random_crop=False):
    aspect_ratio = image_width / image_height
    if not use_resize_random_crop:
        resize = RandomResizedCrop(
            size=(image_height, image_width),  # Crop to target width height
            scale=(1, 1),  # Do not scale.
            ratio=(aspect_ratio, aspect_ratio),  # Keep target aspect ratio.
            interpolation=InterpolationMode.LANCZOS  # Use LANCZO for downsample.
        )
        crop_top_coord, crop_left_coord, _, _ = resize.get_params(image, scale=(1, 1), ratio=(
            aspect_ratio, aspect_ratio))
        crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
        image = resize(image)
    else:
        image_aspect_ratio = image.width / image.height
        if image_aspect_ratio >= aspect_ratio:
            image_resize_h = image_height
            image_resize_w = int(round(image_height * (image.width / image.height)))
            crop_top_coord = 0
            crop_left_coord = random.randint(0, image_resize_w - image_width)
        else:
            image_resize_w = image_width
            image_resize_h = int(round(image_width * (image.height / image.width)))
            crop_top_coord = random.randint(0, image_resize_h - image_height)
            crop_left_coord = 0
        image = TVF.resize(image, size=[image_resize_h, image_resize_w],
                         interpolation=InterpolationMode.LANCZOS)
        image = TVF.crop(image, crop_top_coord, crop_left_coord, image_height,
                       image_width)
        crop_coords_top_left = torch.tensor([crop_top_coord, crop_left_coord])
    return image, crop_coords_top_left


def partition_by_size(data: List[Any], size: int) -> List[List[Any]]:
    """
    Partition a list by size.
    When indivisible, the last group contains fewer items than the target size.

    Examples:
        - data: [1,2,3,4,5]
        - size: 2
        - return: [[1,2], [3,4], [5]]
    """
    return [data[i:i+size] for i in range(0, len(data), size)]


class timer:
    def __init__(self, op, wait_seconds):
        self.op = op
        self.wait_seconds = wait_seconds

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *exc_info):
        self.stop_time = time.time()
        self.elapsed_seconds = self.stop_time - self.start_time
        if self.elapsed_seconds > self.wait_seconds:
            print(f"Op: '{self.op}' took: {round(self.elapsed_seconds, 2)} seconds.", file=sys.stderr)


########## ImageDecoder ##########
import io
from PIL import Image
from base64 import b64decode
from abc import abstractmethod

class ImageDecoder:
    """
    Decode image from json dictionary.
    Return None or raise exception if sample cannot be decoded to skip forward.
    """
    @abstractmethod
    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        raise NotImplementedError()


class GeneralImageDecoder(ImageDecoder):
    """
    Read image from hdfs data entry, usually is in bytes format
    """
    def __init__(self):
        # Avoid image too large warning messages.
        Image.MAX_IMAGE_PIXELS = 1000000000

    def __call__(self, item: Dict[str, Any]) -> Optional[Image.Image]:
        image_data = item.get("image_org") or item.get("image") or item.get("binary")
        if image_data is None:
            return None

        if isinstance(image_data, bytes):
            image_bytes = image_data
        else:
            image_bytes = b64decode(image_data)

        with Image.open(io.BytesIO(image_bytes)) as image:
            if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
                image = image.convert("RGBA")
                white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
                white.paste(image, mask=image.split()[3])
                image = white
            else:
                image = image.convert("RGB")
        return image


########## ImagePredicate ##########
class ImagePredicate:
    """
    Check if image satifiy a certaion requirements.
    Return False if not satisfied and True if pass the check.

    Be sure to pass key-value pair when using
    """
    @abstractmethod
    def __call__(self, image: Image.Image, **kwargs) -> bool:
        raise NotImplementedError()


class ImageMultiPredicate(ImagePredicate):
    def __init__(self, predicates: List[ImagePredicate]):
        self.predicates = predicates

    def __call__(self, image: Image.Image, **kwargs) -> bool:
        for predicate in self.predicates:
            if not predicate(image, **kwargs):
                return False
        return True


class ImageBucketResolutionPredicate(ImagePredicate):
    def __call__(self, image: Image.Image, bucket: Any, **kwargs) -> bool:
        if image.size[0] < bucket.image_width or image.size[1] < bucket.image_height:
            return False
        return True


class ImageAestheticPredicate(ImagePredicate):
    def __init__(self, aes_thed=0):
        self.aes_thed = aes_thed

    def __call__(self, image: Image.Image, content: dict, **kwargs) -> bool:
        return ("aesthetic" not in content) or (content["aesthetic"] >= self.aes_thed)


########## TextCleaner ##########
import re
import ftfy
import html
import urllib.parse as ul
from bs4 import BeautifulSoup

class TextCleaner:
    """
    Clear up a caption with strange/improper contents
    """
    bad_punct_regex = re.compile(
        r'[' + '#®•©™&@·º½¾¿¡§~' + '\)' + '\(' + '\]' + '\[' + '\}' + '\{' + '\|' + '\\' + '\/' + '\*' + r']{1,}')

    def __call__(self, text):
        # The exact text cleaning as was in the training stage:
        text = self.clean_caption(text)
        text = self.clean_caption(text)
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()

    def clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub('<person>', 'person', caption)
        caption = re.sub('<br>', ' ', caption)
        # urls:
        caption = re.sub(
            r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        caption = re.sub(
            r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',
            # noqa
            '', caption)  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features='html.parser').text

        # @<nickname>
        caption = re.sub(r'@[\w\d]+\b', '', caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
        caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
        caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
        caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
        caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
        caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
        caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',
            # noqa
            '-', caption)

        # кавычки к одному стандарту
        caption = re.sub(r'[`´«»“”¨]', '"', caption)
        caption = re.sub(r'[‘’]', "'", caption)

        # &quot;
        caption = re.sub(r'&quot;?', '', caption)
        # &amp
        caption = re.sub(r'&amp', '', caption)

        # ip adresses:
        caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

        # article ids:
        caption = re.sub(r'\d:\d\d\s+$', '', caption)

        # \n
        caption = re.sub(r'\\n', ' ', caption)

        # "#123"
        caption = re.sub(r'#\d{1,3}\b', '', caption)
        # "#12345.."
        caption = re.sub(r'#\d{5,}\b', '', caption)
        # "123456.."
        caption = re.sub(r'\b\d{6,}\b', '', caption)
        # filenames:
        caption = re.sub(
            r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

        #
        caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

        # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(self.bad_punct_regex, r' ', caption)
        caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r'(?:\-|\_)')
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, ' ', caption)

        caption = self.basic_clean(caption)

        caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
        caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
        caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

        caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
        caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
        caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
        caption = re.sub(
            r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
        caption = re.sub(r'\bpage\s+\d+\b', '', caption)

        # j2d1a2a...
        caption = re.sub(
            r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)

        caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

        caption = re.sub(r'\b\s+\:\s+', r': ', caption)
        caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
        caption = re.sub(r'\s+', ' ', caption)

        caption.strip()

        caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
        caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
        caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
        caption = re.sub(r'^\.\S+$', '', caption)

        return caption.strip()


########## T2IHDFSDataset ##########
@dataclass
class Bucket:
    index_files: List[str] = field(default_factory=list) # the .index filenames
    image_count: int       = field(default=0)            # the total number of images
    image_height: int      = field(default=0)            # the image height
    image_width: int       = field(default=0)            # the image width

class T2IHDFSDataset(Dataset):
    def __init__(self, 
                 hdfs_path, 
                 resolution,
                 caption_key,
                 aspect_ratios,
                 debug=False, 
                 use_resize_random_crop=False,
                 skip_caption_ratios=[0, 0.0655]):
        super().__init__()

        self.resolution = resolution
        self.image_decoder = GeneralImageDecoder()
        self.image_predicate = ImageMultiPredicate([
            ImageAestheticPredicate(),
            ImageBucketResolutionPredicate(),
        ])
        self.image_transform = Compose([
            ToTensor(),
            Normalize(mean=0.5, std=0.5),
        ])
        self.text_transform = TextCleaner()
        self.caption_keys = caption_key
        self.debug = debug
        self.rank = 0 # mock value
        self.use_resize_random_crop = use_resize_random_crop
        self.skip_caption_ratios = skip_caption_ratios

        self.buckets = dict()
        self.bucket_override = list(map(lambda ratio: (ratio[0], ratio[1]), aspect_ratios.values())) # w, h

        if isinstance(hdfs_path, str):
            hdfs_path = [hdfs_path]
        filepath_list = hlist_files(hdfs_path, postfix=".index")

        for filepath in filepath_list:
            # Parse name, example:
            #  filepath:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196.index"
            #  filename:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196"
            #  basename:  "2_19_256-896_00002_00196"
            #  extension: ".index"
            filename, extension = os.path.splitext(filepath)
            basename = os.path.basename(filename)

            # Parse basename, example:
            #  {id}_{image_count}_{image_height}-{image_width}_{other_info}
            if extension in [".index", ".snappy"] and "tempstate" not in filename and 'tmp' not in filename:
                image_count, image_height, image_width = basename.replace("_", "-").split("-")[1:4]
                # skip invalid file.
                try:
                    image_count = int(image_count)
                    image_height = int(image_height)
                    image_width = int(image_width)
                except:
                    continue
                if image_width <=0 or image_height<=0:
                    continue

                image_ratio = image_width / image_height
                override_image_width, override_image_height = self._override_resolution_if_needed_v1(image_width,
                                                                                image_height)
                override_image_ratio = override_image_width / override_image_height
                # Omit buckets with unreasonable size ratio, such as (128, 1536)
                if override_image_ratio / image_ratio > 1.5 or override_image_ratio / image_ratio < 0.7:
                    continue

                bucket_key = (override_image_width, override_image_height)
                bucket_entry = self.buckets.get(bucket_key, Bucket())
                bucket_entry.index_files.append(filename)
                bucket_entry.image_count += image_count
                bucket_entry.image_height = override_image_height
                bucket_entry.image_width = override_image_width
                self.buckets[bucket_key] = bucket_entry

        for i, bucket_entry in enumerate(self.buckets.values()):
            print(
                f"Bucket {i}: {bucket_entry.image_width}x{bucket_entry.image_height} " +
                f"contains {bucket_entry.image_count} images."
            )
        print(f"Total samples: {sum([bucket_entry.image_count for bucket_entry in self.buckets.values()])}")

    def _override_resolution_if_needed_v1(self, width: int, height: int) -> Tuple[int, int]:
        """
        Override the bucket resolution if configured:
        Example:
            - bucket override: [(1000, 200), (200, 1000)]
            - current resolution: (300, 900)
            - return (200, 1000) because it is the closest in aspect ratio.
        """
        if self.bucket_override is not None:
            # If bucket override is defined, find a new resolution from the override list that best matches the aspect ratio.
            assert len(self.bucket_override) > 0, "bucket_override must not be an empty list."
            target_aspect_ratio = width / height
            bucket_resolutions = self.bucket_override
            bucket_aspect_ratios = torch.tensor([w / h for w, h in bucket_resolutions], dtype=torch.float64)
            bucket_idx = bucket_aspect_ratios.sub(target_aspect_ratio).abs().argmin().item()
            width, height = bucket_resolutions[bucket_idx]

        if self.resolution != 512:
            # The buckets are defined in 512 resolution. If target resolution is not 512, we need to scale it and make sure divisible by 64.
            ratio = self.resolution / 512
            width = (width * ratio) // 64 * 64
            height = (height * ratio) // 64 * 64

        return int(width), int(height)

    def __len__(self):
        return sum(bucket.image_count for bucket in self.buckets.values())

    def __iter__(self):
        bucket_entries = list(self.buckets.values())
        bucket_weights = list(map(lambda bucket: bucket.image_count, bucket_entries))
        bucket_iterators = list(map(lambda bucket: self._iterate_bucket(bucket), bucket_entries))

        while True:
            try:
                bucket_iterator = random.choices(bucket_iterators, bucket_weights)[0]
                bucket, index_file, key, content, image, original_size_as_tuple = next(bucket_iterator)
                # get caption
                text = self.get_caption(content)
                # Skip sample if text returned None.
                if text is None:
                    if self.debug: print("text is None")
                    continue

                if self.debug:
                    print(f"Original_size_as_tuple {original_size_as_tuple}")
                    print(f"Image size: {image.size}")
                    print(f"Text length: {len(text)}")

                # Resize and crop image
                with timer(op=f"[Rank:{self.rank}] Resize image from {index_file}, key: {key}", wait_seconds=2):
                    image, crop_coords_top_left = resize_crop(image, bucket.image_height,
                                                                bucket.image_width, self.use_resize_random_crop)

                # Transform image and text
                with timer(op=f"[Rank:{self.rank}] Transform image and text from {index_file}, key: {key}",
                            wait_seconds=2):
                    if self.image_transform is not None:
                        image = self.image_transform(image)
                        image = image.unsqueeze(0) # Add temporal dim

                    # filter pure black image
                    if isinstance(image, torch.Tensor) and image.std() < 0.02 and image.mean() < -0.9:
                        if self.debug: print("image is too dark")
                        continue

                    if self.text_transform is not None:
                        text = self.text_transform(text)
                    if text == "":
                        if self.debug: print("text is empty")
                        continue

                if self.debug:
                    print(f"dataset loading current text: en is {text}")

                item = dict(
                    mp4=image,
                    txt=text,
                    num_frames=1
                )
                yield item
            except Exception as ex:
                raise ex
                # Error should not happen here, but we add a guard anyway.
                #print(f"Bucket dataset processing sample received unexpected exception at file: {index_file}", ex,
                #     file=sys.stderr)
                continue

    def _iterate_bucket(self, bucket: Bucket):
        # Copy the list.
        index_files = list(bucket.index_files)
        count_unsatisfy_image_predicor = 0
        while True:
            # Shuffle files
            random.shuffle(index_files)
            # Loop through all the .index files
            for index_file in index_files:
                try:
                    with timer(
                        op=f"[Rank:{self.rank}] KVReader opens and lists keys from index file {index_file}",
                        wait_seconds=3
                    ):
                        reader = FalconReader(index_file)
                        keys = reader.list_keys()

                    # We devide keys to batches then shuffle the batch order.
                    # Note that keys within a batch are still contiguous for faster data loading.
                    keys_batches = partition_by_size(keys, 64)
                    random.shuffle(keys_batches)

                    for key_batch in keys_batches:
                        with timer(
                            op=f"[Rank:{self.rank}] KVReader reads values from index file {index_file}, keys: {key_batch}",
                            wait_seconds=10,
                        ):
                            # Read values. The keys within this batch are contiguous for faster loading.
                            value_batch = reader.read_many(key_batch)

                            # Shuffle samples within this batch.
                            key_value_batch = list(zip(key_batch, value_batch))
                            random.shuffle(key_value_batch)

                        for key, value in key_value_batch:
                            # Decode json
                            with timer(op=f"[Rank:{self.rank}] Decoding bson/json from {index_file}, key: {key}",
                                       wait_seconds=2):
                                try:
                                    content = bson.loads(value)
                                except:
                                    content = json.loads(value)

                            # Decode image
                            with timer(op=f"[Rank:{self.rank}] Decoding image from {index_file}, key: {key}",
                                       wait_seconds=2):
                                image = self.image_decoder(content)
                                original_size_as_tuple = torch.tensor([image.height, image.width])
                            # check if image meets requirements, skip if not
                            if image is None:
                                if self.debug: print("find empty image")
                                continue
                            if self.image_predicate is not None and \
                                    not self.image_predicate(image=image, content=content, bucket=bucket):
                                if self.debug: print("image does not satifiy image predicates", index_file)
                                count_unsatisfy_image_predicor += 1
                                # Find the consecutive 500 samples that do not satisfy image_predicate.
                                # This kv file may cause the dataloader queue to be empty,
                                # leading to program interruption. Therefore, skip this kv file.
                                if count_unsatisfy_image_predicor > 500:
                                    count_unsatisfy_image_predicor = 0
                                    raise RuntimeError("Find invalid kv file, skip!")
                                continue
                            else:
                                count_unsatisfy_image_predicor = 0
                            yield bucket, index_file, key, content, image, original_size_as_tuple

                except Exception as ex:
                    # Error may happen due to network issue when reading from data from this file.
                    # Skip to the next index file regardless.
                    print(f"Bucket dataset reading data received unexpected exception at file: {index_file}", ex, file=sys.stderr)
                    continue

    def get_caption(self, content):
        text_key = None
        if len(self.caption_keys) == 1: # only one key
            res = content.get(self.caption_keys[0], None)
        else: # 2 or more keys
            for caption_key, skip_ratio in zip(self.caption_keys, self.skip_caption_ratios):
                r1 = random.random()
                if r1 >= skip_ratio and content.get(caption_key, None) is not None:
                    text_key = caption_key
                    break
            # if all previous captions are skipped, use the last one (original caption)
            if text_key is None:
                if self.debug:
                    print("v1 {} v2 {} use original caption".format(self.caption_keys[0] in content, self.caption_keys[1] in content))
                res = content.get(self.caption_keys[-1], None)
            else:
                if self.debug:
                    print("v1 {} v2 {} use {}".format(self.caption_keys[0] in content, self.caption_keys[1] in content, text_key))
                res  = content[text_key]
        if res is None:
            return None
        else:
            return res["text"]

    @classmethod
    def create_dataset_function(cls, hdfs_path, args, **kwargs):
        return cls(hdfs_path=hdfs_path, **kwargs)


class T2IHDFSDataset_dump(Dataset):
    def __init__(self, 
                 hdfs_path, 
                 resolution,
                 caption_key,
                 aspect_ratios,
                 debug=False, 
                 use_resize_random_crop=False,
                 skip_caption_ratios=[0, 0.0655]):
        super().__init__()
        ###delete
        self.resolution = resolution
        self.image_decoder = GeneralImageDecoder()
        self.image_predicate = ImageMultiPredicate([
            ImageAestheticPredicate(),
            ImageBucketResolutionPredicate(),
        ])
        self.image_transform = Compose([
            ToTensor(),
            Normalize(mean=0.5, std=0.5),
        ])
        self.text_transform = TextCleaner()
        self.caption_keys = caption_key
        self.debug = debug
        self.rank = 0 # mock value
        self.use_resize_random_crop = use_resize_random_crop
        self.skip_caption_ratios = skip_caption_ratios

        self.buckets = dict()
        self.bucket_override = list(map(lambda ratio: (ratio[0], ratio[1]), aspect_ratios.values())) # w, h
        if isinstance(hdfs_path, str):
            hdfs_path = [hdfs_path]
        filepath_list = hlist_files(hdfs_path, postfix=".parquet")

        for filepath in filepath_list:
            # Parse name, example:
            #  filepath:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196.index"
            #  filename:  "/laion5b_aesv2_512plus_buckets/2_19_256-896_00002_00196"
            #  basename:  "2_19_256-896_00002_00196"
            #  extension: ".index"
            filename, extension = os.path.splitext(filepath)
            basename = os.path.basename(filename)

            # Parse basename, example:
            #  {id}_{image_count}_{image_height}-{image_width}_{other_info}
            if 'good' in filename and extension in [".parquet"]:
                image_count, image_height, image_width = basename.replace("_", "-").split("-")[2:5]
            elif extension in [".parquet"]:
                image_count, image_height, image_width = basename.replace("_", "-").split("-")[1:4]
                # skip invalid file.
            try:
                image_count = int(image_count)
                image_height = int(image_height)
                image_width = int(image_width)
            except:
                continue
            if image_width <=0 or image_height<=0:
                continue

            image_ratio = image_width / image_height
            override_image_width, override_image_height = self._override_resolution_if_needed_v1(image_width,
                                                                            image_height)
            override_image_ratio = override_image_width / override_image_height
            # Omit buckets with unreasonable size ratio, such as (128, 1536)
            if override_image_ratio / image_ratio > 1.5 or override_image_ratio / image_ratio < 0.7:
                continue

            bucket_key = (override_image_width, override_image_height)
            bucket_entry = self.buckets.get(bucket_key, Bucket())
            bucket_entry.index_files.append(filename)
            bucket_entry.image_count += image_count
            bucket_entry.image_height = override_image_height
            bucket_entry.image_width = override_image_width
            self.buckets[bucket_key] = bucket_entry

        for i, bucket_entry in enumerate(self.buckets.values()):
            print(
                f"Bucket {i}: {bucket_entry.image_width}x{bucket_entry.image_height} " +
                f"contains {bucket_entry.image_count} images."
            )
        print(f"Total samples: {sum([bucket_entry.image_count for bucket_entry in self.buckets.values()])}")

    def _override_resolution_if_needed_v1(self, width: int, height: int) -> Tuple[int, int]:
        """
        Override the bucket resolution if configured:
        Example:
            - bucket override: [(1000, 200), (200, 1000)]
            - current resolution: (300, 900)
            - return (200, 1000) because it is the closest in aspect ratio.
        """
        if self.bucket_override is not None:
            # If bucket override is defined, find a new resolution from the override list that best matches the aspect ratio.
            assert len(self.bucket_override) > 0, "bucket_override must not be an empty list."
            target_aspect_ratio = width / height
            bucket_resolutions = self.bucket_override
            bucket_aspect_ratios = torch.tensor([w / h for w, h in bucket_resolutions], dtype=torch.float64)
            bucket_idx = bucket_aspect_ratios.sub(target_aspect_ratio).abs().argmin().item()
            width, height = bucket_resolutions[bucket_idx]

        if self.resolution != 512:
            # The buckets are defined in 512 resolution. If target resolution is not 512, we need to scale it and make sure divisible by 64.
            ratio = self.resolution / 512
            width = (width * ratio) // 64 * 64
            height = (height * ratio) // 64 * 64

        return int(width), int(height)

    def __len__(self):
        return sum(bucket.image_count for bucket in self.buckets.values())

    def __iter__(self):
        bucket_entries = list(self.buckets.values())
        bucket_weights = list(map(lambda bucket: bucket.image_count, bucket_entries))
        bucket_iterators = list(map(lambda bucket: self._iterate_bucket(bucket), bucket_entries))

        while True:
            try:
                bucket_iterator = random.choices(bucket_iterators, bucket_weights)[0]
                bucket, content, image, original_size_as_tuple = next(bucket_iterator)

                if self.resolution == 256:
                    latent = np.frombuffer(content['latent_256'], dtype=np.float32)
                    latent = latent.reshape(content['latent_256_size'])
                    latent = torch.from_numpy(latent).to(torch.bfloat16)
                if self.resolution == 512:
                    latent = np.frombuffer(content['latent_512'], dtype=np.float32)
                    latent = latent.reshape(content['latent_512_size'])
                    latent = torch.from_numpy(latent).to(torch.bfloat16)

                image, crop_coords_top_left = resize_crop(image, bucket.image_height,
                                                            bucket.image_width, self.use_resize_random_crop)
                if self.image_transform is not None:
                    image = self.image_transform(image)
                    image = image.unsqueeze(0) # Add temporal dim
                # get caption
                image_crop_256 = content.get('image_crop_256')
                if image_crop_256 is not None:
                    text = self.get_caption_new(content)
                else:
                    text = self.get_caption(content)
                # Skip sample if text returned None.
                if text is None:
                    if self.debug: print("text is None")
                    continue

                # Transform image and text
                if self.text_transform is not None:
                    text = self.text_transform(text)
                if text == "" or text == 'none':
                    if self.debug: print("text is empty")
                    continue

                if self.debug:
                    print(f"dataset loading current text: en is {text}")

                item = dict(
                    mp4=image,
                    latent = latent,
                    txt=text,
                    num_frames=1
                )
                yield item
            except Exception as ex:
                raise ex
                # Error should not happen here, but we add a guard anyway.
                #print(f"Bucket dataset processing sample received unexpected exception at file: {index_file}", ex,
                #     file=sys.stderr)
                continue

    def _iterate_bucket(self, bucket: Bucket):
        # Copy the list.
        index_files = list(bucket.index_files)
        count_unsatisfy_image_predicor = 0
        while True:
            # Shuffle files
            random.shuffle(index_files)
            # Loop through all the .index files
            for index_file in index_files:
                try:
                    ##read parquet file
                    filesystem = fs.HadoopFileSystem('hdfs://harunasg', 0)
                    index_file = index_file + '.parquet'
                    with pq.ParquetFile(index_file, filesystem=filesystem) as fr:
                        # print(f'--- total: {fr.metadata.num_rows} ---- {fr.num_row_groups}')
                        # keys = []
                        # for i in range(fr.num_row_groups):
                        #     # 读取当前的 Row Group
                        #     row_group = fr.read_row_group(i).to_pylist()
                        #     keys += row_group
                        random_index = random.randint(0, fr.num_row_groups - 1)
                        keys = fr.read_row_group(random_index).to_pylist()   
                                             
                    # We devide keys to batches then shuffle the batch order.
                    # Note that keys within a batch are still contiguous for faster data loading.
                    keys_batches = partition_by_size(keys, 64)
                    random.shuffle(keys_batches)

                    for key_batch in keys_batches:
                        random.shuffle(key_batch)

                        for content in key_batch:
                            if self.resolution == 256:
                                latent = content['latent_256']
                            else:
                                latent = content['latent_512']
                            if not latent:
                                count_unsatisfy_image_predicor += 1
                                # Find the consecutive 500 samples that do not satisfy image_predicate.
                                # This kv file may cause the dataloader queue to be empty,
                                # leading to program interruption. Therefore, skip this kv file.
                                if count_unsatisfy_image_predicor > 500:
                                    count_unsatisfy_image_predicor = 0
                                    raise RuntimeError("Find invalid kv file, skip!")
                                continue                                
                            else:
                                count_unsatisfy_image_predicor = 0
                            image = self.image_decoder(content)
                            original_size_as_tuple = torch.tensor([image.height, image.width])

                            yield bucket, content, image, original_size_as_tuple

                except Exception as ex:
                    # Error may happen due to network issue when reading from data from this file.
                    # Skip to the next index file regardless.
                    print(f"Bucket dataset reading data received unexpected exception at file: {index_file}", ex, file=sys.stderr)
                    continue

    def get_caption(self, content):
        text_key = None
        if len(self.caption_keys) == 1: # only one key
            res = content.get(self.caption_keys[0], None)
        else: # 2 or more keys
            for caption_key, skip_ratio in zip(self.caption_keys, self.skip_caption_ratios):
                r1 = random.random()
                if r1 >= skip_ratio and content.get(caption_key, None) is not None:
                    text_key = caption_key
                    break
            # if all previous captions are skipped, use the last one (original caption)
            if text_key is None:
                if self.debug:
                    print("v1 {} v2 {} use original caption".format(self.caption_keys[0] in content, self.caption_keys[1] in content))
                res = content.get(self.caption_keys[-1], None)
            else:
                if self.debug:
                    print("v1 {} v2 {} use {}".format(self.caption_keys[0] in content, self.caption_keys[1] in content, text_key))
                res  = content[text_key]
        if res is None:
            return None
        else:
            return res
    
    def get_caption_new(self, content):
        caption_dict = json.loads(content['caption_dict'])
        caption_list = []
        for k, v in caption_dict.items():
            if '_en_' in k and '_text' in k:
                caption_list.append(v)
        if len(caption_list) == 0:
            return None
        res = random.choice(caption_list)
        return res

    @classmethod
    def create_dataset_function(cls, hdfs_path, args, **kwargs):
        return cls(hdfs_path=hdfs_path, **kwargs)


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from matplotlib import pyplot as plt
    import numpy as np
    from .collection_dataset import CollectionDataset, collate_fn_map

    hdfs_path = "/mnt/localdisk/hongwei/video_gen/data/hdvg_data"
    config = "/mnt/localdisk/hongwei/video_gen/data/byte_dataset/config.yaml"
    seed = 0

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    configs = OmegaConf.load(config)
    
    train_dataset = CollectionDataset.create_dataset_function(configs['train_data'],
                                                            configs['train_data_weights'],
                                                            **configs['data']['params'])
    # train_dataset = T2IHDFSDataset.create_dataset_function(hdfs_path=hdfs_path, args=None, **configs['data']['params']['dataset_collections']['seedv2-t2i']['params'])

    # sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size,)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        collate_fn=collate_fn_map,
        pin_memory=False
    )

    output_dir = "outputs/test1"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(train_dataloader):
        print(batch.keys())
        print(batch['prompts'])
        print(batch['videos'].size())
        import pdb;pdb.set_trace()
        # print(batch['video_metadata'])
        # print(torch.min(batch['videos']), torch.max(batch['videos']))
        for j in range(batch['videos'].size()[0]):
            plt.imsave(f"{output_dir}/test_{i}_{j}.jpg", ((batch['videos'][j,0,...]+1)*127.5).permute(1,2,0).numpy().astype(np.uint8))
        if i > 20:
            break