import logging
import os
import yaml
import hashlib
import traceback
from typing import Any, Union, List, Optional
#import bytedtos
import io
import decord
import torch
from pyarrow import fs

def hdfs_read(file_path) -> bytes:
    fp = str(file_path)
    filesystem = resolve_fs(fp)

    with filesystem.open_input_stream(fp) as f:
        content = f.readall()
    return content

def sha256_hashs(b: bytes, nbytes=32, bit_len=128) -> bytes:
    m = hashlib.sha256()
    m.update(b)
    mb = m.digest()
    bb = mb[:nbytes] 
    truncated_hashs = bb[: bit_len // 8]
    return truncated_hashs.hex().lower()

def retry(func, retry=3):
    if retry == 0:
        return func

    def wrapper(*args, **kwargs):
        for i in range(retry):
            error = ''
            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                print(f"In {__file__}, retry {i + 1} times!")
                error = traceback.format_exc()
        raise Exception(f"Traceback: {error}")

    return wrapper


def resolve_fs(paths: Union[str, list[str]]) -> fs.FileSystem:
    _p: str = paths  # type: ignore
    if isinstance(paths, list):
        _p = paths[0]
    _p = "/".join(_p.split("/")[:3])
    filesystem, _ = fs._resolve_filesystem_and_path(_p)

    return filesystem



class BaseClient:
    def __init__(self, retry=0, **kwargs):
        self.retry = retry

    def __call__(self, keys: Union[str, List[str]], hashs: Optional[Union[str, List[str]]]=None) -> Union[bytes, List[bytes]]:
        """
        Read bytes from remote data source.
        Args:
            keys (str or list[str]): tos keys or hdfs uri or etc.
            hashs (str or list[str]): hashs of the data.
        Returns:
            bytes (or list[bytes]]): bytes read from remote data source.
        """
        if isinstance(keys, str):
            assert hashs is None or isinstance(hashs, str)
            keys = [keys]
            hashs = [hashs] if hashs is not None else None
            return_list = False
        else:
            return_list = True
        
        if hashs is not None:
            bytes_get = retry(self.get_bytes_and_check, retry=3)(keys, hashs)
        else:
            bytes_get = retry(self.get_bytes, retry=self.retry)(keys)

        if return_list:
            return bytes_get
        else:
            return bytes_get[0]

    def get_bytes_and_check(self, keys: List[bytes], hashs: List[bytes]) -> List[bytes]:
        bytes_get = self.get_bytes(keys)
        for k, b, h in zip(keys, bytes_get, hashs):
            if sha256_hashs(b) != h:
                raise Exception(f"hashs check failed on keyss {k}, {sha256_hashs(b)} != {h}!")
        return bytes_get

    def get_bytes(self, keys: List[bytes]) -> List[bytes]:
        """
        Read bytes from remote data source.
        Args:
            keyss: tos keys or hdfs uri or etc.
        Returns:
            bytes: bytes read from remote data source.
        """
        raise NotImplementedError

class TosClient(BaseClient):
    def __init__(
        self,
        ak,
        bucket,
        idc,
        timeout=10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.tos_client = bytedtos.Client(bucket, ak, timeout=timeout, idc=idc)

    # Input => toskeys
    def get_bytes(self, keys: List[bytes]) -> List[bytes]:
        """
        Read bytes from tos keys.
        Args:
            keys (str or list[str]): tos keys.
        Returns:
            bytes (or list[bytes]]): bytes read from tos.
        """
        return [self.tos_client.get_object(keys).data for keys in keys]

class NebuTosClient(TosClient):
    default_config = {
        "nebudata-us": "hdfs://harunava/home/byte_icaip_nebudata/proj/nebudata/conf/nebuconfig_va_20240925.yaml",
        "nebudata-sg": "hdfs://harunasg/home/byte_icaip_nebudata_sg/proj/nebudata/conf/nebuconfig_sg_20240925.yaml",  # Default
    }

    def __init__(
        self,
        ref_tos_bucket: Union[str, None] = None,
        idc: Union[str, None] = None,
        **kwargs,
    ):
        logging.info(f"NebuTos config: {ref_tos_bucket=} {idc=}")
        if idc is None:
            idc = os.environ.get("RUNTIME_IDC_NAME", "my2")
        
        if ref_tos_bucket is not None:
            assert ref_tos_bucket in self.default_config, f"Unknow tos_bucket {ref_tos_bucket}, please use one of {self.default_config.keyss()}."
            nebuconfig_file = self.default_config.get(ref_tos_bucket)
        else:
            arnold_base_dir = os.environ.get("ARNOLD_BASE_DIR", "hdfs://harunasg")
            for ref_tos_bucket, nebuconfig_file in self.default_config.items():
                if arnold_base_dir in nebuconfig_file:
                    break

        nebuconfig = yaml.safe_load(hdfs_read(nebuconfig_file).decode("utf-8"))
        default_access_keys = nebuconfig['tos_user_access_key']
        tos_ak = os.environ.get("TOS_USER_ACCESS_key", default_access_keys)

        super().__init__(tos_ak, ref_tos_bucket, idc, **kwargs)


if __name__ == "__main__":
    client = NebuTosClient(ref_tos_bucket="nebudata-sg", idc="my2")
    # toskey = 'cas/596ccf6d8de5d16e0ca5a91c0610d9bd'
    toskey = 'cas/0c862903f94897a08bde81ee10104c48'
    results = [client(toskey, hashs=toskey.split('cas/')[-1])]
    # with open('output_video.mp4', 'wb') as f:
    #     f.write(results[0])
    # np_array = np.frombuffer(results[0], dtype=np.uint8)
    file_io = io.BytesIO(results[0])
    reader = decord.VideoReader(file_io, ctx=decord.cpu(0))
    video_length = len(reader)
    # sampler = FrameSamplerCollection(data_configs['samplers'])
    # video_idxs, structure = self.sampler(video_length, params)
    # frames_idxs = copy.deepcopy(video_idxs)
    # in_range_len = len(video_idxs)
    # out_range_idxs = self.add_out_range_sample(
    #     video_idxs, video_length, params)
    # video_idxs = video_idxs + out_range_idxs
    # video_idxs_array = np.array(video_idxs)
    # video_idxs_valid_mask = video_idxs_array >= 0
    # valid_indices = video_idxs_array[video_idxs_valid_mask]
    valid_indices = list(range(121))
    frames_batch = reader.get_batch(valid_indices).asnumpy()
    frames_tensor = torch.from_numpy(frames_batch).float()
    frames_tensor = (frames_tensor / 127.5) - 1
    frames_tensor = frames_tensor.permute(0, 3, 1, 2)
    del reader

# 'clip_toskey': 'cas/596ccf6d8de5d16e0ca5a91c0610d9bd'
# 'clip_tosurl': 'https://tosv.byted.org/obj/nebudata-sg/cas/596ccf6d8de5d16e0ca5a91c0610d9bd'
# 'clip_url': 'https://tosv-sg.tiktok-row.org/obj/nebudata-sg/cas/596ccf6d8de5d16e0ca5a91c0610d9bd'