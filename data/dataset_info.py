# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .blip3_dataset import WebIterableDataset
from .t2v_dataset import T2VIterableDataset
from .byte_dataset import CollectionDataset,CollectionDataset_frame
DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'blip3_web': WebIterableDataset,
    't2v_pretrain':T2VIterableDataset,
    "byte_data":CollectionDataset,
    "byte_frame":CollectionDataset_frame
}
from omegaconf import OmegaConf
configs = OmegaConf.load("env_config.yaml")

DATASET_INFO = {
    't2i_pretrain': {
        'laion-aes': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/laion-aes',
            'image_dir':'/mnt/weka/data_hw/final/laion-aes',
        },
        'laion-nolang': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/laion-nolang',
            'image_dir':'/mnt/weka/data_hw/final/laion-nolang',
        },
        'laion-multi': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/laion-multi',
            'image_dir':'/mnt/weka/data_hw/final/laion-multi',
        },
        'coyo1': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/coyo1',
            'image_dir':'/mnt/weka/data_hw/final/coyo1',
        },
        'coyo2': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/coyo2',
            'image_dir':'/mnt/weka/data_hw/final/coyo2',
        },
        'recap2': {
            'json_list_dir': '/mnt/weka/data_hw/final/img_json/recap2',
            'image_dir':'/mnt/weka/data_hw/final/recap2',
        },
    },
    'blip3_web':
    {
        "long":{"tar_dir":"/mnt/weka/blip3o/long/"},
        "short":{"tar_dir":"/mnt/weka/blip3o/short/"},
        "journey":{"tar_dir":"/mnt/weka/blip3o/journey"},
        "sft":{"tar_dir":"/mnt/weka/blip3o/sft/"}
    },
    'unified_edit':{
        'seedxedit_multi': {
            'data_dir': '',
            'num_files': 0,
            'num_total_samples': 0,
            "parquet_info_path": '',
		},
    },
    'vlm_sft': {
        'llava_ov': {
			'meta_path': '/mnt/localdisk/hongwei/Bagel/data/configs/llava_ov.json',
		},
    },
    't2v_pretrain':{
        'test_data': {
			'video_dir': '/mnt/localdisk/hongwei/Bagel/data/video_gen.json',
		},
    },
    'byte_data':{
        'byte_data':{
            "byte_meta_path":configs["training_data_config"],
        }
    }
}
