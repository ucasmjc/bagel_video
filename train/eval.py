import numpy as np
import os
import torch
import random

from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
from PIL import Image
import sys
sys.path.append('./')
from data.data_utils import add_special_tokens, pil_img2rgb
from data.transforms import ImageTransform
from inferencer import InterleaveInferencer
from modeling.autoencoder import load_ae
from modeling.bagel.qwen2_navit import NaiveCache
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer

import argparse
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

import yaml
# Model Initialization
model_path = "/mnt/localdisk/hongwei/BAGEL-7B-MoT"
eval_config_file="./data/configs/eval.yaml"
checkpoint_dir="/mnt/localdisk/hongwei/Bagel/results/checkpoints"
llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

# vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
# vit_config.rope = False
# vit_config.num_hidden_layers -= 1
from modeling.autoencoder import load_ae,VideoVAE
vae_model=VideoVAE()
vae_config=vae_model.ae_params
config = BagelConfig(
    visual_gen=True,
    visual_und=False,
    llm_config=llm_config, 
    vit_config=None,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act='gelu_pytorch_tanh',
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    model          = Bagel(language_model, None, config)
   # model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(384, 256, 16)
vit_transform = ImageTransform(980, 224, 14)

# Model Loading and Multi GPU Infernece Preparing

device_map = infer_auto_device_map(
    model,
    max_memory={i: "80GiB" for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)
print(device_map)
same_device_modules = [
    'language_model.model.embed_tokens',
    'time_embedder',
    'latent_pos_embed',
    'vae2llm',
    'llm2vae',
    'connector',
    'vit_pos_embed'
]

if torch.cuda.device_count() == 1:
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device
        else:
            device_map[k] = "cuda:0"
else:
    first_device = device_map.get(same_device_modules[0])
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(model_path, "ema.safetensors"),
    device_map=device_map,
    offload_buffers=True,
    dtype=torch.bfloat16,
    force_hooks=True,
    offload_folder="/tmp/offload"
).eval()

# Inferencer Preparing 
inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)
inference_hyper=dict(
            cfg_text_scale=4.0,
            cfg_img_scale=1.0,
            cfg_interval=[0.4, 1.0],
            timestep_shift=3.0,
            num_timesteps=50,
            cfg_renorm_min=1.0,
            cfg_renorm_type="global",
        )
with open(eval_config_file, "r") as stream:
    eval_meta = yaml.safe_load(stream)
prompts=eval_meta["t2i"]["prompt"]
for idx,prompt in enumerate(prompts):
    output_dict = inferencer(text=prompt, **inference_hyper)
    output_dict['image'].save(os.path.join(checkpoint_dir,f"valid_{idx}.png"))