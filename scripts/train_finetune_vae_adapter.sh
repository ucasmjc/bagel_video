# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
export WANDB_API_KEY=0f9e4269eec620b5201843ea9fe23e73c8a14b66
# replace the variables with your own
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
num_nodes=1
node_rank=0
master_addr="localhost"
master_port=26800
model_path=pretrained_ckpts/BAGEL-7B-MoT
train_stage="finetune_vae"
data_config="./data/configs/blip3.yaml"
eval_config="./data/configs/eval.yaml"
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/train_stage1.py \
  --total_steps 50000 \
  --warmup_steps 750 \
  --save_every 10 \
  --lr_scheduler "cosine" \
  --wandb_name $train_stage \
  --train_stage $train_stage \
  --num_shard 8 \
  --vae_change True \
  --dataset_config_file $data_config \
  --eval_config_file $eval_config \
  --model_path $model_path \
  --layer_module Qwen2MoTDecoderLayer \
  --max_latent_size 64 \
  --auto_resume True \
  --finetune-from-ema True \
  --log_every 10 \
  --lr 2e-4 \
  --num_worker 4 \
  --expected_num_tokens 10240 \
  --max_latent_size 32 \
  --max_num_tokens 11520 \
  --max_num_tokens_per_sample 10240 \
  --freeze_und  True \
  --freeze_vit True \
  --freeze_llm True \
  --visual_und False \
  --use_flex True 
    #--resume-from $model_path \
  # --llm_path $llm_path \
  # --vae_path $vae_path \
  # --vit_path $vit_path \
  # 
  # --resume_from $resume_from \
  # --results_dir $output_path \
  # --checkpoint_dir $ckpt_path