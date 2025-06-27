# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
#export WANDB_API_KEY=0f9e4269eec620b5201843ea9fe23e73c8a14b66
export swanlab=opAGVeUsjTN3PcuSFNif3
# replace the variables with your own
#export CUDA_VISIBLE_DEVICES=0,4,6,7
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1  # 关键！使通信错误抛出异常而非卡死
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch分布式调试

num_nodes=1
node_rank=0
master_addr="localhost"
master_port=26800
#stage1 ckpt
model_path=/mnt/data/checkpoints/BAGEL-7B-MoT/
train_stage="t2v"
ckpt_path=results/checkpoints/$train_stage
torchrun \
  --nnodes=$num_nodes \
  --node_rank=$node_rank \
  --nproc_per_node=8 \
  --master_addr=$master_addr \
  --master_port=$master_port \
  train/train_stage1.py \
  --total_steps 100000 \
  --wandb_name $train_stage \
  --train_stage $train_stage \
  --num_shard 8 \
  --vae_change True \
  --wandb_offline true \
  --ema 0.99 \
  --warmup_steps 0 \
  --save_every 300 \
  --dataset_config_file ./data/configs/byte.yaml \
  --model_path $model_path \
  --checkpoint_dir $ckpt_path \
  --resume_model_only True \
  --layer_module Qwen2MoTDecoderLayer \
  --auto_resume True \
  --finetune-from-ema True \
  --log_every 1 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler "cosine" \
  --lr 1e-4 \
  --num_worker 1 \
  --expected_num_tokens 1024 \
  --max_latent_size 64 \
  --max_num_tokens 1280 \
  --max_num_tokens_per_sample 1024 \
  --prefer_buffer_before 1024 \
  --freeze_und  True \
  --freeze_vit True \
  --freeze_llm False \
  --freeze_und True \
  --visual_und False \
  --use_flex True 
    #--resume-from $model_path \
  # --llm_path $llm_path \
  # --vae_path $vae_path \
  # --vit_path $vit_path \
  # 
  # --resume_from $resume_from \
  # --results_dir $output_path \
  # 