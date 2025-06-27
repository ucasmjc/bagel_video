#!/bin/bash
set -x

#T2I权重
model_path="pretrained_ckpts/t2i"
train_stage="t2v"
ckpt_path=results/checkpoints/$train_stage

ulimit -n 1024768
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_TIMEOUT=31


MASTER_ADDR=$ARNOLD_WORKER_0_HOST
MASTER_PORT=$ARNOLD_WORKER_0_PORT
NNODES=$ARNOLD_WORKER_NUM
NODE_RANK=$ARNOLD_ID
NPROC_PER_NODE=${NPROC_PER_NODE:-8}


echo "[INFO] Launching torchrun with:"
echo "  NNODES=$NNODES"
echo "  NODE_RANK=$NODE_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"

torchrun \
  --nproc-per-node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node-rank=$NODE_RANK \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train/train_stage1.py \
  --total_steps 100000 \
  --wandb_name $train_stage \
  --train_stage $train_stage \
  --num_replicate $NNODES \
  --num_shard $NPROC_PER_NODE \
  --vae_change True \
  --wandb_offline true \
  --ema 0.99 \
  --warmup_steps 0 \
  --save_every 500 \
  --dataset_config_file ./data/configs/byte.yaml \
  --model_path $model_path \
  --checkpoint_dir $ckpt_path \
  --resume_model_only True \
  --layer_module Qwen2MoTDecoderLayer \
  --auto_resume True \
  --finetune-from-ema True \
  --log_every 10 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler "cosine" \
  --lr 1e-4 \
  --num_worker 2 \
  --expected_num_tokens 32768 \
  --max_latent_size 64 \
  --max_num_tokens 33280 \
  --max_num_tokens_per_sample 16384 \
  --freeze_und  True \
  --freeze_vit True \
  --freeze_llm False \
  --freeze_und True \
  --visual_und False \
  --use_flex True \
  --resume-from /mnt/workspace/data/code/results/checkpoints/newvae/stage1/0000050 \
  --resume_model_only True