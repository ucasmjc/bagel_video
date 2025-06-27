yaml_file="env_config.yaml"

get_yaml_value() {
    local key=$1
    local value=$(awk -F': ' "/^$key: / {print \$2}" "$yaml_file")
    echo "$value"
}

# 获取值
model_path=$(get_yaml_value "bagel_path")


num_nodes=1
node_rank=0
master_addr="localhost"
master_port=26800
#stage1 ckpt
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
  --use_flex True 
    #--resume-from $model_path \
  # --llm_path $llm_path \
  # --vae_path $vae_path \
  # --vit_path $vit_path \
  # 
  # --resume_from $resume_from \
  # --results_dir $output_path \
  # 