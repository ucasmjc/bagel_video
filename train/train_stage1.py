# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

#from diffusers import AutoencoderKLWan
import sys
sys.path.append('./')
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae,VideoVAE
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper, 
    fsdp_ema_setup, fsdp_ema_update,
)
import swanlab
swanlab.login(api_key="opAGVeUsjTN3PcuSFNif3", save=True)
from inferencer import InterleaveInferencer
@dataclass
class ModelArguments:
    model_path:                 str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_path:                   str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_qk_norm:                bool = field(default=True)
    tie_word_embeddings:        bool = field(default=False)
    layer_module:               str = field(default="Qwen2DecoderLayer")
    vae_path:                   str = field(default="flux/vae/ae.safetensors")
    vit_path:                   str = field(default="hf/siglip-so400m-14-980-flash-attn2-navit/")
    max_latent_size:            int = field(default=32)
    latent_patch_size:          int = field(default=2)
    vit_patch_size:             int = field(default=14)
    vit_max_num_patch_per_side: int = field(default=70)
    connector_act:              str = field(default="gelu_pytorch_tanh")
    interpolate_pos:            bool = field(default=False)
    vit_select_layer:           int = field(default=-2)       
    vit_rope:                   bool = field(default=False)

    text_cond_dropout_prob:     float = field(default=0.1)
    vae_cond_dropout_prob:      float = field(default=0.3)
    vit_cond_dropout_prob:      float = field(default=0.3)
    vae_change:                 bool = field(default=False)


@dataclass
class DataArguments:
    dataset_config_file:        str = field(default="data/configs/example.yaml")
    eval_config_file:        str = field(default="data/configs/eval.yaml")
    prefetch_factor:            int = field(default=2)
    num_workers:                int = field(default=4)
    max_num_tokens_per_sample:  int = field(default=16384)
    max_num_tokens:             int = field(default=36864)
    prefer_buffer_before:       int = field(default=16384)
    max_buffer_size:            int = field(default=50)
    data_seed:                  int = field(default=42)


@dataclass
class TrainingArguments:
    visual_gen:                 bool = field(default=True)
    visual_und:                 bool = field(default=True)
    train_stage:                str = field(default="t2i")
    results_dir:                str = field(default="results")
    checkpoint_dir:             str = field(default="results/checkpoints")
    wandb_project:              str = field(default="bagel")
    wandb_name:                 str = field(default="run")
    wandb_runid:                str = field(default="trial")
    wandb_resume:               str = field(default="allow")
    wandb_offline:              bool = field(default=False)
    global_seed:                int = field(default=4396)
    auto_resume:                bool = field(default=False)
    resume_from:                str = field(default=None)
    resume_model_only:          bool = field(default=False)
    finetune_from_ema:          bool = field(default=False)
    log_every:                  int = field(default=10)
    save_every:                 int = field(default=2000)
    total_steps:                int = field(default=500_000)
    gradient_accumulation_steps:int = field(default=1)

    warmup_steps:               int = field(default=2000)
    lr_scheduler:               str = field(default="constant")
    lr:                         float = field(default=1e-4)
    min_lr:                     float = field(default=1e-6)
    beta1:                      float = field(default=0.9)
    beta2:                      float = field(default=0.95)
    eps:                        float = field(default=1e-15)
    ema:                        float = field(default=0.9999)
    max_grad_norm:              int = field(default=1.0)
    timestep_shift:             float = field(default=1.0)
    mse_weight:                 float = field(default=1.0)
    ce_weight:                  float = field(default=1.0)
    ce_loss_reweighting:        bool = field(default=False)
    expected_num_tokens:        int = field(default=32768)

    num_replicate:              int = field(default=1)
    num_shard:                  int = field(default=8)
    sharding_strategy:          str = field(default="HYBRID_SHARD")
    backward_prefetch:          str = field(default="BACKWARD_PRE")
    cpu_offload:                bool = field(default=False)

    freeze_llm:                 bool = field(default=False)
    freeze_vit:                 bool = field(default=False)
    freeze_vae:                 bool = field(default=True)
    freeze_und:                 bool = field(default=False)
    copy_init_moe:              bool = field(default=True)
    use_flex:                   bool = field(default=False)

from safetensors.torch import load_file

def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging:
    if dist.get_rank() == 0:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
        logger = create_logger(training_args.results_dir, dist.get_rank())
        # wandb.init(
        #     project=training_args.wandb_project, 
        #     #id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
        #     name=training_args.wandb_name, 
        #     #resume=training_args.wandb_resume,
        #     mode="offline" if training_args.wandb_offline else "online"
        # )
        # wandb.config.update(training_args,allow_val_change=True)
        # wandb.config.update(model_args,allow_val_change=True)
        # wandb.config.update(data_args,allow_val_change=True)
        swanlab.init(
            # 设置项目名
            project=training_args.wandb_project,
            experiment_name=training_args.wandb_name,
            
            # 设置超参数
        )
        swanlab.config.update(training_args,allow_val_change=True)
        swanlab.config.update(model_args,allow_val_change=True)
        swanlab.config.update(data_args,allow_val_change=True)
    else:
        logger = create_logger(None, dist.get_rank())
    
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)
    llm_config = Qwen2Config.from_pretrained(os.path.join(model_args.model_path, "llm_config.json"))
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    language_model = Qwen2ForCausalLM(llm_config)
    if training_args.visual_und:
        vit_config = SiglipVisionConfig.from_pretrained(os.path.join(model_args.model_path, "vit_config.json"))
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        vit_model = SiglipVisionModel(vit_config)

    if training_args.visual_gen:
        if model_args.vae_change:
            #from cosmos_tokenizer.video_lib import CausalVideoTokenizer
            vae_model=VideoVAE()
            vae_config=vae_model.ae_params
        else:
            vae_model, vae_config = load_ae(local_path=os.path.join(model_args.model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config, 
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
    )
    
    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)
    
    with open(data_args.eval_config_file, "r") as stream:
        eval_meta = yaml.safe_load(stream)
    
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # model_state_dict_path = os.path.join(model_args.model_path, f"ema.safetensors")
    # model_state_dict=load_file(model_state_dict_path)
    # model_state_dict.pop('latent_pos_embed.pos_embed')
    # model_state_dict.pop('vit_pos_embed.pos_embed')
    # msg = model.load_state_dict(model_state_dict, strict=False)
    # del model_state_dict
    # logger.info(msg)

    # Setup tokenizer for model:
    
    use_orig_params=False
    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        use_orig_params=True
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
   # if training_args.freeze_und:
    else:
        for name, param in model.language_model.named_parameters():
            if 'moe_gen' not in name:
                param.requires_grad = False
        for name, param in model.named_parameters():        
            if 'latent_pos_embed' in name:
                param.requires_grad = True
    use_orig_params=True
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False
    if dist.get_rank() == 0:
        # 使用rank 0的进程写入一次
        with open(f'{training_args.checkpoint_dir}/trainv2_model_parameters.txt', 'w') as f:
            # 遍历所有参数及其是否需要训练
            for name, param in model.named_parameters():
                if param.requires_grad:
                    f.write(f"Parameter name: {name}\n")
                    f.write(f"Requires grad: {param.requires_grad}\n")
                    f.write("-" * 40 + "\n")
                    #logger.info(name)
        print("Parameters have been logged to trainv2_model_parameters.txt")
    

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
        use_orig_params=use_orig_params
    )
    #ema_model = deepcopy(model)
    #model_instance=deepcopy(model).eval()
    finetune_from_ema=False
    # model, ema_model = FSDPCheckpoint.try_load_ckpt(
    #     resume_from, logger, model, ema_model, resume_from_ema=finetune_from_ema
    # )
    model, ema_model = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, None, resume_from_ema=finetune_from_ema
    )
    #ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    # data_status_path = os.path.join(resume_from, "data_status.pt")
    # if os.path.exists(data_status_path):
    #     data_status = torch.load(data_status_path, weights_only=True, map_location="cpu")
    #     local_rank = dist.get_rank()
    #     if local_rank < len(data_status):
    #         data_status = data_status[local_rank]
    #     else:
    #         data_status = None
    # else:
    #     data_status = None
    # ignored_modules = set()
    # for name, module in model.named_modules():
    #     # 若该模块包含可训练参数，则加入忽略列表
    #     if any(p.requires_grad for p in module.parameters(recurse=False)):
    #         ignored_modules.add(module)
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    if dist.get_rank() == 0:
        logger.info(model)
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )

    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
    )
    #train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    #ema_model.eval()

    # train loop
    start_time = time()
    log_loss={"ce":0,"mse":0}
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    accumulation_steps = training_args.gradient_accumulation_steps
    for outer_curr_step, data in enumerate(train_loader, start=train_step*2):
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    #b*c*h*w
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))#bcthw->btchw
                    #bcthw
            if outer_curr_step==0:
                logger.info(data['padded_latent'].shape,data['packed_timesteps'].mean(),data['padded_latent'].mean(),data["packed_text_ids"].float().mean())
            loss_dict = fsdp_model(**data)

        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)
        log_loss["ce"]+=loss_dict["ce"]
        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            loss = loss + mse * training_args.mse_weight
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)
        log_loss["mse"]+=loss_dict["mse"]
        loss=loss / accumulation_steps
        loss.backward()
        curr_step=1
        if (outer_curr_step+1)%accumulation_steps==0:
            total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            #fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

            curr_step=int(outer_curr_step/accumulation_steps)
            # logger.info(
            #     f"(step={curr_step:07d}) curr_step: {curr_step}, log_remainder: {curr_step % training_args.log_every}, save_remainder: {curr_step % training_args.save_every}"
            # )

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in log_loss.items():
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()/training_args.log_every
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            lr=optimizer.param_groups[0]['lr']
            message += f"learning_rate: {lr:.8f}, "
            logger.info(message)

            wandb_log['lr'] = lr
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                #wandb.log(wandb_log, step=curr_step)
                swanlab.log(wandb_log, step=curr_step)
            start_time = time()
            log_loss={"ce":0,"mse":0}

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                #ema_model=ema_model, 
                ema_model=None,
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
       
    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
