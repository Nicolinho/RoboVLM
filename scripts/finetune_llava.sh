#!/bin/bash

#accelerate launch --multi_gpu --gpu_ids 0,1,2,3 --num_processes 4 --mixed_precision no palme_model_openx.py \
accelerate launch --mixed_precision no palme_model_openx.py \
    --image_model_name_or_path "liuhaotian/llava-v1.5-7b" \
    --mm_projector_lr 0.00005 \
    --output_dir 'run_llava' \
    --run_name 'run_llava' \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --batch_size 128 \
    --evaluation_strategy "epoch" \
    --eval_accumulation_steps 2  \
    --fixed_traj_eval_hist_len  5  \
    --gradient_checkpointing True  \
    --save_safetensors False  \
    --save_strategy "epoch" \
    --learning_rate  0.0003  \
    --weight_decay   0.1  \
    --adam_beta2   0.95  \
    --max_grad_norm 0.3  \
    --lr_scheduler_type  "cosine"  \
    --optim "adamw_torch"  \
    --warmup_steps 20  \
    --lora_r 32  \
    --lora_alpha 64  \
    --lora_dropout 0.05  \
    --dataloader_num_workers 2  \
    --remove_unused_columns False  \
    --num_train_epochs 10  \
    --logging_steps 1  \
    --ddp_find_unused_parameters True 