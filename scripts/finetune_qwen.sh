#!/bin/bash

accelerate palme_model_openx.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \