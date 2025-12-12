#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python generator/sample_hidden.py \
    --model_name "Llama-3.1-8B-Instruct" \
    --task tool \
    --batch_size 50 \
    --data_root "data_gen"

