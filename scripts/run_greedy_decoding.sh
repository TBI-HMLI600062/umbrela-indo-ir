#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python generator/greedy_decoding.py \
    --model_name "Llama-3.1-8B-Instruct" \
    --max_new_tokens 300 \
    --task tool \
    --batch_size -1 \
    --temperature 0 \
    --output_path "data_gen/greedy_decoding"

