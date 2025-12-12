#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python generator/sample_res.py \
    --model_name "Llama-3.1-8B-Instruct" \
    --max_new_tokens 512 \
    --num_return_sequences 100 \
    --task tool \
    --batch_size -1 \
    --temperature 1.5

