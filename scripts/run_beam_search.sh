#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python generator/beam_search.py \
    --model_name "Llama-3.1-8B-Instruct" \
    --max_new_tokens 520 \
    --task tool \
    --batch_size 10 \
    --beam 10 \
    --temperature 0 \
    --output_path "data_gen/beam_search"

