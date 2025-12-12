#!/bin/bash
# Code evaluation script
# Function: Evaluate code generation task results

python eval/tool_ev.py \
    --file_path "data_gen/greedy_decoding/Llama-3.1-8B-Instruct/tool/xlam_test.json" \
    --passk 1

