#!/bin/bash
# Labeling results script
# Function: Evaluate and label generated solutions

python eval/labeling.py \
    --data_path "data_gen/Llama-3.1-8B-Instruct/tool/xlam_test" \
    --task tool

