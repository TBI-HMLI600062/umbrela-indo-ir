#!/bin/bash
# Ranker training script
# Function: Train a ranker model to select the best solution from multiple candidates

python ranker/ranker_train.py \
    --hidden_size 4096 \
    --num_attention_heads 1 \
    --dropout_prob 0.1 \
    --num_layers 1 \
    --ranker_structure trans \
    --half \
    --opt sgd \
    --lr 0.001 \
    --momentum 1.0 \
    --weight_decay 1e-4 \
    --lr_schedule constant \
    --batch_size 1024 \
    --num_splits 10 \
    --num_epochs 1 \
    --criterion kl_div \
    --filtered \
    --train_dir "data_gen/Llama-3.1-8B-Instruct/tool/xlam_train" \
    --test_dir "data_gen/Llama-3.1-8B-Instruct/tool/xlam_test" \
    --model_path "Llama-3.1-8B-Instruct" \
    --file_size 100 \
    --save_dir "model/ranker" \
    --cand_num 10 \
    --sample_num 500 \
    --correct_sample random \
    --dataset_layer 32 \
    --dataset_type single \
    --task_name tool

