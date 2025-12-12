#!/bin/bash
# Ranker evaluation script
# Function: Evaluate trained ranker model

python ranker/ranker_eval.py \
    --ranker_path "model/ranker/Llama-3.1-8B-Instruct_cand10_halfTrue/trans-multipos-layers1-lr0.001-epoch1-10-numheads1-bz1024-dp0.1-sgd-momentum1.0-wd0.0001-cand10-correct-random-sample500-filtered-True-criterionkl_div-scheduleconstant-dataset_layer32-20251212132710/Split_1-TestA_30.0-TrainL_0.1.pth" \
    --test_dir "data_gen/Llama-3.1-8B-Instruct/tool/xlam_test" \
    --batchsize 500

