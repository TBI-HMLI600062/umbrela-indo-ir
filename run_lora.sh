#!/bin/bash
# E8T2: LoRA fine-tuning of Qwen2.5-7B-Instruct as LLM judge.
# Training signal: human qrels MIRACL-ID (33k train + 8k val pairs).
# Baseline kappa to beat: 0.3767 (zeroshot_bing, test split).
#
# GPU config (effective batch = 16, gradient checkpointing always on):
#   RTX 5090 32GB and RTX 4090 24GB: --batch-size 2 --grad-accum 8 (default)

set -e
cd "$(dirname "$0")"

export HF_HOME=/workspace/.hf_cache_clean
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

HUMAN_TRAIN="data/miracl-id/qrels/human/train.txt"
HUMAN_VAL="data/miracl-id/qrels/human/val.txt"
HUMAN_TEST="data/miracl-id/qrels/human/test.txt"
DATA_OUT="results/lora_data/qwen/"
MODEL_OUT="results/models/lora_qwen/"
QRELS_OUT="results/qrels/qwen_lora_test.txt"
KAPPA_OUT="results/final/kappa_qwen_lora_test.csv"

# ── Smoke test ────────────────────────────────────────────────────────────────
# Uncomment to run a quick sanity check before the full run:
#
# python lora/prepare_data.py \
#     --human-qrels $HUMAN_TRAIN \
#     --output results/lora_data/qwen_smoke/
#
# RTX 5090:  python lora/train.py --training-data results/lora_data/qwen_smoke/ \
#                --output results/models/lora_qwen_smoke/ --max-steps 10
# RTX 4090:  python lora/train.py --training-data results/lora_data/qwen_smoke/ \
#                --output results/models/lora_qwen_smoke/ --max-steps 10 \
#                --batch-size 2 --grad-accum 8
#
# python qrel_generation/inference_vllm.py \
#     --judge-model Qwen/Qwen2.5-7B-Instruct \
#     --lora-path results/models/lora_qwen_smoke/ \
#     --split test --n-queries 20 --batch-size 16 \
#     --output results/qrels/qwen_lora_smoke_test.txt
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================"
echo "  Step 1: Prepare instruction-tuning data"
echo "========================================"
python lora/prepare_data.py \
    --human-qrels $HUMAN_TRAIN \
    --val-qrels   $HUMAN_VAL \
    --output      $DATA_OUT

echo ""
echo "========================================"
echo "  Step 2: LoRA training (RTX 5090 32GB)"
echo "========================================"
# RTX 4090 24GB: add --batch-size 2 --grad-accum 8
python lora/train.py \
    --training-data $DATA_OUT \
    --output        $MODEL_OUT \
    --epochs        3 \
    --batch-size    2 \
    --grad-accum    8 \
    --lr            2e-4 \
    --lora-r        16 \
    --lora-alpha    32 \
    --max-length    1024

echo ""
echo "========================================"
echo "  Step 3: Inference on test split"
echo "========================================"
python qrel_generation/inference_vllm.py \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --lora-path   $MODEL_OUT \
    --split       test \
    --batch-size  16 \
    --output      $QRELS_OUT

echo ""
echo "========================================"
echo "  Step 4: Eval kappa vs human qrels"
echo "========================================"
python evaluation/metrics.py \
    --llm-qrels   $QRELS_OUT \
    --human-qrels $HUMAN_TEST \
    --output      $KAPPA_OUT

echo ""
echo "Done. Results at $KAPPA_OUT"
echo "Baseline (zeroshot_bing): kappa=0.3767"
