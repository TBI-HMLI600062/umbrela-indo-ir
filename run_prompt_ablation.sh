#!/bin/bash
# Prompt ablation experiment (Epic E7): Qwen judge, test split, 5 prompt variants.
# Baseline (zeroshot_bing) is already at results/qrels/qwen_test.txt (kappa=0.3767).
# Uses batched inference (inference_vllm.py) for ~10x speedup vs sequential pipeline.
# Tuned for RTX 4090 24GB: batch-size=8 (model ~14GB, leaves ~10GB for KV cache).

set -e
cd "$(dirname "$0")"

HUMAN_QRELS="data/miracl-id/qrels/human/test.txt"
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"
BATCH_SIZE=16
N_QUERIES="${1:-}"  # optional: pass number like "100" for quick testing

for MODE in zeroshot_basic fewshot_bing fewshot_basic zeroshot_bing_strict; do
    echo ""
    echo "========================================"
    echo "  Prompt mode: $MODE"
    echo "========================================"

    OUT="results/qrels/qwen_${MODE}_test.txt"
    CMD="python qrel_generation/inference_vllm.py \
        --judge-model $JUDGE_MODEL \
        --split test \
        --prompt-mode $MODE \
        --batch-size $BATCH_SIZE \
        --output $OUT"
    if [ -n "$N_QUERIES" ]; then
        CMD="$CMD --n-queries $N_QUERIES"
    fi
    echo "Running: $CMD"
    $CMD

    echo ""
    echo "--- Kappa for $MODE ---"
    python evaluation/metrics.py \
        --llm-qrels "$OUT" \
        --human-qrels "$HUMAN_QRELS" \
        --output "results/final/kappa_qwen_${MODE}_test.csv"
done

echo ""
echo "========================================"
echo "  Final summary across all modes"
echo "========================================"
python evaluation/metrics.py \
    --compare-all \
    --human-qrels "$HUMAN_QRELS" \
    --output results/final/kappa_prompt_ablation.csv

echo ""
echo "Done. Summary written to results/final/kappa_prompt_ablation.csv"
