#!/bin/bash
# Prompt ablation experiment (Epic E7): SahabatAI-Gemma2 judge, test split, 3 prompt variants.
# Baseline (zeroshot_bing) is already at results/qrels/sahabat-gemma_test.txt (kappa=0.3763).

set -e
cd "$(dirname "$0")"

HUMAN_QRELS="data/miracl-id/qrels/human/test.txt"
JUDGE_MODEL="GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
BATCH_SIZE=64
N_QUERIES="${1:-}"  # optional: pass number like "100" for quick testing

for MODE in zeroshot_basic fewshot_bing fewshot_basic; do
    echo ""
    echo "========================================"
    echo "  Prompt mode: $MODE"
    echo "========================================"

    OUT="results/qrels/sahabat-gemma_${MODE}_test.txt"
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
        --output "results/final/kappa_gemma_${MODE}_test.csv"
done

echo ""
echo "Done."
