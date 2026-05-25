#!/bin/bash
# Prompt ablation experiment (Epic E7): SahabatAI-Gemma2 judge, test split, 5 prompt variants.
# Baseline (zeroshot_bing) is already at results/qrels/sahabat-gemma_test.txt (kappa=0.3763).
# Added zeroshot_bing_strict and zeroshot_bing for full apple-to-apple comparison with Qwen.

set -e
cd "$(dirname "$0")"

HUMAN_QRELS="data/miracl-id/qrels/human/test.txt"
JUDGE_MODEL="GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct"
BATCH_SIZE=64
N_QUERIES="${1:-}"  # optional: pass number like "100" for quick testing

for MODE in zeroshot_basic fewshot_bing fewshot_basic zeroshot_bing_strict zeroshot_bing; do
    echo ""
    echo "========================================"
    echo "  Prompt mode: $MODE"
    echo "========================================"

    OUT="results/qrels/sahabat-gemma_vllm_${MODE}_test.txt"
    # fewshot prompts are longer — use 4096 max-length
    MAX_LEN=2048
    [[ "$MODE" == fewshot* ]] && MAX_LEN=4096
    CMD="python qrel_generation/inference_vllm.py \
        --judge-model $JUDGE_MODEL \
        --split test \
        --prompt-mode $MODE \
        --batch-size $BATCH_SIZE \
        --max-length $MAX_LEN \
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
        --output "results/final/kappa_gemma_vllm_${MODE}_test.csv"
done

echo ""
echo "Done."
