#!/bin/bash
# Prompt ablation experiment (Epic E7): SahabatAI-Llama3-8B judge, test split, 3 remaining variants.
# Already done (skip): zeroshot_bing (kappa=0.2103), zeroshot_bing_strict (kappa=0.3652).
# Runs remaining modes via vLLM batched inference for consistency with Gemma2 ablation.

set -e
cd "$(dirname "$0")"

HUMAN_QRELS="data/miracl-id/qrels/human/test.txt"
JUDGE_MODEL="GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct"
BATCH_SIZE=64
N_QUERIES="${1:-}"  # optional: pass number like "100" for quick testing

for MODE in zeroshot_basic fewshot_bing fewshot_basic; do
    echo ""
    echo "========================================"
    echo "  Prompt mode: $MODE"
    echo "========================================"

    OUT="results/qrels/sahabat-llama_vllm_${MODE}_test.txt"
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
        --output "results/final/kappa_llama_vllm_${MODE}_test.csv"
done

echo ""
echo "Done."
