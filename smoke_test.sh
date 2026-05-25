#!/bin/bash
# End-to-end smoke test for EPIC 1 pipeline (Faiz).
# Runs the full pipeline on N=5 queries to verify everything works
# before kicking off the full GPU run (~4+ hours).
#
# Usage:
#   bash smoke_test.sh              # default N=5
#   bash smoke_test.sh 10           # override N
#
# Expected runtime: < 5 minutes on GPU (model load dominates).
# If this passes, run full pipeline (remove --n-queries / --max-steps limits).

set -e  # stop on first error

N=${1:-5}
OUT="results/smoke"

echo "============================================"
echo " UMBRELA-ID Smoke Test  (N=${N} queries)"
echo "============================================"
echo ""

# ── Step 1: Qwen judge on test split (for kappa) ──────────────────────────
echo "[1/6] Running Qwen judge on test split (${N} queries)..."
python qrel_generation/inference.py \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --split test \
    --n-queries "${N}" \
    --output "${OUT}/qwen_test.txt"

echo ""

# ── Step 2: Qwen judge on train split (for reranker training) ─────────────
echo "[2/6] Running Qwen judge on train split (${N} queries)..."
python qrel_generation/inference.py \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --split train \
    --n-queries "${N}" \
    --output "${OUT}/qwen_train.txt"

echo ""

# ── Step 3: Cohen's kappa (test qrels vs human) ───────────────────────────
echo "[3/6] Computing Cohen's kappa..."
python evaluation/metrics.py \
    --llm-qrels "${OUT}/qwen_test.txt" \
    --human-qrels data/miracl-id/qrels/human/test.txt \
    --output "${OUT}/kappa.csv"

echo ""

# ── Step 4: Prepare training data from train qrels ────────────────────────
echo "[4/6] Preparing reranker training data..."
python reranker/prepare_data.py \
    --qrels "${OUT}/qwen_train.txt" \
    --output "${OUT}/reranker_data/"

echo ""

# ── Step 5: Train reranker (1 epoch, max 10 gradient steps) ───────────────
echo "[5/6] Training reranker (smoke: 1 epoch, max 10 steps)..."
python reranker/train.py \
    --training-data "${OUT}/reranker_data/" \
    --output "${OUT}/reranker/" \
    --epochs 1 \
    --max-steps 10

echo ""

# ── Step 6: Reranker inference + eval (skip if no candidates yet) ──────────
BM25_CANDS="candidates/bm25_top100.jsonl"
if [ -f "${BM25_CANDS}" ]; then
    echo "[6/6] Reranker inference + nDCG@10 eval..."
    python reranker/inference.py \
        --model "${OUT}/reranker/" \
        --candidates "${BM25_CANDS}" \
        --output "${OUT}/reranked.txt" \
        --top-k 10

    python evaluation/eval_pipeline.py \
        --first-stage bm25 \
        --reranker "${OUT}/reranker/" \
        --output "${OUT}/eval.json" \
        --top-k 10
else
    echo "[6/6] SKIPPED — ${BM25_CANDS} not found (need Arvin's retrieval output)."
    echo "       Steps 1-5 still verified the core pipeline."
fi

echo ""
echo "============================================"
echo " SMOKE TEST PASSED ✓"
echo " All outputs in: ${OUT}/"
echo ""
echo " Next: full run"
echo "   python qrel_generation/inference.py --judge-model Qwen/Qwen2.5-7B-Instruct --split test --output results/qrels/qwen_test.txt"
echo "   python qrel_generation/inference.py --judge-model Qwen/Qwen2.5-7B-Instruct --split train --n-queries 1000 --output results/qrels/qwen_train.txt"
echo "   python evaluation/metrics.py --llm-qrels results/qrels/qwen_test.txt --human-qrels data/miracl-id/qrels/human/test.txt --output results/final/kappa.csv"
echo "   python reranker/prepare_data.py --qrels results/qrels/qwen_train.txt --output results/reranker_data/qwen/"
echo "   python reranker/train.py --training-data results/reranker_data/qwen/ --output results/reranker/qwen/ --epochs 3"
echo "============================================"
