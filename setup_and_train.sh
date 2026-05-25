#!/usr/bin/env bash
# ==============================================================================
# UMBRELA LoRA Training Setup — Run on vast.ai RTX 3090/4090 instance
# ==============================================================================
# Usage: SSH into your vast.ai instance and run:
#   bash setup_and_train.sh
#
# Prerequisites:
#   - HF_TOKEN env var set (or run: export HF_TOKEN=hf_xxx)
#   - On vast.ai, pick a PyTorch image (e.g. "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
#
# This script installs deps, downloads data, preps training data, and trains
# a LoRA adapter for Qwen2.5-7B-Instruct using human qrels from MIRACL-ID.
# ==============================================================================

set -e

# ---- Config ----------------------------------------------------------------
# HF_TOKEN is optional. Only needed if you want to upload the trained adapter
# to HuggingFace. Model downloads (Qwen2.5-7B, dataset) are public.
# Get yours at: https://huggingface.co/settings/tokens
HF_TOKEN="${HF_TOKEN:-}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/umbrela-indo-ir/results/lora/qwen}"
HF_REPO="${HF_REPO:-}"              # e.g. fassabilf/umbrela-lora-qwen
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

echo "============================================================"
echo " UMBRELA LoRA Training"
echo " Model    : $MODEL_NAME"
echo " Output   : $OUTPUT_DIR"
echo " Epochs   : $EPOCHS | Batch: $BATCH_SIZE | GradAccum: $GRAD_ACCUM"
echo "============================================================"

# ---- System check ----------------------------------------------------------
echo "[1/6] Checking system..."
nvidia-smi
echo "Disk available:"
df -h / | tail -1
echo "RAM:"
free -h | head -2

# ---- Install dependencies --------------------------------------------------
echo "[2/6] Installing Python dependencies..."
pip install --quiet --no-cache-dir \
    torch transformers accelerate \
    unsloth trl bitsandbytes peft \
    datasets huggingface_hub \
    jsonlines tqdm pandas

echo "  Done. Unsloth version: $(python -c 'import unsloth; print(unsloth.__version__)')"

# ---- Clone repo ------------------------------------------------------------
echo "[3/6] Cloning umbrela-indo-ir..."
cd /workspace
if [ ! -d umbrela-indo-ir ]; then
    git clone https://github.com/TBI-HMLI600062/umbrela-indo-ir.git
fi
cd umbrela-indo-ir
git pull origin main

# ---- Download MIRACL-ID data -----------------------------------------------
echo "[4/6] Downloading MIRACL-ID data..."
huggingface-cli download fassabilf/umbrela-indo-ir \
    --repo-type dataset --local-dir data/miracl-id/

echo "  Data downloaded:"
du -sh data/miracl-id/

# ---- Prepare training data -------------------------------------------------
echo "[5/6] Preparing LoRA training data..."
python lora/prepare_data.py \
    --data-dir data/miracl-id/ \
    --output data/lora/ \
    --prompt-mode zeroshot_bing

# ---- Train LoRA ------------------------------------------------------------
echo "[6/6] Training LoRA adapter..."

# Login to HF only if uploading the adapter
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

UPLOAD_FLAG=""
if [ -n "$HF_REPO" ]; then
    UPLOAD_FLAG="--hf-repo $HF_REPO"
fi

python lora/train.py \
    --data-dir data/lora/ \
    --model "$MODEL_NAME" \
    --output "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    $UPLOAD_FLAG

# ---- Summary ---------------------------------------------------------------
echo ""
echo "============================================================"
echo " Training complete!"
echo " Adapter saved to: $OUTPUT_DIR/adapter/"
echo "============================================================"
echo ""
echo "Next — run inference with LoRA:"
echo "  python qrel_generation/inference.py \\"
echo "    --judge-model $MODEL_NAME --provider hf \\"
echo "    --lora-adapter $OUTPUT_DIR/adapter/ --split test"
echo ""
echo "Or evaluate kappa:"
echo "  python evaluation/metrics.py \\"
echo "    --llm-qrels results/qrels/custom_test.txt \\"
echo "    --human-qrels data/miracl-id/qrels/human/test.txt"
