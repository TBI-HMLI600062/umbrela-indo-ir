#!/bin/bash
# E8T2: LoRA fine-tuning — 1 epoch with resume, val loss, checkpointing, + HF push.
#
# Features:
#   - Auto-resume from latest checkpoint if run is interrupted
#   - Val loss computed at every checkpoint (checkpoint-best saved automatically)
#   - Step checkpoints pruned to last 3; epoch checkpoints always kept
#   - Training logs (train_log.jsonl, training_meta.json, kappa CSV) pushed to HF
#
# GPU: RTX 5090 32GB — effective batch=16, gradient checkpointing always on
# Expected: ~2067 gradient steps, ~3-4h

set -e
cd "$(dirname "$0")"

export HF_HOME=/workspace/.hf_cache_clean
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Set HF_TOKEN before running: export HF_TOKEN=hf_...
# export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN

DATA_OUT="results/lora_data/qwen/"
MODEL_OUT="results/models/lora_qwen_1ep/"
QRELS_OUT="results/qrels/qwen_lora_1ep_test.txt"
KAPPA_OUT="results/final/kappa_qwen_lora_1ep_test.csv"
HUMAN_TEST="data/miracl-id/qrels/human/test.txt"
HUB_MODEL_ID="umbrella_ir/qwen-lora-miracl-id-judge"

echo "========================================"
echo "  Step 1: LoRA training — 1 epoch"
echo "  (auto-resumes if checkpoint exists)"
echo "========================================"
python lora/train.py \
    --training-data $DATA_OUT \
    --val-data      $DATA_OUT \
    --output        $MODEL_OUT \
    --epochs        1 \
    --batch-size    2 \
    --grad-accum    8 \
    --lr            2e-4 \
    --lora-r        16 \
    --lora-alpha    32 \
    --max-length    1024 \
    --save-steps    500 \
    --save-total-limit 3 \
    --push-to-hub \
    --hub-model-id  $HUB_MODEL_ID

echo ""
echo "========================================"
echo "  Step 2: Inference on test split"
echo "========================================"
python qrel_generation/inference_vllm.py \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --lora-path   $MODEL_OUT \
    --split       test \
    --batch-size  16 \
    --output      $QRELS_OUT

echo ""
echo "========================================"
echo "  Step 3: Eval kappa vs human qrels"
echo "========================================"
python evaluation/metrics.py \
    --llm-qrels   $QRELS_OUT \
    --human-qrels $HUMAN_TEST \
    --output      $KAPPA_OUT

echo ""
echo "========================================"
echo "  Step 4: Upload kappa results to HF"
echo "========================================"
python - <<'PYEOF'
import os
from huggingface_hub import HfApi
api = HfApi()
kappa_out = os.environ.get("KAPPA_OUT", "results/final/kappa_qwen_lora_1ep_test.csv")
hub_id    = os.environ.get("HUB_MODEL_ID", "umbrella_ir/qwen-lora-miracl-id-judge")
api.upload_file(
    path_or_fileobj=kappa_out,
    path_in_repo="kappa_lora_1ep_test.csv",
    repo_id=hub_id,
    repo_type="model",
)
print(f"Uploaded kappa CSV → {hub_id}/kappa_lora_1ep_test.csv")
PYEOF

echo ""
echo "Done."
echo "HF repo: https://huggingface.co/$HUB_MODEL_ID"
echo "Baseline (zeroshot_bing): kappa=0.3767"
