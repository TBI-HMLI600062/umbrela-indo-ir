# Handoff — E8 LoRA + ORPO Judge Fine-tuning
**Update: 22 Mei 2026**

---

## Status Sesi Ini

### Yang Sudah Selesai ✅
| Task | Detail |
|------|--------|
| **E7 Prompt ablation** | Semua 4 variant selesai, di-commit + push (commit `7fcef72`) |
| **LoRA SFT pipeline** | Data prep + training + inference, smoke test passed |
| **Calibration analysis** | Threshold sweep: τ=2 sudah optimal, tidak ada gain |
| **ORPO data prep** | Script siap, belum dirun |
| **ORPO training** | Script ada (manual), user minta rewrite pakai TRL |

### Belum Di-commit (unstaged)
```
M  qrel_generation/inference_vllm.py   ← tambah --lora-path arg
M  qrel_generation/judges.py           ← tambah entry qwen-lora
M  requirements.txt                    ← tambah peft>=0.10
?? evaluation/calibrate.py             ← threshold calibration
?? evaluation/logit_inference.py       ← logit-based scoring
?? lora/                               ← semua LoRA scripts
?? results/final/calibration_qwen.csv  ← hasil: τ=2 optimal, kappa 0.3767
?? results/lora_data/                  ← SFT training data (33k pairs)
?? results/models/lora_qwen_smoke/     ← smoke adapter (jangan commit .safetensors)
?? run_lora.sh
```
**Commit semuanya sebelum mulai sesi baru.**

---

## Hasil Kunci Sesi Ini

### E7 Prompt Ablation (selesai)
| Prompt Mode | Kappa | LLM Pos Rate |
|---|---|---|
| **zeroshot_bing (baseline)** | **0.3767** | 0.3074 |
| zeroshot_bing_strict | 0.3218 | 0.2313 |
| fewshot_bing | 0.2720 | 0.1639 |
| zeroshot_basic | 0.2393 | 0.1265 |
| fewshot_basic | 0.2254 | 0.1146 |

**→ zeroshot_bing tetap terbaik. Ini prompt yang dipakai untuk LoRA + ORPO.**

### Calibration Analysis
- Threshold sweep τ ∈ {1,2,3} di val set → **τ=2 sudah optimal**, tidak ada gain
- Qwen sudah well-calibrated secara global (pos rate 30.7% vs human 31.9%)
- Masalah kappa bukan di thresholding tapi di per-pair reasoning
- `evaluation/logit_inference.py` tersedia untuk logit-based scoring kalau mau explore lebih

---

## Langkah Selanjutnya

### 1. Commit semua file yang ada sekarang
```bash
cd /workspace/umbrela-indo-ir
git add qrel_generation/inference_vllm.py qrel_generation/judges.py requirements.txt \
    evaluation/calibrate.py evaluation/logit_inference.py \
    lora/ results/final/calibration_qwen.csv results/lora_data/ run_lora.sh
# Jangan add results/models/ (ada .safetensors gede)
git commit -m "Faiz/E8-setup: LoRA + ORPO pipeline, calibration analysis"
git push origin main
```

### 2. Rewrite train_orpo.py pakai TRL (yang belum selesai)
User minta ini. Install TRL dulu:
```bash
pip install trl
```

Rewrite `lora/train_orpo.py` menggunakan `trl.ORPOTrainer` + `trl.ORPOConfig`.
Data format yang sudah ada di `results/orpo_data/qwen/` compatible langsung.
Format: JSONL dengan kolom `prompt`, `chosen`, `rejected` (plain string, bukan chat dict).

Referensi TRL ORPO: https://huggingface.co/docs/trl/orpo_trainer

### 3. Run ORPO data prep
```bash
python lora/prepare_orpo_data.py \
    --human-qrels data/miracl-id/qrels/human/train.txt \
    --val-qrels   data/miracl-id/qrels/human/val.txt \
    --output      results/orpo_data/qwen/
# Expects: ~19.9k balanced pairs (9.97k pos × 2 setelah upsample)
```

### 4. ORPO smoke run
```bash
HF_HOME=/workspace/.hf_cache_clean \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python lora/train_orpo.py \
    --training-data results/orpo_data/qwen/ \
    --output results/models/orpo_qwen_smoke/ \
    --max-steps 10
```

### 5. Full LoRA SFT run (E8T2) — bisa paralel sama ORPO kalau ada 2 GPU
```bash
HF_HOME=/workspace/.hf_cache_clean bash run_lora.sh
# Estimasi: ~2-3 jam di RTX 5090
# Output: results/models/lora_qwen/ + results/final/kappa_qwen_lora_test.csv
```

### 6. Full ORPO run (E8T2 alternative)
```bash
HF_HOME=/workspace/.hf_cache_clean \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python lora/train_orpo.py \
    --training-data results/orpo_data/qwen/ \
    --output results/models/orpo_qwen/ \
    --epochs 3
# Lalu inference + eval kappa (sama seperti run_lora.sh step 3-4, ganti --lora-path)
```

---

## Environment Wajib

```bash
# Model cache (stale NFS dari session sebelumnya, pakai ini)
export HF_HOME=/workspace/.hf_cache_clean

# Untuk training (cegah CUDA OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**Kalau model ga ketemu / stale lagi**: model sudah ada di `/workspace/.hf_cache_clean/`
tinggal pakai HF_HOME di atas.

---

## File Structure Penting

```
umbrela-indo-ir/
├── lora/
│   ├── prepare_data.py       ← SFT data prep (human qrels → prompt/response)
│   ├── train.py              ← LoRA SFT training (manual loop)
│   ├── prepare_orpo_data.py  ← ORPO data prep (→ prompt/chosen/rejected)
│   └── train_orpo.py         ← ORPO training (REWRITE PAKAI TRL)
├── evaluation/
│   ├── calibrate.py          ← Threshold sweep (τ=2 optimal untuk Qwen)
│   └── logit_inference.py    ← Logit-based scoring (optional, butuh GPU)
├── qrel_generation/
│   └── inference_vllm.py     ← Sudah support --lora-path (merge adapter)
├── results/
│   ├── lora_data/qwen/       ← SFT data: 33k train + 8k val examples
│   ├── orpo_data/qwen/       ← ORPO data: akan terisi setelah step 3
│   ├── models/
│   │   ├── lora_qwen/        ← Full LoRA SFT adapter (akan terisi setelah step 5)
│   │   └── orpo_qwen/        ← ORPO adapter (akan terisi setelah step 6)
│   └── final/
│       ├── calibration_qwen.csv       ← τ sweep results
│       ├── kappa_qwen_lora_test.csv   ← akan terisi setelah step 5
│       └── kappa_orpo_qwen_test.csv   ← akan terisi setelah step 6
├── run_lora.sh               ← End-to-end LoRA SFT run
└── HANDOFF_FAIZ_E8.md        ← File ini
```

---

## Target Metrik

| System | Kappa | Status |
|---|---|---|
| Baseline Qwen zeroshot_bing | 0.3767 | ✅ done |
| DeepSeek-V3 (best judge) | 0.4219 | ✅ done |
| **LoRA SFT (E8T2)** | **> 0.3767?** | 🔲 pending |
| **ORPO (E8T2 alt)** | **> LoRA SFT?** | 🔲 pending |

Kalau ORPO > LoRA SFT > baseline → strong narrative untuk paper.
Kalau tidak ada improvement → juga valid finding (LoRA dari binary signal tidak cukup, butuh lebih rich signal).
