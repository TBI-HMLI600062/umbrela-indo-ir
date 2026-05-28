# UMBRELA-Indo-IR — Wrap-Up Index

> Dibuat: 2026-05-28  
> Dataset: MIRACL-ID (960 test queries, 1.44M passages, Indonesian Wikipedia)  
> Semua angka menggunakan **human qrels** kecuali dinyatakan lain.

---

## RQ1 — Kualitas LLM Judge (Cohen's κ, test split)

| Judge | κ | n_pairs | LLM pos% | Human pos% | File |
|-------|---|---------|----------|------------|------|
| **DeepSeek-V3** | **0.4219** | 9,668 | 27.99% | 31.94% | `final/kappa.csv`, `final/kappa_deepseek_test.csv` |
| ChatGPT (gpt-4o-mini) | 0.3856 | 6,751 | 26.38% | 32.96% | `final/kappa.csv` |
| Qwen2.5-7B-Instruct | 0.3767 | 9,668 | 30.74% | 31.94% | `final/kappa_qwen_test.csv` |
| SahabatAI-Gemma2-9B | 0.3763 | 9,668 | 41.23% | 31.94% | `final/kappa_gemma.csv` |
| Qwen-LoRA SFT (Arvin) | 0.3718 | 9,668 | 14.96% | 31.94% | `final/kappa_qwen_lora_test.csv` |
| SahabatAI-Llama3 (strict) | 0.3652 | 9,668 | 38.79% | 31.94% | `final/kappa_llama_strict.csv` |
| SahabatAI-Llama3 (default) | 0.2103 | 9,668 | 66.66% | 31.94% | `final/kappa_llama.csv` |

### Prompt Ablation (Qwen, test split)
→ `final/kappa_qwen_fewshot_basic_test.csv`, `kappa_qwen_fewshot_bing_test.csv`, `kappa_qwen_zeroshot_basic_test.csv`, `kappa_qwen_zeroshot_bing_strict_test.csv`

### Calibration (τ threshold, Qwen)
→ `final/calibration_qwen.csv`

### Per-split (Qwen)
→ `final/kappa_qwen_train.csv`, `final/kappa_qwen_val.csv`, `final/kappa_train.csv`, `final/kappa_val.csv`

---

## RQ2 — Reranker dari LLM-Generated Qrels

### First-stage retrieval (human qrels, test set)

| System | nDCG@10 | MAP@10 | Recall@100 | File |
|--------|---------|--------|-----------|------|
| BM25 baseline | 0.3053 | — | 0.7634 | `retrieval_scores.csv` |
| BGE-M3 dense | **0.5604** | — | 0.9047 | `retrieval_scores.csv` |
| Hybrid BM25+BGE-M3 | 0.5191 | — | 0.9154 | `retrieval_scores.csv` |

### Size Ablation (Gemma2 qrels → BGE reranker, BM25 first-stage)

| N queries | n_triplets | nDCG@10 | MAP@10 | Recall@100 | File |
|-----------|-----------|---------|--------|-----------|------|
| **100** | 1,937 | **0.5178** | 0.4088 | 0.7634 | `final/size_100.json` |
| 300 | 5,285 | 0.4620 | 0.3491 | 0.7634 | `final/size_300.json` |
| 500 | 9,173 | 0.5011 | 0.3950 | 0.7634 | `final/size_500.json` |
| 1,000 | 18,509 | 0.4072 | 0.2993 | 0.7634 | `final/size_1000.json` |
| full (3,257) | 60,750 | 0.3993 | 0.2917 | 0.7634 | `final/size_full.json` |

→ Ablation summary: `final/ablation_summary.csv` | Curves: `final/learning_curve.png`, `final/ap_vs_ndcg_curve.png`

**Finding:** N=100 optimal — lebih banyak data LLM justru menurunkan kualitas reranker.

### BM25 + Qwen-trained Reranker
nDCG@10 = **0.4478** | MAP@10 = 0.347 | Recall@100 = 0.7634 (+46% vs BM25)  
→ `final/bm25_qwen_rk.json`

---

## RQ3 — Bias Analysis: Self-Reinforcing Bias

> Full writeup: `final/rq3_bias_analysis.md` | Progress notes: `final/karolina_progress.md`

### Result Matrix (test split, nDCG@10)

| System | Retriever | Reranker | nDCG@10 | Δ % vs baseline | Recall@100 | File |
|--------|-----------|----------|---------|-----------------|-----------|------|
| BM25 baseline | BM25 | — | 0.3053 | — | 0.7634 | `retrieval_scores.csv` |
| BGE-M3 baseline | BGE-M3 | — | 0.5604 | — | 0.9047 | `retrieval_scores.csv` |
| Qwen3-embed baseline | Qwen3-embed | — | 0.4958 | — | 0.8508 | — |
| Hybrid BM25+Qwen3 | Hybrid | — | 0.5191 | — | 0.9154 | `retrieval_scores.csv` |
| BGE-M3 + Qwen-rk | BGE-M3 | Qwen-rk | 0.4495 | **-19.8%** | 0.9047 | `final/bgem3_qwenrk.json` |
| BGE-M3 + Gemma2-rk | BGE-M3 | Gemma2-rk | 0.5111 | -8.8% | 0.9047 | `final/bgem3_gemma2rk_test.json` |
| BGE-M3 + Qwen3-hardneg-rk | BGE-M3 | Qwen3-hardneg-rk | 0.5276 | -5.9% | 0.9047 | `final/bgem3_qwen3hardnegrk_test.json` |
| **BGE-M3 + BGE-hardneg-rk** | BGE-M3 | BGE-hardneg-rk | **0.5659** | **+1.0%** | 0.9047 | `final/bgem3_bgem3hardnegrk_test.json` |
| Qwen3-embed + Qwen-rk | Qwen3-embed | Qwen-rk | 0.4585 | -7.5% | 0.8508 | `final/qwen3_qwenrk.json` |
| Qwen3-embed + Gemma2-rk | Qwen3-embed | Gemma2-rk | 0.5160 | +4.1% | 0.8508 | `final/qwen3_gemma2rk_test.json` |
| **Qwen3-embed + BGE-hardneg-rk** | Qwen3-embed | BGE-hardneg-rk | **0.5882** | **+18.6%** 🏆 | 0.8508 | `final/qwen3_bgem3hardnegrk_test.json` |
| Hybrid + Qwen3-hardneg-rk | Hybrid | Qwen3-hardneg-rk | 0.5306 | +15.3% | 0.8792 | `final/hybrid_qwen3_hardnegrk_test.json` |

**Best system: Qwen3-embed + BGE-hardneg-rk → 0.5882 nDCG@10**

### Bias Analysis Artifacts
`final/bias_analysis/`:

| File | Isi |
|------|-----|
| `aggregate_summary.json` | Ringkasan semua baseline + reranked results |
| `judge_matrix.json` | nDCG@10 per judge × retriever |
| `judge_matrix.md` | Tabel judge bias matrix |
| `result_matrix.md` | Full result table |
| `overlap_k10/k20/k50/k100.json` | % overlap top-K antara BGE-M3 dan Qwen3 |
| `perquery_scores.json` | Per-query nDCG semua kombinasi |
| `hardneg_overlap_report.md` | Laporan overlap hard negatives |
| `delta_heatmap.png` | Heatmap perubahan nDCG tiap sistem |
| `judge_delta_chart.png` | Per-judge delta chart |
| `judge_ndcg_matrix.png` | nDCG matrix judge × retriever |
| `leaderboard_correlation.png` | Korelasi ranking antar judge |
| `ndcg_at_k.png` | nDCG@K curves |
| `perquery_violin.png` | Distribusi per-query scores |
| `posrate_effect.png` | Efek positive rate judge |
| `recall_comparison.png` | Recall comparison chart |
| `win_loss.png` | Win/loss analysis |

### TREC Run Output
`final/qwen3_bge_reranked.txt` — Qwen3-embed + BGE-hardneg-rk TREC run format (960 queries × 100 docs)

---

## Models Registry

> Model besar **tidak** di-download lokal. Cukup referensi HuggingFace.

### Reranker Models (BGE/XLM-RoBERTa)

| Model | Status | Deskripsi | HuggingFace | Lokal |
|-------|--------|-----------|-------------|-------|
| `umbrela-indo-ir-reranker-qwen` | Public, 0.6B | BGE reranker, Qwen-generated qrels | [`fassabilf/umbrela-indo-ir-reranker-qwen`](https://huggingface.co/fassabilf/umbrela-indo-ir-reranker-qwen) | `results/models/reranker_qwen/` |
| `qwen-reranker-miracl-id` | 🔐 Private, 0.6B | XLM-RoBERTa reranker, MIRACL-ID | `fassabilf/qwen-reranker-miracl-id` | — |
| `reranker-gemma2-n100` | Public | BGE reranker, Gemma2 qrels (N=100 optimal) | [`karolinajocelyn/umbrela-indo-ir-models`](https://huggingface.co/karolinajocelyn/umbrela-indo-ir-models) (root, ~1.14GB) | — |
| `reranker-bgem3-hardneg` | Public | BGE reranker, Qwen3 hard negatives — **best model** | [`karolinajocelyn/umbrela-indo-ir-models/reranker_qwen3_hardneg`](https://huggingface.co/karolinajocelyn/umbrela-indo-ir-models/tree/main/reranker_qwen3_hardneg) | `results/models/reranker_bgem3_hardneg/` (config only) |
| `reranker-size-100` | — | BGE reranker N=100 ablation | — | `results/models/reranker_100/` |
| `reranker-size-300` | — | BGE reranker N=300 ablation | — | `results/models/reranker_300/` |
| `reranker-size-500` | — | BGE reranker N=500 ablation | — | `results/models/reranker_500/` |
| `reranker-size-1000` | — | BGE reranker N=1000 ablation | — | `results/models/reranker_1000/` |
| `reranker-size-full` | — | BGE reranker N=full ablation | — | `results/models/reranker_full/` |

### LLM Fine-tune Models (Qwen2.5-7B based)

| Model | Status | Metode | Deskripsi | HF Link |
|-------|--------|--------|-----------|---------|
| `lora-qwen-miracl-id-smoke` | 🔐 Private | LoRA (PEFT) | Qwen2.5-7B-Instruct LoRA judge, smoke test | `fassabilf/lora-qwen-miracl-id-smoke` |
| `orpo-qwen-miracl-id-smoke` | 🔐 Private | ORPO (TRL) | Qwen2.5-7B-Instruct ORPO judge, smoke test | `fassabilf/orpo-qwen-miracl-id-smoke` |
| `orpo-qwen-miracl-id-smoke-b3` | 🔐 Private | ORPO (TRL) | ORPO judge variant batch-3 | `fassabilf/orpo-qwen-miracl-id-smoke-b3` |

---

## Datasets Registry

| Dataset | Status | Deskripsi | HuggingFace | Lokal |
|---------|--------|-----------|-------------|-------|
| MIRACL-ID processed | Public | 1.44M passages, topics, human qrels, BM25 index | [`fassabilf/umbrela-indo-ir`](https://huggingface.co/datasets/fassabilf/umbrela-indo-ir) | `data/miracl-id/` |
| MIRACL-ID original | Public | Raw MIRACL Indonesian subset | [`miracl/miracl`](https://huggingface.co/datasets/miracl/miracl) (lang=`id`) | — |
| Hard-neg triplets (Qwen3) | Public | 131,239 triplets dari Qwen3 top-100; cutoff rank-20 | [`karolinajocelyn/umbrela-indo-ir-data`](https://huggingface.co/datasets/karolinajocelyn/umbrela-indo-ir-data) | `data/hardneg/qwen3_hardneg/` |
| umbrela-indo-ir-results | 🔐 Private | Hasil evaluasi lengkap (semua judge, semua retriever) | `fassabilf/umbrela-indo-ir-results` | — (butuh auth) |

### Hard-neg metadata (`data/hardneg/qwen3_hardneg/data_meta.json`):
- n_queries: 3,257 | n_triplets: **131,239** | max_triplets_per_query: 50
- explicit_hardneg (human-labeled irrel): 7,965 | unannotated: 50,532
- Candidates source: `candidates/qwen3_train_top100.jsonl`

---

## Candidates Index

| File | Retriever | Split | n_queries |
|------|-----------|-------|-----------|
| `candidates/bm25_train_top100.jsonl` | BM25 | train | 3,257 |
| `candidates/bm25_val_top100.jsonl` | BM25 | val | 814 |
| `candidates/bm25_test_top100.jsonl` | BM25 | test | 960 |
| `candidates/bgem3_test_top100.jsonl` | BGE-M3 | test | 960 |
| `candidates/bgem3_train_top100.jsonl` | BGE-M3 | train | 3,257 |
| `candidates/bgem3_val_top100.jsonl` | BGE-M3 | val | 814 |
| `candidates/hybrid_test_top100.jsonl` | Hybrid BM25+BGE-M3 | test | 960 |
| `candidates/hybrid_bm25_qwen3_test_top100.jsonl` | Hybrid BM25+Qwen3 | test | 960 |
| `candidates/qwen_train_top100.jsonl` | Qwen-embed (old) | train | 3,257 |
| `candidates/qwen_val_top100.jsonl` | Qwen-embed (old) | val | 814 |
| `candidates/qwen_test_top100.jsonl` | Qwen-embed (old) | test | 960 |
| `candidates/qwen3_train_top100.jsonl` | Qwen3-embed | train | 3,257 |
| `candidates/qwen3_val_top100.jsonl` | Qwen3-embed | val | 814 |
| `candidates/qwen3_test_top100.jsonl` | Qwen3-embed (no instr) | test | 960 |
| `candidates/qwen3_instruct_val_top100.jsonl` | Qwen3-embed (with instr) | val | 814 |

---

## Qrels Index

| File | Judge | Split | n_pairs |
|------|-------|-------|---------|
| `qrels/sahabat_gemma_*.txt` | SahabatAI-Gemma2 | train/val/test | — |
| `qrels/sahabat_llama_*.txt` | SahabatAI-Llama3 | train/val/test | — |
| `qrels/qwen_*.txt` | Qwen2.5-7B | train/val/test | — |
| `qrels_strict/sahabat_llama_strict_*.txt` | Llama3 strict prompt | train/val/test | — |
| `data/miracl-id/qrels/human/test.txt` | Human (MIRACL-ID) | test | 9,668 |

---

## Full File Index: `results/final/`

<details>
<summary>49 file — klik untuk expand</summary>

### Kappa CSVs (RQ1)
- `kappa.csv` — DeepSeek + ChatGPT summary
- `kappa_deepseek_test.csv` — DeepSeek standalone
- `kappa_gemma.csv` — SahabatAI-Gemma2 test
- `kappa_llama.csv` — SahabatAI-Llama3 default test
- `kappa_llama_strict.csv` — SahabatAI-Llama3 strict test
- `kappa_qwen_test.csv` / `kappa_qwen_train.csv` / `kappa_qwen_val.csv` — Qwen per split
- `kappa_train.csv` / `kappa_val.csv` — multi-judge train/val
- `kappa_qwen_lora_test.csv` — Qwen LoRA SFT (Arvin)
- `kappa_prompt_ablation.csv` / `kappa_prompt_ablation_full.csv` — prompt mode ablation
- `kappa_qwen_fewshot_basic_test.csv` / `kappa_qwen_fewshot_bing_test.csv`
- `kappa_qwen_zeroshot_basic_test.csv` / `kappa_qwen_zeroshot_bing_strict_test.csv`
- `kappa_gemma_vllm_fewshot_basic_test.csv` / `kappa_gemma_vllm_fewshot_bing_test.csv`
- `kappa_gemma_vllm_zeroshot_basic_test.csv` / `kappa_gemma_vllm_zeroshot_bing_test.csv` / `kappa_gemma_vllm_zeroshot_bing_strict_test.csv`
- `kappa_llama_vllm_fewshot_basic_test.csv` / `kappa_llama_vllm_fewshot_bing_test.csv` / `kappa_llama_vllm_zeroshot_basic_test.csv`
- `calibration_qwen.csv` — threshold τ=1,2,3 calibration

### Size Ablation JSONs (RQ2)
- `size_100.json` / `size_300.json` / `size_500.json` / `size_1000.json` / `size_full.json`
- `ablation_summary.csv` — rangkuman ablation
- `bm25_qwen_rk.json` — BM25 + Qwen-rk eval
- `learning_curve.png` / `ap_vs_ndcg_curve.png`

### RQ3 Result JSONs
- `bgem3_bgem3hardnegrk_test.json` — BGE-M3 + BGE-hardneg-rk
- `bgem3_gemma2rk_test.json` — BGE-M3 + Gemma2-rk
- `bgem3_qwen3hardnegrk_test.json` — BGE-M3 + Qwen3-hardneg-rk
- `bgem3_qwenrk.json` — BGE-M3 + Qwen-rk
- `qwen3_bgem3hardnegrk_test.json` — Qwen3-embed + BGE-hardneg-rk 🏆
- `qwen3_gemma2rk_test.json` — Qwen3-embed + Gemma2-rk
- `qwen3_qwenrk.json` — Qwen3-embed + Qwen-rk
- `hybrid_bm25_qwen3_test.json` — Hybrid (no reranker)
- `hybrid_bm25_qwen3_qwenrk.json` — Hybrid + Qwen-rk
- `hybrid_qwen3_hardnegrk_test.json` — Hybrid + Qwen3-hardneg-rk

### Docs & Logs
- `rq3_bias_analysis.md` — RQ3 analysis writeup
- `karolina_progress.md` — detailed progress & notes
- `qwen3_bge_reranked.txt` — TREC run output
- `bias_analysis/` — 19 file (charts + data matrices)

</details>

---

## Key Findings Summary

| RQ | Finding | Best Number |
|----|---------|------------|
| RQ1 | DeepSeek-V3 paling mirip human; Qwen calibrated terbaik di open-source | κ = 0.4219 (DeepSeek) |
| RQ2 | N=100 LLM queries optimal; lebih banyak data malah overfit ke label noise | nDCG@10 = 0.5178 |
| RQ3 | Hard neg mining > family alignment; Qwen-rk merusak BGE-M3 -19.8% | nDCG@10 = **0.5882** (Qwen3+BGE-hardneg) |
