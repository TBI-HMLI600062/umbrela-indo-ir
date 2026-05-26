# Karolina — Hasil Retrieval & Reranker
**Last updated: 2026-05-26**

---

## Semua Hasil Eksperimen (test split, 960 queries, human qrels)

| System | First-Stage | Reranker | nDCG@10 | MAP@10 | R@100 | Status |
|--------|-------------|----------|---------|--------|-------|--------|
| BM25 only | BM25 | — | 0.3053 | — | 0.7634 | Faiz |
| BM25 + Qwen-rk | BM25 | Qwen-rk | 0.4478 | 0.3470 | 0.7634 | Faiz |
| BM25 + Gemma2-rk (N=100) | BM25 | Gemma2-rk | 0.5178 | 0.4088 | 0.7634 | Radit |
| BGE-M3 only | BGE-M3 | — | 0.5604 | — | 0.9047 | Arvin |
| Hybrid BM25+BGE-M3 | Hybrid | — | 0.5191 | — | 0.9154 | Arvin |
| BGE-M3 + Qwen-rk | BGE-M3 | Qwen-rk | 0.4495 | — | 0.9047 | Karol ✅ |
| Qwen3-embed (no instr) + Qwen-rk | Qwen3 | Qwen-rk | 0.4585 | 0.3504 | 0.8508 | Karol ✅ |
| Hybrid BM25+Qwen3 (no rk) | BM25+Qwen3 RRF | — | 0.4603 | 0.3502 | 0.8792 | Karol ✅ |
| Hybrid BM25+Qwen3 + Qwen-rk | BM25+Qwen3 RRF | Qwen-rk | 0.4604 | 0.3532 | 0.8792 | Karol ✅ |
| Qwen3-instruct + Qwen-rk (retrained) | Qwen3+instr | Qwen-rk (hardneg) | — | — | — | 🔄 in progress |
| BGE-M3 + Gemma2-rk | BGE-M3 | Gemma2-rk | — | — | — | ❌ blocked |
| Qwen3-embed + Gemma2-rk | Qwen3 | Gemma2-rk | — | — | — | ❌ blocked |

---

## Instruction Prefix — Val Split Evaluation (2026-05-26)

> **Konteks:** Instruction prefix sebelumnya dianggap gagal (0 hits) karena incompatibilitas
> FAISS index lama. Index baru (`embeddings/qwen3-embed-4b/`) di-encode ulang dengan benar,
> dan query encoding dilakukan dengan HF transformers (last-token pooling, identik dengan vLLM).
> vLLM tidak bisa dipakai di environment ini karena CUDA 12.6 vs vLLM compiled for CUDA 13.

**Val split, 814 queries, human qrels:**

| Setup | nDCG@10 | Recall@10 | Recall@100 | MRR |
|---|---|---|---|---|
| no-instruction | 0.4336 | 0.4604 | 0.7910 | 0.5498 |
| **with-instruction** | **0.4923** | **0.5238** | **0.8614** | **0.6095** |
| BM25 baseline | 0.2938 | 0.3290 | 0.7257 | 0.3658 |

**Delta: +0.059 nDCG@10 | +0.070 Recall@100** — signifikan.

**→ Keputusan: semua retrieval selanjutnya pakai `--instruction` flag.**
Termasuk `qwen3_train_top100.jsonl` untuk hard neg mining dan test eval ulang.

---

## RQ3 Matrix — Bias Analysis

|                       | Qwen-reranker | Gemma2-reranker (N=100) |
|-----------------------|---------------|-------------------------|
| **Qwen3-embed**       | ✅ 0.4585     | ❌ blocked (no weights) |
| **BGE-M3**            | ✅ 0.4495     | ❌ blocked (no weights) |

**⚠️ Blocked:** `results/models/reranker_100/` tidak punya `model.safetensors`.
Radit tidak upload ke HF dan Gemma2 train qrels juga tidak tersedia.
**→ Tanya Radit: minta file `model.safetensors` dari `reranker_100`.**

---

## Insight dari Hasil Sejauh Ini

### Instruction prefix sangat penting
- +7pp Recall@100 dan +6pp nDCG@10 di val — ini bukan marginal improvement
- Artinya semua hasil Qwen3 sebelumnya (tanpa instruction) adalah lower bound yang signifikan
- Hard negative mining dari train candidates dengan instruction akan lebih bersih

### Hybrid BM25+Qwen3 (sebelum instruction fix)
- R@100 naik 0.8508 → 0.8792 (+3.4pp) — lebih banyak relevan masuk pool reranker
- nDCG@10 hanya +0.2pp meski R@100 lebih baik
- **Artinya:** Qwen-reranker tidak bisa memanfaatkan extra recall — reranker belum
  ter-kalibrasi dengan pola error Qwen3-embed → strong case untuk **hard negative mining**
- Dengan instruction (R@100 val 0.8614), gap ke BGE-M3 (0.9047) jauh lebih kecil

### BGE-M3 vs Qwen3-embed + Qwen-rk
- BGE-M3 tanpa reranker (0.5604) masih lebih baik dari Qwen3+reranker (0.4585)
- Dengan instruction + reranker ter-kalibrasi, target realistis: >0.52 nDCG@10

---

## Next Steps (urut prioritas)

1. **[NEXT — SIAP]** Generate `qwen3_train_top100.jsonl` dengan `--instruction`
   - Jalankan: `retrieve.py --use-hf --instruction "..." --topics train.tsv`
   - Estimasi: ~8-10 menit (3257 queries, RTX 3090)
2. **[NEXT]** Hard negative mining (`reranker/mine_hard_negatives.py`)
   - Source: `qwen3_train_top100.jsonl` (instruction) × `human/train.txt`
   - Hard neg cutoff: top-20, max 50 triplets per query
3. **[NEXT]** Retrain BGE reranker dengan mixed data:
   - `qwen3_hardneg/train.jsonl` + `llm_triplets/train.jsonl` (sahabat_llama_train)
4. **[NEXT]** Re-evaluate full pipeline: Hybrid BM25+Qwen3-instruct + reranker-hardneg
5. **[TUNGGU RADIT]** Minta `model.safetensors` dari `reranker_100` → lengkapi RQ3 matrix
6. **[BONUS]** HyDE dengan DeepSeek-V3

---

## File Locations

| File | Isi |
|------|-----|
| `candidates/qwen3_test_top100.jsonl` | Qwen3-embed top-100, test, **no instruction** (lama) |
| `candidates/qwen3_val_top100.jsonl` | Qwen3-embed top-100, val, no instruction — nDCG@10=0.4336 |
| `candidates/qwen3_instruct_val_top100.jsonl` | Qwen3-embed top-100, val, **with instruction** — nDCG@10=0.4923 |
| `candidates/hybrid_bm25_qwen3_test_top100.jsonl` | BM25+Qwen3 RRF fused (no instruction) |
| `results/final/qwen3_qwenrk.json` | Qwen3-embed (no instr) + Qwen-rk: nDCG=0.4585 |
| `results/final/bgem3_qwenrk.json` | BGE-M3 + Qwen-rk: nDCG=0.4495 |
| `results/final/hybrid_bm25_qwen3_test.json` | Hybrid no-rk: nDCG=0.4603 |
| `results/final/hybrid_bm25_qwen3_qwenrk.json` | Hybrid + Qwen-rk: nDCG=0.4604 |
| `embeddings/qwen3-embed-4b/` | FAISS indexes (5 chunks, dim=2560, 21GB) |
| `reranker/mine_hard_negatives.py` | Script hard neg mining (baru dibuat) |
