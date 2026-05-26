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
| Qwen3-embed + Qwen-rk | Qwen3 | Qwen-rk | 0.4585 | 0.3504 | 0.8508 | Karol ✅ |
| Hybrid BM25+Qwen3 (no rk) | BM25+Qwen3 RRF | — | 0.4603 | 0.3502 | 0.8792 | Karol ✅ NEW |
| Hybrid BM25+Qwen3 + Qwen-rk | BM25+Qwen3 RRF | Qwen-rk | 0.4604 | 0.3532 | 0.8792 | Karol ✅ NEW |
| Qwen3-instruct + Qwen-rk | Qwen3 + instr | Qwen-rk | — | — | — | ⚠️ aborted |
| BGE-M3 + Gemma2-rk | BGE-M3 | Gemma2-rk | — | — | — | ❌ blocked |
| Qwen3-embed + Gemma2-rk | Qwen3 | Gemma2-rk | — | — | — | ❌ blocked |

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

## Eksperimen Aborted

### ⚠️ Instruction Retrieval (Qwen3 + instruction prefix)
- **Hasil**: 0 hits / 960 queries — tidak ada dokumen relevan di top-100 untuk semua query
- **Root cause**: FAISS index di-encode dengan vLLM menggunakan pooler `use_activation=True`.
  vLLM 0.21.0 menerapkan activation layer ke embedding output. Query yang di-encode sekarang
  (baik dengan transformers maupun vLLM baru) tidak kompatibel dengan doc embeddings lama.
- **Kesimpulan**: Instruction retrieval tidak bisa ditest tanpa re-encode ulang seluruh corpus.
  Non-instruct (0.4585 nDCG@10) tetap valid karena pre-computed dengan setup yang sama.

---

## Insight dari Hasil Sejauh Ini

### Hybrid BM25+Qwen3
- R@100 naik 0.8508 → 0.8792 (+3.4pp) — lebih banyak relevan masuk pool reranker
- nDCG@10 hanya +0.2pp (0.4585 → 0.4604) meski R@100 lebih baik
- **Artinya:** Qwen-reranker tidak bisa memanfaatkan extra recall — reranker belum
  ter-kalibrasi dengan pola error Qwen3-embed → strong case untuk **hard negative mining**

### BGE-M3 vs Qwen3-embed + Qwen-rk
- BGE-M3 tanpa reranker (0.5604) masih jauh lebih baik dari Qwen3+reranker (0.4585)
- Selisih utama ada di first-stage R@100: BGE 0.9047 vs Qwen3 0.8508
- Instruction prefix diharapkan bisa menutup gap R@100 ini

---

## Next Steps (urut prioritas)

1. **[TUNGGU RADIT]** Minta `model.safetensors` dari `reranker_100` → lengkapi RQ3 matrix
2. **[NEXT]** Hard negative mining:
   - Mine negatives dari `candidates/qwen3_test_top100.jsonl` (Qwen3 errors)
   - Re-train BGE reranker dengan Qwen3-specific hard negatives
   - Harapan: reranker lebih ter-kalibrasi → nDCG@10 naik signifikan
4. **[NEXT]** HyDE dengan DeepSeek-V3 (bonus experiment)

---

## File Locations

| File | Isi |
|------|-----|
| `candidates/qwen3_test_top100.jsonl` | Qwen3-embed top-100 candidates (no instruction) |
| `candidates/qwen3_instruct_test_top100.jsonl` | Qwen3-embed + instruction prefix (pending) |
| `candidates/hybrid_bm25_qwen3_test_top100.jsonl` | BM25+Qwen3 RRF fused |
| `results/final/qwen3_qwenrk.json` | Qwen3-embed + Qwen-rk: nDCG=0.4585 |
| `results/final/bgem3_qwenrk.json` | BGE-M3 + Qwen-rk: nDCG=0.4495 |
| `results/final/hybrid_bm25_qwen3_test.json` | Hybrid no-rk: nDCG=0.4603 |
| `results/final/hybrid_bm25_qwen3_qwenrk.json` | Hybrid + Qwen-rk: nDCG=0.4604 |
| `embeddings/qwen3-embed-4b/` | FAISS indexes (5 chunks, dim=2560) |
