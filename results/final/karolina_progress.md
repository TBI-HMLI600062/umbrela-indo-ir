# Karolina — Hasil Retrieval & Reranker

## First-Stage Retrieval: Qwen3-Embedding-4B

**Model:** `Qwen/Qwen3-Embedding-4B`
**Corpus:** MIRACL-ID, 1.44M passages (5 FAISS shards, IndexFlatIP, dim=2560)
**Embeddings di HF:** `karolinajocelyn/umbrela-indo-ir` → folder `qwen3-embed-4b/`
**Candidates:** `candidates/qwen3_test_top100.jsonl` — 960 queries, top-100 each

| Split | Queries | File |
|-------|---------|------|
| test  | 960     | candidates/qwen3_test_top100.jsonl |

First-stage nDCG@10 belum dievaluasi standalone (perlu run eval_pipeline --reranker none).

---

## Reranker: BGE fine-tuned on Qwen qrels

**Model weights:** `fassabilf/umbrela-indo-ir-reranker-qwen`
**Base model:** `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa, 0.6B)
**Training data:** 53,727 triplets dari Qwen2.5-7B-generated qrels (Faiz)
**Input candidates:** `candidates/qwen3_test_top100.jsonl`
**Output TREC run:** `results/final/qwen3_bge_reranked.txt`

### Hasil Evaluasi (human qrels, test split, 960 queries)

| Metric     | Score  |
|------------|--------|
| nDCG@10    | 0.4585 |
| MAP@10     | 0.3504 |
| Recall@100 | 0.8508 |

Saved: `results/final/qwen3_qwenrk.json`

---

## Konteks: Perbandingan Semua Sistem (nDCG@10 @ human qrels)

| System                            | nDCG@10 | R@100  |
|-----------------------------------|---------|--------|
| BM25 (no rerank)                  | 0.3053  | 0.7634 |
| BM25 + Qwen-reranker (Faiz)       | 0.4478  | 0.7634 |
| BM25 + Gemma2-rk N=100 (Radit)    | 0.5178  | -      |
| **Qwen3-embed + Qwen-rk [KAMU]**  | **0.4585**  | **0.8508** |
| BGE-M3 (no rerank, Arvin)         | 0.5604  | 0.9047 |
| Hybrid RRF (no rerank, Arvin)     | 0.5191  | 0.9154 |

---

## Next Steps (RQ3 — Bias Analysis)

Matrix 2×2 yang harus diisi:

|                    | Qwen-reranker              | Gemma2-reranker            |
|--------------------|---------------------------|---------------------------|
| **Qwen3-embed**    | ✅ 0.4585 (selesai)        | ⬜ belum (butuh weights Radit) |
| **BGE-M3**         | ⬜ sedang jalan...         | ⬜ belum (butuh weights Radit) |

- [ ] Tanya Radit: di mana weights `reranker_full` / `reranker_100` (Gemma2-trained)?
- [ ] Implement `evaluation/bias_analysis.py` (masih stub kosong)
- [ ] Evaluasi Qwen3-embed standalone (--reranker none)
