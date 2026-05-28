# Error Analysis — UMBRELA-Indo-IR
## A. Query Difficulty Distribution
| Category | Criterion | Count | % |
|----------|-----------|-------|---|
| Hard | max nDCG@10 < 0.1 | 73 | 7.6% |
| Medium | 0.1 ≤ max nDCG@10 < 0.3 | 67 | 7.0% |
| Easy | max nDCG@10 ≥ 0.3 | 820 | 85.4% |

**Top-10 hardest queries** (by max nDCG@10 across all systems):

| qid | max_nDCG@10 | BGE-M3 | Qwen3-embed | Hybrid BM25+Qwen3 | BM25 |
|-----|------------|---|---|---|---|
| 19 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 22 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 46 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 109 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 161 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 358 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 361 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 419 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 450 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 559 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## B. Judge Disagreement per Query
Computed over 960 queries where ≥2 judges evaluated the same doc.

**Top-10 most disagreed queries:**

| qid | mean_disagreement | n_docs | frac_docs_disagreed |
|-----|-------------------|--------|---------------------|
| 3594 | 0.2248 | 10 | 1.0 |
| 4865 | 0.2179 | 10 | 1.0 |
| 1091 | 0.2176 | 9 | 1.0 |
| 5009 | 0.2166 | 10 | 1.0 |
| 1607 | 0.2161 | 10 | 0.9 |
| 1909 | 0.2147 | 10 | 1.0 |
| 5007 | 0.2133 | 10 | 1.0 |
| 3315 | 0.213 | 10 | 1.0 |
| 5692 | 0.2115 | 10 | 1.0 |
| 919 | 0.2106 | 10 | 1.0 |

## C. Reranker Failure Analysis
System: Qwen3-embed + BGE-hardneg-rk vs Qwen3-embed baseline.
Evaluated on 960 queries with human qrels.

| Category | Criterion | Count | % |
|----------|-----------|-------|---|
| Big failure | δ < -0.2 | 228 | 23.8% |
| Neutral | -0.05 ≤ δ ≤ 0.05 | 291 | 30.3% |
| Big gain | δ > 0.2 | 161 | 16.8% |

**Top-10 worst reranker failures:**

| qid | ndcg_baseline | ndcg_reranked | delta |
|-----|---------------|---------------|-------|
| 1770 | 1.0 | 0.0 | -1.0 |
| 2329 | 1.0 | 0.0 | -1.0 |
| 3738 | 1.0 | 0.0 | -1.0 |
| 3901 | 1.0 | 0.0 | -1.0 |
| 4123 | 1.0 | 0.0 | -1.0 |
| 4180 | 1.0 | 0.0 | -1.0 |
| 4699 | 1.0 | 0.0 | -1.0 |
| 4979 | 1.0 | 0.0 | -1.0 |
| 5629 | 1.0 | 0.0 | -1.0 |
| 5249 | 1.0 | 0.1413 | -0.8587 |

