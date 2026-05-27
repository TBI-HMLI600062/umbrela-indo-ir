| System | Retriever | Reranker | nDCG@10 | Δ abs | Δ % | Recall@100 |
|--------|-----------|----------|---------|-------|-----|------------|
| BM25 only | BM25 | — | **0.3053** | — | — | 0.7634 |
| BGE-M3 only | BGE-M3 | — | **0.5604** | — | — | 0.9047 |
| Hybrid BM25+Qwen3 only | Hybrid BM25+Qwen3 | — | **0.5191** | — | — | 0.9154 |
| BGE-M3 + Qwen-rk | BGE-M3 | Qwen-rk | 0.4495 | -0.1109 | -19.8% | 0.9047 |
| BGE-M3 + Gemma2-rk | BGE-M3 | Gemma2-rk | 0.5111 | -0.0493 | -8.8% | 0.9047 |
| BGE-M3 + Qwen3-hardneg-rk | BGE-M3 | Qwen3-hardneg-rk | 0.5276 | -0.0328 | -5.9% | 0.9047 |
| BGE-M3 + BGE-hardneg-rk | BGE-M3 | BGE-hardneg-rk | 0.5659 | +0.0055 | +1.0% | 0.9047 |
| Qwen3-embed + Qwen-rk | Qwen3-embed | Qwen-rk | 0.4585 | — | — | 0.8508 |
| Qwen3-embed + Gemma2-rk | Qwen3-embed | Gemma2-rk | 0.5160 | — | — | 0.8508 |
| Qwen3-embed + BGE-hardneg-rk | Qwen3-embed | BGE-hardneg-rk | 0.5882 | — | — | 0.8508 |
| Hybrid BM25+Qwen3 + none | Hybrid BM25+Qwen3 | none | 0.4603 | -0.0588 | -11.3% | 0.8792 |
| Hybrid BM25+Qwen3 + Qwen3-hardneg-rk | Hybrid BM25+Qwen3 | Qwen3-hardneg-rk | 0.5306 | +0.0115 | +2.2% | 0.8792 |