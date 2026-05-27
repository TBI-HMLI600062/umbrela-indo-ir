## Judge × Retriever nDCG@10 Matrix

*(each row: judge's binarized qrels as ground truth, score≥2 = relevant)*

| Judge | BM25 | BGE-M3 | Qwen3-embed | Hybrid BM25+Qwen3 | Qwen3−BGE-M3 Δ |
|-------|-------|-------|-------|-------|----------------|
| Human | 0.3055 | 0.5604 | 0.4958 | 0.4603 | -0.0647 |
| SahabatAI-Gemma2 (zeroshot-bing) | 0.2724 | 0.5671 | 0.4907 | 0.4389 | -0.0764 |
| SahabatAI-Gemma2 (strict) | 0.2740 | 0.5515 | 0.4744 | 0.4315 | -0.0771 |
| SahabatAI-Llama3 (zeroshot-basic) | 0.2911 | 0.5349 | 0.4554 | 0.4405 | -0.0795 |
| SahabatAI-Llama3 (fewshot-basic) | 0.2768 | 0.5354 | 0.4670 | 0.4319 | -0.0684 |
| SahabatAI-Llama3 (fewshot-bing) | 0.2682 | 0.5430 | 0.4812 | 0.4330 | -0.0618 |
| DeepSeek | 0.2737 | 0.6107 | 0.5368 | 0.4554 | -0.0739 |
| ChatGPT (GPT-4o) | 0.2398 | 0.5486 | 0.4776 | 0.4010 | -0.0709 |