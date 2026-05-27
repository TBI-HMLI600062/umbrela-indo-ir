# Hard-Negative Overlap: BGE-M3 vs Qwen3-embed

> Low overlap + high exclusive hard-neg rate means hard negatives are
> retriever-specific — which explains why cross-family rerankers struggle.

## K = 10
- Mean overlap rate: **49.2%** (4.9 / 10 docs shared per query)
- Avg confirmed-irrelevant docs in shared pool: 3.49
- Avg confirmed-irrelevant docs exclusive to BGE-M3: **4.65**
- Avg confirmed-irrelevant docs exclusive to Qwen3-embed: **4.91**
- % of BGE-M3 hard-negs not seen by Qwen3: **57.1%**

## K = 20
- Mean overlap rate: **47.0%** (9.4 / 20 docs shared per query)
- Avg confirmed-irrelevant docs in shared pool: 7.58
- Avg confirmed-irrelevant docs exclusive to BGE-M3: **10.16**
- Avg confirmed-irrelevant docs exclusive to Qwen3-embed: **10.44**
- % of BGE-M3 hard-negs not seen by Qwen3: **57.3%**

## K = 50
- Mean overlap rate: **43.4%** (21.7 / 50 docs shared per query)
- Avg confirmed-irrelevant docs in shared pool: 19.39
- Avg confirmed-irrelevant docs exclusive to BGE-M3: **27.93**
- Avg confirmed-irrelevant docs exclusive to Qwen3-embed: **28.19**
- % of BGE-M3 hard-negs not seen by Qwen3: **59.0%**

## K = 100
- Mean overlap rate: **40.5%** (40.5 / 100 docs shared per query)
- Avg confirmed-irrelevant docs in shared pool: 37.89
- Avg confirmed-irrelevant docs exclusive to BGE-M3: **59.23**
- Avg confirmed-irrelevant docs exclusive to Qwen3-embed: **59.44**
- % of BGE-M3 hard-negs not seen by Qwen3: **61.0%**
