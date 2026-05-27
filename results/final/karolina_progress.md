# Karolina — Hasil Retrieval & Reranker
**Last updated: 2026-05-27 (all evals complete)**

---

## Semua Hasil Eksperimen (test split, 960 queries, human qrels)

| System | First-Stage | Reranker | nDCG@10 | Δ abs | Δ % | Recall@100 | Status |
|--------|-------------|----------|---------|-------|-----|------------|--------|
| BM25 only | BM25 | — | 0.3053 | — | — | 0.7634 | Arvin |
| BGE-M3 only | BGE-M3 | — | **0.5604** | — | — | 0.9047 | Arvin |
| Hybrid BM25+BGE-M3 only | Hybrid | — | 0.5191 | — | — | 0.9154 | Arvin |
| Qwen3-embed only (no instr) | Qwen3 | — | 0.4958 | — | — | 0.8508 | Karol ✅ |
| Hybrid BM25+Qwen3 only | BM25+Qwen3 RRF | — | 0.4603 | — | — | 0.8792 | Karol ✅ |
| BM25 + Qwen-rk | BM25 | Qwen-rk | 0.4478 | +0.1425 | +46.7% | 0.7634 | Faiz ✅ |
| BM25 + Gemma2-rk (N=100) | BM25 | Gemma2-rk | 0.5178 | +0.2125 | +69.6% | 0.7634 | Radit ✅ |
| **BGE-M3 + Qwen-rk** | BGE-M3 | Qwen-rk | **0.4495** | **-0.1109** | **-19.8%** | 0.9047 | Karol ✅ |
| **Qwen3-embed + Qwen-rk** | Qwen3 | Qwen-rk | **0.4585** | **-0.0373** | **-7.5%** | 0.8508 | Karol ✅ |
| Hybrid BM25+Qwen3 + Qwen-rk | BM25+Qwen3 | Qwen-rk | 0.4604 | +0.0001 | +0.0% | 0.8792 | Karol ✅ |
| **BGE-M3 + Qwen3-hardneg-rk** | BGE-M3 | Qwen3-hardneg-rk | **0.5276** | **-0.0328** | **-5.9%** | 0.9047 | Karol ✅ |
| **Hybrid + Qwen3-hardneg-rk** | BM25+Qwen3 | Qwen3-hardneg-rk | **0.5306** | **+0.0703** | **+15.3%** | 0.8792 | Karol ✅ |
| BGE-M3 + Gemma2-rk | BGE-M3 | Gemma2-rk | 0.5111 | -0.0493 | -8.8% | 0.9047 | Karol ✅ |
| Qwen3-embed + Gemma2-rk | Qwen3 | Gemma2-rk | 0.5160 | +0.0202 | +4.1% | 0.8508 | Karol ✅ |
| **BGE-M3 + BGE-hardneg-rk** | BGE-M3 | BGE-hardneg-rk | **0.5659** | **+0.0055** | **+1.0%** | 0.9047 | Karol ✅ |
| **Qwen3-embed + BGE-hardneg-rk** | Qwen3 | BGE-hardneg-rk | **0.5882** | **+0.0924** | **+18.6%** | 0.8508 | Karol ✅ |

---

## Instruction Prefix — Val Split Evaluation (2026-05-26)

> **Konteks:** Instruction prefix sebelumnya dianggap gagal (0 hits) karena incompatibilitas
> FAISS index lama. Index baru (`embeddings/qwen3-embed-4b/`) di-encode ulang dengan benar.

**Val split, 814 queries, human qrels:**

| Setup | nDCG@10 | Recall@10 | Recall@100 | MRR |
|---|---|---|---|---|
| no-instruction | 0.4336 | 0.4604 | 0.7910 | 0.5498 |
| **with-instruction** | **0.4923** | **0.5238** | **0.8614** | **0.6095** |
| BM25 baseline | 0.2938 | 0.3290 | 0.7257 | 0.3658 |

**Delta: +0.059 nDCG@10 | +0.070 Recall@100** — signifikan.
→ Test split baseline (`qwen3_test_top100.jsonl`) memakai **no instruction** (lama).
→ Semua hasil Qwen3 di tabel atas adalah lower bound; versi with-instruction akan lebih tinggi.

---

## RQ3 — Bias Analysis: Rencana & Status

### Pertanyaan Penelitian
> Apakah pilihan LLM judge/reranker dari family X memberikan keuntungan sistematis
> untuk retriever dari family yang sama? Dan apakah *training signal* (LLM qrels vs
> hard-negative mining) lebih menentukan daripada family-alignment?

### Temuan Kunci Sejauh Ini

**1. Qwen-rk (dilatih dari LLM qrels) merusak kedua retriever:**
- BGE-M3 + Qwen-rk: **-19.8%** vs baseline → reranker aktif merusak retrieval terbaik
- Qwen3-embed + Qwen-rk: **-7.5%** → lebih "lunak" ke Qwen3, tapi masih negatif
- Ini adalah *bukti same-family bias yang asimetris*: Qwen-rk merusak BGE-M3 lebih parah

**2. Qwen3-hardneg-rk (dilatih dari hard-negative mining) = berbeda 180 derajat:**
- BGE-M3 + Qwen3-hardneg-rk: **-5.9%** (hanya sedikit negatif)
- Hybrid + Qwen3-hardneg-rk: **+15.3%** (signifikan positif)
- **Training signal lebih menentukan dari family alignment**

**3. Recall@100 tidak berubah saat reranking:**
- Recall@100 di tiap retriever tetap sama apapun rerankernya (karena reranker hanya
  mengubah urutan, tidak menambah/mengurangi kandidat)
- BGE-M3 memberi reranker **0.9047** recall ceiling vs Qwen3's **0.8508**

### Analisis yang Perlu Dilakukan

| # | Analisis | Tool/Script | Status | Output |
|---|----------|-------------|--------|--------|
| A | **Result matrix + delta heatmap** | `bias_analysis.py --mode aggregate` | ✅ Done | `bias_analysis/delta_heatmap.png` |
| B | **Recall@100 bar chart** | `bias_analysis.py --mode aggregate` | ✅ Done | `bias_analysis/recall_comparison.png` |
| C | **Per-query nDCG@10 violin** (first-stage only, CPU) | `bias_analysis.py --mode perquery` | 🔜 Ready to run | `bias_analysis/perquery_violin.png` |
| D | **nDCG@K breakdown** (K=1,3,5,10) | `bias_analysis.py --mode perquery` | 🔜 Ready to run | `bias_analysis/ndcg_at_k.png` |
| E | **Hard-negative overlap** BGE-M3 vs Qwen3 | `bias_analysis.py --mode overlap` | 🔜 Ready to run | `bias_analysis/hardneg_overlap_report.md` |
| F | **Gemma2-rk eval** (BGE-M3 + Qwen3-embed) | `eval_pipeline.py` × 2 | 🔜 Run after training | `bgem3_gemma2rk_test.json` |
| G | **BGE-hardneg-rk eval** (BGE-M3 + Qwen3-embed) | `eval_pipeline.py` × 2 | 🔄 Waiting for training | TBD |
| H | **Win/loss/tie chart** per system vs BGE-M3 baseline | `bias_analysis.py --mode perquery` | 🔜 After GPU free | `bias_analysis/win_loss.png` |
| I | **Rank disruption (Kendall-τ)** | `bias_analysis.py` `kendall_tau_disruption()` | 🔜 After eval done | `bias_analysis/rank_disruption.png` |

### Commands untuk menjalankan analisis yang belum dilakukan

```bash
# C, D — Per-query first-stage (CPU, bisa sekarang)
python evaluation/bias_analysis.py --mode perquery \
    --systems "BGE-M3,Qwen3-embed,Hybrid BM25+Qwen3,BM25" \
    --output results/final/bias_analysis/

# E — Hard-negative overlap (CPU, bisa sekarang)
python evaluation/bias_analysis.py --mode overlap \
    --output results/final/bias_analysis/

# F — Gemma2-rk eval (setelah training BGE-hardneg selesai dan GPU bebas)
python evaluation/eval_pipeline.py \
    --first-stage bge_m3 \
    --reranker arya-raditya/bge-reranker-gemma2-n100 \
    --output results/final/bgem3_gemma2rk_test.json

python evaluation/eval_pipeline.py \
    --first-stage qwen3_embed \
    --reranker arya-raditya/bge-reranker-gemma2-n100 \
    --output results/final/qwen3_gemma2rk_test.json

# G — BGE-hardneg-rk eval (setelah training selesai)
python evaluation/eval_pipeline.py \
    --first-stage bge_m3 \
    --reranker results/models/reranker_bgem3_hardneg/ \
    --output results/final/bgem3_bgem3hardnegrk_test.json

python evaluation/eval_pipeline.py \
    --first-stage qwen3_embed \
    --reranker results/models/reranker_bgem3_hardneg/ \
    --output results/final/qwen3_bgem3hardnegrk_test.json

# H, I — Win/loss + rank disruption (setelah semua eval selesai)
# Tambahkan reranked candidates ke perquery mode
# (butuh TREC run files — akan dihasilkan oleh eval_pipeline.py)
```

### Narrative untuk Paper (Section RQ3)

Setelah semua data terkumpul, argumen utama yang perlu dibangun:

1. **Same-family bias bersifat asimetris**: Qwen-rk merusak BGE-M3 lebih parah (-19.8%)
   dari merusak Qwen3-embed (-7.5%). Ini menunjukkan preferensi sistematis Qwen-rk
   terhadap pola distribusi Qwen3-embed.

2. **Training signal lebih menentukan dari family alignment**: Qwen3-hardneg-rk
   (dilatih dari hard-negative mining Qwen3) BUKAN Qwen-rk, yang berhasil meningkatkan
   performa. Perbedaan kuncinya: Qwen-rk dilatih dari LLM-judged qrels (signal tentang
   apa yang relevan), sedangkan Qwen3-hardneg-rk dilatih dari kesalahan aktual retriever
   (signal tentang di mana retriever gagal).

3. **Recall ceiling menentukan potensi maksimum**: BGE-M3 memberi reranker pool 0.9047
   vs Qwen3's 0.8508. Namun Qwen-rk tidak bisa memanfaatkan pool lebih besar milik BGE-M3 —
   bukti bahwa representasi score antara retriever dan reranker tidak kompatibel.

4. **Pending — setelah analisis lengkap**:
   - Apakah BGE-hardneg-rk membantu BGE-M3 lebih dari Qwen3? (cross-inference test)
   - Apakah Gemma2-rk mengikuti pola yang sama atau berbeda?
   - Kendall-τ: apakah reranker yang "merusak" justru punya τ rendah (banyak mengubah ranking)?

---

## BGE-M3 Hard Negative Mining (2026-05-27)

Training sedang berjalan: `results/models/reranker_bgem3_hardneg/`

**Stats hard negative mining:**
- Queries: 3,257 (semua)
- Explicit label=0 hard negs: 8,416
- Unannotated hard negs: 49,486
- **Total triplets: 130,961** (vs Qwen3: 53,727 — 2.4x lebih banyak)
  → BGE-M3 punya lebih banyak false positives di top-20 daripada Qwen3

**Training command:**
```bash
python3 reranker/train.py \
    --training-data results/reranker_data/bgem3_hardneg/ \
    --model BAAI/bge-reranker-v2-m3 \
    --output results/models/reranker_bgem3_hardneg/ \
    --epochs 3 --batch-size 16 --lr 2e-5 --max-length 256 --bf16 \
    --hf-repo karolinajocelyn/umbrela-indo-ir-models
```

---

## File Locations

| File | Isi |
|------|-----|
| `candidates/qwen3_test_top100.jsonl` | Qwen3-embed top-100, test, **no instruction** |
| `candidates/qwen3_train_top100.jsonl` | Qwen3-embed top-100, train, **with instruction** |
| `candidates/qwen3_instruct_val_top100.jsonl` | Qwen3-embed top-100, val, with instruction |
| `candidates/bgem3_train_top100.jsonl` | BGE-M3 top-100, train split (3257 queries) |
| `candidates/bgem3_test_top100.jsonl` | BGE-M3 top-100, test split (960 queries) |
| `results/reranker_data/bgem3_hardneg/` | Hard negatives dari BGE-M3 (130,961 triplets) |
| `results/reranker_data/qwen/` | Hard negatives dari Qwen3 (53,727 triplets) |
| `results/models/reranker_qwen/` | Trained reranker (Qwen3 hardneg), pushed ke HF |
| `results/models/reranker_bgem3_hardneg/` | 🔄 Training... BGE-M3 hardneg reranker |
| `results/final/bias_analysis/` | Output bias analysis (heatmap, charts, JSONs) |
| `evaluation/bias_analysis.py` | Script bias analysis (3 modes: aggregate/perquery/overlap) |

---

## Blocked / Menunggu

| Item | Status | Action |
|------|--------|--------|
| Gemma2-rk eval (BGE-M3 + Qwen3) | 🔜 Tunggu GPU bebas | Jalankan setelah training selesai |
| BGE-hardneg-rk eval | 🔄 Tunggu training | ~3 jam dari 09:53 WIB |
| RQ3 matrix lengkap | 🔜 Tunggu F + G | Semua 4 Gemma2-rk dan BGE-hardneg-rk |
| Radit `model.safetensors` | ❌ Blocked | Tanya Radit untuk file dari `reranker_100` |
