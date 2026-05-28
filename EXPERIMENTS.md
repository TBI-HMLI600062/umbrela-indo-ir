# UMBRELA-Indo-IR — Catatan Eksperimen Lengkap

**Tugas Kelompok IR Genap 2025/2026**  
**Tim:** Faiz · Arvin · Karolina (Karol) · Vincent (Radit) · Radit (Arya/Raditya)

> Dokumen ini merangkum **semua eksperimen** yang telah dilakukan, mulai dari
> qrel generation, evaluasi retrieval, training reranker, fine-tuning LLM judge,
> hingga extended analysis. Ditulis sebagai catatan teknis yang cukup rinci untuk
> dapat mereproduksi semua hasil.

---

## Daftar Isi

1. [Dataset & Setup](#1-dataset--setup)
2. [RQ1 — LLM Judge Agreement](#2-rq1--llm-judge-agreement)
3. [RQ2 — Retrieval & Reranking](#3-rq2--retrieval--reranking)
4. [RQ3 — Bias Analysis](#4-rq3--bias-analysis)
5. [Extended Analysis](#5-extended-analysis)
6. [Kontribusi Per Anggota](#6-kontribusi-per-anggota)
7. [Lokasi File & Artifacts](#7-lokasi-file--artifacts)

---

## 1. Dataset & Setup

### 1.1 MIRACL-ID

| Item | Detail |
|------|--------|
| Dataset | MIRACL (Multilingual Information Retrieval Across a Continuum of Languages) — split Indonesia |
| Corpus | ~1.44 juta passage dari Wikipedia Bahasa Indonesia (1,446,315 passage, 446,330 artikel) |
| Format corpus | JSONL, field: `docid`, `doc` |
| Dev split | **960 queries** dengan human qrels — ini adalah MIRACL **dev** set (bukan "test"; MIRACL test tidak punya public qrels) |
| Train split | **3,257 queries** (subset internal dari MIRACL train 4,071 queries; dipakai untuk hard negative mining & reranker training) |
| Val split | **814 queries** (subset internal lain dari MIRACL train 4,071 queries; dipakai untuk hyperparameter tuning) |
| Human qrels | **3,088 pairs** total di dev split, semua score=1 (positive-only format) |
| Annotation pool | **9,668 pairs** total di dev split (3,088 positif + ~6,580 negatif) — keduanya dianotasi manual oleh native speaker |
| Lokasi | `data/miracl-id/` |

**Asal-usul 9,668 pairs:** MIRACL menggunakan **ensemble baseline retrieval system** (BM25 + mDPR via Pyserini) untuk menghasilkan kandidat awal, lalu native speaker hired menilai relevansi setiap pasang. Penting: **negative passages juga dianotasi eksplisit** oleh native speaker (bukan otomatis diambil dari dokumen yang tidak masuk positive list). Semua judgments (positif + negatif) tersimpan di `data/miracl-id/qrels/candidates/test.jsonl` dengan field `positive_docids` dan `negative_docids`. Total = 9,668 pairs untuk dev split. Angka train (33,076) dan val (8,282) juga berasal dari annotation pool MIRACL train yang sama, di-split secara internal: 33,076 + 8,282 = 41,358 = total judgments MIRACL train resmi ✓.

**Penting tentang format human qrels:** MIRACL human qrels hanya menyimpan dokumen yang dinilai relevan (positive-only). Dokumen yang tidak muncul di qrels dianggap tidak relevan secara implisit. Ini berbeda dari LLM qrels yang menyimpan semua 9,668 pasang query-dokumen beserta skor 0–3.

### 1.2 Lingkungan & Tools

| Tool | Versi / Keterangan |
|------|--------------------|
| Python | 3.12 |
| sentence-transformers | 5.5.0 |
| ranx | evaluasi IR (nDCG@10, MAP@10, Recall@100) |
| bm25s | BM25 retrieval tanpa Java |
| faiss | Dense retrieval (FAISS IVF index) |
| vLLM | Inference LLM (Llama3, Gemma2) |
| TRL | SFTTrainer untuk LoRA fine-tuning |
| scikit-learn | Cohen's κ |
| Platform training | vast.ai (GPU A100/H100) |

### 1.3 Prompt UMBRELA

Proyek ini menggunakan prompt UMBRELA yang diadaptasi ke Bahasa Indonesia. Ada beberapa varian prompt yang diuji (lihat RQ1 ablation):

- **zeroshot_bing** (default, terbaik): Prompt zero-shot dengan gaya Bing, meminta skor 0–3
- **zeroshot_bing_strict**: Versi ketat (kriteria relevansi lebih strict)
- **fewshot_bing**: Versi few-shot dengan contoh
- **zeroshot_basic**: Prompt minimal tanpa konteks web-search
- **fewshot_basic**: Few-shot versi minimal

---

## 2. RQ1 — LLM Judge Agreement

**Pertanyaan:** LLM judge mana yang paling konsisten dengan penilaian human assessor untuk IR berbahasa Indonesia?

**Metrik:** Cohen's κ (inter-rater agreement, chance-corrected), positif rate LLM vs human, jumlah pairs

### 2.1 Qrel Generation (Faiz — Qwen2.5-7B)

**Model:** `Qwen/Qwen2.5-7B-Instruct`

**Proses:**
1. Ambil pasang query–passage dari **annotation pool MIRACL** (`data/miracl-id/qrels/candidates/`) — pasang yang sama yang dinilai oleh human annotator (9,668 pairs untuk dev split MIRACL, dihasilkan dari ensemble BM25+mDPR, bukan BM25 fresh project ini)
2. Format setiap pasang (query, passage) ke dalam prompt UMBRELA
3. Jalankan inference dengan vLLM, ekstrak skor 0–3
4. Simpan sebagai TREC qrels format: `qid 0 docid score`

**Hasil per split:**

| Split | n_pairs | κ vs Human | LLM pos rate | Human pos rate |
|-------|---------|-----------|--------------|----------------|
| Train | 33,076 | 0.3039 | 30.02% | 30.17% |
| Val | 8,282 | 0.2860 | 29.11% | 30.50% |
| Test | **9,668** | **0.3767** | **30.74%** | **31.94%** |

**Threshold:** τ=2 (binary: score ≥ 2 → relevan). Ini dikonfirmasi optimal via calibration sweep (lihat §2.6).

**File:**
- `data/miracl-id/results/qrels/qwen_train.txt`, `qwen_val.txt`, `qwen_test.txt`
- `results/final/kappa_qwen_train.csv`, `kappa_qwen_val.csv`, `kappa_qwen_test.csv`

---

### 2.2 Qrel Generation (Vincent/Radit — SahabatAI-Gemma2)

**Model:** `SahabatAI/sahabatai-v1-gemma2-9b-instruct` (Gemma2-9B, Indonesian-tuned)

**Proses:** Inference menggunakan vLLM di vast.ai (GPU A100).

**Hasil per split:**

| Split | n_pairs | κ vs Human | LLM pos rate | Human pos rate |
|-------|---------|-----------|--------------|----------------|
| Train | 33,076 | 0.3222 | 40.77% | 30.17% |
| Val | 8,282 | 0.3204 | 39.37% | 30.50% |
| Test | **9,668** | **0.3763** | **41.23%** | **31.94%** |

**Catatan:** Gemma2 sedikit overpredicts (pos rate 41% vs human 32%) tapi κ kompetitif dengan Qwen.

**File:**
- `results/qrels/sahabat_llama_test.txt` *(nama file historical, isinya Gemma2)*
- `results/final/kappa_gemma.csv`

---

### 2.3 Qrel Generation (Vincent — SahabatAI-Llama3)

**Model:** `SahabatAI/sahabatai-v1-llama3-8b-instruct` (Llama3-8B, Indonesian-tuned)

**Proses:** Inference unquantized di vast.ai.

**Hasil:**

| Prompt Mode | n_pairs | κ vs Human | LLM pos rate |
|-------------|---------|-----------|--------------|
| default (zeroshot_bing) | 9,668 | **0.2103** | **66.66%** |
| strict | 9,668 | **0.3652** | **38.79%** |

**Temuan kritis:** Tanpa prompt strict, Llama3 menilai 67% dokumen sebagai relevan → κ sangat rendah (0.21). Dengan prompt strict yang membatasi output format, κ naik ke 0.37 dan pos rate turun ke 39% (lebih mendekati human).

**File:**
- `results/qrels/sahabat_llama_test.txt` (default)
- `results/qrels_strict/sahabat_llama_strict_test.txt` (strict)
- `results/final/kappa_llama.csv`, `kappa_llama_strict.csv`

---

### 2.4 Qrel Generation — DeepSeek-V3 & ChatGPT (Arvin)

**Model DeepSeek:** `deepseek-ai/DeepSeek-V3` (API)  
**Model ChatGPT:** `gpt-4o-mini` (API)

**Hasil:**

| Judge | n_pairs | κ vs Human | LLM pos rate |
|-------|---------|-----------|--------------|
| DeepSeek-V3 | 9,668 | **0.4219** | 27.99% |
| ChatGPT (gpt-4o-mini) | 9,668 | 0.3869 | 25.44% |

**DeepSeek-V3 = judge terbaik** (κ=0.42), paling konservatif (28% pos rate), paling mendekati human (32%).
ChatGPT complete 9,668 pairs (κ=0.3869, agreement 74.8%).

**File:**
- `results/qrels/deepseek_test.txt`
- `results/qrels/chatgpt_test.txt`
- `results/final/kappa_deepseek_test.csv`

---

### 2.5 Prompt Ablation (Faiz — Qwen2.5-7B, 5 Variant)

**Pertanyaan:** Apakah format prompt berpengaruh signifikan terhadap κ?

**Setup:** Semua 5 varian dijalankan di test split (9,668 pairs) dengan model Qwen2.5-7B-Instruct.

| Prompt Mode | κ | LLM pos rate |
|-------------|---|--------------|
| **zeroshot_bing** ← baseline terbaik | **0.3767** | 30.74% |
| zeroshot_bing_strict | 0.3218 | 23.13% |
| fewshot_bing | 0.2720 | 16.39% |
| zeroshot_basic | 0.2393 | 12.65% |
| fewshot_basic | 0.2254 | 11.46% |

**Kesimpulan:** `zeroshot_bing` terbaik. Few-shot justru menurunkan κ secara signifikan — kemungkinan karena contoh few-shot membuat model terlalu selektif (pos rate 11–16%).

**File:** `results/final/kappa_prompt_ablation.csv`

---

### 2.6 Prompt Ablation — Gemma2 & Llama3 (Vincent, extended)

Ablation dilanjutkan untuk SahabatAI-Gemma2 dan SahabatAI-Llama3 menggunakan vLLM.

**SahabatAI-Gemma2 (vLLM):**

| Prompt Mode | κ | LLM pos rate |
|-------------|---|--------------|
| **zeroshot_bing** | **0.3778** | 41.03% |
| zeroshot_bing_strict | 0.3753 | 45.29% |
| zeroshot_basic | 0.3628 | 45.36% |
| fewshot_bing | 0.3843 | 41.62% |
| fewshot_basic | 0.3121 | 53.57% |
| zeroshot_basic (partial, 2,151 pairs) | 0.3491 | 29.66% |

**SahabatAI-Llama3 (vLLM):**

| Prompt Mode | κ | LLM pos rate |
|-------------|---|--------------|
| **fewshot_bing** | **0.3415** | 34.90% |
| fewshot_basic | 0.3328 | 44.89% |
| zeroshot_basic | 0.3039 | 58.06% |

**File:** `results/final/kappa_prompt_ablation_full.csv`

---

### 2.7 Calibration Analysis — Threshold Sweep (Faiz)

**Pertanyaan:** Apakah τ=2 sudah optimal untuk binarisasi skor LLM?

**Setup:** Sweep τ ∈ {1, 2, 3} pada val dan test split, Qwen2.5-7B.

| τ | κ (val) | κ (test) | LLM pos rate (test) | Optimal? |
|---|---------|---------|---------------------|---------|
| 1 | 0.2541 | 0.3255 | 52.14% | ✗ |
| **2** | **0.2860** | **0.3767** | **30.74%** | **✓** |
| 3 | 0.1459 | 0.1808 | 7.60% | ✗ |

**Kesimpulan:** τ=2 sudah optimal. Masalah κ bukan di thresholding tapi di per-pair reasoning.

**File:** `results/final/calibration_qwen.csv`
**Script:** `evaluation/calibrate.py`

---

### 2.8 LoRA Fine-tuning Qwen2.5-7B sebagai Judge (Faiz + Arvin)

**Pertanyaan:** Apakah fine-tuning Qwen2.5-7B dengan human qrels sebagai label meningkatkan κ?

**Data:**
- SFT training data: 33,076 pairs dari train split
- Format: `prompt` (query + passage formatted) → `response` (skor relevansi)
- Total: ~33k train examples + 8k val examples
- Lokasi data: `results/lora_data/qwen/`

**Training:**
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Method: LoRA via TRL `SFTTrainer`
- Adapter: `lora/` → `results/models/lora_qwen_smoke/`
- Epochs: 3, bfloat16
- HF push: `fassabilf/lora-qwen-miracl-id-smoke` (private)

**Hasil:**

| System | κ (test) | LLM pos rate |
|--------|---------|--------------|
| Qwen baseline (zeroshot_bing) | 0.3767 | 30.74% |
| **Qwen LoRA fine-tuned** | **0.3718** | **14.96%** |
| DeepSeek-V3 (best judge overall) | 0.4219 | 27.99% |

**Temuan:** LoRA fine-tuning tidak meningkatkan κ secara signifikan (0.3718 vs 0.3767, perbedaan kecil dalam batas noise). Model LoRA menjadi jauh lebih konservatif (pos rate turun dari 30.7% ke 15%), tetapi kappa tidak meningkat. Ini mengindikasikan bahwa sinyal biner dari human qrels tidak cukup kaya untuk mendorong improvement via LoRA; model butuh sinyal yang lebih nuanced.

**ORPO fine-tuning:** Script sudah disiapkan (`lora/train_orpo.py`), ORPO data prep sudah ada di `results/orpo_data/qwen/`. Experiment belum selesai (belum ada hasil final).

**File:** `results/final/kappa_qwen_lora_test.csv`
**Script:** `lora/train.py`, `lora/prepare_data.py`, `lora/train_orpo.py`, `lora/prepare_orpo_data.py`

---

### 2.9 Ringkasan RQ1 — Semua Judge

| Judge | Model | κ (test) | LLM pos rate | n_pairs |
|-------|-------|---------|--------------|---------|
| **DeepSeek-V3** | deepseek-ai/DeepSeek-V3 | **0.4219** | 27.99% | 9,668 |
| ChatGPT (gpt-4o-mini) | gpt-4o-mini | 0.3869 | 25.44% | 9,668 |
| SahabatAI-Gemma2 (zeroshot_bing) | sahabatai-v1-gemma2-9b | 0.3778 | 41.03% | 9,668 |
| Gemma2 (zeroshot_bing_strict) | sahabatai-v1-gemma2-9b | 0.3753 | 45.29% | 9,668 |
| **Qwen2.5-7B (zeroshot_bing)** | Qwen2.5-7B-Instruct | **0.3767** | 30.74% | 9,668 |
| Qwen2.5-7B LoRA fine-tuned | Qwen2.5-7B + LoRA | 0.3718 | 14.96% | 9,668 |
| Gemma2 (fewshot_bing) | sahabatai-v1-gemma2-9b | 0.3843 | 41.62% | 9,668 |
| Llama3-strict | sahabatai-v1-llama3-8b | 0.3652 | 38.79% | 9,668 |
| Gemma2 (zeroshot_basic) | sahabatai-v1-gemma2-9b | 0.3628 | 45.36% | 9,668 |
| Llama3-vllm-fewshot-bing | sahabatai-v1-llama3-8b | 0.3415 | 34.90% | 9,668 |
| Llama3-vllm-fewshot-basic | sahabatai-v1-llama3-8b | 0.3328 | 44.89% | 9,668 |
| Llama3-vllm-zeroshot-basic | sahabatai-v1-llama3-8b | 0.3039 | 58.06% | 9,668 |
| **Llama3-default** | sahabatai-v1-llama3-8b | **0.2103** | **66.66%** | **9,668** |

**Temuan utama RQ1:**
- DeepSeek-V3 adalah judge terbaik (κ=0.42), juga paling konservatif
- Qwen2.5-7B terbaik di antara open-source yang bisa dijalankan sendiri (κ=0.38)
- Llama3 wajib menggunakan prompt strict; tanpa itu κ hanya 0.21
- LoRA fine-tuning tidak memberikan improvement signifikan

---

## 3. RQ2 — Retrieval & Reranking

**Pertanyaan:** Bisakah qrels yang dihasilkan LLM melatih reranker yang meningkatkan BM25?

### 3.1 First-Stage Retrieval (Arvin)

**Sistem yang diimplementasikan:**

| Sistem | Library | Indexing | Keterangan |
|--------|---------|---------|------------|
| BM25 | bm25s | — | Tanpa Java, pure Python |
| BGE-M3 dense | BAAI/bge-m3 | FAISS IVF | XLM-RoBERTa multi-vector |
| Hybrid BM25+BGE-M3 | bm25s + FAISS | RRF | Reciprocal Rank Fusion |
| Qwen-embed | Qwen embedding (old) | FAISS | Legacy, rendah overlap human |
| Qwen3-embed | Qwen/Qwen3-Embedding-4B | FAISS | Tanpa instruction prefix (Karol) |
| Hybrid BM25+Qwen3 | bm25s + FAISS | RRF | Qwen3 + BM25 fusion (Karol) |

**Hasil retrieval (nDCG@10, human qrels, test split, 960 queries):**

| Retriever | nDCG@10 | MAP@10 | Recall@100 | Queries nDCG=0 |
|-----------|---------|--------|------------|----------------|
| **BGE-M3** | **0.5604** | 0.4529 | **0.9047** | 116/960 (12.1%) |
| Hybrid BM25+BGE-M3 | 0.5191 | — | **0.9154** | — |
| Qwen3-embed (no instr) | 0.4958 | 0.3913 | 0.8508 | 156/960 (16.2%) |
| Hybrid BM25+Qwen3 | 0.4603 | 0.3502 | 0.8792 | 161/960 (16.8%) |
| BM25 | 0.3053 | 0.2206 | 0.7634 | 335/960 (34.9%) |
| Qwen-embed (legacy) | ~0.007 | ~0.004 | ~0.04 | — |

**Catatan Qwen-embed legacy:** Nilai sangat rendah karena pool bias — kandidat Qwen-embed hampir tidak ada yang overlap dengan dokumen yang di-judge oleh human (pooling MIRACL dilakukan dari BM25/BGE pool, bukan Qwen-embed).

**Catatan Qwen3-embed:** Dievaluasi tanpa instruction prefix (lower bound). Dengan instruction prefix, nDCG@10 di val split naik +0.059 (dari 0.4336 → 0.4923) — jadi hasil test juga diperkirakan lebih tinggi ~0.05.

**File candidates:**
- `candidates/bm25_test_top100.jsonl`
- `candidates/bgem3_test_top100.jsonl`
- `candidates/hybrid_test_top100.jsonl` (BM25+BGE)
- `candidates/qwen_test_top100.jsonl` (Qwen-embed legacy)
- `candidates/qwen3_test_top100.jsonl` (Qwen3-embed, no instruction)
- `candidates/hybrid_bm25_qwen3_test_top100.jsonl`

**nDCG@K breakdown (4 retriever utama):**

| Retriever | @1 | @3 | @5 | @10 |
|-----------|-----|-----|-----|-----|
| BGE-M3 | 0.5563 | 0.5168 | 0.5189 | 0.5604 |
| Qwen3-embed | 0.4844 | 0.4537 | 0.4647 | 0.4958 |
| Hybrid BM25+Qwen3 | 0.3979 | 0.3914 | 0.4156 | 0.4603 |
| BM25 | 0.2437 | 0.2470 | 0.2618 | 0.3055 |

---

### 3.2 Qwen3-embed Instruction Prefix Ablation (Karol)

**Pertanyaan:** Apakah menambah instruction prefix ke Qwen3-embed signifikan?

**Val split, 814 queries:**

| Setup | nDCG@10 | Recall@10 | Recall@100 | MRR |
|-------|---------|-----------|------------|-----|
| no-instruction | 0.4336 | 0.4604 | 0.7910 | 0.5498 |
| **with-instruction** | **0.4923** | **0.5238** | **0.8614** | **0.6095** |
| BM25 baseline | 0.2938 | 0.3290 | 0.7257 | 0.3658 |

**Delta: +0.059 nDCG@10 | +0.070 Recall@100 | +0.060 MRR**

Ini sangat signifikan — instruction prefix memberikan improvement hampir seperti selisih antara Qwen3 dan Hybrid. Semua evaluasi test split Qwen3-embed adalah **lower bound**.

---

### 3.3 Reranker Training dari LLM Qrels — Size Ablation (Radit/Vincent)

**Pertanyaan:** Berapa banyak training data dari LLM qrels yang dibutuhkan untuk melatih reranker efektif?

**Setup:**
- First-stage: BM25 (baseline 0.3053)
- Base reranker model: `BAAI/bge-reranker-v2-m3` (XLM-RoBERTa for sequence classification)
- Training data: qrels SahabatAI-Gemma2 (format: triplet positif/negatif dari top-100 BM25)
- Loss: Binary Cross-Entropy
- Epochs: 3, batch 16, lr 5e-5

**Hasil size ablation (test split, BM25 first-stage):**

| N queries | N triplets | N train examples | nDCG@10 | MAP@10 | Recall@100 | Val AP (LLM) |
|-----------|-----------|-----------------|---------|--------|------------|--------------|
| **100** | 1,937 | **3,874** | **0.5178** | 0.4088 | 0.7634 | 0.865 |
| 300 | 5,285 | 10,570 | 0.4620 | 0.3491 | 0.7634 | 0.873 |
| 500 | 9,173 | 18,346 | 0.5011 | 0.3950 | 0.7634 | 0.905 |
| 1000 | 18,509 | 37,018 | 0.4072 | 0.2993 | 0.7634 | 0.916 |
| full (3,257) | 60,750 | 121,500 | 0.3993 | 0.2917 | 0.7634 | 1.000 |

**Temuan utama:**
- **N=100 menghasilkan nDCG@10 tertinggi (0.5178)** — lebih baik dari N=full (0.3993)
- Val AP pada LLM qrels naik monoton ke 100% → reranker **overfitting** pada noise LLM judge
- BM25 + Gemma2-rk (N=100): 0.5178 ≈ BGE-M3 baseline 0.5604 (selisih 0.04)
- **Counterintuitive:** Lebih banyak data LLM = lebih rendah nDCG@10 vs human qrels

**Model:** Semua 5 model tersimpan di `results/models/reranker_{100,300,500,1000,full}/`  
**File:** `results/final/ablation_summary.csv`, `size_{100,300,500,1000,full}.json`

---

### 3.4 Reranker Training dari LLM Qrels — Qwen (Faiz)

**Setup:**
- First-stage: BM25
- Base model: `BAAI/bge-reranker-v2-m3`
- Training data: Qwen2.5-7B qrels (full dataset: 53,727 triplets)
- Epochs: 3, bf16, max-length 256, lr 2e-5

**Hasil:**

| System | nDCG@10 | MAP@10 | Recall@100 |
|--------|---------|--------|------------|
| BM25 baseline | 0.3053 | — | 0.7634 |
| **BM25 + Qwen-rk** | **0.4478** | — | **0.7634** |

**Delta vs BM25: +0.1425 (+46.7%)**

Model: `results/models/reranker_qwen/` → HF: `fassabilf/umbrela-indo-ir-reranker-qwen` (public)

---

### 3.5 Hard Negative Mining (Karol)

**Pertanyaan:** Apakah reranker yang dilatih dari hard negatives (bukan LLM qrels) lebih efektif?

**Konsep hard negative mining:**
- Ambil top-20 kandidat retriever
- Dokumen yang ada di top-20 tapi dinilai tidak relevan oleh human (atau tidak muncul di human qrels) = hard negative
- Ini lebih informatif dari random negative karena secara semantik mirip query

**Hard negatives dari Qwen3-embed:**
- Queries: 3,257 (semua train)
- Triplets: **53,727**
- Lokasi: `results/reranker_data/qwen/`
- Model dilatih: `results/models/reranker_qwen/` (dengan Qwen3 hard negatives)

**Hard negatives dari BGE-M3:**
- Queries: 3,257 (semua train)
- Explicit label=0 hard negs: 8,416
- Unannotated hard negs: 49,486
- Total triplets: **130,961** (2.4× lebih banyak dari Qwen3 hard negs)
- Lokasi: `results/reranker_data/bgem3_hardneg/`
- Model dilatih: `results/models/reranker_bgem3_hardneg/` → HF: `karolinajocelyn/umbrela-indo-ir-models` (BGE-hardneg-rk)

**Mengapa BGE-M3 punya lebih banyak hard negs?**  
BGE-M3 punya Recall@100 lebih tinggi (0.9047 vs 0.8508) → lebih banyak dokumen masuk top-20 → lebih banyak false positives yang bisa dijadikan hard negative.

**Script:** `reranker/mine_hard_negatives.py`

---

### 3.6 Hasil Lengkap Reranking (semua kombinasi, Karol)

**Evaluasi:** nDCG@10 dengan human qrels, test split, 960 queries.

| First-Stage | Reranker | Training Signal | nDCG@10 | Δ abs | Δ % | Recall@100 |
|-------------|----------|----------------|---------|-------|-----|------------|
| BM25 | — | — | 0.3053 | — | — | 0.7634 |
| BM25 | Qwen-rk | Qwen LLM qrels | **0.4478** | +0.1425 | +46.7% | 0.7634 |
| BM25 | Gemma2-rk (N=100) | Gemma2 LLM qrels | **0.5178** | +0.2125 | +69.6% | 0.7634 |
| BGE-M3 | — | — | 0.5604 | — | — | 0.9047 |
| BGE-M3 | Qwen-rk | Qwen LLM qrels | 0.4495 | −0.1109 | −19.8% | 0.9047 |
| BGE-M3 | Gemma2-rk | Gemma2 LLM qrels | 0.5111 | −0.0493 | −8.8% | 0.9047 |
| BGE-M3 | Qwen3-hardneg-rk | Hard negs (Qwen3) | 0.5276 | −0.0328 | −5.9% | 0.9047 |
| BGE-M3 | **BGE-hardneg-rk** | **Hard negs (BGE-M3)** | **0.5659** | **+0.0055** | **+1.0%** | 0.9047 |
| Qwen3-embed | — | — | 0.4958 | — | — | 0.8508 |
| Qwen3-embed | Qwen-rk | Qwen LLM qrels | 0.4585 | −0.0373 | −7.5% | 0.8508 |
| Qwen3-embed | Gemma2-rk | Gemma2 LLM qrels | 0.5160 | +0.0202 | +4.1% | 0.8508 |
| Qwen3-embed | Qwen3-hardneg-rk | Hard negs (Qwen3) | — | — | — | — |
| Qwen3-embed | **BGE-hardneg-rk** | **Hard negs (BGE-M3)** | **0.5882** | **+0.0924** | **+18.6%** | 0.8508 |
| Hybrid BM25+Qwen3 | — | — | 0.4603 | — | — | 0.8792 |
| Hybrid BM25+Qwen3 | Qwen-rk | Qwen LLM qrels | 0.4604 | +0.0001 | ~0% | 0.8792 |
| Hybrid BM25+Qwen3 | Qwen3-hardneg-rk | Hard negs (Qwen3) | **0.5306** | +0.0703 | +15.3% | 0.8792 |

**Recall@100 tidak berubah setelah reranking** — reranker hanya mengubah urutan, tidak menambah/mengurangi kandidat.

**File JSON hasil eval:**
- `results/final/bm25_qwen_rk.json` — BM25 + Qwen-rk
- `results/final/bgem3_qwenrk.json` — BGE-M3 + Qwen-rk
- `results/final/bgem3_gemma2rk_test.json` — BGE-M3 + Gemma2-rk
- `results/final/bgem3_qwen3hardnegrk_test.json` — BGE-M3 + Qwen3-hardneg-rk
- `results/final/bgem3_bgem3hardnegrk_test.json` — BGE-M3 + BGE-hardneg-rk ★
- `results/final/qwen3_qwenrk.json` — Qwen3 + Qwen-rk
- `results/final/qwen3_gemma2rk_test.json` — Qwen3 + Gemma2-rk
- `results/final/qwen3_bgem3hardnegrk_test.json` — Qwen3 + BGE-hardneg-rk ★
- `results/final/hybrid_bm25_qwen3_test.json` — Hybrid (no reranker)
- `results/final/hybrid_bm25_qwen3_qwenrk.json` — Hybrid + Qwen-rk
- `results/final/hybrid_qwen3_hardnegrk_test.json` — Hybrid + Qwen3-hardneg-rk

---

## 4. RQ3 — Bias Analysis

**Pertanyaan:** Apakah pilihan LLM judge/reranker dari family X memberikan keuntungan sistematis untuk retriever dari family yang sama? Apakah training signal lebih menentukan dari family alignment?

### 4.1 Reranker Bias — Temuan

Lihat §3.6 untuk semua angka. Temuan kunci:

**1. Qwen-rk merusak BGE-M3 lebih parah dari Qwen3-embed (-19.8% vs -7.5%)**  
→ Ini adalah asimetri yang mengindikasikan bias, tapi arahnya paradoks: Qwen-rk (family Qwen) justru merusak Qwen3-embed (-7.5%) dan BGE-M3 (-19.8%) — tidak ada yang diuntungkan.

**2. BGE-hardneg-rk membantu Qwen3 lebih besar (+18.6%) dari BGE-M3 sendiri (+1.0%)**  
→ Reranker yang dilatih dari kesalahan retriever kuat menghasilkan sinyal yang bersifat general/transfer.

**3. Hard negative mining > LLM-judged qrels**  
→ Perbedaan antara BGE-hardneg-rk (+18.6%) dan Qwen-rk (-7.5%) jauh lebih besar dari variasi antar-family.

### 4.2 Hard Negative Overlap Analysis

**Seberapa berbeda kandidat BGE-M3 vs Qwen3 pada query yang sama?**

| K | Overlap Rate | Excl BGE-M3 hard-negs | Excl Qwen3 hard-negs | % BGE-M3 HN tidak di Qwen3 |
|---|---|---|---|---|
| 10 | 49.2% | 4.65/query | 4.91/query | 57.1% |
| 20 | 47.0% | 10.16/query | 10.44/query | 57.3% |
| 50 | 43.4% | 27.93/query | 28.19/query | 59.0% |
| 100 | 40.5% | 59.23/query | 59.44/query | 61.0% |

**Interpretasi:** Di K=10, hanya 49.2% dokumen yang sama antara BGE-M3 dan Qwen3. Masing-masing retriever punya blind spot yang berbeda dan hampir seimbang jumlahnya (4.65 vs 4.91 per query di K=10).

**Implikasi untuk reranker:** Ketika reranker dilatih dari hard negs salah satu retriever, 57% dari apa yang dipelajarinya tidak relevan untuk retriever lain (dokumen tersebut tidak ada di kandidat retriever lain). Ini menjelaskan mengapa Qwen3-hardneg-rk kurang efektif pada BGE-M3. Sebaliknya, BGE-M3 hard negs masih ~41% overlap dengan kandidat Qwen3, cukup untuk memberikan transfer.

**File:** `results/final/bias_analysis/hardneg_overlap_report.md`

### 4.3 Judge Bias Analysis — Matrix nDCG@10 per Judge per Retriever

*Semua nilai dihitung dengan threshold ≥2 kecuali Human (≥1)*

| Judge | BM25 | BGE-M3 | Qwen3-embed | Hybrid BM25+Qwen3 | BGE−Qwen3 Δ |
|-------|------|--------|-------------|-------------------|------------|
| **Human** | 0.3055 | **0.5604** | 0.4958 | 0.4603 | **−0.0647** |
| DeepSeek-V3 | 0.2980 | **0.5805** | 0.5063 | 0.4642 | −0.0739 |
| Qwen2.5-7B | 0.2833 | **0.5687** | 0.4988 | 0.4499 | −0.0700 |
| ChatGPT | 0.2739 | **0.5579** | 0.4827 | 0.4324 | −0.0752 |
| Gemma2-vllm-zs-bing | 0.2543 | 0.5006 | 0.4299 | 0.3981 | −0.0707 |
| Gemma2-vllm-zs-bing-strict | 0.2474 | 0.4713 | 0.4012 | 0.3803 | −0.0701 |
| Llama3-strict | 0.2521 | 0.4757 | 0.4072 | 0.3877 | −0.0685 |
| Llama3-vllm-fs-basic | 0.2506 | 0.4903 | 0.4245 | 0.3934 | −0.0658 |
| Llama3-vllm-fs-bing | 0.2422 | 0.4949 | 0.4377 | 0.3930 | −0.0572 |
| Llama3-vllm-zs-basic | 0.2606 | 0.4887 | 0.4164 | 0.3985 | −0.0723 |
| **Llama3-default** | **0.2352** | **0.4241** | **0.3552** | **0.3542** | **−0.0689** |

**Temuan utama judge bias:**
- **Ranking konsisten di semua judge:** BGE-M3 > Qwen3 > Hybrid > BM25 — tidak ada judge yang "membalik" leaderboard
- **BGE−Qwen3 Δ selalu negatif** (range −0.06 s/d −0.08) — semua judge setuju BGE-M3 lebih baik
- **Tidak ada bukti same-family bias** — Qwen2.5-7B judge tidak menaikkan Qwen3-embed relatif terhadap human assessment
- **DeepSeek menginflasi skor absolut** (+9% untuk BGE-M3) tapi tidak mengubah ranking
- **ChatGPT deflate skor Hybrid** (Hybrid-Qwen3: 0.4324 vs Human 0.4603) — paling skeptis terhadap hybrid retrieval
- **Llama3-default sangat rendah** — terpengaruh parah oleh pos rate 67%

**File:** `results/final/bias_analysis/judge_matrix.md`, `results/final/extended/full_matrix.csv`  
**MD analisis lengkap:** `results/final/rq3_bias_analysis.md`

---

## 5. Extended Analysis

Semua output di `results/final/extended/`. Dijalankan via `evaluation/extended_analysis.py`.

### 5.1 Inter-Judge Agreement Matrix (Mode: `inter_judge`)

**Setup:** Pairwise Cohen's κ dan Agreement % untuk semua C(11,2) = 55 pasang judge.

**Hasil κ antar judge (terpilih):**

| Judge A | Judge B | κ | Agree % | Both Pos % | Both Neg % |
|---------|---------|---|---------|-----------|-----------|
| ChatGPT | DeepSeek-V3 | **0.7144** | 88.8% | 21.1% | 67.7% |
| ChatGPT | Qwen2.5-7B | **0.5877** | 83.3% | 19.7% | 63.6% |
| DeepSeek-V3 | Qwen2.5-7B | 0.6xxx | ~86% | — | — |
| DeepSeek-V3 | Human | **0.4219** | 75.7% | 17.8% | 57.9% |
| ChatGPT | Human | **0.3869** | 74.8% | 16.1% | 58.7% |
| Qwen2.5-7B | Human | 0.3767 | 73.2% | 17.9% | 55.2% |
| Gemma2-vllm-zs-bing | Human | 0.3778 | 70.9% | 21.9% | 49.0% |
| Llama3-default | Human | 0.2103 | 55.8% | 27.2% | 28.6% |

**Positive rate per judge:**

| Judge | Pos Rate | n_pairs |
|-------|---------|---------|
| Human | 1.000* | 3,088 |
| Llama3-default | 0.6666 | 9,668 |
| Llama3-vllm-zs-basic | 0.5806 | 9,668 |
| Gemma2-vllm-fs-basic | 0.5357 | 9,668 |
| Gemma2-vllm-zs-bing-strict | 0.4529 | 9,668 |
| Llama3-vllm-fs-basic | 0.4489 | 9,668 |
| Gemma2-vllm-zs-basic | 0.4536 | 9,668 |
| Gemma2-vllm-zs-bing | 0.4103 | 9,668 |
| Llama3-strict | 0.3879 | 9,668 |
| Llama3-vllm-fs-bing | 0.3490 | 9,668 |
| Qwen2.5-7B | 0.3074 | 9,668 |
| DeepSeek-V3 | 0.2799 | 9,668 |
| ChatGPT | 0.2544 | 9,668 |

*Human pos rate = 1.0 karena format positive-only; 3,088 pairs semua score=1

**File:** `results/final/extended/inter_judge_kappa.csv`, `inter_judge_kappa_heatmap.png`, `inter_judge_agree_heatmap.png`, `inter_judge_posrate.csv`

---

### 5.2 Label Distribution per Rank Bin (Mode: `label_dist`)

**Setup:** Untuk setiap kombinasi retriever × judge, hitung precision di setiap bin rank: [1-5], [6-10], [11-25], [26-50], [51-100].

**Human precision (positivity rate) per retriever × rank bin:**

| Retriever | Rank 1-5 | Rank 6-10 | Rank 11-25 | Rank 26-50 | Rank 51-100 |
|-----------|---------|----------|-----------|-----------|------------|
| BGE-M3 | **0.280** | 0.093 | 0.034 | 0.012 | 0.004 |
| Qwen3-embed | 0.250 | 0.071 | 0.032 | 0.013 | 0.005 |
| Hybrid BM25+BGE | 0.215 | 0.079 | — | — | — |
| Hybrid BM25+Qwen3 | — | — | — | — | — |
| BM25 | 0.144 | 0.075 | 0.034 | 0.016 | 0.007 |
| Qwen-embed (legacy) | 0.003 | 0.001 | 0.001 | 0.002 | 0.001 |

*Human precision = fraksi dokumen dalam rank bin yang muncul di human positive qrels*

**Interpretasi:** BGE-M3 menempatkan 28% dari 5 dokumen teratas yang relevan menurut human (vs BM25 hanya 14.4%). Semua retriever menunjukkan gradien menurun yang jelas dari rank 1-5 ke rank 51-100, mengkonfirmasi kualitas ranking.

**File:** `results/final/extended/label_dist.csv` (330 baris: 6 retriever × 11 judge × 5 bin), `label_dist.png`

---

### 5.3 Full Retriever × Judge Matrix (Mode: `full_matrix`)

**Setup:** nDCG@10 untuk semua 66 kombinasi (6 retriever × 11 judge).

Lihat §4.3 untuk tabel lengkap. Summary tambahan dari full_matrix.csv:

**Retriever Hybrid-BGE (BM25 + BGE-M3, nDCG@10 per judge):**

| Judge | nDCG@10 |
|-------|---------|
| Human | 0.5175 |
| DeepSeek-V3 | 0.5233 |
| Qwen2.5-7B | 0.5062 |
| ChatGPT | 0.4893 |

*Hybrid-BGE = 0.5191 dari evaluasi langsung, sedikit berbeda karena perbedaan versi candidates file*

**File:** `results/final/extended/full_matrix.csv`, `full_matrix_ndcg_heatmap.png`, `full_matrix_map_heatmap.png`

---

### 5.4 Error Analysis (Mode: `error_analysis`)

#### A. Query Difficulty Distribution

| Category | Kriteria | Count | % |
|----------|---------|-------|---|
| **Hard** | max nDCG@10 < 0.1 | **73** | **7.6%** |
| Medium | 0.1 ≤ max nDCG@10 < 0.3 | 67 | 7.0% |
| Easy | max nDCG@10 ≥ 0.3 | 820 | 85.4% |

73 query (7.6%) gagal total di semua sistem (nDCG@10 = 0 untuk semua retriever). Ini kemungkinan karena:
- Dokumen relevan tidak ada di Wikipedia Indonesia
- Query memerlukan reasoning multi-step
- Pooling bias (dokumen relevan tidak pernah di-retrieve oleh sistem yang membentuk pool)

**File:** `results/final/extended/hard_queries.csv`

#### B. Judge Disagreement per Query

| qid | mean_disagreement | n_docs | frac_docs_disagreed |
|-----|-------------------|--------|---------------------|
| 3594 | 0.2248 | 10 | 1.00 |
| 4865 | 0.2179 | 10 | 1.00 |
| 1091 | 0.2176 | 9 | 1.00 |
| 5009 | 0.2166 | 10 | 1.00 |

Query-query ini memiliki disagreement tertinggi di antara semua judge — semua 10 dokumen dinilai berbeda oleh judge yang berbeda. Ini adalah kasus borderline relevance yang genuinely sulit.

**File:** `results/final/extended/judge_disagreement.csv`

#### C. Reranker Failure Analysis

**System:** Qwen3-embed + BGE-hardneg-rk vs Qwen3-embed baseline (human qrels)

| Category | Kriteria | Count | % |
|----------|---------|-------|---|
| Big failure | δ < -0.2 | 228 | 23.8% |
| Neutral | -0.05 ≤ δ ≤ 0.05 | 291 | 30.3% |
| Big gain | δ > 0.2 | 161 | 16.8% |

**Worst failures (δ = -1.0):** Query 1770, 2329, 3738, 3901, 4123, 4180, 4699, 4979, 5629 — baseline nDCG=1.0 → reranked nDCG=0.0. Ini terjadi karena query yang hanya memiliki 1 dokumen relevan, dan reranker mendorong dokumen tersebut keluar dari top-10.

**File:** `results/final/extended/reranker_failure.csv`, `error_analysis.md`

---

## 6. Kontribusi Per Anggota

### Faiz (E-series)

| Eksperimen | Detail | Output |
|-----------|--------|--------|
| **Qwen2.5-7B Judge** | Full inference pipeline, 51k pairs (train+val+test) | `qrels/qwen_*.txt` |
| **Prompt ablation** | 5 varian prompt × Qwen2.5-7B | `kappa_prompt_ablation.csv` |
| **Calibration** | Threshold sweep τ=1,2,3 | `calibration_qwen.csv` |
| **LoRA SFT** | Fine-tuning Qwen2.5-7B sebagai judge (TRL SFTTrainer, Unsloth) | `kappa_qwen_lora_test.csv` |
| **ORPO prep** | Script + data preparation (belum ada hasil final) | `lora/train_orpo.py` |
| **BM25 + Qwen-rk eval** | CrossEncoder reranker dari Qwen qrels | `bm25_qwen_rk.json` |
| **Extended analysis** | Script `evaluation/extended_analysis.py`, semua 4 mode | `results/final/extended/` |
| **WRAP_UP.md** | Dokumentasi komprehensif assets | `results/WRAP_UP.md` |

### Arvin

| Eksperimen | Detail | Output |
|-----------|--------|--------|
| **BM25 retrieval** | bm25s, no Java dependency | `candidates/bm25_*_top100.jsonl` |
| **BGE-M3 dense retrieval** | BAAI/bge-m3 + FAISS, semua split | `candidates/bgem3_*_top100.jsonl` |
| **Hybrid BM25+BGE-M3** | RRF fusion | `candidates/hybrid_test_top100.jsonl` |
| **DeepSeek-V3 judge** | API inference, 9,668 pairs | `results/qrels/deepseek_test.txt` |
| **ChatGPT judge** | gpt-4o-mini API, 9,668 pairs (complete) | `results/qrels/chatgpt_test.txt` |
| **LoRA SFT (TRL)** | Refactor ke TRL SFTTrainer, setup_and_train.sh | `lora/` |

### Karolina (Karol)

| Eksperimen | Detail | Output |
|-----------|--------|--------|
| **Qwen3-embed retrieval** | Qwen/Qwen3-Embedding-4B, no instruction, semua split | `candidates/qwen3_*_top100.jsonl` |
| **Qwen3-embed with instruction** | val split ablation +0.059 nDCG@10 | `candidates/qwen3_instruct_val_top100.jsonl` |
| **Hybrid BM25+Qwen3** | RRF fusion + eval | `candidates/hybrid_bm25_qwen3_test_top100.jsonl` |
| **Hard negative mining (Qwen3)** | 53,727 triplets dari Qwen3-embed false positives | `results/reranker_data/qwen/` |
| **Hard negative mining (BGE-M3)** | 130,961 triplets dari BGE-M3 false positives | `results/reranker_data/bgem3_hardneg/` |
| **Reranker training (Qwen3 HN)** | BGE-reranker-v2-m3 + Qwen3 hard negs | `results/models/reranker_qwen/` |
| **Reranker training (BGE-M3 HN)** | BGE-reranker-v2-m3 + BGE-M3 hard negs | `results/models/reranker_bgem3_hardneg/` |
| **All reranker evals** | 12 kombinasi first-stage × reranker | Semua JSON di `results/final/` |
| **RQ3 bias analysis** | Full analysis, visualisasi, laporan | `results/final/rq3_bias_analysis.md` |
| **Bias analysis scripts** | `evaluation/bias_analysis.py` | 3 mode: aggregate/perquery/overlap |

### Vincent (Radit)

| Eksperimen | Detail | Output |
|-----------|--------|--------|
| **SahabatAI-Gemma2 judge** | Gemma2-9B inference vLLM, train+val+test (9,668+33,076+8,282) | `results/qrels/sahabat_llama_*.txt` |
| **SahabatAI-Llama3 judge** | Llama3-8B inference (default + strict prompt) | `results/qrels/sahabat_llama_*.txt`, `results/qrels_strict/` |
| **Gemma2 prompt ablation** | 5 varian prompt × Gemma2 (vLLM) | `kappa_prompt_ablation_full.csv` |
| **Llama3 prompt ablation** | 3 varian prompt × Llama3 (vLLM) | `kappa_llama_vllm_*.csv` |
| **Size ablation (RQ2)** | BGE reranker dilatih dari Gemma2 qrels, N=100,300,500,1000,full | `ablation_summary.csv`, `size_*.json` |
| **Learning curve analysis** | Kurva learning N vs nDCG@10 vs Val AP | `learning_curve.png`, `ap_vs_ndcg_curve.png` |
| **Reranker models × 5** | Semua 5 model dari size ablation | `results/models/reranker_{100,300,500,1000,full}/` |

---

## 7. Lokasi File & Artifacts

### 7.1 Qrels Files

| File | Judge | n_pairs | Split |
|------|-------|---------|-------|
| `data/miracl-id/qrels/human/test.txt` | Human | 3,088 | test |
| `data/miracl-id/results/qrels/qwen_test.txt` | Qwen2.5-7B | 9,668 | test |
| `data/miracl-id/results/qrels/qwen_train.txt` | Qwen2.5-7B | 33,076 | train |
| `data/miracl-id/results/qrels/qwen_val.txt` | Qwen2.5-7B | 8,282 | val |
| `results/qrels/deepseek_test.txt` | DeepSeek-V3 | 9,668 | test |
| `results/qrels/chatgpt_test.txt` | ChatGPT (gpt-4o-mini) | 9,668 | test |
| `results/qrels/sahabat_llama_test.txt` | SahabatAI-Gemma2 (default) | 9,668 | test |
| `results/qrels/sahabat_llama_train.txt` | SahabatAI-Gemma2 | 33,076 | train |
| `results/qrels/sahabat_llama_val.txt` | SahabatAI-Gemma2 | 8,282 | val |
| `results/qrels/sahabat-gemma_vllm_zeroshot_bing_test.txt` | Gemma2 (vllm, zs-bing) | 9,668 | test |
| `results/qrels/sahabat-gemma_vllm_zeroshot_bing_strict_test.txt` | Gemma2 (vllm, strict) | 9,668 | test |
| `results/qrels/sahabat-gemma_zeroshot_basic_test.txt` | Gemma2 (partial) | 2,151 | test |
| `results/qrels/sahabat-llama_vllm_fewshot_basic_test.txt` | Llama3 (vllm, fs-basic) | 9,668 | test |
| `results/qrels/sahabat-llama_vllm_fewshot_bing_test.txt` | Llama3 (vllm, fs-bing) | 9,668 | test |
| `results/qrels/sahabat-llama_vllm_zeroshot_basic_test.txt` | Llama3 (vllm, zs-basic) | 9,668 | test |
| `results/qrels_strict/sahabat_llama_strict_test.txt` | Llama3 (strict) | 9,668 | test |
| `results/qrels_strict/sahabat_llama_strict_train.txt` | Llama3 (strict) | 33,076 | train |
| `results/qrels_strict/sahabat_llama_strict_val.txt` | Llama3 (strict) | 8,282 | val |

### 7.2 Candidates Files

| File | Retriever | Split |
|------|-----------|-------|
| `candidates/bm25_test_top100.jsonl` | BM25 | test |
| `candidates/bm25_train_top100.jsonl` | BM25 | train |
| `candidates/bm25_val_top100.jsonl` | BM25 | val |
| `candidates/bgem3_test_top100.jsonl` | BGE-M3 | test |
| `candidates/bgem3_train_top100.jsonl` | BGE-M3 | train |
| `candidates/bgem3_val_top100.jsonl` | BGE-M3 | val |
| `candidates/hybrid_test_top100.jsonl` | Hybrid BM25+BGE-M3 | test |
| `candidates/qwen_test_top100.jsonl` | Qwen-embed (legacy) | test |
| `candidates/qwen_train_top100.jsonl` | Qwen-embed (legacy) | train |
| `candidates/qwen_val_top100.jsonl` | Qwen-embed (legacy) | val |
| `candidates/qwen3_test_top100.jsonl` | Qwen3-embed (no instr) | test |
| `candidates/qwen3_train_top100.jsonl` | Qwen3-embed | train |
| `candidates/qwen3_val_top100.jsonl` | Qwen3-embed | val |
| `candidates/qwen3_instruct_val_top100.jsonl` | Qwen3-embed (with instr) | val |
| `candidates/hybrid_bm25_qwen3_test_top100.jsonl` | Hybrid BM25+Qwen3 | test |

### 7.3 Model Artifacts

| Model | Lokasi | HuggingFace | Training |
|-------|--------|------------|---------|
| BGE-reranker (Gemma2, N=100) | `results/models/reranker_100/` | — | 3,874 triplets, 3 epoch |
| BGE-reranker (Gemma2, N=300) | `results/models/reranker_300/` | — | 10,570 triplets |
| BGE-reranker (Gemma2, N=500) | `results/models/reranker_500/` | — | 18,346 triplets |
| BGE-reranker (Gemma2, N=1000) | `results/models/reranker_1000/` | — | 37,018 triplets |
| BGE-reranker (Gemma2, full) | `results/models/reranker_full/` | — | 121,500 triplets |
| BGE-reranker (Qwen HN) | `results/models/reranker_qwen/` | [`fassabilf/umbrela-indo-ir-reranker-qwen`](https://hf.co/fassabilf/umbrela-indo-ir-reranker-qwen) | 53,727 triplets (Qwen HN) |
| BGE-reranker (Qwen smoke) | `results/models/reranker_qwen_smoke/` | — | Smoke test only |
| BGE-reranker (BGE-M3 HN) | `results/models/reranker_bgem3_hardneg/` | [`karolinajocelyn/umbrela-indo-ir-models`](https://hf.co/karolinajocelyn/umbrela-indo-ir-models) | 130,961 triplets (BGE HN) |
| LoRA Qwen2.5-7B (smoke) | `results/models/lora_qwen_smoke/` | `fassabilf/lora-qwen-miracl-id-smoke` (🔐) | 33k pairs SFT |

### 7.4 Script Utama

| Script | Fungsi |
|--------|--------|
| `evaluation/eval_pipeline.py` | Evaluasi first-stage + reranking (nDCG@10, MAP@10, R@100) |
| `evaluation/eval_retrieval.py` | Evaluasi first-stage saja |
| `evaluation/metrics.py` | `parse_qrels()`, `compute_kappa()` |
| `evaluation/bias_analysis.py` | Bias analysis (3 mode: aggregate/perquery/overlap) |
| `evaluation/calibrate.py` | Threshold sweep untuk binarisasi |
| `evaluation/logit_inference.py` | Logit-based scoring (optional) |
| `evaluation/extended_analysis.py` | Extended analysis (4+1 mode) |
| `evaluation/plot_ablation_analysis.py` | Plot size ablation |
| `evaluation/plot_learning_curve.py` | Plot learning curve |
| `qrel_generation/inference.py` | Inference LLM judge (standard) |
| `qrel_generation/inference_vllm.py` | Inference vLLM (support --lora-path) |
| `qrel_generation/judges.py` | Registry judge models |
| `reranker/prepare_data.py` | Persiapan training data reranker (dari LLM qrels) |
| `reranker/train.py` | Training CrossEncoder reranker |
| `reranker/inference.py` | Inference reranker → TREC run |
| `reranker/mine_hard_negatives.py` | Hard negative mining dari candidates |
| `lora/prepare_data.py` | SFT data prep (human qrels → prompt/response) |
| `lora/train.py` | LoRA SFT training (manual loop) |
| `lora/prepare_orpo_data.py` | ORPO data prep |
| `lora/train_orpo.py` | ORPO training (TRL ORPOTrainer) |

### 7.5 Extended Analysis Output

| File | Konten |
|------|--------|
| `results/final/extended/inter_judge_kappa.csv` | 55 pasang κ + agree% + breakdown (56 baris) |
| `results/final/extended/inter_judge_kappa_heatmap.png` | 11×11 heatmap Cohen's κ |
| `results/final/extended/inter_judge_agree_heatmap.png` | 11×11 heatmap Agreement % |
| `results/final/extended/inter_judge_posrate.csv` | Positive rate per judge |
| `results/final/extended/label_dist.csv` | 330 baris (6 retriever × 11 judge × 5 bin) |
| `results/final/extended/label_dist.png` | Grouped bar chart label distribution |
| `results/final/extended/full_matrix.csv` | 66 kombinasi nDCG@10/MAP@10/R@100 |
| `results/final/extended/full_matrix_ndcg_heatmap.png` | Heatmap nDCG@10 |
| `results/final/extended/full_matrix_map_heatmap.png` | Heatmap MAP@10 |
| `results/final/extended/hard_queries.csv` | 960 queries dengan kategori hard/medium/easy |
| `results/final/extended/judge_disagreement.csv` | Per-query judge disagreement |
| `results/final/extended/reranker_failure.csv` | Query yang nDCG turun setelah reranking |
| `results/final/extended/error_analysis.md` | Ringkasan error analysis |

### 7.6 HuggingFace Assets

**`fassabilf` (Faiz):**
| Asset | Tipe | Status |
|-------|------|--------|
| [`fassabilf/umbrela-indo-ir`](https://hf.co/datasets/fassabilf/umbrela-indo-ir) | Dataset (results, qrels) | 🌐 Public |
| `fassabilf/umbrela-indo-ir-results` | Dataset (results detail) | 🔐 Private |
| [`fassabilf/umbrela-indo-ir-reranker-qwen`](https://hf.co/fassabilf/umbrela-indo-ir-reranker-qwen) | Model (BGE-reranker Qwen HN) | 🌐 Public |
| `fassabilf/qwen-reranker-miracl-id` | Model (BGE-reranker, MIRACL-ID) | 🔐 Private |
| `fassabilf/lora-qwen-miracl-id-smoke` | Model (LoRA Qwen2.5-7B) | 🔐 Private |
| `fassabilf/orpo-qwen-miracl-id-smoke` | Model (ORPO LoRA) | 🔐 Private |
| `fassabilf/orpo-qwen-miracl-id-smoke-b3` | Model (ORPO LoRA variant) | 🔐 Private |

**`karolinajocelyn` (Karol):**
| Asset | Tipe | Status |
|-------|------|--------|
| [`karolinajocelyn/umbrela-indo-ir-models`](https://hf.co/karolinajocelyn/umbrela-indo-ir-models) | Model (BGE-hardneg-rk, Gemma2-rk) | 🌐 Public |
| [`karolinajocelyn/umbrela-indo-ir-data`](https://hf.co/datasets/karolinajocelyn/umbrela-indo-ir-data) | Dataset (hard negative triplets) | 🌐 Public |

**`arya-raditya` (Radit):**
| Asset | Tipe | Status |
|-------|------|--------|
| [`arya-raditya/bge-reranker-gemma2-n100`](https://hf.co/arya-raditya/bge-reranker-gemma2-n100) | Model (Gemma2-rk, N=100) | 🌐 Public |

---

## Catatan Reproduksi

### Urutan Eksperimen

```
1. Setup data
   └── data/miracl-id/ (download dari MIRACL)

2. First-stage retrieval
   ├── python retriever/run_bm25.py → candidates/bm25_*
   ├── python retriever/run_bgem3.py → candidates/bgem3_*
   └── python retriever/run_qwen3.py → candidates/qwen3_*

3. LLM Judge inference
   ├── python qrel_generation/inference_vllm.py --judge qwen → qrels/qwen_*
   ├── python qrel_generation/inference_vllm.py --judge gemma2 → qrels/sahabat_llama_*
   └── [API] deepseek, chatgpt → qrels/deepseek_*, chatgpt_*

4. Kappa evaluation
   └── python evaluation/metrics.py → kappa_*.csv

5. Reranker training
   ├── python reranker/prepare_data.py (dari LLM qrels)
   ├── python reranker/mine_hard_negatives.py (dari candidates)
   └── python reranker/train.py → models/reranker_*/

6. Retrieval evaluation
   └── python evaluation/eval_pipeline.py → results/final/*.json

7. Bias analysis
   └── python evaluation/bias_analysis.py --mode all → results/final/bias_analysis/

8. Extended analysis
   └── python evaluation/extended_analysis.py --mode all → results/final/extended/
```

### Dependency Utama

```bash
pip install sentence-transformers ranx bm25s faiss-cpu \
            vllm scikit-learn matplotlib seaborn \
            huggingface-hub trl peft unsloth
```

Atau untuk environment yang managed (macOS dengan Homebrew Python):
```bash
uv run --with ranx --with scikit-learn --with matplotlib --with seaborn \
    python3 evaluation/extended_analysis.py --mode all
```

---

*Dokumen ini digenerate otomatis dari data eksperimen, diperbarui terakhir: 2026-05-28*
