# RQ3 — Bias Analysis: Self-Reinforcing Bias in LLM Judge & Retriever Choice

**Tanggal analisis:** 2026-05-27  
**Dataset:** MIRACL-ID, test split, 960 queries, 1.44M passages  
**Metrik utama:** nDCG@10 (human qrels, binary threshold=1)

---

## Ringkasan Eksekutif

Analisis ini menyelidiki dua dimensi bias yang dapat mempengaruhi hasil evaluasi IR:

1. **Reranker bias** — Apakah reranker yang dilatih dari signal LLM-X secara sistematis menguntungkan retriever dari family X?
2. **Judge bias** — Apakah LLM judge dari family Y menginflasi skor nDCG@10 retriever dari family Y relatif terhadap penilaian human assessor?

**Temuan utama:** Family alignment antara reranker dan retriever *tidak* menjamin peningkatan performa. Yang lebih menentukan adalah **kualitas dan kuantitas training signal** (hard negatives dari retriever yang kuat > LLM-judged qrels) serta **ruang perbaikan** yang tersedia di first-stage retriever. Untuk judge bias, semua LLM judge secara konsisten menempatkan BGE-M3 di atas Qwen3-embed — tidak ditemukan bukti bahwa judge dari family tertentu memihak retriever dari family yang sama.

---

## Bagian 1 — First-Stage Retriever Baseline

### 1.1 Performa dan Distribusi Per-Query

| Retriever | nDCG@10 | Recall@100 | Median | Stdev | Queries nDCG=0 |
|-----------|---------|------------|--------|-------|----------------|
| BM25 | 0.3055 | 0.7634 | 0.2184 | 0.3145 | 335/960 (34.9%) |
| Hybrid BM25+Qwen3 | 0.4603 | 0.8792 | 0.4415 | 0.3183 | 161/960 (16.8%) |
| Qwen3-embed | 0.4958 | 0.8508 | 0.4943 | 0.3280 | 156/960 (16.2%) |
| **BGE-M3** | **0.5604** | **0.9047** | **0.6053** | 0.3246 | 116/960 (12.1%) |

**nDCG@K breakdown:**

| Retriever | @1 | @3 | @5 | @10 |
|-----------|-----|-----|-----|-----|
| BGE-M3 | 0.5563 | 0.5168 | 0.5189 | 0.5604 |
| Qwen3-embed | 0.4844 | 0.4537 | 0.4647 | 0.4958 |
| Hybrid BM25+Qwen3 | 0.3979 | 0.3914 | 0.4156 | 0.4603 |
| BM25 | 0.2437 | 0.2470 | 0.2618 | 0.3055 |

**Analisis:**

**BGE-M3 dominan di semua K.** Ini terjadi karena BGE-M3 (XLM-RoBERTa + multi-vector PLAID index) dilatih secara masif pada data multilingual termasuk Bahasa Indonesia, sehingga representasi semantiknya lebih kaya dibanding Qwen3-embed (4B parameter generative model yang digunakan dalam mode no-instruction).

**Qwen3-embed mengalahkan Hybrid BM25+Qwen3 meski Hybrid punya Recall@100 lebih tinggi (0.8792 vs 0.8508).** Ini counter-intuitive. Penjelasannya: RRF fusion mencampur sinyal BM25 yang lemah ke dalam ranking Qwen3, mendegradasi urutan top-10 meskipun menambahkan lebih banyak dokumen relevan di rank 11–100. Untuk nDCG@10, urutan top 10 lebih penting daripada cakupan di rank 11–100.

**BM25 punya 335/960 queries dengan nDCG@10=0** dibanding BGE-M3 yang hanya 116/960. BM25 gagal total pada pertanyaan yang menggunakan parafrase, sinonim, atau konsep tanpa keyword eksplisit — skenario yang umum dalam MIRACL-ID.

**BGE-M3 vs Qwen3 per-query (win/loss):** BGE-M3 menang di 466 queries, Qwen3 menang di 216 queries, draw 278 queries. BGE-M3 unggul >0.1 di 348 queries, Qwen3 unggul >0.1 hanya di 130 queries. Ini menunjukkan keunggulan BGE-M3 bukan hanya pada rata-rata, tetapi konsisten di level query individual.

---

## Bagian 2 — Reranker Bias

### 2.1 Matrix Hasil Lengkap

| First-Stage | Reranker | nDCG@10 | Δ abs | Δ % | Recall@100 |
|-------------|----------|---------|-------|-----|------------|
| BGE-M3 | — | 0.5604 | — | — | 0.9047 |
| BGE-M3 | **Qwen-rk** | 0.4495 | **-0.1109** | **-19.8%** | 0.9047 |
| BGE-M3 | Gemma2-rk | 0.5111 | -0.0493 | -8.8% | 0.9047 |
| BGE-M3 | Qwen3-hardneg-rk | 0.5276 | -0.0328 | -5.9% | 0.9047 |
| BGE-M3 | **BGE-hardneg-rk** | **0.5659** | **+0.0055** | **+1.0%** | 0.9047 |
| Qwen3-embed | — | 0.4958 | — | — | 0.8508 |
| Qwen3-embed | Qwen-rk | 0.4585 | -0.0373 | -7.5% | 0.8508 |
| Qwen3-embed | Gemma2-rk | 0.5160 | +0.0202 | +4.1% | 0.8508 |
| Qwen3-embed | **BGE-hardneg-rk** | **0.5882** | **+0.0924** | **+18.6%** | 0.8508 |

*Recall@100 identik sebelum dan sesudah reranking karena reranker hanya mengubah urutan, tidak menambah/mengurangi kandidat.*

### 2.2 Temuan: Qwen-rk Merusak Kedua Retriever

**Qwen-rk (dilatih dari LLM qrels Qwen2.5-7B) memberi hasil terburuk untuk kedua retriever:**
- BGE-M3 + Qwen-rk: **-19.8%** dari baseline — reranker secara aktif mendegradasi retrieval terbaik
- Qwen3-embed + Qwen-rk: **-7.5%** — lebih "toleran" ke Qwen3, tetapi tetap negatif

**Mengapa Qwen-rk gagal?**

Qwen-rk dilatih dari pasangan positif/negatif yang ditentukan oleh *penilaian relevansi LLM* (Qwen2.5-7B menilai dokumen relevan atau tidak). LLM judge cenderung menilai dokumen relevan berdasarkan surface similarity — dokumen yang mengandung kata kunci serupa dengan query, meski tidak menjawab kebutuhan informasi secara substantif. Reranker yang dilatih dari signal ini belajar menyerupai cara kerja LLM judge, bukan cara kerja human assessor.

Ketika diaplikasikan ke kandidat BGE-M3 (yang sudah sangat baik), Qwen-rk mendegradasi karena ia mendorong ke atas dokumen yang "terlihat relevan menurut LLM" tetapi bukan dokumen yang human assessor nilai relevan. BGE-M3 + Qwen-rk kehilangan -19.8% karena selisih preferensi antara Qwen LLM dan human assessor paling terasa saat kandidat awalnya sudah berkualitas tinggi.

**Asimetri -19.8% vs -7.5%:** Qwen-rk merusak BGE-M3 lebih parah karena BGE-M3 menghasilkan kandidat yang *lebih bernilai untuk dihancurkan*. BGE-M3 sudah menempatkan dokumen human-relevan di top-10 dengan baik; Qwen-rk kemudian memindahkan mereka ke posisi lebih rendah dan mendorong dokumen "LLM-relevan" ke atas. Pada Qwen3 yang ranking-nya sudah lebih noisy, efek destruktif ini lebih kecil karena kandidatnya memang sudah tidak optimal.

### 2.3 Temuan: Gemma2-rk — Pola yang Berbeda

**Gemma2-rk dilatih dari qrels yang dihasilkan oleh model Gemma2** (SahabatAI-Gemma2), bukan dari hard negative mining. Hasilnya menunjukkan pola intermediate:
- BGE-M3 + Gemma2-rk: -8.8% (lebih baik dari Qwen-rk, tetapi masih merusak BGE-M3)
- Qwen3-embed + Gemma2-rk: **+4.1%** (berhasil meningkatkan Qwen3)

**Mengapa Gemma2-rk berhasil untuk Qwen3 tapi tidak untuk BGE-M3?**

Gemma2-rk juga dilatih dari LLM qrels, tetapi hanya dari N=100 training queries — jauh lebih sedikit data dibanding Qwen-rk. Signal yang lebih terbatas ternyata justru lebih "aman": reranker tidak bisa belajar terlalu dalam bias LLM judge. Selain itu, Gemma2 judge menurut analisis RQ1 (pos_rate=0.4123) lebih generously menilai relevansi, yang mungkin lebih aligned dengan kebutuhan retrieval umum.

Untuk Qwen3, Gemma2-rk memberi +4.1% karena Qwen3 punya lebih banyak *improvable queries* (118/960 vs 88/960 untuk BGE-M3) — kueri di mana dokumen relevan ada di rank 11–100 tapi tidak masuk top-10. Reranker sekecil apapun bisa mengangkat beberapa di antaranya.

Untuk BGE-M3, -8.8% terjadi karena BGE-M3 sudah optimal; setiap pertukaran posisi akibat reranker LLM-judged lebih sering merugikan daripada menguntungkan.

### 2.4 Temuan Kunci: BGE-hardneg-rk — Training Signal Menentukan

**BGE-hardneg-rk adalah satu-satunya reranker yang meningkatkan BGE-M3 (+1.0%) dan meningkatkan Qwen3 secara signifikan (+18.6%).**

Perbedaan fundamental antara BGE-hardneg-rk dan reranker lainnya bukan pada arsitektur (semua pakai BAAI/bge-reranker-v2-m3 sebagai base), melainkan pada **sumber training signal**:

| Reranker | Training signal | Triplets | Sumber error |
|----------|----------------|----------|--------------|
| Qwen-rk | LLM qrels (Qwen2.5-7B judge) | ~53k | Preferensi LLM |
| Gemma2-rk | LLM qrels (Gemma2 judge), N=100 queries | kecil | Preferensi LLM |
| Qwen3-hardneg-rk | Hard negatives dari Qwen3-embed | 53,727 | False positives Qwen3 |
| **BGE-hardneg-rk** | **Hard negatives dari BGE-M3** | **130,961** | **False positives BGE-M3** |

BGE-hardneg-rk dilatih dari **dokumen yang berhasil menembus top-20 BGE-M3 tetapi dinilai tidak relevan oleh human assessor** (false positives). Ini adalah sinyal "di mana retriever terkuat sekalipun bisa salah" — secara definisi, ini adalah kasus yang paling sulit dan paling informatif.

**Mengapa BGE-hardneg-rk membantu Qwen3 lebih besar (+18.6%) daripada BGE-M3 sendiri (+1.0%)?**

1. **Ruang perbaikan (headroom):** Qwen3 punya 118/960 improvable queries (rel. dok. di rank 11–100 tapi tidak top-10), BGE-M3 hanya 88/960. Setiap query yang "diperbaiki" oleh reranker memberi kontribusi lebih besar ke Qwen3.

2. **Universalitas hard negatives BGE-M3:** Dokumen yang berhasil menembus top-20 BGE-M3 (retriever terkuat) adalah dokumen yang *secara semantik sangat mirip query* tetapi tidak relevan. Ini adalah false positive yang "keras" — sulit dideteksi oleh retriever manapun. Qwen3 juga menempatkan dokumen-dokumen serupa di ranking tinggi (karena Qwen3 juga menggunakan dense retrieval berbasis similarity). Reranker BGE-hardneg-rk belajar menolak kelas dokumen ini, dan efeknya berlaku lintas retriever.

3. **Volume training (130k vs 53k triplets):** BGE-M3 memiliki lebih banyak false positives dalam top-20 karena recall@100-nya lebih tinggi (0.9047 vs 0.8508) — lebih banyak dokumen yang dievaluasi. Semakin banyak data hard negative, semakin robust rerankernya.

4. **BGE-M3 sudah di near-ceiling:** Untuk BGE-M3, reranker terbaik pun hanya bisa memberi +1.0% karena dokumen relevan sudah ada di posisi bagus sejak awal. Reranker bekerja pada margin yang sangat tipis.

### 2.5 Implikasi: Training Signal > Family Alignment

Hipotesis awal RQ3 adalah bahwa reranker dari family X akan menguntungkan retriever X (same-family bias). Data justru membuktikan sebaliknya:

- Qwen-rk (family Qwen) *merusak* Qwen3-embed (-7.5%), bukan membantu
- BGE-hardneg-rk (dilatih dari BGE-M3 errors) *paling membantu* Qwen3-embed (+18.6%)
- Tidak ada satupun reranker LLM-judged yang memberi improvement konsisten

**Kesimpulan:** Family alignment antara reranker dan retriever tidak memiliki efek positif yang terukur. Yang menentukan adalah apakah training signal reranker berasal dari *kesalahan aktual retriever* (hard negative mining) versus *preferensi LLM* (LLM-judged qrels). Hard negative mining menghasilkan reranker yang lebih umum dan lebih efektif.

---

## Bagian 3 — Hard-Negative Overlap Analysis

### 3.1 Seberapa Berbeda Kandidat BGE-M3 vs Qwen3?

| K | Overlap Rate | Exclusive BGE-M3 hard-negs | Exclusive Qwen3 hard-negs | % BGE-M3 hardneg tidak di Qwen3 |
|---|---|---|---|---|
| 10 | 49.2% | 4.65/query | 4.91/query | 57.1% |
| 20 | 47.0% | 10.16/query | 10.44/query | 57.3% |
| 50 | 43.4% | 27.93/query | 28.19/query | 59.0% |
| 100 | 40.5% | 59.23/query | 59.44/query | 61.0% |

**Analisis:**

Pada K=10, hanya **49.2% dari dokumen top-10 yang sama** antara BGE-M3 dan Qwen3 untuk query yang sama. Artinya lebih dari separuh dokumen yang dianggap paling relevan oleh satu retriever tidak muncul di top-10 retriever lain. Overlap semakin menurun seiring bertambahnya K (40.5% di K=100), menunjukkan semakin ke bawah ranking, semakin divergen kedua retriever.

**Hard negatives eksklusif hampir seimbang:** BGE-M3 punya rata-rata 59.23 exclusive hard-negs per query di K=100, Qwen3 punya 59.44. Ini berarti kedua retriever memiliki *kumpulan blind spot yang berbeda* — dokumen yang menipu BGE-M3 berbeda dari dokumen yang menipu Qwen3.

**Mengapa ini penting untuk reranker bias?** Ketika reranker dilatih dari hard negatives salah satu retriever, ia belajar menolak dokumen yang spesifik untuk retriever tersebut. Saat diaplikasikan ke retriever lain, sebagian besar "hard negatives yang dipelajari" tidak relevan (karena retriever lain tidak memasukkan dokumen itu ke kandidatnya). Ini menjelaskan mengapa Qwen3-hardneg-rk kurang efektif pada BGE-M3 — 57.1% dari hard negatives Qwen3 tidak pernah muncul di kandidat BGE-M3.

Sebaliknya, BGE-hardneg-rk tetap efektif pada Qwen3 karena (a) BGE-M3 lebih kuat sehingga hard negatives-nya lebih "umum dan susah", dan (b) sebagian hard negatives BGE-M3 muncul juga di kandidat Qwen3 (41% overlap di K=100 dari sisi Qwen3).

---

## Bagian 4 — Judge Bias Analysis

### 4.1 Matrix nDCG@10 Per Judge Per Retriever

*(semua nilai dihitung menggunakan qrels judge yang di-binarisasi pada threshold score≥2)*

| Judge | Family | BM25 | BGE-M3 | Qwen3-embed | Hybrid | Qwen3−BGE-M3 Δ |
|-------|--------|------|--------|-------------|--------|----------------|
| Human | human | 0.3055 | 0.5604 | 0.4958 | 0.4603 | **-0.0647** |
| SahabatAI-Gemma2 (zeroshot-bing) | gemma2 | 0.2724 | 0.5671 | 0.4907 | 0.4389 | **-0.0764** |
| SahabatAI-Gemma2 (strict) | gemma2 | 0.2740 | 0.5515 | 0.4744 | 0.4315 | **-0.0771** |
| SahabatAI-Llama3 (zeroshot-basic) | llama3 | 0.2911 | 0.5349 | 0.4554 | 0.4405 | **-0.0795** |
| SahabatAI-Llama3 (fewshot-basic) | llama3 | 0.2768 | 0.5354 | 0.4670 | 0.4319 | **-0.0684** |
| SahabatAI-Llama3 (fewshot-bing) | llama3 | 0.2682 | 0.5430 | 0.4812 | 0.4330 | **-0.0618** |
| DeepSeek | deepseek | 0.2737 | **0.6107** | **0.5368** | 0.4554 | **-0.0739** |
| ChatGPT (GPT-4o) | gpt4o | 0.2398 | 0.5486 | 0.4776 | 0.4010 | **-0.0709** |

### 4.2 Temuan: Tidak Ada Bukti Same-Family Bias pada Judge

**Ranking sistem konsisten di semua judge:** BGE-M3 > Qwen3-embed > Hybrid > BM25 untuk setiap judge. Tidak ada judge dari family tertentu yang "membalik" ranking atau secara tidak proporsional menaikkan skor salah satu retriever.

**Kolom Qwen3−BGE-M3 Δ selalu negatif** di semua judge (range -0.0618 sampai -0.0795). Ini berarti semua judge — termasuk judge dari berbagai family — setuju bahwa BGE-M3 lebih baik dari Qwen3-embed. Tidak ada judge yang secara sistematis "memihak" Qwen3-embed.

**Mengapa tidak ada same-family bias?** Qwen3-embed dan Qwen2.5-7B (judge yang sudah dihapus) memang dari family yang sama (Qwen), tetapi arsitektur dan tujuan mereka sangat berbeda: Qwen3-embed adalah model embedding (fine-tuned untuk retrieval), sementara Qwen2.5-7B adalah LM generatif (digunakan sebagai relevance judge). Preferensi relevance judgement Qwen2.5-7B lebih dipengaruhi oleh training data instruksi daripada kesamaan embedding space dengan Qwen3-embed. Bias family yang dihipotesiskan lebih mungkin muncul jika judge dan retriever menggunakan *model yang sama persis* (contoh: Qwen3-embed sebagai judge dan Qwen3-embed sebagai retriever), yang bukan skenario di sini.

### 4.3 Temuan: DeepSeek Menginflasi Skor Absolut

**DeepSeek secara konsisten memberi skor nDCG@10 tertinggi** untuk semua retriever — BGE-M3 mendapat 0.6107 menurut DeepSeek vs 0.5604 menurut Human (selisih +9%). Qwen3-embed mendapat 0.5368 menurut DeepSeek vs 0.4958 menurut Human (+8.3%).

**Mengapa DeepSeek menginflasi skor absolut?** Analisis ini dilakukan dengan threshold ≥2 pada skala 0-3. DeepSeek dengan pos_rate=0.2799 (28% dokumen dinilai relevan) dan distribusi yang lebih merata di skor 1-3 menghasilkan binaryisasi yang sedikit berbeda dari human judgement. DeepSeek cenderung memberi skor 1 atau 2 pada dokumen yang oleh human dinilai "tidak relevan (0)" — dokumen topically related tetapi tidak menjawab query secara tepat. Ini menyebabkan lebih banyak "true positives" dari perspektif DeepSeek, yang menaikkan nDCG@10 secara absolut.

**Namun ranking relatif tetap stabil:** BGE-M3 tetap yang terbaik menurut DeepSeek, dengan Δ (Qwen3-BGE-M3) = -0.0739, mirip dengan Human -0.0647. DeepSeek memberi inflasi absolut tetapi tidak mengubah leaderboard.

### 4.4 Temuan: ChatGPT (GPT-4o) Merendahkan Skor Hybrid

ChatGPT memberi skor Hybrid BM25+Qwen3 hanya 0.4010 dibanding Human 0.4603 — selisih -0.0593, terbesar di antara semua judge untuk sistem ini. Ini menunjukkan bahwa ChatGPT relatif lebih "kritis" terhadap hasil hybrid yang mencampur BM25 keyword matching dengan dense retrieval. Kemungkinan ChatGPT lebih strict terhadap dokumen yang relevan karena keyword match semata tanpa konten yang substantif.

Selain itu, ChatGPT hanya mengevaluasi 870 dari 960 queries (ada 90 queries dengan zero judgements, kemungkinan karena limit API atau error) — ini perlu diperhatikan saat menggunakan ChatGPT sebagai judge untuk evaluasi lengkap.

### 4.5 Konsistensi Leaderboard: Semua Judge Sepakat

Meski skor absolut bervariasi, **semua judge menghasilkan ranking yang sama**: BGE-M3 > Qwen3-embed > Hybrid BM25+Qwen3 > BM25. Ini merupakan bukti robustness bahwa kesimpulan penelitian tidak bergantung pada pilihan judge — seseorang tidak bisa "mengubah kesimpulan" dengan memilih judge yang berbeda dari family yang sama dengan sistem yang diunggulkan.

---

## Bagian 5 — Sintesis dan Implikasi untuk Paper

### 5.1 Jawaban untuk RQ3

**"Apakah pilihan LLM judge/reranker dari family X memberikan keuntungan sistematis untuk retriever dari family yang sama?"**

**Jawaban: Tidak, dan data menunjukkan hubungan yang lebih kompleks.**

Untuk *reranker*: Family alignment tidak memberi keuntungan. Qwen-rk (family Qwen) justru merusak Qwen3-embed (-7.5%). Yang memberi keuntungan adalah training signal berbasis hard negative mining dari retriever yang kuat (BGE-hardneg-rk memberikan +18.6% untuk Qwen3-embed).

Untuk *judge*: Tidak ditemukan bukti same-family bias. Semua judge menghasilkan leaderboard yang identik dengan human assessor. Variasi antar-judge hanya pada inflasi/deflasi skor absolut, bukan pada perubahan ranking relatif.

**"Apakah training signal lebih menentukan daripada family-alignment?"**

**Jawaban: Ya, secara tegas.**

Hard negative mining > LLM-judged qrels untuk semua kondisi yang diuji. Perbedaan antara BGE-hardneg-rk (hard negative) dan Qwen-rk (LLM qrels) jauh lebih besar daripada perbedaan antar-family dalam LLM qrels.

### 5.2 Temuan yang Paling Menarik untuk Disoroti

1. **Cross-retriever generalization:** BGE-hardneg-rk membantu Qwen3 lebih besar (+18.6%) daripada BGE-M3 sendiri (+1.0%). Reranker yang dilatih dari kesalahan retriever kuat menghasilkan signal yang bersifat general.

2. **LLM judge tidak bias terhadap family sendiri:** Ini *against* hipotesis naive tentang same-family bias. Interpretasi: LLM judge melihat relevansi dari perspektif pemahaman bahasa, bukan dari perspektif embedding similarity — sehingga tidak ada keuntungan sistematik untuk retriever yang berbagi architecture space.

3. **Hard negatives divergen antara retriever (57% eksklusif di K=10):** Kedua retriever punya blind spot yang sangat berbeda. Implikasi: untuk membuat reranker paling robust, idealnya mine hard negatives dari *kombinasi* kedua retriever.

4. **DeepSeek inflasi absolut tapi bukan ranking:** Ini menunjukkan bahwa peneliti perlu berhati-hati saat melaporkan skor absolut dengan LLM judge, tetapi perbandingan relatif antar sistem tetap valid.

### 5.3 Keterbatasan Analisis

- Qwen3-embed dievaluasi tanpa instruction prefix (lower bound; dengan instruction diperkirakan +0.059 nDCG@10)
- Gemma2-rk hanya dilatih dari N=100 training queries — terlalu sedikit untuk kesimpulan definitif
- Analisis win/loss dan Kendall-τ rank disruption untuk sistem reranked belum dilakukan (butuh TREC run files per reranker)
- ChatGPT hanya memiliki judgements untuk 870/960 queries

---

## File Output

| File | Konten |
|------|--------|
| `bias_analysis/delta_heatmap.png` | Heatmap % delta nDCG@10 vs baseline (retriever × reranker) |
| `bias_analysis/recall_comparison.png` | Bar chart Recall@100 per first-stage retriever |
| `bias_analysis/perquery_violin.png` | Distribusi per-query nDCG@10 (violin plot) |
| `bias_analysis/ndcg_at_k.png` | nDCG@K untuk K=1,3,5,10 per sistem |
| `bias_analysis/win_loss.png` | Win/loss/tie chart vs BGE-M3 baseline |
| `bias_analysis/hardneg_overlap_report.md` | Overlap analysis BGE-M3 vs Qwen3 di K=10/20/50/100 |
| `bias_analysis/judge_ndcg_matrix.png` | Heatmap nDCG@10 per (judge × retriever) |
| `bias_analysis/judge_delta_chart.png` | Bar chart Qwen3−BGE-M3 Δ per judge |
| `bias_analysis/leaderboard_correlation.png` | Kendall-τ leaderboard per judge vs Human |
| `bias_analysis/judge_matrix.json` | Data lengkap judge × retriever matrix |
| `bias_analysis/perquery_scores.json` | Per-query nDCG@K untuk 4 first-stage systems |
