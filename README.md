# UMBRELA for Indonesian IR — LLM-as-Judge + Reranker Training + Bias Analysis

**Tugas Kelompok IR Genap 2025/2026** | Faiz · Vincent · Radit · Arvin · Karol

Extension of the UMBRELA framework (SIGIR 2025) to Indonesian Information Retrieval using MIRACL-ID.
We compare multilingual vs. Indonesian-specific LLM judges, train rerankers from LLM-generated qrels,
and analyze self-reinforcing bias (RQ3).

---

## Progress Report

> Last updated: 2026-05-17

### Base Paper

This project extends **UMBRELA** — an LLM-as-Judge framework for automatic relevance assessment — to the Indonesian language domain. The original paper:

> Naghmeh Farzi and Laura Dietz. *UMBRELA: UMbrela is the Replacement for BERT-based Relevance LAbeling.* SIGIR 2025.

Our extension uses the **MIRACL-ID** dataset (~1.44M Indonesian Wikipedia passages, 960 test queries with human qrels) to evaluate three research questions: (RQ1) which LLM judge best aligns with human relevance judgments in Indonesian, (RQ2) whether LLM-generated qrels can train a reranker that improves over BM25, and (RQ3) whether judge choice introduces self-reinforcing bias.

---

### Team Contributions

**Faiz — Qwen2.5-7B-Instruct Judge + Reranker Training**
Implemented the full end-to-end pipeline: LLM judge inference, Cohen's kappa evaluation against human qrels, reranker training (triplet preparation → CrossEncoder fine-tuning → TREC-format inference → nDCG@10 evaluation). Full qrel generation across all 51k query-document pairs completed.

**Radit — SahabatAI-Gemma2 Judge + Size Ablation (RQ2)**
Qrel generation completed for all splits: test (9,668 pairs), train (33,076 pairs), val (8,282 pairs). Cohen's kappa computed for all three splits. Size ablation (RQ2) fully completed: BGE reranker fine-tuned on 5 training subsets (N=100, 300, 500, 1000, full) using Gemma2-generated qrels, evaluated on test set with BM25 first-stage. Learning curve and divergence analysis (AP vs nDCG) generated.

**Vincent — SahabatAI-Llama3 Judge**
Inference on the test split (9,668 pairs) is completed using an unquantized model hosted on vast.ai. Full inference run on the train split is currently underway with 16,829 out of 33,076 pairs processed. Scripts are updated to ensure full precision model loading and bypass previous memory constraints.

**Arvin — First-Stage Retrieval**
Implemented BM25 (bm25s, no Java dependency), BGE-M3 dense retrieval with FAISS, and hybrid RRF fusion. Generated top-100 candidate files for all splits (train/val/test). Retrieval scores evaluated and added to results table.

**Karol — Corpus Encoding + Bias Analysis (RQ3)**
Qwen2.5-7B corpus encoding completed in chunked batches (~20.8 GB total, 5 FAISS shards, uploaded to HuggingFace). RQ3 bias analysis implementation in progress.

---

### Preliminary Results

**RQ1 — Judge Agreement** (Cohen's κ vs. human qrels, test set)

| Judge Model | κ | LLM pos. rate | Human pos. rate | n_pairs |
|---|---|---|---|---|
| DeepSeek-V3 | **0.4219** | 27.99% | 31.94% | 9,668 |
| Qwen2.5-7B-Instruct | 0.3767 | 30.74% | 31.94% | 9,668 |
| ChatGPT (gpt-4o-mini) | 0.3856 | 26.38% | 32.96% | 6,751 |
| SahabatAI-Gemma2-9B | 0.3763 | 41.23% | 31.94% | 9,668 |
| SahabatAI-Llama3-8B (strict prompt) | 0.3652 | 38.79% | 31.94% | 9,668 |
| SahabatAI-Llama3-8B (default prompt) | 0.2103 | 66.66% | 31.94% | 9,668 |

DeepSeek-V3 achieves the highest agreement (κ=0.42) and is also the most conservative judge (28% pos rate vs 32% human). Qwen2.5-7B is best-calibrated among open-source models. Gemma2 overpredicts slightly (41%). Llama3 requires a strict output-constrained prompt to perform competitively — without it, pos rate reaches 67% and κ drops to 0.21. Note: ChatGPT evaluated on 6,751 pairs (partial test set due to API cost).

**RQ2 — Retrieval & Reranking** (nDCG@10, test set, 960 queries)

| System | nDCG@10 | R@100 |
|---|---|---|
| BM25 (baseline) | 0.3053 | 0.7634 |
| BGE-M3 dense retrieval | **0.5604** | 0.9047 |
| Hybrid BM25 + BGE-M3 (RRF) | 0.5191 | 0.9154 |
| BM25 + BGE reranker (Gemma2 qrels, N=100) | 0.5178 | — |
| Qwen-embed dense retrieval | 0.0066 | 0.0406 |

BGE-M3 substantially outperforms BM25 on MIRACL-ID. Reranking BM25 candidates with a BGE reranker trained on only 100 queries of Gemma2-generated qrels approaches BGE-M3 performance (0.5178 vs 0.5604). Reranker eval for Qwen-trained model pending.

From a training size perspective, Gemma2-trained BGE rerankers (BM25 first-stage) consistently beat the BM25 baseline across all training sizes, with N=100 achieving the highest nDCG@10=0.5178. Counterintuitively, performance degrades as training size increases toward N=full (0.3993), while val average precision on LLM qrels rises monotonically to 99.9% — indicating the reranker overfits to LLM judge noise as more training data is added.

| N (queries) | n_triplets | nDCG@10 | MAP@10 | Val AP (LLM) |
|---|---|---|---|---|
| 100 | 1,937 | **0.5178** | 0.4088 | 0.865 |
| 300 | 5,285 | 0.4620 | 0.3491 | 0.873 |
| 500 | 9,173 | 0.5011 | 0.3950 | 0.905 |
| 1000 | 18,509 | 0.4072 | 0.2993 | 0.916 |
| full | 60,750 | 0.3993 | 0.2917 | 1.000 |

**RQ3 — Bias Analysis**: pending (Karol)

---

## Quick Start (per person)

### Step 0 — Clone & install
```bash
git clone https://github.com/TBI-HMLI600062/umbrela-indo-ir
cd umbrela-indo-ir
pip install -r requirements.txt
```

### Step 1 — Download processed MIRACL-ID data (one command, ~2.5 GB)
```bash
huggingface-cli download fassabilf/umbrela-indo-ir \
    --repo-type dataset --local-dir data/miracl-id/
```
Data splits: **train** (~3257 queries), **val** (~814 queries), **test** (960 queries, has human qrels).
Split seed: `42`. Test = MIRACL dev with human relevance judgments.

### Step 2 — Run your LLM judge (replace MODEL_ID)
```bash
# Faiz: Qwen
python qrel_generation/inference.py \
    --judge-model Qwen/Qwen2.5-7B-Instruct \
    --split train --n-queries 1000 \
    --output results/qrels/qwen_train.txt

# Vincent: SahabatAI-Llama3
python qrel_generation/inference.py \
    --judge-model GoToCompany/llama3-8b-cpt-sahabatai-v1-instruct \
    --split train --n-queries 1000 \
    --output results/qrels/sahabat_llama_train.txt --batch-size 8

# Radit: SahabatAI-Gemma2
python qrel_generation/inference.py \
    --judge-model GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct \
    --split train --n-queries 1000 \
    --output results/qrels/sahabat_gemma_train.txt --batch-size 8
```
Inference supports **resume** (safe to Ctrl+C and restart) and **CUDA OOM recovery**.

### Step 3 — Evaluation
```bash
# Cohen's kappa (RQ1)
python evaluation/metrics.py \
    --llm-qrels results/qrels/qwen_train.txt \
    --human-qrels data/miracl-id/qrels/human/test.txt \
    --output results/final/kappa.csv

# nDCG@10 (RQ2)
python evaluation/eval_pipeline.py \
    --first-stage bm25 \
    --reranker results/reranker/qwen/ \
    --output results/final/bm25_qwen_rk.json
```

---

## Repository Structure

```
umbrela-indo-ir/
├── requirements.txt
├── data/
│   ├── download_miracl.py      # Download + preprocess MIRACL-ID → upload to HF
│   ├── dl2019/, dl2020/, dl2023/   # Original TREC DL data (from paper)
├── qrel_generation/
│   └── inference.py            # UMBRELA judge on MIRACL-ID (E0-T2, Faiz)
├── retrieval/
│   ├── bm25/index.py, retrieve.py         # Arvin (E4)
│   └── dense/embed_corpus.py, retrieve.py, hybrid.py  # Arvin + Karol
├── reranker/
│   ├── prepare_data.py, train.py, inference.py  # Faiz (E1)
├── evaluation/
│   ├── metrics.py              # Cohen's kappa
│   ├── eval_pipeline.py        # nDCG@10 evaluation
│   └── bias_analysis.py        # RQ3 bias chart (Karol)
├── src/                        # Original UMBRELA source (TREC DL)
├── prompts/                    # UMBRELA prompt templates
├── results/final/              # Tracked: final tables + charts
└── paper/                      # ACL format LaTeX
```

---

## Research Questions

| RQ | Question | Owner |
|----|----------|-------|
| RQ1 | Which LLM judge gives best relevance judgments for Indonesian? (Qwen vs SahabatAI-Llama vs SahabatAI-Gemma) | Faiz + Vincent + Radit |
| RQ2 | Can LLM-generated qrels train a reranker that beats baseline? How many qrels? Effect of first-stage? | Faiz (main) + Vincent + Radit + Arvin |
| RQ3 | Does LLM judge choice introduce self-reinforcing bias? | Karol |

Dataset: **MIRACL-ID** — Indonesian Wikipedia, ~1.44M passages.
Train/val split: 80/20 of MIRACL train, seed=42. Test = MIRACL dev.

---

# Appendix for "Does UMBRELA Work on Other LLMs?"

Contact: Naghmeh.Farzi@unh.edu

Paper Link: [paper.pdf](paper.pdf)

Citation: Naghmeh Farzi and Laura Dietz. 2025. *Does UMBRELA Work on Other
LLMs?.* In Proceedings of the 48th International ACM SIGIR Conference on
Research and Development in Information Retrieval (SIGIR ’25), July 13–18,
2025, Padua, Italy. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3726302.3730317

This repository contains code and resources for reproducing the experiments in **"Does UMBRELA Work on Other LLMs?"**. Our study investigates the generalizability of the UMBRELA LLM Judge evaluation framework across different large language models (LLMs), assessing its effectiveness beyond the original study.

## Overview
The UMBRELA framework provides a zero-shot, structured prompting approach for generating graded relevance labels. This work explores:
- How different LLMs impact relevance assessment accuracy.
- The effect of model scale on leaderboard rank correlation and per-label agreement.
- The reproducibility of UMBRELA's effectiveness across various LLM families.


## Models Evaluated

We evaluated UMBRELA on:

- Flan-T5-Large
- Meta-Llama-3-8B-Instruct
- Meta-Llama-3.3-70B-Instruct-Turbo
- DeepSeek-V3


## Repository Structure
```
├── data/ 
│ ├── dl2019 
│ ├── qdl2020
│ ├── dl2023 
├── prompts/ # Prompt templates for different settings 
│ ├── qrel_fewshot_basic.txt 
│ ├── qrel_fewshot_bing.txt 
│ ├── qrel_zeroshot_basic.txt 
│ ├── qrel_zeroshot_bing.txt 
├── results/ # Stores evaluation outputs 
├── src/ # Source code for processing and evaluation 
│ ├── data_processing.py # Handles input/output
│ ├── main.py # Main script to run the evaluation 
│ ├── make_rubric_format.py # Converts data into rubric-based format 
│ ├── model_utils.py # Functions to load and interact with LLMs 
│ ├── prompts.py # Handles prompt construction 
│ ├── relevance_processors.py 
│ ├── relevance_scoring.py 
├── run_command.sh # Example script for running the evaluation 
└── README.md # Project documentation
```


## Data Directory

The `data/` directory contains the datasets used for evaluation. It includes the following folders:

- `dl2019/`: Contains data for the DL2019 dataset.
- `dl2020/`: Contains data for the DL2020 dataset.
- `dl2023/`: Contains data for the DL2023 dataset (LLLMJudge challenge dataset).

Each of these directories contains files that are required for running the experiments (queries, documents, and qrels).


## Main Arguments of Commands

| Argument             | What it means                                      |
|----------------------|----------------------------------------------------|
| `--model_id`         | Model name (e.g., `deepseek-ai/DeepSeek-V3`) |
| `--prompt_mode`      | Prompt strategy to use (`zeroshot_bing`, etc.)    |
| `--test_qrel_path`   | Path to qrel (ground-truth relevance) file        |
| `--queries_path`     | Path to the file with search queries               |
| `--docs_path`        | Path to the document file                          |
| `--result_file_path` | Where to save the model’s output                   |
| `-together`          | Use Together.ai API to run the model (optional)   |



<details>
  <summary>Full Evaluation Command Example (Click to expand)</summary>


```
python src/main.py \
  --model_id "<MODEL_ID>" \
  --test_qrel_path "<PATH_TO_QREL_FILE>" \
  --queries_path "<PATH_TO_QUERIES_FILE>" \
  --docs_path "<PATH_TO_DOCUMENTS_FILE>" \
  --prompt_mode "<PROMPT_MODE>" \
  --result_file_path "<OUTPUT_RESULT_FILE>" \
  -together (Only if you want to run a model with Together AI API )
```

Example of Usage (DeepSeek-V3 on TREC DL2020)
```
python src/main.py \
  --model_id "deepseek-ai/DeepSeek-V3" \
  --test_qrel_path "./data/dl2020/2020qrels-pass.txt" \
  --queries_path "./data/dl2020/msmarco-test2020-queries.tsv" \
  --docs_path "./data/dl2020/dl2020_document.jsonl" \
  --prompt_mode "zeroshot_bing" \
  --result_file_path "./results/dl20_test_zeroshot_bing_DSV3.txt" \
  -together


```
</details>
