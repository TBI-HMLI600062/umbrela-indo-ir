# Language Ranker

A lightweight ranking framework for LLM decoding that introduces a lightweight module to rerank candidate responses using features extracted by the base model. The paper [Language Ranker: A Lightweight Ranking framework for LLM Decoding](https://www.arxiv.org/abs/2510.21883) has been accepted by NeurIPS 2025.

## Overview

Language Ranker revisits LLM generation through the lens of recommender systems, conceptualizing the decoding process as analogous to the ranking stage in recommendation pipelines. The ranker introduces a lightweight module that uses the internal representations (hidden states) extracted by the base language model to rerank candidate responses. This approach significantly reduces computational overhead during both training and inference stages while maintaining competitive performance, requiring only <0.5M additional parameters.


## Installation

### Requirements

- Python 3.10
- PyTorch 2.6.0

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd language_ranker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Prepare Dataset

For each model, sample multiple candidate responses and extract hidden states:

```bash
python generator/sample_res.py \
    --model_name <path_to_model> \
    --task tool \
    --num_return_sequences 100 \
    --temperature 1.5 \
    --max_new_tokens 512

python generator/sample_hidden.py \
    --model_name <path_to_model> \
    --task tool \
    --batch_size 50 \
    --data_root <path_to_data_root>
```

Evaluate and label the generated responses:

```bash
python eval/labeling.py \
    --data_path <path_to_data> \
    --task tool
```

**Supported tasks**: `math`, `code`, `tool`


### Step 2: Train Ranker

Train a ranker model with the prepared dataset:

```bash
python ranker/ranker_train.py \
    --hidden_size 4096 \
    --ranker_structure trans \
    --num_layers 1 \
    --num_attention_heads 1 \
    --dropout_prob 0.1 \
    --lr 0.001 \
    --batch_size 1024 \
    --num_epochs 50 \
    --train_dir <path_to_train_data> \
    --test_dir <path_to_test_data> \
    --cand_num 10 \
    --sample_num 500 \
    --dataset_layer 32 \
    --half
```

### Step 3: Evaluate Ranker

Evaluate the trained ranker:

```bash
python ranker/ranker_eval.py \
    --ranker_path <path_to_trained_ranker> \
    --test_dir <path_to_test_data> \
    --batchsize 500
```

## Configuration

### Ranker Architectures

- **Transformer** (`trans`): Uses self-attention to model relationships between context and candidates
- **MLP** (`mlp`): Simple feedforward network

## Data Format

### Input Data

Each task expects JSON files with the following structure:

**Math:**
```json
{
    "item_id": "mathprealgebra_389",
    "question": "Compute: $(3^2)(2^4)(37)(5^3)$",
    "answer_cot": "Since multiplication ...",
    "answer_value": "666000",
    "level": "Level 2",
    "question_type": "Prealgebra"
  },
```

**Code:**
```json
{
    "item_id": "mbpp_1",
    "question": "Write a function to...",
    "test_list": ["assert function_name(...) == ..."],
    "code": "def function_name ...",
    "function_name": "function_name",
}
```

**Tool:**
```json
{
    "id": "00001",
    "query": "What is the weather?",
    "answers": "[{\"name\": ...}, ...]",
    "tools": "[{\"name\": \"get_weather\", ...}]"
}
```


## Examples

See the `scripts/` directory for example shell scripts that demonstrate how to run experiments with different configurations.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2025language,
  title={Language Ranker: A Lightweight Ranking framework for LLM Decoding},
  author={Zhang, Chenheng and Du, Tianqi and Zhang, Jizhe and Xiao, Mingqing and Wang, Yifei and Wang, Yisen and Lin, Zhouchen},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://www.arxiv.org/abs/2510.21883}
}
```