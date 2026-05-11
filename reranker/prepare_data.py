"""
Prepare training data for BGE reranker fine-tuning from LLM qrels.

Converts TREC-format LLM qrels + MIRACL-ID corpus into (query, pos_doc, neg_doc) triplets
for sentence-transformers CrossEncoder training.

Args:
    --qrels         path to LLM qrels file (TREC format: qid 0 docid score)
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output directory for training data
    --min-pos-score minimum LLM score to treat as positive (default: 2)

Example:
    python reranker/prepare_data.py \\
        --qrels results/qrels/qwen_train.txt \\
        --output results/reranker_data/qwen/
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare reranker training data from LLM qrels.")
    parser.add_argument("--qrels", required=True, help="LLM qrels file (TREC format)")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Processed MIRACL-ID directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min-pos-score", type=int, default=2,
                        help="Min LLM score to treat as positive (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Faiz, E1-T5): implement reranker data preparation")


if __name__ == "__main__":
    main()
