"""
Fine-tune BAAI/bge-reranker-v2-m3 on LLM-generated qrels.

Args:
    --training-data     path to prepared training data directory
    --model             base model ID (default: BAAI/bge-reranker-v2-m3)
    --output            output directory for fine-tuned model
    --epochs            number of training epochs (default: 3)
    --batch-size        training batch size (default: 32)
    --lr                learning rate (default: 2e-5)
    --val-data          optional validation data directory (for early stopping)

Example:
    python reranker/train.py \\
        --training-data results/reranker_data/qwen/ \\
        --model BAAI/bge-reranker-v2-m3 \\
        --output results/reranker/qwen/ \\
        --epochs 3
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BGE reranker on LLM qrels.")
    parser.add_argument("--training-data", required=True, help="Training data directory")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                        help="Base model (default: BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-data", default=None, help="Validation data for early stopping")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Faiz, E1-T6): implement reranker fine-tuning")


if __name__ == "__main__":
    main()
