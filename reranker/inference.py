"""
Run BGE reranker on candidate passages and re-score.

Args:
    --model         path to fine-tuned reranker model directory
    --candidates    candidates JSONL file (output of retrieval scripts)
    --data-dir      processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output reranked results file (TREC run format)
    --zero-shot     use zero-shot (un-fine-tuned) BAAI/bge-reranker-v2-m3
    --top-k         top-k passages to rerank per query (default: 100)

Example:
    python reranker/inference.py \\
        --model results/reranker/qwen/ \\
        --candidates candidates/bm25_top100.jsonl \\
        --output results/final/bm25_qwen_rk.json
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Reranker inference on MIRACL-ID candidates.")
    parser.add_argument("--model", required=True, help="Fine-tuned reranker path")
    parser.add_argument("--candidates", required=True, help="Candidates JSONL file")
    parser.add_argument("--data-dir", default="data/miracl-id/")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use zero-shot BAAI/bge-reranker-v2-m3 (ignore --model)")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Faiz, E1-T7): implement reranker inference")


if __name__ == "__main__":
    main()
