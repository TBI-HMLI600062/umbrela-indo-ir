"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF) of BM25 + dense candidates.

Args:
    --bm25      BM25 candidates JSONL (candidates/bm25_top100.jsonl)
    --dense     Dense candidates JSONL (candidates/bgem3_top100.jsonl)
    --output    Output fused candidates JSONL (candidates/hybrid_top100.jsonl)
    --k         RRF constant k (default: 60)
    --top-n     Number of top results to keep per query (default: 100)

Example:
    python retrieval/dense/hybrid.py \\
        --bm25 candidates/bm25_top100.jsonl \\
        --dense candidates/bgem3_top100.jsonl \\
        --output candidates/hybrid_top100.jsonl
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RRF hybrid fusion of BM25 + dense retrieval.")
    parser.add_argument("--bm25", required=True, help="BM25 candidates JSONL")
    parser.add_argument("--dense", required=True, help="Dense candidates JSONL")
    parser.add_argument("--output", required=True, help="Output fused candidates JSONL")
    parser.add_argument("--k", type=int, default=60, help="RRF k constant (default: 60)")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Arvin, E4-T4): implement RRF hybrid fusion")


if __name__ == "__main__":
    main()
