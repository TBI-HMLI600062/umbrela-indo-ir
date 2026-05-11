"""
BM25 retrieval for MIRACL-ID dev/test queries using Pyserini.

Args:
    --topics    topics TSV file (data/miracl-id/topics/test.tsv)
    --index     BM25 index directory (data/miracl-id/bm25-index/)
    --output    output candidates JSONL file (candidates/bm25_top100.jsonl)
    --k         number of candidates per query (default: 100)
    --lang      language (default: id)

Example:
    python retrieval/bm25/retrieve.py \\
        --topics data/miracl-id/topics/test.tsv \\
        --index data/miracl-id/bm25-index/ \\
        --output candidates/bm25_top100.jsonl --k 100
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="BM25 retrieval with Pyserini.")
    parser.add_argument("--topics", required=True, help="Topics TSV file")
    parser.add_argument("--index", required=True, help="BM25 index directory")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--k", type=int, default=100, help="Top-k per query")
    parser.add_argument("--lang", default="id")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Arvin, E4-T3): implement BM25 retrieval")


if __name__ == "__main__":
    main()
