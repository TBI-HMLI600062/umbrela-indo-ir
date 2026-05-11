"""
Build BM25 index for MIRACL-ID corpus using Pyserini.

Requires Java 21: sudo apt install openjdk-21-jdk -y

Args:
    --corpus    path to corpus JSONL file (data/miracl-id/corpus/corpus.jsonl)
    --index     output index directory (default: data/miracl-id/bm25-index/)
    --lang      language for tokenization (default: id)

Example:
    python retrieval/bm25/index.py \\
        --corpus data/miracl-id/corpus/corpus.jsonl \\
        --index data/miracl-id/bm25-index/
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Build Pyserini BM25 index for MIRACL-ID.")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--index", default="data/miracl-id/bm25-index/")
    parser.add_argument("--lang", default="id")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Arvin, E4-T3): implement BM25 indexing with Pyserini")


if __name__ == "__main__":
    main()
