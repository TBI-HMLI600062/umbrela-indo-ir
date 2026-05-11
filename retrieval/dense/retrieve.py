"""
Dense retrieval for MIRACL-ID queries using prebuilt FAISS index.

Args:
    --embeddings    path to embeddings directory (with FAISS index)
    --topics        topics TSV file (e.g. data/miracl-id/topics/test.tsv)
    --output        output candidates JSONL file
    --k             top-k per query (default: 100)
    --model         encoder model (same as used in embed_corpus.py)

Example:
    python retrieval/dense/retrieve.py \\
        --embeddings embeddings/bge-m3/ \\
        --topics data/miracl-id/topics/test.tsv \\
        --output candidates/bgem3_top100.jsonl --k 100
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Dense retrieval with FAISS.")
    parser.add_argument("--embeddings", required=True, help="Embeddings directory")
    parser.add_argument("--topics", required=True, help="Topics TSV file")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--model", default=None, help="Encoder model (for query encoding)")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "TODO (Arvin E4-T4 / Karol E5-T3): implement dense retrieval"
    )


if __name__ == "__main__":
    main()
