"""
Encode MIRACL-ID corpus with a dense encoder and build a FAISS index.

Run in background (4-5h for ~1.44M passages):
    nohup python retrieval/dense/embed_corpus.py \\
        --model BAAI/bge-m3 --corpus data/miracl-id/corpus/corpus.jsonl \\
        --output embeddings/bge-m3/ > log_bge_encode.txt &

Args:
    --model     HF encoder model ID (BAAI/bge-m3 or Qwen/Qwen2.5-7B-Instruct)
    --corpus    corpus JSONL file
    --output    output directory for embeddings + FAISS index
    --batch-size    encoding batch size (default: 64)
    --device    cuda | cpu (default: cuda)

Example (Arvin — BGE-M3):
    python retrieval/dense/embed_corpus.py \\
        --model BAAI/bge-m3 --output embeddings/bge-m3/

Example (Karol — Qwen-embed):
    python retrieval/dense/embed_corpus.py \\
        --model Qwen/Qwen2.5-7B-Instruct --mode embedding \\
        --output embeddings/qwen/ --batch-size 32
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Encode corpus with dense encoder + FAISS.")
    parser.add_argument("--model", required=True, help="HF encoder model ID")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--output", required=True, help="Output embeddings directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="embedding",
                        help="Encoding mode for Qwen (default: embedding)")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError(
        "TODO (Arvin E4-T2 / Karol E5-T2): implement corpus encoding with FlagEmbedding or Qwen"
    )


if __name__ == "__main__":
    main()
