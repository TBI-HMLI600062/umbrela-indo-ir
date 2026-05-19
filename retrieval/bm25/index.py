"""
Build BM25 index for MIRACL-ID corpus using bm25s (no Java required).

Args:
    --corpus    path to corpus JSONL file (data/miracl-id/corpus/corpus.jsonl)
    --index     output index directory (default: data/miracl-id/bm25-index/)

Example:
    python retrieval/bm25/index.py \
        --corpus data/miracl-id/corpus/corpus.jsonl \
        --index data/miracl-id/bm25-index/
"""

import argparse
import json
import time
from pathlib import Path

import bm25s
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build bm25s index for MIRACL-ID corpus.")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--index", default="data/miracl-id/bm25-index/")
    return parser.parse_args()


def load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    docids, texts = [], []
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus"):
            obj = json.loads(line)
            docids.append(obj["docid"])
            texts.append(obj.get("doc", obj.get("text", "")) or "")
    return docids, texts


def main():
    args = parse_args()
    corpus_path = Path(args.corpus)
    index_dir = Path(args.index)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus_path}...")
    docids, texts = load_corpus(corpus_path)
    print(f"Loaded {len(docids):,} passages")

    DOC_TRUNC = 4000
    texts = [t[:DOC_TRUNC] for t in texts]

    t0 = time.time()
    print("Tokenizing...")
    tokenized = bm25s.tokenize(texts, stopwords=None, stemmer=None, show_progress=True)
    print(f"Tokenized in {time.time() - t0:.1f}s | vocab={len(tokenized.vocab):,}")

    t0 = time.time()
    print("Building BM25 index (Lucene: k1=1.5, b=0.75)...")
    index = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    index.index(tokenized, show_progress=True)
    print(f"Indexed in {time.time() - t0:.1f}s")

    index.save(str(index_dir), corpus=docids)
    np.save(index_dir / "docids.npy", np.array(docids))
    print(f"Index saved to {index_dir}")
    print(f"Total passages indexed: {len(docids):,}")


if __name__ == "__main__":
    main()
