"""
BM25 retrieval for MIRACL-ID queries using a saved bm25s index.

Args:
    --topics    topics TSV file (data/miracl-id/topics/test.tsv)
    --index     BM25 index directory (data/miracl-id/bm25-index/)
    --output    output candidates JSONL (candidates/bm25_top100.jsonl)
    --k         number of candidates per query (default: 100)

Example:
    python retrieval/bm25/retrieve.py \
        --topics data/miracl-id/topics/test.tsv \
        --index data/miracl-id/bm25-index/ \
        --output candidates/bm25_top100.jsonl --k 100
"""

import argparse
import json
import time
from pathlib import Path

import bm25s
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="BM25 retrieval with bm25s.")
    parser.add_argument("--topics", required=True, help="Topics TSV file")
    parser.add_argument("--index", required=True, help="BM25 index directory")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--k", type=int, default=100, help="Top-k per query")
    return parser.parse_args()


def load_topics(topics_path: Path) -> list[tuple[str, str]]:
    topics = []
    with open(topics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            topics.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return topics


def main():
    args = parse_args()
    index_dir = Path(args.index)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading BM25 index from {index_dir}...")
    index = bm25s.BM25.load(str(index_dir), load_corpus=True)
    docids = index.corpus
    print(f"Index loaded: {len(docids):,} passages")

    print(f"Loading topics from {args.topics}...")
    topics = load_topics(Path(args.topics))
    print(f"Topics: {len(topics):,} queries")

    t0 = time.time()
    queries = [q for _, q in topics]
    tokenized_queries = bm25s.tokenize(queries, stopwords=None, stemmer=None)

    results, scores = index.retrieve(tokenized_queries, k=args.k, show_progress=True)

    elapsed = time.time() - t0
    print(f"Retrieved in {elapsed:.1f}s ({len(topics) / elapsed:.0f} queries/s)")

    with open(output_path, "w") as f:
        for i, (qid, _) in enumerate(tqdm(topics, desc="Writing")):
            candidates = [
                {"docid": str(docids[results[i, j]]), "score": float(scores[i, j])}
                for j in range(results.shape[1])
            ]
            f.write(json.dumps({"qid": qid, "candidates": candidates}) + "\n")

    print(f"Candidates saved to {output_path}")


if __name__ == "__main__":
    main()
