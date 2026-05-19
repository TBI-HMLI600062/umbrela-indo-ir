"""
Hybrid retrieval via Reciprocal Rank Fusion (RRF) of BM25 + dense candidates.

Args:
    --bm25    BM25 candidates JSONL (candidates/bm25_top100.jsonl)
    --dense   Dense candidates JSONL (candidates/bgem3_top100.jsonl)
    --output  Output fused candidates JSONL (candidates/hybrid_top100.jsonl)
    --k       RRF constant k (default: 60)
    --top-n   Number of top results to keep per query (default: 100)

Example:
    python retrieval/dense/hybrid.py \
        --bm25 candidates/bm25_top100.jsonl \
        --dense candidates/bgem3_top100.jsonl \
        --output candidates/hybrid_top100.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="RRF hybrid fusion of BM25 + dense candidates.")
    parser.add_argument("--bm25", required=True, help="BM25 candidates JSONL")
    parser.add_argument("--dense", required=True, help="Dense candidates JSONL")
    parser.add_argument("--output", required=True, help="Output fused candidates JSONL")
    parser.add_argument("--k", type=int, default=60, help="RRF k constant (default: 60)")
    parser.add_argument("--top-n", type=int, default=100)
    return parser.parse_args()


def load_candidates(path: Path) -> dict[str, list[str]]:
    """Returns {qid: [docid, ...]} in ranked order."""
    result = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            result[obj["qid"]] = [c["docid"] for c in obj["candidates"]]
    return result


def rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, docid in enumerate(ranked):
            scores[docid] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading BM25 candidates from {args.bm25}...")
    bm25_cands = load_candidates(Path(args.bm25))
    print(f"Loading dense candidates from {args.dense}...")
    dense_cands = load_candidates(Path(args.dense))

    qids = sorted(set(bm25_cands) | set(dense_cands))
    print(f"Fusing {len(qids):,} queries (k={args.k}, top-n={args.top_n})...")

    with open(output_path, "w") as f:
        for qid in tqdm(qids):
            lists = []
            if qid in bm25_cands:
                lists.append(bm25_cands[qid])
            if qid in dense_cands:
                lists.append(dense_cands[qid])
            fused = rrf_fuse(lists, k=args.k)[: args.top_n]
            candidates = [{"docid": d, "score": s} for d, s in fused]
            f.write(json.dumps({"qid": qid, "candidates": candidates}) + "\n")

    print(f"Hybrid candidates saved to {output_path}")
    print(f"Total queries: {len(qids):,}")


if __name__ == "__main__":
    main()
