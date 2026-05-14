"""
Run BGE reranker on candidate passages and re-score.

Loads candidate passages from retrieval output (JSONL), scores each (query, doc) pair
with a CrossEncoder, and writes a TREC run file sorted by score.

Args:
    --model         path to fine-tuned reranker model directory
    --candidates    candidates JSONL file (output of retrieval scripts)
                    Format: {"qid": str, "candidates": [{"docid": str, "score": float}, ...]}
    --data-dir      processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output reranked results file (TREC run format)
    --zero-shot     use zero-shot (un-fine-tuned) BAAI/bge-reranker-v2-m3
    --top-k         top-k passages to rerank per query (default: 100)
    --batch-size    CrossEncoder predict batch size (default: 64)

Example:
    python reranker/inference.py \\
        --model results/reranker/qwen/ \\
        --candidates candidates/bm25_top100.jsonl \\
        --output results/final/bm25_qwen_rk.txt

Zero-shot baseline:
    python reranker/inference.py \\
        --zero-shot \\
        --candidates candidates/bm25_top100.jsonl \\
        --output results/final/bm25_zeroshot_rk.txt
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Reranker inference on MIRACL-ID candidates.")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                        help="Fine-tuned reranker path or HF model ID")
    parser.add_argument("--candidates", required=True, help="Candidates JSONL file")
    parser.add_argument("--data-dir", default="data/miracl-id/")
    parser.add_argument("--output", required=True, help="Output TREC run file")
    parser.add_argument("--zero-shot", action="store_true",
                        help="Use zero-shot BAAI/bge-reranker-v2-m3 (ignore --model path)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top-k candidates to rerank per query (default: 100)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="CrossEncoder predict batch size (default: 64)")
    return parser.parse_args()


def load_all_topics(data_dir: Path) -> dict:
    """Load topics from all splits → {qid: query_text}."""
    topics = {}
    for tsv in (data_dir / "topics").glob("*.tsv"):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0]] = parts[1]
    return topics


def load_corpus_subset(data_dir: Path, needed_docids: set) -> dict:
    """Load only the corpus passages referenced by needed_docids."""
    corpus_path = data_dir / "corpus" / "corpus.jsonl"
    corpus = {}
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus", unit=" passages"):
            obj = json.loads(line)
            if obj["docid"] in needed_docids:
                corpus[obj["docid"]] = obj["doc"]
            if len(corpus) == len(needed_docids):
                break
    missing = needed_docids - corpus.keys()
    if missing:
        print(f"Warning: {len(missing)} docids not found in corpus")
    return corpus


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load candidates
    print(f"Loading candidates from {args.candidates}...")
    candidates_by_qid = {}
    with open(args.candidates) as f:
        for line in f:
            obj = json.loads(line)
            candidates_by_qid[obj["qid"]] = obj["candidates"][: args.top_k]
    print(f"  {len(candidates_by_qid):,} queries, top-{args.top_k} candidates each")

    # Load topics
    print("Loading topics...")
    topics = load_all_topics(data_dir)

    # Collect needed docids
    needed_docids = {
        c["docid"]
        for cands in candidates_by_qid.values()
        for c in cands
    }
    print(f"Loading corpus subset ({len(needed_docids):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed_docids)

    # Load reranker
    from sentence_transformers import CrossEncoder

    model_id = "BAAI/bge-reranker-v2-m3" if args.zero_shot else args.model
    run_name = "zero_shot_bge" if args.zero_shot else Path(args.model).name
    print(f"Loading reranker: {model_id}")
    model = CrossEncoder(model_id, max_length=512)

    # Score and write TREC run
    n_missing_topic = 0
    with open(output_path, "w") as out_f:
        for qid, cands in tqdm(candidates_by_qid.items(), desc="Reranking"):
            if qid not in topics:
                n_missing_topic += 1
                continue

            query = topics[qid]
            pairs = [(query, corpus.get(c["docid"], "")) for c in cands]

            scores = model.predict(pairs, batch_size=args.batch_size,
                                   show_progress_bar=False)

            ranked = sorted(zip(cands, scores), key=lambda x: float(x[1]), reverse=True)
            for rank, (cand, score) in enumerate(ranked, 1):
                out_f.write(f"{qid} Q0 {cand['docid']} {rank} {float(score):.6f} {run_name}\n")

    print(f"\nReranked {len(candidates_by_qid) - n_missing_topic:,} queries")
    if n_missing_topic:
        print(f"Warning: {n_missing_topic} queries skipped (not in topics files)")
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
