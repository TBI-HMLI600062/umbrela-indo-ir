"""
Prepare training data for BGE reranker fine-tuning from LLM qrels.

Converts TREC-format LLM qrels + MIRACL-ID corpus into (query, pos_doc, neg_doc) triplets
for sentence-transformers CrossEncoder training.

Args:
    --qrels         path to LLM qrels file (TREC format: qid 0 docid score)
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output directory for training data
    --min-pos-score minimum LLM score to treat as positive (default: 2)
    --max-triplets  max triplets per query, 0 = no limit (default: 50)

Example:
    python reranker/prepare_data.py \\
        --qrels results/qrels/qwen_train.txt \\
        --output results/reranker_data/qwen/
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare reranker training data from LLM qrels.")
    parser.add_argument("--qrels", required=True, help="LLM qrels file (TREC format)")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Processed MIRACL-ID directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--min-pos-score", type=int, default=2,
                        help="Min LLM score to treat as positive (default: 2)")
    parser.add_argument("--max-triplets", type=int, default=50,
                        help="Max triplets per query, 0=unlimited (default: 50)")
    return parser.parse_args()


def parse_qrels(path: Path) -> dict:
    """Parse TREC qrels → {qid: {docid: score}}."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, score = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    print(f"Loading LLM qrels from {args.qrels}...")
    qrels = parse_qrels(Path(args.qrels))
    print(f"  {len(qrels):,} queries, "
          f"{sum(len(v) for v in qrels.values()):,} total pairs")

    print("Loading topics...")
    topics = load_all_topics(data_dir)
    print(f"  {len(topics):,} queries loaded")

    # Collect needed docids
    needed_docids = {docid for doc_scores in qrels.values() for docid in doc_scores}
    print(f"Loading corpus subset ({len(needed_docids):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed_docids)

    # Build triplets
    triplets = []
    n_skipped_no_topic = 0
    n_skipped_no_pos = 0
    n_skipped_no_neg = 0

    for qid, doc_scores in tqdm(qrels.items(), desc="Building triplets"):
        if qid not in topics:
            n_skipped_no_topic += 1
            continue

        query = topics[qid]
        positives = [d for d, s in doc_scores.items()
                     if s >= args.min_pos_score and d in corpus]
        negatives = [d for d, s in doc_scores.items()
                     if s < args.min_pos_score and d in corpus]

        if not positives:
            n_skipped_no_pos += 1
            continue
        if not negatives:
            n_skipped_no_neg += 1
            continue

        pairs = [(p, n) for p in positives for n in negatives]
        if args.max_triplets > 0 and len(pairs) > args.max_triplets:
            pairs = random.sample(pairs, args.max_triplets)

        for pos_id, neg_id in pairs:
            triplets.append({
                "query": query,
                "pos": corpus[pos_id],
                "neg": corpus[neg_id],
            })

    output_path = output_dir / "train.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Queries processed : {len(qrels) - n_skipped_no_topic - n_skipped_no_pos - n_skipped_no_neg}"
          f" / {len(qrels)}")
    print(f"  Skipped (no topic): {n_skipped_no_topic}")
    print(f"  Skipped (no pos)  : {n_skipped_no_pos}")
    print(f"  Skipped (no neg)  : {n_skipped_no_neg}")
    print(f"  Triplets written  : {len(triplets):,}")
    print(f"  Output            : {output_path}")


if __name__ == "__main__":
    main()
