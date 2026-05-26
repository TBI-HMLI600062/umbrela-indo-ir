"""
Mine Qwen3-specific hard negatives from dense retrieval candidates + human qrels.

A "hard negative" here is a document that Qwen3-Embedding ranks highly (within
--hard-neg-cutoff) for a given query but is labelled irrelevant in human qrels.
These are the exact false positives that BGE reranker struggles with after hybrid
BM25+Qwen3 retrieval.

Hard negative priority (per query, within top-K):
  1. Docs with explicit human label=0  (confirmed irrelevant, in top-K → hardest)
  2. Docs absent from human qrels       (unannotated, likely irrelevant, in top-K)
  Positives (label=1) are always excluded from the negative pool.

Args:
    --candidates        Qwen3 candidates JSONL (e.g. candidates/qwen3_train_top100.jsonl)
                        Format: {"qid": str|int, "candidates": [{"docid": str, "score": float}, ...]}
    --human-qrels       Human qrels file in TREC format (e.g. data/miracl-id/qrels/human/train.txt)
    --data-dir          Processed MIRACL-ID directory (default: data/miracl-id/)
    --hard-neg-cutoff   Max Qwen3 rank to consider as hard negative (default: 20)
    --min-explicit      If >0, skip queries with fewer than N explicit label=0 hard negs in top-K.
                        Use 0 to also include unannotated docs as negatives (default: 0)
    --output            Output directory; writes train.jsonl + data_meta.json
    --max-triplets      Max (pos, neg) triplets per query, 0=no limit (default: 50)

Examples:
    # Standard: mine from Qwen3 train candidates + human train qrels
    python reranker/mine_hard_negatives.py \\
        --candidates candidates/qwen3_train_top100.jsonl \\
        --human-qrels data/miracl-id/qrels/human/train.txt \\
        --output results/reranker_data/qwen3_hardneg/

    # Stricter: only use queries where Qwen3 puts >=1 explicit label=0 doc in top-10
    python reranker/mine_hard_negatives.py \\
        --candidates candidates/qwen3_train_top100.jsonl \\
        --human-qrels data/miracl-id/qrels/human/train.txt \\
        --hard-neg-cutoff 10 \\
        --min-explicit 1 \\
        --output results/reranker_data/qwen3_hardneg_strict/
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Mine Qwen3 hard negatives for BGE reranker fine-tuning."
    )
    parser.add_argument("--candidates", required=True,
                        help="Qwen3 candidates JSONL file")
    parser.add_argument("--human-qrels", required=True,
                        help="Human qrels file (TREC format: qid 0 docid score)")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Processed MIRACL-ID directory (default: data/miracl-id/)")
    parser.add_argument("--hard-neg-cutoff", type=int, default=20,
                        help="Max Qwen3 rank to consider as hard neg (default: 20)")
    parser.add_argument("--min-explicit", type=int, default=0,
                        help="Min explicit label=0 hard negs required per query; "
                             "0 = also allow unannotated docs as negatives (default: 0)")
    parser.add_argument("--max-triplets", type=int, default=50,
                        help="Max triplets per query, 0=unlimited (default: 50)")
    parser.add_argument("--output", required=True, help="Output directory")
    return parser.parse_args()


def parse_qrels(path: Path) -> dict:
    """Parse TREC qrels → {qid: {docid: score}}. Handles both label=0 and label=1."""
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


def load_candidates(path: Path) -> list[dict]:
    """Load candidates JSONL → list of {qid: str, candidates: [...]}."""
    records = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            obj["qid"] = str(obj["qid"])  # normalise to str (file uses int)
            records.append(obj)
    return records


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
        print(f"  Warning: {len(missing)} docids not found in corpus")
    return corpus


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)

    print(f"Loading candidates from {args.candidates}...")
    all_candidates = load_candidates(Path(args.candidates))
    print(f"  {len(all_candidates):,} queries")

    print(f"Loading human qrels from {args.human_qrels}...")
    human_qrels = parse_qrels(Path(args.human_qrels))
    print(f"  {len(human_qrels):,} queries with annotations")

    print("Loading topics...")
    topics = load_all_topics(data_dir)
    print(f"  {len(topics):,} queries loaded")

    # Collect all docids we need from the corpus:
    # positives (label=1 in human qrels) + hard neg candidates (top-K in Qwen3)
    needed_docids = set()
    for rec in all_candidates:
        qid = rec["qid"]
        # Top-K candidates from Qwen3
        for c in rec["candidates"][: args.hard_neg_cutoff]:
            needed_docids.add(c["docid"])
        # All human-annotated positives for this query (may not appear in top-K)
        for docid, score in human_qrels.get(qid, {}).items():
            if score >= 1:
                needed_docids.add(docid)

    print(f"Loading corpus subset ({len(needed_docids):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed_docids)

    # Build triplets
    triplets = []
    stats = {
        "n_skipped_no_topic": 0,
        "n_skipped_no_pos": 0,
        "n_skipped_no_hardneg": 0,
        "n_skipped_min_explicit": 0,
        "n_explicit_hardneg": 0,
        "n_unannotated_hardneg": 0,
    }

    for rec in tqdm(all_candidates, desc="Building triplets"):
        qid = rec["qid"]

        if qid not in topics:
            stats["n_skipped_no_topic"] += 1
            continue

        query = topics[qid]
        q_human = human_qrels.get(qid, {})

        # Positives: human label=1, must be in corpus
        positives = [d for d, s in q_human.items() if s >= 1 and d in corpus]
        if not positives:
            stats["n_skipped_no_pos"] += 1
            continue

        # Hard negatives from Qwen3 top-K:
        #   Tier 1: explicit label=0 (confirmed irrelevant, high-ranked by Qwen3)
        #   Tier 2: unannotated (not in qrels at all, high-ranked by Qwen3)
        explicit_hard_negs = []
        unannotated_hard_negs = []
        for c in rec["candidates"][: args.hard_neg_cutoff]:
            docid = c["docid"]
            if docid not in corpus:
                continue
            label = q_human.get(docid, None)
            if label == 0:
                explicit_hard_negs.append(docid)
            elif label is None:
                unannotated_hard_negs.append(docid)
            # label=1 → skip (it's a positive)

        # Enforce --min-explicit filter
        if args.min_explicit > 0 and len(explicit_hard_negs) < args.min_explicit:
            stats["n_skipped_min_explicit"] += 1
            continue

        # Merge: explicit first, then unannotated
        hard_negs = explicit_hard_negs + unannotated_hard_negs
        if not hard_negs:
            stats["n_skipped_no_hardneg"] += 1
            continue

        stats["n_explicit_hardneg"] += len(explicit_hard_negs)
        stats["n_unannotated_hardneg"] += len(unannotated_hard_negs)

        # Build all (pos, neg) pairs, cap if needed
        pairs = [(p, n) for p in positives for n in hard_negs]
        if args.max_triplets > 0 and len(pairs) > args.max_triplets:
            # Bias sampling: prefer pairs where neg comes from explicit_hard_negs
            explicit_pairs = [(p, n) for p, n in pairs if n in explicit_hard_negs]
            unannotated_pairs = [(p, n) for p, n in pairs if n in unannotated_hard_negs]

            n_explicit_take = min(len(explicit_pairs), args.max_triplets)
            n_unann_take = args.max_triplets - n_explicit_take

            sampled = random.sample(explicit_pairs, n_explicit_take)
            if n_unann_take > 0 and unannotated_pairs:
                sampled += random.sample(unannotated_pairs,
                                         min(n_unann_take, len(unannotated_pairs)))
            pairs = sampled

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

    n_processed = (len(all_candidates)
                   - stats["n_skipped_no_topic"]
                   - stats["n_skipped_no_pos"]
                   - stats["n_skipped_no_hardneg"]
                   - stats["n_skipped_min_explicit"])

    meta = {
        "candidates": str(args.candidates),
        "human_qrels": str(args.human_qrels),
        "hard_neg_cutoff": args.hard_neg_cutoff,
        "min_explicit": args.min_explicit,
        "max_triplets": args.max_triplets,
        "n_queries_input": len(all_candidates),
        "n_queries_processed": n_processed,
        "n_triplets": len(triplets),
        **stats,
    }
    with open(output_dir / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone.")
    print(f"  Queries processed  : {n_processed:,} / {len(all_candidates):,}")
    print(f"  Skipped no topic   : {stats['n_skipped_no_topic']:,}")
    print(f"  Skipped no pos     : {stats['n_skipped_no_pos']:,}")
    print(f"  Skipped no hardneg : {stats['n_skipped_no_hardneg']:,}")
    if args.min_explicit > 0:
        print(f"  Skipped min-explicit: {stats['n_skipped_min_explicit']:,}")
    print(f"  Explicit label=0 hard negs used: {stats['n_explicit_hardneg']:,}")
    print(f"  Unannotated hard negs used      : {stats['n_unannotated_hardneg']:,}")
    print(f"  Triplets written   : {len(triplets):,}")
    print(f"  Output             : {output_path}")


if __name__ == "__main__":
    main()
