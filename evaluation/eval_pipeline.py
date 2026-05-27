"""
Evaluate retrieval + reranking pipeline with nDCG@10 on MIRACL-ID test split.

Loads first-stage candidates, optionally reranks, then computes nDCG@10
against human qrels from data/miracl-id/qrels/human/test.txt.

Args:
    --first-stage   first-stage type (bm25|bge_m3|hybrid|qwen_embed) or path to JSONL
    --reranker      reranker model path, or 'none' | 'zero_shot'
    --data-dir      processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output JSON file with nDCG@10 results
    --candidates-dir  directory with first-stage candidates (default: candidates/)
    --top-k         cutoff for reranking (default: 100)
    --batch-size    CrossEncoder batch size (default: 64)

Example:
    python evaluation/eval_pipeline.py \\
        --first-stage bm25 \\
        --reranker results/reranker/qwen/ \\
        --output results/final/bm25_qwen_rk.json

No reranker (first-stage only):
    python evaluation/eval_pipeline.py \\
        --first-stage bm25 --reranker none \\
        --output results/final/bm25_only.json
"""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


_FIRST_STAGE_FILES = {
    "bm25":        "bm25_test_top100.jsonl",
    "bge_m3":      "bgem3_test_top100.jsonl",
    "hybrid":      "hybrid_test_top100.jsonl",
    "qwen_embed":  "qwen_test_top100.jsonl",
    "qwen3_embed": "qwen3_test_top100.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline on MIRACL-ID.")
    parser.add_argument("--first-stage", required=True,
                        help="First-stage type (bm25|bge_m3|hybrid|qwen_embed) or JSONL path")
    parser.add_argument("--reranker", default="none",
                        help="Reranker model path or 'none'/'zero_shot'")
    parser.add_argument("--data-dir", default="data/miracl-id/")
    parser.add_argument("--output", required=True, help="Output JSON results file")
    parser.add_argument("--candidates-dir", default="candidates/")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def load_candidates_jsonl(path: Path) -> dict:
    """Load candidates JSONL → {qid: [{"docid": str, "score": float}, ...]}."""
    result = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            result[obj["qid"]] = obj["candidates"]
    return result


def load_all_topics(data_dir: Path) -> dict:
    topics = {}
    for tsv in (data_dir / "topics").glob("*.tsv"):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0]] = parts[1]
    return topics


def load_corpus_subset(data_dir: Path, needed_docids: set) -> dict:
    corpus_path = data_dir / "corpus" / "corpus.jsonl"
    corpus = {}
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus", unit=" passages"):
            obj = json.loads(line)
            if obj["docid"] in needed_docids:
                corpus[obj["docid"]] = obj["doc"]
            if len(corpus) == len(needed_docids):
                break
    return corpus


def load_human_qrels(data_dir: Path) -> dict:
    """Load human qrels → {qid: {docid: rel}} for ranx."""
    qrels_path = data_dir / "qrels" / "human" / "test.txt"
    qrels = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def rerank_candidates(candidates_by_qid: dict, topics: dict, corpus: dict,
                       model_id: str, top_k: int, batch_size: int) -> dict:
    """Apply CrossEncoder reranker, return {qid: [{"docid": str, "score": float}, ...]}."""
    from sentence_transformers import CrossEncoder

    print(f"Loading reranker: {model_id}")
    model = CrossEncoder(model_id, max_length=512)

    reranked = {}
    for qid, cands in tqdm(candidates_by_qid.items(), desc="Reranking"):
        query = topics.get(qid, "")
        top_cands = cands[:top_k]
        pairs = [(query, corpus.get(c["docid"], "")) for c in top_cands]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        ranked = sorted(zip(top_cands, scores), key=lambda x: float(x[1]), reverse=True)
        reranked[qid] = [{"docid": c["docid"], "score": float(s)} for c, s in ranked]
    return reranked


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve first-stage candidates file
    if args.first_stage in _FIRST_STAGE_FILES:
        cands_path = Path(args.candidates_dir) / _FIRST_STAGE_FILES[args.first_stage]
    else:
        cands_path = Path(args.first_stage)

    print(f"Loading first-stage candidates from {cands_path}...")
    candidates_by_qid = load_candidates_jsonl(cands_path)
    print(f"  {len(candidates_by_qid):,} queries")

    # Optionally rerank
    if args.reranker in ("none", ""):
        run_data = {qid: {c["docid"]: c["score"] for c in cands}
                    for qid, cands in candidates_by_qid.items()}
        reranker_label = "none"
    else:
        model_id = ("BAAI/bge-reranker-v2-m3" if args.reranker == "zero_shot"
                    else args.reranker)
        reranker_label = "zero_shot" if args.reranker == "zero_shot" else Path(args.reranker).name

        # Load topics + corpus for reranking
        print("Loading topics and corpus for reranking...")
        topics = load_all_topics(data_dir)
        needed_docids = {c["docid"] for cands in candidates_by_qid.values()
                         for c in cands[:args.top_k]}
        corpus = load_corpus_subset(data_dir, needed_docids)

        reranked = rerank_candidates(candidates_by_qid, topics, corpus,
                                      model_id, args.top_k, args.batch_size)
        run_data = {qid: {c["docid"]: c["score"] for c in cands}
                    for qid, cands in reranked.items()}

    # Load human qrels
    print("Loading human qrels...")
    qrels_dict = load_human_qrels(data_dir)
    print(f"  {len(qrels_dict):,} queries with human judgments")

    # Compute nDCG@10 with ranx
    from ranx import Qrels, Run, evaluate

    qrels = Qrels(qrels_dict)
    run = Run(run_data)
    metrics = evaluate(qrels, run, ["ndcg@10", "map@10", "recall@100"])

    result = {
        "ndcg@10":    round(float(metrics["ndcg@10"]), 4),
        "map@10":     round(float(metrics["map@10"]), 4),
        "recall@100": round(float(metrics["recall@100"]), 4),
        "first_stage": args.first_stage,
        "reranker":    reranker_label,
        "top_k":       args.top_k,
        "n_queries":   len(run_data),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nnDCG@10  : {result['ndcg@10']:.4f}")
    print(f"MAP@10   : {result['map@10']:.4f}")
    print(f"R@100    : {result['recall@100']:.4f}")
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
