"""
Evaluate retrieval candidates against qrels.

Args:
    --candidates   glob or space-separated JSONL candidate files
    --qrels        TREC-format qrel file
    --label        short label for this qrel set (e.g. "human" or "qwen")
    --output       output CSV file (default: results/retrieval_scores.csv)

Example:
    python evaluation/eval_retrieval.py \
        --candidates "candidates/bm25_*_top100.jsonl candidates/qwen_*_top100.jsonl" \
        --qrels data/miracl-id/qrels/human/test.txt \
        --label human
"""

import argparse
import csv
import json
import sys
from glob import glob
from pathlib import Path

import pytrec_eval


METRICS = {
    "ndcg_cut_10": "nDCG@10",
    "ndcg_cut_100": "nDCG@100",
    "recall_10": "Recall@10",
    "recall_100": "Recall@100",
    "recip_rank": "MRR",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, nargs="+",
                        help="Candidate JSONL files (globs OK)")
    parser.add_argument("--qrels", required=True, nargs="+",
                        help="TREC qrel files")
    parser.add_argument("--output", default="results/retrieval_scores.csv")
    return parser.parse_args()


def load_qrels(path: str) -> dict:
    qrels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            qid, _, docid, score = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_run(path: str) -> dict:
    run = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["qid"]
            run[qid] = {c["docid"]: c["score"] for c in obj["candidates"]}
    return run


def evaluate(run: dict, qrels: dict) -> dict[str, float]:
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(METRICS.keys()))
    results = evaluator.evaluate(run)
    agg = {m: 0.0 for m in METRICS}
    n = 0
    for qid_scores in results.values():
        for m in METRICS:
            agg[m] += qid_scores.get(m, 0.0)
        n += 1
    if n > 0:
        agg = {m: v / n for m, v in agg.items()}
    return agg


def expand_globs(patterns):
    files = []
    for p in patterns:
        expanded = glob(p)
        files.extend(expanded if expanded else [p])
    return sorted(set(files))


def main():
    args = parse_args()

    candidate_files = expand_globs(args.candidates)
    qrel_files = expand_globs(args.qrels)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for qrel_path in qrel_files:
        qrel_label = Path(qrel_path).stem  # e.g. "train", "qwen_test"
        qrels = load_qrels(qrel_path)
        n_qids = len(qrels)
        print(f"\n=== Qrels: {qrel_path} ({n_qids} queries) ===")

        for cand_path in candidate_files:
            cand_name = Path(cand_path).stem  # e.g. "bm25_train_top100"
            run = load_run(cand_path)
            # filter run to only queries present in qrels
            run_filtered = {qid: docs for qid, docs in run.items() if qid in qrels}
            if not run_filtered:
                print(f"  {cand_name}: no overlapping queries — skipping")
                continue

            scores = evaluate(run_filtered, qrels)
            row = {
                "qrel": qrel_label,
                "candidates": cand_name,
                "n_queries": len(run_filtered),
                **{METRICS[m]: f"{v:.4f}" for m, v in scores.items()},
            }
            rows.append(row)

            print(f"  {cand_name} ({len(run_filtered)} q) | "
                  + " | ".join(f"{METRICS[m]}={v:.4f}" for m, v in scores.items()))

    if not rows:
        print("No results.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["qrel", "candidates", "n_queries"] + list(METRICS.values())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
