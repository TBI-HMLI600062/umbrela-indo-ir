"""
Compute Cohen's kappa between LLM qrels and human qrels (RQ1).

Binarizes LLM scores (score >= threshold → relevant) and computes kappa against
human relevance judgments from MIRACL-ID (both rel=0 and rel=1 entries).

Args:
    --llm-qrels     LLM-generated qrels file (TREC format: qid 0 docid score)
    --human-qrels   human qrels file (TREC format, from data/miracl-id/qrels/human/test.txt)
    --output        output CSV file with kappa scores (default: results/final/kappa.csv)
    --compare-all   compare all .txt files in results/qrels/ against human qrels
    --threshold     min LLM score to treat as relevant (default: 2)

Example:
    python evaluation/metrics.py \\
        --llm-qrels results/qrels/qwen_test.txt \\
        --human-qrels data/miracl-id/qrels/human/test.txt \\
        --output results/final/kappa.csv
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for LLM vs human qrels.")
    parser.add_argument("--llm-qrels", help="LLM qrels file (TREC format)")
    parser.add_argument("--human-qrels", help="Human qrels file (TREC format)")
    parser.add_argument("--output", default="results/final/kappa.csv")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all results/qrels/*.txt files against human qrels")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Min LLM score for relevant (default: 2)")
    return parser.parse_args()


def parse_qrels(path: Path) -> dict:
    """Parse TREC qrels → {qid: {docid: int_score}}."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, score = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def compute_kappa(llm_qrels: dict, human_qrels: dict, threshold: int = 2) -> dict:
    """Compute Cohen's kappa between binarized LLM and human labels.

    Universe = all (qid, docid) pairs present in llm_qrels.
    Human label defaults to 0 for any pair not in human_qrels.
    """
    from sklearn.metrics import cohen_kappa_score

    llm_labels = []
    human_labels = []

    for qid, doc_scores in llm_qrels.items():
        for docid, score in doc_scores.items():
            llm_labels.append(1 if score >= threshold else 0)
            human_labels.append(human_qrels.get(qid, {}).get(docid, 0))

    if not llm_labels:
        return {"kappa": float("nan"), "n_pairs": 0,
                "llm_pos_rate": 0.0, "human_pos_rate": 0.0}

    # kappa is undefined if one side has only one class; guard against that
    if len(set(llm_labels)) < 2 or len(set(human_labels)) < 2:
        kappa = float("nan")
    else:
        kappa = float(cohen_kappa_score(human_labels, llm_labels))

    return {
        "kappa": kappa,
        "n_pairs": len(llm_labels),
        "llm_pos_rate": round(sum(llm_labels) / len(llm_labels), 4),
        "human_pos_rate": round(sum(human_labels) / len(human_labels), 4),
    }


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    human_qrels = parse_qrels(Path(args.human_qrels))
    print(f"Human qrels: {sum(len(v) for v in human_qrels.values()):,} pairs "
          f"across {len(human_qrels):,} queries")

    if args.compare_all:
        qrels_dir = Path("results/qrels")
        judge_files = sorted(qrels_dir.glob("*_test.txt"))
        if not judge_files:
            judge_files = sorted(qrels_dir.glob("*.txt"))
        rows = []
        for path in judge_files:
            llm_qrels = parse_qrels(path)
            stats = compute_kappa(llm_qrels, human_qrels, args.threshold)
            judge_name = path.stem
            rows.append({"judge": judge_name, **stats})
            print(f"  {judge_name}: κ={stats['kappa']:.4f}  "
                  f"n={stats['n_pairs']}  "
                  f"llm_pos={stats['llm_pos_rate']:.3f}  "
                  f"human_pos={stats['human_pos_rate']:.3f}")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["judge", "kappa", "n_pairs",
                                                    "llm_pos_rate", "human_pos_rate"])
            writer.writeheader()
            writer.writerows(rows)
    else:
        if not args.llm_qrels:
            raise ValueError("--llm-qrels is required when --compare-all is not set")
        llm_qrels = parse_qrels(Path(args.llm_qrels))
        print(f"LLM qrels: {sum(len(v) for v in llm_qrels.values()):,} pairs "
              f"across {len(llm_qrels):,} queries")
        stats = compute_kappa(llm_qrels, human_qrels, args.threshold)
        judge_name = Path(args.llm_qrels).stem
        print(f"\nCohen's κ ({judge_name} vs human, threshold={args.threshold}):")
        print(f"  κ = {stats['kappa']:.4f}")
        print(f"  n_pairs = {stats['n_pairs']:,}")
        print(f"  LLM positive rate  = {stats['llm_pos_rate']:.3f}")
        print(f"  Human positive rate = {stats['human_pos_rate']:.3f}")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["judge", "kappa", "n_pairs",
                                                    "llm_pos_rate", "human_pos_rate"])
            writer.writeheader()
            writer.writerow({"judge": judge_name, **stats})

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
