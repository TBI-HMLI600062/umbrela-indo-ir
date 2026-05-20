"""
Sample N queries from a TREC-format qrels file for training size ablation.

Randomly samples N unique query IDs (seed=42) and writes all their pairs
to separate output files. Produces one file per requested size + a "full" copy.

Args:
    --qrels     input qrels file (TREC format: qid 0 docid score)
    --sizes     list of query counts to sample (default: 100 300 500 1000)
    --output    output directory for subset files
    --seed      random seed (default: 42)

Outputs (in --output dir):
    qrels_100.txt, qrels_300.txt, qrels_500.txt, qrels_1000.txt, qrels_full.txt

Example:
    python data/sample_qrels.py \\
        --qrels results/qrels/sahabat-gemma_train.txt \\
        --sizes 100 300 500 1000 \\
        --output results/qrels/subsets/
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Sample N queries from a qrels file.")
    parser.add_argument("--qrels", required=True, help="Input qrels file (TREC format)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 300, 500, 1000],
                        help="Query counts to sample (default: 100 300 500 1000)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def load_qrels(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Load TREC qrels → ordered unique qids + {qid: [raw_lines]}."""
    lines_by_qid: dict[str, list[str]] = defaultdict(list)
    qid_order: list[str] = []
    seen: set[str] = set()

    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            qid = parts[0]
            if qid not in seen:
                seen.add(qid)
                qid_order.append(qid)
            lines_by_qid[qid].append(line)

    return qid_order, lines_by_qid


def write_subset(qids: list[str], lines_by_qid: dict, output_path: Path) -> None:
    with open(output_path, "w") as f:
        for qid in qids:
            for line in lines_by_qid[qid]:
                f.write(line + "\n")


def main():
    args = parse_args()
    qrels_path = Path(args.qrels)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading qrels from {qrels_path}...")
    qid_order, lines_by_qid = load_qrels(qrels_path)
    total_queries = len(qid_order)
    total_pairs = sum(len(v) for v in lines_by_qid.values())
    print(f"  {total_queries:,} unique queries | {total_pairs:,} pairs total")

    rng = random.Random(args.seed)
    shuffled = qid_order[:]
    rng.shuffle(shuffled)

    full_path = out_dir / "qrels_full.txt"
    write_subset(shuffled, lines_by_qid, full_path)
    print(f"  qrels_full.txt  : {total_queries:>5} queries | {total_pairs:>6} pairs")

    for n in sorted(args.sizes):
        if n > total_queries:
            print(f"  WARNING: N={n} > available {total_queries}, skipping.")
            continue
        subset_qids = shuffled[:n]
        n_pairs = sum(len(lines_by_qid[q]) for q in subset_qids)
        out_path = out_dir / f"qrels_{n}.txt"
        write_subset(subset_qids, lines_by_qid, out_path)
        print(f"  qrels_{n}.txt   : {n:>5} queries | {n_pairs:>6} pairs → {out_path}")

    print(f"\nDone. Subsets written to {out_dir}")


if __name__ == "__main__":
    main()
