"""
Threshold calibration for LLM judge — zero GPU required.

Sweeps binarization threshold τ on val set and reports kappa per threshold.
Finds optimal τ, then reports test kappa using that threshold.

The default τ=2 in metrics.py is arbitrary. This script finds the τ that
maximizes kappa on held-out val set, then applies it to test set.

Works on any judge that outputs integer 0-3 scores (Qwen, SahabatAI, LoRA, etc).
Since Qwen outputs ∈ {0,1,2,3}, valid thresholds are 1, 2, 3:
  τ=1 → score ≥ 1 = relevant  (only 0 = not relevant)
  τ=2 → score ≥ 2 = relevant  (0,1 = not relevant)   ← current default
  τ=3 → score ≥ 3 = relevant  (only 3 = relevant)

Args:
    --val-qrels-llm     LLM qrels on val split (TREC format)
    --val-qrels-human   Human qrels on val split
    --test-qrels-llm    LLM qrels on test split (TREC format)
    --test-qrels-human  Human qrels on test split
    --output            Output CSV (default: results/final/calibration.csv)
    --judge             Label for judge name in output (default: from filename)

Example:
    python evaluation/calibrate.py \\
        --val-qrels-llm   results/qrels/qwen_val.txt \\
        --val-qrels-human data/miracl-id/qrels/human/val.txt \\
        --test-qrels-llm  results/qrels/qwen_test.txt \\
        --test-qrels-human data/miracl-id/qrels/human/test.txt
"""

import argparse
import csv
from pathlib import Path

from metrics import parse_qrels, compute_kappa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-qrels-llm",    required=True)
    parser.add_argument("--val-qrels-human",  required=True)
    parser.add_argument("--test-qrels-llm",   required=True)
    parser.add_argument("--test-qrels-human", required=True)
    parser.add_argument("--output", default="results/final/calibration.csv")
    parser.add_argument("--judge", default=None,
                        help="Judge label (default: stem of val-qrels-llm file)")
    return parser.parse_args()


def score_distribution(qrels: dict) -> dict:
    """Count how many pairs have each integer score."""
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for doc_scores in qrels.values():
        for score in doc_scores.values():
            counts[score] = counts.get(score, 0) + 1
    return counts


def main():
    args = parse_args()
    judge = args.judge or Path(args.val_qrels_llm).stem

    print(f"Judge: {judge}")
    print("=" * 50)

    val_llm   = parse_qrels(Path(args.val_qrels_llm))
    val_human = parse_qrels(Path(args.val_qrels_human))
    test_llm  = parse_qrels(Path(args.test_qrels_llm))
    test_human = parse_qrels(Path(args.test_qrels_human))

    # Show score distribution on val set
    dist = score_distribution(val_llm)
    total = sum(dist.values())
    print(f"\nVal score distribution ({total:,} pairs):")
    for s, n in sorted(dist.items()):
        bar = "█" * int(30 * n / total)
        print(f"  score={s}: {n:>5,}  ({100*n/total:5.1f}%)  {bar}")

    # Threshold sweep on val set
    thresholds = [1, 2, 3]
    print(f"\nThreshold sweep on val set:")
    print(f"  {'τ':>3}  {'κ (val)':>10}  {'llm_pos':>8}  {'human_pos':>10}")
    print("  " + "-" * 40)

    val_results = []
    for tau in thresholds:
        stats = compute_kappa(val_llm, val_human, threshold=tau)
        val_results.append((tau, stats))
        marker = " ← current default" if tau == 2 else ""
        print(f"  τ={tau}  κ={stats['kappa']:>8.4f}  "
              f"pos={stats['llm_pos_rate']:.3f}  "
              f"human_pos={stats['human_pos_rate']:.3f}{marker}")

    best_tau, best_val_stats = max(val_results, key=lambda x: x[1]["kappa"])
    print(f"\n  Best val threshold: τ={best_tau}  (κ={best_val_stats['kappa']:.4f})")

    # Apply best threshold to test set
    print(f"\nTest set results:")
    print(f"  {'τ':>3}  {'κ (test)':>10}  {'llm_pos':>8}  {'human_pos':>10}")
    print("  " + "-" * 40)

    test_results = []
    for tau in thresholds:
        stats = compute_kappa(test_llm, test_human, threshold=tau)
        test_results.append((tau, stats))
        marker = " ← best val" if tau == best_tau else (
                 " ← current default" if tau == 2 else "")
        print(f"  τ={tau}  κ={stats['kappa']:>8.4f}  "
              f"pos={stats['llm_pos_rate']:.3f}  "
              f"human_pos={stats['human_pos_rate']:.3f}{marker}")

    best_test_tau, best_test_stats = max(test_results, key=lambda x: x[1]["kappa"])
    default_test_stats = next(s for t, s in test_results if t == 2)

    print(f"\n  Default (τ=2):     κ={default_test_stats['kappa']:.4f}")
    print(f"  Best val→test τ={best_tau}: κ={best_test_stats['kappa']:.4f}  "
          f"(Δ={best_test_stats['kappa'] - default_test_stats['kappa']:+.4f})")

    # Save CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for (tv, vs), (tt, ts) in zip(val_results, test_results):
        rows.append({
            "judge":          judge,
            "threshold":      tv,
            "kappa_val":      round(vs["kappa"], 4),
            "kappa_test":     round(ts["kappa"], 4),
            "llm_pos_val":    vs["llm_pos_rate"],
            "llm_pos_test":   ts["llm_pos_rate"],
            "human_pos_val":  vs["human_pos_rate"],
            "human_pos_test": ts["human_pos_rate"],
            "best_val":       tv == best_tau,
        })
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
