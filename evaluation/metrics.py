"""
Compute Cohen's kappa between LLM qrels and human qrels (RQ1).

Binarizes LLM scores (score >= 2 → relevant) and computes kappa against
human binary relevance from MIRACL-ID test split.

Args:
    --llm-qrels     LLM-generated qrels file (TREC format)
    --human-qrels   human qrels file (TREC format, from data/miracl-id/qrels/human/test.txt)
    --output        output CSV file with kappa scores
    --compare-all   compare all LLM qrels files in results/qrels/ and merge into one table
    --threshold     min LLM score to treat as relevant (default: 2)

Example:
    python evaluation/metrics.py \\
        --llm-qrels results/qrels/qwen_train.txt \\
        --human-qrels data/miracl-id/qrels/human/test.txt \\
        --output results/final/kappa.csv
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for LLM vs human qrels.")
    parser.add_argument("--llm-qrels", help="LLM qrels file (TREC format)")
    parser.add_argument("--human-qrels", help="Human qrels file (TREC format)")
    parser.add_argument("--output", default="results/final/kappa.csv")
    parser.add_argument("--compare-all", action="store_true",
                        help="Merge all judge results into one kappa table")
    parser.add_argument("--threshold", type=int, default=2,
                        help="Min LLM score for relevant (default: 2)")
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Faiz, E1-T3/E1-T4): implement Cohen's kappa computation")


if __name__ == "__main__":
    main()
