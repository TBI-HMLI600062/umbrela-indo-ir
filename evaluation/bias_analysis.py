"""
Bias analysis for RQ3: self-reinforcing bias from LLM judge choice.

Computes delta nDCG@10 (Qwen-embed vs BGE-M3 first-stage) per reranker type,
and generates a bias chart showing whether using a Qwen-based judge/reranker
favors Qwen-embed retrieval.

Args:
    --results-dir   directory containing JSON result files from eval_pipeline.py
    --output        output bias chart image (e.g. results/final/bias_chart.png)
    --format        output format: png | pdf (default: png)

Example:
    python evaluation/bias_analysis.py \\
        --results-dir results/final/ \\
        --output results/final/bias_chart.png
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Bias analysis for RQ3.")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with eval_pipeline.py JSON outputs")
    parser.add_argument("--output", required=True, help="Output chart path")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Karol, E5-T6): implement bias analysis chart")


if __name__ == "__main__":
    main()
