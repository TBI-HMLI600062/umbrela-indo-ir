"""
Evaluate retrieval + reranking pipeline with nDCG@10 on MIRACL-ID test split.

Computes nDCG@10 against human qrels for any combination of first-stage retriever
and optional reranker.

Args:
    --first-stage   first-stage retrieval run (bm25 | bge_m3 | hybrid | qwen_embed)
                    or path to a TREC run file
    --reranker      reranker model path or 'none' | 'zero_shot' | 'human'
    --data-dir      processed MIRACL-ID directory (default: data/miracl-id/)
    --output        output JSON file with nDCG@10 results
    --candidates-dir  directory with first-stage candidates (default: candidates/)
    --top-k         cutoff for reranking (default: 100)

Example:
    python evaluation/eval_pipeline.py \\
        --first-stage bm25 \\
        --reranker results/reranker/qwen/ \\
        --output results/final/bm25_qwen_rk.json
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval pipeline on MIRACL-ID.")
    parser.add_argument("--first-stage", required=True,
                        help="First-stage type or path to TREC run file")
    parser.add_argument("--reranker", default="none",
                        help="Reranker path or 'none'/'zero_shot'/'human'")
    parser.add_argument("--data-dir", default="data/miracl-id/")
    parser.add_argument("--output", required=True, help="Output JSON results file")
    parser.add_argument("--candidates-dir", default="candidates/")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    raise NotImplementedError("TODO (Faiz, E1-T7): implement nDCG@10 evaluation pipeline")


if __name__ == "__main__":
    main()
