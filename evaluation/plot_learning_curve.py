"""
Plot nDCG@10 learning curve for training size ablation (RQ2 — Radit).

Reads size_100.json, size_300.json, ..., size_full.json from --results-dir
(produced by eval_pipeline.py) and plots nDCG@10 vs number of training queries.

Optionally overlays baseline lines (e.g. zero-shot reranker, no reranker).

Args:
    --results-dir   directory containing size_*.json result files
    --output        output image path (default: <results-dir>/learning_curve.png)
    --baselines     zero or more JSON result files to draw as horizontal baselines
    --format        png | pdf (default: png)
    --title         optional plot title override

Example:
    python evaluation/plot_learning_curve.py \\
        --results-dir results/final/ \\
        --baselines results/final/bm25_only.json results/final/bm25_zeroshot_rk.json \\
        --output results/final/learning_curve.png
"""

import argparse
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training size ablation learning curve.")
    parser.add_argument("--results-dir", required=True,
                        help="Directory with size_*.json files from eval_pipeline.py")
    parser.add_argument("--output", default=None,
                        help="Output image path (default: <results-dir>/learning_curve.png)")
    parser.add_argument("--baselines", nargs="*", default=[],
                        help="JSON result files to draw as horizontal baselines")
    parser.add_argument("--format", default="png", choices=["png", "pdf"])
    parser.add_argument("--title", default=None, help="Plot title override")
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def stem_to_n(stem: str) -> int | None:
    m = re.match(r"size_(\d+)$", stem)
    if m:
        return int(m.group(1))
    if stem == "size_full":
        return 999_999
    return None


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    points = []
    for path in results_dir.glob("size_*.json"):
        n = stem_to_n(path.stem)
        if n is None:
            continue
        data = load_result(path)
        label = "full" if n == 999_999 else str(n)
        points.append((n, label, data["ndcg@10"]))

    if not points:
        raise SystemExit(
            f"No size_*.json files found in {results_dir}. "
            "Run eval_pipeline.py for each ablation size first."
        )

    points.sort(key=lambda x: x[0])
    x_labels = [p[1] for p in points]
    x_vals = list(range(len(points)))
    y_vals = [p[2] for p in points]

    print("Training size ablation results:")
    for _, label, ndcg in points:
        print(f"  N={label:>6}: nDCG@10 = {ndcg:.4f}")

    baselines = []
    for bp in args.baselines:
        bpath = Path(bp)
        if not bpath.exists():
            print(f"WARNING: baseline file not found: {bpath}, skipping.")
            continue
        data = load_result(bpath)
        label = data.get("reranker", bpath.stem)
        if label == "none":
            label = "no reranker (BM25 only)"
        elif label == "zero_shot":
            label = "zero-shot reranker"
        baselines.append((label, data["ndcg@10"]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_vals, y_vals, marker="o", linewidth=2, markersize=7,
            label="SahabatAI-Gemma2 reranker", color="#2196F3")
    ax.set_xticks(x_vals)
    ax.set_xticklabels(x_labels)

    colors = ["#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    for i, (blabel, bndcg) in enumerate(baselines):
        ax.axhline(bndcg, linestyle="--", linewidth=1.5,
                   color=colors[i % len(colors)], label=blabel)
        ax.text(x_vals[-1] + 0.05, bndcg, f"{bndcg:.4f}",
                va="center", fontsize=8, color=colors[i % len(colors)])

    ax.set_xlabel("Number of training queries (N)", fontsize=11)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_title(args.title or "Training Size Ablation — SahabatAI-Gemma2 Reranker (RQ2)",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    out_path = Path(args.output) if args.output else \
        results_dir / f"learning_curve.{args.format}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, format=args.format)
    plt.close(fig)

    print(f"\nLearning curve saved to {out_path}")


if __name__ == "__main__":
    main()
