"""
Generate comprehensive size ablation analysis: summary CSV + dual-axis AP vs nDCG@10 plot.

Reads:
  results/final/size_*.json          — nDCG@10, MAP@10 from eval_pipeline
  results/models/reranker_*/eval/    — val AP, accuracy, F1 from training
  results/models/reranker_*/training_meta.json — n_triplets

Outputs:
  results/final/ablation_summary.csv
  results/final/ap_vs_ndcg_curve.png

Example:
    python evaluation/plot_ablation_analysis.py
    python evaluation/plot_ablation_analysis.py --results-dir results/final/ --output-dir results/final/
"""

import argparse
import csv
import json
from pathlib import Path


SIZES = [100, 300, 500, 1000, "full"]
EVAL_CSV = "eval/CrossEncoderClassificationEvaluator_val_results.csv"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/final/")
    parser.add_argument("--models-dir", default="results/models/")
    parser.add_argument("--output-dir", default="results/final/")
    return parser.parse_args()


def load_eval_json(results_dir: Path, size) -> dict:
    path = results_dir / f"size_{size}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_training_meta(models_dir: Path, size) -> dict:
    path = models_dir / f"reranker_{size}" / "training_meta.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_val_metrics(models_dir: Path, size) -> dict:
    path = models_dir / f"reranker_{size}" / EVAL_CSV
    if not path.exists():
        return {}
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return {}
    # take last epoch
    last = rows[-1]
    return {
        "val_accuracy":          round(float(last["Accuracy"]), 4),
        "val_f1":                round(float(last["F1"]), 4),
        "val_precision":         round(float(last["Precision"]), 4),
        "val_recall":            round(float(last["Recall"]), 4),
        "val_average_precision": round(float(last["Average_Precision"]), 4),
    }


def build_rows(results_dir, models_dir):
    rows = []
    for size in SIZES:
        eval_data = load_eval_json(results_dir, size)
        meta = load_training_meta(models_dir, size)
        val = load_val_metrics(models_dir, size)

        row = {
            "n_queries":             size,
            "n_triplets":            meta.get("n_triplets", ""),
            "n_train_examples":      meta.get("n_train_examples", ""),
            # IR metrics (vs human qrels)
            "ndcg@10":               eval_data.get("ndcg@10", ""),
            "map@10":                eval_data.get("map@10", ""),
            "recall@100":            eval_data.get("recall@100", ""),
            # Val metrics (vs LLM qrels, last epoch)
            "val_average_precision": val.get("val_average_precision", ""),
            "val_accuracy":          val.get("val_accuracy", ""),
            "val_f1":                val.get("val_f1", ""),
            "val_precision":         val.get("val_precision", ""),
            "val_recall":            val.get("val_recall", ""),
        }
        rows.append(row)
    return rows


def write_csv(rows, out_path: Path):
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {out_path}")


def plot_dual_axis(rows, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_labels = [str(r["n_queries"]) for r in rows]
    x_vals = list(range(len(rows)))
    ap_vals   = [r["val_average_precision"] for r in rows]
    ndcg_vals = [r["ndcg@10"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_ap   = "#FF5722"
    color_ndcg = "#2196F3"

    ax1.plot(x_vals, ap_vals, marker="s", linewidth=2, markersize=7,
             color=color_ap, label="Val Avg Precision (LLM qrels)")
    ax1.set_xlabel("Number of training queries (N)", fontsize=11)
    ax1.set_ylabel("Val Average Precision", fontsize=11, color=color_ap)
    ax1.tick_params(axis="y", labelcolor=color_ap)
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylim(0.8, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, ndcg_vals, marker="o", linewidth=2, markersize=7,
             color=color_ndcg, label="nDCG@10 (human qrels)")
    ax2.set_ylabel("nDCG@10", fontsize=11, color=color_ndcg)
    ax2.tick_params(axis="y", labelcolor=color_ndcg)
    ax2.set_ylim(0.3, 0.6)

    # combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="center right")

    ax1.set_title("LLM Judge Alignment vs Human IR Quality (RQ2)", fontsize=11)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {out_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    models_dir  = Path(args.models_dir)
    out_dir     = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(results_dir, models_dir)

    print("\nAblation summary:")
    print(f"{'N':>6} | {'n_triplets':>10} | {'nDCG@10':>8} | {'MAP@10':>7} | {'Val AP':>7} | {'Val Acc':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{str(r['n_queries']):>6} | {str(r['n_triplets']):>10} | "
              f"{r['ndcg@10']:>8.4f} | {r['map@10']:>7.4f} | "
              f"{r['val_average_precision']:>7.4f} | {r['val_accuracy']:>8.4f}")

    write_csv(rows, out_dir / "ablation_summary.csv")
    plot_dual_axis(rows, out_dir / "ap_vs_ndcg_curve.png")


if __name__ == "__main__":
    main()
