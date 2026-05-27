"""
RQ3 Bias Analysis — Self-reinforcing bias from LLM judge / retriever choice.

Two orthogonal bias dimensions are analysed:

  DIMENSION 1 — Reranker bias
    Does a reranker trained on LLM-X judge signal systematically favour retrieval
    system X over a different-family retriever?

  DIMENSION 2 — Judge bias (added per team suggestion)
    Does LLM-X judge inflate the apparent nDCG@10 of retriever X relative to
    how human assessors would rank the same systems?

Analyses implemented
--------------------
Reranker bias (aggregate / perquery / overlap modes):
  1.  result_matrix      nDCG@10 / MAP@10 / Recall@100 for every retriever x reranker
                         combination, plus normalised delta over first-stage baseline
  2.  delta_heatmap      colour-coded grid: % gain/loss vs first-stage baseline
  3.  recall_comparison  bar chart of Recall@100 per first-stage retriever
  4.  perquery_violin    per-query nDCG@10 distribution per system
  5.  ndcg_at_k          mean nDCG at K in {1, 3, 5, 10} per system
  6.  win_loss           per-query win/loss/tie vs reference system
  7.  rank_disruption    Kendall-tau between first-stage and reranked order
  8.  hardneg_overlap    top-K overlap + exclusive hard-negatives between BGE-M3 / Qwen3

Judge bias (judge_matrix mode):
  9.  judge_ndcg_matrix  nDCG@10 per (judge, retriever) — rows=judges, cols=retrievers
  10. judge_delta_chart  nDCG(Qwen3-embed) − nDCG(BGE-M3) per judge; positive = Qwen-bias
  11. leaderboard_corr   Kendall-tau of per-judge system ranking vs human ranking
  12. posrate_effect     scatter: judge positivity rate vs nDCG(Qwen3)/nDCG(BGE-M3) ratio

Modes
-----
  aggregate      No GPU. Reads pre-computed JSON result files.
                 Outputs: result_matrix.md, delta_heatmap.png, recall_comparison.png

  perquery       Reads candidates JSONL; optionally applies a reranker model.
                 CPU-only for first-stage; GPU needed for reranked combos.
                 Outputs: perquery_violin.png, ndcg_at_k.png, win_loss.png

  overlap        CPU-only. Hard-negative overlap at K = 10/20/50/100.
                 Outputs: hardneg_overlap_report.md, overlap_k{K}.json

  judge_matrix   CPU-only. Evaluates all retriever candidates against each LLM judge's
                 qrels (binarised at score >= 2), then builds bias matrix.
                 Outputs: judge_ndcg_matrix.png, judge_delta_chart.png,
                          leaderboard_correlation.png, judge_matrix.json

Usage examples
--------------
  python evaluation/bias_analysis.py --mode aggregate \\
      --results-dir results/final/ \\
      --output results/final/bias_analysis/

  python evaluation/bias_analysis.py --mode perquery \\
      --systems "BGE-M3,Qwen3-embed,Hybrid BM25+Qwen3" \\
      --output results/final/bias_analysis/

  python evaluation/bias_analysis.py --mode overlap \\
      --output results/final/bias_analysis/

  python evaluation/bias_analysis.py --mode judge_matrix \\
      --candidates-dir candidates/ \\
      --output results/final/bias_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from tqdm import tqdm


# ── canonical system registry ─────────────────────────────────────────────────

# Maps result JSON file stem -> (retriever_label, reranker_label)
# Add new entries here as new eval results become available
RESULT_FILE_MAP: Dict[str, Tuple[str, str]] = {
    "hybrid_bm25_qwen3_test":        ("Hybrid BM25+Qwen3",  "none"),
    "bgem3_qwenrk":                  ("BGE-M3",             "Qwen-rk"),
    "qwen3_qwenrk":                  ("Qwen3-embed",         "Qwen-rk"),
    "bgem3_gemma2rk_test":           ("BGE-M3",             "Gemma2-rk"),
    "qwen3_gemma2rk_test":           ("Qwen3-embed",         "Gemma2-rk"),
    "bgem3_qwen3hardnegrk_test":     ("BGE-M3",             "Qwen3-hardneg-rk"),
    "hybrid_qwen3_hardnegrk_test":   ("Hybrid BM25+Qwen3",  "Qwen3-hardneg-rk"),
    "bgem3_bgem3hardnegrk_test":     ("BGE-M3",             "BGE-hardneg-rk"),
    "qwen3_bgem3hardnegrk_test":     ("Qwen3-embed",         "BGE-hardneg-rk"),
}

# Maps candidates column name in retrieval_scores.csv -> retriever label
CSV_RETRIEVER_MAP = {
    "bgem3_test_top100":    "BGE-M3",
    "hybrid_test_top100":   "Hybrid BM25+Qwen3",
    "bm25_test_top100":     "BM25",
    "qwen3_test_top100":    "Qwen3-embed",
}

# Maps retriever label -> candidates JSONL filename in candidates/
CANDIDATES_FILES = {
    "BGE-M3":            "bgem3_test_top100.jsonl",
    "Qwen3-embed":       "qwen3_test_top100.jsonl",
    "Hybrid BM25+Qwen3": "hybrid_bm25_qwen3_test_top100.jsonl",
    "BM25":              "bm25_test_top100.jsonl",
}

RETRIEVER_ORDER = ["BM25", "BGE-M3", "Qwen3-embed", "Hybrid BM25+Qwen3"]
RERANKER_ORDER  = ["none", "Qwen-rk", "Gemma2-rk", "Qwen3-hardneg-rk", "BGE-hardneg-rk"]

# ── LLM judge registry (Dimension 2 — judge bias) ────────────────────────────

# Each entry: display_name -> {path, binarize_threshold, pos_rate (from RQ1), family}
# binarize_threshold: minimum score to treat as relevant (score >= threshold = rel)
JUDGE_REGISTRY: Dict[str, dict] = {
    "Human": {
        "path":      "data/miracl-id/qrels/human/test.txt",
        "threshold": 1,       # binary: label=1 is already relevant
        "pos_rate":  0.3194,
        "family":    "human",
        "kappa":     1.0,
    },
    "SahabatAI-Gemma2\n(zeroshot-bing)": {
        "path":      "data/miracl-id/results/qrels/sahabat-gemma_vllm_zeroshot_bing_test.txt",
        "threshold": 2,
        "pos_rate":  0.4123,
        "family":    "gemma2",
        "kappa":     0.3763,
    },
    "SahabatAI-Gemma2\n(strict)": {
        "path":      "data/miracl-id/results/qrels/sahabat-gemma_vllm_zeroshot_bing_strict_test.txt",
        "threshold": 2,
        "pos_rate":  None,
        "family":    "gemma2",
        "kappa":     None,
    },
    "SahabatAI-Llama3\n(zeroshot-basic)": {
        "path":      "data/miracl-id/results/qrels/sahabat-llama_vllm_zeroshot_basic_test.txt",
        "threshold": 2,
        "pos_rate":  0.5700,
        "family":    "llama3",
        "kappa":     0.3652,
    },
    "SahabatAI-Llama3\n(fewshot-basic)": {
        "path":      "data/miracl-id/results/qrels/sahabat-llama_vllm_fewshot_basic_test.txt",
        "threshold": 2,
        "pos_rate":  None,
        "family":    "llama3",
        "kappa":     None,
    },
    "SahabatAI-Llama3\n(fewshot-bing)": {
        "path":      "data/miracl-id/results/qrels/sahabat-llama_vllm_fewshot_bing_test.txt",
        "threshold": 2,
        "pos_rate":  None,
        "family":    "llama3",
        "kappa":     None,
    },
    "DeepSeek": {
        "path":      "results/qrels/deepseek_test.txt",
        "threshold": 2,
        "pos_rate":  0.2799,
        "family":    "deepseek",
        "kappa":     None,
    },
    "ChatGPT\n(GPT-4o)": {
        "path":      "results/qrels/chatgpt_test.txt",
        "threshold": 2,
        "pos_rate":  0.2494,
        "family":    "gpt4o",
        "kappa":     None,
    },
}

# Retrievers to include in judge_matrix mode (order = left-to-right in plots)
JUDGE_MATRIX_RETRIEVERS = ["BM25", "BGE-M3", "Qwen3-embed", "Hybrid BM25+Qwen3"]

# Family colours for judge_matrix plots
FAMILY_COLORS = {
    "human":   "#2c7bb6",
    "qwen":    "#d7191c",
    "gemma2":  "#1a9641",
    "llama3":  "#fdae61",
    "deepseek": "#7b2d8b",
    "gpt4o":   "#555555",
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_result_jsons(results_dir: Path) -> Dict[Tuple[str, str], dict]:
    """Load pre-computed eval JSON files; returns {(retriever, reranker): metrics}."""
    data = {}
    for stem, (ret, rk) in RESULT_FILE_MAP.items():
        path = results_dir / f"{stem}.json"
        if path.exists():
            with open(path) as f:
                data[(ret, rk)] = json.load(f)
        else:
            print(f"  [missing] {path.name}")
    return data


def load_baselines_from_csv(csv_path: Path) -> Dict[str, dict]:
    """Load first-stage baselines from retrieval_scores.csv -> {retriever: metrics}."""
    baselines: Dict[str, dict] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("qrel") != "test":
                continue
            key = row["candidates"]
            if key not in CSV_RETRIEVER_MAP:
                continue
            ret = CSV_RETRIEVER_MAP[key]
            baselines[ret] = {
                "ndcg@10":    float(row["nDCG@10"]),
                "recall@100": float(row["Recall@100"]),
            }
    return baselines


def load_candidates_jsonl(path: Path) -> Dict[str, List[dict]]:
    """Load candidates JSONL -> {qid: [{"docid": str, "score": float}, ...]}."""
    result: Dict[str, List[dict]] = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            result[str(obj["qid"])] = obj["candidates"]
    return result


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """Load TREC qrels -> {qid: {docid: rel}}."""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rel = str(parts[0]), parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


# ── per-query metric computation ──────────────────────────────────────────────

def _dcg(rels: List[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(ranked_docids: List[str], q_qrels: Dict[str, int], k: int) -> float:
    """Compute nDCG@k for a single query."""
    ranked   = ranked_docids[:k]
    rels     = [q_qrels.get(d, 0) for d in ranked]
    ideal    = sorted(q_qrels.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0


def compute_perquery_ndcg(
    candidates: Dict[str, List[dict]],
    qrels: Dict[str, Dict[str, int]],
    ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, Dict[int, float]]:
    """Compute per-query nDCG@k for all queries; returns {qid: {k: score}}."""
    scores: Dict[str, Dict[int, float]] = {}
    for qid, q_qrels in qrels.items():
        if qid not in candidates:
            continue
        ranked = [c["docid"] for c in candidates[qid]]
        scores[qid] = {k: ndcg_at_k(ranked, q_qrels, k) for k in ks}
    return scores


# ── rank disruption (Kendall-tau) ─────────────────────────────────────────────

def kendall_tau_disruption(
    first_stage: Dict[str, List[dict]],
    reranked: Dict[str, List[dict]],
    top_k: int = 20,
) -> Dict[str, float]:
    """Kendall-tau between first-stage and reranked order per query.

    tau = 1.0  -> identical order (reranker agreed completely)
    tau = 0.0  -> random reordering
    tau = -1.0 -> completely inverted
    """
    taus: Dict[str, float] = {}
    for qid in first_stage:
        if qid not in reranked:
            continue
        fs_docs = [c["docid"] for c in first_stage[qid][:top_k]]
        rk_docs = [c["docid"] for c in reranked[qid][:top_k]]
        common  = [d for d in fs_docs if d in set(rk_docs)]
        if len(common) < 2:
            continue
        fs_rank = {d: i for i, d in enumerate(fs_docs)}
        rk_rank = {d: i for i, d in enumerate(rk_docs)}
        x = np.array([fs_rank[d] for d in common])
        y = np.array([rk_rank[d] for d in common])
        tau, _ = stats.kendalltau(x, y)
        taus[qid] = float(tau)
    return taus


# ── hard-negative overlap ─────────────────────────────────────────────────────

def compute_hardneg_overlap(
    cands_a: Dict[str, List[dict]],
    cands_b: Dict[str, List[dict]],
    qrels: Dict[str, Dict[str, int]],
    top_k: int = 20,
) -> dict:
    """Overlap between top-K candidates of two retrievers.

    Returns per-query averages of:
      overlap_rate         : |top-k(A) ∩ top-k(B)| / k
      hardneg_both         : confirmed-irrelevant docs in shared pool
      hardneg_a_only       : confirmed-irrelevant docs unique to A
      hardneg_b_only       : confirmed-irrelevant docs unique to B
      pct_hardneg_excl_a   : fraction of A's hard-negs not seen by B
    """
    overlap_rates: List[float] = []
    hn_both: List[float] = []
    hn_a_only: List[float] = []
    hn_b_only: List[float] = []

    for qid in set(cands_a) & set(cands_b):
        q_qrels = qrels.get(qid, {})
        top_a   = {c["docid"] for c in cands_a[qid][:top_k]}
        top_b   = {c["docid"] for c in cands_b[qid][:top_k]}
        shared  = top_a & top_b
        a_only  = top_a - top_b
        b_only  = top_b - top_a

        def n_hardneg(docs):
            # unlisted docs in TREC qrels = implicitly irrelevant (label 0)
            return sum(1 for d in docs if q_qrels.get(d, 0) < 1)

        overlap_rates.append(len(shared) / top_k)
        hn_both.append(n_hardneg(shared))
        hn_a_only.append(n_hardneg(a_only))
        hn_b_only.append(n_hardneg(b_only))

    mean_hn_a    = float(np.mean(hn_a_only)) if hn_a_only else 0.0
    mean_hn_both = float(np.mean(hn_both))   if hn_both   else 0.0
    return {
        "n_queries":            len(overlap_rates),
        "top_k":                top_k,
        "mean_overlap_rate":    float(np.mean(overlap_rates)) if overlap_rates else 0.0,
        "mean_hardneg_both":    mean_hn_both,
        "mean_hardneg_a_only":  mean_hn_a,
        "mean_hardneg_b_only":  float(np.mean(hn_b_only)) if hn_b_only else 0.0,
        "pct_hardneg_excl_a":   mean_hn_a / (mean_hn_a + mean_hn_both + 1e-9),
    }


# ── win / loss / tie ──────────────────────────────────────────────────────────

def win_loss_tie(
    scores_a: Dict[str, float],
    scores_b: Dict[str, float],
    eps: float = 1e-4,
) -> dict:
    """Per-query win/loss/tie for system A versus reference system B."""
    wins = losses = ties = 0
    for qid in set(scores_a) & set(scores_b):
        diff = scores_a[qid] - scores_b[qid]
        if diff > eps:
            wins += 1
        elif diff < -eps:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "n": len(set(scores_a) & set(scores_b))}


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_delta_heatmap(
    results: Dict[Tuple[str, str], dict],
    baselines: Dict[str, dict],
    output_path: Path,
) -> None:
    """Heatmap of delta nDCG@10 (% relative improvement vs first-stage baseline).
    Red = hurts performance, green = helps.
    """
    retrievers = [r for r in RETRIEVER_ORDER if r in baselines]
    rerankers  = [rk for rk in RERANKER_ORDER if rk != "none"]

    matrix = np.full((len(retrievers), len(rerankers)), np.nan)
    annot  = [["" for _ in rerankers] for _ in retrievers]

    for ri, ret in enumerate(retrievers):
        base = baselines.get(ret, {}).get("ndcg@10")
        if base is None:
            continue
        for ci, rk in enumerate(rerankers):
            val = results.get((ret, rk), {}).get("ndcg@10")
            if val is None:
                annot[ri][ci] = "—"
                continue
            delta_pct = (val - base) / base * 100
            matrix[ri, ci] = delta_pct
            sign = "+" if delta_pct >= 0 else ""
            annot[ri][ci] = f"{sign}{delta_pct:.1f}%\n({val:.4f})"

    vmax = max(abs(np.nanmax(matrix)) if not np.all(np.isnan(matrix)) else 5,
               abs(np.nanmin(matrix)) if not np.all(np.isnan(matrix)) else 5, 5)

    fig, ax = plt.subplots(figsize=(len(rerankers) * 2.6 + 1.4, len(retrievers) * 1.8 + 1.4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(rerankers)))
    ax.set_yticks(range(len(retrievers)))
    ax.set_xticklabels(rerankers, fontsize=10)
    ax.set_yticklabels(retrievers, fontsize=10)
    ax.set_xlabel("Reranker", fontsize=11)
    ax.set_ylabel("First-Stage Retriever", fontsize=11)
    ax.set_title(
        "RQ3 Bias Analysis — Δ nDCG@10 vs first-stage baseline (%)\n"
        "Red = hurts  |  Green = helps", fontsize=12
    )

    for ri in range(len(retrievers)):
        for ci in range(len(rerankers)):
            val = matrix[ri, ci]
            txt = annot[ri][ci]
            color = "white" if (not np.isnan(val) and abs(val) > vmax * 0.55) else "black"
            ax.text(ci, ri, txt, ha="center", va="center", fontsize=8.5, color=color)

    plt.colorbar(im, ax=ax, label="Δ nDCG@10 (%)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_recall_comparison(
    baselines: Dict[str, dict],
    output_path: Path,
) -> None:
    """Horizontal bar chart of Recall@100 per first-stage retriever."""
    retrievers  = [r for r in RETRIEVER_ORDER if r in baselines]
    recall_vals = [baselines[r].get("recall@100", 0) for r in retrievers]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][:len(retrievers)]

    fig, ax = plt.subplots(figsize=(8, max(3, len(retrievers) * 0.9 + 1)))
    bars = ax.barh(retrievers, recall_vals, color=colors)
    ax.set_xlabel("Recall@100 (test split, human qrels)", fontsize=11)
    ax.set_title(
        "First-Stage Recall@100\n"
        "(upper bound of relevant docs available to reranker)", fontsize=12
    )
    ax.set_xlim(0, 1.02)
    for bar, val in zip(bars, recall_vals):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_ndcg_at_k(
    systems: Dict[str, Dict[str, Dict[int, float]]],
    output_path: Path,
    ks: Tuple[int, ...] = (1, 3, 5, 10),
) -> None:
    """Line plot of mean nDCG@K for K in {1,3,5,10} per system."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10.colors
    for (sys_name, per_query), color in zip(systems.items(), colors):
        means = [np.mean([s[k] for s in per_query.values() if k in s]) for k in ks]
        ax.plot(ks, means, marker="o", label=sys_name, color=color, linewidth=2)

    ax.set_xticks(ks)
    ax.set_xticklabels([f"nDCG@{k}" for k in ks])
    ax.set_ylabel("Mean nDCG", fontsize=11)
    ax.set_title("nDCG@K breakdown — bias visible at low K?", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_perquery_violin(
    systems: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Violin plot of per-query nDCG@10 distribution per system."""
    sys_names = list(systems.keys())
    data      = [list(systems[s].values()) for s in sys_names]

    fig, ax = plt.subplots(figsize=(max(8, len(sys_names) * 1.9), 5))
    parts = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(plt.cm.tab10.colors[i % 10])
        pc.set_alpha(0.7)

    ax.set_xticks(range(1, len(sys_names) + 1))
    ax.set_xticklabels(sys_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("nDCG@10", fontsize=11)
    ax.set_title("Per-query nDCG@10 distribution", fontsize=12)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_win_loss(
    wlt_dict: Dict[str, dict],
    reference_name: str,
    output_path: Path,
) -> None:
    """Stacked bar: win/loss/tie per system vs reference system."""
    labels = list(wlt_dict.keys())
    wins   = [wlt_dict[s]["wins"]   for s in labels]
    losses = [wlt_dict[s]["losses"] for s in labels]
    ties   = [wlt_dict[s]["ties"]   for s in labels]
    x      = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.8), 5))
    ax.bar(x, wins,   0.5, label="Win",  color="#2ca02c")
    ax.bar(x, losses, 0.5, bottom=wins,  label="Loss", color="#d62728")
    ax.bar(x, ties,   0.5, bottom=[wins[i] + losses[i] for i in range(len(labels))],
           label="Tie", color="#aec7e8")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Number of queries", fontsize=11)
    ax.set_title(f"Per-query Win / Loss / Tie vs {reference_name}", fontsize=12)
    ax.legend()
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_rank_disruption(
    taus: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Box plot of Kendall-tau per reranker. Low tau = aggressive reordering."""
    labels = list(taus.keys())
    data   = [list(taus[s].values()) for s in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 2.2), 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(plt.cm.tab10.colors[i % 10])
        patch.set_alpha(0.7)

    ax.axhline(y=1.0, color="green",  linestyle="--", alpha=0.4, label="tau=1 (no change)")
    ax.axhline(y=0.0, color="orange", linestyle="--", alpha=0.4, label="tau=0 (random)")
    ax.set_ylabel("Kendall-tau (first-stage vs reranked top-20)", fontsize=11)
    ax.set_title(
        "Rank Disruption per Reranker\n"
        "Low tau = reranker disagrees with retriever ordering", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── text builders ─────────────────────────────────────────────────────────────

def build_result_matrix_md(
    results: Dict[Tuple[str, str], dict],
    baselines: Dict[str, dict],
) -> str:
    """Markdown table of all results with absolute and relative delta."""
    rows = []
    rows.append(
        "| System | Retriever | Reranker | nDCG@10 | Δ abs | Δ % | Recall@100 |"
    )
    rows.append("|--------|-----------|----------|---------|-------|-----|------------|")

    for ret in RETRIEVER_ORDER:
        if ret not in baselines:
            continue
        m = baselines[ret]
        rows.append(
            f"| {ret} only | {ret} | — "
            f"| **{m['ndcg@10']:.4f}** | — | — "
            f"| {m.get('recall@100', '—'):.4f} |"
        )

    def sort_key(item):
        (ret, rk), _ = item
        ri = RETRIEVER_ORDER.index(ret) if ret in RETRIEVER_ORDER else 99
        rki = RERANKER_ORDER.index(rk) if rk in RERANKER_ORDER else 99
        return (ri, rki)

    for (ret, rk), m in sorted(results.items(), key=sort_key):
        base = baselines.get(ret, {}).get("ndcg@10")
        if base is not None:
            delta_abs = m["ndcg@10"] - base
            delta_pct = delta_abs / base * 100
            sign = "+" if delta_abs >= 0 else ""
            delta_str = f"{sign}{delta_abs:.4f}"
            pct_str   = f"{sign}{delta_pct:.1f}%"
        else:
            delta_str = pct_str = "—"
        recall = m.get("recall@100", "—")
        recall_str = f"{recall:.4f}" if isinstance(recall, float) else "—"
        rows.append(
            f"| {ret} + {rk} | {ret} | {rk} "
            f"| {m['ndcg@10']:.4f} | {delta_str} | {pct_str} "
            f"| {recall_str} |"
        )
    return "\n".join(rows)


# ── mode runners ──────────────────────────────────────────────────────────────

def run_aggregate(args) -> None:
    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Mode: aggregate ===")
    print(f"Loading result JSONs from {results_dir}...")
    results = load_result_jsons(results_dir)
    print(f"  {len(results)} result files loaded")

    print(f"Loading first-stage baselines from {args.retrieval_csv}...")
    baselines = load_baselines_from_csv(Path(args.retrieval_csv))
    print(f"  Baselines: {list(baselines.keys())}")

    # 1. Result matrix markdown
    table = build_result_matrix_md(results, baselines)
    (output_dir / "result_matrix.md").write_text(table)
    print(f"\n{table}\n")

    # 2. Delta heatmap
    print("Generating delta heatmap...")
    plot_delta_heatmap(results, baselines, output_dir / "delta_heatmap.png")

    # 3. Recall bar chart
    print("Generating recall comparison chart...")
    plot_recall_comparison(baselines, output_dir / "recall_comparison.png")

    # 4. JSON summary
    summary = {
        "baselines": baselines,
        "results": {f"{ret}+{rk}": m for (ret, rk), m in results.items()},
    }
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"\nAll outputs saved to: {output_dir}")


def run_perquery(args) -> None:
    cands_dir  = Path(args.candidates_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(Path(args.qrels))
    ks    = (1, 3, 5, 10)

    system_names = [s.strip() for s in args.systems.split(",")]
    print(f"=== Mode: perquery — systems: {system_names} ===")

    perquery_scores: Dict[str, Dict[str, Dict[int, float]]] = {}
    perquery_k10:   Dict[str, Dict[str, float]] = {}

    for sys_name in system_names:
        if sys_name not in CANDIDATES_FILES:
            print(f"  [skip] '{sys_name}' not in CANDIDATES_FILES")
            continue
        path = cands_dir / CANDIDATES_FILES[sys_name]
        if not path.exists():
            print(f"  [skip] {path} not found")
            continue
        print(f"  Loading {sys_name} from {path}...")
        cands = load_candidates_jsonl(path)

        if args.reranker_model:
            print(f"  Applying reranker {args.reranker_model}...")
            import sys as _sys
            _sys.path.insert(0, str(Path(__file__).parent.parent))
            from evaluation.eval_pipeline import (
                load_all_topics, load_corpus_subset, rerank_candidates,
            )
            data_dir = Path(args.data_dir)
            topics   = load_all_topics(data_dir)
            needed   = {c["docid"] for cs in cands.values() for c in cs[:100]}
            corpus   = load_corpus_subset(data_dir, needed)
            cands    = rerank_candidates(cands, topics, corpus,
                                          args.reranker_model, 100, 64)

        scores = compute_perquery_ndcg(cands, qrels, ks=ks)
        perquery_scores[sys_name] = scores
        perquery_k10[sys_name]    = {qid: s[10] for qid, s in scores.items()}
        mean_k10 = np.mean(list(perquery_k10[sys_name].values()))
        print(f"    mean nDCG@10 = {mean_k10:.4f}  ({len(scores)} queries)")

    if len(perquery_scores) < 2:
        print("Need >= 2 systems for comparison plots.")
        return

    print("\nGenerating violin plot...")
    plot_perquery_violin(perquery_k10, output_dir / "perquery_violin.png")

    print("Generating nDCG@K plot...")
    plot_ndcg_at_k(perquery_scores, output_dir / "ndcg_at_k.png", ks=ks)

    ref_name = system_names[0]
    wlt = {s: win_loss_tie(perquery_k10[s], perquery_k10[ref_name])
           for s in system_names[1:] if s in perquery_k10}
    if wlt:
        print("Generating win/loss chart...")
        plot_win_loss(wlt, ref_name, output_dir / "win_loss.png")

    serialisable = {
        s: {qid: {str(k): v for k, v in ks_scores.items()}
            for qid, ks_scores in sq.items()}
        for s, sq in perquery_scores.items()
    }
    (output_dir / "perquery_scores.json").write_text(
        json.dumps(serialisable, indent=2)
    )
    print(f"\nAll outputs saved to: {output_dir}")


def run_overlap(args) -> None:
    cands_dir  = Path(args.candidates_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    qrels = load_qrels(Path(args.qrels))

    print("=== Mode: overlap ===")
    for path in [cands_dir / CANDIDATES_FILES["BGE-M3"],
                 cands_dir / CANDIDATES_FILES["Qwen3-embed"]]:
        if not path.exists():
            print(f"  [missing] {path}")
            return

    print("Loading BGE-M3 candidates...")
    bgem3 = load_candidates_jsonl(cands_dir / CANDIDATES_FILES["BGE-M3"])
    print("Loading Qwen3-embed candidates...")
    qwen3 = load_candidates_jsonl(cands_dir / CANDIDATES_FILES["Qwen3-embed"])

    lines = ["# Hard-Negative Overlap: BGE-M3 vs Qwen3-embed\n",
             "> Low overlap + high exclusive hard-neg rate means hard negatives are",
             "> retriever-specific — which explains why cross-family rerankers struggle.\n"]

    for k in (10, 20, 50, 100):
        res = compute_hardneg_overlap(bgem3, qwen3, qrels, top_k=k)
        excl_a = res["pct_hardneg_excl_a"]
        print(f"  @K={k}: overlap={res['mean_overlap_rate']:.1%}, "
              f"hardneg BGE-M3-only={res['mean_hardneg_a_only']:.2f}, "
              f"hardneg Qwen3-only={res['mean_hardneg_b_only']:.2f}, "
              f"excl_BGE-M3={excl_a:.1%}")
        lines += [
            f"## K = {k}",
            f"- Mean overlap rate: **{res['mean_overlap_rate']:.1%}** "
            f"({res['mean_overlap_rate']*k:.1f} / {k} docs shared per query)",
            f"- Avg confirmed-irrelevant docs in shared pool: {res['mean_hardneg_both']:.2f}",
            f"- Avg confirmed-irrelevant docs exclusive to BGE-M3: **{res['mean_hardneg_a_only']:.2f}**",
            f"- Avg confirmed-irrelevant docs exclusive to Qwen3-embed: **{res['mean_hardneg_b_only']:.2f}**",
            f"- % of BGE-M3 hard-negs not seen by Qwen3: **{excl_a:.1%}**",
            "",
        ]
        (output_dir / f"overlap_k{k}.json").write_text(json.dumps(res, indent=2))

    report_path = output_dir / "hardneg_overlap_report.md"
    report_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {report_path}")


# ── judge bias helpers ────────────────────────────────────────────────────────

def load_qrels_binarized(
    qrels_path: Path,
    threshold: int = 2,
) -> Dict[str, Dict[str, int]]:
    """Load LLM qrels and binarize: score >= threshold -> 1, else -> 0.

    Queries with no judgements for a doc are treated as 0 (not relevant).
    """
    raw: Dict[str, Dict[str, int]] = {}
    with open(qrels_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, score = str(parts[0]), parts[2], int(parts[3])
            raw.setdefault(qid, {})[docid] = 1 if score >= threshold else 0
    return raw


def compute_mean_ndcg(
    candidates: Dict[str, List[dict]],
    qrels: Dict[str, Dict[str, int]],
    k: int = 10,
) -> float:
    """Compute mean nDCG@k across all queries that have qrels."""
    scores = []
    for qid, q_qrels in qrels.items():
        if qid not in candidates:
            continue
        ranked = [c["docid"] for c in candidates[qid]]
        scores.append(ndcg_at_k(ranked, q_qrels, k))
    return float(np.mean(scores)) if scores else 0.0


def compute_system_ranking(
    ndcg_per_system: Dict[str, float],
) -> Dict[str, int]:
    """Return rank of each system (1 = best) based on nDCG@10."""
    sorted_systems = sorted(ndcg_per_system.items(), key=lambda x: x[1], reverse=True)
    return {sys: rank + 1 for rank, (sys, _) in enumerate(sorted_systems)}


# ── judge bias plots ──────────────────────────────────────────────────────────

def plot_judge_ndcg_matrix(
    matrix: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Heatmap: rows = judges, cols = retrievers, cells = nDCG@10.

    Two colour scales: absolute nDCG (background) and family annotation (left axis).
    """
    judge_names = list(matrix.keys())
    ret_names   = JUDGE_MATRIX_RETRIEVERS

    data = np.array([
        [matrix[j].get(r, np.nan) for r in ret_names]
        for j in judge_names
    ])

    fig, ax = plt.subplots(figsize=(len(ret_names) * 2.2 + 2, len(judge_names) * 1.0 + 1.8))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(ret_names)))
    ax.set_yticks(range(len(judge_names)))
    ax.set_xticklabels(ret_names, fontsize=10)
    ax.set_yticklabels(judge_names, fontsize=9)
    ax.set_xlabel("Retriever", fontsize=11)
    ax.set_ylabel("Judge (ground truth)", fontsize=11)
    ax.set_title(
        "nDCG@10 per Judge × Retriever\n"
        "(each row: judge's qrels used as ground truth, score≥2 = relevant)",
        fontsize=12,
    )

    for ri in range(len(judge_names)):
        for ci in range(len(ret_names)):
            val = data[ri, ci]
            if not np.isnan(val):
                txt_color = "white" if val > 0.65 else "black"
                ax.text(ci, ri, f"{val:.4f}", ha="center", va="center",
                        fontsize=8.5, color=txt_color)
            else:
                ax.text(ci, ri, "—", ha="center", va="center", fontsize=9, color="gray")

    # Colour-coded family marker on y-axis labels
    for i, jname in enumerate(judge_names):
        meta   = JUDGE_REGISTRY.get(jname, {})
        family = meta.get("family", "")
        color  = FAMILY_COLORS.get(family, "black")
        ax.get_yticklabels()[i].set_color(color)

    plt.colorbar(im, ax=ax, label="nDCG@10")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_judge_delta_chart(
    matrix: Dict[str, Dict[str, float]],
    output_path: Path,
    ret_a: str = "Qwen3-embed",
    ret_b: str = "BGE-M3",
) -> None:
    """Bar chart: nDCG(ret_a) − nDCG(ret_b) per judge.

    Positive = judge favours ret_a, negative = favours ret_b.
    Bars coloured by LLM family.
    """
    judge_names = list(matrix.keys())
    deltas      = [matrix[j].get(ret_a, 0) - matrix[j].get(ret_b, 0)
                   for j in judge_names]
    colors      = [FAMILY_COLORS.get(JUDGE_REGISTRY.get(j, {}).get("family", ""), "#888")
                   for j in judge_names]

    fig, ax = plt.subplots(figsize=(max(8, len(judge_names) * 1.5), 5))
    bars = ax.bar(range(len(judge_names)), deltas, color=colors, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)

    ax.set_xticks(range(len(judge_names)))
    ax.set_xticklabels(judge_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(f"nDCG@10 ({ret_a}) − nDCG@10 ({ret_b})", fontsize=10)
    ax.set_title(
        f"Judge Bias: {ret_a} vs {ret_b}\n"
        f"Positive bar = judge favours {ret_a}  |  Negative = favours {ret_b}",
        fontsize=12,
    )

    for bar, delta in zip(bars, deltas):
        sign = "+" if delta >= 0 else ""
        ax.text(bar.get_x() + bar.get_width() / 2,
                delta + (0.003 if delta >= 0 else -0.006),
                f"{sign}{delta:.4f}", ha="center", va="bottom" if delta >= 0 else "top",
                fontsize=8)

    # Legend for family colours
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f.capitalize())
                       for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_leaderboard_correlation(
    rankings_per_judge: Dict[str, Dict[str, int]],
    reference_judge: str,
    output_path: Path,
) -> None:
    """Bar chart: Kendall-tau of each judge's system ranking vs reference (Human).

    Higher tau = judge ranks systems similarly to humans.
    """
    if reference_judge not in rankings_per_judge:
        print(f"  [skip] reference judge '{reference_judge}' not found")
        return

    ref_ranks = rankings_per_judge[reference_judge]
    judge_names = [j for j in rankings_per_judge if j != reference_judge]
    taus        = []

    for jname in judge_names:
        j_ranks  = rankings_per_judge[jname]
        common   = sorted(set(ref_ranks) & set(j_ranks))
        if len(common) < 2:
            taus.append(np.nan)
            continue
        x = np.array([ref_ranks[s] for s in common])
        y = np.array([j_ranks[s]   for s in common])
        tau, _ = stats.kendalltau(x, y)
        taus.append(float(tau))

    colors = [FAMILY_COLORS.get(JUDGE_REGISTRY.get(j, {}).get("family", ""), "#888")
              for j in judge_names]

    fig, ax = plt.subplots(figsize=(max(8, len(judge_names) * 1.6), 5))
    bars = ax.bar(range(len(judge_names)), taus, color=colors, edgecolor="white")
    ax.axhline(y=1.0, color="green",  linestyle="--", alpha=0.4, label="τ=1 (identical ranking)")
    ax.axhline(y=0.0, color="orange", linestyle="--", alpha=0.4, label="τ=0 (no correlation)")
    ax.set_ylim(-0.1, 1.15)
    ax.set_xticks(range(len(judge_names)))
    ax.set_xticklabels(judge_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Kendall-τ vs Human ranking", fontsize=11)
    ax.set_title(
        f"Leaderboard Rank Correlation vs {reference_judge}\n"
        "How consistently does each judge rank retrieval systems vs humans?",
        fontsize=12,
    )

    for bar, tau in zip(bars, taus):
        if not np.isnan(tau):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    tau + 0.02, f"{tau:.3f}", ha="center", va="bottom", fontsize=9)

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f.capitalize())
                       for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][2:], fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_posrate_effect(
    matrix: Dict[str, Dict[str, float]],
    output_path: Path,
    ret_a: str = "Qwen3-embed",
    ret_b: str = "BGE-M3",
) -> None:
    """Scatter: judge positivity rate vs nDCG(ret_a)/nDCG(ret_b) ratio.

    Shows whether high-recall judges (generous pos_rate) distort the comparison.
    """
    xs, ys, labels, colors = [], [], [], []
    for jname, meta in JUDGE_REGISTRY.items():
        if meta.get("pos_rate") is None:
            continue
        if jname not in matrix:
            continue
        ndcg_a = matrix[jname].get(ret_a)
        ndcg_b = matrix[jname].get(ret_b)
        if ndcg_a is None or ndcg_b is None or ndcg_b == 0:
            continue
        xs.append(meta["pos_rate"])
        ys.append(ndcg_a / ndcg_b)
        labels.append(jname.replace("\n", " "))
        colors.append(FAMILY_COLORS.get(meta.get("family", ""), "#888"))

    if not xs:
        print("  [skip] not enough data for posrate_effect plot")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, s=120, zorder=3, edgecolors="white", linewidths=0.8)
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.4,
               label=f"ratio=1 (equal nDCG for {ret_a} and {ret_b})")

    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, alpha=0.9)

    ax.set_xlabel("Judge positivity rate (fraction labelled relevant)", fontsize=11)
    ax.set_ylabel(f"nDCG({ret_a}) / nDCG({ret_b})", fontsize=11)
    ax.set_title(
        f"Positivity Rate Effect on {ret_a} vs {ret_b} Comparison\n"
        "Ratio > 1 = judge scores Qwen3-embed higher; does leniency drive this?",
        fontsize=12,
    )

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=f.capitalize())
                       for f, c in FAMILY_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── judge_matrix mode runner ──────────────────────────────────────────────────

def run_judge_matrix(args) -> None:
    """Evaluate all retrievers against each LLM judge's qrels, build bias matrix."""
    cands_dir  = Path(args.candidates_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Mode: judge_matrix ===")

    # Load all candidate files
    print("Loading retriever candidates...")
    all_cands: Dict[str, Dict[str, List[dict]]] = {}
    for ret_name in JUDGE_MATRIX_RETRIEVERS:
        fname = CANDIDATES_FILES.get(ret_name)
        if fname is None:
            continue
        path = cands_dir / fname
        if not path.exists():
            print(f"  [skip] {ret_name}: {path} not found")
            continue
        all_cands[ret_name] = load_candidates_jsonl(path)
        print(f"  {ret_name}: {len(all_cands[ret_name])} queries")

    if not all_cands:
        print("No candidate files found. Exiting.")
        return

    # For each judge: compute nDCG@10 per retriever
    # matrix[judge_name][retriever_name] = nDCG@10
    matrix: Dict[str, Dict[str, float]] = {}
    rankings_per_judge: Dict[str, Dict[str, int]] = {}

    for judge_name, meta in JUDGE_REGISTRY.items():
        qrels_path = Path(meta["path"])
        if not qrels_path.exists():
            print(f"  [missing] {judge_name}: {qrels_path}")
            continue

        print(f"\n  Judge: {judge_name}")
        qrels = load_qrels_binarized(qrels_path, threshold=meta["threshold"])
        print(f"    {len(qrels)} queries, threshold={meta['threshold']}, "
              f"pos_rate={meta.get('pos_rate') or '?'}")

        judge_scores: Dict[str, float] = {}
        for ret_name, cands in all_cands.items():
            ndcg = compute_mean_ndcg(cands, qrels, k=10)
            judge_scores[ret_name] = ndcg
            print(f"    {ret_name:25s} nDCG@10 = {ndcg:.4f}")

        matrix[judge_name] = judge_scores
        rankings_per_judge[judge_name] = compute_system_ranking(judge_scores)

    if not matrix:
        print("No results computed. Check judge qrels paths.")
        return

    # Save raw matrix JSON
    (output_dir / "judge_matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False)
    )

    # Save markdown summary table
    ret_names   = JUDGE_MATRIX_RETRIEVERS
    judge_names = list(matrix.keys())
    md_lines    = ["## Judge × Retriever nDCG@10 Matrix\n",
                   "*(each row: judge's binarized qrels as ground truth, score≥2 = relevant)*\n"]
    header = "| Judge | " + " | ".join(ret_names) + " | Qwen3−BGE-M3 Δ |"
    sep    = "|-------|" + "|".join(["-------"] * len(ret_names)) + "|----------------|"
    md_lines += [header, sep]

    for jname in judge_names:
        row_vals = [f"{matrix[jname].get(r, '—'):.4f}"
                    if isinstance(matrix[jname].get(r), float) else "—"
                    for r in ret_names]
        qwen_ndcg = matrix[jname].get("Qwen3-embed")
        bge_ndcg  = matrix[jname].get("BGE-M3")
        if qwen_ndcg is not None and bge_ndcg is not None:
            delta = qwen_ndcg - bge_ndcg
            sign  = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.4f}"
        else:
            delta_str = "—"
        kappa = meta.get("kappa")
        kappa_str = f" (κ={kappa:.3f})" if kappa and kappa < 1.0 else ""
        row_label = jname.replace("\n", " ") + kappa_str
        md_lines.append(f"| {row_label} | " + " | ".join(row_vals) + f" | {delta_str} |")

    (output_dir / "judge_matrix.md").write_text("\n".join(md_lines))
    print("\n" + "\n".join(md_lines))

    # Generate all plots
    print("\nGenerating plots...")
    plot_judge_ndcg_matrix(matrix, output_dir / "judge_ndcg_matrix.png")
    plot_judge_delta_chart(matrix, output_dir / "judge_delta_chart.png")
    plot_leaderboard_correlation(
        rankings_per_judge, reference_judge="Human",
        output_path=output_dir / "leaderboard_correlation.png",
    )
    plot_posrate_effect(matrix, output_dir / "posrate_effect.png")

    print(f"\nAll outputs saved to: {output_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="RQ3 Bias Analysis — umbrela-indo-ir")
    p.add_argument("--mode", required=True,
                   choices=["aggregate", "perquery", "overlap", "judge_matrix"],
                   help="aggregate | perquery | overlap | judge_matrix")
    p.add_argument("--results-dir",    default="results/final/")
    p.add_argument("--retrieval-csv",  default="results/retrieval_scores.csv")
    p.add_argument("--candidates-dir", default="candidates/")
    p.add_argument("--qrels",          default="data/miracl-id/qrels/human/test.txt")
    p.add_argument("--data-dir",       default="data/miracl-id/")
    p.add_argument("--systems",
                   default="BGE-M3,Qwen3-embed,Hybrid BM25+Qwen3",
                   help="Comma-separated retriever names (perquery mode)")
    p.add_argument("--reranker-model", default=None,
                   help="HF model ID or path for reranking (perquery mode, needs GPU)")
    p.add_argument("--output",         default="results/final/bias_analysis/")
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "aggregate":
        run_aggregate(args)
    elif args.mode == "perquery":
        run_perquery(args)
    elif args.mode == "overlap":
        run_overlap(args)
    elif args.mode == "judge_matrix":
        run_judge_matrix(args)


if __name__ == "__main__":
    main()
