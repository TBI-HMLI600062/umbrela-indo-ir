"""
Extended analysis for UMBRELA-Indo-IR.

Computes four analyses from existing data (CPU-only, no models needed):

  --mode inter_judge    Pairwise Cohen's κ + agreement % matrix (11 judges)
  --mode label_dist     Label distribution in top-K rank bins per retriever × judge
  --mode full_matrix    nDCG@10 / MAP@10 for all retriever × judge combinations
  --mode error_analysis Hard queries, judge disagreement, reranker failures
  --mode all            Run all four modes

Output: results/final/extended/

Example:
    cd umbrela-indo-ir
    python evaluation/extended_analysis.py --mode all
    python evaluation/extended_analysis.py --mode inter_judge
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths & judge registry
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent  # umbrela-indo-ir/
OUT_DIR  = BASE_DIR / "results" / "final" / "extended"

JUDGE_REGISTRY = {
    "Human":              BASE_DIR / "data/miracl-id/qrels/human/test.txt",
    "Qwen2.5-7B":         BASE_DIR / "data/miracl-id/results/qrels/qwen_test.txt",
    "DeepSeek-V3":        BASE_DIR / "results/qrels/deepseek_test.txt",
    "ChatGPT":            BASE_DIR / "results/qrels/chatgpt_test.txt",
    "Llama3-default":     BASE_DIR / "results/qrels/sahabat_llama_test.txt",
    "Llama3-strict":      BASE_DIR / "results/qrels_strict/sahabat_llama_strict_test.txt",
    "Llama3-vllm-fs-basic":  BASE_DIR / "results/qrels/sahabat-llama_vllm_fewshot_basic_test.txt",
    "Llama3-vllm-fs-bing":   BASE_DIR / "results/qrels/sahabat-llama_vllm_fewshot_bing_test.txt",
    "Llama3-vllm-zs-basic":  BASE_DIR / "results/qrels/sahabat-llama_vllm_zeroshot_basic_test.txt",
    "Gemma2-vllm-zs-bing":   BASE_DIR / "results/qrels/sahabat-gemma_vllm_zeroshot_bing_test.txt",
    "Gemma2-vllm-zs-bing-strict": BASE_DIR / "results/qrels/sahabat-gemma_vllm_zeroshot_bing_strict_test.txt",
}

CANDIDATE_REGISTRY = {
    "BM25":          BASE_DIR / "candidates/bm25_test_top100.jsonl",
    "BGE-M3":        BASE_DIR / "candidates/bgem3_test_top100.jsonl",
    "Qwen-embed":    BASE_DIR / "candidates/qwen_test_top100.jsonl",
    "Qwen3-embed":   BASE_DIR / "candidates/qwen3_test_top100.jsonl",
    "Hybrid-BGE":    BASE_DIR / "candidates/hybrid_test_top100.jsonl",
    "Hybrid-Qwen3":  BASE_DIR / "candidates/hybrid_bm25_qwen3_test_top100.jsonl",
}

RELEVANCE_THRESHOLD = 2  # score >= threshold → relevant (LLM judges, 0-3 scale)

# Per-judge threshold (Human uses binary 0/1, only positives listed)
JUDGE_THRESHOLD: dict = {"Human": 1}  # default for others = RELEVANCE_THRESHOLD

# Judges that use positive-only format (no negatives in qrels file)
POSITIVE_ONLY_JUDGES: set = {"Human"}


# ---------------------------------------------------------------------------
# Shared utilities (mirrors evaluation/metrics.py & eval_pipeline.py)
# ---------------------------------------------------------------------------

def parse_qrels(path: Path) -> dict:
    """TREC qrels → {qid: {docid: int_score}}."""
    qrels: dict = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, score = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def binarize(qrels: dict, threshold: int = RELEVANCE_THRESHOLD) -> dict:
    """→ {qid: {docid: 0|1}}."""
    return {qid: {d: int(s >= threshold) for d, s in docs.items()}
            for qid, docs in qrels.items()}


def get_threshold(judge_name: str) -> int:
    return JUDGE_THRESHOLD.get(judge_name, RELEVANCE_THRESHOLD)


def load_candidates(path: Path) -> dict:
    """JSONL candidates → {qid: [(docid, score), ...]} sorted by score desc."""
    result = {}
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            result[obj["qid"]] = [(c["docid"], c["score"]) for c in obj["candidates"]]
    return result


def load_available_judges() -> dict:
    """Return {name: path} for judges whose files exist."""
    return {name: path for name, path in JUDGE_REGISTRY.items() if path.exists()}


def load_available_candidates() -> dict:
    return {name: path for name, path in CANDIDATE_REGISTRY.items() if path.exists()}


# ---------------------------------------------------------------------------
# Mode 1: inter_judge — pairwise Cohen's κ + agreement %
# ---------------------------------------------------------------------------

def run_inter_judge(out_dir: Path) -> None:
    print("\n=== Mode: inter_judge ===")
    from sklearn.metrics import cohen_kappa_score

    judges = load_available_judges()
    print(f"  Loaded {len(judges)} judges: {list(judges.keys())}")

    # Parse raw qrels (not binarized yet, we binarize per judge threshold)
    raw_qrels: dict = {}
    for name, path in judges.items():
        raw_qrels[name] = parse_qrels(path)

    judge_names = sorted(judges.keys())
    rows = []

    for i, a in enumerate(judge_names):
        for b in judge_names[i+1:]:
            ra, rb = raw_qrels[a], raw_qrels[b]
            thr_a = get_threshold(a)
            thr_b = get_threshold(b)
            a_pos_only = a in POSITIVE_ONLY_JUDGES
            b_pos_only = b in POSITIVE_ONLY_JUDGES

            # Universe: use the non-positive-only judge's pairs when one is pos-only
            # If both pos-only or both full: use intersection
            if a_pos_only and not b_pos_only:
                # Universe = b's pairs; human label = 1 if in ra else 0
                universe = [(qid, docid) for qid in rb for docid in rb[qid]]
            elif b_pos_only and not a_pos_only:
                universe = [(qid, docid) for qid in ra for docid in ra[qid]]
            else:
                # Both full or both pos-only: intersection
                universe = [(qid, docid) for qid in ra if qid in rb
                            for docid in ra[qid] if docid in rb.get(qid, {})]

            labels_a, labels_b = [], []
            for qid, docid in universe:
                score_a = ra.get(qid, {}).get(docid, 0)
                score_b = rb.get(qid, {}).get(docid, 0)
                labels_a.append(int(score_a >= thr_a))
                labels_b.append(int(score_b >= thr_b))

            if not labels_a:
                continue
            n = len(labels_a)

            if n < 2:
                continue

            # agree %
            agree_n = sum(la == lb for la, lb in zip(labels_a, labels_b))
            both_pos = sum(la == 1 and lb == 1 for la, lb in zip(labels_a, labels_b))
            both_neg = sum(la == 0 and lb == 0 for la, lb in zip(labels_a, labels_b))
            a_pos_b_neg = sum(la == 1 and lb == 0 for la, lb in zip(labels_a, labels_b))
            a_neg_b_pos = sum(la == 0 and lb == 1 for la, lb in zip(labels_a, labels_b))

            agree_pct = round(agree_n / n, 4)
            both_pos_pct = round(both_pos / n, 4)
            both_neg_pct = round(both_neg / n, 4)
            disagree_ab_pct = round(a_pos_b_neg / n, 4)
            disagree_ba_pct = round(a_neg_b_pos / n, 4)

            # Cohen's κ
            if len(set(labels_a)) < 2 or len(set(labels_b)) < 2:
                kappa = float("nan")
            else:
                kappa = round(float(cohen_kappa_score(labels_a, labels_b)), 4)

            posrate_a = round(sum(labels_a) / n, 4)
            posrate_b = round(sum(labels_b) / n, 4)

            rows.append({
                "judge_a": a, "judge_b": b,
                "kappa": kappa, "agree_pct": agree_pct,
                "both_pos_pct": both_pos_pct, "both_neg_pct": both_neg_pct,
                "disagree_ab_pct": disagree_ab_pct, "disagree_ba_pct": disagree_ba_pct,
                "posrate_a": posrate_a, "posrate_b": posrate_b,
                "n_common_pairs": n,
            })
            print(f"  {a} vs {b}: κ={kappa:.4f}  agree={agree_pct:.1%}  n={n:,}")

    # Save CSV
    csv_path = out_dir / "inter_judge_kappa.csv"
    fields = ["judge_a","judge_b","kappa","agree_pct","both_pos_pct","both_neg_pct",
              "disagree_ab_pct","disagree_ba_pct","posrate_a","posrate_b","n_common_pairs"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {csv_path} ({len(rows)} pairs)")

    # posrate CSV
    pr_rows = []
    for name, rq in raw_qrels.items():
        thr = get_threshold(name)
        labels = [int(v >= thr) for docs in rq.values() for v in docs.values()]
        pr_rows.append({"judge": name, "posrate": round(sum(labels)/len(labels), 4) if labels else 0,
                        "n_pairs": len(labels)})
    pr_path = out_dir / "inter_judge_posrate.csv"
    with open(pr_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["judge","posrate","n_pairs"])
        w.writeheader(); w.writerows(pr_rows)
    print(f"  → {pr_path}")

    # Heatmaps
    _plot_inter_judge_heatmaps(rows, judge_names, out_dir)


def _plot_inter_judge_heatmaps(rows: list, judge_names: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip heatmap — matplotlib not available]")
        return

    # Build symmetric matrices
    n = len(judge_names)
    idx = {name: i for i, name in enumerate(judge_names)}
    kappa_mat  = np.full((n, n), float("nan"))
    agree_mat  = np.full((n, n), float("nan"))
    for r in rows:
        i, j = idx[r["judge_a"]], idx[r["judge_b"]]
        kappa_mat[i, j] = kappa_mat[j, i] = float(r["kappa"]) if r["kappa"] == r["kappa"] else float("nan")
        agree_mat[i, j] = agree_mat[j, i] = float(r["agree_pct"])
    np.fill_diagonal(kappa_mat, 1.0)
    np.fill_diagonal(agree_mat, 1.0)

    short = [n.replace("Llama3-vllm-","L3-").replace("Gemma2-vllm-","G2-")
               .replace("Qwen2.5-7B","Qwen2.5") for n in judge_names]

    for mat, title, fname, vmin, vmax, fmt in [
        (kappa_mat, "Inter-Judge Cohen's κ", "inter_judge_kappa_heatmap.png", -0.1, 1.0, ".3f"),
        (agree_mat * 100, "Inter-Judge Agreement %", "inter_judge_agree_heatmap.png", 40, 100, ".1f"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 9))
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, vmin=vmin, vmax=vmax, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                if val == val:  # not nan
                    ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=7,
                            color="black" if 0.3 < (val - vmin)/(vmax - vmin + 1e-9) < 0.8 else "white")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{title}\n(MIRACL-ID test, threshold≥2)", fontsize=11)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print(f"  → {out_dir / fname}")


# ---------------------------------------------------------------------------
# Mode 2: label_dist — precision per rank bin
# ---------------------------------------------------------------------------

RANK_BINS = [(1, 5), (6, 10), (11, 25), (26, 50), (51, 100)]


def run_label_dist(out_dir: Path) -> None:
    print("\n=== Mode: label_dist ===")
    judges = load_available_judges()
    candidates_map = load_available_candidates()

    rows = []
    for ret_name, cands_path in candidates_map.items():
        cands = load_candidates(cands_path)  # {qid: [(docid, score), ...]}
        for judge_name, qrels_path in judges.items():
            thr = get_threshold(judge_name)
            raw_q = parse_qrels(qrels_path)
            pos_only = judge_name in POSITIVE_ONLY_JUDGES
            qrels = binarize(raw_q, threshold=thr)
            for lo, hi in RANK_BINS:
                precisions = []
                n_judged_total = 0
                for qid, ranked_docs in cands.items():
                    bin_docs = [docid for docid, _ in ranked_docs[lo-1:hi]]
                    if pos_only:
                        # Precision = fraction of bin docs present in positive-only qrels
                        if not bin_docs:
                            continue
                        n_relevant = sum(1 for d in bin_docs
                                         if qrels.get(qid, {}).get(d, 0) == 1)
                        prec = n_relevant / len(bin_docs)
                        precisions.append(prec)
                        n_judged_total += len(bin_docs)
                    else:
                        if qid not in qrels:
                            continue
                        judged = [(docid, qrels[qid][docid]) for docid in bin_docs
                                  if docid in qrels[qid]]
                        if not judged:
                            continue
                        prec = sum(rel for _, rel in judged) / len(judged)
                        precisions.append(prec)
                        n_judged_total += len(judged)
                if precisions:
                    rows.append({
                        "retriever": ret_name,
                        "judge": judge_name,
                        "bin": f"{lo}-{hi}",
                        "mean_precision": round(sum(precisions)/len(precisions), 4),
                        "n_queries": len(precisions),
                        "n_docs_judged": n_judged_total,
                    })
        print(f"  {ret_name} ✓")

    csv_path = out_dir / "label_dist.csv"
    fields = ["retriever","judge","bin","mean_precision","n_queries","n_docs_judged"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {csv_path} ({len(rows)} rows)")

    _plot_label_dist(rows, out_dir)


def _plot_label_dist(rows: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip chart — matplotlib not available]"); return

    from collections import defaultdict
    ret_names = list(dict.fromkeys(r["retriever"] for r in rows))
    judge_names = list(dict.fromkeys(r["judge"] for r in rows))
    bins = [f"{lo}-{hi}" for lo, hi in RANK_BINS]

    # One subplot per retriever; lines = judges; x = rank bin
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes_flat = axes.flatten()
    colors = plt.cm.tab10.colors

    for ax_idx, ret in enumerate(ret_names[:6]):
        ax = axes_flat[ax_idx]
        for j_idx, judge in enumerate(judge_names):
            yvals = []
            for b in bins:
                match = [r["mean_precision"] for r in rows
                         if r["retriever"] == ret and r["judge"] == judge and r["bin"] == b]
                yvals.append(match[0] if match else float("nan"))
            ax.plot(bins, yvals, marker="o", label=judge,
                    color=colors[j_idx % len(colors)], linewidth=1.5, markersize=4)
        ax.set_title(ret, fontsize=9)
        ax.set_xlabel("Rank bin", fontsize=8)
        ax.set_ylabel("Mean precision", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for ax in axes_flat[len(ret_names):]:
        ax.set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=2, fontsize=7)
    fig.suptitle("Mean Precision per Rank Bin (retriever × judge)\nMIRACL-ID test, threshold≥2",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "label_dist.png", dpi=150)
    plt.close()
    print(f"  → {out_dir / 'label_dist.png'}")


# ---------------------------------------------------------------------------
# Mode 3: full_matrix — nDCG@10 for all retriever × judge combos
# ---------------------------------------------------------------------------

def run_full_matrix(out_dir: Path) -> None:
    print("\n=== Mode: full_matrix ===")
    try:
        from ranx import Qrels, Run, evaluate
    except ImportError:
        print("  [ERROR] ranx not installed — skipping full_matrix"); return

    judges = load_available_judges()
    candidates_map = load_available_candidates()

    rows = []
    for ret_name, cands_path in candidates_map.items():
        cands = load_candidates(cands_path)
        run_data = {qid: {docid: score for docid, score in docs}
                    for qid, docs in cands.items()}
        run_obj = Run(run_data, name=ret_name)
        for judge_name, qrels_path in judges.items():
            raw_qrels = parse_qrels(qrels_path)
            # ranx needs int relevance
            q_obj = Qrels({qid: {d: int(s) for d, s in docs.items()}
                           for qid, docs in raw_qrels.items()})
            metrics = evaluate(q_obj, run_obj, ["ndcg@10", "map@10", "recall@100"],
                               make_comparable=True)
            n_judged = sum(1 for qid in run_data if qid in raw_qrels)
            rows.append({
                "retriever": ret_name, "judge": judge_name,
                "ndcg@10":    round(float(metrics["ndcg@10"]), 4),
                "map@10":     round(float(metrics["map@10"]), 4),
                "recall@100": round(float(metrics["recall@100"]), 4),
                "n_judged_queries": n_judged,
            })
            print(f"  {ret_name} × {judge_name}: nDCG@10={metrics['ndcg@10']:.4f}")

    csv_path = out_dir / "full_matrix.csv"
    fields = ["retriever","judge","ndcg@10","map@10","recall@100","n_judged_queries"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"  → {csv_path} ({len(rows)} rows)")

    _plot_full_matrix(rows, out_dir)


def _plot_full_matrix(rows: list, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [skip heatmap — matplotlib not available]"); return

    ret_names = list(dict.fromkeys(r["retriever"] for r in rows))
    judge_names = list(dict.fromkeys(r["judge"] for r in rows))
    n_ret, n_judge = len(ret_names), len(judge_names)
    ret_idx = {r: i for i, r in enumerate(ret_names)}
    judge_idx = {j: i for i, j in enumerate(judge_names)}

    for metric, fname, title in [
        ("ndcg@10", "full_matrix_ndcg_heatmap.png", "nDCG@10"),
        ("map@10",  "full_matrix_map_heatmap.png",  "MAP@10"),
    ]:
        mat = np.zeros((n_ret, n_judge))
        for r in rows:
            mat[ret_idx[r["retriever"]], judge_idx[r["judge"]]] = r[metric]

        fig, ax = plt.subplots(figsize=(max(10, n_judge * 1.1), max(4, n_ret * 0.9)))
        im = ax.imshow(mat, vmin=0, vmax=mat.max()*1.05, cmap="YlOrRd", aspect="auto")
        short_judges = [j.replace("Llama3-vllm-","L3-").replace("Gemma2-vllm-","G2-") for j in judge_names]
        ax.set_xticks(range(n_judge)); ax.set_xticklabels(short_judges, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_ret)); ax.set_yticklabels(ret_names, fontsize=8)
        for i in range(n_ret):
            for j in range(n_judge):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Retriever × Judge {title}\nMIRACL-ID test (threshold≥2 for binary)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
        print(f"  → {out_dir / fname}")


# ---------------------------------------------------------------------------
# Per-query nDCG helper (manual, no ranx per_query flag needed)
# ---------------------------------------------------------------------------

def _dcg(rels: list, k: int) -> float:
    import math
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def _compute_perquery_ndcg(run_data: dict, qrels: dict, k: int = 10) -> dict:
    """Return {qid: ndcg@k} using human qrels. Queries with no human qrels → 0."""
    result = {}
    for qid, docs in run_data.items():
        if qid not in qrels:
            result[qid] = 0.0
            continue
        q_rels = qrels[qid]
        # rank by score descending
        ranked = sorted(docs.items(), key=lambda x: x[1], reverse=True)[:k]
        rels_at_k = [q_rels.get(docid, 0) for docid, _ in ranked]
        dcg_val = _dcg(rels_at_k, k)
        # ideal DCG
        ideal = sorted(q_rels.values(), reverse=True)[:k]
        idcg = _dcg(ideal, k)
        result[qid] = float(dcg_val / idcg) if idcg > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Mode 4: error_analysis
# ---------------------------------------------------------------------------

def run_error_analysis(out_dir: Path) -> None:
    print("\n=== Mode: error_analysis ===")
    report_lines = ["# Error Analysis — UMBRELA-Indo-IR\n"]

    # A. Hard queries
    perquery_path = BASE_DIR / "results/final/bias_analysis/perquery_scores.json"
    if perquery_path.exists():
        with open(perquery_path) as f:
            pq = json.load(f)
        # pq: {system: {qid: {k: ndcg}}}
        # collect nDCG@10 per query for each system
        ndcg_by_qid: dict = defaultdict(dict)
        for sys_name, qid_dict in pq.items():
            for qid, k_dict in qid_dict.items():
                ndcg_by_qid[qid][sys_name] = float(k_dict.get("10", 0))

        hard_rows = []
        for qid, sys_scores in sorted(ndcg_by_qid.items(), key=lambda x: max(x[1].values())):
            max_ndcg = max(sys_scores.values())
            category = "hard" if max_ndcg < 0.1 else ("medium" if max_ndcg < 0.3 else "easy")
            hard_rows.append({"qid": qid, "category": category, "max_ndcg": round(max_ndcg, 4),
                               **{k: round(v, 4) for k, v in sys_scores.items()}})

        hard_path = out_dir / "hard_queries.csv"
        sys_names = list(pq.keys())
        fields = ["qid","category","max_ndcg"] + sys_names
        with open(hard_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader(); w.writerows(hard_rows)

        n_hard   = sum(1 for r in hard_rows if r["category"] == "hard")
        n_medium = sum(1 for r in hard_rows if r["category"] == "medium")
        n_easy   = sum(1 for r in hard_rows if r["category"] == "easy")
        print(f"  Hard queries: hard={n_hard}, medium={n_medium}, easy={n_easy}")
        print(f"  → {hard_path}")

        report_lines += [
            "## A. Query Difficulty Distribution\n",
            f"| Category | Criterion | Count | % |\n|----------|-----------|-------|---|\n",
            f"| Hard | max nDCG@10 < 0.1 | {n_hard} | {n_hard/len(hard_rows):.1%} |\n",
            f"| Medium | 0.1 ≤ max nDCG@10 < 0.3 | {n_medium} | {n_medium/len(hard_rows):.1%} |\n",
            f"| Easy | max nDCG@10 ≥ 0.3 | {n_easy} | {n_easy/len(hard_rows):.1%} |\n\n",
            f"**Top-10 hardest queries** (by max nDCG@10 across all systems):\n\n",
            "| qid | max_nDCG@10 | " + " | ".join(sys_names) + " |\n",
            "|-----|------------|" + "|".join(["---"]*len(sys_names)) + "|\n",
        ]
        for r in sorted(hard_rows, key=lambda x: x["max_ndcg"])[:10]:
            vals = " | ".join(str(r.get(s, "?")) for s in sys_names)
            report_lines.append(f"| {r['qid']} | {r['max_ndcg']} | {vals} |\n")
        report_lines.append("\n")
    else:
        print(f"  [skip hard queries — {perquery_path} not found]")

    # B. Judge disagreement per query
    judges = load_available_judges()
    all_bin_qrels = {name: binarize(parse_qrels(path)) for name, path in judges.items()}

    # Build {qid: {docid: {judge: label}}}
    pair_labels: dict = defaultdict(lambda: defaultdict(dict))
    for judge_name, qrels in all_bin_qrels.items():
        for qid, docs in qrels.items():
            for docid, label in docs.items():
                pair_labels[qid][docid][judge_name] = label

    disagree_rows = []
    for qid, doc_dict in pair_labels.items():
        if not doc_dict:
            continue
        doc_disagree = []
        for docid, judge_labels in doc_dict.items():
            if len(judge_labels) < 2:
                continue
            vals = list(judge_labels.values())
            mean_val = sum(vals) / len(vals)
            # variance-based disagreement: std of binary labels
            variance = sum((v - mean_val)**2 for v in vals) / len(vals)
            doc_disagree.append(variance)
        if not doc_disagree:
            continue
        mean_disagree = sum(doc_disagree) / len(doc_disagree)
        n_docs = len(doc_disagree)
        frac_any_disagree = sum(1 for v in doc_disagree if v > 0) / n_docs
        disagree_rows.append({
            "qid": qid,
            "mean_disagreement": round(mean_disagree, 4),
            "n_docs_judged": n_docs,
            "frac_docs_disagreed": round(frac_any_disagree, 4),
        })

    disagree_rows.sort(key=lambda x: -x["mean_disagreement"])
    dis_path = out_dir / "judge_disagreement.csv"
    with open(dis_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid","mean_disagreement","n_docs_judged","frac_docs_disagreed"])
        w.writeheader(); w.writerows(disagree_rows)
    print(f"  → {dis_path} ({len(disagree_rows)} queries)")

    top10_dis = disagree_rows[:10]
    report_lines += [
        "## B. Judge Disagreement per Query\n",
        f"Computed over {len(disagree_rows):,} queries where ≥2 judges evaluated the same doc.\n\n",
        "**Top-10 most disagreed queries:**\n\n",
        "| qid | mean_disagreement | n_docs | frac_docs_disagreed |\n",
        "|-----|-------------------|--------|---------------------|\n",
    ]
    for r in top10_dis:
        report_lines.append(f"| {r['qid']} | {r['mean_disagreement']} | {r['n_docs_judged']} | {r['frac_docs_disagreed']} |\n")
    report_lines.append("\n")

    # C. Reranker failure per query
    reranked_path = BASE_DIR / "results/final/qwen3_bge_reranked.txt"
    human_qrels_path = BASE_DIR / "data/miracl-id/qrels/human/test.txt"
    if reranked_path.exists() and human_qrels_path.exists() and perquery_path.exists():
        try:
            from ranx import Qrels, Run, evaluate

            # Load reranked run (TREC format)
            run_data: dict = {}
            with open(reranked_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    qid, _, docid, rank, score, _ = parts[:6]
                    run_data.setdefault(qid, {})[docid] = float(score)

            human_qrels = parse_qrels(human_qrels_path)

            # Compute per-query nDCG@10 manually (ranx per_query not supported)
            rk_ndcg = _compute_perquery_ndcg(run_data, human_qrels, k=10)

            # Baseline: Qwen3 from perquery_scores
            with open(perquery_path) as f:
                pq = json.load(f)
            qwen3_ndcg = {qid: float(vals.get("10", 0))
                          for qid, vals in pq.get("Qwen3-embed", {}).items()}

            failure_rows = []
            for qid in rk_ndcg:
                if qid not in qwen3_ndcg:
                    continue
                baseline = qwen3_ndcg[qid]
                reranked = rk_ndcg[qid]
                delta = reranked - baseline
                failure_rows.append({
                    "qid": qid, "ndcg_baseline": round(baseline, 4),
                    "ndcg_reranked": round(reranked, 4), "delta": round(delta, 4),
                })
            failure_rows.sort(key=lambda x: x["delta"])
            fail_path = out_dir / "reranker_failure.csv"
            with open(fail_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["qid","ndcg_baseline","ndcg_reranked","delta"])
                w.writeheader(); w.writerows(failure_rows)

            n_fail = sum(1 for r in failure_rows if r["delta"] < -0.2)
            n_neutral = sum(1 for r in failure_rows if -0.05 <= r["delta"] <= 0.05)
            n_gain = sum(1 for r in failure_rows if r["delta"] > 0.2)
            print(f"  Reranker failures (delta<-0.2): {n_fail} | neutral: {n_neutral} | big gains: {n_gain}")
            print(f"  → {fail_path}")

            report_lines += [
                "## C. Reranker Failure Analysis\n",
                f"System: Qwen3-embed + BGE-hardneg-rk vs Qwen3-embed baseline.\n",
                f"Evaluated on {len(failure_rows):,} queries with human qrels.\n\n",
                "| Category | Criterion | Count | % |\n|----------|-----------|-------|---|\n",
                f"| Big failure | δ < -0.2 | {n_fail} | {n_fail/max(1,len(failure_rows)):.1%} |\n",
                f"| Neutral | -0.05 ≤ δ ≤ 0.05 | {n_neutral} | {n_neutral/max(1,len(failure_rows)):.1%} |\n",
                f"| Big gain | δ > 0.2 | {n_gain} | {n_gain/max(1,len(failure_rows)):.1%} |\n\n",
                "**Top-10 worst reranker failures:**\n\n",
                "| qid | ndcg_baseline | ndcg_reranked | delta |\n|-----|---------------|---------------|-------|\n",
            ]
            for r in failure_rows[:10]:
                report_lines.append(f"| {r['qid']} | {r['ndcg_baseline']} | {r['ndcg_reranked']} | {r['delta']} |\n")
            report_lines.append("\n")
        except Exception as e:
            print(f"  [reranker failure] error: {e}")
    else:
        print("  [skip reranker failure — missing files]")

    # Write report
    md_path = out_dir / "error_analysis.md"
    with open(md_path, "w") as f:
        f.writelines(report_lines)
    print(f"  → {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extended analysis for UMBRELA-Indo-IR.")
    p.add_argument("--mode", required=True,
                   choices=["inter_judge","label_dist","full_matrix","error_analysis","all"])
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = (["inter_judge","label_dist","full_matrix","error_analysis"]
             if args.mode == "all" else [args.mode])

    dispatch = {
        "inter_judge":    run_inter_judge,
        "label_dist":     run_label_dist,
        "full_matrix":    run_full_matrix,
        "error_analysis": run_error_analysis,
    }
    for mode in modes:
        dispatch[mode](OUT_DIR)

    print(f"\n✓ Done. Output in {OUT_DIR}/")


if __name__ == "__main__":
    main()
