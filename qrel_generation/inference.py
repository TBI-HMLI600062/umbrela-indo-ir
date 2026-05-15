"""
UMBRELA LLM Judge inference for MIRACL-ID.

Runs an LLM judge on MIRACL-ID query-passage pairs and writes TREC-format qrels.
Supports resume (append mode) and CUDA OOM recovery.

Usage (from registry):
    python qrel_generation/inference.py --judge chatgpt --split test
    python qrel_generation/inference.py --judge deepseek --split train
    python qrel_generation/inference.py --judge qwen     --split test --n-queries 100

    # List all available judges:
    python qrel_generation/inference.py --list-judges

Usage (custom / one-off model):
    python qrel_generation/inference.py \\
        --judge-model Qwen/Qwen2.5-7B-Instruct --provider hf \\
        --split test --output results/qrels/custom_test.txt

Output is auto-derived as results/qrels/<judge>_<split>.txt when using --judge.
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMBRELA judge inference on MIRACL-ID.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--list-judges", action="store_true",
                        help="Print all available judges and exit.")

    judge_group = parser.add_mutually_exclusive_group()
    judge_group.add_argument(
        "--judge", metavar="NAME",
        help="Judge name from registry (e.g. chatgpt, qwen, deepseek). "
             "Run --list-judges to see all options.",
    )
    judge_group.add_argument(
        "--judge-model", metavar="MODEL_ID",
        help="Custom model ID (HF or API). Requires --provider and --output.",
    )

    parser.add_argument("--provider", default="hf",
                        choices=["hf", "together", "openai", "deepseek"],
                        help="Provider for --judge-model (ignored with --judge).")
    parser.add_argument("--prompt-mode", default=None,
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic"],
                        help="Prompt mode override (default: zeroshot_bing).")
    parser.add_argument("--split", choices=["train", "val", "test"],
                        help="Dataset split (required unless --list-judges).")
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Max unique queries to process (default: all).")
    parser.add_argument("--output", default=None,
                        help="Output TREC qrels file. "
                             "Auto-set to results/qrels/<judge>_<split>.txt when using --judge.")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Path to processed MIRACL-ID directory.")
    parser.add_argument("--token", default=None,
                        help="HuggingFace token for private models.")
    return parser.parse_args()


def resolve_judge_config(args):
    """Return (model_id, provider, prompt_mode, output_path) from args."""
    from judges import JUDGES

    if args.judge:
        if args.judge not in JUDGES:
            available = ", ".join(JUDGES)
            raise SystemExit(
                f"Unknown judge '{args.judge}'. Available: {available}\n"
                "Run --list-judges for details."
            )
        cfg = JUDGES[args.judge]
        model_id = cfg["model"]
        provider = cfg["provider"]
        prompt_mode = args.prompt_mode or cfg.get("prompt_mode", "zeroshot_bing")
        output_path = Path(
            args.output or f"results/qrels/{args.judge}_{args.split}.txt"
        )
    else:
        if not args.judge_model:
            raise SystemExit("Provide either --judge <name> or --judge-model <id>.")
        if not args.output:
            raise SystemExit("--output is required when using --judge-model.")
        model_id = args.judge_model
        provider = args.provider
        prompt_mode = args.prompt_mode or "zeroshot_bing"
        output_path = Path(args.output)

    return model_id, provider, prompt_mode, output_path


def load_topics(data_dir: Path, split: str) -> dict:
    path = data_dir / "topics" / f"{split}.tsv"
    topics = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def load_candidates(data_dir: Path, split: str) -> list:
    """Returns list of (qid, docid) pairs from positive + negative passages."""
    path = data_dir / "qrels" / "candidates" / f"{split}.jsonl"
    pairs = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            qid = obj["qid"]
            for docid in obj.get("positive_docids", []):
                pairs.append((qid, docid))
            for docid in obj.get("negative_docids", []):
                pairs.append((qid, docid))
    return pairs


def filter_by_n_queries(pairs: list, n_queries: int) -> list:
    """Keep all pairs for the first n_queries unique query IDs."""
    seen = {}
    result = []
    for qid, docid in pairs:
        if qid not in seen:
            if len(seen) >= n_queries:
                continue
            seen[qid] = True
        result.append((qid, docid))
    return result


def load_processed(output_path: Path) -> set:
    """Read already-written (qid, docid) pairs for resume."""
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    processed.add((parts[0], parts[2]))
    return processed


def load_corpus_subset(data_dir: Path, needed_docids: set) -> dict:
    """Load only the corpus entries referenced by needed_docids."""
    corpus_path = data_dir / "corpus" / "corpus.jsonl"
    corpus = {}
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus", unit=" passages"):
            obj = json.loads(line)
            if obj["docid"] in needed_docids:
                corpus[obj["docid"]] = obj["doc"]
            if len(corpus) == len(needed_docids):
                break
    missing = needed_docids - corpus.keys()
    if missing:
        print(f"Warning: {len(missing)} docids not found in corpus")
    return corpus


def main():
    args = parse_args()

    if args.list_judges:
        sys.path.insert(0, str(Path(__file__).parent))
        from judges import list_judges
        list_judges()
        return

    if not args.split:
        raise SystemExit("--split is required (train | val | test).")

    sys.path.insert(0, str(Path(__file__).parent))
    model_id, provider, prompt_mode, output_path = resolve_judge_config(args)

    data_dir = Path(args.data_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load topics and candidate pairs
    topics = load_topics(data_dir, args.split)
    candidates = load_candidates(data_dir, args.split)

    if args.n_queries is not None:
        candidates = filter_by_n_queries(candidates, args.n_queries)

    # Resume: skip already processed pairs
    processed = load_processed(output_path)
    if processed:
        print(f"Resuming: skipping {len(processed)} already processed pairs")
    remaining = [(q, d) for q, d in candidates if (q, d) not in processed]
    print(f"Pairs to process: {len(remaining)} "
          f"({len(candidates) - len(remaining)} skipped)")

    if not remaining:
        print("All pairs already processed. Done.")
        return

    # Load only needed corpus passages
    needed_docids = {docid for _, docid in remaining}
    corpus = load_corpus_subset(data_dir, needed_docids)

    # Heavy imports — defer torch so --help and --list-judges work without GPU
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from model_utils import get_model_baseline, APIBasePipeline
    from relevance_scoring import grade_each_pq_pair

    is_api_model = provider in ("together", "openai", "deepseek")
    if not is_api_model:
        import torch

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    print(f"Loading model: {model_id}  (provider={provider}, prompt={prompt_mode})")
    model = get_model_baseline(model_id, provider=provider)

    # Setup log/error paths
    logs_path = output_path.parent / "logs" / output_path.name.replace(".txt", ".jsonl")
    errors_path = output_path.parent / "cuda_errors" / output_path.name
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as out_f, open(errors_path, "a") as err_f:
        for qid, docid in tqdm(remaining, desc="Judging"):
            if qid not in topics:
                err_f.write(f"missing_query\t{qid}\n")
                continue
            if docid not in corpus:
                err_f.write(f"missing_doc\t{qid}\t{docid}\n")
                continue

            try:
                score, _ = grade_each_pq_pair(
                    query=topics[qid],
                    passage=corpus[docid],
                    pipeline=model,
                    log_file_path=str(logs_path),
                    system_message="",
                    qidx=qid,
                    docidx=docid,
                    mode=prompt_mode,
                )

            except Exception as e:
                oom = (not is_api_model
                       and isinstance(e, torch.cuda.OutOfMemoryError))
                if oom:
                    torch.cuda.empty_cache()
                    print(f"\nCUDA OOM on {qid}/{docid}, retrying after cache clear...")
                    try:
                        score, _ = grade_each_pq_pair(
                            query=topics[qid],
                            passage=corpus[docid],
                            pipeline=model,
                            log_file_path=str(logs_path),
                            system_message="",
                            qidx=qid,
                            docidx=docid,
                            mode=prompt_mode,
                        )
                    except Exception as e2:
                        err_f.write(f"oom_skip\t{qid}\t{docid}\t{e2}\n")
                        score = 0
                else:
                    err_f.write(f"error\t{qid}\t{docid}\t{e}\n")
                    score = 0

            out_f.write(f"{qid} 0 {docid} {score}\n")
            out_f.flush()

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
