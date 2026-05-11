"""
UMBRELA LLM Judge inference for MIRACL-ID.

Runs an LLM judge on MIRACL-ID query-passage pairs and writes TREC-format qrels.
Supports resume (append mode) and CUDA OOM recovery.

Args:
    --judge-model   HF model ID (e.g. Qwen/Qwen2.5-7B-Instruct)
    --split         train | val | test
    --n-queries     max number of unique queries to process (default: all)
    --output        output TREC qrels file path
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --prompt-mode   zeroshot_bing | zeroshot_basic (default: zeroshot_bing)
    --token         HuggingFace token for private models (optional)

Example:
    python qrel_generation/inference.py \\
        --judge-model Qwen/Qwen2.5-7B-Instruct \\
        --split train --n-queries 1000 \\
        --output results/qrels/qwen_train.txt
"""

import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="UMBRELA judge inference on MIRACL-ID.")
    parser.add_argument("--judge-model", required=True, help="HF model ID")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Max unique queries to process (default: all)")
    parser.add_argument("--output", required=True, help="Output TREC qrels file")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Path to processed MIRACL-ID directory")
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic"])
    parser.add_argument("--token", default=None, help="HF token for private models")
    return parser.parse_args()


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
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
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

    # Heavy imports here so --help works without GPU dependencies installed
    import torch
    import os
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from model_utils import get_model_baseline
    from relevance_scoring import grade_each_pq_pair

    # Load model
    if args.token:
        os.environ["HF_TOKEN"] = args.token
    print(f"Loading model: {args.judge_model}")
    model = get_model_baseline(args.judge_model, use_together=False)

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
                    mode=args.prompt_mode,
                )

            except torch.cuda.OutOfMemoryError:
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
                        mode=args.prompt_mode,
                    )
                except Exception as e2:
                    err_f.write(f"oom_skip\t{qid}\t{docid}\t{e2}\n")
                    score = 0

            except Exception as e:
                err_f.write(f"error\t{qid}\t{docid}\t{e}\n")
                score = 0

            out_f.write(f"{qid} 0 {docid} {score}\n")
            out_f.flush()

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
