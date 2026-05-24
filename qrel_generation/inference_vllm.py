"""
UMBRELA LLM Judge inference with vLLM on MIRACL-ID.
Uses vLLM continuous batching + max_tokens=1 (prefix forcing) for fast scoring.

Args:
    --judge-model   HF model ID (e.g. GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct)
    --split         train | val | test
    --n-queries     max number of unique queries to process (default: all)
    --output        output TREC qrels file path
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --prompt-mode   zeroshot_bing | zeroshot_basic | fewshot_bing | fewshot_basic
    --token         HuggingFace token for private models (optional)
    --batch-size    chunk size for resume flushing (default: 500, vLLM batches internally)
    --max-length    max input token length (default: 2048)

Example:
    python qrel_generation/inference_vllm.py \\
        --judge-model GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct \\
        --split test --output results/qrels/sahabat-gemma_test.txt
"""

import sys
import re
import json
import time
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMBRELA judge inference (vLLM) on MIRACL-ID."
    )
    parser.add_argument("--judge-model", required=True, help="HF model ID")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Max unique queries to process (default: all)")
    parser.add_argument("--output", required=True, help="Output TREC qrels file")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Path to processed MIRACL-ID directory")
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic",
                                 "zeroshot_bing_strict"])
    parser.add_argument("--token", default=None, help="HF token for private models")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Chunk size for resume flushing (vLLM batches internally)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max input token length (default: 2048)")
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
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    processed.add((parts[0], parts[2]))
    return processed


def load_corpus_subset(data_dir: Path, needed_docids: set) -> dict:
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


def extract_score(text: str) -> int:
    match = re.search(r'##\s*final score:\s*([0-3])', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'O:\s*([0-3])', text)
    if match:
        return int(match.group(1))
    match = re.search(r'\b[0-3]\b', text)
    if match:
        return int(match.group())
    return 0


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import os
    if args.token:
        os.environ["HF_TOKEN"] = args.token

    topics = load_topics(data_dir, args.split)
    candidates = load_candidates(data_dir, args.split)

    if args.n_queries is not None:
        candidates = filter_by_n_queries(candidates, args.n_queries)

    processed = load_processed(output_path)
    if processed:
        print(f"Resuming: skipping {len(processed)} already processed pairs")
    remaining = [(q, d) for q, d in candidates if (q, d) not in processed]
    print(f"Pairs to process: {len(remaining)} "
          f"({len(candidates) - len(remaining)} skipped)")

    if not remaining:
        print("All pairs already processed. Done.")
        return

    needed_docids = {docid for _, docid in remaining}
    corpus = load_corpus_subset(data_dir, needed_docids)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from prompts import get_umbrella_prompt

    print(f"Loading model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)

    t_load = time.time()
    llm = LLM(
        model=args.judge_model,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_model_len=args.max_length,
    )
    # max_tokens=1: prefix "##final score: " already added, model outputs just the digit
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
    print(f"vLLM engine ready. (load+compile: {time.time()-t_load:.1f}s)")

    logs_path = output_path.parent / "logs" / output_path.name.replace(".txt", ".jsonl")
    errors_path = output_path.parent / "cuda_errors" / output_path.name
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(remaining)
    chunk_size = args.batch_size  # flush interval for resume safety
    t_infer_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Model  : {args.judge_model.split('/')[-1]}")
    print(f"  Mode   : {args.prompt_mode}")
    print(f"  Split  : {args.split}  |  Pairs: {total}")
    print(f"  Output : {output_path}")
    print(f"{'='*60}\n")

    with open(output_path, "a") as out_f, \
         open(logs_path, "a") as log_f, \
         open(errors_path, "a") as err_f:

        pbar = tqdm(total=total, desc="Judging", unit="pairs",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        for chunk_start in range(0, total, chunk_size):
            chunk = remaining[chunk_start:chunk_start + chunk_size]

            valid = []
            for qid, docid in chunk:
                if qid not in topics:
                    err_f.write(f"missing_query\t{qid}\n")
                    pbar.update(1)
                    continue
                if docid not in corpus:
                    err_f.write(f"missing_doc\t{qid}\t{docid}\n")
                    pbar.update(1)
                    continue
                user_content = get_umbrella_prompt(
                    query=topics[qid], passage=corpus[docid], mode=args.prompt_mode
                )
                messages = [{"role": "user", "content": user_content}]
                formatted = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                # Prefix forces model to output just the digit score
                formatted += "##final score: "
                valid.append((qid, docid, formatted))

            if not valid:
                continue

            prompts = [p for _, _, p in valid]
            outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

            for (qid, docid, _), output in zip(valid, outputs):
                text = output.outputs[0].text.strip()
                score = extract_score(text)
                out_f.write(f"{qid} 0 {docid} {score}\n")
                log_f.write(json.dumps({
                    "prompt_mode": args.prompt_mode,
                    "qidx": qid,
                    "docidx": docid,
                    "query": topics[qid],
                    "passage": corpus[docid],
                    "LLMs_output": text,
                    "final_relevance_score": score,
                }) + "\n")

            out_f.flush()
            log_f.flush()
            pbar.update(len(valid))

            # verbose chunk summary
            done = chunk_start + len(chunk)
            elapsed = time.time() - t_infer_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else 0
            print(f"  [chunk] {done}/{total} pairs | "
                  f"{rate:.1f} pairs/s | ETA {eta/60:.1f} min", flush=True)

        pbar.close()

    elapsed_total = time.time() - t_infer_start
    print(f"\n{'='*60}")
    print(f"  DONE: {total} pairs in {elapsed_total/60:.1f} min  ({total/elapsed_total:.1f} pairs/s)")
    print(f"  Results → {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
