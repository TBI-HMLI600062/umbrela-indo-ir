"""
UMBRELA LLM Judge inference with batched HuggingFace generation on MIRACL-ID.
Optimized for RTX 5090 (32GB VRAM) — processes pairs in large batches instead of one-by-one.

Note: vLLM requires CUDA driver 572+ (for CUDA 12.9 PTX). On driver 570.x (CUDA 12.8),
      this script uses batched HF Transformers instead (still 10-20x faster than sequential).

Args:
    --judge-model   HF model ID (e.g. Qwen/Qwen2.5-7B-Instruct)
    --split         train | val | test
    --n-queries     max number of unique queries to process (default: all)
    --output        output TREC qrels file path
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --prompt-mode   zeroshot_bing | zeroshot_basic | fewshot_bing | fewshot_basic
    --token         HuggingFace token for private models (optional)
    --batch-size    pairs per forward pass (default: 64 for RTX 5090 32GB)
    --max-length    max input token length (default: 2048)

Example:
    python qrel_generation/inference_vllm.py \\
        --judge-model Qwen/Qwen2.5-7B-Instruct \\
        --split test --output results/qrels/qwen_test.txt
"""

import sys
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="UMBRELA judge inference (batched HF) on MIRACL-ID."
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
                                 "fewshot_bing", "fewshot_basic"])
    parser.add_argument("--token", default=None, help="HF token for private models")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Pairs per GPU forward pass (default: 64 for RTX 5090 32GB)")
    parser.add_argument("--max-length", type=int, default=2048,
                        help="Max input token length with truncation (default: 2048)")
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


def run_batch(model, tokenizer, prompts: list, max_new_tokens: int, max_length: int) -> list:
    """Tokenize and generate for a batch of formatted prompts. Returns list of generated strings."""
    import torch
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
    )


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

    import torch
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from prompts import get_umbrella_prompt

    print(f"Loading model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"Model loaded on: {next(model.parameters()).device}")

    logs_path = output_path.parent / "logs" / output_path.name.replace(".txt", ".jsonl")
    errors_path = output_path.parent / "cuda_errors" / output_path.name
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(remaining)
    batch_size = args.batch_size
    n_batches = (total + batch_size - 1) // batch_size

    with open(output_path, "a") as out_f, \
         open(logs_path, "a") as log_f, \
         open(errors_path, "a") as err_f:

        pbar = tqdm(total=total, desc="Judging")

        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch = remaining[batch_start:batch_start + batch_size]

            valid = []
            for qid, docid in batch:
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
                valid.append((qid, docid, formatted))

            if not valid:
                continue

            prompts = [p for _, _, p in valid]
            try:
                generated_texts = run_batch(
                    model, tokenizer, prompts,
                    max_new_tokens=100,
                    max_length=args.max_length,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                # Retry with half-batch
                mid = len(prompts) // 2
                try:
                    g1 = run_batch(model, tokenizer, prompts[:mid], 100, args.max_length)
                    g2 = run_batch(model, tokenizer, prompts[mid:], 100, args.max_length)
                    generated_texts = g1 + g2
                except Exception as e2:
                    for qid, docid, _ in valid:
                        err_f.write(f"oom_skip\t{qid}\t{docid}\t{e2}\n")
                        out_f.write(f"{qid} 0 {docid} 0\n")
                    pbar.update(len(valid))
                    continue

            for (qid, docid, _), generated in zip(valid, generated_texts):
                score = extract_score(generated)
                out_f.write(f"{qid} 0 {docid} {score}\n")
                log_f.write(json.dumps({
                    "prompt_mode": args.prompt_mode,
                    "qidx": qid,
                    "docidx": docid,
                    "query": topics[qid],
                    "passage": corpus[docid],
                    "LLMs_output": generated,
                    "final_relevance_score": score,
                }) + "\n")

            out_f.flush()
            log_f.flush()
            pbar.update(len(valid))

        pbar.close()

    print(f"\nDone. Results written to {output_path}")


if __name__ == "__main__":
    main()
