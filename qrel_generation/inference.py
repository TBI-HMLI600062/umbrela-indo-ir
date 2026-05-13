"""
UMBRELA LLM Judge inference for MIRACL-ID.

Runs an LLM judge on MIRACL-ID query-passage pairs and writes TREC-format qrels.
Supports resume (append mode) and CUDA OOM recovery.

Args:
    --judge-model   HF model ID (e.g. Qwen/Qwen2.5-7B-Instruct)
    --split         train | val | test
    --n-queries     max number of unique queries to process (default: all)
    --max-pairs     max number of query-passage pairs to process (default: all)
    --output        output TREC qrels file path
    --data-dir      path to processed MIRACL-ID directory (default: data/miracl-id/)
    --prompt-mode   zeroshot_bing | zeroshot_basic (default: zeroshot_bing)
    --batch-size    number of pairs to judge per generation call (default: 1)
    --max-new-tokens max generated tokens per judgment (default: 100)
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
import re
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="UMBRELA judge inference on MIRACL-ID.")
    parser.add_argument("--judge-model", required=True, help="HF model ID")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--n-queries", type=int, default=None,
                        help="Max unique queries to process (default: all)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max query-passage pairs to process after query filtering (default: all)")
    parser.add_argument("--output", required=True, help="Output TREC qrels file")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="Path to processed MIRACL-ID directory")
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic",
                                 "zeroshot_bing_strict"])
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Number of pairs per generation call (default: 1)")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Max generated tokens per judgment (default: 100)")
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


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def get_terminators(model_pipeline):
    unk_id = model_pipeline.tokenizer.unk_token_id
    return [token_id for token_id in [
        model_pipeline.tokenizer.eos_token_id,
        model_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ] if token_id is not None and token_id != unk_id]


def build_prompt(model_pipeline, user_prompt: str, system_message: str) -> str:
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    tokenizer = model_pipeline.tokenizer
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return user_prompt


def parse_vincent_score(text: str) -> int:
    """Vincent-only parser for strict experiments; does not alter shared scoring code."""
    final_score_match = re.search(
        r"(?:^|\n)\s*(?:##\s*)?final score:\s*([0-3])",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    if final_score_match:
        return int(final_score_match.group(1))

    o_score_match = re.search(r"(?:^|\n)\s*O:\s*([0-3])", text, re.IGNORECASE | re.MULTILINE)
    if o_score_match:
        return int(o_score_match.group(1))

    if re.fullmatch(r"\s*[0-3]\s*", text):
        return int(text.strip())

    general_match = re.search(r"\b[0-3]\b", text)
    if general_match:
        return int(general_match.group())

    return 0


def score_batch(batch: list, topics: dict, corpus: dict, model_pipeline,
                log_file_path: str, system_message: str, prompt_mode: str,
                max_new_tokens: int) -> list:
    """Judge a batch of valid (qid, docid) pairs and return (qid, docid, score)."""
    from prompts import get_umbrella_prompt

    prompts = []
    metadata = []
    for qid, docid in batch:
        umbrella_prompt = get_umbrella_prompt(
            query=topics[qid],
            passage=corpus[docid],
            mode=prompt_mode,
        )
        prompts.append(build_prompt(model_pipeline, umbrella_prompt, system_message))
        metadata.append((qid, docid, umbrella_prompt))

    terminators = get_terminators(model_pipeline)
    pad_token_id = (
        model_pipeline.tokenizer.pad_token_id
        or model_pipeline.tokenizer.eos_token_id
        or (terminators[0] if terminators else None)
    )
    outputs = model_pipeline(
        prompts,
        max_new_tokens=max_new_tokens,
        eos_token_id=terminators,
        pad_token_id=pad_token_id,
        do_sample=False,
        temperature=None,
        top_p=None,
        batch_size=len(prompts),
    )

    results = []
    with open(log_file_path, "a") as log_f:
        for prompt, output, (qid, docid, _) in zip(prompts, outputs, metadata):
            generated_text = output[0]["generated_text"] if isinstance(output, list) else output["generated_text"]
            if getattr(model_pipeline.tokenizer, "chat_template", None) is not None:
                generated_text = generated_text[len(prompt):]

            score = parse_vincent_score(generated_text)
            if score not in [0, 1, 2, 3]:
                score = 0

            scoring_log = {
                "prompt_mode": prompt_mode,
                "qidx": qid,
                "docidx": docid,
                "query": topics[qid],
                "passage": corpus[docid],
                "LLMs_output": generated_text,
                "final_relevance_score": score,
            }
            log_f.write(json.dumps(scoring_log) + "\n")
            results.append((qid, docid, score))
    return results


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
    if args.max_pairs is not None:
        candidates = candidates[:args.max_pairs]

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
    os.environ["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    print(f"Loading model: {args.judge_model}")
    model = get_model_baseline(args.judge_model, use_together=False)

    # Setup log/error paths
    logs_path = output_path.parent / "logs" / output_path.name.replace(".txt", ".jsonl")
    errors_path = output_path.parent / "cuda_errors" / output_path.name
    logs_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    with open(output_path, "a") as out_f, open(errors_path, "a") as err_f:
        progress = tqdm(total=len(remaining), desc="Judging")
        for raw_batch in batched(remaining, args.batch_size):
            valid_batch = []
            for qid, docid in raw_batch:
                if qid not in topics:
                    err_f.write(f"missing_query\t{qid}\n")
                    progress.update(1)
                    continue
                if docid not in corpus:
                    err_f.write(f"missing_doc\t{qid}\t{docid}\n")
                    progress.update(1)
                    continue
                valid_batch.append((qid, docid))

            if not valid_batch:
                continue

            if args.batch_size > 1:
                try:
                    scored = score_batch(
                        valid_batch, topics, corpus, model,
                        str(logs_path), "", args.prompt_mode, args.max_new_tokens,
                    )
                    for qid, docid, score in scored:
                        out_f.write(f"{qid} 0 {docid} {score}\n")
                    out_f.flush()
                    progress.update(len(valid_batch))
                    continue

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("\nCUDA OOM on batch, falling back to single-pair scoring...")

                except Exception as e:
                    err_f.write(f"batch_error\t{valid_batch[0][0]}\t{valid_batch[0][1]}\t{e}\n")
                    print(f"\nBatch error, falling back to single-pair scoring: {e}")

            for qid, docid in valid_batch:
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
                progress.update(1)
        progress.close()

    print(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    main()
