"""
Logit-based LLM judge scoring with temperature calibration (Level 2 calibration).

Instead of generating text and regex-extracting the score, this script:
1. Runs a single forward pass on each prompt
2. Extracts logits at the NEXT-token position (where model would predict the score)
3. Computes P("0"), P("1"), P("2"), P("3") via softmax
4. Outputs a continuous score = 0·P(0) + 1·P(1) + 2·P(2) + 3·P(3)

This continuous score can be thresholded at any float value (e.g., 1.5, 2.0),
or temperature-scaled to tune calibration without retraining.

Temperature scaling:
  - T > 1: softer distribution (model less confident) → helps if model overconfident
  - T < 1: sharper distribution (model more confident)
  - T=1:   no change (default)

Calibration workflow:
  1. Run on val split with T=1, find best float threshold τ
  2. Optionally also tune T on val (grid search T ∈ {0.5, 0.75, 1.0, 1.5, 2.0})
  3. Apply best (T, τ) to test split

Args:
    --judge-model       HF model ID
    --split             train | val | test
    --output            Output TREC qrels file (scores are continuous, multiplied by 100)
    --temperature       Softmax temperature (default: 1.0)
    --batch-size        Pairs per forward pass (default: 32)
    --prompt-mode       Prompt mode (default: zeroshot_bing)
    --lora-path         Optional LoRA adapter path
    --n-queries         Max queries (default: all)
    --data-dir          MIRACL-ID directory

Example:
    python evaluation/logit_inference.py \\
        --judge-model Qwen/Qwen2.5-7B-Instruct \\
        --split val \\
        --output results/qrels/qwen_logit_val.txt

Then calibrate:
    python evaluation/calibrate_logit.py \\
        --val-scores  results/qrels/qwen_logit_val.txt \\
        --test-scores results/qrels/qwen_logit_test.txt \\
        --val-human   data/miracl-id/qrels/human/val.txt \\
        --test-human  data/miracl-id/qrels/human/test.txt
"""

import sys
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Softmax temperature for score token logits (default: 1.0)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic",
                                 "zeroshot_bing_strict"])
    parser.add_argument("--lora-path", default=None)
    parser.add_argument("--n-queries", type=int, default=None)
    parser.add_argument("--data-dir", default="data/miracl-id/")
    return parser.parse_args()


# --- reuse loaders from inference_vllm.py ---
def load_topics(data_dir, split):
    path = data_dir / "topics" / f"{split}.tsv"
    topics = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def load_candidates(data_dir, split):
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


def load_corpus_subset(data_dir, needed_docids):
    corpus_path = data_dir / "corpus" / "corpus.jsonl"
    corpus = {}
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus", unit=" passages"):
            obj = json.loads(line)
            if obj["docid"] in needed_docids:
                corpus[obj["docid"]] = obj["doc"]
            if len(corpus) == len(needed_docids):
                break
    return corpus


def load_processed(output_path):
    processed = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    processed.add((parts[0], parts[2]))
    return processed


def filter_by_n_queries(pairs, n_queries):
    seen = {}
    result = []
    for qid, docid in pairs:
        if qid not in seen:
            if len(seen) >= n_queries:
                continue
            seen[qid] = True
        result.append((qid, docid))
    return result


def get_score_token_ids(tokenizer):
    """Get token IDs for '0', '1', '2', '3' (single digit tokens)."""
    ids = {}
    for digit in ["0", "1", "2", "3"]:
        tokens = tokenizer.encode(digit, add_special_tokens=False)
        # Use the last token if the digit tokenizes to multiple tokens
        ids[int(digit)] = tokens[-1]
    return ids


def run_logit_batch(model, tokenizer, prompts, score_token_ids, temperature):
    """
    Forward pass on a batch of prompts.
    Returns list of continuous scores (float, range 0-3).

    The score is computed as the expected value over {0,1,2,3} probabilities
    extracted from the logits at the next-token position after the prompt.
    """
    import torch
    import torch.nn.functional as F

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits shape: (batch, seq_len, vocab_size)
    # We want logits at the LAST non-padded position for each example
    logits = outputs.logits  # (B, L, V)

    # Find last non-pad position per example
    attn = inputs["attention_mask"]  # (B, L)
    last_positions = attn.sum(dim=1) - 1  # (B,)

    scores = []
    for i, last_pos in enumerate(last_positions):
        token_logits = logits[i, last_pos, :]  # (V,)

        # Extract logits for score tokens {0,1,2,3}
        score_logits = torch.tensor(
            [token_logits[score_token_ids[d]].item() for d in range(4)],
            dtype=torch.float32,
        )

        # Temperature scaling
        if temperature != 1.0:
            score_logits = score_logits / temperature

        probs = F.softmax(score_logits, dim=0)  # P(0), P(1), P(2), P(3)

        # Continuous expected score in [0, 3]
        expected = sum(d * probs[d].item() for d in range(4))
        scores.append(expected)

    return scores


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompts import get_umbrella_prompt

    topics = load_topics(data_dir, args.split)
    candidates = load_candidates(data_dir, args.split)

    if args.n_queries:
        candidates = filter_by_n_queries(candidates, args.n_queries)

    processed = load_processed(output_path)
    if processed:
        print(f"Resuming: {len(processed)} pairs already done")
    remaining = [(q, d) for q, d in candidates if (q, d) not in processed]
    print(f"Pairs to process: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    needed_docids = {d for _, d in remaining}
    corpus = load_corpus_subset(data_dir, needed_docids)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

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

    if args.lora_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()

    model.eval()

    score_token_ids = get_score_token_ids(tokenizer)
    print(f"Score token IDs: {score_token_ids}")
    print(f"Temperature: {args.temperature}")

    total = len(remaining)
    n_batches = (total + args.batch_size - 1) // args.batch_size

    logs_path = output_path.parent / "logs" / (output_path.stem + "_logit.jsonl")
    logs_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a") as out_f, open(logs_path, "a") as log_f:
        pbar = tqdm(total=total, desc="Scoring")
        for batch_idx in range(n_batches):
            batch = remaining[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]

            valid = []
            for qid, docid in batch:
                if qid not in topics or docid not in corpus:
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
                cont_scores = run_logit_batch(
                    model, tokenizer, prompts, score_token_ids, args.temperature
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                mid = len(prompts) // 2
                try:
                    s1 = run_logit_batch(model, tokenizer, prompts[:mid], score_token_ids, args.temperature)
                    s2 = run_logit_batch(model, tokenizer, prompts[mid:], score_token_ids, args.temperature)
                    cont_scores = s1 + s2
                except Exception:
                    cont_scores = [0.0] * len(valid)

            for (qid, docid, _), score in zip(valid, cont_scores):
                # Store as integer × 1000 to preserve precision in TREC format
                # e.g., 2.347 → stored as 2347 (divide by 1000 when reading)
                score_int = round(score * 1000)
                out_f.write(f"{qid} 0 {docid} {score_int}\n")
                log_f.write(json.dumps({
                    "qid": qid, "docid": docid,
                    "continuous_score": score, "temperature": args.temperature,
                }) + "\n")

            out_f.flush()
            pbar.update(len(valid))

        pbar.close()

    print(f"Done. Written to {output_path}")
    print(f"Note: scores are stored as int(score × 1000). Use calibrate_logit.py to evaluate.")


if __name__ == "__main__":
    main()
