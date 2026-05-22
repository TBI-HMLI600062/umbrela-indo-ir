"""
Prepare preference pairs for ORPO fine-tuning of Qwen2.5-7B-Instruct LLM judge.

For each (query, doc, rel) in human qrels, creates a contrastive pair on the
same prompt context — model learns correct vs incorrect response for that specific pair:
  rel=1 → chosen="##final score: 3",  rejected="##final score: 0"
  rel=0 → chosen="##final score: 0",  rejected="##final score: 3"

Dataset is balanced: positives upsampled to match negatives (or vice versa).

Args:
    --human-qrels   TREC human qrels (train split)
    --val-qrels     TREC human qrels (val split, optional)
    --data-dir      MIRACL-ID directory (default: data/miracl-id/)
    --output        Output directory
    --prompt-mode   Prompt template (default: zeroshot_bing)
    --balance       Balance pos/neg pairs (default: True)

Example:
    python lora/prepare_orpo_data.py \\
        --human-qrels data/miracl-id/qrels/human/train.txt \\
        --val-qrels   data/miracl-id/qrels/human/val.txt \\
        --output      results/orpo_data/qwen/
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-qrels", required=True)
    parser.add_argument("--val-qrels",   default=None)
    parser.add_argument("--data-dir",    default="data/miracl-id/")
    parser.add_argument("--output",      required=True)
    parser.add_argument("--prompt-mode", default="zeroshot_bing")
    parser.add_argument("--no-balance",  action="store_true",
                        help="Disable pos/neg balancing (default: balance enabled)")
    return parser.parse_args()


def parse_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def load_all_topics(data_dir):
    topics = {}
    for tsv in (data_dir / "topics").glob("*.tsv"):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0]] = parts[1]
    return topics


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
    missing = needed_docids - corpus.keys()
    if missing:
        print(f"  Warning: {len(missing)} docids not found in corpus")
    return corpus


def build_orpo_examples(qrels, topics, corpus, get_prompt):
    """
    Build (prompt, chosen, rejected) triples.
    rel=1 → chosen=score3, rejected=score0
    rel=0 → chosen=score0, rejected=score3
    """
    pos_examples = []  # rel=1 pairs
    neg_examples = []  # rel=0 pairs

    for qid, doc_rels in tqdm(qrels.items(), desc="Building pairs"):
        if qid not in topics:
            continue
        query = topics[qid]
        for docid, rel in doc_rels.items():
            if docid not in corpus:
                continue
            prompt_content = get_prompt(query=query, passage=corpus[docid])
            if rel == 1:
                pos_examples.append({
                    "prompt":   prompt_content,
                    "chosen":   "##final score: 3",
                    "rejected": "##final score: 0",
                })
            else:
                neg_examples.append({
                    "prompt":   prompt_content,
                    "chosen":   "##final score: 0",
                    "rejected": "##final score: 3",
                })

    return pos_examples, neg_examples


def balance_and_merge(pos_examples, neg_examples, seed=42):
    """Upsample minority class to match majority class size."""
    rng = random.Random(seed)
    n_pos, n_neg = len(pos_examples), len(neg_examples)
    print(f"  Before balance: {n_pos:,} positive, {n_neg:,} negative")

    if n_pos < n_neg:
        # Upsample positives
        extra = n_neg - n_pos
        upsampled = rng.choices(pos_examples, k=extra)
        all_examples = pos_examples + upsampled + neg_examples
    elif n_neg < n_pos:
        extra = n_pos - n_neg
        upsampled = rng.choices(neg_examples, k=extra)
        all_examples = pos_examples + neg_examples + upsampled
    else:
        all_examples = pos_examples + neg_examples

    rng.shuffle(all_examples)
    print(f"  After balance:  {len(all_examples):,} total pairs ({len(all_examples)//2:,} pos, {len(all_examples)//2:,} neg)")
    return all_examples


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    random.seed(42)

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompts import get_umbrella_prompt

    def get_prompt(query, passage):
        return get_umbrella_prompt(query=query, passage=passage, mode=args.prompt_mode)

    print(f"Loading human qrels from {args.human_qrels}...")
    train_qrels = parse_qrels(Path(args.human_qrels))
    n_pairs = sum(len(v) for v in train_qrels.values())
    print(f"  {len(train_qrels):,} queries, {n_pairs:,} pairs")

    val_qrels = None
    if args.val_qrels:
        print(f"Loading val qrels from {args.val_qrels}...")
        val_qrels = parse_qrels(Path(args.val_qrels))
        print(f"  {len(val_qrels):,} queries")

    print("Loading topics...")
    topics = load_all_topics(data_dir)

    needed = {d for dq in train_qrels.values() for d in dq}
    if val_qrels:
        needed |= {d for dq in val_qrels.values() for d in dq}
    print(f"Loading corpus subset ({len(needed):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed)

    print("Building train preference pairs...")
    pos_ex, neg_ex = build_orpo_examples(train_qrels, topics, corpus, get_prompt)

    if not args.no_balance:
        train_examples = balance_and_merge(pos_ex, neg_ex)
    else:
        train_examples = pos_ex + neg_ex
        random.shuffle(train_examples)
        print(f"  Total (unbalanced): {len(train_examples):,} pairs")

    write_jsonl(train_examples, output_dir / "train.jsonl")
    print(f"  Written to {output_dir / 'train.jsonl'}")

    if val_qrels:
        print("Building val preference pairs...")
        val_pos, val_neg = build_orpo_examples(val_qrels, topics, corpus, get_prompt)
        val_examples = val_pos + val_neg
        random.shuffle(val_examples)
        write_jsonl(val_examples, output_dir / "val.jsonl")
        print(f"  {len(val_examples):,} val pairs → {output_dir / 'val.jsonl'}")

    meta = {
        "human_qrels":  str(args.human_qrels),
        "prompt_mode":  args.prompt_mode,
        "n_train":      len(train_examples),
        "n_pos_raw":    len(pos_ex),
        "n_neg_raw":    len(neg_ex),
        "balanced":     not args.no_balance,
        "label_map":    {"rel=1": "chosen=3,rejected=0", "rel=0": "chosen=0,rejected=3"},
    }
    with open(output_dir / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Data written to {output_dir}")


if __name__ == "__main__":
    main()
