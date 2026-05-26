"""
Prepare instruction-tuning data for LoRA fine-tuning of Qwen2.5-7B-Instruct LLM judge.

Converts human qrels (binary TREC format) + MIRACL-ID corpus into chat-formatted
SFT examples. Human rel=1 → target score 3, human rel=0 → target score 0.
Mapping to scale extremes gives cleaner loss signal than intermediate values.

Args:
    --human-qrels   path to human qrels file (TREC format: qid 0 docid rel)
    --val-qrels     path to val qrels (optional; produces val.jsonl)
    --data-dir      path to MIRACL-ID directory (default: data/miracl-id/)
    --output        output directory for train.jsonl [+ val.jsonl] + data_meta.json
    --prompt-mode   prompt template to use (default: zeroshot_bing)

Example:
    python lora/prepare_data.py \\
        --human-qrels data/miracl-id/qrels/human/train.txt \\
        --val-qrels data/miracl-id/qrels/human/val.txt \\
        --output results/lora_data/qwen/
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-qrels", required=True,
                        help="Human qrels file (TREC format)")
    parser.add_argument("--val-qrels", default=None,
                        help="Optional val qrels for validation set")
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="MIRACL-ID directory (default: data/miracl-id/)")
    parser.add_argument("--output", required=True,
                        help="Output directory")
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic",
                                 "zeroshot_bing_strict"])
    return parser.parse_args()


def parse_qrels(path: Path) -> dict:
    """Parse TREC qrels → {qid: {docid: score}}."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, docid, score = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_all_topics(data_dir: Path) -> dict:
    """Load topics from all splits → {qid: query_text}."""
    topics = {}
    for tsv in (data_dir / "topics").glob("*.tsv"):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0]] = parts[1]
    return topics


def load_corpus_subset(data_dir: Path, needed_docids: set) -> dict:
    """Stream corpus and keep only passages referenced by needed_docids."""
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


def build_examples(qrels: dict, topics: dict, corpus: dict,
                   get_prompt, label_map: dict) -> list:
    """Build SFT examples as {prompt, response} dicts."""
    examples = []
    n_skipped = 0
    for qid, doc_scores in tqdm(qrels.items(), desc="Building examples"):
        if qid not in topics:
            n_skipped += 1
            continue
        query = topics[qid]
        for docid, rel in doc_scores.items():
            if docid not in corpus:
                continue
            passage = corpus[docid]
            prompt_content = get_prompt(query=query, passage=passage)
            target_score = label_map[rel]
            examples.append({
                "prompt": prompt_content,
                "response": f"##final score: {target_score}",
            })
    if n_skipped:
        print(f"  Skipped {n_skipped} queries with no topic entry")
    return examples


def write_jsonl(examples: list, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompts import get_umbrella_prompt

    def get_prompt(query, passage):
        return get_umbrella_prompt(query=query, passage=passage, mode=args.prompt_mode)

    # rel=0 (not relevant) → score 0; rel=1 (relevant) → score 3
    # Mapping to scale extremes gives unambiguous training signal
    label_map = {0: 0, 1: 3}

    print(f"Loading human qrels from {args.human_qrels}...")
    train_qrels = parse_qrels(Path(args.human_qrels))
    n_pairs = sum(len(v) for v in train_qrels.values())
    print(f"  {len(train_qrels):,} queries, {n_pairs:,} pairs")

    val_qrels = None
    if args.val_qrels:
        print(f"Loading val qrels from {args.val_qrels}...")
        val_qrels = parse_qrels(Path(args.val_qrels))
        n_val = sum(len(v) for v in val_qrels.values())
        print(f"  {len(val_qrels):,} queries, {n_val:,} pairs")

    print("Loading topics...")
    topics = load_all_topics(data_dir)
    print(f"  {len(topics):,} queries loaded")

    # Collect all needed docids in one corpus pass
    needed = {d for dq in train_qrels.values() for d in dq}
    if val_qrels:
        needed |= {d for dq in val_qrels.values() for d in dq}
    print(f"Loading corpus subset ({len(needed):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed)

    print("Building train examples...")
    train_examples = build_examples(train_qrels, topics, corpus, get_prompt, label_map)
    print(f"  {len(train_examples):,} examples")
    write_jsonl(train_examples, output_dir / "train.jsonl")
    print(f"  Written to {output_dir / 'train.jsonl'}")

    if val_qrels:
        print("Building val examples...")
        val_examples = build_examples(val_qrels, topics, corpus, get_prompt, label_map)
        print(f"  {len(val_examples):,} examples")
        write_jsonl(val_examples, output_dir / "val.jsonl")
        print(f"  Written to {output_dir / 'val.jsonl'}")

    meta = {
        "human_qrels": str(args.human_qrels),
        "val_qrels": str(args.val_qrels) if args.val_qrels else None,
        "prompt_mode": args.prompt_mode,
        "label_map": label_map,
        "n_train": len(train_examples),
        "n_val": len(val_examples) if val_qrels else 0,
    }
    with open(output_dir / "data_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Data written to {output_dir}")


if __name__ == "__main__":
    main()
