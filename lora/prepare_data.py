"""
Prepare LoRA training data from MIRACL-ID human qrels.

Converts human relevance labels (0/1) into instruction-tuning pairs using
the UMBRELA prompt template. Output is compatible with Unsloth's dataset loader.

Human binary labels are mapped:
    0 (irrelevant) → score 0
    1 (relevant)   → score 3

Splits: merge train+val human qrels → 90/10 train/val for LoRA.
Test human qrels are held out for final evaluation.

Args:
    --data-dir        path to MIRACL-ID directory (default: data/miracl-id/)
    --output          output directory for LoRA training data
    --prompt-mode     UMBRELA prompt template (default: zeroshot_bing)
    --train-splits    qrel splits to use for training (default: train val)
    --val-ratio       fraction held out for validation (default: 0.1)
    --seed            random seed (default: 42)

Outputs:
    train.jsonl       ~37k pairs (90% of 41,358)
    val.jsonl         ~4k pairs  (10%)

Example:
    python lora/prepare_data.py --output data/lora/
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare LoRA training data from human qrels."
    )
    parser.add_argument("--data-dir", default="data/miracl-id/",
                        help="MIRACL-ID directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--prompt-mode", default="zeroshot_bing",
                        choices=["zeroshot_bing", "zeroshot_basic",
                                 "fewshot_bing", "fewshot_basic",
                                 "zeroshot_bing_strict"],
                        help="UMBRELA prompt template (default: zeroshot_bing)")
    parser.add_argument("--train-splits", nargs="+", default=["train", "val"],
                        help="Qrel splits for training (default: train val)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation ratio (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_human_qrels(data_dir: Path, split: str) -> list[tuple[str, str, int]]:
    """Load TREC qrels for a split → list of (qid, docid, rel)."""
    path = data_dir / "qrels" / "human" / f"{split}.txt"
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid = parts[0]
            docid = parts[2]
            rel = int(parts[3])
            pairs.append((qid, docid, rel))
    return pairs


def load_all_topics(data_dir: Path) -> dict[str, str]:
    """Load query texts from all splits → {qid: query_text}."""
    topics = {}
    for tsv in (data_dir / "topics").glob("*.tsv"):
        with open(tsv) as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0]] = parts[1]
    return topics


def load_corpus_subset(data_dir: Path, needed_docids: set[str]) -> dict[str, str]:
    """Load only the corpus passages referenced by needed_docids."""
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


def map_label(rel: int) -> int:
    """Map binary human label to UMBRELA score."""
    return 3 if rel == 1 else 0


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # Import here so --help works without heavy deps
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from prompts import get_umbrella_prompt

    # Load human qrels from requested splits
    all_pairs = []
    for split in args.train_splits:
        pairs = load_human_qrels(data_dir, split)
        print(f"Human qrels ({split}): {len(pairs):,} pairs")
        all_pairs.extend(pairs)
    print(f"Total pairs: {len(all_pairs):,}")

    # Load topics
    print("Loading topics...")
    topics = load_all_topics(data_dir)
    print(f"  {len(topics):,} queries loaded")

    # Load only needed corpus passages
    needed_docids = {docid for _, docid, _ in all_pairs}
    print(f"Loading corpus subset ({len(needed_docids):,} passages)...")
    corpus = load_corpus_subset(data_dir, needed_docids)

    # Build training examples
    examples = []
    n_skipped = 0
    for qid, docid, rel in tqdm(all_pairs, desc="Building examples"):
        if qid not in topics or docid not in corpus:
            n_skipped += 1
            continue

        prompt = get_umbrella_prompt(
            query=topics[qid],
            passage=corpus[docid],
            mode=args.prompt_mode,
        )
        score = map_label(rel)
        examples.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"##final score: {score}"},
            ],
        })

    print(f"Built {len(examples):,} examples ({n_skipped} skipped)")

    # Shuffle and split
    random.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio))
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    # Write output files
    for name, data in [("train.jsonl", train_examples),
                       ("val.jsonl", val_examples)]:
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for ex in tqdm(data, desc=f"Writing {name}", unit=" examples"):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(data):,} examples → {path}")

    # Stats
    train_pos = sum(1 for e in train_examples
                    if e["messages"][1]["content"].endswith("3"))
    val_pos = sum(1 for e in val_examples
                  if e["messages"][1]["content"].endswith("3"))
    print(f"\nTrain: {len(train_examples):,} examples "
          f"(pos={train_pos:,}, pos_rate={train_pos / len(train_examples):.1%})")
    print(f"Val:   {len(val_examples):,} examples "
          f"(pos={val_pos:,}, pos_rate={val_pos / len(val_examples):.1%})")
    print(f"\nDone. Training data saved to {output_dir}")
    print(f"Next: python lora/train.py --data-dir {output_dir} "
          f"--model Qwen/Qwen2.5-7B-Instruct --output results/lora/qwen/")


if __name__ == "__main__":
    main()
