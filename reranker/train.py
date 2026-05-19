"""
Fine-tune BAAI/bge-reranker-v2-m3 on LLM-generated qrels.

Loads (query, pos_doc, neg_doc) triplets from prepare_data.py output and trains
a CrossEncoder with binary cross-entropy loss.

Args:
    --training-data     path to prepared training data directory (contains train.jsonl)
    --model             base model ID (default: BAAI/bge-reranker-v2-m3)
    --output            output directory for fine-tuned model
    --epochs            number of training epochs (default: 3)
    --batch-size        training batch size per device (default: 32)
    --lr                learning rate (default: 2e-5)
    --max-steps         max gradient steps total; 0=no limit. Use for smoke tests (default: 0)
    --val-data          optional validation data directory (train.jsonl inside)

Example:
    python reranker/train.py \\
        --training-data results/reranker_data/qwen/ \\
        --model BAAI/bge-reranker-v2-m3 \\
        --output results/reranker/qwen/ \\
        --epochs 3

Smoke test:
    python reranker/train.py \\
        --training-data results/smoke/reranker_data/ \\
        --output results/smoke/reranker/ \\
        --epochs 1 --max-steps 10
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BGE reranker on LLM qrels.")
    parser.add_argument("--training-data", required=True, help="Training data directory")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3",
                        help="Base model (default: BAAI/bge-reranker-v2-m3)")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max gradient steps (0=no limit, use for smoke test)")
    parser.add_argument("--val-data", default=None, help="Validation data directory")
    return parser.parse_args()


def load_triplets(data_dir: Path):
    """Load triplets from train.jsonl → list of dicts with query/pos/neg."""
    path = data_dir / "train.jsonl"
    triplets = []
    with open(path) as f:
        for line in f:
            triplets.append(json.loads(line))
    return triplets


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    print(f"Loading training data from {args.training_data}...")
    triplets = load_triplets(Path(args.training_data))
    print(f"  {len(triplets):,} triplets loaded")

    # Expand triplets → paired examples (pos=1, neg=0)
    train_examples = []
    for t in triplets:
        train_examples.append(InputExample(texts=[t["query"], t["pos"]], label=1.0))
        train_examples.append(InputExample(texts=[t["query"], t["neg"]], label=0.0))

    # Apply max-steps limit for smoke test (cap training data)
    if args.max_steps > 0:
        max_examples = args.max_steps * args.batch_size * 2
        if len(train_examples) > max_examples:
            train_examples = train_examples[:max_examples]
            print(f"  Smoke test: capped to {len(train_examples)} examples "
                  f"({args.max_steps} steps × batch {args.batch_size})")

    print(f"  Total training examples: {len(train_examples):,}")

    train_dataloader = DataLoader(
        train_examples, shuffle=True, batch_size=args.batch_size
    )

    # Optional validation data
    evaluator = None
    if args.val_data:
        from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
        val_triplets = load_triplets(Path(args.val_data))
        val_pairs = [(t["query"], t["pos"]) for t in val_triplets] + \
                    [(t["query"], t["neg"]) for t in val_triplets]
        val_labels = [1] * len(val_triplets) + [0] * len(val_triplets)
        evaluator = CEBinaryClassificationEvaluator(val_pairs, val_labels,
                                                     name="val", show_progress_bar=False)
        print(f"  Validation: {len(val_triplets):,} triplets from {args.val_data}")

    print(f"\nLoading model: {args.model}")
    model = CrossEncoder(args.model, num_labels=1, max_length=512)

    warmup_steps = min(100, max(1, len(train_dataloader) * args.epochs // 10))
    print(f"Training: epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, warmup={warmup_steps}")

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=args.epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": args.lr},
        show_progress_bar=True,
        output_path=str(output_dir),
        save_best_model=evaluator is not None,
    )

    model.save(str(output_dir))

    meta = {
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "n_triplets": len(triplets),
        "n_train_examples": len(train_examples),
        "training_data": str(args.training_data),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"Training metadata saved to {output_dir}/training_meta.json")


if __name__ == "__main__":
    main()
