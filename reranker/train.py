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

    import random
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
    from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm

    print(f"Loading training data from {args.training_data}...")
    triplets = load_triplets(Path(args.training_data))
    print(f"  {len(triplets):,} triplets loaded")

    # Expand triplets → (text_a, text_b, label) pairs
    pairs, labels_all = [], []
    for t in triplets:
        pairs.append((t["query"], t["pos"])); labels_all.append(1.0)
        pairs.append((t["query"], t["neg"])); labels_all.append(0.0)

    # Apply max-steps limit for smoke test
    if args.max_steps > 0:
        cap = args.max_steps * args.batch_size * 2
        pairs, labels_all = pairs[:cap], labels_all[:cap]
        print(f"  Smoke test: capped to {len(pairs)} examples")

    combined = list(zip(pairs, labels_all))
    random.seed(42)
    random.shuffle(combined)
    pairs, labels_all = [x[0] for x in combined], [x[1] for x in combined]
    print(f"  Total training examples: {len(pairs):,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model: {args.model}  (device={device})")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_hf = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model_hf = model_hf.to(device)

    total_steps = (len(pairs) // args.batch_size) * args.epochs
    warmup_steps = min(100, max(1, total_steps // 10))
    print(f"Training: epochs={args.epochs}, batch={args.batch_size}, "
          f"lr={args.lr}, warmup={warmup_steps}, total_steps={total_steps}")

    optimizer = AdamW(model_hf.parameters(), lr=args.lr, weight_decay=0.0)
    warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps)
    decay_scheduler  = LinearLR(optimizer, start_factor=1.0, end_factor=0.0,
                                 total_iters=max(1, total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                              milestones=[warmup_steps])
    loss_fn = nn.BCEWithLogitsLoss()

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model_hf.train()
        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(range(0, len(pairs), args.batch_size),
                    desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for i in pbar:
            batch_pairs  = pairs[i:i + args.batch_size]
            batch_labels = torch.tensor(labels_all[i:i + args.batch_size],
                                        dtype=torch.float32, device=device)
            enc = tokenizer(
                [p[0] for p in batch_pairs],
                [p[1] for p in batch_pairs],
                padding=True, truncation=True, max_length=512,
                return_tensors="pt",
            ).to(device)

            optimizer.zero_grad()
            logits = model_hf(**enc).logits.squeeze(-1)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model_hf.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1
            if global_step % 500 == 0:
                pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}", step=global_step)

            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        print(f"  Epoch {epoch}: avg_loss={epoch_loss / max(n_batches, 1):.4f}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    model_hf.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "n_triplets": len(triplets),
        "n_train_examples": len(pairs),
        "training_data": str(args.training_data),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved to {output_dir}")
    print(f"Training metadata saved to {output_dir}/training_meta.json")


if __name__ == "__main__":
    main()
