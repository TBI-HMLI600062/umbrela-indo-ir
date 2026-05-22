"""
LoRA fine-tuning of Qwen2.5-7B-Instruct as LLM relevance judge (E8T2).

Uses PEFT library for parameter-efficient fine-tuning on human-labeled
query-passage pairs from MIRACL-ID. Training signal: human qrels (binary)
mapped to 0/3 target scores in the judge's ##final score: X output format.
Loss is computed only on response tokens (prompt tokens are masked).

Args:
    --training-data  path to data directory containing train.jsonl
    --model          base model ID (default: Qwen/Qwen2.5-7B-Instruct)
    --output         output directory for LoRA adapter
    --epochs         number of training epochs (default: 3)
    --batch-size     per-device batch size (default: 4 for RTX 5090 32GB)
    --grad-accum     gradient accumulation steps (default: 4; effective batch = 16)
    --lr             learning rate (default: 2e-4)
    --lora-r         LoRA rank (default: 16)
    --lora-alpha     LoRA alpha (default: 32)
    --max-length     max sequence length in tokens (default: 1024)
    --max-steps      max gradient steps; 0=no limit. Use for smoke tests.
    --val-data       optional val data directory (val.jsonl inside)

Both RTX 5090 32GB and RTX 4090 24GB use the same defaults (batch=2, grad_accum=8).
Gradient checkpointing is always enabled, keeping VRAM ~18-20GB.

Example:
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --output results/models/lora_qwen/ \\
        --epochs 3

Smoke test (RTX 5090):
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --output results/models/lora_qwen_smoke/ \\
        --epochs 1 --max-steps 10

Smoke test (RTX 4090):
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --output results/models/lora_qwen_smoke/ \\
        --epochs 1 --max-steps 10 --batch-size 2 --grad-accum 8
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-7B-Instruct as LLM judge."
    )
    parser.add_argument("--training-data", required=True,
                        help="Training data directory (contains train.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model HF ID")
    parser.add_argument("--output", required=True,
                        help="Output directory for LoRA adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2; use --grad-accum to scale "
                             "effective batch. Works on both RTX 5090 32GB and RTX 4090 24GB)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8; "
                             "effective batch = batch_size × grad_accum = 16)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Max sequence length in tokens (default: 1024)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max gradient steps; 0=no limit (use for smoke test)")
    parser.add_argument("--val-data", default=None,
                        help="Val data directory (val.jsonl inside); optional")
    return parser.parse_args()


def load_examples(data_dir: Path, filename: str = "train.jsonl") -> list:
    path = data_dir / filename
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def tokenize_example(example: dict, tokenizer, max_length: int):
    """
    Tokenize one SFT example. Returns (input_ids, labels) where
    prompt tokens are masked with -100 so loss is only on response tokens.
    """
    messages = [{"role": "user", "content": example["prompt"]}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    full_str = prompt_str + example["response"] + tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = tokenizer.encode(
        full_str, add_special_tokens=False, truncation=True, max_length=max_length
    )

    prompt_len = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_len + full_ids[prompt_len:]
    return full_ids, labels


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import random
    import torch
    import torch.nn as nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LinearLR, SequentialLR
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from tqdm import tqdm

    random.seed(42)
    torch.manual_seed(42)

    data_dir = Path(args.training_data)
    print(f"Loading training data from {data_dir}...")
    train_examples = load_examples(data_dir)
    print(f"  {len(train_examples):,} examples")

    if args.max_steps > 0:
        # Cap dataset for smoke test; keep a bit more than strictly needed
        cap = args.max_steps * args.batch_size * args.grad_accum * 2
        train_examples = train_examples[:cap]
        print(f"  Smoke test: capped to {len(train_examples)} examples")

    random.shuffle(train_examples)

    val_examples = None
    if args.val_data:
        val_examples = load_examples(Path(args.val_data))
        print(f"  {len(val_examples):,} val examples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model: {args.model}  (device={device})")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    # Gradient checkpointing: recomputes activations during backward to save ~10GB VRAM.
    # Required for 7B model + LoRA on 32GB GPU. Adds ~30% compute overhead.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print("Tokenizing examples...")
    tokenized = []
    for ex in tqdm(train_examples, desc="Tokenizing"):
        input_ids, labels = tokenize_example(ex, tokenizer, args.max_length)
        tokenized.append((input_ids, labels))

    class SFTDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, i):
            return self.data[i]

    def collate_fn(batch):
        input_ids_list = [torch.tensor(ids) for ids, _ in batch]
        labels_list    = [torch.tensor(lbl) for _, lbl in batch]
        input_ids = nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return {
            "input_ids":      input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels":         labels.to(device),
        }

    loader = DataLoader(
        SFTDataset(tokenized), batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn
    )

    steps_per_epoch = len(loader) // args.grad_accum
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = min(100, max(1, int(total_steps * 0.03)))

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum} (effective={eff_batch}), "
          f"lr={args.lr}, lora_r={args.lora_r}, "
          f"warmup={warmup_steps}, total_steps={total_steps}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
    )
    decay_sched = LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0,
        total_iters=max(1, total_steps - warmup_steps)
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, decay_sched], milestones=[warmup_steps]
    )

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for step, batch in enumerate(pbar):
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()
            n_batches  += 1

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 50 == 0:
                    pbar.set_postfix(
                        loss=f"{epoch_loss / n_batches:.4f}", step=global_step
                    )
                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        print(f"  Epoch {epoch}: avg_loss={epoch_loss / max(n_batches, 1):.4f}")
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    # Save only the LoRA adapter (~50MB), not the full 14GB base weights
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model":       args.model,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  args.batch_size * args.grad_accum,
        "lr":               args.lr,
        "max_steps":        args.max_steps,
        "n_train_examples": len(train_examples),
        "training_data":    str(args.training_data),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLoRA adapter saved to {output_dir}")
    print(f"Training metadata saved to {output_dir}/training_meta.json")


if __name__ == "__main__":
    main()
