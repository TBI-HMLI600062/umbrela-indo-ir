"""
LoRA fine-tune an LLM judge on human qrels using Unsloth + QLoRA.

Loads instruction-tuning data from lora/prepare_data.py, trains a LoRA adapter,
and saves the adapter for use with qrel_generation/inference.py.

Requires: unsloth, bitsandbytes, trl, peft, accelerate

Args:
    --data-dir        path to training data directory (contains train.jsonl and val.jsonl)
    --model           base model ID (default: Qwen/Qwen2.5-7B-Instruct)
    --output          output directory for LoRA adapter
    --epochs          number of training epochs (default: 3)
    --batch-size      per-device batch size (default: 4)
    --grad-accum      gradient accumulation steps (default: 4)
    --lr              learning rate (default: 2e-4)
    --lora-r          LoRA rank (default: 16)
    --lora-alpha      LoRA alpha (default: 16)
    --max-seq-length  max sequence length in tokens (default: 2048)
    --max-steps       max training steps; 0=no limit. Use for smoke test (default: 0)
    --hf-repo         optional HuggingFace repo to push adapter to
    --no-upload       skip HuggingFace upload

Smoke test:
    python lora/train.py --data-dir data/lora/ --output results/lora/smoke/ \\
        --epochs 1 --max-steps 20

Full training (Qwen):
    python lora/train.py --data-dir data/lora/ \\
        --model Qwen/Qwen2.5-7B-Instruct --output results/lora/qwen/

Full training (SahabatAI-Gemma2):
    python lora/train.py --data-dir data/lora/ \\
        --model GoToCompany/gemma2-9b-cpt-sahabatai-v1-instruct --output results/lora/gemma/
"""

import argparse
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune LLM judge with Unsloth."
    )
    parser.add_argument("--data-dir", required=True,
                        help="Directory with train.jsonl and val.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model ID")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (default: 4)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max training steps (0=no limit)")
    parser.add_argument("--hf-repo", default=None,
                        help="HF repo to push adapter (e.g. fassabilf/umbrela-lora-qwen)")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload")
    parser.add_argument("--token", default=None, help="HF token")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_datasets(data_dir: Path):
    """Load train and val JSONL files → HuggingFace DatasetDict."""
    from datasets import Dataset, DatasetDict

    def _load(path):
        examples = []
        with open(path) as f:
            for line in f:
                examples.append(json.loads(line))
        return Dataset.from_list(examples)

    train = _load(data_dir / "train.jsonl")
    val = _load(data_dir / "val.jsonl")
    return DatasetDict({"train": train, "validation": val})


def format_conversation(examples, tokenizer):
    """Apply chat template to messages list for SFT."""
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


def main():
    args = parse_args()

    # Defer heavy imports so --help works without GPU
    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import DatasetDict

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load datasets ----
    print(f"Loading training data from {data_dir}...")
    dataset = load_datasets(data_dir)
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Val:   {len(dataset['validation']):,} examples")

    # ---- Load model ----
    print(f"\nLoading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,       # auto-detect best dtype
        load_in_4bit=True,
    )

    # ---- Apply LoRA ----
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({trainable / total * 100:.2f}%)")

    # ---- Pre-format dataset with chat template ----
    print("Applying chat template...")
    formatted = DatasetDict({
        split: ds.map(
            lambda x: format_conversation(x, tokenizer),
            batched=True,
            remove_columns=ds.column_names,
        )
        for split, ds in dataset.items()
    })

    # ---- Train ----
    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, eff_batch={eff_batch}, lr={args.lr}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted["train"],
        eval_dataset=formatted["validation"],
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps if args.max_steps > 0 else -1,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=args.seed,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
    )

    trainer.train()

    # ---- Save adapter ----
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to {adapter_dir}")

    # Save training metadata
    meta = {
        "base_model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "max_steps": args.max_steps,
        "train_examples": len(dataset["train"]),
        "val_examples": len(dataset["validation"]),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Upload to HuggingFace ----
    if not args.no_upload and args.hf_repo:
        from huggingface_hub import HfApi, create_repo

        print(f"\nUploading adapter to HuggingFace: {args.hf_repo}")
        api = HfApi(token=args.token or os.environ.get("HF_TOKEN"))
        try:
            create_repo(args.hf_repo, repo_type="model",
                        token=args.token or True, exist_ok=True)
        except Exception as e:
            print(f"Warning: create_repo failed ({e}), attempting upload anyway")

        api.upload_folder(
            repo_id=args.hf_repo,
            folder_path=str(adapter_dir),
            repo_type="model",
            commit_message=f"LoRA adapter: {args.model} trained {len(dataset['train'])} examples",
        )
        print(f"Uploaded: https://huggingface.co/{args.hf_repo}")

    print("\nDone.")
    print(f"Next: python qrel_generation/inference.py "
          f"--judge-model {args.model} --provider hf "
          f"--lora-adapter {adapter_dir} --split test")


if __name__ == "__main__":
    main()
