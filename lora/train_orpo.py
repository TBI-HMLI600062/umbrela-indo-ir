"""
ORPO fine-tuning of Qwen2.5-7B-Instruct using TRL's ORPOTrainer.

Combines SFT + preference loss in one pass without a reference model (same as
the manual implementation), but delegates loss computation and training loop
to TRL for cleaner code and built-in logging.

Args:
    --training-data     Directory with train.jsonl (from prepare_orpo_data.py)
    --model             Base model HF ID (default: Qwen/Qwen2.5-7B-Instruct)
    --output            Output directory for LoRA adapter
    --epochs            Training epochs (default: 3)
    --batch-size        Per-device batch size (default: 2)
    --grad-accum        Gradient accumulation steps (default: 8; effective batch=16)
    --lr                Learning rate (default: 2e-4)
    --lambda-orpo       ORPO β weight for preference loss (default: 0.1)
    --lora-r            LoRA rank (default: 16)
    --lora-alpha        LoRA alpha (default: 32)
    --max-length        Max total sequence length (default: 1024)
    --max-steps         Max gradient steps; 0=no limit (for smoke test)

Both RTX 5090 32GB and RTX 4090 24GB: --batch-size 2 --grad-accum 8 (default)

Example:
    python lora/train_orpo.py \\
        --training-data results/orpo_data/qwen/ \\
        --output results/models/orpo_qwen/ \\
        --epochs 3

Smoke test:
    python lora/train_orpo.py \\
        --training-data results/orpo_data/qwen/ \\
        --output results/models/orpo_qwen_smoke/ \\
        --max-steps 10
"""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--model",       default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output",      required=True)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--batch-size",  type=int,   default=2)
    parser.add_argument("--grad-accum",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--lambda-orpo", type=float, default=0.1,
                        help="ORPO β weight for preference loss (default: 0.1)")
    parser.add_argument("--lora-r",      type=int,   default=16)
    parser.add_argument("--lora-alpha",  type=int,   default=32)
    parser.add_argument("--max-length",  type=int,   default=1024)
    parser.add_argument("--max-steps",   type=int,   default=0,
                        help="Max gradient steps; 0=no limit (use for smoke test)")
    return parser.parse_args()


def load_jsonl(path):
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def to_conversational(example):
    """Convert flat prompt/chosen/rejected strings to chat message lists.

    TRL applies the model's chat template when given message lists, matching
    how inference_vllm.py formats prompts at eval time.
    """
    return {
        "prompt":   [{"role": "user",      "content": example["prompt"]}],
        "chosen":   [{"role": "assistant", "content": example["chosen"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, TaskType
    from trl import ORPOTrainer, ORPOConfig

    data_dir = Path(args.training_data)
    print(f"Loading training data from {data_dir}...")
    train_data = load_jsonl(data_dir / "train.jsonl")
    print(f"  {len(train_data):,} preference pairs")

    train_dataset = Dataset.from_list(train_data).map(
        to_conversational, remove_columns=["prompt", "chosen", "rejected"]
    )

    eval_dataset = None
    val_path = data_dir / "val.jsonl"
    if val_path.exists():
        val_data = load_jsonl(val_path)
        eval_dataset = Dataset.from_list(val_data).map(
            to_conversational, remove_columns=["prompt", "chosen", "rejected"]
        )
        print(f"  {len(val_data):,} val pairs")

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # TRL 0.13 accesses model.warnings_issued which was removed in transformers 5.x
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    max_steps = args.max_steps if args.max_steps > 0 else -1
    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum} (effective={eff_batch}), "
          f"lr={args.lr}, beta={args.lambda_orpo}, lora_r={args.lora_r}")

    orpo_cfg = ORPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.lambda_orpo,
        max_length=args.max_length,
        max_prompt_length=args.max_length - 32,  # response is very short (~5 tokens)
        max_completion_length=32,
        max_steps=max_steps,
        gradient_checkpointing=True,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        logging_steps=50,
        save_strategy="no",
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_cfg,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_cfg,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "method":           "orpo_trl",
        "trl_version":      __import__("trl").__version__,
        "base_model":       args.model,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  eff_batch,
        "lr":               args.lr,
        "lambda_orpo":      args.lambda_orpo,
        "max_steps":        args.max_steps,
        "n_train_examples": len(train_data),
        "training_data":    str(args.training_data),
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLoRA adapter (ORPO/TRL) saved to {output_dir}")


if __name__ == "__main__":
    main()
