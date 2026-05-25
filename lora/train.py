"""
LoRA fine-tuning of Qwen2.5-7B-Instruct as LLM relevance judge (E8T2).

Fine-tunes the model on human-labeled query-passage pairs from MIRACL-ID.
Training signal: human qrels (binary) mapped to 0/3 target scores in the
judge's "##final score: X" output format.

Stack:
  - Unsloth  → fast 4-bit (QLoRA) model loading + fused kernels + gradient
               checkpointing. Roughly 2x faster, ~half the VRAM vs vanilla HF.
  - TRL SFTTrainer → the training loop itself: checkpointing, auto-resume,
               validation, best-checkpoint tracking, LR schedule, HF Hub push.
               (Unsloth does NOT provide a training loop; SFTTrainer is the
               recommended pairing.)

Loss is computed on the assistant response only — the prompt is masked via
DataCollatorForCompletionOnlyLM, which zeroes the labels of every token before
the assistant turn marker. This matches the old manual loop's masking.

The CLI is unchanged from the manual-loop version, so setup_and_train.sh and
existing invocations keep working.

Args:
    --training-data   path to data directory containing train.jsonl
    --model           base model ID (default: Qwen/Qwen2.5-7B-Instruct)
    --output          output directory for LoRA adapter + checkpoints
    --epochs          number of training epochs (default: 3)
    --batch-size      per-device batch size (default: 2)
    --grad-accum      gradient accumulation steps (default: 8; effective batch=16)
    --lr              learning rate (default: 2e-4)
    --lora-r          LoRA rank (default: 16)
    --lora-alpha      LoRA alpha (default: 32)
    --max-length      max sequence length in tokens (default: 1024)
    --max-steps       max gradient steps; 0=no limit. Use for smoke tests.
    --val-data        val data directory (val.jsonl inside); enables val loss +
                      load-best-model-at-end
    --save-steps      save checkpoint every N steps; 0=epoch-only (default: 500)
    --save-total-limit  max checkpoints to keep; 0=unlimited (default: 3)
    --eval-steps      evaluate val loss every --save-steps steps (default: epoch-end only)
    --no-resume       disable auto-resume from latest checkpoint
    --no-4bit         disable 4-bit quantization (full bf16 LoRA, more VRAM)
    --push-to-hub     push final adapter + logs to HuggingFace Hub
    --hub-model-id    HF repo ID (e.g. umbrella_ir/qwen-lora-miracl-id-judge)
    --hub-public      make HF repo public (default: private)

Example — 1-epoch with val + HF push:
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --val-data      results/lora_data/qwen/ \\
        --output        results/models/lora_qwen_1ep/ \\
        --epochs 1 --save-steps 500 --save-total-limit 3 \\
        --push-to-hub --hub-model-id umbrella_ir/qwen-lora-miracl-id-judge

Smoke test (15 steps):
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --output results/models/lora_qwen_smoke/ \\
        --epochs 1 --max-steps 15 --save-steps 5 --save-total-limit 2
"""

import argparse
import inspect
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-7B-Instruct as LLM judge (Unsloth + TRL SFTTrainer)."
    )
    parser.add_argument("--training-data", required=True,
                        help="Training data directory (contains train.jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model HF ID")
    parser.add_argument("--output", required=True,
                        help="Output directory for LoRA adapter + checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Per-device batch size (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8; effective batch=16)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Max gradient steps; 0=no limit (use for smoke test)")
    parser.add_argument("--val-data", default=None,
                        help="Val data directory (val.jsonl inside); enables val loss tracking")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps; 0=epoch-only (default: 500)")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Max checkpoints to keep; 0=unlimited (default: 3). "
                             "Best checkpoint is always kept when --val-data is set.")
    parser.add_argument("--eval-steps", action="store_true",
                        help="Evaluate val loss every --save-steps steps (default: epoch-end only)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable auto-resume from latest checkpoint")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization (full bfloat16 LoRA, uses more VRAM)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push final adapter + logs to HuggingFace Hub")
    parser.add_argument("--hub-model-id", default=None,
                        help="HF repo ID for --push-to-hub")
    parser.add_argument("--hub-public", action="store_true",
                        help="Make HF repo public (default: private)")
    return parser.parse_args()


def load_examples(data_dir: Path, filename: str = "train.jsonl") -> list:
    path = data_dir / filename
    with open(path) as f:
        return [json.loads(line) for line in f]


def build_dataset(examples: list):
    """Emit TRL prompt-completion conversational format: separate `prompt` and
    `completion` message lists. SFTTrainer applies the chat template itself and,
    because the dataset is prompt-completion, defaults completion_only_loss=True —
    it builds a completion_mask that zeroes the prompt tokens, so loss lands on
    the assistant response only (same intent as the old collator masking)."""
    from datasets import Dataset

    def render(ex):
        return {
            "prompt": [{"role": "user", "content": ex["prompt"]}],
            "completion": [{"role": "assistant", "content": ex["response"]}],
        }

    return Dataset.from_list([render(ex) for ex in examples])


def filter_supported(target, kwargs: dict) -> dict:
    """Drop kwargs the installed version of `target` doesn't accept. Guards against
    parameters that were renamed across TRL/transformers releases."""
    valid = set(inspect.signature(target).parameters)
    return {k: v for k, v in kwargs.items() if k in valid}


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig

    # ── Model (Unsloth) ──────────────────────────────────────────────────────────
    print(f"Loading model: {args.model}  (4bit={not args.no_4bit})")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_length,
        dtype=None,                       # auto-detect bf16/fp16
        load_in_4bit=not args.no_4bit,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    model.print_trainable_parameters()

    # ── Data ───────────────────────────────────────────────────────────────────
    print(f"Loading training data from {args.training_data}...")
    train_examples = load_examples(Path(args.training_data))
    if args.max_steps > 0:
        cap = args.max_steps * args.batch_size * args.grad_accum * 2
        train_examples = train_examples[:cap]
        print(f"  Smoke test: capped to {len(train_examples)} examples")
    print(f"  {len(train_examples):,} train examples")
    train_ds = build_dataset(train_examples)

    eval_ds = None
    if args.val_data:
        val_examples = load_examples(Path(args.val_data), filename="val.jsonl")
        print(f"  {len(val_examples):,} val examples")
        eval_ds = build_dataset(val_examples)

    # ── Trainer config ───────────────────────────────────────────────────────────
    has_val = eval_ds is not None
    # load_best_model_at_end requires eval and save strategies to match.
    if has_val and args.eval_steps and args.save_steps > 0:
        eval_strategy = save_strategy = "steps"
    elif has_val:
        eval_strategy = save_strategy = "epoch"
    else:
        eval_strategy = "no"
        save_strategy = "steps" if args.save_steps > 0 else "epoch"

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nTraining: epochs={args.epochs}, batch={args.batch_size}, "
          f"grad_accum={args.grad_accum} (effective={eff_batch}), "
          f"lr={args.lr}, lora_r={args.lora_r}")

    cfg_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        optim="adamw_torch",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=1,
        save_strategy=save_strategy,
        save_steps=args.save_steps if args.save_steps > 0 else 500,
        save_total_limit=args.save_total_limit if args.save_total_limit > 0 else None,
        eval_strategy=eval_strategy,
        eval_steps=args.save_steps if args.save_steps > 0 else None,
        load_best_model_at_end=has_val,
        metric_for_best_model="eval_loss" if has_val else None,
        greater_is_better=False,
        seed=42,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_private_repo=not args.hub_public,
        # SFT-specific. max_seq_length was renamed to max_length in newer TRL;
        # filter_supported() keeps whichever the installed version accepts.
        max_seq_length=args.max_length,
        max_length=args.max_length,
        # Dataset is prompt-completion → completion_only_loss defaults to True,
        # masking the prompt. No dataset_text_field / collator needed.
        packing=False,
    )
    sft_config = SFTConfig(**filter_supported(SFTConfig, cfg_kwargs))

    trainer_kwargs = filter_supported(SFTTrainer.__init__, dict(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,   # renamed from `tokenizer` in newer TRL
        tokenizer=tokenizer,
    ))
    trainer = SFTTrainer(**trainer_kwargs)

    # ── Resume detection ─────────────────────────────────────────────────────────
    resume = False
    if not args.no_resume and any(output_dir.glob("checkpoint-*")):
        resume = True
        print("Resuming from latest checkpoint in", output_dir)

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=resume)

    # ── Save adapter + metadata ──────────────────────────────────────────────────
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    meta = {
        "base_model":       args.model,
        "lora_r":           args.lora_r,
        "lora_alpha":       args.lora_alpha,
        "epochs":           args.epochs,
        "batch_size":       args.batch_size,
        "grad_accum":       args.grad_accum,
        "effective_batch":  eff_batch,
        "lr":               args.lr,
        "max_steps":        args.max_steps,
        "load_in_4bit":     not args.no_4bit,
        "n_train_examples": len(train_examples),
        "training_data":    str(args.training_data),
        "val_data":         str(args.val_data) if args.val_data else None,
        "best_metric":      trainer.state.best_metric,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nLoRA adapter saved to {output_dir}")

    # ── HuggingFace Hub push ──────────────────────────────────────────────────────
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set")
        visibility = "public" if args.hub_public else "private"
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id} ({visibility}) ...")
        trainer.push_to_hub()
        print(f"Pushed: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
