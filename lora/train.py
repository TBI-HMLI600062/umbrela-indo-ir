"""
LoRA fine-tuning of Qwen2.5-7B-Instruct as LLM relevance judge (E8T2).

Uses PEFT library for parameter-efficient fine-tuning on human-labeled
query-passage pairs from MIRACL-ID. Training signal: human qrels (binary)
mapped to 0/3 target scores in the judge's ##final score: X output format.
Loss is computed only on response tokens (prompt tokens are masked).

Key features:
  - Auto-resume from latest checkpoint (optimizer + scheduler state restored)
  - Validation loss evaluated at every checkpoint (requires --val-data)
  - checkpoint-best/ saved whenever val_loss improves
  - Step checkpoints pruned to --save-total-limit (epoch checkpoints always kept)
  - Per-step loss (step_loss) and running epoch average both logged to train_log.jsonl
  - Training logs + metadata pushed to HuggingFace Hub alongside the adapter

Args:
    --training-data   path to data directory containing train.jsonl
    --model           base model ID (default: Qwen/Qwen2.5-7B-Instruct)
    --output          output directory for LoRA adapter
    --epochs          number of training epochs (default: 3)
    --batch-size      per-device batch size (default: 2)
    --grad-accum      gradient accumulation steps (default: 8; effective batch = 16)
    --lr              learning rate (default: 2e-4)
    --lora-r          LoRA rank (default: 16)
    --lora-alpha      LoRA alpha (default: 32)
    --max-length      max sequence length in tokens (default: 1024)
    --max-steps       max gradient steps; 0=no limit. Use for smoke tests.
    --val-data        val data directory (val.jsonl inside); enables val loss + checkpoint-best
    --save-steps      save checkpoint every N gradient steps; 0=epoch-only (default: 500)
    --save-total-limit  max step checkpoints to keep; 0=unlimited (default: 3)
    --no-resume       disable auto-resume from latest checkpoint
    --push-to-hub     push final adapter + logs to HuggingFace Hub
    --hub-model-id    HF repo ID (e.g. umbrella_ir/qwen-lora-miracl-id-judge)
    --hub-public      make HF repo public (default: private)

Example — 1-epoch with resume + val + HF push:
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --val-data      results/lora_data/qwen/ \\
        --output        results/models/lora_qwen_1ep/ \\
        --epochs 1 --save-steps 500 --save-total-limit 3 \\
        --push-to-hub --hub-model-id umbrella_ir/qwen-lora-miracl-id-judge

Smoke test (resume test: run 15 steps, kill at 10, restart — should resume from step 10):
    python lora/train.py \\
        --training-data results/lora_data/qwen/ \\
        --output results/models/lora_qwen_smoke/ \\
        --epochs 1 --max-steps 15 --save-steps 5 --save-total-limit 2
"""

import argparse
import itertools
import json
import shutil
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
                        help="Save checkpoint every N gradient steps; 0=epoch-only (default: 500)")
    parser.add_argument("--save-total-limit", type=int, default=3,
                        help="Max step checkpoints to keep; 0=unlimited (default: 3). "
                             "Epoch and best checkpoints are never pruned.")
    parser.add_argument("--eval-steps", action="store_true",
                        help="Also evaluate val loss at every step checkpoint (default: epoch-end only)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable auto-resume from latest checkpoint")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push final adapter + logs to HuggingFace Hub")
    parser.add_argument("--hub-model-id", default=None,
                        help="HF repo ID for --push-to-hub")
    parser.add_argument("--hub-public", action="store_true",
                        help="Make HF repo public (default: private)")
    return parser.parse_args()


def load_examples(data_dir: Path, filename: str = "train.jsonl") -> list:
    path = data_dir / filename
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def tokenize_example(example: dict, tokenizer, max_length: int):
    """Tokenize one SFT example. Prompt tokens are masked (-100) so loss is on response only."""
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


def find_latest_checkpoint(output_dir: Path):
    """
    Scan output_dir for checkpoint dirs that have optimizer.pt.
    Return (ckpt_path, state_dict) for the one with the highest global_step,
    or (None, None) if none found.
    """
    import torch
    best_path = None
    best_state = None
    best_step = -1

    if not output_dir.exists():
        return None, None

    for ckpt_dir in output_dir.iterdir():
        if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith("checkpoint-"):
            continue
        opt_path = ckpt_dir / "optimizer.pt"
        adapter_path = ckpt_dir / "adapter_model.safetensors"
        if not opt_path.exists() or not adapter_path.exists():
            continue
        try:
            state = torch.load(opt_path, map_location="cpu", weights_only=False)
            step = state.get("global_step", -1)
            if step > best_step:
                best_step = step
                best_path = ckpt_dir
                best_state = state
        except Exception:
            continue

    return best_path, best_state


def compute_val_loss(model, val_loader) -> float:
    """Compute average val loss over the full val set."""
    import torch
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(**batch)
            total_loss += outputs.loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def prune_step_checkpoints(output_dir: Path, save_total_limit: int):
    """Delete the oldest step checkpoints (checkpoint-{N}) beyond save_total_limit.
    Epoch checkpoints (checkpoint-epochN) and checkpoint-best are never deleted."""
    if save_total_limit <= 0:
        return
    step_ckpts = []
    for d in output_dir.iterdir():
        name = d.name
        if d.is_dir() and name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit():
            step_ckpts.append((int(name[len("checkpoint-"):]), d))
    step_ckpts.sort(key=lambda x: x[0])
    to_delete = step_ckpts[:-save_total_limit] if len(step_ckpts) > save_total_limit else []
    for _, d in to_delete:
        shutil.rmtree(d)
        print(f"  Pruned checkpoint: {d.name}")


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
    from peft import LoraConfig, get_peft_model, PeftModel, TaskType
    from tqdm import tqdm

    random.seed(42)
    torch.manual_seed(42)

    # ── Data ────────────────────────────────────────────────────────────────────
    data_dir = Path(args.training_data)
    print(f"Loading training data from {data_dir}...")
    train_examples = load_examples(data_dir)
    print(f"  {len(train_examples):,} examples")

    if args.max_steps > 0:
        cap = args.max_steps * args.batch_size * args.grad_accum * 2
        train_examples = train_examples[:cap]
        print(f"  Smoke test: capped to {len(train_examples)} examples")

    # Deterministic pre-shuffle (seed 42 set above). DataLoader will use shuffle=False
    # so position-based resume can skip the right number of batches.
    random.shuffle(train_examples)

    val_examples = None
    if args.val_data:
        val_examples = load_examples(Path(args.val_data), filename="val.jsonl")
        print(f"  {len(val_examples):,} val examples")

    # ── Resume detection ────────────────────────────────────────────────────────
    resume_ckpt_path = None
    resume_state = None
    if not args.no_resume:
        resume_ckpt_path, resume_state = find_latest_checkpoint(output_dir)
        if resume_ckpt_path is not None:
            print(f"\nFound checkpoint: {resume_ckpt_path.name} "
                  f"(global_step={resume_state['global_step']}, "
                  f"next_epoch={resume_state['next_epoch']}, "
                  f"skip_batches={resume_state['skip_batches']})")

    # ── Model ───────────────────────────────────────────────────────────────────
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

    if resume_ckpt_path is not None:
        print(f"Loading LoRA adapter from {resume_ckpt_path.name}...")
        model = PeftModel.from_pretrained(base_model, str(resume_ckpt_path), is_trainable=True)
    else:
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

    # Gradient checkpointing saves ~10GB VRAM at ~30% compute overhead.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # ── Tokenize ─────────────────────────────────────────────────────────────────
    print("Tokenizing examples...")
    tokenized = []
    for ex in tqdm(train_examples, desc="Tokenizing"):
        input_ids, labels = tokenize_example(ex, tokenizer, args.max_length)
        tokenized.append((input_ids, labels))

    val_tokenized = None
    if val_examples is not None:
        print("Tokenizing val examples...")
        val_tokenized = []
        for ex in tqdm(val_examples, desc="Tokenizing val"):
            input_ids, labels = tokenize_example(ex, tokenizer, args.max_length)
            val_tokenized.append((input_ids, labels))

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

    # shuffle=False: we pre-shuffled with seed 42, so position is deterministic for resume
    loader = DataLoader(
        SFTDataset(tokenized), batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    val_loader = None
    if val_tokenized is not None:
        val_loader = DataLoader(
            SFTDataset(val_tokenized), batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn
        )

    # ── Optimizer + Scheduler ────────────────────────────────────────────────────
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
    start_epoch = 1
    skip_batches = 0

    if resume_state is not None:
        global_step = resume_state["global_step"]
        start_epoch = resume_state["next_epoch"]
        skip_batches = resume_state["skip_batches"]
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        # Move optimizer tensor states to the right device
        for param_state in optimizer.state.values():
            for k, v in param_state.items():
                if isinstance(v, torch.Tensor):
                    param_state[k] = v.to(device)
        print(f"Resumed: global_step={global_step}, start_epoch={start_epoch}, "
              f"skip_batches={skip_batches}")

    optimizer.zero_grad()

    # ── Logging + checkpoint helpers ─────────────────────────────────────────────
    train_log_path = output_dir / "train_log.jsonl"
    log_mode = "a" if resume_state is not None else "w"
    train_log_f = open(train_log_path, log_mode)

    best_val_loss = float("inf")

    def save_checkpoint(tag: str, epoch: int, next_epoch: int,
                        skip_for_next: int, val_loss: float = None):
        ckpt_dir = output_dir / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        torch.save({
            "global_step":  global_step,
            "next_epoch":   next_epoch,
            "skip_batches": skip_for_next,
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
        }, ckpt_dir / "optimizer.pt")
        info = f"  Checkpoint saved → {ckpt_dir.name}  (step={global_step})"
        if val_loss is not None:
            info += f"  val_loss={val_loss:.4f}"
        print(info)

    def maybe_eval_val(context_tag: str) -> float:
        """Compute val loss if val_loader exists. Also updates checkpoint-best."""
        nonlocal best_val_loss
        if val_loader is None:
            return None
        val_loss = compute_val_loss(model, val_loader)
        print(f"  [{context_tag}] val_loss={val_loss:.4f}  (best={best_val_loss:.4f})")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_dir = output_dir / "checkpoint-best"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))
            # Save a small meta file so we know which step this came from
            (best_dir / "best_meta.json").write_text(
                json.dumps({"global_step": global_step, "val_loss": round(val_loss, 6)}, indent=2)
            )
            print(f"  New best checkpoint saved → checkpoint-best (val_loss={val_loss:.4f})")
        return val_loss

    # ── Training loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        step_accum_loss = 0.0  # accumulates over grad_accum micro-batches → step_loss

        # On the resume epoch, skip already-processed batches
        this_epoch_skip = skip_batches if epoch == start_epoch else 0
        epoch_iter = iter(loader)
        if this_epoch_skip > 0:
            print(f"  Epoch {epoch}: skipping first {this_epoch_skip} batches (resume)...")
            epoch_iter = itertools.islice(epoch_iter, this_epoch_skip, None)

        pbar = tqdm(epoch_iter, desc=f"Epoch {epoch}/{args.epochs}", unit="batch",
                    initial=this_epoch_skip, total=len(loader))

        for step, batch in enumerate(pbar, start=this_epoch_skip):
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            epoch_loss      += outputs.loss.item()
            step_accum_loss += outputs.loss.item()
            n_batches       += 1

            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                step_loss  = step_accum_loss / args.grad_accum
                step_accum_loss = 0.0

                log_entry = {
                    "step":         global_step,
                    "epoch":        epoch,
                    "step_loss":    round(step_loss, 6),
                    "running_loss": round(epoch_loss / n_batches, 6),
                    "lr":           round(current_lr, 8),
                }

                # Step-level checkpoint (val eval only if --eval-steps is set)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    val_loss = maybe_eval_val(f"step {global_step}") if args.eval_steps else None
                    if val_loss is not None:
                        log_entry["val_loss"] = round(val_loss, 6)
                    save_checkpoint(
                        tag=f"checkpoint-{global_step}",
                        epoch=epoch, next_epoch=epoch,
                        skip_for_next=(step + 1),
                        val_loss=val_loss,
                    )
                    prune_step_checkpoints(output_dir, args.save_total_limit)

                train_log_f.write(json.dumps(log_entry) + "\n")
                train_log_f.flush()

                if global_step % 50 == 0:
                    pbar.set_postfix(loss=f"{step_loss:.4f}", step=global_step)

                if args.max_steps > 0 and global_step >= args.max_steps:
                    break

        avg_epoch_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch}: avg_loss={avg_epoch_loss:.4f}")

        # Epoch-end checkpoint (always kept, never pruned)
        val_loss = maybe_eval_val(f"epoch {epoch} end")
        epoch_log = {
            "epoch_end":   epoch,
            "global_step": global_step,
            "avg_loss":    round(avg_epoch_loss, 6),
        }
        if val_loss is not None:
            epoch_log["val_loss"] = round(val_loss, 6)
        train_log_f.write(json.dumps(epoch_log) + "\n")
        train_log_f.flush()

        save_checkpoint(
            tag=f"checkpoint-epoch{epoch}",
            epoch=epoch, next_epoch=epoch + 1,
            skip_for_next=0,
            val_loss=val_loss,
        )

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    train_log_f.close()
    print(f"Training log saved to {train_log_path}")

    # ── Final adapter save ───────────────────────────────────────────────────────
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
        "val_data":         str(args.val_data) if args.val_data else None,
        "best_val_loss":    round(best_val_loss, 6) if best_val_loss < float("inf") else None,
    }
    with open(output_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nLoRA adapter saved to {output_dir}")

    # ── HuggingFace Hub push ─────────────────────────────────────────────────────
    if args.push_to_hub:
        if not args.hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is set")
        private = not args.hub_public
        print(f"\nPushing to HuggingFace Hub: {args.hub_model_id} (private={private}) ...")
        model.push_to_hub(args.hub_model_id, private=private)
        tokenizer.push_to_hub(args.hub_model_id, private=private)

        from huggingface_hub import HfApi
        api = HfApi()
        for fname in ["training_meta.json", "train_log.jsonl"]:
            fpath = output_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=args.hub_model_id,
                    repo_type="model",
                )
                print(f"  Uploaded {fname}")

        visibility = "public" if args.hub_public else "private"
        print(f"Pushed ({visibility}): https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
