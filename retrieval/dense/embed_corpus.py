"""
Encode MIRACL-ID corpus with Qwen3-Embedding-4B via vLLM and build per-chunk FAISS indexes.

Processes the corpus in shards of --chunk-size docs. Each shard is encoded,
saved as index_chunk_X.faiss + docids_chunk_X.npy, optionally uploaded to a
HuggingFace dataset repo subfolder, then deleted locally to keep disk usage low.
Script is resume-safe — existing chunk files are skipped automatically.

Args:
    --model           HF encoder model ID (default: Qwen/Qwen3-Embedding-4B)
    --corpus          corpus JSONL file (default: data/miracl-id/corpus/corpus.jsonl)
    --output          output directory for chunk files
    --chunk-size      documents per shard (default: 300000)
    --tensor-parallel number of GPUs for tensor parallelism (default: all available)
    --gpu-mem-util    vLLM GPU memory utilization 0.0–1.0 (default: 0.90)
    --max-model-len   max token length passed to vLLM (default: 8192)
    --hf-repo         HuggingFace dataset repo ID; if set, each chunk is uploaded
                      then deleted locally (requires HF_TOKEN env var)
    --hf-folder       subfolder inside hf-repo for this model's chunks (default: qwen3-embed-4b)
    --smoke-docs      smoke-test mode: encode only N docs, print diagnostics, exit without saving

Full run (Karol — Qwen3-Embedding-4B, multi-GPU + upload):
    nohup python retrieval/dense/embed_corpus.py \\
        --model Qwen/Qwen3-Embedding-4B \\
        --output embeddings/qwen3-embed-4b/ \\
        --hf-repo karolinajocelyn/umbrela-indo-ir \\
        --hf-folder qwen3-embed-4b \\
        > log_qwen3_encode.txt 2>&1 &

Smoke test (sanity check before full run, ~1 min):
    python retrieval/dense/embed_corpus.py \\
        --model Qwen/Qwen3-Embedding-4B \\
        --output embeddings/qwen3-embed-4b/ \\
        --smoke-docs 32

Output files per chunk X:
    {output}/index_chunk_{X}.faiss   — FAISS IndexFlatIP (cosine sim, L2-normalised)
    {output}/docids_chunk_{X}.npy    — numpy string array; docids[i] maps to vector i
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Encode corpus with Qwen3-Embedding via vLLM + FAISS.")
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-4B", help="HF encoder model ID")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for chunk files")
    parser.add_argument("--chunk-size", type=int, default=300_000,
                        help="Documents per shard (default: 300000)")
    parser.add_argument("--tensor-parallel", type=int, default=None,
                        help="GPUs for tensor parallelism (default: all available)")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="vLLM GPU memory utilization (default: 0.90)")
    parser.add_argument("--max-model-len", type=int, default=8192,
                        help="Max token length for vLLM (default: 8192)")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace dataset repo ID for upload + local cleanup")
    parser.add_argument("--hf-folder", default="qwen3-embed-4b",
                        help="Subfolder inside hf-repo for this model's chunks (default: qwen3-embed-4b)")
    parser.add_argument("--smoke-docs", type=int, default=None,
                        help="Smoke-test mode: encode only N docs, print diagnostics, exit without saving")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def load_llm(args):
    from vllm import LLM

    n_gpus = args.tensor_parallel or torch.cuda.device_count() or 1
    print(f"Loading model  : {args.model}")
    print(f"Tensor parallel: {n_gpus} GPU(s)")
    print(f"GPU mem util   : {args.gpu_mem_util}")
    print(f"Max model len  : {args.max_model_len}")

    return LLM(
        model=args.model,
        runner="pooling",   # vLLM >=0.21: replaces task="embed"
        convert="embed",
        tensor_parallel_size=n_gpus,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
        enforce_eager=False,
    )


def _truncate_to_max_tokens(tokenizer, text: str, content_limit: int) -> str:
    """Truncate to content_limit tokens, excluding special tokens (BOS/EOS added by vLLM later)."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= content_limit:
        return text
    return tokenizer.decode(ids[:content_limit], skip_special_tokens=False)


def embed_texts(llm, texts: list[str], max_tokens: int = 8192) -> np.ndarray:
    """Encode texts with vLLM; return L2-normalised float32 array (vLLM normalises internally)."""
    # Reserve 2 slots for BOS/EOS that vLLM adds; truncate content to the remainder.
    # Char threshold = content_limit: a doc with ≤ content_limit chars cannot exceed
    # content_limit tokens even in the worst case (1 char = 1 token).
    content_limit = max_tokens - 2
    tokenizer = llm.get_tokenizer()
    texts = [
        _truncate_to_max_tokens(tokenizer, t, content_limit) if len(t) > content_limit else t
        for t in texts
    ]

    # Sort by length so vLLM can bucket similar-length sequences together,
    # reducing wasted padding even within its continuous batching scheduler.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    sorted_texts = [texts[i] for i in order]

    outputs = llm.embed(sorted_texts)
    embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)

    # Restore original order
    restore = [0] * len(order)
    for new_i, orig_i in enumerate(order):
        restore[orig_i] = new_i
    return embs[restore]


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------

def encode_chunk(
    chunk_idx: int,
    docs: list[str],
    docids: list[str],
    llm,
    out_dir: Path,
    max_tokens: int = 8192,
) -> tuple[Path, Path]:
    t0 = time.time()
    print(f"  Encoding {len(docs):,} docs...")

    embeddings = embed_texts(llm, docs, max_tokens)

    # Save as float16 numpy — half the size of FAISS float32 (~1.5 GB vs ~3 GB
    # per 300k-doc chunk at dim=2560). FAISS index is built on the fly at
    # retrieval time from these arrays.
    emb_path = out_dir / f"embeddings_chunk_{chunk_idx}.fp16.npy"
    docids_path = out_dir / f"docids_chunk_{chunk_idx}.npy"
    np.save(emb_path, embeddings.astype(np.float16))
    np.save(docids_path, np.array(docids))

    elapsed = time.time() - t0
    rate = len(docs) / max(elapsed, 1e-6)
    size_gb = emb_path.stat().st_size / 1e9
    print(
        f"  Chunk {chunk_idx}: {len(docs):,} vectors | dim={embeddings.shape[1]} | "
        f"{size_gb:.2f} GB | {elapsed:.0f}s ({rate:.0f} docs/s)"
    )
    return emb_path, docids_path


def upload_and_clean(api, hf_repo: str, hf_folder: str, emb_path: Path, docids_path: Path, chunk_idx: int):
    """Upload both chunk files to HF subfolder in parallel, then delete locally."""
    import concurrent.futures

    def _upload(path: Path):
        repo_path = f"{hf_folder}/{path.name}"
        size_gb = path.stat().st_size / 1e9
        print(f"  Uploading {path.name} ({size_gb:.2f} GB) → {hf_repo}/{repo_path} ...")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=hf_repo,
            repo_type="dataset",
        )
        print(f"  {path.name} done in {time.time() - t0:.0f}s")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_upload, p) for p in (emb_path, docids_path)]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()  # propagate any upload exception immediately

    for path in (emb_path, docids_path):
        os.remove(path)
    # Leave a marker so the resume logic can skip this chunk even after local files are deleted.
    (emb_path.parent / f"chunk_{chunk_idx}.done").touch()
    print(f"  Chunk {chunk_idx} uploaded and local files removed.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(args, llm):
    """Encode --smoke-docs documents, print diagnostics, exit. No files saved."""
    print(f"\n{'='*60}")
    print(f"SMOKE TEST — {args.smoke_docs} docs from {args.corpus}")
    print(f"{'='*60}")

    corpus_path = Path(args.corpus)
    docs, docids = [], []
    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            docids.append(row["docid"])
            docs.append(row.get("doc", row.get("text", "")))
            if len(docs) == args.smoke_docs:
                break

    if not docs:
        print("ERROR: corpus is empty or not found.")
        return

    print(f"Loaded {len(docs)} docs. Encoding...")
    t0 = time.time()
    embs = embed_texts(llm, docs, args.max_model_len)
    elapsed = time.time() - t0

    norms = np.linalg.norm(embs, axis=1)
    sim_matrix = embs @ embs.T

    print(f"\n--- Results ---")
    print(f"Shape          : {embs.shape}")
    print(f"Dtype          : {embs.dtype}")
    print(f"Elapsed        : {elapsed:.1f}s  ({len(docs)/elapsed:.1f} docs/s)")
    print(f"Norm (mean/min/max): {norms.mean():.4f} / {norms.min():.4f} / {norms.max():.4f}  (expect ~1.0)")
    print(f"Self-sim diag  : {np.diag(sim_matrix).mean():.4f}  (expect ~1.0)")
    if len(docs) >= 2:
        off_diag = sim_matrix[np.triu_indices(len(docs), k=1)]
        print(f"Cross-sim mean : {off_diag.mean():.4f}  (expect < 1.0)")
        print(f"Cross-sim max  : {off_diag.max():.4f}")

    print(f"\nSample docids  : {docids[:3]}")
    print(f"Sample doc[0]  : {docs[0][:120]}...")

    all_ok = (
        embs.shape[1] > 0
        and norms.mean() > 0.99
        and np.diag(sim_matrix).mean() > 0.99
    )
    print(f"\n{'SMOKE TEST PASSED' if all_ok else 'SMOKE TEST FAILED'}")
    if not all_ok:
        print("  Check: norms should be ~1.0 and self-sim should be ~1.0")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _count_lines(path: Path) -> int:
    """Fast line count without parsing JSON."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def main():
    args = parse_args()

    llm = load_llm(args)

    if args.smoke_docs:
        run_smoke_test(args, llm)
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional HF upload setup
    api = None
    if args.hf_repo:
        # hf_transfer is a Rust-based uploader (pip install hf_transfer) that splits
        # large files into parallel chunks — typically 5–10x faster than pure-Python.
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("--hf-repo requires HF_TOKEN environment variable to be set.")
        api = HfApi(token=hf_token)
        try:
            import hf_transfer  # noqa: F401
            print(f"HuggingFace upload enabled → {args.hf_repo}/{args.hf_folder}/  [hf_transfer active]")
        except ImportError:
            print(f"HuggingFace upload enabled → {args.hf_repo}/{args.hf_folder}/  "
                  f"[tip: pip install hf_transfer for faster uploads]")

    corpus_path = Path(args.corpus)
    print(f"\nCounting docs in {corpus_path} ...")
    total_docs = _count_lines(corpus_path)
    n_chunks = math.ceil(total_docs / args.chunk_size)
    print(f"Corpus: {total_docs:,} docs → {n_chunks} chunk(s) of {args.chunk_size:,}")

    chunk_docs: list[str] = []
    chunk_docids: list[str] = []
    chunk_idx = 0
    docs_done = 0
    run_start = time.time()

    chunk_bar = tqdm(total=n_chunks, desc="Chunks", unit="chunk", position=0)

    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            chunk_docids.append(row["docid"])
            chunk_docs.append(row.get("doc", row.get("text", "")))

            if len(chunk_docs) == args.chunk_size:
                emb_path = out_dir / f"embeddings_chunk_{chunk_idx}.fp16.npy"
                docids_path = out_dir / f"docids_chunk_{chunk_idx}.npy"

                done_marker = out_dir / f"chunk_{chunk_idx}.done"
                chunk_bar.set_postfix(chunk=chunk_idx, docs=f"{docs_done:,}", status="encoding")
                if done_marker.exists() or (emb_path.exists() and docids_path.exists()):
                    tqdm.write(f"[chunk {chunk_idx}/{n_chunks-1}] checkpoint found — skipping")
                else:
                    emb_path, docids_path = encode_chunk(
                        chunk_idx, chunk_docs, chunk_docids, llm, out_dir, args.max_model_len,
                    )
                    if api:
                        chunk_bar.set_postfix(chunk=chunk_idx, docs=f"{docs_done:,}", status="uploading")
                        upload_and_clean(api, args.hf_repo, args.hf_folder, emb_path, docids_path, chunk_idx)

                docs_done += len(chunk_docs)
                elapsed = time.time() - run_start
                rate = docs_done / max(elapsed, 1e-6)
                eta_min = (total_docs - docs_done) / max(rate, 1e-6) / 60
                chunk_bar.set_postfix(
                    chunk=f"{chunk_idx}/{n_chunks-1}",
                    docs=f"{docs_done:,}/{total_docs:,}",
                    rate=f"{rate:.0f} d/s",
                    ETA=f"{eta_min:.0f}m",
                )
                chunk_bar.update(1)

                chunk_docs, chunk_docids = [], []
                chunk_idx += 1

    # Final partial chunk
    if chunk_docs:
        emb_path = out_dir / f"embeddings_chunk_{chunk_idx}.fp16.npy"
        docids_path = out_dir / f"docids_chunk_{chunk_idx}.npy"

        done_marker = out_dir / f"chunk_{chunk_idx}.done"
        chunk_bar.set_postfix(chunk=chunk_idx, docs=f"{docs_done:,}", status="encoding")
        if done_marker.exists() or (emb_path.exists() and docids_path.exists()):
            tqdm.write(f"[chunk {chunk_idx}/{n_chunks-1}] checkpoint found — skipping")
        else:
            emb_path, docids_path = encode_chunk(
                chunk_idx, chunk_docs, chunk_docids, llm, out_dir, args.max_model_len,
            )
            if api:
                chunk_bar.set_postfix(chunk=chunk_idx, status="uploading")
                upload_and_clean(api, args.hf_repo, args.hf_folder, emb_path, docids_path, chunk_idx)

        docs_done += len(chunk_docs)
        chunk_bar.update(1)

    chunk_bar.close()
    total_elapsed = time.time() - run_start
    print(f"\nDone. {docs_done:,} docs | {chunk_idx + 1} chunk(s) | {total_elapsed/60:.1f} min total")


if __name__ == "__main__":
    main()
