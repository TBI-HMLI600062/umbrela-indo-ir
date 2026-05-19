"""
Encode MIRACL-ID corpus with BGE-M3 and save embeddings + docid mapping.
Supports checkpoint/resume — safe to kill and restart at any time.

Run in background (3-4h for ~1.44M passages on A100/RTX4090):
    nohup python retrieval/dense/embed_corpus.py \
        --model BAAI/bge-m3 --output embeddings/bge-m3/ > log_bge_encode.txt &

Args:
    --model       HF encoder model ID (BAAI/bge-m3 or Qwen/Qwen2.5-7B-Instruct)
    --corpus      corpus JSONL file (default: data/miracl-id/corpus/corpus.jsonl)
    --output      output directory for embeddings
    --batch-size  encoding batch size (default: 64)
    --chunk-size  passages per checkpoint chunk (default: 100000)
    --max-length  max token length per passage (default: 512)
    --device      cuda | cpu (default: cuda)

Example (Arvin — BGE-M3):
    python retrieval/dense/embed_corpus.py \
        --model BAAI/bge-m3 --output embeddings/bge-m3/

Example (Karol — Qwen-embed):
    python retrieval/dense/embed_corpus.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --output embeddings/qwen/ --batch-size 32
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Encode corpus with BGE-M3 + save embeddings.")
    parser.add_argument("--model", required=True, help="HF encoder model ID")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_corpus(corpus_path: Path) -> tuple[list[str], list[str]]:
    docids, texts = [], []
    with open(corpus_path) as f:
        for line in tqdm(f, desc="Loading corpus"):
            obj = json.loads(line)
            docids.append(obj["docid"])
            texts.append((obj.get("doc", obj.get("text", "")) or "")[:4000])
    return docids, texts


def main():
    args = parse_args()
    out_dir = Path(args.output)
    chunk_dir = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(exist_ok=True)

    final_emb_path = out_dir / "doc_emb.fp16.npy"
    final_pid_path = out_dir / "doc_pids.npy"

    if final_emb_path.exists() and final_pid_path.exists():
        print(f"Embeddings already exist at {final_emb_path}, skipping.")
        emb = np.load(final_emb_path, mmap_mode="r")
        print(f"Shape: {emb.shape} | dtype: {emb.dtype}")
        return

    print(f"Loading corpus from {args.corpus}...")
    docids, texts = load_corpus(Path(args.corpus))
    n_docs = len(docids)
    print(f"Corpus: {n_docs:,} passages")

    from FlagEmbedding import BGEM3FlagModel
    print(f"Loading model {args.model}...")
    model = BGEM3FlagModel(args.model, use_fp16=True, device=args.device)

    n_chunks = (n_docs + args.chunk_size - 1) // args.chunk_size
    print(f"Encoding in {n_chunks} chunks of {args.chunk_size:,}...")

    for chunk_i in range(n_chunks):
        chunk_path = chunk_dir / f"chunk_{chunk_i:04d}.fp16.npy"
        if chunk_path.exists():
            print(f"Chunk {chunk_i+1}/{n_chunks}: cached, skipping")
            continue

        s = chunk_i * args.chunk_size
        e = min(s + args.chunk_size, n_docs)
        chunk_texts = texts[s:e]

        t0 = time.time()
        output = model.encode(
            chunk_texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        chunk_emb = output["dense_vecs"].astype(np.float16)
        np.save(chunk_path, chunk_emb)

        elapsed = time.time() - t0
        rate = (e - s) / max(elapsed, 1e-6)
        eta_min = (n_docs - e) / max(rate, 1e-6) / 60
        print(f"Chunk {chunk_i+1}/{n_chunks}: {e-s:,} docs in {elapsed:.0f}s "
              f"({rate:.0f} docs/s, ETA={eta_min:.0f} min)")

        del chunk_emb, chunk_texts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Concatenating chunks...")
    chunks = [np.load(chunk_dir / f"chunk_{i:04d}.fp16.npy") for i in range(n_chunks)]
    doc_emb = np.concatenate(chunks, axis=0)
    doc_pids = np.array(docids)

    np.save(final_emb_path, doc_emb)
    np.save(final_pid_path, doc_pids)
    print(f"Saved: {final_emb_path} | shape={doc_emb.shape} | dtype={doc_emb.dtype}")
    print(f"Saved: {final_pid_path} | {len(doc_pids):,} docids")


if __name__ == "__main__":
    main()
