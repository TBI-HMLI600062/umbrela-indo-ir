"""
Encode MIRACL-ID corpus with a dense encoder and build per-chunk FAISS indexes.

Processes the corpus in shards of --chunk-size docs. Each shard is encoded,
saved as index_chunk_X.faiss + docids_chunk_X.npy, optionally uploaded to a
Hugging Face dataset repo, then deleted locally to keep disk usage low.

Args:
    --model         HF encoder model ID (BAAI/bge-m3 or Qwen/Qwen2.5-7B-Instruct)
    --corpus        corpus JSONL file (default: data/miracl-id/corpus/corpus.jsonl)
    --output        output directory for chunk files
    --batch-size    encoding batch size (default: 64)
    --chunk-size    documents per shard (default: 300000)
    --device        cuda | cpu (default: cuda)
    --mode          encoding mode label, informational only (default: embedding)
    --hf-repo       HuggingFace dataset repo ID; if set, each chunk is uploaded
                    then deleted locally (requires HF_TOKEN env var)

Example (Karol — Qwen, chunked + upload):
    nohup python retrieval/dense/embed_corpus.py \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --output embeddings/qwen/ \\
        --batch-size 64 \\
        --hf-repo username/my-embeddings \\
        > log_qwen_encode.txt 2>&1 &

Output files per chunk X:
    {output}/index_chunk_{X}.faiss   — FAISS IndexFlatIP (cosine sim, L2-normalised)
    {output}/docids_chunk_{X}.npy    — numpy string array; docids[i] maps to vector i
"""

import argparse
import json
import os
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Decoder-only architectures use last-token pooling; all others use mean pooling.
_DECODER_MODEL_TYPES = {"qwen2", "llama", "mistral", "gemma", "phi", "falcon"}


def parse_args():
    parser = argparse.ArgumentParser(description="Encode corpus with dense encoder + FAISS.")
    parser.add_argument("--model", required=True, help="HF encoder model ID")
    parser.add_argument("--corpus", default="data/miracl-id/corpus/corpus.jsonl")
    parser.add_argument("--output", required=True, help="Output directory for chunk files")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=300_000,
                        help="Documents per shard (default: 300000)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="embedding",
                        help="Encoding mode label (default: embedding)")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace dataset repo ID for upload + local cleanup")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pooling helpers
# ---------------------------------------------------------------------------

def last_token_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Return the last token's hidden state.

    Correct only when tokenizer.padding_side == "left": left-padding pushes
    all real tokens to the right, so the last real token is always at index -1
    regardless of sequence length. Using attention_mask.sum-1 here would be
    wrong because that calculation assumes right-padded sequences.
    """
    return hidden[:, -1, :]


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pool over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: str,
    use_last_token: bool,
    max_length: int = 512,
) -> np.ndarray:
    """Tokenise and encode one batch; return L2-normalised float32 numpy array."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**encoded)

    embs = (
        last_token_pool(out.last_hidden_state)
        if use_last_token
        else mean_pool(out.last_hidden_state, encoded["attention_mask"])
    )
    embs = F.normalize(embs.float(), dim=-1)
    return embs.cpu().numpy()


# ---------------------------------------------------------------------------
# Per-chunk pipeline
# ---------------------------------------------------------------------------

def encode_chunk(
    chunk_idx: int,
    docs: list[str],
    docids: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    args,
    use_last_token: bool,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Encode one chunk, build a FAISS index, write both files, return their paths."""
    all_embs: list[np.ndarray] = []
    n_batches = (len(docs) + args.batch_size - 1) // args.batch_size

    for i in tqdm(
        range(0, len(docs), args.batch_size),
        total=n_batches,
        desc=f"  Chunk {chunk_idx} — encoding",
        leave=False,
    ):
        batch = docs[i : i + args.batch_size]
        all_embs.append(encode_batch(model, tokenizer, batch, args.device, use_last_token))
        if args.device == "cuda" and (i // args.batch_size) % 100 == 0:
            torch.cuda.empty_cache()

    embeddings = np.concatenate(all_embs, axis=0).astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss_path = out_dir / f"index_chunk_{chunk_idx}.faiss"
    docids_path = out_dir / f"docids_chunk_{chunk_idx}.npy"
    faiss.write_index(index, str(faiss_path))
    np.save(docids_path, np.array(docids))

    print(
        f"  Chunk {chunk_idx}: {index.ntotal:,} vectors  "
        f"({faiss_path.name}, {docids_path.name})"
    )
    return faiss_path, docids_path


def upload_and_clean(api, hf_repo: str, faiss_path: Path, docids_path: Path, chunk_idx: int):
    """Upload both chunk files to HF then delete them locally."""
    for path in (faiss_path, docids_path):
        print(f"  Uploading {path.name} → {hf_repo} ...")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=hf_repo,
            repo_type="dataset",
        )
    os.remove(faiss_path)
    os.remove(docids_path)
    print(f"  Chunk {chunk_idx} uploaded and local files removed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional HF upload setup
    api = None
    if args.hf_repo:
        from huggingface_hub import HfApi
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError("--hf-repo requires HF_TOKEN environment variable to be set.")
        api = HfApi(token=hf_token)
        print(f"HuggingFace upload enabled → {args.hf_repo}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Left-padding so the last real token is always at position -1, which is
    # what last_token_pool relies on. Right-padding (the default for most
    # tokenizers) would place padding tokens at -1 and break the pooling.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(args.device).eval()

    use_last_token = model.config.model_type in _DECODER_MODEL_TYPES
    print(
        f"Model type : {model.config.model_type}  |  "
        f"Pooling : {'last-token (left-pad)' if use_last_token else 'mean'}  |  "
        f"Precision : bfloat16"
    )

    # --- Stream corpus, flush a chunk whenever it fills ---
    corpus_path = Path(args.corpus)
    print(f"\nReading corpus: {corpus_path}")

    chunk_docs: list[str] = []
    chunk_docids: list[str] = []
    chunk_idx = 0
    total_docs = 0

    with open(corpus_path) as f:
        for line in f:
            row = json.loads(line)
            chunk_docids.append(row["docid"])
            chunk_docs.append(row["doc"])
            total_docs += 1

            if len(chunk_docs) == args.chunk_size:
                print(f"\nChunk {chunk_idx}  ({len(chunk_docs):,} docs, total so far: {total_docs:,})")
                faiss_path, docids_path = encode_chunk(
                    chunk_idx, chunk_docs, chunk_docids,
                    model, tokenizer, args, use_last_token, out_dir,
                )
                if api:
                    upload_and_clean(api, args.hf_repo, faiss_path, docids_path, chunk_idx)
                chunk_docs, chunk_docids = [], []
                chunk_idx += 1

    # Final partial chunk
    if chunk_docs:
        print(f"\nChunk {chunk_idx}  ({len(chunk_docs):,} docs, total: {total_docs:,})")
        faiss_path, docids_path = encode_chunk(
            chunk_idx, chunk_docs, chunk_docids,
            model, tokenizer, args, use_last_token, out_dir,
        )
        if api:
            upload_and_clean(api, args.hf_repo, faiss_path, docids_path, chunk_idx)

    print(f"\nDone. {total_docs:,} documents encoded across {chunk_idx + 1} chunk(s).")


if __name__ == "__main__":
    main()
