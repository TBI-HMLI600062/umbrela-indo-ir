"""
Dense retrieval for MIRACL-ID queries using prebuilt embeddings + FAISS.

Args:
    --embeddings   path to embeddings directory (output of embed_corpus.py)
                   expects doc_emb.fp16.npy + doc_pids.npy
    --faiss-chunks path to directory with pre-built chunked FAISS indices
                   expects index_chunk_N.faiss + docids_chunk_N.npy
                   (mutually exclusive with --embeddings)
    --topics       topics TSV file (e.g. data/miracl-id/topics/test.tsv)
    --output       output candidates JSONL file
    --model        encoder model ID
    --k            top-k per query (default: 100)
    --device       cuda | cpu (default: cuda)
    --batch-size   query encoding batch size (default: auto)

Examples:
    # BGE-M3 from raw embeddings:
    python retrieval/dense/retrieve.py \
        --embeddings embeddings/bge-m3/ \
        --topics data/miracl-id/topics/test.tsv \
        --output candidates/bgem3_test_top100.jsonl

    # Qwen3-Embedding from pre-built chunks (no instruction — original):
    python retrieval/dense/retrieve.py \
        --faiss-chunks embeddings/qwen3-embed-4b/ \
        --model Qwen/Qwen3-Embedding-4B \
        --topics data/miracl-id/topics/test.tsv \
        --output candidates/qwen3_test_top100.jsonl

    # Qwen3-Embedding with instruction prefix (recommended for Qwen3):
    python retrieval/dense/retrieve.py \
        --faiss-chunks embeddings/qwen3-embed-4b/ \
        --model Qwen/Qwen3-Embedding-4B \
        --topics data/miracl-id/topics/test.tsv \
        --output candidates/qwen3_instruct_test_top100.jsonl \
        --instruction "Instruct: Given a question, retrieve relevant passages that answer the question\nQuery: "
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_DECODER_TYPES = {"qwen2", "qwen3", "llama", "mistral", "gemma", "phi", "falcon"}


def parse_args():
    parser = argparse.ArgumentParser(description="Dense retrieval with FAISS.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--embeddings", help="Embeddings dir (doc_emb.fp16.npy + doc_pids.npy)")
    group.add_argument("--faiss-chunks", help="Dir with index_chunk_N.faiss + docids_chunk_N.npy")
    parser.add_argument("--topics", required=True, help="Topics TSV file")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Encoder model for query encoding")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Query encoding batch size (default: 256 for BGE-M3, 32 for LLMs)")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Instruction prefix prepended to each query before encoding. "
                             "Recommended for Qwen3-Embedding: "
                             "'Instruct: Given a question, retrieve relevant passages that answer the question\\nQuery: '")
    return parser.parse_args()


def load_topics(topics_path: Path) -> list[tuple[str, str]]:
    topics = []
    with open(topics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            topics.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return topics


def _load_one_chunk(f: Path, chunks_dir: Path):
    idx = faiss.read_index(str(f))
    docids = np.load(
        str(chunks_dir / f.name.replace("index_chunk_", "docids_chunk_").replace(".faiss", ".npy")),
        allow_pickle=True,
    )
    return idx, docids


def load_chunks_parallel(chunks_dir: Path, n_threads: int = 4):
    """Load all FAISS chunks in parallel using IO threads."""
    chunk_files = sorted(chunks_dir.glob("index_chunk_*.faiss"))
    if not chunk_files:
        raise FileNotFoundError(f"No index_chunk_*.faiss in {chunks_dir}")
    chunks = [None] * len(chunk_files)
    with ThreadPoolExecutor(max_workers=min(n_threads, len(chunk_files))) as pool:
        futures = {pool.submit(_load_one_chunk, f, chunks_dir): i
                   for i, f in enumerate(chunk_files)}
        with tqdm(total=len(chunk_files), desc="Loading FAISS chunks") as pbar:
            for fut in as_completed(futures):
                i = futures[fut]
                chunks[i] = fut.result()
                pbar.update(1)
    return chunks


class QueryDataset(Dataset):
    def __init__(self, queries, tokenizer, max_length=128):
        self.queries = queries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx]

    def collate(self, batch):
        return self.tokenizer(
            batch, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )


def encode_queries(model_or_encoder, encoder_type: str, queries: list[str],
                   device: str, batch_size: int) -> np.ndarray:
    if encoder_type == "vllm":
        llm = model_or_encoder
        outputs = llm.embed(queries)
        embs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = np.where(norms > 0, embs / norms, embs)
        return embs

    if encoder_type == "bgem3":
        out = model_or_encoder.encode(
            queries, batch_size=batch_size, max_length=128,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        q_emb = out["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(q_emb)
        return q_emb

    model, tokenizer = model_or_encoder
    use_last_token = model.config.model_type in _DECODER_TYPES
    pool_mode = "last-token" if use_last_token else "mean"
    print(f"  Architecture: {model.config.model_type} | Pooling: {pool_mode}")

    dataset = QueryDataset(queries, tokenizer)
    # num_workers > 0: tokenization runs on CPU workers, overlapping GPU inference
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=min(4, batch_size),
        collate_fn=dataset.collate,
        pin_memory=(device == "cuda"),
    )

    all_embs = []
    n_batches = (len(queries) + batch_size - 1) // batch_size
    with tqdm(total=n_batches, desc="Encoding queries") as pbar:
        for enc in loader:
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
            if use_last_token:
                emb = out.last_hidden_state[:, -1, :]
            else:
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            emb = F.normalize(emb.float(), dim=-1)
            all_embs.append(emb.cpu().numpy())
            pbar.update(1)

    return np.concatenate(all_embs, axis=0)


def search_chunks(chunks: list, q_emb: np.ndarray, k: int) -> tuple[np.ndarray, list]:
    all_scores, all_docids = [], []
    for chunk_index, chunk_docids in tqdm(chunks, desc="Searching chunks"):
        scores, local_idx = chunk_index.search(q_emb, k)
        docids = np.where(local_idx >= 0, chunk_docids[local_idx], None)
        all_scores.append(scores)
        all_docids.append(docids)

    merged_scores = np.concatenate(all_scores, axis=1)
    merged_docids = np.concatenate(all_docids, axis=1)

    top_scores, top_docids = [], []
    for i in range(q_emb.shape[0]):
        order = np.argsort(-merged_scores[i])[:k]
        top_scores.append(merged_scores[i][order])
        top_docids.append(merged_docids[i][order])
    return np.array(top_scores), top_docids


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_chunks = args.faiss_chunks is not None

    # --- Kick off FAISS chunk loading in background BEFORE loading the model ---
    # This overlaps IO (reading ~20 GB from disk) with model download/load time.
    chunk_future = None
    if use_chunks:
        print(f"Starting background FAISS chunk load from {args.faiss_chunks}...")
        _executor = ThreadPoolExecutor(max_workers=1)
        chunk_future = _executor.submit(load_chunks_parallel, Path(args.faiss_chunks))

    # --- Load encoder (runs while chunks load in background) ---
    print(f"\nLoading encoder: {args.model}")
    if "bge-m3" in args.model.lower():
        from FlagEmbedding import BGEM3FlagModel
        encoder = BGEM3FlagModel(args.model, use_fp16=True, device=args.device)
        encoder_type = "bgem3"
        batch_size = args.batch_size or 256
    elif "qwen3" in args.model.lower() and "embedding" in args.model.lower():
        # Use vLLM (same as embed_corpus.py) so query embeddings are in the same space
        # as the pre-built FAISS document embeddings.
        from vllm import LLM
        encoder = LLM(
            model=args.model,
            runner="pooling",
            convert="embed",
            gpu_memory_utilization=0.85,
            max_model_len=8192,
            dtype="bfloat16",
            enforce_eager=False,
        )
        encoder_type = "vllm"
        batch_size = args.batch_size or 960
    else:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            trust_remote_code=True,
        ).to(args.device).eval()
        encoder = (model, tokenizer)
        encoder_type = "transformer"
        batch_size = args.batch_size or 32

    # --- Load topics ---
    print(f"\nLoading topics from {args.topics}...")
    topics = load_topics(Path(args.topics))
    queries = [q for _, q in topics]
    if args.instruction:
        instruction = args.instruction.replace("\\n", "\n")
        queries = [instruction + q for q in queries]
        print(f"Topics: {len(topics):,} queries | instruction prefix: {repr(instruction[:60])}...")
    else:
        print(f"Topics: {len(topics):,} queries")

    # --- Encode queries ---
    t0 = time.time()
    q_emb = encode_queries(encoder, encoder_type, queries, args.device, batch_size)
    print(f"Query encoding done in {time.time() - t0:.1f}s")

    # --- Wait for FAISS chunks (should already be done by now) ---
    if use_chunks:
        print("\nWaiting for FAISS chunks (if still loading)...")
        t_wait = time.time()
        chunks = chunk_future.result()
        waited = time.time() - t_wait
        if waited > 1:
            print(f"Waited {waited:.1f}s for chunks")
        _executor.shutdown(wait=False)
        total = sum(c[0].ntotal for c in chunks)
        print(f"Chunks ready: {len(chunks)} chunks | {total:,} passages")
    else:
        emb_dir = Path(args.embeddings)
        print(f"\nLoading embeddings from {emb_dir}...")
        doc_emb = np.load(emb_dir / "doc_emb.fp16.npy").astype(np.float32)
        doc_pids = np.load(emb_dir / "doc_pids.npy", allow_pickle=True)
        print(f"Loaded {len(doc_pids):,} passages | dim={doc_emb.shape[1]}")
        print("Building FAISS IndexFlatIP...")
        faiss.normalize_L2(doc_emb)
        index = faiss.IndexFlatIP(doc_emb.shape[1])
        for i in tqdm(range(0, len(doc_emb), 200_000), desc="Adding to FAISS"):
            index.add(doc_emb[i : i + 200_000])
        del doc_emb

    # --- Search ---
    print("\nSearching...")
    t0 = time.time()
    if use_chunks:
        scores, docid_rows = search_chunks(chunks, q_emb, args.k)
        print(f"Search done in {time.time() - t0:.2f}s")
        with open(output_path, "w") as f:
            for i, (qid, _) in enumerate(tqdm(topics, desc="Writing")):
                candidates = [
                    {"docid": str(docid_rows[i][j]), "score": float(scores[i, j])}
                    for j in range(args.k)
                    if docid_rows[i][j] is not None
                ]
                f.write(json.dumps({"qid": qid, "candidates": candidates}) + "\n")
    else:
        faiss.normalize_L2(q_emb)
        scores, indices = index.search(q_emb, args.k)
        print(f"Search done in {time.time() - t0:.2f}s")
        with open(output_path, "w") as f:
            for i, (qid, _) in enumerate(tqdm(topics, desc="Writing")):
                candidates = [
                    {"docid": str(doc_pids[indices[i, j]]), "score": float(scores[i, j])}
                    for j in range(args.k)
                    if indices[i, j] >= 0
                ]
                f.write(json.dumps({"qid": qid, "candidates": candidates}) + "\n")

    print(f"\nCandidates saved to {output_path}")


if __name__ == "__main__":
    main()
