"""
Dense retrieval for MIRACL-ID queries using prebuilt BGE-M3 embeddings + FAISS.

Args:
    --embeddings  path to embeddings directory (output of embed_corpus.py)
    --topics      topics TSV file (e.g. data/miracl-id/topics/test.tsv)
    --output      output candidates JSONL file
    --model       encoder model ID (must match embed_corpus.py)
    --k           top-k per query (default: 100)
    --device      cuda | cpu (default: cuda)

Example:
    python retrieval/dense/retrieve.py \
        --embeddings embeddings/bge-m3/ \
        --topics data/miracl-id/topics/test.tsv \
        --output candidates/bgem3_top100.jsonl --k 100
"""

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Dense retrieval with FAISS.")
    parser.add_argument("--embeddings", required=True, help="Embeddings directory")
    parser.add_argument("--topics", required=True, help="Topics TSV file")
    parser.add_argument("--output", required=True, help="Output candidates JSONL")
    parser.add_argument("--model", default="BAAI/bge-m3", help="Encoder model for query encoding")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--device", default="cuda")
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


def main():
    args = parse_args()
    emb_dir = Path(args.embeddings)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading document embeddings from {emb_dir}...")
    doc_emb = np.load(emb_dir / "doc_emb.fp16.npy").astype(np.float32)
    doc_pids = np.load(emb_dir / "doc_pids.npy", allow_pickle=True)
    print(f"Loaded {len(doc_pids):,} passages | dim={doc_emb.shape[1]}")

    print("Building FAISS IndexFlatIP...")
    faiss.normalize_L2(doc_emb)
    index = faiss.IndexFlatIP(doc_emb.shape[1])
    # add in chunks to avoid peak RAM spike
    chunk = 200_000
    for i in tqdm(range(0, len(doc_emb), chunk), desc="Adding to FAISS"):
        index.add(doc_emb[i:i+chunk])
    del doc_emb

    print(f"Loading topics from {args.topics}...")
    topics = load_topics(Path(args.topics))
    queries = [q for _, q in topics]
    print(f"Topics: {len(topics):,} queries")

    from FlagEmbedding import BGEM3FlagModel
    print(f"Encoding queries with {args.model}...")
    model = BGEM3FlagModel(args.model, use_fp16=True, device=args.device)
    q_output = model.encode(queries, batch_size=256, max_length=128,
                             return_dense=True, return_sparse=False,
                             return_colbert_vecs=False)
    q_emb = q_output["dense_vecs"].astype(np.float32)
    faiss.normalize_L2(q_emb)

    t0 = time.time()
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

    print(f"Candidates saved to {output_path}")


if __name__ == "__main__":
    main()
