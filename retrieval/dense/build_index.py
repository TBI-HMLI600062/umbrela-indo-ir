"""
Build per-chunk FAISS IndexFlatIP from embeddings produced by embed_corpus.py.

Reads:   {embeddings}/embeddings_chunk_N.fp16.npy
Writes:  {output}/index_chunk_N.faiss
         {output}/docids_chunk_N.npy  (copied/symlinked if output != embeddings)

The docids_chunk_N.npy files produced by embed_corpus.py are already in the
format expected by retrieve.py — they do not need to be rebuilt.

FAISS strategy (matches retrieve.py):
    IndexFlatIP over L2-normalised float32 vectors (exact cosine similarity).

Resume-safe: existing index_chunk_N.faiss files are skipped automatically.

Args:
    --embeddings   directory with embeddings_chunk_N.fp16.npy files
                   (or HF repo subfolder when --hf-repo is set)
    --output       output directory for index files (default: same as --embeddings)
    --hf-repo      HuggingFace dataset repo ID to download embeddings from
    --hf-folder    subfolder inside hf-repo (default: qwen3-embed-4b)

Examples:
    # From local files (embed_corpus.py already ran locally):
    python retrieval/dense/build_index.py \\
        --embeddings embeddings/qwen3-embed-4b/

    # Download from HF then build:
    python retrieval/dense/build_index.py \\
        --hf-repo karolinajocelyn/umbrela-indo-ir \\
        --hf-folder qwen3-embed-4b \\
        --embeddings embeddings/qwen3-embed-4b/

    Then run retrieval:
    python retrieval/dense/retrieve.py \\
        --faiss-chunks embeddings/qwen3-embed-4b/ \\
        --model Qwen/Qwen3-Embedding-4B \\
        --topics data/miracl-id/topics/test.tsv \\
        --output candidates/qwen3_test_top100.jsonl
"""

import argparse
import os
import shutil
import time
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS IndexFlatIP from chunked fp16 embeddings.")
    parser.add_argument("--embeddings", required=True,
                        help="Directory with embeddings_chunk_N.fp16.npy (and docids_chunk_N.npy)")
    parser.add_argument("--output", default=None,
                        help="Output directory for index_chunk_N.faiss (default: same as --embeddings)")
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace dataset repo to download embeddings from")
    parser.add_argument("--hf-folder", default="qwen3-embed-4b",
                        help="Subfolder inside hf-repo (default: qwen3-embed-4b)")
    return parser.parse_args()


def download_from_hf(hf_repo: str, hf_folder: str, local_dir: Path):
    """Download all embeddings_chunk_* and docids_chunk_* files from HF."""
    from huggingface_hub import HfApi, hf_hub_download

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_xet  # noqa: F401
        print("hf_xet active — fast parallel download")
    except ImportError:
        try:
            import hf_transfer  # noqa: F401
            print("hf_transfer active — fast parallel download")
        except ImportError:
            print("tip: pip install hf-xet for faster downloads")

    hf_token = os.environ.get("HF_TOKEN")
    api = HfApi(token=hf_token)
    all_files = list(api.list_repo_files(hf_repo, repo_type="dataset"))

    targets = [
        f for f in all_files
        if f.startswith(f"{hf_folder}/")
        and (
            "/embeddings_chunk_" in f
            or "/docids_chunk_" in f
        )
    ]
    if not targets:
        raise FileNotFoundError(
            f"No embeddings/docids chunk files found under {hf_folder}/ in {hf_repo}"
        )

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(targets)} file(s) from {hf_repo}/{hf_folder}/ ...")
    for repo_path in targets:
        fname = Path(repo_path).name
        dest = local_dir / fname
        if dest.exists():
            print(f"  {fname} already exists — skipping download")
            continue
        print(f"  Downloading {fname} ...")
        tmp = hf_hub_download(
            repo_id=hf_repo,
            filename=repo_path,
            repo_type="dataset",
            token=hf_token,
            local_dir=str(local_dir),
        )
        # hf_hub_download saves to a cache subdir; move to flat local_dir
        tmp_path = Path(tmp)
        if tmp_path != dest:
            shutil.move(str(tmp_path), str(dest))
    print("Download complete.\n")


def build_chunk_index(emb_path: Path, index_path: Path):
    t0 = time.time()
    print(f"[{emb_path.name}] Loading ...")
    emb = np.load(emb_path).astype(np.float32)
    print(f"  {len(emb):,} vectors | dim={emb.shape[1]}")

    # Vectors from embed_corpus.py are already L2-normalised, but normalise
    # again in-place to guard against any fp16 rounding drift.
    faiss.normalize_L2(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    batch = 200_000
    for i in tqdm(range(0, len(emb), batch), desc="  Adding to FAISS", leave=False):
        index.add(emb[i : i + batch])
    del emb

    faiss.write_index(index, str(index_path))
    elapsed = time.time() - t0
    size_gb = index_path.stat().st_size / 1e9
    print(f"  Saved {index_path.name} ({size_gb:.2f} GB) in {elapsed:.0f}s")


def main():
    args = parse_args()
    emb_dir = Path(args.embeddings)
    out_dir = Path(args.output) if args.output else emb_dir

    if args.hf_repo:
        download_from_hf(args.hf_repo, args.hf_folder, emb_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(emb_dir.glob("embeddings_chunk_*.fp16.npy"))
    if not chunk_files:
        raise FileNotFoundError(
            f"No embeddings_chunk_*.fp16.npy found in {emb_dir}\n"
            f"If files are on HuggingFace, add --hf-repo <repo-id>"
        )

    print(f"Found {len(chunk_files)} chunk(s) to index.\n")
    t_total = time.time()

    for emb_path in chunk_files:
        # embeddings_chunk_3.fp16.npy → chunk index 3
        chunk_idx = emb_path.name.split("embeddings_chunk_")[1].split(".")[0]
        index_path = out_dir / f"index_chunk_{chunk_idx}.faiss"

        if index_path.exists():
            print(f"[chunk {chunk_idx}] index already exists — skipping")
            continue

        build_chunk_index(emb_path, index_path)

        # If output dir differs from embeddings dir, copy the docids file too
        # so retrieve.py can find both files in one place.
        if out_dir != emb_dir:
            docids_src = emb_dir / f"docids_chunk_{chunk_idx}.npy"
            docids_dst = out_dir / f"docids_chunk_{chunk_idx}.npy"
            if docids_src.exists() and not docids_dst.exists():
                shutil.copy2(str(docids_src), str(docids_dst))

    total_elapsed = time.time() - t_total
    print(f"\nDone. {len(chunk_files)} chunk(s) indexed in {total_elapsed/60:.1f} min")
    print(f"\nNext step — run retrieval:")
    print(f"  python retrieval/dense/retrieve.py \\")
    print(f"      --faiss-chunks {out_dir} \\")
    print(f"      --model Qwen/Qwen3-Embedding-4B \\")
    print(f"      --topics data/miracl-id/topics/test.tsv \\")
    print(f"      --output candidates/qwen3_test_top100.jsonl")


if __name__ == "__main__":
    main()
