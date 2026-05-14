"""
Download and preprocess MIRACL-ID dataset from HuggingFace.

Saves corpus, topics, and qrels in the format expected by qrel_generation/inference.py.
Splits MIRACL train (4071) into train (80%) and val (20%) with seed=42.
MIRACL dev (960) becomes test split (has human qrels).

Optionally uploads processed data to a HuggingFace dataset repo.

Args:
    --lang          language code (default: id)
    --output        local output directory (default: data/miracl-id/)
    --token         HuggingFace token (required for download + upload)
    --hf-repo       HF dataset repo to upload to (e.g. fassabilf/umbrela-indo-ir)
    --no-upload     skip upload to HuggingFace
    --val-ratio     fraction of train to use for val (default: 0.2)
    --seed          random seed for train/val split (default: 42)

Example:
    python data/download_miracl.py \\
        --token hf_xxx \\
        --hf-repo fassabilf/umbrela-indo-ir

After upload, team members can download with:
    huggingface-cli download fassabilf/umbrela-indo-ir \\
        --repo-type dataset --local-dir data/miracl-id/
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Download and preprocess MIRACL-ID.")
    parser.add_argument("--lang", default="id", help="Language code (default: id)")
    parser.add_argument("--output", default="data/miracl-id/",
                        help="Local output directory")
    parser.add_argument("--token", default=None, help="HuggingFace token")
    parser.add_argument("--hf-repo", default="fassabilf/umbrela-indo-ir",
                        help="HF dataset repo for upload")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip upload to HuggingFace")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Fraction of train for val split (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split (default: 42)")
    return parser.parse_args()


def download_corpus(lang: str, token: str, out_dir: Path):
    """Download MIRACL corpus and save as corpus/corpus.jsonl."""
    from datasets import load_dataset

    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / "corpus.jsonl"

    if out_path.exists():
        print(f"Corpus already exists at {out_path}, skipping download.")
        return

    print(f"Downloading MIRACL-{lang.upper()} corpus...")
    ds = load_dataset("miracl/miracl-corpus", lang,
                      token=token, trust_remote_code=True)["train"]

    print(f"Saving {len(ds)} passages to {out_path}...")
    with open(out_path, "w") as f:
        for row in tqdm(ds, desc="Writing corpus", unit=" passages"):
            doc_text = f"{row['title']}\n{row['text']}".strip()
            f.write(json.dumps({"docid": row["docid"], "doc": doc_text}) + "\n")

    print(f"Corpus saved: {len(ds)} passages")


def save_split(df, split_name: str, out_dir: Path, save_human_qrels: bool = False):
    """Save topics, candidates, and optionally human qrels for a split."""
    topics_dir = out_dir / "topics"
    cands_dir = out_dir / "qrels" / "candidates"
    human_dir = out_dir / "qrels" / "human"
    topics_dir.mkdir(parents=True, exist_ok=True)
    cands_dir.mkdir(parents=True, exist_ok=True)

    topics_path = topics_dir / f"{split_name}.tsv"
    cands_path = cands_dir / f"{split_name}.jsonl"

    # topics TSV
    with open(topics_path, "w") as f:
        for _, row in df.iterrows():
            qid = str(row["query_id"])
            query = row["query"].replace("\t", " ")
            f.write(f"{qid}\t{query}\n")

    # candidates JSONL
    with open(cands_path, "w") as f:
        for _, row in df.iterrows():
            qid = str(row["query_id"])
            pos_ids = [p["docid"] for p in row.get("positive_passages", [])]
            neg_ids = [p["docid"] for p in row.get("negative_passages", [])]
            f.write(json.dumps({
                "qid": qid,
                "positive_docids": pos_ids,
                "negative_docids": neg_ids,
            }) + "\n")

    # human qrels — save all annotated pairs (rel=1 for positives, rel=0 for negatives)
    if save_human_qrels:
        human_dir.mkdir(parents=True, exist_ok=True)
        qrels_path = human_dir / f"{split_name}.txt"
        with open(qrels_path, "w") as f:
            for _, row in df.iterrows():
                qid = str(row["query_id"])
                for p in row.get("positive_passages", []):
                    f.write(f"{qid} 0 {p['docid']} 1\n")
                for p in row.get("negative_passages", []):
                    f.write(f"{qid} 0 {p['docid']} 0\n")
        print(f"Human qrels saved: {qrels_path}")

    n_queries = len(df)
    n_pairs = sum(
        len(r.get("positive_passages", [])) + len(r.get("negative_passages", []))
        for _, r in df.iterrows()
    )
    print(f"Split '{split_name}': {n_queries} queries, {n_pairs} candidate pairs")


def main():
    args = parse_args()

    if args.token:
        os.environ["HF_TOKEN"] = args.token

    from datasets import load_dataset

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Download corpus ---
    download_corpus(args.lang, args.token, out_dir)

    # --- Download MIRACL train → split 80/20 into train + val ---
    print(f"\nDownloading MIRACL-{args.lang.upper()} train split...")
    raw_train = load_dataset("miracl/miracl", args.lang,
                             split="train", token=args.token, trust_remote_code=True)
    train_df = raw_train.to_pandas().sample(frac=1, random_state=args.seed).reset_index(drop=True)

    n_val = int(len(train_df) * args.val_ratio)
    val_df = train_df.iloc[:n_val].reset_index(drop=True)
    train_split_df = train_df.iloc[n_val:].reset_index(drop=True)

    print(f"Train/val split (seed={args.seed}, val_ratio={args.val_ratio}):")
    save_split(train_split_df, "train", out_dir, save_human_qrels=True)
    save_split(val_df, "val", out_dir, save_human_qrels=True)

    # --- Download MIRACL dev → becomes test (has human qrels) ---
    print(f"\nDownloading MIRACL-{args.lang.upper()} dev split (→ test)...")
    raw_dev = load_dataset("miracl/miracl", args.lang,
                           split="dev", token=args.token, trust_remote_code=True)
    test_df = raw_dev.to_pandas()
    save_split(test_df, "test", out_dir, save_human_qrels=True)

    # --- Summary ---
    print(f"\nAll data saved to: {out_dir}")
    print("Structure:")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.relative_to(out_dir)}  ({size_mb:.1f} MB)")

    # --- Upload to HuggingFace ---
    if not args.no_upload:
        from huggingface_hub import HfApi, create_repo

        print(f"\nUploading to HuggingFace: {args.hf_repo}")
        api = HfApi(token=args.token)

        try:
            create_repo(args.hf_repo, repo_type="dataset",
                        token=args.token, exist_ok=True)
            print(f"Repo ready: {args.hf_repo}")
        except Exception as e:
            print(f"Warning: could not create repo ({e}), attempting upload anyway")

        api.upload_folder(
            repo_id=args.hf_repo,
            folder_path=str(out_dir),
            repo_type="dataset",
            commit_message="Add processed MIRACL-ID data (train/val/test splits)",
        )
        print(f"Upload complete: https://huggingface.co/datasets/{args.hf_repo}")
        print("\nTeam members can download with:")
        print(f"  huggingface-cli download {args.hf_repo} "
              f"--repo-type dataset --local-dir data/miracl-id/")
    else:
        print("Skipping HuggingFace upload (--no-upload set)")


if __name__ == "__main__":
    main()
