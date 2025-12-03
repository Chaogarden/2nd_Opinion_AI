#!/usr/bin/env python3
"""
Quick semantic search over the guidelines FAISS index.
Prints top-k chunks with corpus + title + (optional) heading_path.
"""

import argparse, json
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="rag/index/guidelines.faiss")
    ap.add_argument("--chunks", default="rag/index/guidelines_chunks.jsonl")
    ap.add_argument("--model", default="BAAI/bge-m3")
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--filter_corpus", default=None, help="e.g., infectious_disease")
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    embedder = SentenceTransformer(args.model)

    rows = list(read_jsonl(Path(args.chunks)))
    if args.filter_corpus:
        rows = [r for r in rows if r.get("corpus") == args.filter_corpus]

    qv = embedder.encode([args.q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, args.k)

    print(f"\nTop {args.k} for: {args.q}\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        row = rows[int(idx)]
        print(f"[{rank}] score={float(score):.3f} | corpus={row['corpus']} | title={row['title']}")
        if row.get("heading_path"):
            print(f"    heading: {row['heading_path']}")
        print(f"    src: {row['source_path']}")
        preview = row["text"][:500].replace("\n", " ")
        print("    " + preview + ("…" if len(row['text']) > 500 else ""))
        print("-" * 80)

if __name__ == "__main__":
    main()
