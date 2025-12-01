#!/usr/bin/env python3
"""
Simple semantic search over the Merck19 FAISS index.
Prints top-k chunks with headings for inspection, ready to feed your LLM.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_meta(meta_path: Path) -> List[Dict]:
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=str, default="rag/index/merck19.faiss")
    ap.add_argument("--meta", type=str, default="rag/index/merck19_meta.jsonl")
    ap.add_argument("--model", type=str, default="BAAI/bge-m3")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--q", type=str, required=True, help="Query")
    args = ap.parse_args()

    index = faiss.read_index(str(Path(args.index)))
    meta = load_meta(Path(args.meta))
    model = SentenceTransformer(args.model)

    q_vec = model.encode([args.q], normalize_embeddings=True)
    D, I = index.search(np.asarray(q_vec, dtype="float32"), args.k)

    print(f"\nTop {args.k} for: {args.q}\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        row = meta[int(idx)]
        print(f"[{rank}] score={float(score):.3f}")
        print(f"HEADINGS: {row['heading_path']}")
        print(row["text"][:1000] + ("…" if len(row["text"]) > 1000 else ""))
        print("-" * 80)


if __name__ == "__main__":
    main()
