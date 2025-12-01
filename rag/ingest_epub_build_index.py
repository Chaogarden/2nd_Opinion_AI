#!/usr/bin/env python3
"""
Parse a Merck 19th EPUB, chunk with headings preserved, embed, and build a FAISS index.

Outputs:
  - index/merck19.faiss        (FAISS index)
  - index/merck19_meta.jsonl   (one JSON line per chunk with metadata)
  - index/merck19_embeds.npy   (optional numpy backup of vectors)
"""

import os
import json
import re
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from tqdm import tqdm
from ebooklib import epub
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize

import faiss
from sentence_transformers import SentenceTransformer

# One-time download for sentence splitting
nltk.download("punkt", quiet=True)

# ---------- Helpers ----------

def html_to_text_keep_structure(html: str) -> List[Tuple[str, str]]:
    """
    Convert an HTML doc into a list of (heading_path, paragraph_text) pairs.
    heading_path is a breadcrumb-like string built from h1/h2/h3...
    We keep paragraphs only (no tables/images).
    """
    soup = BeautifulSoup(html, "lxml")
    # Track current heading path
    path = []
    out: List[Tuple[str, str]] = []

    # We’ll walk DOM in order; when we see a heading, adjust path
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        name = el.name.lower()

        if name in {"h1", "h2", "h3", "h4"}:
            level = int(name[1])
            text = " ".join(el.get_text(" ", strip=True).split())
            if not text:
                continue
            # shrink/expand path to level
            while len(path) >= level:
                path.pop()
            path.append(text)
        elif name in {"p", "li"}:
            text = " ".join(el.get_text(" ", strip=True).split())
            if not text:
                continue
            breadcrumb = " > ".join(path) if path else "Untitled"
            out.append((breadcrumb, text))
    return out


def chunk_paragraphs_with_headings(
    items: List[Tuple[str, str]],
    max_words: int = 400,
    min_words: int = 120,
    overlap_sents: int = 1,
) -> List[Dict]:
    """
    Chunk by sentences, respecting headings.
    Each chunk carries: text, heading_path.
    A chunk aims for ~[min_words, max_words] words.
    """
    chunks: List[Dict] = []
    buffer_sents: List[str] = []
    buffer_head = None

    def flush():
        if not buffer_sents:
            return
        text = " ".join(buffer_sents).strip()
        if len(text.split()) < min_words and chunks:
            # append to previous if too small
            chunks[-1]["text"] += " " + text
            return
        chunks.append({"heading_path": buffer_head or "Untitled", "text": text})
        buffer_sents.clear()

    for heading_path, para in items:
        sents = sent_tokenize(para)
        if not sents:
            continue
        # If heading changes drastically, flush
        if buffer_head != heading_path and buffer_sents:
            flush()
        buffer_head = heading_path

        for s in sents:
            buffer_sents.append(s)
            # flush when near max_words
            if len(" ".join(buffer_sents).split()) >= max_words:
                flush()
                # seed overlap from the tail
                if overlap_sents > 0 and sents:
                    buffer_sents.extend(sents[max(0, sents.index(s)+1 - overlap_sents):sents.index(s)+1])
                buffer_head = heading_path

    flush()
    # final cleanup for tiny trailing chunk already handled in flush
    return chunks


def load_epub_items(epub_path: Path) -> List[Tuple[str, str]]:
    """
    Parse EPUB spine documents, turn each into (heading_path, paragraph_text) items.
    Works across ebooklib versions (no ITEM_DOCUMENT constant required).
    """
    book = epub.read_epub(str(epub_path))
    items = []
    for item in book.get_items():
        mt = getattr(item, "media_type", "") or ""
        name = (getattr(item, "get_name", lambda: "")() or "").lower()
        is_doc = (
            mt in ("application/xhtml+xml", "text/html")
            or name.endswith(".xhtml")
            or name.endswith(".html")
        )
        if is_doc:
            html = item.get_content().decode("utf-8", errors="ignore")
            items.extend(html_to_text_keep_structure(html))
    return items


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vectors, dtype="float32")


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epub", type=str, required=True, help="Path to Merck 19 EPUB file")
    ap.add_argument("--outdir", type=str, default="rag/index", help="Output directory for index and metadata")
    ap.add_argument("--model", type=str, default="BAAI/bge-m3", help="SentenceTransformer model")
    ap.add_argument("--max_words", type=int, default=400)
    ap.add_argument("--min_words", type=int, default=120)
    args = ap.parse_args()

    epub_path = Path(args.epub).expanduser().resolve()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading EPUB: {epub_path}")
    items = load_epub_items(epub_path)
    print(f"Parsed blocks (paragraph-level): {len(items)}")

    print("Chunking with headings…")
    chunks = chunk_paragraphs_with_headings(items, max_words=args.max_words, min_words=args.min_words, overlap_sents=1)
    print(f"Chunks: {len(chunks)}")

    texts = [c["text"] for c in chunks]
    print(f"Embedding with model: {args.model}")
    model = SentenceTransformer(args.model)
    vecs = embed_texts(model, texts)

    # Build FAISS
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine since we normalized
    index.add(vecs)

    # Save artifacts
    faiss_path = outdir / "merck19.faiss"
    meta_path = outdir / "merck19_meta.jsonl"
    npy_path  = outdir / "merck19_embeds.npy"

    faiss.write_index(index, str(faiss_path))
    np.save(npy_path, vecs)

    with meta_path.open("w", encoding="utf-8") as f:
        for i, c in enumerate(chunks):
            f.write(json.dumps({
                "id": i,
                "heading_path": c["heading_path"],
                "text": c["text"]
            }, ensure_ascii=False) + "\n")

    print(f"Done.\nIndex: {faiss_path}\nMeta: {meta_path}\nEmbeds: {npy_path}")

if __name__ == "__main__":
    main()
