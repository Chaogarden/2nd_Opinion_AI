#!/usr/bin/env python3
"""
Ingest mixed PDFs + EPUBs under a guidelines/ root into a single FAISS index.

Outputs (in --outdir):
  - guidelines.faiss
  - guidelines_meta.jsonl      (metadata per chunk, no text)
  - guidelines_chunks.jsonl    (text + metadata per line)
  - guidelines_embeds.npy      (optional vector cache)

Does not touch your existing Merck19 index files.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import unicodedata
import collections
import zipfile
import warnings
from bs4 import XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import numpy as np
from tqdm import tqdm

# PDF
import fitz  # PyMuPDF

# EPUB
from ebooklib import epub
from bs4 import BeautifulSoup

# Embeddings / FAISS
import faiss
from sentence_transformers import SentenceTransformer

# Sentence split
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt", quiet=True)


# ------------------------
# Utilities
# ------------------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def normalize_ws(text: str) -> str:
    # de-hyphenate line breaks like "exam-\nple" -> "example"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    # collapse internal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# helpers for noise reduction e.g. removing citations and abstracts that distract that FAISS

CIT_SUPERSCRIPTS = "".join(chr(c) for c in (
    list(range(0x00B9, 0x00BB)) + list(range(0x2070, 0x207A))
))

def strip_superscripts(text: str) -> str:
    # Remove Unicode superscript number clusters attached to words
    return re.sub(rf"(?<=\S)[{re.escape(CIT_SUPERSCRIPTS)}]+", "", text)

def strip_numeric_bracket_citations(text: str) -> str:
    # Remove [1], [2-5], [6,7,10] style citations
    text = re.sub(r"\[(?:\s*\d+\s*(?:-\s*\d+)?\s*(?:,\s*\d+\s*(?:-\s*\d+)?\s*)*)\]", "", text)
    # Remove bare numeric ^1,2 style at end of words (fallback)
    text = re.sub(r"(?<=\S)\s*\^\s*\d+(?:\s*,\s*\d+)*", "", text)
    return text

def strip_author_year_citations(text: str) -> str:
    # Remove parenthetical author-year citations like (Smith 2023), (Doe et al., 2019; Roe 2020)
    text = re.sub(
        r"\((?:[A-Z][A-Za-z\-\u00C0-\u024F]+(?:\s+et al\.)?(?:,\s*)?\s*(?:19|20)\d{2}[a-z]?"
        r"(?:\s*;\s*[A-Z][A-Za-z\-\u00C0-\u024F]+(?:\s+et al\.)?(?:,\s*)?\s*(?:19|20)\d{2}[a-z]?)*?)\)",
        "", text)
    return text

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def is_page_number_line(line: str) -> bool:
    return bool(re.fullmatch(r"\s*(page\s*)?\d{1,4}(\s*/\s*\d{1,4})?\s*", line.strip(), flags=re.I))

def common_lines_to_drop(page_lines: List[List[str]], threshold: float = 0.4) -> set:
    """
    Identify header/footer lines repeating on >= threshold of pages.
    """
    counter = collections.Counter()
    n_pages = len(page_lines)
    for lines in page_lines:
        # consider only very short lines or ALLCAPS short lines for header/footer candidates
        candidates = [l.strip() for l in lines if len(l.strip()) <= 80]
        counter.update(set(candidates))
    drop = {l for l, c in counter.items() if c / max(1, n_pages) >= threshold}
    # also drop pure page-number lines
    drop |= {l for l in list(counter) if is_page_number_line(l)}
    return drop

def remove_references_tail(text: str) -> str:
    """
    If a REFERENCES/BIBLIOGRAPHY section exists at the end, drop it.
    """
    # Find last occurrence to avoid false positives in TOC
    lines = text.splitlines()
    idx = None
    for i, l in enumerate(lines):
        s = l.strip().lower()
        if s in {"references", "bibliography", "works cited"} or re.fullmatch(r"references\s*[:\-]?\s*", s):
            idx = i
    if idx is not None and idx > 50:  # avoid chopping early TOCs
        return "\n".join(lines[:idx]).strip()
    return text

def strip_toc_block(text: str) -> str:
    """
    Remove an early table-of-contents block with leader dots ...... 12
    """
    lines = text.splitlines()
    # look at first ~200 lines
    window = lines[:200]
    pattern = re.compile(r".{10,}\.{2,}\s*\d{1,4}\s*$")
    hits = [i for i, l in enumerate(window) if pattern.search(l)]
    if len(hits) >= 5:
        # remove from min(hits)-5 up to max(hits)+5 as TOC block
        start = max(min(hits) - 5, 0)
        end = min(max(hits) + 5, len(lines))
        del lines[start:end]
    return "\n".join(lines)

def filter_and_clean_epub_items(items: List[Tuple[Optional[str], str]]) -> List[Tuple[Optional[str], str]]:
    out = []
    for heading, para in items:
        # skip explicit References/Bibliography sections
        if heading and heading.strip().lower().split(" > ")[-1] in {"references", "bibliography", "works cited"}:
            continue
        txt = remove_urls(strip_author_year_citations(strip_numeric_bracket_citations(strip_superscripts(para))))
        if txt.strip():
            out.append((heading, txt))
    return out


# ------------------------
# PDF extraction (PyMuPDF)
# ------------------------

def extract_text_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)

    # First pass: collect raw lines per page
    pages_lines: List[List[str]] = []
    for page in doc:
        txt = page.get_text("text")
        lines = [l.rstrip() for l in txt.splitlines()]
        pages_lines.append(lines)

    # Detect repeated header/footer lines
    drop = common_lines_to_drop(pages_lines, threshold=0.35)

    # Second pass: rebuild text without headers/footers/captions/page numbers
    cleaned_pages = []
    for lines in pages_lines:
        kept = []
        for l in lines:
            s = l.strip()
            if not s:
                kept.append("")
                continue
            if s in drop or is_page_number_line(s):
                continue
            # Remove obvious captions
            if re.match(r"^(figure|table)\s*\d+[:.\-]\s*", s, flags=re.I):
                continue
            kept.append(l)
        cleaned_pages.append("\n".join(kept))

    raw = "\n\n".join(cleaned_pages)

    # Global cleans
    raw = strip_toc_block(raw)
    raw = normalize_ws(raw)
    raw = strip_superscripts(raw)
    raw = strip_numeric_bracket_citations(raw)
    raw = strip_author_year_citations(raw)
    raw = remove_urls(raw)
    raw = remove_references_tail(raw)

    return raw.strip()



# ------------------------
# EPUB extraction (ebooklib)
# ------------------------

def html_to_items_with_headings(html: str) -> List[Tuple[Optional[str], str]]:
    """
    Convert one EPUB HTML document into a list of (heading_path, paragraph_text).
    heading_path is "H1 > H2 > H3 ..." if present.
    """
    soup = BeautifulSoup(html, "lxml")
    path: List[str] = []
    out: List[Tuple[Optional[str], str]] = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        name = el.name.lower()
        if name in {"h1", "h2", "h3", "h4"}:
            level = int(name[1])
            text = " ".join(el.get_text(" ", strip=True).split())
            if not text:
                continue
            while len(path) >= level:
                path.pop()
            path.append(text)
        elif name in {"p", "li"}:
            text = " ".join(el.get_text(" ", strip=True).split())
            if not text:
                continue
            breadcrumb = " > ".join(path) if path else None
            out.append((breadcrumb, text))
    return out

def split_very_long_sentence(s: str, max_words_per_piece: int = 120) -> List[str]:
    """If a single 'sentence' is enormous (e.g., table/no punctuation), split by words."""
    words = s.split()
    if len(words) <= max_words_per_piece:
        return [s]
    out = []
    for i in range(0, len(words), max_words_per_piece):
        out.append(" ".join(words[i:i+max_words_per_piece]))
    return out

def extract_items_epub(epub_path: Path) -> List[Tuple[Optional[str], str]]:
    """
    Returns a list of (heading_path_or_None, paragraph_text) across the entire EPUB.
    """
    book = epub.read_epub(str(epub_path))
    items: List[Tuple[Optional[str], str]] = []
    for it in book.get_items():
        mt = getattr(it, "media_type", "") or ""
        name = (getattr(it, "get_name", lambda: "")() or "").lower()
        is_doc = (
            mt in ("application/xhtml+xml", "text/html")
            or name.endswith(".xhtml")
            or name.endswith(".html")
        )
        if not is_doc:
            continue
        html = it.get_content().decode("utf-8", errors="ignore")
        items.extend(html_to_items_with_headings(html))
    return items


# ------------------------
# Chunking
# ------------------------

def chunk_by_sentences(
    text: str,
    max_words: int = 500,
    min_words: int = 180,
    overlap_sents: int = 1,
) -> List[str]:
    """
    Simple sentence-aware chunker for plain text (no headings).
    """
    chunks: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if not buf:
            return
        chunk = " ".join(buf).strip()
        if len(chunk.split()) < min_words and chunks:
            chunks[-1] = (chunks[-1] + " " + chunk).strip()
        else:
            chunks.append(chunk)
        buf = []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for p in paragraphs:
        sents = sent_tokenize(p)
        for i, s in enumerate(sents):
            for piece in split_very_long_sentence(s, max_words_per_piece=120):
                buf.append(piece)
                if len(" ".join(buf).split()) >= max_words:
                    flush()
                    if overlap_sents > 0:
                        # emulate a 1-sent overlap with the last 'piece'
                        buf.append(piece)
    flush()
    return chunks


def chunk_items_with_headings(
    items: List[Tuple[Optional[str], str]],
    max_words: int = 500,
    min_words: int = 180,
    overlap_sents: int = 1,
) -> List[Dict]:
    """
    Chunk a list of (heading_path, paragraph_text) pairs. Keeps heading_path in metadata.
    """
    chunks: List[Dict] = []
    buf_sents: List[str] = []
    current_heading: Optional[str] = None

    def flush():
        nonlocal buf_sents
        if not buf_sents:
            return
        text = " ".join(buf_sents).strip()
        if len(text.split()) < min_words and chunks:
            chunks[-1]["text"] = (chunks[-1]["text"] + " " + text).strip()
        else:
            chunks.append({"heading_path": current_heading, "text": text})
        buf_sents = []

    for heading, para in items:
        sents = sent_tokenize(para)
        if not sents:
            continue
        if current_heading != heading and buf_sents:
            flush()
        current_heading = heading
        for i, s in enumerate(sents):
            for piece in split_very_long_sentence(s, max_words_per_piece=120):
                buf_sents.append(piece)
                if len(" ".join(buf_sents).split()) >= max_words:
                    flush()
                    if overlap_sents > 0:
                        buf_sents.append(piece)
    flush()
    return chunks


# ------------------------
# Embedding / Index
# ------------------------

def embed_in_batches(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(vecs, dtype="float32")


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="guidelines", help="Root folder containing your guideline subfolders")
    ap.add_argument("--outdir", type=str, default="rag/index", help="Output folder for index and jsonl files")
    ap.add_argument("--model", type=str, default="BAAI/bge-m3", help="SentenceTransformer embedding model")
    ap.add_argument("--max_words", type=int, default=500)
    ap.add_argument("--min_words", type=int, default=180)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_texts: List[str] = []
    all_meta: List[Dict] = []

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])

    for corpus_dir in subdirs:
        corpus = corpus_dir.name
        files = sorted(list(corpus_dir.glob("**/*")))
        files = [f for f in files if f.is_file() and f.suffix.lower() in {".pdf", ".epub"}]
        if not files:
            continue

        print(f"\n=== Corpus: {corpus} ({len(files)} files) ===")
        for path in tqdm(files):
            ext = path.suffix.lower()
            title = path.stem.replace("_", " ").strip()
            sha = sha256_bytes(path.read_bytes())
            doc_id = f"{corpus}:{path.name}"

            try:
                if ext == ".pdf":
                    raw = extract_text_pdf(path)
                    if not raw:
                        continue
                    chunks = chunk_by_sentences(raw, max_words=args.max_words, min_words=args.min_words, overlap_sents=1)
                    for ch in chunks:
                        all_meta.append({
                            "corpus": corpus,
                            "doc_id": doc_id,
                            "title": title,
                            "section": None,
                            "heading_path": None,
                            "source_path": str(path),
                            "sha256": sha
                        })
                        # skip micro or low-signal chunks
                        if len(ch.strip().split()) < 40:
                            continue
                        all_texts.append(ch)

                elif ext == ".epub":
                    items = extract_items_epub(path)
                    items = filter_and_clean_epub_items(items)
                    if not items:
                        continue
                    cdicts = chunk_items_with_headings(items, max_words=args.max_words, min_words=args.min_words, overlap_sents=1)

                    
                    
                    if not items:
                        continue
                    cdicts = chunk_items_with_headings(items, max_words=args.max_words, min_words=args.min_words, overlap_sents=1)
                    for c in cdicts:
                        all_meta.append({
                            "corpus": corpus,
                            "doc_id": doc_id,
                            "title": title,
                            "section": None,
                            "heading_path": c.get("heading_path"),
                            "source_path": str(path),
                            "sha256": sha
                        })
                        if len(c["text"].strip().split()) < 40:
                            continue
                        all_texts.append(c["text"])

            except Exception as e:
                print(f"[WARN] Skipping {path} due to error: {e}")

    print(f"\nTotal chunks: {len(all_texts)}")
    if not all_texts:
        print("No chunks extracted. Exiting.")
        return

    model = SentenceTransformer(args.model)

    # Hard cap to avoid huge SDPA buffers (MPS/CPU friendly)
    try:
        model.max_seq_length = 512   # 384–512 is typical for SBERT-style models
    except Exception:
        pass

    # Hard char cap per chunk as a final safety net (~4–5k chars ≈ <1024 tokens for most BPEs)
    MAX_CHARS = 5000
    all_texts = [t[:MAX_CHARS] for t in all_texts]

    vecs = embed_in_batches(model, all_texts, batch_size=args.batch_size)

    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)          # cosine via normalized vectors
    index.add(vecs)

    # Save artifacts (names won't clash with Merck files)
    faiss_path = outdir / "guidelines.faiss"
    meta_path  = outdir / "guidelines_meta.jsonl"
    chunks_path= outdir / "guidelines_chunks.jsonl"
    vecs_path  = outdir / "guidelines_embeds.npy"

    faiss.write_index(index, str(faiss_path))
    np.save(vecs_path, vecs)

    with meta_path.open("w", encoding="utf-8") as f:
        for m in all_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    with chunks_path.open("w", encoding="utf-8") as f:
        for txt, m in zip(all_texts, all_meta):
            row = {"text": txt, **m}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved:\n  {faiss_path}\n  {meta_path}\n  {chunks_path}\n  {vecs_path}\nDone.")

if __name__ == "__main__":
    main()
