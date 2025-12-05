#!/usr/bin/env python3
"""
Ingest mixed PDFs + EPUBs under a guidelines/ root into a single FAISS index.

Outputs (in --outdir):
  - guidelines.faiss
  - guidelines_meta.jsonl      (metadata per chunk, no text)
  - guidelines_chunks.jsonl    (text + metadata per line)
  - guidelines_embeds.npy      (optional vector cache)

Does not touch your existing Merck19 index files.

## LLM Cleaning Hook (optional)
Set --llm_cleaner to a provider name (e.g. "openai") and ensure the corresponding
environment variables are set (e.g. OPENAI_API_KEY). By default, no LLM calls are made.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
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
nltk.download("punkt_tab", quiet=True)


# ========================
# CONFIGURATION CONSTANTS
# ========================

# Minimum word count for a chunk to be embedded
MIN_CHUNK_WORDS = 40

# Front-matter: skip first N paragraphs if they look like titles/intro
FRONT_MATTER_SKIP_LINES = 15

# Boilerplate phrases that indicate non-guideline content (case-insensitive)
BOILERPLATE_PHRASES = [
    "all rights reserved",
    "isbn",
    "writing committee",
    "acknowledgments",
    "acknowledgements",
    "conflict of interest",
    "conflicts of interest",
    "disclosures",
    "funding",
    "financial support",
    "author contributions",
    "authors' contributions",
    "correspondence:",
    "reprint requests",
    "reprints:",
    "copyright ©",
    "copyright (c)",
    "published by",
    "doi:",
    "e-mail:",
    "email:",
    "fax:",
    "tel:",
    "address:",
    "received:",
    "accepted:",
    "revised:",
    "online publication",
    "epub ahead",
    "supplementary material",
    "supplemental material",
    "appendix",
    "abbreviations:",
    "keywords:",
    "key words:",
]

# Section headings to skip entirely (case-insensitive, leaf heading match)
SKIP_SECTION_HEADINGS = {
    "references",
    "bibliography",
    "works cited",
    "acknowledgments",
    "acknowledgements",
    "about the authors",
    "author information",
    "conflicts of interest",
    "disclosures",
    "funding",
    "abbreviations",
    "table of contents",
    "contents",
    "index",
    "appendix",
    "supplementary",
    "supplemental",
    "author contributions",
}


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


# ------------------------
# NEW: Enhanced noise filters
# ------------------------

def contains_boilerplate_phrase(text: str) -> bool:
    """Check if text contains any boilerplate phrase indicating non-guideline content."""
    lower = text.lower()
    for phrase in BOILERPLATE_PHRASES:
        if phrase in lower:
            return True
    return False


def is_reference_entry(para: str) -> bool:
    """
    Detect if a paragraph looks like a numbered reference entry.
    E.g., "1. Smith J, Doe A. Title of paper. J Med. 2020;45(3):123-130."
    """
    para = para.strip()
    if not para:
        return False
    
    # Pattern: starts with number followed by period/parenthesis
    numbered_start = re.match(r"^\d{1,3}[\.\)]\s*", para)
    if not numbered_start:
        return False
    
    # Look for journal-style patterns: volume(issue):pages or year;volume
    journal_patterns = [
        r"\d{4}\s*;\s*\d+",  # 2020;45
        r"\d+\s*\(\s*\d+\s*\)\s*:\s*\d+",  # 45(3):123
        r"\d+\s*:\s*\d+\s*[-–]\s*\d+",  # 45:123-130
        r"[A-Z][a-z]+\s+[A-Z][A-Z]?\s*[,;]",  # Author initials pattern
        r"et al\.",  # et al.
        r"pp?\.\s*\d+",  # p. 123 or pp. 123
    ]
    
    matches = sum(1 for p in journal_patterns if re.search(p, para))
    return matches >= 2


def is_front_matter_line(line: str, line_idx: int) -> bool:
    """
    Detect if a line in the first N lines looks like front-matter (title, authors, affiliations).
    """
    if line_idx >= FRONT_MATTER_SKIP_LINES:
        return False
    
    line = line.strip()
    if not line:
        return False
    
    # Very short lines at the start are often titles/headers
    words = line.split()
    if len(words) <= 3 and line_idx < 5:
        return True
    
    # Lines with mostly uppercase (titles)
    alpha_chars = [c for c in line if c.isalpha()]
    if alpha_chars and sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) > 0.7:
        return True
    
    # Lines with author-like patterns (names with degrees/affiliations)
    author_patterns = [
        r"^[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+",  # John A. Smith
        r",\s*(MD|PhD|MPH|RN|DO|FACC|FAHA|FACS)\b",  # Degrees
        r"^\d+\s*[A-Z][a-z]+\s+(University|Hospital|Institute|College|School|Center|Centre)",  # Affiliations
    ]
    for p in author_patterns:
        if re.search(p, line):
            return True
    
    return False


def filter_paragraphs_pdf(paragraphs: List[str]) -> List[str]:
    """
    Filter out front-matter, reference entries, and boilerplate paragraphs from PDF text.
    """
    filtered = []
    for idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        # Skip front-matter (first few paragraphs that look like titles/authors)
        lines = para.split('\n')
        if idx < 3 and all(is_front_matter_line(l, i) for i, l in enumerate(lines) if l.strip()):
            continue
        
        # Skip reference entries
        if is_reference_entry(para):
            continue
        
        # Skip boilerplate
        if contains_boilerplate_phrase(para):
            # Only skip if it's a short paragraph dominated by boilerplate
            if len(para.split()) < 100:
                continue
        
        # Skip paragraphs that are mostly numbers/punctuation (tables, etc.)
        alpha_ratio = sum(1 for c in para if c.isalpha()) / max(1, len(para))
        if alpha_ratio < 0.4:
            continue
        
        filtered.append(para)
    
    return filtered


def should_skip_heading(heading: Optional[str]) -> bool:
    """Check if a heading indicates a section we should skip entirely."""
    if not heading:
        return False
    
    # Get the leaf heading (last part after " > ")
    parts = heading.strip().lower().split(" > ")
    leaf = parts[-1].strip() if parts else ""
    
    # Check against skip list
    for skip_term in SKIP_SECTION_HEADINGS:
        if skip_term in leaf:
            return True
    
    return False


def filter_and_clean_epub_items(items: List[Tuple[Optional[str], str]]) -> List[Tuple[Optional[str], str]]:
    """
    Filter EPUB items to remove references, boilerplate, and front-matter.
    """
    out = []
    seen_substantive = False
    
    for idx, (heading, para) in enumerate(items):
        # Skip sections by heading
        if should_skip_heading(heading):
            continue
        
        # Clean the paragraph text
        txt = para.strip()
        txt = strip_superscripts(txt)
        txt = strip_numeric_bracket_citations(txt)
        txt = strip_author_year_citations(txt)
        txt = remove_urls(txt)
        txt = normalize_ws(txt)
        
        if not txt:
            continue
        
        # Skip reference entries
        if is_reference_entry(txt):
            continue
        
        # Skip front-matter: first few short paragraphs before substantive content
        words = txt.split()
        if not seen_substantive:
            # Mark as substantive if we have a paragraph with decent length
            if len(words) >= 50:
                seen_substantive = True
            elif idx < 10:
                # Skip short early paragraphs (titles, intros)
                if len(words) < 30:
                    # Check if it looks like a title or author line
                    if contains_boilerplate_phrase(txt) or len(words) < 10:
                        continue
        
        # Skip boilerplate paragraphs
        if contains_boilerplate_phrase(txt) and len(words) < 100:
            continue
        
        # Skip paragraphs with low alphabetic content
        alpha_ratio = sum(1 for c in txt if c.isalpha()) / max(1, len(txt))
        if alpha_ratio < 0.4:
            continue
        
        out.append((heading, txt))
    
    return out


# ------------------------
# PDF extraction (PyMuPDF)
# ------------------------

def extract_text_pdf(pdf_path: Path) -> Tuple[str, List[str]]:
    """
    Extract and clean text from a PDF.
    Returns (full_cleaned_text, list_of_filtered_paragraphs).
    """
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

    # Split into paragraphs and filter out noise
    paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
    filtered_paragraphs = filter_paragraphs_pdf(paragraphs)
    
    # Rejoin for backward compatibility
    cleaned_text = "\n\n".join(filtered_paragraphs)

    return cleaned_text.strip(), filtered_paragraphs



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
) -> List[Dict]:
    """
    Simple sentence-aware chunker for plain text (no headings).
    Returns list of dicts with 'text' and 'position_in_doc'.
    """
    chunks: List[Dict] = []
    buf: List[str] = []
    chunk_idx = 0

    def flush():
        nonlocal buf, chunk_idx
        if not buf:
            return
        chunk = " ".join(buf).strip()
        if len(chunk.split()) < min_words and chunks:
            chunks[-1]["text"] = (chunks[-1]["text"] + " " + chunk).strip()
        else:
            chunks.append({"text": chunk, "position_in_doc": chunk_idx})
            chunk_idx += 1
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
    Chunk a list of (heading_path, paragraph_text) pairs. 
    Keeps heading_path and position_in_doc in metadata.
    """
    chunks: List[Dict] = []
    buf_sents: List[str] = []
    current_heading: Optional[str] = None
    chunk_idx = 0

    def flush():
        nonlocal buf_sents, chunk_idx
        if not buf_sents:
            return
        text = " ".join(buf_sents).strip()
        if len(text.split()) < min_words and chunks:
            chunks[-1]["text"] = (chunks[-1]["text"] + " " + text).strip()
        else:
            chunks.append({
                "heading_path": current_heading, 
                "text": text,
                "position_in_doc": chunk_idx
            })
            chunk_idx += 1
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
# Post-chunk filtering
# ------------------------

def is_low_signal_chunk(text: str) -> bool:
    """
    Check if a chunk is low-signal (mostly boilerplate, names, or non-content).
    """
    words = text.split()
    if len(words) < MIN_CHUNK_WORDS:
        return True
    
    # Check alphabetic ratio
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    if alpha_ratio < 0.5:
        return True
    
    # Check for boilerplate dominance in short chunks
    if len(words) < 80 and contains_boilerplate_phrase(text):
        return True
    
    return False


# ------------------------
# Optional LLM Cleaning Hook
# ------------------------

def maybe_llm_clean(text: str, meta: Dict, args: Any) -> str:
    """
    Optional LLM-based cleaning hook.
    By default (--llm_cleaner none), returns text unchanged.
    
    To implement LLM cleaning:
    1. Set --llm_cleaner to a provider name (e.g., "openai")
    2. Set appropriate environment variables (e.g., OPENAI_API_KEY)
    3. Implement the cleaning logic below for your provider
    
    Example prompt for cleaning:
    "Clean this medical guideline text by removing any remaining citations,
    author credits, or boilerplate. Keep only the clinical recommendations
    and explanatory content. Return the cleaned text only."
    """
    if not hasattr(args, 'llm_cleaner') or args.llm_cleaner == "none":
        return text
    
    # Placeholder for LLM integration
    # Users can implement their preferred LLM client here
    provider = args.llm_cleaner
    
    if provider == "openai":
        # Example OpenAI integration (requires openai package and OPENAI_API_KEY)
        try:
            import os
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical text cleaner. Remove citations, author credits, and boilerplate from medical guideline text. Keep only clinical recommendations and explanatory content. Return cleaned text only, no commentary."},
                    {"role": "user", "content": text[:4000]}  # Truncate to avoid token limits
                ],
                max_tokens=2000,
                temperature=0.0
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[WARN] LLM cleaning failed: {e}")
            return text
    
    # Add other providers as needed
    print(f"[WARN] Unknown LLM provider: {provider}, returning original text")
    return text


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
    ap = argparse.ArgumentParser(
        description="Ingest medical guideline PDFs and EPUBs into a FAISS index.",
        epilog="""
LLM Cleaning Hook:
  Set --llm_cleaner to a provider name (e.g., "openai") for optional LLM-based
  chunk cleaning. Requires appropriate environment variables (e.g., OPENAI_API_KEY).
  Default is "none" (deterministic heuristics only).
        """
    )
    ap.add_argument("--root", type=str, default="guidelines", help="Root folder containing your guideline subfolders")
    ap.add_argument("--outdir", type=str, default="rag/index", help="Output folder for index and jsonl files")
    ap.add_argument("--model", type=str, default="BAAI/bge-m3", help="SentenceTransformer embedding model")
    ap.add_argument("--max_words", type=int, default=500)
    ap.add_argument("--min_words", type=int, default=180)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--llm_cleaner", type=str, default="none", 
                    help="LLM provider for optional cleaning (none, openai). Default: none")
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
                    cleaned_text, _ = extract_text_pdf(path)
                    if not cleaned_text:
                        continue
                    chunk_dicts = chunk_by_sentences(
                        cleaned_text, 
                        max_words=args.max_words, 
                        min_words=args.min_words, 
                        overlap_sents=1
                    )
                    
                    for c in chunk_dicts:
                        chunk_text = c["text"]
                        
                        # Skip low-signal chunks
                        if is_low_signal_chunk(chunk_text):
                            continue
                        
                        # Optional LLM cleaning
                        chunk_text = maybe_llm_clean(chunk_text, c, args)
                        
                        # Build metadata (only for chunks we keep)
                        meta = {
                            "corpus": corpus,
                            "doc_id": doc_id,
                            "title": title,
                            "section": None,
                            "heading_path": None,
                            "position_in_doc": c.get("position_in_doc", 0),
                            "source_path": str(path),
                            "sha256": sha
                        }
                        all_meta.append(meta)
                        all_texts.append(chunk_text)

                elif ext == ".epub":
                    items = extract_items_epub(path)
                    items = filter_and_clean_epub_items(items)
                    if not items:
                        continue
                    
                    chunk_dicts = chunk_items_with_headings(
                        items, 
                        max_words=args.max_words, 
                        min_words=args.min_words, 
                        overlap_sents=1
                    )
                    
                    for c in chunk_dicts:
                        chunk_text = c["text"]
                        
                        # Skip low-signal chunks
                        if is_low_signal_chunk(chunk_text):
                            continue
                        
                        # Optional LLM cleaning
                        chunk_text = maybe_llm_clean(chunk_text, c, args)
                        
                        # Build metadata (only for chunks we keep)
                        meta = {
                            "corpus": corpus,
                            "doc_id": doc_id,
                            "title": title,
                            "section": None,
                            "heading_path": c.get("heading_path"),
                            "position_in_doc": c.get("position_in_doc", 0),
                            "source_path": str(path),
                            "sha256": sha
                        }
                        all_meta.append(meta)
                        all_texts.append(chunk_text)

            except Exception as e:
                print(f"[WARN] Skipping {path} due to error: {e}")

    print(f"\nTotal chunks: {len(all_texts)}")
    if not all_texts:
        print("No chunks extracted. Exiting.")
        return

    # Verify metadata alignment
    assert len(all_texts) == len(all_meta), f"Metadata mismatch: {len(all_texts)} texts vs {len(all_meta)} meta"

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
