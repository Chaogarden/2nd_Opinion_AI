# rag/engine.py
# ==============================
# Reusable RAG engine for guidelines and Merck corpora
# - Shared sentence embedder
# - GuidelinesRAG and MerckRAG search classes
# - Query construction from extracted facts
# ==============================

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

# Add project root to path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.schema import EvidenceChunk
from extractor.schema import ExtractorJSON


class SentenceEmbedder:
    """
    Wrapper around SentenceTransformer for embedding text.
    Uses BAAI/bge-m3 by default (same as index build).
    """
    
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        Initialize the embedder.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings.
        """
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Embed a list of texts.
        
        Args:
            texts: List of strings to embed.
            normalize: Whether to L2-normalize the embeddings.
        
        Returns:
            numpy array of shape (len(texts), embed_dim) as float32.
        """
        embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        return embeddings.astype("float32")
    
    def embed_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """Embed a single text string."""
        return self.embed([text], normalize=normalize)


class GuidelinesRAG:
    """
    RAG search over the clinical guidelines FAISS index.
    """
    
    def __init__(
        self,
        index_path: str = "rag/index/guidelines.faiss",
        chunks_path: str = "rag/index/guidelines_chunks.jsonl",
        embedder: Optional[SentenceEmbedder] = None
    ):
        """
        Initialize the Guidelines RAG.
        
        Args:
            index_path: Path to the FAISS index file.
            chunks_path: Path to the JSONL file with chunk metadata.
            embedder: Optional shared SentenceEmbedder instance.
        """
        import faiss
        
        # Resolve paths relative to project root if needed
        self.index_path = Path(ROOT) / index_path if not Path(index_path).is_absolute() else Path(index_path)
        self.chunks_path = Path(ROOT) / chunks_path if not Path(chunks_path).is_absolute() else Path(chunks_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load chunk metadata
        self.chunks = self._load_jsonl(self.chunks_path)
        
        # Embedder (lazy init if not provided)
        self._embedder = embedder
    
    @property
    def embedder(self) -> SentenceEmbedder:
        """Get or create the embedder."""
        if self._embedder is None:
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file into a list of dicts."""
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_corpus: Optional[str] = None
    ) -> List[EvidenceChunk]:
        """
        Search for relevant guideline chunks.
        
        Args:
            query: Natural language query.
            k: Number of results to return.
            filter_corpus: Optional corpus name filter (e.g., 'cardiology_aha_acc').
        
        Returns:
            List of EvidenceChunk objects with evidence_ids like 'guidelines:123'.
        """
        # Embed query
        q_vec = self.embedder.embed_single(query)
        
        # Search (fetch extra if filtering)
        fetch_k = k * 3 if filter_corpus else k
        D, I = self.index.search(q_vec, fetch_k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            row = self.chunks[int(idx)]
            
            # Apply corpus filter if specified
            if filter_corpus and row.get("corpus") != filter_corpus:
                continue
            
            chunk = EvidenceChunk(
                evidence_id=f"guidelines:{idx}",
                source="guidelines",
                score=float(score),
                title=row.get("title") or "",
                heading_path=row.get("heading_path") or "",
                text=row.get("text") or "",
                meta={
                    "corpus": row.get("corpus") or "",
                    "doc_id": row.get("doc_id") or "",
                    "source_path": row.get("source_path") or "",
                    "position_in_doc": row.get("position_in_doc", 0),
                }
            )
            results.append(chunk)
            
            if len(results) >= k:
                break
        
        return results


class MerckRAG:
    """
    RAG search over the Merck Manual FAISS index.
    """
    
    def __init__(
        self,
        index_path: str = "rag/index/merck19.faiss",
        meta_path: str = "rag/index/merck19_meta.jsonl",
        embedder: Optional[SentenceEmbedder] = None
    ):
        """
        Initialize the Merck RAG.
        
        Args:
            index_path: Path to the FAISS index file.
            meta_path: Path to the JSONL file with chunk metadata.
            embedder: Optional shared SentenceEmbedder instance.
        """
        import faiss
        
        # Resolve paths relative to project root if needed
        self.index_path = Path(ROOT) / index_path if not Path(index_path).is_absolute() else Path(index_path)
        self.meta_path = Path(ROOT) / meta_path if not Path(meta_path).is_absolute() else Path(meta_path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_path))
        
        # Load metadata
        self.meta = self._load_jsonl(self.meta_path)
        
        # Embedder (lazy init if not provided)
        self._embedder = embedder
    
    @property
    def embedder(self) -> SentenceEmbedder:
        """Get or create the embedder."""
        if self._embedder is None:
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def _load_jsonl(self, path: Path) -> List[Dict]:
        """Load JSONL file into a list of dicts."""
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    
    def search(self, query: str, k: int = 5) -> List[EvidenceChunk]:
        """
        Search for relevant Merck Manual chunks.
        
        Args:
            query: Natural language query.
            k: Number of results to return.
        
        Returns:
            List of EvidenceChunk objects with evidence_ids like 'merck:123'.
        """
        # Embed query
        q_vec = self.embedder.embed_single(query)
        
        # Search
        D, I = self.index.search(q_vec, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            
            row = self.meta[int(idx)]
            
            # Use the 'id' field from meta if available, otherwise use index
            chunk_id = row.get("id", idx)
            
            chunk = EvidenceChunk(
                evidence_id=f"merck:{chunk_id}",
                source="merck",
                score=float(score),
                title="Merck Manual 19th Edition",
                heading_path=row.get("heading_path") or "",
                text=row.get("text") or "",
                meta={}
            )
            results.append(chunk)
        
        return results


def build_rag_query(extracted: ExtractorJSON) -> str:
    """
    Build a natural language query from extracted clinical facts.
    
    Composes a concise query from:
    - Chief complaint
    - Top present symptoms
    - Key vitals
    - Current medications
    - Risk factors
    
    Args:
        extracted: ExtractorJSON with normalized clinical findings.
    
    Returns:
        A query string suitable for semantic search.
    """
    parts = []
    
    # Chief complaint
    if extracted.chief_complaint:
        parts.append(f"Chief complaint: {extracted.chief_complaint}")
    
    # Symptoms (only present ones)
    present_symptoms = [
        s.name_norm or s.name_surface 
        for s in extracted.symptoms 
        if s.assertion == "present"
    ]
    if present_symptoms:
        symptom_str = ", ".join(present_symptoms[:5])  # Limit to top 5
        parts.append(f"Symptoms: {symptom_str}")
    
    # Key vitals
    vitals_strs = []
    for v in extracted.vitals:
        vitals_strs.append(f"{v.kind}={v.value}")
    if vitals_strs:
        parts.append(f"Vitals: {', '.join(vitals_strs)}")
    
    # Current medications
    current_meds = [
        m.name_norm or m.name_surface
        for m in extracted.meds
        if m.assertion == "present"
    ]
    if current_meds:
        meds_str = ", ".join(current_meds[:5])
        parts.append(f"Current medications: {meds_str}")
    
    # Risk factors
    if extracted.risk_factors:
        rf_str = ", ".join(extracted.risk_factors[:5])
        parts.append(f"Risk factors: {rf_str}")
    
    # Allergies
    allergies = [a.substance_surface for a in extracted.allergies if a.assertion == "present"]
    if allergies:
        parts.append(f"Allergies: {', '.join(allergies)}")
    
    if not parts:
        return "clinical presentation"
    
    return ". ".join(parts)


def collect_evidence(
    extracted: ExtractorJSON,
    k_guidelines: int = 5,
    k_merck: int = 5,
    embedder: Optional[SentenceEmbedder] = None,
    guidelines_filter: Optional[str] = None
) -> List[EvidenceChunk]:
    """
    Collect evidence from both RAG sources.
    
    Args:
        extracted: ExtractorJSON with clinical findings.
        k_guidelines: Number of guideline chunks to retrieve.
        k_merck: Number of Merck chunks to retrieve.
        embedder: Optional shared embedder instance.
        guidelines_filter: Optional corpus filter for guidelines.
    
    Returns:
        Combined list of EvidenceChunk objects from both sources.
    """
    # Build query from extracted facts
    query = build_rag_query(extracted)
    
    # Create shared embedder if not provided
    if embedder is None:
        embedder = SentenceEmbedder()
    
    results = []
    
    # Search guidelines
    try:
        guidelines_rag = GuidelinesRAG(embedder=embedder)
        guidelines_chunks = guidelines_rag.search(query, k=k_guidelines, filter_corpus=guidelines_filter)
        results.extend(guidelines_chunks)
    except Exception as e:
        print(f"Warning: Guidelines RAG search failed: {e}")
    
    # Search Merck
    try:
        merck_rag = MerckRAG(embedder=embedder)
        merck_chunks = merck_rag.search(query, k=k_merck)
        results.extend(merck_chunks)
    except Exception as e:
        print(f"Warning: Merck RAG search failed: {e}")
    
    return results


# CLI test entry point
if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser(description="Test RAG engine")
    ap.add_argument("--query", "-q", required=True, help="Search query")
    ap.add_argument("--k", type=int, default=3, help="Number of results per source")
    args = ap.parse_args()
    
    print(f"\nSearching for: {args.query}\n")
    
    embedder = SentenceEmbedder()
    
    print("=" * 60)
    print("GUIDELINES RESULTS")
    print("=" * 60)
    guidelines = GuidelinesRAG(embedder=embedder)
    for chunk in guidelines.search(args.query, k=args.k):
        print(f"\n[{chunk.evidence_id}] score={chunk.score:.3f}")
        print(f"Title: {chunk.title}")
        print(f"Heading: {chunk.heading_path}")
        print(f"Text: {chunk.text[:300]}...")
    
    print("\n" + "=" * 60)
    print("MERCK RESULTS")
    print("=" * 60)
    merck = MerckRAG(embedder=embedder)
    for chunk in merck.search(args.query, k=args.k):
        print(f"\n[{chunk.evidence_id}] score={chunk.score:.3f}")
        print(f"Heading: {chunk.heading_path}")
        print(f"Text: {chunk.text[:300]}...")

