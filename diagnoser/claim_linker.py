# diagnoser/claim_linker.py
# ==============================
# Claim Extraction and Evidence Linkage
# Links note claims to dialogue evidence using embeddings and NLI
# ==============================

import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Literal
from dataclasses import dataclass, field

import numpy as np

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.critic_schema import (
    ClaimJudgment, EvidenceCitation, SOAPSection, VerdictType
)


# ============================================================
# Data Structures
# ============================================================

@dataclass
class DialogueSentence:
    """A sentence/turn from the dialogue."""
    idx: int
    text: str
    role: str  # "DOCTOR" or "PATIENT"
    start: int = 0
    end: int = 0


@dataclass
class NoteClaim:
    """An atomic claim extracted from a clinical note."""
    claim_id: str
    text: str
    section: SOAPSection
    claim_type: Literal["finding", "diagnosis", "medication", "test", "plan", "other"]
    sentence_idx: int = 0
    normalized_form: Optional[str] = None


@dataclass
class LinkageResult:
    """Result of linking a claim to dialogue evidence."""
    claim: NoteClaim
    verdict: VerdictType
    confidence: float
    evidence_sentences: List[DialogueSentence]
    similarity_scores: List[float]


# ============================================================
# Dialogue Parsing
# ============================================================

def parse_dialogue_sentences(dialogue_text: str) -> List[DialogueSentence]:
    """
    Parse dialogue text into sentences with role information.
    
    Args:
        dialogue_text: Raw dialogue with [doctor]/[patient] tags
    
    Returns:
        List of DialogueSentence objects
    """
    # Pattern for speaker tags
    speaker_pattern = re.compile(r'\[(?P<role>doctor|patient)\]', re.IGNORECASE)
    
    sentences = []
    
    # Find all speaker tag positions
    matches = list(speaker_pattern.finditer(dialogue_text))
    
    if not matches:
        # No tags, treat as single sentence
        if dialogue_text.strip():
            sentences.append(DialogueSentence(
                idx=0,
                text=dialogue_text.strip(),
                role="UNKNOWN",
                start=0,
                end=len(dialogue_text)
            ))
        return sentences
    
    for i, match in enumerate(matches):
        role = match.group("role").upper()
        
        # Get text between this tag and next
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(dialogue_text)
        
        turn_text = dialogue_text[start_pos:end_pos].strip()
        
        if turn_text:
            sentences.append(DialogueSentence(
                idx=len(sentences),
                text=turn_text,
                role=role,
                start=start_pos,
                end=end_pos
            ))
    
    return sentences


def parse_dialogue_turns(turns: List[Dict[str, Any]]) -> List[DialogueSentence]:
    """
    Convert turn dicts to DialogueSentence objects.
    
    Args:
        turns: List of turn dicts with 'utt_id', 'role', 'text'
    
    Returns:
        List of DialogueSentence objects
    """
    return [
        DialogueSentence(
            idx=t.get("utt_id", i),
            text=t.get("text", ""),
            role=t.get("role", "UNKNOWN").upper()
        )
        for i, t in enumerate(turns)
        if t.get("text", "").strip()
    ]


# ============================================================
# Note Claim Extraction
# ============================================================

def detect_soap_sections(note_text: str) -> Dict[str, Tuple[int, int]]:
    """
    Detect SOAP section boundaries in a note.
    
    Args:
        note_text: SOAP note text
    
    Returns:
        Dict mapping section name to (start, end) positions
    """
    sections = {}
    
    patterns = [
        (r'(?i)^\s*(subjective|s)[:\s]', 'S'),
        (r'(?i)^\s*(objective|o)[:\s]', 'O'),
        (r'(?i)^\s*(assessment|a)[:\s]', 'A'),
        (r'(?i)^\s*(plan|p)[:\s]', 'P'),
        (r'(?i)^\s*(history\s+of\s+present\s+illness|hpi)[:\s]', 'S'),
        (r'(?i)^\s*(physical\s+exam|pe|exam)[:\s]', 'O'),
        (r'(?i)^\s*(impression|dx|diagnosis|diagnoses)[:\s]', 'A'),
        (r'(?i)^\s*(recommendations?|rx|treatment)[:\s]', 'P'),
        (r'(?i)^\s*(review\s+of\s+systems|ros)[:\s]', 'S'),
        (r'(?i)^\s*(vitals?|vital\s+signs?)[:\s]', 'O'),
    ]
    
    section_starts = []
    for pattern, section_name in patterns:
        for match in re.finditer(pattern, note_text, re.MULTILINE):
            section_starts.append((match.start(), match.end(), section_name))
    
    section_starts.sort(key=lambda x: x[0])
    
    for i, (start, header_end, name) in enumerate(section_starts):
        if i + 1 < len(section_starts):
            end = section_starts[i + 1][0]
        else:
            end = len(note_text)
        
        # Only keep first occurrence of each section
        if name not in sections:
            sections[name] = (header_end, end)
    
    return sections


def extract_claims_from_note(
    note_text: str,
    note_type: Literal["prose", "soap"] = "prose"
) -> List[NoteClaim]:
    """
    Extract atomic claims from a clinical note.
    
    Args:
        note_text: Raw note text
        note_type: "prose" or "soap"
    
    Returns:
        List of NoteClaim objects
    """
    claims = []
    claim_idx = 0
    
    if note_type == "soap":
        sections = detect_soap_sections(note_text)
        
        for section_name, (start, end) in sections.items():
            section_text = note_text[start:end].strip()
            section_claims = _extract_claims_from_section(
                section_text, section_name, claim_idx
            )
            claims.extend(section_claims)
            claim_idx += len(section_claims)
        
        # Handle any text not in sections
        if not sections:
            claims = _extract_claims_from_section(note_text, "unknown", 0)
    else:
        # Prose note - treat as single section
        claims = _extract_claims_from_section(note_text, "unknown", 0)
    
    return claims


def _extract_claims_from_section(
    section_text: str,
    section_name: str,
    start_idx: int
) -> List[NoteClaim]:
    """Extract claims from a note section."""
    claims = []
    
    # Split into sentences
    sentences = _split_into_sentences(section_text)
    
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence) < 5:
            continue
        
        # Determine claim type based on content and section
        claim_type = _classify_claim_type(sentence, section_name)
        
        claims.append(NoteClaim(
            claim_id=f"claim_{start_idx + len(claims)}",
            text=sentence,
            section=section_name if section_name in ["S", "O", "A", "P"] else "unknown",
            claim_type=claim_type,
            sentence_idx=i,
            normalized_form=_normalize_claim(sentence)
        ))
    
    return claims


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Split on common sentence boundaries
    # Handle numbered lists, bullet points, and standard punctuation
    
    sentences = []
    
    # First split by newlines (preserving list items)
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Remove bullet points and numbers at start
        line = re.sub(r'^[\-\*\•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        
        # Split on sentence-ending punctuation
        parts = re.split(r'(?<=[.!?])\s+', line)
        sentences.extend([p.strip() for p in parts if p.strip()])
    
    return sentences


def _classify_claim_type(sentence: str, section: str) -> str:
    """Classify the type of claim based on content and section."""
    lower = sentence.lower()
    
    # Medication patterns
    med_patterns = [
        r'\b(mg|mcg|ml|tablet|capsule|daily|bid|tid|qid|prn)\b',
        r'\b(prescribe|start|continue|take|taking|medication|drug)\b',
    ]
    for pattern in med_patterns:
        if re.search(pattern, lower):
            return "medication"
    
    # Diagnosis patterns
    dx_patterns = [
        r'\b(diagnos|dx|rule out|r/o|suspect|consistent with|impression)\b',
        r'\b(disease|syndrome|disorder|infection|failure)\b',
    ]
    for pattern in dx_patterns:
        if re.search(pattern, lower):
            return "diagnosis"
    
    # Test/workup patterns
    test_patterns = [
        r'\b(order|lab|test|ct|mri|xray|x-ray|ecg|ekg|ultrasound|blood|urine)\b',
        r'\b(check|measure|obtain|send|draw)\b',
    ]
    for pattern in test_patterns:
        if re.search(pattern, lower):
            return "test"
    
    # Plan patterns
    plan_patterns = [
        r'\b(follow.?up|return|refer|consult|admit|discharge)\b',
        r'\b(recommend|advise|counsel|educate)\b',
    ]
    for pattern in plan_patterns:
        if re.search(pattern, lower):
            return "plan"
    
    # Section-based defaults
    if section == "A":
        return "diagnosis"
    elif section == "P":
        return "plan"
    elif section in ["S", "O"]:
        return "finding"
    
    return "other"


def _normalize_claim(sentence: str) -> str:
    """Normalize a claim for comparison."""
    # Lowercase and remove extra whitespace
    normalized = sentence.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    
    # Remove punctuation
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    return normalized


# ============================================================
# Embedding-based Evidence Linkage
# ============================================================

class EvidenceLinker:
    """
    Links note claims to dialogue evidence using embeddings.
    
    Uses sentence embeddings to find semantically similar dialogue
    sentences for each note claim.
    """
    
    def __init__(self, embedder=None, nli_model=None):
        """
        Initialize the evidence linker.
        
        Args:
            embedder: SentenceEmbedder instance (will create if None)
            nli_model: Optional NLI model for verdict classification
        """
        self._embedder = embedder
        self._nli_model = nli_model
        
        # Cache for dialogue embeddings
        self._dialogue_cache: Dict[int, np.ndarray] = {}
    
    @property
    def embedder(self):
        """Get or create the sentence embedder."""
        if self._embedder is None:
            from rag.engine import SentenceEmbedder
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    def link_claims_to_dialogue(
        self,
        claims: List[NoteClaim],
        dialogue_sentences: List[DialogueSentence],
        top_k: int = 3,
        similarity_threshold: float = 0.3,
        use_nli: bool = False
    ) -> List[LinkageResult]:
        """
        Link each claim to supporting dialogue evidence.
        
        Args:
            claims: List of note claims
            dialogue_sentences: List of dialogue sentences
            top_k: Number of top evidence sentences to return
            similarity_threshold: Minimum similarity to consider as evidence
            use_nli: Whether to use NLI for verdict classification
        
        Returns:
            List of LinkageResult for each claim
        """
        if not dialogue_sentences:
            return [
                LinkageResult(
                    claim=claim,
                    verdict="unsupported",
                    confidence=1.0,
                    evidence_sentences=[],
                    similarity_scores=[]
                )
                for claim in claims
            ]
        
        # Embed dialogue sentences
        dialogue_texts = [s.text for s in dialogue_sentences]
        dialogue_embeds = self.embedder.embed(dialogue_texts)
        
        # Embed claims
        claim_texts = [c.text for c in claims]
        claim_embeds = self.embedder.embed(claim_texts)
        
        # Compute similarities
        similarities = np.dot(claim_embeds, dialogue_embeds.T)
        
        results = []
        for i, claim in enumerate(claims):
            # Get top-k most similar dialogue sentences
            sim_scores = similarities[i]
            top_indices = np.argsort(sim_scores)[::-1][:top_k]
            
            # Filter by threshold
            evidence_sents = []
            evidence_scores = []
            for idx in top_indices:
                score = sim_scores[idx]
                if score >= similarity_threshold:
                    evidence_sents.append(dialogue_sentences[idx])
                    evidence_scores.append(float(score))
            
            # Determine verdict
            if use_nli and evidence_sents:
                verdict, confidence = self._nli_verdict(claim.text, evidence_sents)
            else:
                verdict, confidence = self._heuristic_verdict(
                    claim, evidence_sents, evidence_scores
                )
            
            results.append(LinkageResult(
                claim=claim,
                verdict=verdict,
                confidence=confidence,
                evidence_sentences=evidence_sents,
                similarity_scores=evidence_scores
            ))
        
        return results
    
    def _heuristic_verdict(
        self,
        claim: NoteClaim,
        evidence_sents: List[DialogueSentence],
        scores: List[float]
    ) -> Tuple[VerdictType, float]:
        """
        Determine verdict using heuristics.
        
        Args:
            claim: The claim being evaluated
            evidence_sents: Supporting dialogue sentences
            scores: Similarity scores
        
        Returns:
            (verdict, confidence) tuple
        """
        if not evidence_sents:
            return "unsupported", 0.9
        
        max_score = max(scores) if scores else 0
        
        # Check for negation patterns
        claim_lower = claim.text.lower()
        has_negation = any(neg in claim_lower for neg in ["no ", "denies", "negative", "absent", "without"])
        
        for sent, score in zip(evidence_sents, scores):
            sent_lower = sent.text.lower()
            sent_has_negation = any(neg in sent_lower for neg in ["no ", "denies", "don't", "doesn't", "negative"])
            
            # If claim says something is absent but dialogue says present (or vice versa)
            if has_negation != sent_has_negation and score > 0.5:
                return "contradicted", min(0.9, score)
        
        # High similarity = entailed
        if max_score > 0.6:
            return "entailed", max_score
        elif max_score > 0.4:
            return "entailed", max_score * 0.8  # Lower confidence
        else:
            return "unsupported", 1.0 - max_score
    
    def _nli_verdict(
        self,
        claim_text: str,
        evidence_sents: List[DialogueSentence]
    ) -> Tuple[VerdictType, float]:
        """
        Determine verdict using NLI model.
        
        Args:
            claim_text: The claim text
            evidence_sents: Supporting dialogue sentences
        
        Returns:
            (verdict, confidence) tuple
        """
        if self._nli_model is None:
            return self._heuristic_verdict(
                NoteClaim(claim_id="", text=claim_text, section="unknown", claim_type="other"),
                evidence_sents,
                [0.5] * len(evidence_sents)
            )
        
        # Combine evidence sentences as premise
        premise = " ".join(s.text for s in evidence_sents)
        
        # Run NLI
        try:
            result = self._nli_model(premise, claim_text)
            
            # Map NLI labels to our verdicts
            label_map = {
                "entailment": "entailed",
                "contradiction": "contradicted",
                "neutral": "unsupported"
            }
            
            verdict = label_map.get(result["label"], "unsupported")
            confidence = result.get("score", 0.5)
            
            return verdict, confidence
        except Exception:
            return "unsupported", 0.5


def build_claim_judgments(
    linkage_results: List[LinkageResult]
) -> List[ClaimJudgment]:
    """
    Convert linkage results to ClaimJudgment objects.
    
    Args:
        linkage_results: List of LinkageResult from evidence linker
    
    Returns:
        List of ClaimJudgment for the critic output
    """
    judgments = []
    
    for result in linkage_results:
        # Build evidence citations
        citations = []
        for sent, score in zip(result.evidence_sentences, result.similarity_scores):
            citations.append(EvidenceCitation(
                evidence_id=f"dialogue:{sent.idx}",
                source="dialogue",
                snippet=sent.text[:100] + "..." if len(sent.text) > 100 else sent.text,
                score=score
            ))
        
        judgments.append(ClaimJudgment(
            claim_id=result.claim.claim_id,
            claim_text=result.claim.text,
            section=result.claim.section,
            verdict=result.verdict,
            dialogue_sentence_ids=[s.idx for s in result.evidence_sentences],
            dialogue_evidence=citations,
            confidence=result.confidence
        ))
    
    return judgments


# ============================================================
# LLM-based NLI Verification (Optional)
# ============================================================

NLI_PROMPT = """You are an expert at natural language inference. Given a PREMISE (from a doctor-patient dialogue) and a CLAIM (from a clinical note), determine if the claim is:
- ENTAILED: The claim is supported by the dialogue
- CONTRADICTED: The claim conflicts with the dialogue
- UNSUPPORTED: The claim has no evidence in the dialogue

PREMISE (dialogue excerpt):
{premise}

CLAIM (from note):
{claim}

Respond with ONLY one word: ENTAILED, CONTRADICTED, or UNSUPPORTED"""


class LLMNLIVerifier:
    """
    LLM-based NLI verification for claim-dialogue linkage.
    """
    
    def __init__(self, client, model: str):
        """
        Initialize the LLM NLI verifier.
        
        Args:
            client: LLM client with .chat() method
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def verify(
        self,
        claim_text: str,
        dialogue_sentences: List[DialogueSentence]
    ) -> Tuple[VerdictType, float]:
        """
        Verify a claim against dialogue evidence.
        
        Args:
            claim_text: The claim to verify
            dialogue_sentences: Supporting dialogue sentences
        
        Returns:
            (verdict, confidence) tuple
        """
        if not dialogue_sentences:
            return "unsupported", 1.0
        
        # Combine dialogue as premise
        premise = " ".join(f"[{s.role}]: {s.text}" for s in dialogue_sentences)
        
        prompt = NLI_PROMPT.format(premise=premise, claim=claim_text)
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            if isinstance(response, str):
                content = response
            elif isinstance(response, dict):
                content = response.get("content", "")
            else:
                content = str(response)
            
            content = content.strip().upper()
            
            if "ENTAIL" in content:
                return "entailed", 0.85
            elif "CONTRADICT" in content:
                return "contradicted", 0.85
            else:
                return "unsupported", 0.85
                
        except Exception as e:
            print(f"LLM NLI failed: {e}")
            return "unsupported", 0.5


# ============================================================
# Convenience Functions
# ============================================================

def link_note_to_dialogue(
    note_text: str,
    dialogue_text: str,
    note_type: Literal["prose", "soap"] = "prose",
    use_llm_nli: bool = False,
    llm_client=None,
    llm_model: str = ""
) -> Tuple[List[NoteClaim], List[ClaimJudgment]]:
    """
    High-level function to link a note to dialogue evidence.
    
    Args:
        note_text: Clinical note text
        dialogue_text: Dialogue text with [doctor]/[patient] tags
        note_type: "prose" or "soap"
        use_llm_nli: Whether to use LLM for NLI verification
        llm_client: LLM client (required if use_llm_nli=True)
        llm_model: LLM model name
    
    Returns:
        (claims, judgments) tuple
    """
    # Parse dialogue
    dialogue_sentences = parse_dialogue_sentences(dialogue_text)
    
    # Extract claims from note
    claims = extract_claims_from_note(note_text, note_type)
    
    # Set up linker
    nli_model = None
    if use_llm_nli and llm_client:
        nli_model = LLMNLIVerifier(llm_client, llm_model)
    
    linker = EvidenceLinker(nli_model=nli_model)
    
    # Link claims to dialogue
    linkage_results = linker.link_claims_to_dialogue(
        claims, dialogue_sentences, use_nli=use_llm_nli
    )
    
    # Convert to judgments
    judgments = build_claim_judgments(linkage_results)
    
    return claims, judgments


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    # Simple test
    test_dialogue = """
    [doctor] What brings you in today?
    [patient] I've been having chest pain for the past 2 days.
    [doctor] Can you describe the pain?
    [patient] It's a sharp pain in my chest, worse when I breathe deeply.
    [doctor] Any shortness of breath?
    [patient] Yes, a little bit.
    [doctor] Do you have any allergies to medications?
    [patient] I'm allergic to penicillin, I get a rash.
    """
    
    test_note = """
    S: Patient presents with 2-day history of chest pain, described as sharp and pleuritic.
    Associated with mild dyspnea. Denies fever or cough.
    
    A: Likely pleurisy vs musculoskeletal chest pain.
    
    P: Order chest X-ray. Start ibuprofen 400mg TID for pain.
    """
    
    print("Testing claim-dialogue linkage...")
    print("=" * 60)
    
    claims, judgments = link_note_to_dialogue(
        test_note, test_dialogue, note_type="soap"
    )
    
    print(f"\nExtracted {len(claims)} claims:")
    for claim in claims:
        print(f"  [{claim.section}] {claim.claim_type}: {claim.text[:60]}...")
    
    print(f"\nLinkage judgments:")
    for j in judgments:
        print(f"  [{j.verdict}] (conf={j.confidence:.2f}) {j.claim_text[:50]}...")
        if j.dialogue_evidence:
            print(f"    -> Evidence: {j.dialogue_evidence[0].snippet[:40]}...")
