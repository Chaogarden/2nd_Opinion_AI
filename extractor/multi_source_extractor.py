# extractor/multi_source_extractor.py
# ==============================
# Unified Multi-Source Clinical Information Extractor
# Handles dialogue, prose notes, and SOAP notes
# ==============================

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.schema import ExtractorJSON, Evidence
from extractor.hybrid_extractor import HybridExtractor
from extractor.note_extractor import NoteExtractor

SourceType = Literal["dialogue", "note", "augmented_note"]


# ============================================================
# Cached Extractors
# ============================================================

_cached_hybrid_extractor: Optional[HybridExtractor] = None
_cached_note_extractor: Optional[NoteExtractor] = None


def get_hybrid_extractor(
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    force_new: bool = False
) -> HybridExtractor:
    """
    Get or create a HybridExtractor instance.
    
    Args:
        llm_config: LLM configuration
        use_llm: Whether to use LLM extraction
        force_new: Force creation of new instance
    
    Returns:
        HybridExtractor instance
    """
    global _cached_hybrid_extractor
    
    # When using LLM, don't cache since config may change
    if not force_new and not use_llm and _cached_hybrid_extractor is not None:
        return _cached_hybrid_extractor
    
    # Get LLM client if available
    llm_client = None
    qa_model = None
    if llm_config is not None and use_llm:
        llm_client = llm_config.client
        qa_model = llm_config.diagnoser_model
    
    # RxNorm path
    rxnorm_path = ROOT / "models" / "rxnorm_names.tsv"
    
    extractor = HybridExtractor(
        symptom_backend="scispacy",
        med_backend="scispacy",
        rxnorm_tsv_path=str(rxnorm_path) if rxnorm_path.exists() else None,
        llm_client=llm_client,
        qa_model_name=qa_model,
        use_llm_extraction=use_llm and llm_client is not None,
        use_ner_enrichment=True
    )
    
    if not use_llm:
        _cached_hybrid_extractor = extractor
    
    return extractor


def get_note_extractor(llm_config: Any) -> NoteExtractor:
    """
    Get or create a NoteExtractor instance.
    
    Args:
        llm_config: LLM configuration (required)
    
    Returns:
        NoteExtractor instance
    """
    global _cached_note_extractor
    
    if _cached_note_extractor is not None:
        # Check if same model
        if _cached_note_extractor.model == llm_config.diagnoser_model:
            return _cached_note_extractor
    
    extractor = NoteExtractor(
        client=llm_config.client,
        model=llm_config.diagnoser_model
    )
    _cached_note_extractor = extractor
    
    return extractor


def clear_extractor_caches():
    """Clear cached extractors."""
    global _cached_hybrid_extractor, _cached_note_extractor
    _cached_hybrid_extractor = None
    _cached_note_extractor = None


# ============================================================
# ACI Dataset Helpers
# ============================================================

def load_aci_encounter(encounter_id: str, include_notes: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load a specific encounter from the ACI dataset.
    
    Args:
        encounter_id: The encounter ID to find
        include_notes: Whether to include note and augmented_note fields
    
    Returns:
        Dict with dialogue, note, augmented_note fields, or None if not found
    """
    from data.load_ACI_dataset import extract_aci_dialogues
    
    for split in ["test", "validation", "train"]:
        try:
            dialogues = extract_aci_dialogues(
                split=split,
                max_examples=None,
                include_notes=include_notes
            )
            for d in dialogues:
                if d["encounter_id"] == encounter_id:
                    return d
        except Exception:
            continue
    
    return None


def parse_dialogue_to_turns(dialogue_text: str) -> List[Dict[str, Any]]:
    """
    Parse ACI dialogue text to turn format.
    
    Args:
        dialogue_text: Raw dialogue with [doctor]/[patient] tags
    
    Returns:
        List of turn dicts
    """
    from data.load_ACI_dataset import aci_dialogue_to_turns
    return aci_dialogue_to_turns(dialogue_text)


def segment_note_to_sentences(note_text: str) -> List[Dict[str, Any]]:
    """
    Segment a note into sentences with indices.
    
    Args:
        note_text: Raw note text
    
    Returns:
        List of dicts with 'idx', 'text', 'start', 'end'
    """
    import re
    
    # Simple sentence splitting (can be improved with spaCy)
    # Split on period, exclamation, question mark followed by space and capital or end
    pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
    
    sentences = []
    current_pos = 0
    
    # Split by newlines first (for SOAP sections)
    lines = note_text.split('\n')
    
    idx = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split line into sentences
        parts = re.split(pattern, line)
        for part in parts:
            part = part.strip()
            if part:
                start = note_text.find(part, current_pos)
                if start == -1:
                    start = current_pos
                end = start + len(part)
                
                sentences.append({
                    'idx': idx,
                    'text': part,
                    'start': start,
                    'end': end
                })
                idx += 1
                current_pos = end
    
    return sentences


def detect_soap_sections(note_text: str) -> Dict[str, Tuple[int, int]]:
    """
    Detect SOAP section boundaries in a note.
    
    Args:
        note_text: SOAP note text
    
    Returns:
        Dict mapping section name to (start, end) character positions
    """
    import re
    
    sections = {}
    
    # Common SOAP section headers
    patterns = [
        (r'(?i)^\s*(subjective|s)[:\s]', 'S'),
        (r'(?i)^\s*(objective|o)[:\s]', 'O'),
        (r'(?i)^\s*(assessment|a)[:\s]', 'A'),
        (r'(?i)^\s*(plan|p)[:\s]', 'P'),
        (r'(?i)^\s*(history\s+of\s+present\s+illness|hpi)[:\s]', 'S'),
        (r'(?i)^\s*(physical\s+exam|pe|exam)[:\s]', 'O'),
        (r'(?i)^\s*(impression|dx|diagnosis)[:\s]', 'A'),
        (r'(?i)^\s*(recommendations?|rx|treatment)[:\s]', 'P'),
    ]
    
    # Find all section starts
    section_starts = []
    for pattern, section_name in patterns:
        for match in re.finditer(pattern, note_text, re.MULTILINE):
            section_starts.append((match.start(), match.end(), section_name))
    
    # Sort by position
    section_starts.sort(key=lambda x: x[0])
    
    # Assign boundaries
    for i, (start, header_end, name) in enumerate(section_starts):
        # End is start of next section or end of text
        if i + 1 < len(section_starts):
            end = section_starts[i + 1][0]
        else:
            end = len(note_text)
        
        sections[name] = (header_end, end)
    
    return sections


# ============================================================
# Main Extraction Interface
# ============================================================

def run_extraction(
    encounter_id: Optional[str] = None,
    source_type: SourceType = "dialogue",
    text: Optional[str] = None,
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    verbose: bool = False
) -> ExtractorJSON:
    """
    Unified extraction interface for all source types.
    
    Args:
        encounter_id: ACI encounter ID (optional if text provided)
        source_type: "dialogue", "note", or "augmented_note"
        text: Pre-loaded text (optional, loads from ACI if not provided)
        llm_config: LLM configuration for extraction
        use_llm: Whether to use LLM-based extraction
        verbose: Print progress information
    
    Returns:
        ExtractorJSON with extracted clinical information
    """
    # Load text from ACI if not provided
    if text is None:
        if encounter_id is None:
            raise ValueError("Either encounter_id or text must be provided")
        
        encounter = load_aci_encounter(encounter_id, include_notes=True)
        if encounter is None:
            raise ValueError(f"Could not find encounter {encounter_id} in ACI dataset")
        
        if source_type == "dialogue":
            text = encounter.get("dialogue", "")
        elif source_type == "note":
            text = encounter.get("note", "")
        elif source_type == "augmented_note":
            text = encounter.get("augmented_note", "")
        else:
            raise ValueError(f"Unknown source_type: {source_type}")
        
        if not text:
            raise ValueError(f"No {source_type} text found for encounter {encounter_id}")
    
    if verbose:
        print(f"Extracting from {source_type}: {len(text)} chars")
    
    # Route to appropriate extractor
    if source_type == "dialogue":
        return _extract_dialogue(text, llm_config, use_llm, verbose)
    else:
        return _extract_note(text, source_type, llm_config, use_llm, verbose)


def _extract_dialogue(
    dialogue_text: str,
    llm_config: Optional[Any],
    use_llm: bool,
    verbose: bool
) -> ExtractorJSON:
    """Extract from dialogue text."""
    # Parse to turns
    turns = parse_dialogue_to_turns(dialogue_text)
    
    if verbose:
        print(f"  Parsed {len(turns)} dialogue turns")
    
    # Get extractor
    extractor = get_hybrid_extractor(llm_config, use_llm)
    
    # Run extraction
    result = extractor.extract(turns)
    
    if verbose:
        print(f"  Extracted: {len(result.symptoms)} symptoms, {len(result.meds)} meds")
    
    return result


def _extract_note(
    note_text: str,
    source_type: SourceType,
    llm_config: Optional[Any],
    use_llm: bool,
    verbose: bool
) -> ExtractorJSON:
    """Extract from note text (prose or SOAP)."""
    if not use_llm or llm_config is None:
        if verbose:
            print("  Warning: Note extraction requires LLM, returning empty result")
        return ExtractorJSON()
    
    # Determine note type
    note_type = "soap" if source_type == "augmented_note" else "prose"
    
    if verbose:
        print(f"  Note type: {note_type}")
    
    # Get extractor
    extractor = get_note_extractor(llm_config)
    
    # Run extraction
    result = extractor.extract(note_text, note_type=note_type)
    
    if verbose:
        print(f"  Extracted: {len(result.symptoms)} symptoms, {len(result.meds)} meds")
    
    return result


# ============================================================
# Batch Extraction with Evidence Mapping
# ============================================================

def extract_with_evidence_mapping(
    encounter_id: Optional[str] = None,
    source_type: SourceType = "dialogue",
    text: Optional[str] = None,
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    verbose: bool = False
) -> Tuple[ExtractorJSON, Dict[str, List[int]]]:
    """
    Extract with mapping from claims to source sentences.
    
    Args:
        encounter_id: ACI encounter ID
        source_type: "dialogue", "note", or "augmented_note"
        text: Pre-loaded text
        llm_config: LLM configuration
        use_llm: Whether to use LLM
        verbose: Print progress
    
    Returns:
        Tuple of:
          - ExtractorJSON with extractions
          - Dict mapping entity keys to sentence/turn indices
    """
    # Run standard extraction
    ej = run_extraction(
        encounter_id=encounter_id,
        source_type=source_type,
        text=text,
        llm_config=llm_config,
        use_llm=use_llm,
        verbose=verbose
    )
    
    # Build evidence mapping from the Evidence fields
    evidence_map = {}
    
    # Symptoms
    for s in ej.symptoms:
        key = f"symptom:{s.name_norm or s.name_surface}"
        if s.evidence and s.evidence.utt_ids:
            evidence_map[key] = list(s.evidence.utt_ids)
    
    # Medications
    for m in ej.meds:
        key = f"med:{m.name_norm or m.name_surface}"
        if m.evidence and m.evidence.utt_ids:
            evidence_map[key] = list(m.evidence.utt_ids)
    
    # Allergies
    for a in ej.allergies:
        key = f"allergy:{a.substance_norm or a.substance_surface}"
        if a.evidence and a.evidence.utt_ids:
            evidence_map[key] = list(a.evidence.utt_ids)
    
    # Vitals
    for v in ej.vitals:
        key = f"vital:{v.kind}:{v.value}"
        if v.evidence and v.evidence.utt_ids:
            evidence_map[key] = list(v.evidence.utt_ids)
    
    # QA extractions
    for qa in ej.qa_extractions:
        key = f"qa:{qa.concept}"
        if qa.evidence and qa.evidence.utt_ids:
            evidence_map[key] = list(qa.evidence.utt_ids)
    
    return ej, evidence_map


def extract_all_sources(
    encounter_id: str,
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    verbose: bool = False
) -> Dict[str, ExtractorJSON]:
    """
    Extract from all available sources for an encounter.
    
    Args:
        encounter_id: ACI encounter ID
        llm_config: LLM configuration
        use_llm: Whether to use LLM
        verbose: Print progress
    
    Returns:
        Dict mapping source_type to ExtractorJSON
    """
    results = {}
    
    # Load encounter data
    encounter = load_aci_encounter(encounter_id, include_notes=True)
    if encounter is None:
        raise ValueError(f"Encounter {encounter_id} not found")
    
    # Extract from each available source
    for source_type in ["dialogue", "note", "augmented_note"]:
        text = encounter.get(source_type if source_type != "augmented_note" else "augmented_note", "")
        
        if not text:
            if verbose:
                print(f"  No {source_type} available for {encounter_id}")
            continue
        
        try:
            if verbose:
                print(f"\nExtracting {source_type} for {encounter_id}...")
            
            ej = run_extraction(
                encounter_id=None,  # Already have text
                source_type=source_type,
                text=text,
                llm_config=llm_config,
                use_llm=use_llm,
                verbose=verbose
            )
            results[source_type] = ej
            
        except Exception as e:
            if verbose:
                print(f"  Failed to extract {source_type}: {e}")
            results[source_type] = ExtractorJSON()
    
    return results


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test multi-source extraction")
    parser.add_argument("--encounter", "-e", type=str, help="ACI encounter ID")
    parser.add_argument("--source", "-s", type=str, default="dialogue",
                        choices=["dialogue", "note", "augmented_note"],
                        help="Source type to extract from")
    parser.add_argument("--all-sources", action="store_true",
                        help="Extract from all available sources")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM extraction")
    parser.add_argument("--mode", "-m", type=str, default="test",
                        choices=["test", "prod"],
                        help="LLM mode")
    
    args = parser.parse_args()
    
    # Set up LLM config
    llm_config = None
    use_llm = not args.no_llm
    
    if use_llm:
        try:
            from core.llm_config import LLMConfig
            llm_config = LLMConfig(mode=args.mode)
            print(f"Using LLM mode: {args.mode}")
        except Exception as e:
            print(f"Could not initialize LLM: {e}")
            use_llm = False
    
    # Get a sample encounter if none specified
    if not args.encounter:
        from data.load_ACI_dataset import extract_aci_dialogues
        dialogues = extract_aci_dialogues("test", max_examples=1)
        if dialogues:
            args.encounter = dialogues[0]["encounter_id"]
            print(f"Using sample encounter: {args.encounter}")
        else:
            print("No encounters found")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Multi-Source Extraction Test")
    print(f"{'='*60}")
    print(f"Encounter: {args.encounter}")
    print(f"LLM: {use_llm}")
    
    if args.all_sources:
        results = extract_all_sources(
            encounter_id=args.encounter,
            llm_config=llm_config,
            use_llm=use_llm,
            verbose=True
        )
        
        for source_type, ej in results.items():
            print(f"\n--- {source_type.upper()} ---")
            print(f"Chief Complaint: {ej.chief_complaint}")
            print(f"Symptoms: {len(ej.symptoms)}")
            print(f"Medications: {len(ej.meds)}")
            print(f"Allergies: {len(ej.allergies)}")
            print(f"Risk Factors: {len(ej.risk_factors)}")
    else:
        ej = run_extraction(
            encounter_id=args.encounter,
            source_type=args.source,
            llm_config=llm_config,
            use_llm=use_llm,
            verbose=True
        )
        
        print(f"\n--- RESULTS ({args.source}) ---")
        print(f"Chief Complaint: {ej.chief_complaint}")
        print(f"\nSymptoms ({len(ej.symptoms)}):")
        for s in ej.symptoms[:5]:
            print(f"  - {s.name_surface} ({s.assertion})")
        
        print(f"\nMedications ({len(ej.meds)}):")
        for m in ej.meds[:5]:
            print(f"  - {m.name_surface} {m.dose or ''} {m.freq or ''}")
        
        print(f"\nAllergies ({len(ej.allergies)}):")
        for a in ej.allergies:
            print(f"  - {a.substance_surface}")
        
        print(f"\nRisk Factors ({len(ej.risk_factors)}):")
        for rf in ej.risk_factors[:5]:
            print(f"  - {rf}")
