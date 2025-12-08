# data/load_ACI_dataset.py
# ==============================
# ACI-Bench-Refined Dataset Loader
# https://huggingface.co/datasets/ClinicianFOCUS/ACI-Bench-Refined
#
# This dataset contains doctor-patient dialogues with clinical notes.
# Columns:
#   - dataset: source dataset identifier
#   - encounter_id: unique ID for each encounter
#   - dialogue: raw dialogue text with [doctor]/[patient] tags
#   - note: clinical note
#   - augmented_note: enhanced clinical note (SOAP format)
#
# Splits: train (177 rows), validation (10 rows), test (20 rows)
#
# Usage:
#   pip install datasets
#   python data/load_ACI_dataset.py --split test --max 5
# ==============================

import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator

from datasets import load_dataset, Dataset


# ============================================================
# Dataset Loading
# ============================================================

def load_aci_dataset(split: str = "train") -> Dataset:
    """
    Load the ACI-Bench-Refined dataset from Hugging Face.
    
    Args:
        split: Dataset split to load ("train", "validation", or "test").
    
    Returns:
        A HuggingFace Dataset object for the specified split.
    
    Example:
        >>> ds = load_aci_dataset("test")
        >>> print(len(ds))
        20
    """
    ds = load_dataset("ClinicianFOCUS/ACI-Bench-Refined", split=split)
    return ds


def load_all_splits() -> Dict[str, Dataset]:
    """
    Load all splits of the ACI-Bench-Refined dataset.
    
    Returns:
        Dict mapping split names to Dataset objects.
    """
    return {
        "train": load_aci_dataset("train"),
        "validation": load_aci_dataset("validation"),
        "test": load_aci_dataset("test"),
    }


# ============================================================
# Dialogue Extraction
# ============================================================

def extract_aci_dialogues(
    split: str = "test",
    max_examples: Optional[int] = None,
    include_notes: bool = False
) -> List[Dict[str, Any]]:
    """
    Extract dialogues from the ACI-Bench-Refined dataset.
    
    Args:
        split: Dataset split to extract from.
        max_examples: Maximum number of examples to extract (None = all).
        include_notes: Whether to include clinical notes in output.
    
    Returns:
        List of dicts with keys:
          - encounter_id: str
          - dialogue: str (raw dialogue with [doctor]/[patient] tags)
          - dataset_source: str (original dataset source)
          - note: str (only if include_notes=True)
          - augmented_note: str (only if include_notes=True)
    
    Example:
        >>> dialogues = extract_aci_dialogues("test", max_examples=5)
        >>> print(dialogues[0]["encounter_id"])
    """
    ds = load_aci_dataset(split)
    
    results = []
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        
        item = {
            "encounter_id": row["encounter_id"],
            "dialogue": row["dialogue"],
            "dataset_source": row.get("dataset", "unknown"),
        }
        
        if include_notes:
            item["note"] = row.get("note", "")
            item["augmented_note"] = row.get("augmented note", "")
        
        results.append(item)
    
    return results


def save_aci_dialogues_jsonl(
    output_path: str,
    split: str = "test",
    max_examples: Optional[int] = 20,
    include_notes: bool = False
) -> int:
    """
    Save extracted dialogues to a JSONL file.
    
    Args:
        output_path: Path to output JSONL file.
        split: Dataset split to extract from.
        max_examples: Maximum number of examples to save.
        include_notes: Whether to include clinical notes.
    
    Returns:
        Number of dialogues saved.
    
    Example:
        >>> n = save_aci_dialogues_jsonl("data/aci_test.jsonl", "test", 10)
        >>> print(f"Saved {n} dialogues")
    """
    dialogues = extract_aci_dialogues(split, max_examples, include_notes)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for d in dialogues:
            d["dataset_split"] = split
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    return len(dialogues)


# ============================================================
# Dialogue Parsing (ACI format -> Internal Turn Format)
# ============================================================

# Regex to match [doctor] or [patient] tags (case-insensitive)
SPEAKER_TAG_RE = re.compile(r'\[(?P<role>doctor|patient)\]', re.IGNORECASE)


def aci_dialogue_to_turns(dialogue_text: str) -> List[Dict[str, Any]]:
    """
    Parse ACI-format dialogue text into internal turn format.
    
    The ACI dataset uses [doctor] and [patient] tags to mark speaker turns.
    This function parses those tags into structured turn dictionaries
    compatible with the clinical pipeline.
    
    Args:
        dialogue_text: Raw dialogue string with [doctor]/[patient] tags.
    
    Returns:
        List of turn dicts, each with:
          - utt_id: int (sequential ID starting at 0)
          - role: str ("DOCTOR" or "PATIENT")
          - speaker: str (same as role)
          - start: float (0.0, no timestamps in text data)
          - end: float (0.0)
          - text: str (the utterance text)
    
    Example:
        >>> text = "[doctor] How are you? [patient] I have a headache."
        >>> turns = aci_dialogue_to_turns(text)
        >>> print(turns[0])
        {'utt_id': 0, 'role': 'DOCTOR', 'speaker': 'DOCTOR', 'start': 0.0, 'end': 0.0, 'text': 'How are you?'}
    """
    turns = []
    
    # Find all speaker tag positions
    matches = list(SPEAKER_TAG_RE.finditer(dialogue_text))
    
    if not matches:
        # No tags found - treat entire text as patient utterance
        if dialogue_text.strip():
            turns.append({
                "utt_id": 0,
                "role": "PATIENT",
                "speaker": "PATIENT",
                "start": 0.0,
                "end": 0.0,
                "text": dialogue_text.strip(),
            })
        return turns
    
    for i, match in enumerate(matches):
        role = match.group("role").upper()
        
        # Get text between this tag and the next (or end of string)
        start_pos = match.end()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(dialogue_text)
        
        text = dialogue_text[start_pos:end_pos].strip()
        
        if text:  # Only add non-empty turns
            turns.append({
                "utt_id": len(turns),
                "role": role,
                "speaker": role,
                "start": 0.0,
                "end": 0.0,
                "text": text,
            })
    
    return turns


def make_transcript_result_from_aci(dialogue_text: str) -> Dict[str, Any]:
    """
    Create a transcript_result dict from ACI dialogue text.
    
    This creates a structure compatible with the Streamlit app and
    the run_clinical_pipeline function, mimicking what would come
    from WhisperX ASR output.
    
    Args:
        dialogue_text: Raw dialogue string with [doctor]/[patient] tags.
    
    Returns:
        Dict with keys:
          - text: str (full dialogue text, tags stripped for readability)
          - segments: List[Dict] (parsed turns with role info)
          - info: Dict with language and duration metadata
    
    Example:
        >>> result = make_transcript_result_from_aci("[doctor] Hi [patient] Hello")
        >>> print(result["info"]["language"])
        'en'
    """
    turns = aci_dialogue_to_turns(dialogue_text)
    
    # Build clean text (without tags) for the transcript view
    clean_text = " ".join(t["text"] for t in turns)
    
    # Convert turns to segment format (similar to WhisperX output)
    segments = []
    for turn in turns:
        segments.append({
            "start": turn["start"],
            "end": turn["end"],
            "text": turn["text"],
            "speaker": turn["speaker"],  # For diarization compatibility
        })
    
    return {
        "text": clean_text,
        "segments": segments,
        "info": {
            "language": "en",
            "duration": None,  # No audio duration for text data
        },
        "_aci_turns": turns,  # Store parsed turns for direct extractor use
    }


def get_prebuilt_role_mapping_from_aci() -> Dict[str, str]:
    """
    Get the role mapping for ACI dialogues.
    
    Since ACI dialogues already have explicit [doctor]/[patient] tags,
    the mapping is trivial: DOCTOR -> DOCTOR, PATIENT -> PATIENT.
    
    Returns:
        Dict mapping speaker IDs to roles.
    """
    return {
        "DOCTOR": "DOCTOR",
        "PATIENT": "PATIENT",
    }


# ============================================================
# Simple Pipeline Test (No Pipeline Changes Required)
# ============================================================

def get_aci_test_inputs(
    split: str = "test",
    max_examples: int = 5
) -> List[Dict[str, Any]]:
    """
    Get ACI dialogues ready to plug directly into run_clinical_pipeline().
    
    This function returns a list of inputs that can be passed directly
    to the existing pipeline WITHOUT ANY MODIFICATIONS to the pipeline code.
    
    Args:
        split: Dataset split to use.
        max_examples: Number of examples to return.
    
    Returns:
        List of dicts, each with:
          - encounter_id: str (for tracking)
          - transcript_result: dict (plug directly into run_clinical_pipeline)
          - role_mapping: dict (plug directly into run_clinical_pipeline)
    
    Example:
        >>> from data.load_ACI_dataset import get_aci_test_inputs
        >>> from core.pipeline import run_clinical_pipeline
        >>> 
        >>> # Get test inputs
        >>> test_inputs = get_aci_test_inputs("test", max_examples=3)
        >>> 
        >>> # Run through pipeline (no changes needed!)
        >>> for item in test_inputs:
        >>>     result = run_clinical_pipeline(
        >>>         transcript_result=item["transcript_result"],
        >>>         role_mapping=item["role_mapping"]
        >>>     )
        >>>     print(f"{item['encounter_id']}: {result.extracted_facts.chief_complaint}")
    """
    dialogues = extract_aci_dialogues(split=split, max_examples=max_examples)
    
    test_inputs = []
    for d in dialogues:
        transcript_result = make_transcript_result_from_aci(d["dialogue"])
        role_mapping = get_prebuilt_role_mapping_from_aci()
        
        test_inputs.append({
            "encounter_id": d["encounter_id"],
            "transcript_result": transcript_result,
            "role_mapping": role_mapping,
        })
    
    return test_inputs


# ============================================================
# Test Harness for E2E Pipeline
# ============================================================

def run_aci_pipeline_test(
    max_examples: int = 3,
    split: str = "test",
    skip_consultant: bool = False,
    skip_revision: bool = True,
    k_guidelines: int = 3,
    k_merck: int = 3,
    verbose: bool = True,
    save_detailed_output: bool = True,
    output_file: str = "data/pipeline_test_output.txt"
) -> List[Dict[str, Any]]:
    """
    Run a batch of ACI dialogues through the clinical reasoning pipeline.
    
    This is a test harness for sanity-checking the end-to-end pipeline
    using pre-diarized dialogues from the ACI-Bench-Refined dataset.
    
    Args:
        max_examples: Number of dialogues to process.
        split: Dataset split to use.
        skip_consultant: Skip the Consultant/Arbiter stage for speed.
        skip_revision: Skip the Diagnoser revision pass (default: True for speed).
        k_guidelines: Number of guideline chunks to retrieve.
        k_merck: Number of Merck chunks to retrieve.
        verbose: Print progress and results.
        save_detailed_output: Save detailed output including RAG evidence to file.
        output_file: Path for detailed output file.
    
    Returns:
        List of result dicts, each containing:
          - encounter_id: str
          - extracted_facts: dict summary
          - differential: list of conditions
          - error: str or None
    
    Example:
        >>> results = run_aci_pipeline_test(max_examples=2, verbose=True)
    """
    import sys
    from pathlib import Path
    
    # Add project root to path
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    
    from core.llm_config import LLMConfig
    from core.pipeline import run_clinical_pipeline
    from datetime import datetime
    
    # Load dialogues
    dialogues = extract_aci_dialogues(split=split, max_examples=max_examples)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ACI Pipeline Test: {len(dialogues)} dialogues from '{split}' split")
        print(f"{'='*60}\n")
    
    # Build LLM config (test mode by default)
    llm_config = LLMConfig(mode="test")
    
    # Prepare detailed output file
    detailed_output_lines = []
    if save_detailed_output:
        detailed_output_lines.append(f"=" * 80)
        detailed_output_lines.append(f"ACI PIPELINE TEST - DETAILED OUTPUT")
        detailed_output_lines.append(f"Generated: {datetime.now().isoformat()}")
        detailed_output_lines.append(f"Split: {split}, Max Examples: {max_examples}")
        detailed_output_lines.append(f"k_guidelines: {k_guidelines}, k_merck: {k_merck}")
        detailed_output_lines.append(f"=" * 80)
        detailed_output_lines.append("")
    
    results = []
    
    for i, dialogue_data in enumerate(dialogues):
        encounter_id = dialogue_data["encounter_id"]
        dialogue_text = dialogue_data["dialogue"]
        
        if verbose:
            print(f"\n[{i+1}/{len(dialogues)}] Processing encounter: {encounter_id}")
            print("-" * 40)
        
        try:
            # Convert ACI dialogue to transcript_result format
            transcript_result = make_transcript_result_from_aci(dialogue_text)
            role_mapping = get_prebuilt_role_mapping_from_aci()
            
            # Show dialogue preview
            if verbose:
                turns = transcript_result.get("_aci_turns", [])
                print(f"\n  === DIALOGUE PREVIEW ({len(turns)} turns) ===")
                for t in turns[:6]:  # Show first 6 turns
                    role_tag = "DOC" if t["role"] == "DOCTOR" else "PAT"
                    text_preview = t["text"][:80] + "..." if len(t["text"]) > 80 else t["text"]
                    print(f"  [{role_tag}] {text_preview}")
                if len(turns) > 6:
                    print(f"  ... ({len(turns) - 6} more turns)")
            
            # Run the pipeline
            pipeline_result = run_clinical_pipeline(
                transcript_result=transcript_result,
                role_mapping=role_mapping,
                llm_config=llm_config,
                k_guidelines=k_guidelines,
                k_merck=k_merck,
                skip_consultant=skip_consultant,
                skip_revision=skip_revision,
            )
            
            # Extract summary info
            result = {
                "encounter_id": encounter_id,
                "error": pipeline_result.error,
            }
            
            if pipeline_result.extracted_facts:
                facts = pipeline_result.extracted_facts
                result["extracted_facts"] = {
                    "chief_complaint": facts.chief_complaint,
                    "num_symptoms": len(facts.symptoms),
                    "num_meds": len(facts.meds),
                    "num_allergies": len(facts.allergies),
                    "risk_factors": facts.risk_factors,
                }
            
            if pipeline_result.diagnoser_output:
                diag = pipeline_result.diagnoser_output
                result["differential"] = [
                    {"condition": dx.condition, "likelihood": dx.likelihood}
                    for dx in diag.differential
                ]
                result["red_flags"] = [rf.description for rf in diag.red_flags]
                result["uncertainty"] = diag.overall_uncertainty
            
            results.append(result)
            
            if verbose:
                if result.get("error"):
                    print(f"  ERROR: {result['error']}")
                else:
                    # Show summary in console
                    ef = result.get("extracted_facts", {})
                    print(f"  Chief Complaint: {ef.get('chief_complaint', 'N/A')[:60]}...")
                    print(f"  Symptoms: {ef.get('num_symptoms', 0)}, Meds: {ef.get('num_meds', 0)}")
                    if result.get("differential"):
                        print(f"  Top Dx: {result['differential'][0]['condition'] if result['differential'] else 'N/A'}")
                    if result.get("red_flags"):
                        print(f"  Red Flags: {len(result['red_flags'])}")
                    if pipeline_result.revised_diagnoser_output:
                        print(f"  ✓ Revision: Plan was revised based on Consultant feedback")
                    if pipeline_result.evidence_chunks:
                        print(f"  RAG Evidence: {len(pipeline_result.evidence_chunks)} chunks (see output file for details)")
            
            # Save detailed output to file
            if save_detailed_output:
                import json
                detailed_output_lines.append(f"\n{'#' * 80}")
                detailed_output_lines.append(f"# ENCOUNTER: {encounter_id}")
                detailed_output_lines.append(f"{'#' * 80}\n")
                
                # Dialogue
                detailed_output_lines.append("=" * 60)
                detailed_output_lines.append("DIALOGUE")
                detailed_output_lines.append("=" * 60)
                turns = transcript_result.get("_aci_turns", [])
                for t in turns:
                    role_tag = "DOCTOR" if t["role"] == "DOCTOR" else "PATIENT"
                    detailed_output_lines.append(f"[{role_tag}]: {t['text']}")
                detailed_output_lines.append("")
                
                # Extracted Facts
                detailed_output_lines.append("=" * 60)
                detailed_output_lines.append("EXTRACTED FACTS")
                detailed_output_lines.append("=" * 60)
                if pipeline_result.extracted_facts:
                    detailed_output_lines.append(json.dumps(
                        pipeline_result.extracted_facts.model_dump(), 
                        indent=2, 
                        default=str
                    ))
                detailed_output_lines.append("")
                
                # RAG Evidence
                detailed_output_lines.append("=" * 60)
                detailed_output_lines.append(f"RAG EVIDENCE ({len(pipeline_result.evidence_chunks)} chunks)")
                detailed_output_lines.append("=" * 60)
                
                if pipeline_result.evidence_chunks:
                    # Guidelines chunks
                    guidelines_chunks = [c for c in pipeline_result.evidence_chunks if c.source == "guidelines"]
                    if guidelines_chunks:
                        detailed_output_lines.append(f"\n--- GUIDELINES ({len(guidelines_chunks)} chunks) ---\n")
                        for chunk in guidelines_chunks:
                            detailed_output_lines.append(f"[{chunk.evidence_id}] Score: {chunk.score:.4f}")
                            detailed_output_lines.append(f"Title: {chunk.title}")
                            detailed_output_lines.append(f"Section: {chunk.heading_path}")
                            detailed_output_lines.append(f"Text:\n{chunk.text[:2000]}{'...' if len(chunk.text) > 2000 else ''}")
                            detailed_output_lines.append("-" * 40)
                    
                    # Merck chunks
                    merck_chunks = [c for c in pipeline_result.evidence_chunks if c.source == "merck"]
                    if merck_chunks:
                        detailed_output_lines.append(f"\n--- MERCK MANUAL ({len(merck_chunks)} chunks) ---\n")
                        for chunk in merck_chunks:
                            detailed_output_lines.append(f"[{chunk.evidence_id}] Score: {chunk.score:.4f}")
                            detailed_output_lines.append(f"Heading: {chunk.heading_path}")
                            detailed_output_lines.append(f"Text:\n{chunk.text[:2000]}{'...' if len(chunk.text) > 2000 else ''}")
                            detailed_output_lines.append("-" * 40)
                else:
                    detailed_output_lines.append("No evidence chunks retrieved.")
                detailed_output_lines.append("")
                
                # Diagnoser Output
                if pipeline_result.diagnoser_output:
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append("DIAGNOSER OUTPUT")
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append(json.dumps(
                        pipeline_result.diagnoser_output.model_dump(),
                        indent=2,
                        default=str
                    ))
                    detailed_output_lines.append("")
                
                # Consultant Critique
                if pipeline_result.consultant_critique:
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append("CONSULTANT CRITIQUE")
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append(json.dumps(
                        pipeline_result.consultant_critique.model_dump(),
                        indent=2,
                        default=str
                    ))
                    detailed_output_lines.append("")
                
                # Revised Diagnoser Output (if revision occurred)
                if pipeline_result.revised_diagnoser_output:
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append("REVISED DIAGNOSER OUTPUT (after Consultant feedback)")
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append(json.dumps(
                        pipeline_result.revised_diagnoser_output.model_dump(),
                        indent=2,
                        default=str
                    ))
                    detailed_output_lines.append("")
                
                # Arbiter Result
                if pipeline_result.arbiter_result:
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append("ARBITER RESULT")
                    detailed_output_lines.append("=" * 60)
                    detailed_output_lines.append(json.dumps(
                        pipeline_result.arbiter_result.model_dump(),
                        indent=2,
                        default=str
                    ))
                    detailed_output_lines.append("")
                        
        except Exception as e:
            results.append({
                "encounter_id": encounter_id,
                "error": str(e),
            })
            if verbose:
                print(f"  EXCEPTION: {e}")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Completed: {len([r for r in results if not r.get('error')])} success, "
              f"{len([r for r in results if r.get('error')])} errors")
        print(f"{'='*60}\n")
    
    # Write detailed output file
    if save_detailed_output and detailed_output_lines:
        output_path = Path(ROOT) / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add summary at end
        detailed_output_lines.append(f"\n{'=' * 80}")
        detailed_output_lines.append("SUMMARY")
        detailed_output_lines.append(f"{'=' * 80}")
        detailed_output_lines.append(f"Total encounters processed: {len(results)}")
        detailed_output_lines.append(f"Successful: {len([r for r in results if not r.get('error')])}")
        detailed_output_lines.append(f"Errors: {len([r for r in results if r.get('error')])}")
        
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(detailed_output_lines))
        
        if verbose:
            print(f"Detailed output saved to: {output_path}")
    
    return results


# ============================================================
# Utility: Iterate dialogues as generator
# ============================================================

def iter_aci_dialogues(
    split: str = "test",
    max_examples: Optional[int] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Iterate over ACI dialogues as a generator.
    
    Memory-efficient alternative to extract_aci_dialogues for large batches.
    
    Args:
        split: Dataset split to iterate.
        max_examples: Maximum number of examples (None = all).
    
    Yields:
        Dict with encounter_id, dialogue, and dataset_source.
    """
    ds = load_aci_dataset(split)
    
    for i, row in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            break
        
        yield {
            "encounter_id": row["encounter_id"],
            "dialogue": row["dialogue"],
            "dataset_source": row.get("dataset", "unknown"),
        }


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    """CLI entry point for dataset inspection and extraction."""
    parser = argparse.ArgumentParser(
        description="ACI-Bench-Refined Dataset Loader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset stats
  python data/load_ACI_dataset.py --split test
  
  # Extract dialogues to JSONL
  python data/load_ACI_dataset.py --split test --max 10 --output data/aci_test.jsonl
  
  # Run pipeline test harness
  python data/load_ACI_dataset.py --test --max 3
  
  # Parse a single dialogue
  python data/load_ACI_dataset.py --parse "[doctor] Hi [patient] Hello"
        """
    )
    
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "validation", "test"],
                        help="Dataset split to use")
    parser.add_argument("--max", type=int, default=None,
                        help="Maximum number of examples")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output JSONL file path")
    parser.add_argument("--include-notes", action="store_true",
                        help="Include clinical notes in output")
    parser.add_argument("--test", action="store_true",
                        help="Run pipeline test harness")
    parser.add_argument("--parse", type=str, default=None,
                        help="Parse a dialogue string and show turns")
    parser.add_argument("--skip-consultant", action="store_true",
                        help="Skip the Consultant/Arbiter stage for speed")
    parser.add_argument("--with-revision", action="store_true",
                        help="Enable Diagnoser revision pass after Consultant critique")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    
    args = parser.parse_args()
    
    # Parse mode
    if args.parse:
        print("\nParsing dialogue:")
        print(f"  Input: {args.parse[:80]}...")
        turns = aci_dialogue_to_turns(args.parse)
        print(f"\nParsed {len(turns)} turns:")
        for turn in turns:
            print(f"  [{turn['utt_id']}] {turn['role']}: {turn['text'][:60]}...")
        return
    
    # Test harness mode
    if args.test:
        run_aci_pipeline_test(
            max_examples=args.max or 3,
            split=args.split,
            skip_consultant=args.skip_consultant,
            skip_revision=not args.with_revision,  # Default: skip revision
            verbose=True
        )
        return
    
    # Load and display stats
    print(f"\nLoading ACI-Bench-Refined dataset (split: {args.split})...")
    ds = load_aci_dataset(args.split)
    
    print(f"\nDataset Stats:")
    print(f"  Split: {args.split}")
    print(f"  Rows: {len(ds)}")
    print(f"  Columns: {list(ds.column_names)}")
    
    # Show example
    if len(ds) > 0:
        example = ds[0]
        print(f"\nExample (encounter_id: {example['encounter_id']}):")
        print(f"  Dataset source: {example.get('dataset', 'N/A')}")
        dialogue_preview = example["dialogue"][:200].replace("\n", " ")
        print(f"  Dialogue preview: {dialogue_preview}...")
    
    # Extract to JSONL if requested
    if args.output:
        n = save_aci_dialogues_jsonl(
            args.output,
            split=args.split,
            max_examples=args.max,
            include_notes=args.include_notes
        )
        print(f"\nSaved {n} dialogues to: {args.output}")
    
    # Extract and show dialogues if verbose
    if args.verbose and not args.output:
        dialogues = extract_aci_dialogues(
            split=args.split,
            max_examples=args.max or 5
        )
        print(f"\nExtracted {len(dialogues)} dialogues:")
        for d in dialogues:
            print(f"  - {d['encounter_id']}: {len(d['dialogue'])} chars")


if __name__ == "__main__":
    main()
