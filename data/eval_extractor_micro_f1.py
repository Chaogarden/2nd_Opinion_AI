# data/eval_extractor_micro_f1.py
# ==============================
# Micro-F1 Evaluation Script for Clinical Information Extractor
# Evaluates extractor against GOLD annotations
# ==============================

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.eval_utils import (
    load_gold_file, evaluate_extraction, aggregate_results,
    format_metrics_table, format_error_analysis, EvalResult, MetricCounts
)
from extractor.schema import ExtractorJSON


# ============================================================
# Default paths
# ============================================================

DEFAULT_GOLD_PATH = ROOT / "data" / "aci_extractor_gold_15.jsonl"


# ============================================================
# Extraction Runner (delegates to multi_source_extractor when available)
# ============================================================

def run_extraction_for_eval(
    encounter_id: str,
    source_type: str,
    text: Optional[str] = None,
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    verbose: bool = False
) -> ExtractorJSON:
    """
    Run extraction for a single GOLD example.
    
    Delegates to extractor/multi_source_extractor.py if available,
    otherwise falls back to direct HybridExtractor usage.
    
    Args:
        encounter_id: ACI encounter ID
        source_type: "dialogue", "note", or "augmented_note"
        text: Optional pre-loaded text (if not provided, loads from ACI)
        llm_config: LLM configuration for extraction
        use_llm: Whether to use LLM-based extraction
        verbose: Print progress
    
    Returns:
        ExtractorJSON with extracted information
    """
    try:
        # Try using multi_source_extractor if available
        from extractor.multi_source_extractor import run_extraction
        return run_extraction(
            encounter_id=encounter_id,
            source_type=source_type,
            text=text,
            llm_config=llm_config,
            use_llm=use_llm,
            verbose=verbose
        )
    except ImportError:
        # Fall back to direct implementation
        pass
    
    # Fallback: direct HybridExtractor for dialogue
    if source_type == "dialogue":
        return _extract_dialogue_fallback(encounter_id, text, llm_config, use_llm, verbose)
    elif source_type in ("note", "augmented_note"):
        return _extract_note_fallback(text, source_type, llm_config, use_llm, verbose)
    else:
        raise ValueError(f"Unknown source_type: {source_type}")


def _extract_dialogue_fallback(
    encounter_id: str,
    text: Optional[str],
    llm_config: Optional[Any],
    use_llm: bool,
    verbose: bool
) -> ExtractorJSON:
    """Fallback dialogue extraction using HybridExtractor directly."""
    from data.load_ACI_dataset import aci_dialogue_to_turns, extract_aci_dialogues
    from extractor.hybrid_extractor import HybridExtractor
    
    # Load text if not provided
    if text is None:
        # Find the encounter in ACI dataset
        for split in ["test", "validation", "train"]:
            try:
                dialogues = extract_aci_dialogues(split=split, max_examples=None)
                for d in dialogues:
                    if d["encounter_id"] == encounter_id:
                        text = d["dialogue"]
                        break
                if text:
                    break
            except Exception:
                continue
        
        if text is None:
            raise ValueError(f"Could not find encounter_id {encounter_id} in ACI dataset")
    
    # Parse dialogue to turns
    turns = aci_dialogue_to_turns(text)
    
    if verbose:
        print(f"  Extracted {len(turns)} turns from dialogue")
    
    # Set up extractor
    llm_client = None
    qa_model = None
    if llm_config is not None and use_llm:
        llm_client = llm_config.client
        qa_model = llm_config.diagnoser_model
    
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
    
    return extractor.extract(turns)


def _extract_note_fallback(
    text: Optional[str],
    source_type: str,
    llm_config: Optional[Any],
    use_llm: bool,
    verbose: bool
) -> ExtractorJSON:
    """Fallback note extraction (requires multi_source_extractor for full support)."""
    if text is None:
        raise ValueError(f"Text must be provided for {source_type} extraction")
    
    try:
        from extractor.note_extractor import NoteExtractor
        note_type = "soap" if source_type == "augmented_note" else "prose"
        
        llm_client = None
        model = None
        if llm_config is not None and use_llm:
            llm_client = llm_config.client
            model = llm_config.diagnoser_model
        
        if llm_client is None:
            raise ImportError("LLM client required for note extraction")
        
        extractor = NoteExtractor(client=llm_client, model=model)
        return extractor.extract(text, note_type=note_type)
    except ImportError:
        # Return empty extraction if note extractor not available
        if verbose:
            print(f"  Warning: NoteExtractor not available, returning empty extraction")
        return ExtractorJSON()


# ============================================================
# Main Evaluation Loop
# ============================================================

def run_evaluation(
    gold_path: Path,
    llm_config: Optional[Any] = None,
    use_llm: bool = True,
    source_type_filter: Optional[str] = None,
    max_examples: Optional[int] = None,
    verbose: bool = True,
    save_predictions: bool = False,
    predictions_path: Optional[Path] = None
) -> EvalResult:
    """
    Run full evaluation on GOLD file.
    
    Args:
        gold_path: Path to GOLD JSONL file
        llm_config: LLM configuration
        use_llm: Whether to use LLM extraction
        source_type_filter: Only evaluate specific source type
        max_examples: Maximum examples to evaluate
        verbose: Print progress
        save_predictions: Save predictions to file
        predictions_path: Where to save predictions
    
    Returns:
        Aggregated EvalResult
    """
    # Load GOLD file
    gold_entries = load_gold_file(gold_path)
    
    if verbose:
        print(f"Loaded {len(gold_entries)} GOLD examples from {gold_path}")
    
    # Filter by source type if requested
    if source_type_filter:
        gold_entries = [e for e in gold_entries if e.get("source_type") == source_type_filter]
        if verbose:
            print(f"Filtered to {len(gold_entries)} examples with source_type={source_type_filter}")
    
    # Limit examples if requested
    if max_examples and len(gold_entries) > max_examples:
        gold_entries = gold_entries[:max_examples]
        if verbose:
            print(f"Limited to {max_examples} examples")
    
    # Skip template entries
    gold_entries = [e for e in gold_entries if not e.get("encounter_id", "").startswith("TEMPLATE")]
    
    if not gold_entries:
        print("No valid GOLD entries to evaluate (all templates or empty)")
        return EvalResult()
    
    results = []
    predictions = []
    
    for i, entry in enumerate(gold_entries):
        encounter_id = entry.get("encounter_id", f"unknown_{i}")
        source_type = entry.get("source_type", "dialogue")
        text = entry.get("text")
        gold = entry.get("gold", {})
        
        if verbose:
            print(f"\n[{i+1}/{len(gold_entries)}] Evaluating {encounter_id} ({source_type})")
        
        try:
            # Run extraction
            pred_ej = run_extraction_for_eval(
                encounter_id=encounter_id,
                source_type=source_type,
                text=text,
                llm_config=llm_config,
                use_llm=use_llm,
                verbose=verbose
            )
            
            # Evaluate against gold
            result = evaluate_extraction(gold, pred_ej)
            results.append(result)
            
            if verbose:
                print(f"  Entities: TP={result.overall.tp}, FP={result.overall.fp}, FN={result.overall.fn}")
                print(f"  Entity F1: {result.overall.f1():.3f}")
            
            if save_predictions:
                predictions.append({
                    "encounter_id": encounter_id,
                    "source_type": source_type,
                    "prediction": pred_ej.model_dump()
                })
                
        except Exception as e:
            print(f"  ERROR: {e}")
            # Add empty result so aggregation still works
            results.append(EvalResult())
    
    # Aggregate results
    agg_result = aggregate_results(results)
    
    # Save predictions if requested
    if save_predictions and predictions:
        pred_path = predictions_path or (gold_path.parent / "eval_predictions.jsonl")
        with pred_path.open("w", encoding="utf-8") as f:
            for p in predictions:
                f.write(json.dumps(p, default=str) + "\n")
        if verbose:
            print(f"\nPredictions saved to {pred_path}")
    
    return agg_result


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate clinical extractor using micro-F1 metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation (NER only, no LLM)
  python data/eval_extractor_micro_f1.py --no-llm
  
  # Evaluation with LLM extraction
  python data/eval_extractor_micro_f1.py --mode test
  
  # Evaluate only dialogue examples
  python data/eval_extractor_micro_f1.py --source-type dialogue
  
  # Limit to first 5 examples
  python data/eval_extractor_micro_f1.py --max 5
  
  # Save predictions for analysis
  python data/eval_extractor_micro_f1.py --save-predictions
        """
    )
    
    parser.add_argument(
        "--gold", "-g",
        type=str,
        default=str(DEFAULT_GOLD_PATH),
        help=f"Path to GOLD JSONL file (default: {DEFAULT_GOLD_PATH})"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["test", "prod"],
        default="test",
        help="LLM mode: test (Ollama) or prod (default: test)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM extraction, use NER/patterns only"
    )
    parser.add_argument(
        "--source-type", "-s",
        type=str,
        choices=["dialogue", "note", "augmented_note"],
        default=None,
        help="Filter to specific source type"
    )
    parser.add_argument(
        "--max", "-n",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions to JSONL file"
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        default=None,
        help="Path to save predictions (default: eval_predictions.jsonl)"
    )
    parser.add_argument(
        "--show-errors", "-e",
        type=int,
        default=10,
        help="Number of errors to show in analysis (default: 10)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-example output"
    )
    
    args = parser.parse_args()
    
    # Set up LLM config if using LLM
    llm_config = None
    use_llm = not args.no_llm
    
    if use_llm:
        try:
            from core.llm_config import LLMConfig
            llm_config = LLMConfig(mode=args.mode)
            print(f"Using LLM mode: {args.mode}")
        except Exception as e:
            print(f"Warning: Could not initialize LLM config: {e}")
            print("Falling back to NER-only extraction")
            use_llm = False
    
    # Run evaluation
    gold_path = Path(args.gold)
    if not gold_path.exists():
        print(f"Error: GOLD file not found: {gold_path}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("EXTRACTOR MICRO-F1 EVALUATION")
    print("=" * 70)
    print(f"GOLD file: {gold_path}")
    print(f"LLM extraction: {use_llm}")
    if args.source_type:
        print(f"Source type filter: {args.source_type}")
    if args.max:
        print(f"Max examples: {args.max}")
    print("=" * 70)
    
    predictions_path = Path(args.predictions_path) if args.predictions_path else None
    
    result = run_evaluation(
        gold_path=gold_path,
        llm_config=llm_config,
        use_llm=use_llm,
        source_type_filter=args.source_type,
        max_examples=args.max,
        verbose=not args.quiet,
        save_predictions=args.save_predictions,
        predictions_path=predictions_path
    )
    
    # Print results
    print("\n")
    print(format_metrics_table(result))
    
    if args.show_errors > 0:
        print(format_error_analysis(result, max_errors=args.show_errors))
    
    # Summary - CRUCIAL INFORMATION ONLY
    print("\n" + "=" * 70)
    print("SUMMARY - CRUCIAL INFORMATION MICRO-F1")
    print("=" * 70)
    
    # Get medication and allergy counts from entity breakdown
    med_counts = result.by_entity_type.get("med", MetricCounts())
    allergy_counts = result.by_entity_type.get("allergy", MetricCounts())
    
    # Calculate crucial information combined score
    crucial_tp = result.demographics.tp + med_counts.tp + allergy_counts.tp + result.vitals.tp
    crucial_fp = result.demographics.fp + med_counts.fp + allergy_counts.fp + result.vitals.fp
    crucial_fn = result.demographics.fn + med_counts.fn + allergy_counts.fn + result.vitals.fn
    
    if crucial_tp + crucial_fp > 0:
        crucial_p = crucial_tp / (crucial_tp + crucial_fp)
    else:
        crucial_p = 0.0
    
    if crucial_tp + crucial_fn > 0:
        crucial_r = crucial_tp / (crucial_tp + crucial_fn)
    else:
        crucial_r = 0.0
    
    if crucial_p + crucial_r > 0:
        crucial_f1 = 2 * crucial_p * crucial_r / (crucial_p + crucial_r)
    else:
        crucial_f1 = 0.0
    
    print(f"\n{'Category':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-" * 70)
    
    # Demographics
    m = result.demographics
    print(f"{'Demographics':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Medications
    m = med_counts
    print(f"{'Medications':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Allergies
    m = allergy_counts
    print(f"{'Allergies':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Vitals
    m = result.vitals
    print(f"{'Vitals':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    print("-" * 70)
    print(f"{'COMBINED':<20} {crucial_p:>8.3f} {crucial_r:>8.3f} {crucial_f1:>8.3f} {crucial_tp:>6} {crucial_fp:>6} {crucial_fn:>6}")
    
    print("\n" + "=" * 70)
    print(f">>> MICRO-F1 SCORE: {crucial_f1:.4f} <<<")
    print("=" * 70)


if __name__ == "__main__":
    main()
