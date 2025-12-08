# core/pipeline.py
# ==============================
# End-to-end clinical reasoning pipeline orchestration
# ASR -> Extraction -> RAG -> Diagnoser -> Consultant -> Arbiter
# ==============================

from typing import Optional, Dict, Any, List
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dialogue import (
    build_dialogue, build_dialogue_from_text, build_dialogue_brief,
    infer_role_mapping
)
from core.llm_config import LLMConfig, build_clients
from extractor.schema import ExtractorJSON
from extractor.hybrid_extractor import HybridExtractor
from rag.engine import collect_evidence, SentenceEmbedder
from diagnoser.schema import (
    DiagnoserInput, DiagnoserOutput, ConsultantCritique,
    ArbiterResult, PipelineResult, EvidenceChunk
)
from diagnoser.engine import DiagnoserEngine, ConsultantEngine
from diagnoser.arbiter import apply_critique
from diagnoser.diagnosis_explainer import DiagnosisExplainer
from diagnoser.critic_engine import NoteCriticEngine
from diagnoser.critique_explainer import CritiqueExplainer


# ============================================================
# Cached resource loader (for Streamlit compatibility)
# ============================================================

_cached_extractor: Optional[HybridExtractor] = None
_cached_embedder: Optional[SentenceEmbedder] = None


def get_hybrid_extractor(
    llm_config: Optional[LLMConfig] = None,
    rxnorm_tsv_path: str = "models/rxnorm_names.tsv",
    use_llm_extraction: bool = True,
    force_new: bool = False
) -> HybridExtractor:
    """
    Get or create a HybridExtractor instance.
    
    Args:
        llm_config: LLM configuration for extraction.
        rxnorm_tsv_path: Path to RxNorm TSV file for medication normalization.
        use_llm_extraction: Whether to use LLM as primary extractor.
        force_new: If True, always create a new instance (don't use cache).
    
    Returns:
        HybridExtractor instance.
    """
    global _cached_extractor
    
    # For LLM extraction, we don't cache since config may change
    # For NER-only, we can cache the heavy NER models
    if not force_new and not use_llm_extraction and _cached_extractor is not None:
        return _cached_extractor
    
    # Resolve path
    full_path = Path(ROOT) / rxnorm_tsv_path
    tsv_path = str(full_path) if full_path.exists() else None
    
    # Get LLM client for extraction
    llm_client = None
    qa_model = None
    if llm_config is not None:
        llm_client = llm_config.client
        qa_model = llm_config.diagnoser_model  # Use diagnoser model for extraction
    
    extractor = HybridExtractor(
        symptom_backend="scispacy",
        med_backend="scispacy",
        rxnorm_tsv_path=tsv_path,
        llm_client=llm_client,
        qa_model_name=qa_model,
        use_llm_extraction=use_llm_extraction,
        use_ner_enrichment=True
    )
    
    # Only cache if not using LLM extraction
    if not use_llm_extraction:
        _cached_extractor = extractor
    
    return extractor


def get_embedder() -> SentenceEmbedder:
    """Get or create a cached SentenceEmbedder."""
    global _cached_embedder
    
    if _cached_embedder is None:
        _cached_embedder = SentenceEmbedder()
    
    return _cached_embedder


def clear_caches():
    """Clear all cached resources."""
    global _cached_extractor, _cached_embedder
    _cached_extractor = None
    _cached_embedder = None


# ============================================================
# Pipeline Execution
# ============================================================

def run_clinical_pipeline(
    transcript_result: Dict[str, Any],
    role_mapping: Optional[Dict[str, str]] = None,
    llm_config: Optional[LLMConfig] = None,
    k_guidelines: int = 5,
    k_merck: int = 5,
    guidelines_filter: Optional[str] = None,
    skip_consultant: bool = False,
    skip_revision: bool = True,  # Skip revision pass by default (saves an LLM call)
    use_llm_extraction: bool = True,
    doctor_note: Optional[str] = None,  # Optional doctor's note to critique
    note_type: str = "soap"  # "prose" or "soap"
) -> PipelineResult:
    """
    Run the full clinical reasoning pipeline.
    
    Pipeline stages:
    1. Build dialogue from transcript
    2. Extract medical facts using HybridExtractor (LLM-first with NER enrichment)
    3. Collect evidence from RAG (guidelines + Merck)
    4. Run Diagnoser LLM
    5. Run Consultant LLM (critique)
    6. Apply Arbiter rules
    7. Generate Diagnosis Explanation
    8. (Optional) Critique doctor's note if provided
    
    Args:
        transcript_result: Dict with 'text' and optionally 'segments' from ASR.
        role_mapping: Optional speaker->role mapping from diarization.
        llm_config: LLM configuration (models and client).
        k_guidelines: Number of guideline chunks to retrieve.
        k_merck: Number of Merck chunks to retrieve.
        guidelines_filter: Optional corpus filter for guidelines.
        skip_consultant: If True, skip Consultant and Arbiter stages.
        skip_revision: If True, skip the Diagnoser revision pass (default: True).
        use_llm_extraction: If True, use LLM as primary extractor (recommended).
        doctor_note: Optional doctor's clinical note to critique.
        note_type: Type of note - "prose" or "soap".
    
    Returns:
        PipelineResult with all outputs.
    """
    # Default LLM config
    if llm_config is None:
        llm_config = LLMConfig()
    
    # --------------------------------------------------------
    # Stage 1: Build dialogue
    # --------------------------------------------------------
    segments = transcript_result.get("segments", [])
    full_text = transcript_result.get("text", "")
    
    if segments:
        # Diarized segments available
        if role_mapping is None:
            role_mapping = infer_role_mapping(segments)
        dialogue = build_dialogue(segments, mapping=role_mapping)
    elif full_text:
        # Free-text input only
        dialogue = build_dialogue_from_text(full_text, default_role="PATIENT")
    else:
        # No input
        return PipelineResult(
            extracted_facts=ExtractorJSON(),
            error="No transcript data provided"
        )
    
    if not dialogue:
        return PipelineResult(
            extracted_facts=ExtractorJSON(),
            error="Could not build dialogue from transcript"
        )
    
    # Build dialogue brief for LLM context
    dialogue_brief = build_dialogue_brief(dialogue)
    
    # --------------------------------------------------------
    # Stage 2: Extract medical facts (LLM-first with NER enrichment)
    # --------------------------------------------------------
    try:
        extractor = get_hybrid_extractor(
            llm_config=llm_config,
            use_llm_extraction=use_llm_extraction,
            force_new=use_llm_extraction  # Don't cache when using LLM
        )
        extracted_facts = extractor.extract(dialogue)
    except Exception as e:
        return PipelineResult(
            extracted_facts=ExtractorJSON(),
            dialogue_brief=dialogue_brief,
            error=f"Extraction failed: {str(e)}"
        )
    
    # --------------------------------------------------------
    # Stage 3: Collect RAG evidence
    # --------------------------------------------------------
    try:
        embedder = get_embedder()
        evidence_chunks = collect_evidence(
            extracted_facts,
            k_guidelines=k_guidelines,
            k_merck=k_merck,
            embedder=embedder,
            guidelines_filter=guidelines_filter
        )
    except Exception as e:
        # Continue without evidence if RAG fails
        evidence_chunks = []
        print(f"Warning: RAG evidence collection failed: {e}")
    
    # --------------------------------------------------------
    # Stage 4: Run Diagnoser
    # --------------------------------------------------------
    diagnoser_input = DiagnoserInput(
        patient_facts=extracted_facts,
        dialogue_brief=dialogue_brief,
        evidence_chunks=evidence_chunks
    )
    
    try:
        diagnoser_engine = DiagnoserEngine(
            client=llm_config.client,
            model=llm_config.diagnoser_model
        )
        diagnoser_output = diagnoser_engine.run(diagnoser_input)
    except Exception as e:
        return PipelineResult(
            extracted_facts=extracted_facts,
            evidence_chunks=evidence_chunks,
            dialogue_brief=dialogue_brief,
            error=f"Diagnoser failed: {str(e)}"
        )
    
    # --------------------------------------------------------
    # Stage 5: Run Consultant (if not skipped)
    # --------------------------------------------------------
    consultant_critique = None
    arbiter_result = None
    revised_output = None
    
    if not skip_consultant:
        try:
            consultant_engine = ConsultantEngine(
                client=llm_config.client,
                model=llm_config.consultant_model
            )
            consultant_critique = consultant_engine.run(
                diagnoser_input,
                diagnoser_output
            )
            
            # --------------------------------------------------------
            # Stage 5b: Revision Pass (if enabled and critique has addressable issues)
            # --------------------------------------------------------
            if not skip_revision:
                has_addressable_issues = (
                    consultant_critique.issues and
                    consultant_critique.overall_safety_rating != "unsafe" and
                    not any(issue.severity == "critical" for issue in consultant_critique.issues)
                )
                
                if has_addressable_issues:
                    print("  Consultant raised issues - requesting Diagnoser revision...")
                    revised_output = diagnoser_engine.revise(
                        diagnoser_input,
                        diagnoser_output,
                        consultant_critique
                    )
                    # Use revised output for Arbiter
                    output_for_arbiter = revised_output
                else:
                    output_for_arbiter = diagnoser_output
            else:
                output_for_arbiter = diagnoser_output
            
            # --------------------------------------------------------
            # Stage 6: Apply Arbiter
            # --------------------------------------------------------
            arbiter_result = apply_critique(output_for_arbiter, consultant_critique)
            
        except Exception as e:
            # Continue without critique if Consultant fails
            print(f"Warning: Consultant/Arbiter failed: {e}")
            arbiter_result = ArbiterResult(
                final_plan=diagnoser_output,
                abstained=False,
                patches_applied=[],
                notes_on_missing_info=f"Consultant review skipped due to error: {str(e)}"
            )
    else:
        # Skip Consultant - use Diagnoser output directly
        arbiter_result = ArbiterResult(
            final_plan=diagnoser_output,
            abstained=False,
            patches_applied=[],
            notes_on_missing_info="Consultant review was skipped"
        )
    
    # --------------------------------------------------------
    # Stage 7: Generate Diagnosis Explanation (Patient-Friendly)
    # --------------------------------------------------------
    diagnosis_explanation = None
    final_output = arbiter_result.final_plan if arbiter_result and arbiter_result.final_plan else diagnoser_output
    
    if final_output:
        try:
            print("  Generating patient-friendly diagnosis explanation...")
            explainer = DiagnosisExplainer(
                client=llm_config.client,
                model=llm_config.diagnoser_model
            )
            explanation = explainer.explain(
                final_output,
                extracted_facts,
                evidence_chunks
            )
            diagnosis_explanation = explanation.to_dict()
        except Exception as e:
            print(f"Warning: Diagnosis explanation generation failed: {e}")
            diagnosis_explanation = None
    
    # --------------------------------------------------------
    # Stage 8: Critique Doctor's Note (if provided)
    # --------------------------------------------------------
    critique_explanation = None
    
    print(f"  Doctor note provided: {bool(doctor_note and doctor_note.strip())}")
    if doctor_note:
        print(f"  Note length: {len(doctor_note)} chars")
    
    if doctor_note and doctor_note.strip():
        try:
            print("  Critiquing doctor's note...")
            # Run note critic
            critic_engine = NoteCriticEngine(
                llm_client=llm_config.client,
                llm_model=llm_config.diagnoser_model
            )
            note_critique = critic_engine.run(
                dialogue_text=full_text,
                note_text=doctor_note,
                note_type=note_type,
                verbose=False
            )
            
            # Generate explanation
            print("  Generating critique explanation...")
            critique_explainer = CritiqueExplainer(
                client=llm_config.client,
                model=llm_config.diagnoser_model
            )
            crit_explanation = critique_explainer.explain(note_critique)
            critique_explanation = crit_explanation.to_dict()
            
        except Exception as e:
            import traceback
            print(f"Warning: Note critique failed: {e}")
            print(traceback.format_exc())
            critique_explanation = None
    
    return PipelineResult(
        extracted_facts=extracted_facts,
        evidence_chunks=evidence_chunks,
        diagnoser_output=diagnoser_output,
        revised_diagnoser_output=revised_output,
        consultant_critique=consultant_critique,
        arbiter_result=arbiter_result,
        dialogue_brief=dialogue_brief,
        diagnosis_explanation=diagnosis_explanation,
        critique_explanation=critique_explanation
    )


# ============================================================
# Streamlit-specific helpers
# ============================================================

def get_streamlit_cached_extractor():
    """
    Factory function for use with @st.cache_resource.
    
    Usage in Streamlit:
        @st.cache_resource
        def get_extractor():
            return get_streamlit_cached_extractor()
    """
    return get_hybrid_extractor()


def get_streamlit_cached_embedder():
    """
    Factory function for use with @st.cache_resource.
    
    Usage in Streamlit:
        @st.cache_resource
        def get_embedder():
            return get_streamlit_cached_embedder()
    """
    return get_embedder()

