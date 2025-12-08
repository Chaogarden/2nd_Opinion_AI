# diagnoser/critic_engine.py
# ==============================
# Note Critic Engine
# Orchestrates extraction, linkage, guideline retrieval, and scoring
# Produces structured NoteCritique JSON output
# ==============================

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal, Tuple
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.critic_schema import (
    NoteCritique, SourceNoteConsistency, CoverageSalientFindings,
    AssessmentQuality, PlanQualitySafety, OmittedFinding,
    ClaimJudgment, EvidenceCitation, score_to_verdict
)
from diagnoser.claim_linker import (
    parse_dialogue_sentences, extract_claims_from_note,
    EvidenceLinker, build_claim_judgments, LLMNLIVerifier,
    DialogueSentence, NoteClaim
)
from diagnoser.guideline_scorer import (
    GuidelineScorer, retrieve_guidelines_for_critique
)
from diagnoser.schema import EvidenceChunk
from extractor.schema import ExtractorJSON


# ============================================================
# Note Critic Engine
# ============================================================

class NoteCriticEngine:
    """
    Retrieval-Augmented Critic for clinical note evaluation.
    
    Evaluates clinical notes against:
    - Source dialogue (for consistency and coverage)
    - Clinical guidelines (for assessment and plan quality)
    
    Produces structured NoteCritique JSON with evidence-linked verdicts.
    """
    
    def __init__(
        self,
        llm_client,
        llm_model: str,
        embedder=None,
        use_llm_nli: bool = False
    ):
        """
        Initialize the critic engine.
        
        Args:
            llm_client: LLM client with .chat() method
            llm_model: Model name for LLM calls
            embedder: Optional shared SentenceEmbedder
            use_llm_nli: Whether to use LLM for NLI verification
        """
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._embedder = embedder
        self.use_llm_nli = use_llm_nli
        
        # Initialize components
        self._evidence_linker: Optional[EvidenceLinker] = None
        self._guideline_scorer: Optional[GuidelineScorer] = None
    
    @property
    def embedder(self):
        """Get or create the sentence embedder."""
        if self._embedder is None:
            from rag.engine import SentenceEmbedder
            self._embedder = SentenceEmbedder()
        return self._embedder
    
    @property
    def evidence_linker(self) -> EvidenceLinker:
        """Get or create the evidence linker."""
        if self._evidence_linker is None:
            nli_model = None
            if self.use_llm_nli:
                nli_model = LLMNLIVerifier(self.llm_client, self.llm_model)
            self._evidence_linker = EvidenceLinker(
                embedder=self.embedder,
                nli_model=nli_model
            )
        return self._evidence_linker
    
    @property
    def guideline_scorer(self) -> GuidelineScorer:
        """Get or create the guideline scorer."""
        if self._guideline_scorer is None:
            self._guideline_scorer = GuidelineScorer(
                self.llm_client, self.llm_model, self.embedder
            )
        return self._guideline_scorer
    
    def run(
        self,
        dialogue_text: str,
        note_text: str,
        note_type: Literal["prose", "soap"] = "soap",
        extracted_dialogue: Optional[ExtractorJSON] = None,
        extracted_note: Optional[ExtractorJSON] = None,
        guideline_chunks: Optional[List[EvidenceChunk]] = None,
        encounter_id: str = "",
        verbose: bool = False
    ) -> NoteCritique:
        """
        Run the full note critique.
        
        Args:
            dialogue_text: Raw dialogue with [doctor]/[patient] tags
            note_text: Clinical note text (prose or SOAP)
            note_type: "prose" or "soap"
            extracted_dialogue: Pre-extracted dialogue facts (optional)
            extracted_note: Pre-extracted note facts (optional)
            guideline_chunks: Pre-retrieved guidelines (optional)
            encounter_id: Identifier for the encounter
            verbose: Print progress
        
        Returns:
            NoteCritique with all rubric evaluations
        """
        critique = NoteCritique(
            encounter_id=encounter_id,
            note_type=note_type
        )
        
        if verbose:
            print(f"Running note critique for {encounter_id or 'encounter'}...")
        
        # --------------------------------------------------------
        # Step 1: Parse dialogue into sentences
        # --------------------------------------------------------
        if verbose:
            print("  Parsing dialogue...")
        
        dialogue_sentences = parse_dialogue_sentences(dialogue_text)
        
        if verbose:
            print(f"  Found {len(dialogue_sentences)} dialogue sentences")
        
        # --------------------------------------------------------
        # Step 2: Extract if not provided
        # --------------------------------------------------------
        if extracted_dialogue is None:
            if verbose:
                print("  Extracting dialogue facts...")
            extracted_dialogue = self._extract_dialogue(dialogue_text)
        
        if extracted_note is None:
            if verbose:
                print("  Extracting note facts...")
            extracted_note = self._extract_note(note_text, note_type)
        
        # --------------------------------------------------------
        # Step 3: Extract claims from note
        # --------------------------------------------------------
        if verbose:
            print("  Extracting claims from note...")
        
        claims = extract_claims_from_note(note_text, note_type)
        
        if verbose:
            print(f"  Extracted {len(claims)} claims")
        
        # --------------------------------------------------------
        # Step 4: Link claims to dialogue evidence
        # --------------------------------------------------------
        if verbose:
            print("  Linking claims to dialogue...")
        
        linkage_results = self.evidence_linker.link_claims_to_dialogue(
            claims, dialogue_sentences, use_nli=self.use_llm_nli
        )
        claim_judgments = build_claim_judgments(linkage_results)
        
        # --------------------------------------------------------
        # Step 5: Compute source-note consistency
        # --------------------------------------------------------
        if verbose:
            print("  Computing source-note consistency...")
        
        critique.source_note_consistency = self._compute_consistency(
            claim_judgments
        )
        
        # --------------------------------------------------------
        # Step 6: Compute coverage of salient findings
        # --------------------------------------------------------
        if verbose:
            print("  Computing coverage metrics...")
        
        critique.coverage_of_salient_findings = self._compute_coverage(
            extracted_dialogue, extracted_note, dialogue_sentences
        )
        
        # --------------------------------------------------------
        # Step 7: Retrieve guidelines
        # --------------------------------------------------------
        if guideline_chunks is None:
            if verbose:
                print("  Retrieving relevant guidelines...")
            guideline_chunks = retrieve_guidelines_for_critique(
                extracted_dialogue, k=5, embedder=self.embedder
            )
        
        if verbose:
            print(f"  Retrieved {len(guideline_chunks)} guideline chunks")
        
        # --------------------------------------------------------
        # Step 8: Extract diagnoses and plan items from note
        # --------------------------------------------------------
        diagnoses = self._extract_diagnoses(extracted_note, claims)
        plan_items = self._extract_plan_items(extracted_note, claims)
        
        if verbose:
            print(f"  Found {len(diagnoses)} diagnoses, {len(plan_items)} plan items")
        
        # --------------------------------------------------------
        # Step 9: Score assessment quality
        # --------------------------------------------------------
        if verbose:
            print("  Scoring assessment quality...")
        
        critique.assessment_quality = self.guideline_scorer.score_assessment(
            extracted_dialogue, diagnoses, guideline_chunks
        )
        
        # --------------------------------------------------------
        # Step 10: Score plan quality and safety
        # --------------------------------------------------------
        if verbose:
            print("  Scoring plan safety...")
        
        critique.plan_quality_and_safety = self.guideline_scorer.score_plan(
            extracted_dialogue, diagnoses, plan_items, guideline_chunks
        )
        
        # --------------------------------------------------------
        # Step 11: Finalize critique
        # --------------------------------------------------------
        critique = critique.finalize()
        
        if verbose:
            print(f"  Overall score: {critique.overall_score:.2f}")
            print(f"  Overall verdict: {critique.overall_verdict}")
            print(f"  Safety: {critique.overall_safety}")
        
        return critique
    
    def _extract_dialogue(self, dialogue_text: str) -> ExtractorJSON:
        """Extract facts from dialogue."""
        try:
            from data.load_ACI_dataset import aci_dialogue_to_turns
            from extractor.hybrid_extractor import HybridExtractor
            
            # Parse dialogue to turns
            turns = aci_dialogue_to_turns(dialogue_text)
            
            if not turns:
                return ExtractorJSON()
            
            # RxNorm path
            rxnorm_path = ROOT / "models" / "rxnorm_names.tsv"
            
            # Create extractor with our LLM client
            extractor = HybridExtractor(
                symptom_backend="scispacy",
                med_backend="scispacy",
                rxnorm_tsv_path=str(rxnorm_path) if rxnorm_path.exists() else None,
                llm_client=self.llm_client,
                qa_model_name=self.llm_model,
                use_llm_extraction=True,
                use_ner_enrichment=True
            )
            
            return extractor.extract(turns)
        except Exception as e:
            print(f"Dialogue extraction failed: {e}")
            return ExtractorJSON()
    
    def _extract_note(self, note_text: str, note_type: str) -> ExtractorJSON:
        """Extract facts from note."""
        try:
            from extractor.note_extractor import NoteExtractor
            
            extractor = NoteExtractor(self.llm_client, self.llm_model)
            return extractor.extract(note_text, note_type=note_type)
        except Exception as e:
            print(f"Note extraction failed: {e}")
            return ExtractorJSON()
    
    def _compute_consistency(
        self,
        judgments: List[ClaimJudgment]
    ) -> SourceNoteConsistency:
        """Compute source-note consistency from claim judgments."""
        if not judgments:
            return SourceNoteConsistency()
        
        total = len(judgments)
        entailed = sum(1 for j in judgments if j.verdict == "entailed")
        contradicted = sum(1 for j in judgments if j.verdict == "contradicted")
        unsupported = sum(1 for j in judgments if j.verdict == "unsupported")
        
        support_score = entailed / total if total > 0 else 0
        contradiction_score = contradicted / total if total > 0 else 0
        hallucination_density = unsupported / total if total > 0 else 0
        
        # Determine verdict
        if support_score >= 0.7 and contradiction_score == 0:
            verdict = "good"
        elif contradiction_score > 0.1 or hallucination_density > 0.3:
            verdict = "poor"
        else:
            verdict = "borderline"
        
        return SourceNoteConsistency(
            support_score=support_score,
            contradiction_score=contradiction_score,
            hallucination_density=hallucination_density,
            verdict=verdict,
            claim_judgments=judgments,
            total_claims_evaluated=total
        )
    
    def _compute_coverage(
        self,
        extracted_dialogue: ExtractorJSON,
        extracted_note: ExtractorJSON,
        dialogue_sentences: List[DialogueSentence]
    ) -> CoverageSalientFindings:
        """Compute coverage of salient dialogue findings in the note."""
        result = CoverageSalientFindings()
        omissions = []
        
        # Compare symptoms
        dialogue_symptoms = {
            (s.name_norm or s.name_surface).lower()
            for s in extracted_dialogue.symptoms
            if s.assertion == "present"
        }
        note_symptoms = {
            (s.name_norm or s.name_surface).lower()
            for s in extracted_note.symptoms
            if s.assertion == "present"
        }
        
        if dialogue_symptoms:
            covered = dialogue_symptoms & note_symptoms
            result.symptom_recall = len(covered) / len(dialogue_symptoms)
            
            for missed in dialogue_symptoms - note_symptoms:
                omissions.append(OmittedFinding(
                    finding_type="symptom",
                    finding_text=missed,
                    importance="moderate"
                ))
        
        # Compare medications
        dialogue_meds = {
            (m.name_norm or m.name_surface).lower()
            for m in extracted_dialogue.meds
            if m.assertion == "present"
        }
        note_meds = {
            (m.name_norm or m.name_surface).lower()
            for m in extracted_note.meds
            if m.assertion == "present"
        }
        
        if dialogue_meds:
            covered = dialogue_meds & note_meds
            result.medication_recall = len(covered) / len(dialogue_meds)
            
            for missed in dialogue_meds - note_meds:
                omissions.append(OmittedFinding(
                    finding_type="medication",
                    finding_text=missed,
                    importance="high"
                ))
        
        # Compare allergies (critical)
        dialogue_allergies = {
            (a.substance_norm or a.substance_surface).lower()
            for a in extracted_dialogue.allergies
        }
        note_allergies = {
            (a.substance_norm or a.substance_surface).lower()
            for a in extracted_note.allergies
        }
        
        if dialogue_allergies:
            covered = dialogue_allergies & note_allergies
            result.allergy_recall = len(covered) / len(dialogue_allergies)
            
            for missed in dialogue_allergies - note_allergies:
                omissions.append(OmittedFinding(
                    finding_type="allergy",
                    finding_text=missed,
                    importance="critical"
                ))
                result.critical_omissions += 1
        
        # Compare risk factors
        dialogue_rf = {rf.lower() for rf in extracted_dialogue.risk_factors}
        note_rf = {rf.lower() for rf in extracted_note.risk_factors}
        
        if dialogue_rf:
            covered = dialogue_rf & note_rf
            result.risk_factor_recall = len(covered) / len(dialogue_rf)
            
            for missed in dialogue_rf - note_rf:
                omissions.append(OmittedFinding(
                    finding_type="risk_factor",
                    finding_text=missed,
                    importance="moderate"
                ))
        
        # Overall completeness
        recalls = [
            result.symptom_recall,
            result.medication_recall,
            result.allergy_recall,
            result.risk_factor_recall
        ]
        non_zero = [r for r in recalls if r > 0]
        result.completeness_score = sum(non_zero) / len(non_zero) if non_zero else 0.5
        
        # Omission rate
        total_dialogue = (
            len(dialogue_symptoms) + len(dialogue_meds) +
            len(dialogue_allergies) + len(dialogue_rf)
        )
        if total_dialogue > 0:
            result.omission_rate = len(omissions) / total_dialogue
        
        result.omissions = omissions
        
        # Determine verdict
        if result.completeness_score >= 0.8 and result.critical_omissions == 0:
            result.verdict = "good"
        elif result.critical_omissions > 0 or result.completeness_score < 0.5:
            result.verdict = "poor"
        else:
            result.verdict = "borderline"
        
        return result
    
    def _extract_diagnoses(
        self,
        extracted_note: ExtractorJSON,
        claims: List[NoteClaim]
    ) -> List[str]:
        """Extract diagnoses from note extraction and claims."""
        diagnoses = []
        
        # From QA extractions
        for qa in extracted_note.qa_extractions:
            if qa.concept.startswith("diagnosis"):
                diagnoses.append(qa.value or qa.concept)
        
        # From claims in Assessment section
        for claim in claims:
            if claim.section == "A" and claim.claim_type == "diagnosis":
                diagnoses.append(claim.text)
        
        return diagnoses[:10]  # Limit
    
    def _extract_plan_items(
        self,
        extracted_note: ExtractorJSON,
        claims: List[NoteClaim]
    ) -> List[str]:
        """Extract plan items from note extraction and claims."""
        items = []
        
        # From QA extractions
        for qa in extracted_note.qa_extractions:
            if qa.concept.startswith("plan"):
                items.append(qa.value or qa.concept)
        
        # From claims in Plan section
        for claim in claims:
            if claim.section == "P":
                items.append(claim.text)
        
        # From medications in note
        for med in extracted_note.meds:
            if med.assertion == "present":
                dose_info = f"{med.dose or ''} {med.freq or ''}".strip()
                items.append(f"{med.name_surface} {dose_info}".strip())
        
        return items[:15]  # Limit


# ============================================================
# Explanation Integration
# ============================================================

def run_critique_with_explanation(
    dialogue_text: str,
    note_text: str,
    note_type: Literal["prose", "soap"] = "soap",
    llm_client=None,
    llm_model: str = "",
    encounter_id: str = "",
    verbose: bool = False
) -> Tuple[NoteCritique, 'CritiqueExplanation']:
    """
    Run critique and generate natural language explanation.
    
    Args:
        dialogue_text: Raw dialogue text
        note_text: Clinical note text
        note_type: "prose" or "soap"
        llm_client: LLM client (required)
        llm_model: LLM model name (required)
        encounter_id: Encounter identifier
        verbose: Print progress
    
    Returns:
        Tuple of (NoteCritique, CritiqueExplanation)
    """
    from diagnoser.critique_explainer import CritiqueExplainer
    
    if llm_client is None:
        raise ValueError("llm_client is required")
    if not llm_model:
        raise ValueError("llm_model is required")
    
    # Run critique
    engine = NoteCriticEngine(llm_client, llm_model)
    critique = engine.run(
        dialogue_text=dialogue_text,
        note_text=note_text,
        note_type=note_type,
        encounter_id=encounter_id,
        verbose=verbose
    )
    
    # Generate explanation
    if verbose:
        print("Generating natural language explanation...")
    
    explainer = CritiqueExplainer(llm_client, llm_model)
    explanation = explainer.explain(critique)
    
    return critique, explanation


# ============================================================
# Convenience Function
# ============================================================

def run_note_critic(
    dialogue_text: str,
    note_text: str,
    note_type: Literal["prose", "soap"] = "soap",
    llm_client=None,
    llm_model: str = "",
    extracted_dialogue: Optional[ExtractorJSON] = None,
    extracted_note: Optional[ExtractorJSON] = None,
    guideline_chunks: Optional[List[EvidenceChunk]] = None,
    encounter_id: str = "",
    verbose: bool = False
) -> NoteCritique:
    """
    High-level function to run note critique.
    
    Args:
        dialogue_text: Raw dialogue text
        note_text: Clinical note text
        note_type: "prose" or "soap"
        llm_client: LLM client (required)
        llm_model: LLM model name (required)
        extracted_dialogue: Pre-extracted dialogue (optional)
        extracted_note: Pre-extracted note (optional)
        guideline_chunks: Pre-retrieved guidelines (optional)
        encounter_id: Encounter identifier
        verbose: Print progress
    
    Returns:
        NoteCritique with full evaluation
    """
    if llm_client is None:
        raise ValueError("llm_client is required")
    if not llm_model:
        raise ValueError("llm_model is required")
    
    engine = NoteCriticEngine(llm_client, llm_model)
    
    return engine.run(
        dialogue_text=dialogue_text,
        note_text=note_text,
        note_type=note_type,
        extracted_dialogue=extracted_dialogue,
        extracted_note=extracted_note,
        guideline_chunks=guideline_chunks,
        encounter_id=encounter_id,
        verbose=verbose
    )


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run note critique on ACI dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Critique a single encounter with explanation
  python -m diagnoser.critic_engine --encounter enc_123 --explain --verbose
  
  # Batch critique
  python -m diagnoser.critic_engine --split test --max 5 --output critiques/
  
  # Batch critique with natural language explanations
  python -m diagnoser.critic_engine --split test --max 5 --output critiques/ --explain
        """
    )
    
    parser.add_argument("--encounter", "-e", type=str, help="ACI encounter ID")
    parser.add_argument("--split", "-s", type=str, default="test",
                        choices=["test", "validation", "train"],
                        help="Dataset split for batch processing")
    parser.add_argument("--max", "-n", type=int, default=5,
                        help="Maximum encounters to process")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for critique JSONs")
    parser.add_argument("--mode", "-m", type=str, default="test",
                        choices=["test", "prod"],
                        help="LLM mode")
    parser.add_argument("--explain", "-x", action="store_true",
                        help="Generate natural language explanations of critiques")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress")
    
    args = parser.parse_args()
    
    # Set up LLM config
    try:
        from core.llm_config import LLMConfig
        config = LLMConfig(mode=args.mode)
        print(f"Using LLM mode: {args.mode}")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        sys.exit(1)
    
    # Load encounters
    from data.load_ACI_dataset import extract_aci_dialogues
    
    if args.encounter:
        # Single encounter
        encounters = extract_aci_dialogues(args.split, include_notes=True)
        encounter = next((e for e in encounters if e["encounter_id"] == args.encounter), None)
        
        if encounter is None:
            print(f"Encounter {args.encounter} not found")
            sys.exit(1)
        
        encounters = [encounter]
    else:
        # Batch
        encounters = extract_aci_dialogues(
            args.split, max_examples=args.max, include_notes=True
        )
    
    print(f"\nProcessing {len(encounters)} encounters...")
    print("=" * 60)
    
    # Initialize engine
    engine = NoteCriticEngine(config.client, config.diagnoser_model)
    
    # Initialize explainer if requested
    explainer = None
    if args.explain:
        from diagnoser.critique_explainer import CritiqueExplainer, format_explanation_text
        explainer = CritiqueExplainer(config.client, config.diagnoser_model)
        print("Explanation generation: ENABLED")
    
    # Output directory
    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for i, enc in enumerate(encounters):
        encounter_id = enc["encounter_id"]
        dialogue = enc.get("dialogue", "")
        note = enc.get("augmented_note", "") or enc.get("note", "")
        note_type = "soap" if enc.get("augmented_note") else "prose"
        
        print(f"\n[{i+1}/{len(encounters)}] {encounter_id}")
        
        if not dialogue or not note:
            print("  Skipping: missing dialogue or note")
            continue
        
        try:
            critique = engine.run(
                dialogue_text=dialogue,
                note_text=note,
                note_type=note_type,
                encounter_id=encounter_id,
                verbose=args.verbose
            )
            
            print(f"  Overall: {critique.overall_verdict} (score={critique.overall_score:.2f})")
            print(f"  Safety: {critique.overall_safety}")
            print(f"  Consistency: {critique.source_note_consistency.verdict}")
            print(f"  Coverage: {critique.coverage_of_salient_findings.verdict}")
            
            if output_dir:
                out_path = output_dir / f"{encounter_id}_critique.json"
                with out_path.open("w") as f:
                    json.dump(critique.model_dump(), f, indent=2, default=str)
                print(f"  Saved to {out_path}")
            
            # Generate explanation if requested
            explanation = None
            if explainer:
                print("  Generating explanation...")
                explanation = explainer.explain(critique)
                print(f"  Rating: {explanation.overall_rating}")
                
                if output_dir:
                    # Save explanation JSON
                    exp_path = output_dir / f"{encounter_id}_explanation.json"
                    with exp_path.open("w") as f:
                        json.dump(explanation.to_dict(), f, indent=2)
                    
                    # Save explanation as readable text
                    txt_path = output_dir / f"{encounter_id}_explanation.txt"
                    with txt_path.open("w") as f:
                        f.write(format_explanation_text(explanation))
                    
                    print(f"  Explanation saved to {txt_path}")
                else:
                    # Print explanation to console
                    print("\n" + format_explanation_text(explanation))
            
            results.append({
                "encounter_id": encounter_id,
                "overall_score": critique.overall_score,
                "overall_verdict": critique.overall_verdict,
                "safety": critique.overall_safety,
                "explanation_rating": explanation.overall_rating if explanation else None
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "encounter_id": encounter_id,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if "error" not in r]
    if successful:
        avg_score = sum(r["overall_score"] for r in successful) / len(successful)
        print(f"Processed: {len(successful)}/{len(results)}")
        print(f"Average score: {avg_score:.2f}")
        
        verdicts = {}
        for r in successful:
            v = r["overall_verdict"]
            verdicts[v] = verdicts.get(v, 0) + 1
        print(f"Verdicts: {verdicts}")
        
        safety = {}
        for r in successful:
            s = r["safety"]
            safety[s] = safety.get(s, 0) + 1
        print(f"Safety: {safety}")


if __name__ == "__main__":
    main()
