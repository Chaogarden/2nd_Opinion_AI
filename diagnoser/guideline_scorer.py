# diagnoser/guideline_scorer.py
# ==============================
# Guideline-based Assessment and Plan Scoring
# Uses RAG to retrieve guidelines and LLM to score adherence
# ==============================

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.critic_schema import (
    GuidelineCitation, GuidelineCheck, SafetyCheck,
    DiagnosisEvaluation, MissedDifferential,
    AssessmentQuality, PlanQualitySafety
)
from diagnoser.schema import EvidenceChunk
from extractor.schema import ExtractorJSON


# ============================================================
# LLM Prompts for Guideline-Based Scoring
# ============================================================

ASSESSMENT_SCORING_PROMPT = """You are a clinical expert evaluating the quality of a diagnostic assessment.

PATIENT FACTS (extracted from dialogue):
{patient_facts}

DIAGNOSES FROM NOTE:
{diagnoses}

RELEVANT CLINICAL GUIDELINES:
{guidelines}

Evaluate the assessment and respond with ONLY a JSON object (no other text):

{{
  "dx_coherence_score": 0.0-1.0 (how well diagnoses match the evidence profile),
  "differential_breadth_score": 0.0-1.0 (whether plausible alternatives were considered),
  "verdict": "good" | "borderline" | "poor",
  "diagnosis_evaluations": [
    {{
      "diagnosis": "diagnosis name",
      "coherence_with_evidence": 0.0-1.0,
      "guideline_support": ["evidence_id1", "evidence_id2"],
      "concerns": ["concern1", "concern2"]
    }}
  ],
  "missed_differentials": [
    {{
      "condition": "missed diagnosis",
      "reasoning": "why it should be considered",
      "is_red_flag": true/false,
      "guideline_evidence": ["evidence_id"]
    }}
  ]
}}

IMPORTANT:
- Output ONLY valid JSON, no explanation
- Reference guidelines by their evidence_id when citing
- Score conservatively (0.7+ = clearly good, 0.4-0.7 = borderline)
- Flag red-flag diagnoses that could be dangerous if missed"""


PLAN_SCORING_PROMPT = """You are a clinical expert evaluating a treatment plan for guideline adherence and safety.

PATIENT FACTS:
{patient_facts}

DIAGNOSES:
{diagnoses}

TREATMENT PLAN:
{plan_items}

ALLERGIES:
{allergies}

CURRENT MEDICATIONS:
{current_meds}

RELEVANT CLINICAL GUIDELINES:
{guidelines}

Evaluate the plan and respond with ONLY a JSON object (no other text):

{{
  "guideline_adherence_score": 0.0-1.0,
  "safety_score": 0.0-1.0,
  "verdict": "safe" | "needs_review" | "unsafe",
  "guideline_checks": [
    {{
      "plan_item": "item description",
      "item_type": "diagnostic_test" | "medication" | "referral" | "lifestyle" | "follow_up" | "other",
      "guideline_adherence": "aligned" | "deviation" | "uncertain",
      "guideline_citations": ["evidence_id"],
      "deviation_explanation": "if deviation, explain why"
    }}
  ],
  "safety_checks": [
    {{
      "check_type": "drug_allergy" | "drug_interaction" | "dosing_range" | "contraindication" | "missing_follow_up" | "other",
      "status": "pass" | "warning" | "fail",
      "description": "description of the check",
      "related_items": ["item1", "item2"],
      "severity": "critical" | "high" | "moderate" | "low"
    }}
  ],
  "missing_return_precautions": true/false,
  "missing_follow_up": true/false
}}

IMPORTANT:
- Output ONLY valid JSON, no explanation
- Check for drug-allergy conflicts
- Check for drug-drug interactions
- Verify dosing is within normal ranges
- Flag missing safety nets (return precautions, follow-up)
- Reference guidelines by evidence_id"""


# ============================================================
# Guideline Retrieval
# ============================================================

def retrieve_guidelines_for_critique(
    extracted: ExtractorJSON,
    k: int = 5,
    embedder=None
) -> List[EvidenceChunk]:
    """
    Retrieve relevant guidelines for critiquing an assessment/plan.
    
    Args:
        extracted: ExtractorJSON with patient facts
        k: Number of guideline chunks to retrieve
        embedder: Optional shared embedder
    
    Returns:
        List of EvidenceChunk from guidelines corpus
    """
    from rag.engine import GuidelinesRAG, build_rag_query
    
    # Build query from extracted facts
    query = build_rag_query(extracted)
    
    try:
        rag = GuidelinesRAG(embedder=embedder)
        chunks = rag.search(query, k=k)
        return chunks
    except Exception as e:
        print(f"Guideline retrieval failed: {e}")
        return []


def format_guidelines_for_prompt(chunks: List[EvidenceChunk], max_chars: int = 4000) -> str:
    """
    Format guideline chunks for inclusion in LLM prompt.
    
    Args:
        chunks: List of EvidenceChunk
        max_chars: Maximum total characters
    
    Returns:
        Formatted string with evidence IDs
    """
    lines = []
    total_chars = 0
    
    for chunk in chunks:
        entry = f"[{chunk.evidence_id}] {chunk.title} - {chunk.heading_path}\n{chunk.text[:500]}"
        
        if total_chars + len(entry) > max_chars:
            break
        
        lines.append(entry)
        total_chars += len(entry)
    
    if not lines:
        return "No relevant guidelines found."
    
    return "\n\n".join(lines)


def format_patient_facts(extracted: ExtractorJSON) -> str:
    """Format extracted facts for prompt."""
    parts = []
    
    if extracted.chief_complaint:
        parts.append(f"Chief Complaint: {extracted.chief_complaint}")
    
    symptoms = [s.name_surface for s in extracted.symptoms if s.assertion == "present"]
    if symptoms:
        parts.append(f"Symptoms: {', '.join(symptoms[:10])}")
    
    absent = [s.name_surface for s in extracted.symptoms if s.assertion == "absent"]
    if absent:
        parts.append(f"Pertinent Negatives: {', '.join(absent[:5])}")
    
    vitals = [f"{v.kind}={v.value}" for v in extracted.vitals]
    if vitals:
        parts.append(f"Vitals: {', '.join(vitals)}")
    
    meds = [m.name_surface for m in extracted.meds if m.assertion == "present"]
    if meds:
        parts.append(f"Current Medications: {', '.join(meds)}")
    
    if extracted.risk_factors:
        parts.append(f"Risk Factors: {', '.join(extracted.risk_factors[:5])}")
    
    return "\n".join(parts) if parts else "No patient facts available."


# ============================================================
# LLM-based Scoring
# ============================================================

class GuidelineScorer:
    """
    Scores assessments and plans against clinical guidelines using LLM.
    """
    
    def __init__(self, client, model: str, embedder=None):
        """
        Initialize the guideline scorer.
        
        Args:
            client: LLM client with .chat() method
            model: Model name for scoring
            embedder: Optional shared embedder for RAG
        """
        self.client = client
        self.model = model
        self.embedder = embedder
    
    def score_assessment(
        self,
        extracted: ExtractorJSON,
        diagnoses: List[str],
        guideline_chunks: Optional[List[EvidenceChunk]] = None
    ) -> AssessmentQuality:
        """
        Score the quality of diagnostic assessment.
        
        Args:
            extracted: Patient facts from extraction
            diagnoses: List of diagnoses from the note
            guideline_chunks: Pre-retrieved guidelines (will retrieve if None)
        
        Returns:
            AssessmentQuality with scores and evaluations
        """
        # Retrieve guidelines if not provided
        if guideline_chunks is None:
            guideline_chunks = retrieve_guidelines_for_critique(
                extracted, k=5, embedder=self.embedder
            )
        
        # Format prompt inputs
        patient_facts = format_patient_facts(extracted)
        guidelines = format_guidelines_for_prompt(guideline_chunks)
        diagnoses_str = "\n".join(f"- {dx}" for dx in diagnoses) if diagnoses else "None documented"
        
        prompt = ASSESSMENT_SCORING_PROMPT.format(
            patient_facts=patient_facts,
            diagnoses=diagnoses_str,
            guidelines=guidelines
        )
        
        # Call LLM
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = self._extract_content(response)
            data = self._parse_json(content)
            
            return self._build_assessment_quality(data, guideline_chunks)
            
        except Exception as e:
            print(f"Assessment scoring failed: {e}")
            return AssessmentQuality(
                dx_coherence_score=0.5,
                differential_breadth_score=0.5,
                verdict="borderline"
            )
    
    def score_plan(
        self,
        extracted: ExtractorJSON,
        diagnoses: List[str],
        plan_items: List[str],
        guideline_chunks: Optional[List[EvidenceChunk]] = None
    ) -> PlanQualitySafety:
        """
        Score the quality and safety of treatment plan.
        
        Args:
            extracted: Patient facts from extraction
            diagnoses: List of diagnoses
            plan_items: List of plan items from the note
            guideline_chunks: Pre-retrieved guidelines
        
        Returns:
            PlanQualitySafety with scores and checks
        """
        # Retrieve guidelines if not provided
        if guideline_chunks is None:
            guideline_chunks = retrieve_guidelines_for_critique(
                extracted, k=5, embedder=self.embedder
            )
        
        # Format inputs
        patient_facts = format_patient_facts(extracted)
        guidelines = format_guidelines_for_prompt(guideline_chunks)
        diagnoses_str = "\n".join(f"- {dx}" for dx in diagnoses) if diagnoses else "None"
        plan_str = "\n".join(f"- {item}" for item in plan_items) if plan_items else "None documented"
        
        allergies = [f"{a.substance_surface} ({a.reaction or 'unknown reaction'})" 
                     for a in extracted.allergies]
        allergies_str = ", ".join(allergies) if allergies else "NKDA"
        
        current_meds = [f"{m.name_surface} {m.dose or ''} {m.freq or ''}" 
                        for m in extracted.meds if m.assertion == "present"]
        meds_str = ", ".join(current_meds) if current_meds else "None"
        
        prompt = PLAN_SCORING_PROMPT.format(
            patient_facts=patient_facts,
            diagnoses=diagnoses_str,
            plan_items=plan_str,
            allergies=allergies_str,
            current_meds=meds_str,
            guidelines=guidelines
        )
        
        # Call LLM
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            
            content = self._extract_content(response)
            data = self._parse_json(content)
            
            return self._build_plan_quality(data, guideline_chunks)
            
        except Exception as e:
            print(f"Plan scoring failed: {e}")
            return PlanQualitySafety(
                guideline_adherence_score=0.5,
                safety_score=0.5,
                verdict="needs_review"
            )
    
    def _extract_content(self, response) -> str:
        """Extract content from LLM response."""
        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return response.get("content", "")
        elif isinstance(response, list) and response:
            first = response[0]
            return first.get("content", "") if isinstance(first, dict) else str(first)
        return str(response) if response else ""
    
    def _parse_json(self, content: str) -> dict:
        """Parse JSON from LLM response."""
        content = content.strip()
        
        # Handle code fences
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                if "{" in part:
                    content = part
                    break
        
        # Remove json tag
        if content.startswith("json"):
            content = content[4:].strip()
        
        # Find JSON object
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            content = content[start:end]
        
        return json.loads(content)
    
    def _build_assessment_quality(
        self,
        data: dict,
        chunks: List[EvidenceChunk]
    ) -> AssessmentQuality:
        """Build AssessmentQuality from LLM response."""
        # Create evidence_id to chunk mapping
        chunk_map = {c.evidence_id: c for c in chunks}
        
        result = AssessmentQuality(
            dx_coherence_score=data.get("dx_coherence_score", 0.5),
            differential_breadth_score=data.get("differential_breadth_score", 0.5),
            verdict=data.get("verdict", "borderline")
        )
        
        # Parse diagnosis evaluations
        for dx_eval in data.get("diagnosis_evaluations", []):
            guideline_support = []
            for eid in dx_eval.get("guideline_support", []):
                if eid in chunk_map:
                    chunk = chunk_map[eid]
                    guideline_support.append(GuidelineCitation(
                        evidence_id=eid,
                        guideline_name=chunk.title,
                        section=chunk.heading_path,
                        snippet=chunk.text[:200],
                        relevance_score=chunk.score
                    ))
            
            result.diagnosis_evaluations.append(DiagnosisEvaluation(
                diagnosis=dx_eval.get("diagnosis", ""),
                coherence_with_evidence=dx_eval.get("coherence_with_evidence", 0.5),
                guideline_support=guideline_support,
                concerns=dx_eval.get("concerns", [])
            ))
        
        # Parse missed differentials
        for missed in data.get("missed_differentials", []):
            guideline_evidence = []
            for eid in missed.get("guideline_evidence", []):
                if eid in chunk_map:
                    chunk = chunk_map[eid]
                    guideline_evidence.append(GuidelineCitation(
                        evidence_id=eid,
                        guideline_name=chunk.title,
                        section=chunk.heading_path,
                        snippet=chunk.text[:200],
                        relevance_score=chunk.score
                    ))
            
            result.missed_differentials.append(MissedDifferential(
                condition=missed.get("condition", ""),
                reasoning=missed.get("reasoning", ""),
                is_red_flag=missed.get("is_red_flag", False),
                guideline_evidence=guideline_evidence
            ))
            
            if missed.get("is_red_flag"):
                result.red_flag_misses += 1
        
        return result
    
    def _build_plan_quality(
        self,
        data: dict,
        chunks: List[EvidenceChunk]
    ) -> PlanQualitySafety:
        """Build PlanQualitySafety from LLM response."""
        chunk_map = {c.evidence_id: c for c in chunks}
        
        result = PlanQualitySafety(
            guideline_adherence_score=data.get("guideline_adherence_score", 0.5),
            safety_score=data.get("safety_score", 0.5),
            verdict=data.get("verdict", "needs_review"),
            missing_return_precautions=data.get("missing_return_precautions", False),
            missing_follow_up=data.get("missing_follow_up", False)
        )
        
        # Parse guideline checks
        VALID_ITEM_TYPES = {"diagnostic_test", "medication", "referral", "lifestyle", "follow_up", "other"}
        VALID_ADHERENCE = {"aligned", "deviation", "uncertain"}
        
        for check in data.get("guideline_checks", []):
            citations = []
            for eid in check.get("guideline_citations", []):
                if eid in chunk_map:
                    chunk = chunk_map[eid]
                    citations.append(GuidelineCitation(
                        evidence_id=eid,
                        guideline_name=chunk.title,
                        section=chunk.heading_path,
                        snippet=chunk.text[:200],
                        relevance_score=chunk.score
                    ))
            
            # Normalize item_type and guideline_adherence
            raw_item_type = check.get("item_type", "other")
            item_type = raw_item_type if raw_item_type in VALID_ITEM_TYPES else "other"
            
            raw_adherence = check.get("guideline_adherence", "uncertain")
            adherence = raw_adherence if raw_adherence in VALID_ADHERENCE else "uncertain"
            
            result.guideline_checks.append(GuidelineCheck(
                plan_item=check.get("plan_item", ""),
                item_type=item_type,
                guideline_adherence=adherence,
                guideline_citations=citations,
                deviation_explanation=check.get("deviation_explanation")
            ))
            
            if adherence == "deviation":
                result.guideline_deviations += 1
        
        # Parse safety checks
        VALID_CHECK_TYPES = {"drug_allergy", "drug_interaction", "dosing_range", "contraindication", "missing_follow_up", "other"}
        VALID_STATUSES = {"pass", "warning", "fail"}
        VALID_SEVERITIES = {"critical", "high", "moderate", "low"}
        
        for check in data.get("safety_checks", []):
            # Normalize check_type - map invalid values to "other"
            raw_check_type = check.get("check_type", "other")
            check_type = raw_check_type if raw_check_type in VALID_CHECK_TYPES else "other"
            
            # Normalize status
            raw_status = check.get("status", "pass")
            status = raw_status if raw_status in VALID_STATUSES else "pass"
            
            # Normalize severity
            raw_severity = check.get("severity", "moderate")
            severity = raw_severity if raw_severity in VALID_SEVERITIES else "moderate"
            
            result.safety_checks.append(SafetyCheck(
                check_type=check_type,
                status=status,
                description=check.get("description", ""),
                related_items=check.get("related_items", []),
                severity=severity
            ))
            
            if check.get("status") == "fail":
                result.safety_failures += 1
                if check.get("severity") == "critical":
                    result.critical_safety_issues += 1
        
        return result


# ============================================================
# Convenience Functions
# ============================================================

def score_note_against_guidelines(
    extracted: ExtractorJSON,
    diagnoses: List[str],
    plan_items: List[str],
    llm_client,
    llm_model: str,
    embedder=None
) -> Tuple[AssessmentQuality, PlanQualitySafety, List[EvidenceChunk]]:
    """
    Score a note's assessment and plan against guidelines.
    
    Args:
        extracted: Patient facts from extraction
        diagnoses: Diagnoses from the note
        plan_items: Plan items from the note
        llm_client: LLM client
        llm_model: LLM model name
        embedder: Optional shared embedder
    
    Returns:
        (assessment_quality, plan_quality, guideline_chunks) tuple
    """
    # Retrieve guidelines once
    guideline_chunks = retrieve_guidelines_for_critique(
        extracted, k=5, embedder=embedder
    )
    
    scorer = GuidelineScorer(llm_client, llm_model, embedder)
    
    assessment = scorer.score_assessment(extracted, diagnoses, guideline_chunks)
    plan = scorer.score_plan(extracted, diagnoses, plan_items, guideline_chunks)
    
    return assessment, plan, guideline_chunks


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    print("Guideline scorer module loaded.")
    print("Use score_note_against_guidelines() to score notes.")
