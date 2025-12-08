# diagnoser/critic_schema.py
# ==============================
# Pydantic models for the Retrieval-Augmented Critic
# Structured output for rubric-based note critique
# ==============================

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


# ============================================================
# Verdict Types
# ============================================================

VerdictType = Literal["entailed", "contradicted", "unsupported"]
QualityVerdict = Literal["good", "borderline", "poor"]
SafetyVerdict = Literal["safe", "needs_review", "unsafe"]
SOAPSection = Literal["S", "O", "A", "P", "unknown"]


# ============================================================
# Evidence Citation
# ============================================================

class EvidenceCitation(BaseModel):
    """A citation linking a claim to supporting evidence."""
    evidence_id: str = Field(..., description="ID like 'dialogue:3' or 'guidelines:42'")
    source: Literal["dialogue", "guidelines", "merck", "note"] = Field(..., description="Evidence source type")
    snippet: Optional[str] = Field(default=None, description="Short quoted text from the evidence")
    score: Optional[float] = Field(default=None, description="Relevance or similarity score")


class GuidelineCitation(BaseModel):
    """Citation to a clinical guideline chunk."""
    evidence_id: str = Field(..., description="Guidelines evidence ID (e.g., 'guidelines:123')")
    guideline_name: str = Field(default="", description="Name of the guideline (e.g., 'ACC/AHA Chest Pain')")
    section: str = Field(default="", description="Section or heading path")
    snippet: str = Field(default="", description="Relevant text snippet from guideline")
    relevance_score: float = Field(default=0.0, description="Semantic similarity score")


# ============================================================
# Source-Note Consistency
# ============================================================

class ClaimJudgment(BaseModel):
    """Judgment for a single claim in the note."""
    claim_id: str = Field(..., description="Unique identifier for this claim")
    claim_text: str = Field(..., description="The claim being evaluated")
    section: SOAPSection = Field(default="unknown", description="SOAP section this claim is from")
    verdict: VerdictType = Field(..., description="Whether claim is entailed, contradicted, or unsupported by dialogue")
    dialogue_sentence_ids: List[int] = Field(default_factory=list, description="Dialogue turn/sentence IDs supporting this verdict")
    dialogue_evidence: List[EvidenceCitation] = Field(default_factory=list, description="Evidence citations from dialogue")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this verdict")


class SourceNoteConsistency(BaseModel):
    """
    Rubric: Source-Note Consistency
    Evaluates how well the note content is supported by the dialogue.
    """
    support_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Fraction of claims supported by dialogue")
    contradiction_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims contradicted by dialogue")
    hallucination_density: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of claims with no supporting evidence")
    verdict: QualityVerdict = Field(default="borderline", description="Overall consistency verdict")
    claim_judgments: List[ClaimJudgment] = Field(default_factory=list, description="Per-claim verdicts with evidence")
    total_claims_evaluated: int = Field(default=0, description="Number of claims evaluated")


# ============================================================
# Coverage of Salient Findings
# ============================================================

class OmittedFinding(BaseModel):
    """A salient finding from the dialogue that was omitted from the note."""
    finding_type: Literal["symptom", "medication", "allergy", "vital", "risk_factor", "red_flag", "ros_positive", "ros_negative"] = Field(
        ..., description="Type of the omitted finding"
    )
    finding_text: str = Field(..., description="Description of the omitted finding")
    dialogue_sentence_ids: List[int] = Field(default_factory=list, description="Where this was mentioned in dialogue")
    importance: Literal["critical", "high", "moderate", "low"] = Field(default="moderate", description="Clinical importance of this omission")


class CoverageSalientFindings(BaseModel):
    """
    Rubric: Coverage of Salient Findings
    Evaluates whether the note captures important findings from the dialogue.
    """
    completeness_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Recall over key entity classes")
    omission_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of salient findings omitted")
    verdict: QualityVerdict = Field(default="borderline", description="Overall coverage verdict")
    
    # Per-class recall
    symptom_recall: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of dialogue symptoms captured")
    medication_recall: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of dialogue meds captured")
    allergy_recall: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of dialogue allergies captured")
    risk_factor_recall: float = Field(default=0.0, ge=0.0, le=1.0, description="Fraction of dialogue risk factors captured")
    
    # Omitted items
    omissions: List[OmittedFinding] = Field(default_factory=list, description="List of important omitted findings")
    critical_omissions: int = Field(default=0, description="Count of critical omissions")


# ============================================================
# Assessment Quality
# ============================================================

class DiagnosisEvaluation(BaseModel):
    """Evaluation of a single diagnosis in the assessment."""
    diagnosis: str = Field(..., description="The diagnosis being evaluated")
    coherence_with_evidence: float = Field(default=0.5, ge=0.0, le=1.0, description="How well diagnosis matches symptom profile")
    evidence_support: List[EvidenceCitation] = Field(default_factory=list, description="Supporting evidence from dialogue/exam")
    guideline_support: List[GuidelineCitation] = Field(default_factory=list, description="Guideline citations supporting this diagnosis")
    concerns: List[str] = Field(default_factory=list, description="Concerns about this diagnosis")


class MissedDifferential(BaseModel):
    """A plausible differential that should have been considered."""
    condition: str = Field(..., description="The missed diagnosis")
    reasoning: str = Field(..., description="Why this should have been considered")
    is_red_flag: bool = Field(default=False, description="Whether this is a dangerous miss")
    guideline_evidence: List[GuidelineCitation] = Field(default_factory=list, description="Guideline support")


class AssessmentQuality(BaseModel):
    """
    Rubric: Assessment Quality
    Evaluates the diagnoses and differential reasoning.
    """
    dx_coherence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How well diagnoses match evidence profile")
    differential_breadth_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Whether plausible ddx are considered")
    verdict: QualityVerdict = Field(default="borderline", description="Overall assessment quality verdict")
    
    # Individual diagnosis evaluations
    diagnosis_evaluations: List[DiagnosisEvaluation] = Field(default_factory=list, description="Per-diagnosis evaluation")
    
    # Missed differentials
    missed_differentials: List[MissedDifferential] = Field(default_factory=list, description="Plausible diagnoses not considered")
    red_flag_misses: int = Field(default=0, description="Count of dangerous missed differentials")


# ============================================================
# Plan Quality and Safety
# ============================================================

class GuidelineCheck(BaseModel):
    """Result of checking plan item against guidelines."""
    plan_item: str = Field(..., description="The plan item being checked")
    item_type: Literal["diagnostic_test", "medication", "referral", "lifestyle", "follow_up", "other"] = Field(
        default="other", description="Type of plan item"
    )
    guideline_adherence: Literal["aligned", "deviation", "uncertain"] = Field(
        default="uncertain", description="Whether item aligns with guidelines"
    )
    guideline_citations: List[GuidelineCitation] = Field(default_factory=list, description="Relevant guideline citations")
    deviation_explanation: Optional[str] = Field(default=None, description="If deviation, explanation of the discrepancy")


class SafetyCheck(BaseModel):
    """Result of a safety check on the plan."""
    check_type: Literal["drug_allergy", "drug_interaction", "dosing_range", "contraindication", "missing_follow_up", "other"] = Field(
        ..., description="Type of safety check"
    )
    status: Literal["pass", "warning", "fail"] = Field(default="pass", description="Safety check result")
    description: str = Field(..., description="Description of the safety concern")
    related_items: List[str] = Field(default_factory=list, description="Related medications/allergies/conditions")
    severity: Literal["critical", "high", "moderate", "low"] = Field(default="moderate", description="Severity if failed")


class PlanQualitySafety(BaseModel):
    """
    Rubric: Plan Quality and Safety
    Evaluates the treatment plan for guideline adherence and safety.
    """
    guideline_adherence_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How well plan aligns with guidelines")
    safety_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Safety rating (1=no issues, 0=critical issues)")
    verdict: SafetyVerdict = Field(default="needs_review", description="Overall safety verdict")
    
    # Guideline checks
    guideline_checks: List[GuidelineCheck] = Field(default_factory=list, description="Per-item guideline checks")
    guideline_deviations: int = Field(default=0, description="Count of guideline deviations")
    
    # Safety checks
    safety_checks: List[SafetyCheck] = Field(default_factory=list, description="Safety check results")
    safety_failures: int = Field(default=0, description="Count of failed safety checks")
    critical_safety_issues: int = Field(default=0, description="Count of critical safety issues")
    
    # Missing items
    missing_return_precautions: bool = Field(default=False, description="Whether return precautions are missing")
    missing_follow_up: bool = Field(default=False, description="Whether follow-up plan is missing")


# ============================================================
# Complete Note Critique
# ============================================================

class NoteCritique(BaseModel):
    """
    Complete critic output for a clinical note.
    
    Structured JSON output containing:
    - Source-note consistency evaluation
    - Coverage of salient findings
    - Assessment quality scoring
    - Plan quality and safety checks
    
    All sections contain verdicts with cited evidence, no chain-of-thought.
    """
    encounter_id: str = Field(default="", description="Identifier for the encounter being critiqued")
    note_type: Literal["prose", "soap"] = Field(default="prose", description="Type of note critiqued")
    
    # Main rubric sections
    source_note_consistency: SourceNoteConsistency = Field(
        default_factory=SourceNoteConsistency,
        description="Source-Note Consistency evaluation"
    )
    coverage_of_salient_findings: CoverageSalientFindings = Field(
        default_factory=CoverageSalientFindings,
        description="Coverage of Salient Findings evaluation"
    )
    assessment_quality: AssessmentQuality = Field(
        default_factory=AssessmentQuality,
        description="Assessment Quality evaluation"
    )
    plan_quality_and_safety: PlanQualitySafety = Field(
        default_factory=PlanQualitySafety,
        description="Plan Quality and Safety evaluation"
    )
    
    # Overall summary
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0, description="Composite score across all rubrics")
    overall_verdict: QualityVerdict = Field(default="borderline", description="Overall note quality verdict")
    overall_safety: SafetyVerdict = Field(default="needs_review", description="Overall safety determination")
    
    # Metadata
    num_claims_evaluated: int = Field(default=0, description="Total claims evaluated")
    num_guideline_checks: int = Field(default=0, description="Total guideline checks performed")
    num_safety_checks: int = Field(default=0, description="Total safety checks performed")
    
    def compute_overall_score(self) -> float:
        """Compute weighted overall score from component scores."""
        weights = {
            "source_note_consistency": 0.25,
            "coverage": 0.25,
            "assessment": 0.25,
            "plan_safety": 0.25
        }
        
        score = (
            weights["source_note_consistency"] * self.source_note_consistency.support_score +
            weights["coverage"] * self.coverage_of_salient_findings.completeness_score +
            weights["assessment"] * self.assessment_quality.dx_coherence_score +
            weights["plan_safety"] * self.plan_quality_and_safety.safety_score
        )
        
        return score
    
    def determine_overall_verdict(self) -> QualityVerdict:
        """Determine overall verdict from component verdicts."""
        verdicts = [
            self.source_note_consistency.verdict,
            self.coverage_of_salient_findings.verdict,
            self.assessment_quality.verdict,
        ]
        
        if all(v == "good" for v in verdicts):
            return "good"
        elif any(v == "poor" for v in verdicts):
            return "poor"
        return "borderline"
    
    def determine_overall_safety(self) -> SafetyVerdict:
        """Determine overall safety from plan evaluation."""
        if self.plan_quality_and_safety.critical_safety_issues > 0:
            return "unsafe"
        elif (self.plan_quality_and_safety.safety_failures > 0 or
              self.assessment_quality.red_flag_misses > 0):
            return "needs_review"
        elif self.plan_quality_and_safety.verdict == "safe":
            return "safe"
        return "needs_review"
    
    def finalize(self) -> 'NoteCritique':
        """Compute derived fields and return self."""
        self.overall_score = self.compute_overall_score()
        self.overall_verdict = self.determine_overall_verdict()
        self.overall_safety = self.determine_overall_safety()
        
        self.num_claims_evaluated = self.source_note_consistency.total_claims_evaluated
        self.num_guideline_checks = len(self.plan_quality_and_safety.guideline_checks)
        self.num_safety_checks = len(self.plan_quality_and_safety.safety_checks)
        
        return self


# ============================================================
# Helper Functions
# ============================================================

def create_empty_critique(encounter_id: str = "", note_type: str = "prose") -> NoteCritique:
    """Create an empty NoteCritique with default values."""
    return NoteCritique(
        encounter_id=encounter_id,
        note_type=note_type
    )


def verdict_to_score(verdict: QualityVerdict) -> float:
    """Convert quality verdict to numeric score."""
    mapping = {"good": 1.0, "borderline": 0.5, "poor": 0.0}
    return mapping.get(verdict, 0.5)


def safety_to_score(verdict: SafetyVerdict) -> float:
    """Convert safety verdict to numeric score."""
    mapping = {"safe": 1.0, "needs_review": 0.5, "unsafe": 0.0}
    return mapping.get(verdict, 0.5)


def score_to_verdict(score: float) -> QualityVerdict:
    """Convert numeric score to quality verdict."""
    if score >= 0.7:
        return "good"
    elif score >= 0.4:
        return "borderline"
    return "poor"


def score_to_safety(score: float) -> SafetyVerdict:
    """Convert numeric score to safety verdict."""
    if score >= 0.8:
        return "safe"
    elif score >= 0.4:
        return "needs_review"
    return "unsafe"
