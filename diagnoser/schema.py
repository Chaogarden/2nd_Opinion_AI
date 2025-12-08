# diagnoser/schema.py
# ==============================
# Pydantic models for the two-doctor diagnostic pipeline:
# - DiagnoserInput / DiagnoserOutput
# - ConsultantCritique
# - ArbiterResult
# - Supporting data structures
# ==============================

from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator

# Import ExtractorJSON from the extractor module
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extractor.schema import ExtractorJSON


# ============================================================
# Evidence Chunk (from RAG)
# ============================================================

class EvidenceChunk(BaseModel):
    """A single chunk of evidence retrieved from RAG (guidelines or Merck)."""
    evidence_id: str = Field(..., description="Unique ID like 'guidelines:123' or 'merck:456'")
    source: Literal["guidelines", "merck"] = Field(..., description="Which corpus this came from")
    score: float = Field(..., description="Similarity score from vector search")
    title: str = Field(default="", description="Document title")
    heading_path: str = Field(default="", description="Section/heading path within document")
    text: str = Field(..., description="The actual text content of the chunk")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# ============================================================
# Diagnoser Input
# ============================================================

class DiagnoserInput(BaseModel):
    """Input bundle for the Diagnoser LLM."""
    patient_facts: ExtractorJSON = Field(..., description="Normalized clinical findings from extraction")
    dialogue_brief: str = Field(default="", description="Short summary of the doctor-patient dialogue")
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list, description="Top-k evidence from RAG")


# ============================================================
# Diagnoser Output Components
# ============================================================

class DifferentialDx(BaseModel):
    """A single differential diagnosis with supporting evidence."""
    condition: str = Field(..., description="Name of the condition/diagnosis")
    likelihood: Literal["high", "moderate", "low"] = Field(default="moderate", description="Estimated likelihood")
    rationale: str = Field(default="", description="Clinical reasoning for this diagnosis")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs supporting this diagnosis")


class RedFlag(BaseModel):
    """A clinical red flag requiring urgent attention."""
    description: str = Field(..., description="Description of the red flag")
    risk_level: Literal["critical", "high", "moderate"] = Field(default="high")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs supporting this red flag")


class WorkupItem(BaseModel):
    """A recommended diagnostic test or workup item."""
    test_name: str = Field(..., description="Name of the test/procedure")
    priority: Literal["urgent", "routine", "optional"] = Field(default="routine")
    rationale: str = Field(default="", description="Why this test is recommended")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs supporting this recommendation")


class NonPharmRec(BaseModel):
    """A non-pharmacological recommendation."""
    intervention: str = Field(..., description="Description of the intervention")
    rationale: str = Field(default="", description="Why this is recommended")
    evidence_ids: List[str] = Field(default_factory=list)


class PharmRec(BaseModel):
    """A pharmacological recommendation with dosing details."""
    drug: str = Field(..., description="Drug name (generic preferred)")
    route: str = Field(default="oral", description="Route of administration")
    dose: str = Field(default="", description="Dose amount (e.g., '10 mg')")
    frequency: str = Field(default="", description="Dosing frequency (e.g., 'once daily')")
    duration: str = Field(default="", description="Duration of therapy (e.g., '7 days', 'ongoing')")
    indication: str = Field(default="", description="What this medication is treating")
    evidence_ids: List[str] = Field(default_factory=list)


class ManagementPlan(BaseModel):
    """Combined management plan with non-pharm and pharm recommendations."""
    non_pharm: List[NonPharmRec] = Field(default_factory=list)
    pharm: List[PharmRec] = Field(default_factory=list)


class ContraindicationOrInteraction(BaseModel):
    """A contraindication or drug interaction warning."""
    issue: str = Field(..., description="Description of the contraindication/interaction")
    related_meds: List[str] = Field(default_factory=list, description="Medications involved")
    severity: Literal["critical", "high", "moderate", "low"] = Field(default="moderate")
    evidence_ids: List[str] = Field(default_factory=list)


class CodingAndLabels(BaseModel):
    """Clinical coding suggestions."""
    icd10_codes: List[str] = Field(default_factory=list, description="Suggested ICD-10 codes")
    snomed_codes: List[str] = Field(default_factory=list, description="Suggested SNOMED CT codes")
    labels: List[str] = Field(default_factory=list, description="Free-text clinical labels")


# ============================================================
# Diagnoser Output (Full)
# ============================================================

class DiagnoserOutput(BaseModel):
    """
    Complete output from the Diagnoser LLM.
    Every clinical claim must be citation-anchored via evidence_ids.
    """
    differential: List[DifferentialDx] = Field(default_factory=list, description="Differential diagnoses")
    red_flags: List[RedFlag] = Field(default_factory=list, description="Clinical red flags")
    initial_workup: List[WorkupItem] = Field(default_factory=list, description="Recommended diagnostic workup")
    management_plan: ManagementPlan = Field(default_factory=ManagementPlan, description="Treatment plan")
    contraindications_and_interactions: List[ContraindicationOrInteraction] = Field(
        default_factory=list, description="Drug contraindications and interactions"
    )
    coding_and_labels: CodingAndLabels = Field(default_factory=CodingAndLabels, description="Clinical coding")
    overall_uncertainty: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence calibration (0=certain, 1=very uncertain)")
    groundedness_score: float = Field(default=0.5, ge=0.0, le=1.0, description="How well grounded in evidence (0=poor, 1=excellent)")
    notes_on_missing_info: Optional[str] = Field(default=None, description="What additional info would help")
    
    @field_validator("overall_uncertainty", "groundedness_score", mode="before")
    @classmethod
    def clamp_scores(cls, v):
        """Ensure scores are within [0, 1] range."""
        if v is None:
            return 0.5
        try:
            v = float(v)
            return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            return 0.5


# ============================================================
# Consultant Critique
# ============================================================

CritiqueKind = Literal[
    "missing_differential",
    "unsafe_recommendation", 
    "guideline_conflict",
    "safety_gap",
    "dosing_error",
    "interaction_missed",
    "other"
]

CritiqueSeverity = Literal["critical", "high", "moderate", "low"]


class CritiqueItem(BaseModel):
    """A single critique point from the Consultant."""
    kind: CritiqueKind = Field(..., description="Type of critique")
    description: str = Field(..., description="Detailed description of the issue")
    severity: CritiqueSeverity = Field(default="moderate")
    target_path: str = Field(default="", description="JSON pointer to the problematic element (e.g., '/management_plan/pharm/0')")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence supporting this critique")
    suggested_fix: Optional[str] = Field(default=None, description="How to address this issue")


class ConsultantCritique(BaseModel):
    """Complete critique from the Consultant LLM."""
    issues: List[CritiqueItem] = Field(default_factory=list, description="List of identified issues")
    overall_assessment: str = Field(default="", description="Summary assessment of the plan")
    overall_safety_rating: Literal["safe", "needs_review", "unsafe"] = Field(
        default="needs_review", description="Overall safety rating"
    )


# ============================================================
# Arbiter Result
# ============================================================

class ArbiterResult(BaseModel):
    """Result from the Arbiter after processing critiques."""
    final_plan: Optional[DiagnoserOutput] = Field(default=None, description="Patched plan if not abstaining")
    abstained: bool = Field(default=False, description="True if critical issues prevent safe output")
    patches_applied: List[str] = Field(default_factory=list, description="Human-readable list of changes made")
    notes_on_missing_info: Optional[str] = Field(default=None, description="Explanation if abstaining")
    unresolved_issues: List[str] = Field(default_factory=list, description="Issues that couldn't be auto-resolved")


# ============================================================
# Pipeline Result (combines everything)
# ============================================================

class PipelineResult(BaseModel):
    """Complete result from the clinical reasoning pipeline."""
    extracted_facts: ExtractorJSON = Field(..., description="Extracted medical facts")
    evidence_chunks: List[EvidenceChunk] = Field(default_factory=list, description="Retrieved evidence")
    diagnoser_output: Optional[DiagnoserOutput] = Field(default=None, description="Initial diagnosis plan")
    revised_diagnoser_output: Optional[DiagnoserOutput] = Field(default=None, description="Revised plan after Consultant feedback")
    consultant_critique: Optional[ConsultantCritique] = Field(default=None, description="Consultant's review")
    arbiter_result: Optional[ArbiterResult] = Field(default=None, description="Final arbitrated result")
    dialogue_brief: str = Field(default="", description="Summary of the dialogue")
    error: Optional[str] = Field(default=None, description="Error message if pipeline failed")
    # Natural language explanations (patient-friendly)
    diagnosis_explanation: Optional[Dict[str, Any]] = Field(default=None, description="Patient-friendly diagnosis explanation")
    critique_explanation: Optional[Dict[str, Any]] = Field(default=None, description="Patient-friendly critique of doctor's note")

