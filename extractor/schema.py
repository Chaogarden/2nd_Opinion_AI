# =============================================================
# extractor/schema.py
# =============================================================
from __future__ import annotations
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

Assertion = Literal["present", "absent", "possible"]

class Evidence(BaseModel):
    utt_ids: List[int] = Field(default_factory=list)
    spans: List[Dict[str, Any]] = Field(default_factory=list) # optional: {start,end,text}

class Symptom(BaseModel):
    name_surface: str
    name_norm: Optional[str] = None # e.g., UMLS/SNOMED label
    cui: Optional[str] = None
    assertion: Assertion = "present"
    duration: Optional[str] = None
    severity: Optional[str] = None
    onset: Optional[str] = None
    modifiers: List[str] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)

class Medication(BaseModel):
    name_surface: str
    name_norm: Optional[str] = None # normalized generic
    rxcui: Optional[str] = None
    dose: Optional[str] = None
    strength: Optional[str] = None
    freq: Optional[str] = None
    route: Optional[str] = None
    form: Optional[str] = None
    prn: Optional[bool] = None
    assertion: Assertion = "present" # e.g., "not taking ibuprofen" -> absent
    evidence: Evidence = Field(default_factory=Evidence)

class Allergy(BaseModel):
    substance_surface: str
    substance_norm: Optional[str] = None
    rxcui: Optional[str] = None
    reaction: Optional[str] = None
    assertion: Assertion = "present"
    evidence: Evidence = Field(default_factory=Evidence)

class Vital(BaseModel):
    kind: Literal["temp","hr","bp","rr","spo2"]
    value: str
    evidence: Evidence = Field(default_factory=Evidence)

class QAExtracted(BaseModel):
    concept: str # e.g., "heavy_exercise"
    value: Optional[str] = None # "yes"/"no"/free text
    assertion: Assertion = "present"
    evidence: Evidence = Field(default_factory=Evidence)

class ExtractorJSON(BaseModel):
    chief_complaint: Optional[str] = None
    demographics: Dict[str, Any] = Field(default_factory=dict) # optional future
    symptoms: List[Symptom] = Field(default_factory=list)
    meds: List[Medication] = Field(default_factory=list)
    allergies: List[Allergy] = Field(default_factory=list)
    vitals: List[Vital] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    qa_extractions: List[QAExtracted] = Field(default_factory=list)