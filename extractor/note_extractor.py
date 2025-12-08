# extractor/note_extractor.py
# ==============================
# LLM-based Clinical Note Extractor
# Extracts structured information from prose and SOAP notes
# ==============================

import json
from typing import List, Dict, Optional, Any, Literal

from .schema import (
    ExtractorJSON, Symptom, Medication, Allergy, Vital,
    QAExtracted, Evidence
)


# ============================================================
# System Prompts for Note Extraction
# ============================================================

PROSE_NOTE_SYSTEM_PROMPT = """You are an expert clinical documentation specialist. Your task is to extract structured medical information from a clinical note.

Analyze the note and extract all clinical information, just as a physician would when reviewing documentation.

OUTPUT FORMAT:
Return a single valid JSON object with the following structure. Use null for any field not mentioned in the note. Do not make up information.

{
  "chief_complaint": "The primary reason for the visit (1-2 sentences)",
  
  "symptoms": [
    {
      "name": "standardized symptom name (e.g., 'chest pain', 'shortness of breath')",
      "assertion": "present|absent|possible",
      "duration": "how long (e.g., '2 weeks', '3 days')"
    }
  ],
  
  "medications": [
    {
      "name": "medication name (generic preferred)",
      "dose": "dose amount (e.g., '10 mg')",
      "frequency": "how often (e.g., 'once daily', 'twice daily')",
      "assertion": "present (currently taking)|absent (stopped)|possible (considering)"
    }
  ],
  
  "allergies": [
    {
      "substance": "allergen name",
      "reaction": "type of reaction"
    }
  ],
  
  "vitals": {
    "blood_pressure": "if mentioned",
    "heart_rate": "if mentioned",
    "temperature": "if mentioned",
    "respiratory_rate": "if mentioned",
    "oxygen_saturation": "if mentioned"
  },
  
  "risk_factors": ["list of risk factors like smoking, diabetes, hypertension, family history"],
  
  "diagnoses": ["list of diagnoses or assessments mentioned"],
  
  "plan_items": ["list of plan items, tests ordered, follow-up instructions"]
}

IMPORTANT:
1. Extract ONLY information explicitly stated in the note
2. Distinguish between present symptoms and denied/absent symptoms
3. Include pertinent negatives when documented
4. Use standardized medical terminology
5. Return ONLY the JSON object, no additional text"""


SOAP_NOTE_SYSTEM_PROMPT = """You are an expert clinical documentation specialist. Your task is to extract structured medical information from a SOAP-formatted clinical note.

SOAP notes have sections:
- S (Subjective): Patient's reported symptoms and history
- O (Objective): Exam findings, vitals, lab results
- A (Assessment): Diagnoses and clinical reasoning
- P (Plan): Treatment plan, medications, follow-up

OUTPUT FORMAT:
Return a single valid JSON object. Use null for fields not documented. Do not make up information.

{
  "chief_complaint": "Primary reason for visit from Subjective section",
  
  "symptoms": [
    {
      "name": "symptom name (from S section primarily)",
      "assertion": "present|absent|possible",
      "duration": "duration if mentioned",
      "section": "S|O"
    }
  ],
  
  "medications": [
    {
      "name": "medication name (generic)",
      "dose": "dose amount",
      "frequency": "dosing frequency",
      "assertion": "present|absent|possible",
      "section": "S|P"
    }
  ],
  
  "allergies": [
    {
      "substance": "allergen",
      "reaction": "reaction type"
    }
  ],
  
  "vitals": {
    "blood_pressure": "from O section",
    "heart_rate": "from O section",
    "temperature": "from O section",
    "respiratory_rate": "from O section",
    "oxygen_saturation": "from O section"
  },
  
  "exam_findings": [
    {
      "system": "body system examined",
      "finding": "what was found",
      "normal": true|false
    }
  ],
  
  "risk_factors": ["list of identified risk factors"],
  
  "diagnoses": [
    {
      "condition": "diagnosis from A section",
      "status": "primary|secondary|differential|ruled_out"
    }
  ],
  
  "plan_items": [
    {
      "category": "medication|test|referral|lifestyle|follow_up|other",
      "description": "what is being done/ordered"
    }
  ]
}

IMPORTANT:
1. Map information to the correct SOAP section
2. Diagnoses come from Assessment (A) section
3. New medications come from Plan (P) section; current meds from Subjective (S)
4. Vitals and exam findings from Objective (O) section
5. Return ONLY the JSON object"""


# ============================================================
# JSON Parsing Helpers
# ============================================================

def _extract_json_block(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from text."""
    if not text:
        return None
    
    t = text.strip()
    
    # Handle code fences
    if t.startswith("```"):
        parts = t.split("```")
        for part in parts:
            if "{" in part:
                t = part
                break
    
    # Remove json language tag
    if t.strip().startswith("json"):
        t = t.strip()[4:].strip()
    
    # Find balanced braces
    start = None
    depth = 0
    for i, ch in enumerate(t):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return t[start:i+1]
    
    return None


def _normalize_assertion(value: str) -> str:
    """Normalize assertion to valid schema value."""
    if not value or not isinstance(value, str):
        return "present"
    
    value_lower = value.lower().strip()
    
    if value_lower in ("present", "absent", "possible"):
        return value_lower
    
    present_terms = {"yes", "true", "current", "active", "taking", "has", "confirmed", "positive"}
    absent_terms = {"no", "false", "none", "not", "negative", "denied", "denies", "stopped"}
    possible_terms = {"maybe", "uncertain", "unclear", "possible", "suspected", "likely"}
    
    if value_lower in present_terms:
        return "present"
    if value_lower in absent_terms:
        return "absent"
    if value_lower in possible_terms:
        return "possible"
    
    return "present"


# ============================================================
# Note Extractor
# ============================================================

class NoteExtractor:
    """
    LLM-based extractor for clinical notes (prose and SOAP format).
    
    Produces ExtractorJSON compatible with the rest of the pipeline.
    """
    
    def __init__(self, client, model: str):
        """
        Initialize the note extractor.
        
        Args:
            client: LLM client with .chat() method
            model: Model name for extraction
        """
        self.client = client
        self.model = model
    
    def extract(
        self,
        note_text: str,
        note_type: Literal["prose", "soap"] = "prose",
        max_retries: int = 2
    ) -> ExtractorJSON:
        """
        Extract clinical information from a note.
        
        Args:
            note_text: Raw note text
            note_type: "prose" or "soap"
            max_retries: Number of retry attempts on parse failure
        
        Returns:
            ExtractorJSON with extracted information
        """
        # Select appropriate prompt
        system_prompt = SOAP_NOTE_SYSTEM_PROMPT if note_type == "soap" else PROSE_NOTE_SYSTEM_PROMPT
        
        user_prompt = f"""CLINICAL NOTE:

{note_text}

Extract all clinical information from this note and return as JSON:"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Try extraction with retries
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )
                
                # Handle various response formats
                if isinstance(response, str):
                    content = response
                elif isinstance(response, dict):
                    content = response.get("content", "")
                elif isinstance(response, list) and response:
                    first = response[0]
                    content = first.get("content", "") if isinstance(first, dict) else str(first)
                else:
                    content = str(response) if response else ""
                
                # Extract JSON
                json_str = _extract_json_block(content)
                if not json_str:
                    json_str = content.strip()
                
                # Parse JSON
                data = json.loads(json_str)
                
                # Convert to ExtractorJSON
                return self._parse_note_output(data, note_type)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt + "\n\nReturn ONLY valid JSON, no other text."}
                    ]
                else:
                    print(f"Note extraction JSON parse failed: {e}")
                    return ExtractorJSON()
            except Exception as e:
                if attempt >= max_retries:
                    print(f"Note extraction failed: {e}")
                    return ExtractorJSON()
        
        return ExtractorJSON()
    
    def _parse_note_output(self, data: dict, note_type: str) -> ExtractorJSON:
        """Convert LLM JSON output to ExtractorJSON schema."""
        ej = ExtractorJSON()
        
        # Chief Complaint
        cc = data.get("chief_complaint")
        if cc and isinstance(cc, str):
            ej.chief_complaint = cc[:200]
        
        # Symptoms
        symptoms_data = data.get("symptoms", [])
        if isinstance(symptoms_data, list):
            for s in symptoms_data:
                if not isinstance(s, dict):
                    continue
                name = s.get("name")
                if not name:
                    continue
                
                ej.symptoms.append(Symptom(
                    name_surface=name,
                    name_norm=name.lower().replace(" ", "_"),
                    assertion=_normalize_assertion(s.get("assertion", "present")),
                    duration=s.get("duration"),
                    evidence=Evidence(utt_ids=[], spans=[{"text": name}])
                ))
        
        # Medications
        meds_data = data.get("medications", [])
        if isinstance(meds_data, list):
            for m in meds_data:
                if not isinstance(m, dict):
                    continue
                name = m.get("name")
                if not name:
                    continue
                
                ej.meds.append(Medication(
                    name_surface=name,
                    name_norm=name.lower(),
                    dose=m.get("dose"),
                    freq=m.get("frequency"),
                    assertion=_normalize_assertion(m.get("assertion", "present")),
                    evidence=Evidence(utt_ids=[], spans=[{"text": name}])
                ))
        
        # Allergies
        allergies_data = data.get("allergies", [])
        if isinstance(allergies_data, list):
            for a in allergies_data:
                if not isinstance(a, dict):
                    continue
                substance = a.get("substance")
                if not substance:
                    continue
                
                ej.allergies.append(Allergy(
                    substance_surface=substance,
                    substance_norm=substance.lower(),
                    reaction=a.get("reaction"),
                    assertion="present",
                    evidence=Evidence(utt_ids=[], spans=[])
                ))
        
        # Vitals
        vitals_data = data.get("vitals", {})
        if isinstance(vitals_data, dict):
            vital_mapping = {
                "blood_pressure": "bp",
                "heart_rate": "hr",
                "temperature": "temp",
                "respiratory_rate": "rr",
                "oxygen_saturation": "spo2"
            }
            for key, kind in vital_mapping.items():
                value = vitals_data.get(key)
                if value and isinstance(value, str):
                    ej.vitals.append(Vital(
                        kind=kind,
                        value=value,
                        evidence=Evidence(utt_ids=[], spans=[])
                    ))
        
        # Risk Factors
        risk_data = data.get("risk_factors", [])
        if isinstance(risk_data, list):
            ej.risk_factors = [r for r in risk_data if isinstance(r, str)]
        
        # Diagnoses -> QA extractions
        diagnoses_data = data.get("diagnoses", [])
        if isinstance(diagnoses_data, list):
            for dx in diagnoses_data:
                if isinstance(dx, str) and dx:
                    ej.qa_extractions.append(QAExtracted(
                        concept=f"diagnosis_{dx.lower().replace(' ', '_')[:30]}",
                        value=dx,
                        assertion="present",
                        evidence=Evidence(utt_ids=[], spans=[])
                    ))
                elif isinstance(dx, dict):
                    condition = dx.get("condition", "")
                    if condition:
                        status = dx.get("status", "primary")
                        assertion = "absent" if status == "ruled_out" else "present"
                        ej.qa_extractions.append(QAExtracted(
                            concept=f"diagnosis_{condition.lower().replace(' ', '_')[:30]}",
                            value=f"{condition} ({status})",
                            assertion=assertion,
                            evidence=Evidence(utt_ids=[], spans=[])
                        ))
        
        # Plan items -> QA extractions
        plan_data = data.get("plan_items", [])
        if isinstance(plan_data, list):
            for i, item in enumerate(plan_data):
                if isinstance(item, str) and item:
                    ej.qa_extractions.append(QAExtracted(
                        concept=f"plan_item_{i}",
                        value=item,
                        assertion="present",
                        evidence=Evidence(utt_ids=[], spans=[])
                    ))
                elif isinstance(item, dict):
                    category = item.get("category", "other")
                    desc = item.get("description", "")
                    if desc:
                        ej.qa_extractions.append(QAExtracted(
                            concept=f"plan_{category}_{i}",
                            value=desc,
                            assertion="present",
                            evidence=Evidence(utt_ids=[], spans=[])
                        ))
        
        # Exam findings -> QA extractions (for SOAP notes)
        exam_data = data.get("exam_findings", [])
        if isinstance(exam_data, list):
            for finding in exam_data:
                if isinstance(finding, dict):
                    system = finding.get("system", "general")
                    finding_text = finding.get("finding", "")
                    if finding_text:
                        normal = finding.get("normal", True)
                        ej.qa_extractions.append(QAExtracted(
                            concept=f"exam_{system.lower().replace(' ', '_')}",
                            value=finding_text,
                            assertion="present" if normal else "present",  # Both present, value describes finding
                            evidence=Evidence(utt_ids=[], spans=[])
                        ))
        
        return ej


def create_note_extractor(client, model: str) -> NoteExtractor:
    """Factory function to create a note extractor."""
    return NoteExtractor(client, model)
