# extractor/llm_extractor.py
# ==============================
# LLM-First Full-Dialogue Clinical Extraction
# Processes the entire conversation at once for comprehensive extraction
# ==============================

import json
from typing import List, Dict, Optional, Any

from .schema import (
    ExtractorJSON, Symptom, Medication, Allergy, Vital, 
    QAExtracted, Evidence
)

# ============================================================
# System Prompt for Full-Dialogue Extraction
# ============================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert clinical documentation specialist. Your task is to extract structured medical information from a doctor-patient conversation transcript.

You must analyze the ENTIRE dialogue and extract comprehensive clinical information, just as a physician would when documenting an encounter.

OUTPUT FORMAT:
Return a single valid JSON object with the following structure. Use null for any field not discussed in the conversation. Do not make up information.

{
  "demographics": {
    "age": "patient's age in years as a number string (e.g., '45', '72') - extract if mentioned",
    "sex": "male|female - extract if mentioned or clearly implied",
    "race_ethnicity": "patient's race or ethnicity if mentioned (e.g., 'african_american', 'hispanic', 'caucasian', 'asian')",
    "sexual_orientation": "patient's sexual orientation if mentioned (e.g., 'heterosexual', 'homosexual', 'bisexual')"
  },
  
  "chief_complaint": "The primary reason for the visit, in the patient's own words or summarized. Should be 1-2 sentences capturing why they came in.",
  
  "history_of_present_illness": "A brief narrative summary of the current problem including onset, duration, severity, associated symptoms, and relevant context.",
  
  "symptoms": [
    {
      "name": "standardized symptom name (e.g., 'chest pain', 'shortness of breath', 'fatigue')",
      "assertion": "present|absent|possible",
      "duration": "how long the symptom has been present (e.g., '2 weeks', '3 days')",
      "severity": "mild|moderate|severe (if mentioned)",
      "onset": "when/how it started (e.g., 'gradual', 'sudden', 'after eating')",
      "location": "body location if applicable",
      "character": "quality/description (e.g., 'sharp', 'dull', 'burning', 'constant', 'intermittent')",
      "aggravating_factors": "what makes it worse",
      "relieving_factors": "what makes it better"
    }
  ],
  
  "medications": [
    {
      "name": "LOWERCASE generic drug name only (e.g., 'lisinopril', 'metformin', 'aspirin', 'atorvastatin')",
      "dose": "STRICT FORMAT: number + space + unit (e.g., '10 mg', '500 mg', '81 mg', '2.5 mg', '1 g')",
      "frequency": "USE ONLY THESE VALUES: 'once_daily', 'twice_daily', 'three_times_daily', 'four_times_daily', 'every_4_hours', 'every_6_hours', 'every_8_hours', 'every_12_hours', 'as_needed', 'at_bedtime', 'in_morning', 'once_weekly', 'twice_weekly'",
      "assertion": "USE ONLY: 'present' (currently taking) | 'absent' (stopped/not taking) | 'possible' (considering)"
    }
  ],
  
  "allergies": [
    {
      "substance": "USE STANDARDIZED NAMES - Drug allergies: 'penicillin', 'sulfa', 'sulfamethoxazole', 'aspirin', 'nsaids', 'ibuprofen', 'codeine', 'morphine', 'hydrocodone', 'oxycodone', 'erythromycin', 'azithromycin', 'tetracycline', 'doxycycline', 'cephalosporins', 'cephalexin', 'fluoroquinolones', 'ciprofloxacin', 'levofloxacin', 'amoxicillin', 'ampicillin', 'vancomycin', 'metronidazole', 'trimethoprim', 'nitrofurantoin', 'ace_inhibitors', 'statins', 'contrast_dye', 'iodine', 'latex'. Food allergies: 'shellfish', 'peanuts', 'tree_nuts', 'eggs', 'milk', 'dairy', 'soy', 'wheat', 'gluten', 'fish', 'sesame'. Environmental: 'pollen', 'dust', 'dust_mites', 'mold', 'pet_dander', 'bee_stings', 'insect_stings'. Use lowercase with underscores.",
      "reaction": "Standardized: 'rash', 'hives', 'itching', 'swelling', 'anaphylaxis', 'shortness_of_breath', 'nausea', 'vomiting', 'diarrhea', 'throat_swelling', 'tongue_swelling', 'difficulty_breathing', 'hypotension', 'unknown'",
      "assertion": "USE ONLY: 'present' | 'absent' | 'possible'"
    }
  ],
  
  "past_medical_history": ["list of prior diagnoses, conditions, surgeries, hospitalizations"],
  
  "family_history": ["list of relevant family medical history with relationship if mentioned"],
  
  "social_history": {
    "smoking": {
      "status": "current|former|never",
      "details": "pack-years, amount, quit date if applicable"
    },
    "alcohol": {
      "status": "current|former|never",
      "details": "amount/frequency if mentioned"
    },
    "drugs": "recreational drug use if mentioned",
    "occupation": "job/work if mentioned",
    "living_situation": "if mentioned",
    "exercise": "if mentioned"
  },
  
  "vitals": {
    "blood_pressure": "FORMAT: 'systolic/diastolic' as numbers only (e.g., '120/80', '140/90') - no units",
    "heart_rate": "FORMAT: number only, no units (e.g., '72', '88', '100') - bpm assumed",
    "temperature": "FORMAT: number only in Fahrenheit (e.g., '98.6', '101.2', '99.0') - °F assumed",
    "respiratory_rate": "FORMAT: number only (e.g., '16', '20', '12') - per minute assumed",
    "oxygen_saturation": "FORMAT: number only (e.g., '98', '95', '100') - percentage assumed"
  },
  
  "review_of_systems": {
    "constitutional": ["fever", "chills", "weight_loss", "fatigue", etc. - list what was asked/mentioned],
    "cardiovascular": ["chest_pain", "palpitations", "edema", etc.],
    "respiratory": ["cough", "shortness_of_breath", "wheezing", etc.],
    "gastrointestinal": ["nausea", "vomiting", "diarrhea", "constipation", etc.],
    "genitourinary": ["dysuria", "frequency", "hematuria", etc.],
    "musculoskeletal": ["joint_pain", "back_pain", "muscle_weakness", etc.],
    "neurological": ["headache", "dizziness", "numbness", "weakness", etc.],
    "psychiatric": ["depression", "anxiety", "sleep_problems", etc.],
    "skin": ["rash", "itching", "lesions", etc.]
  },
  
  "risk_factors": ["list of identified risk factors like smoking, obesity, diabetes, hypertension, family history of heart disease, etc."]
}

IMPORTANT GUIDELINES:
1. Extract ONLY information explicitly stated or clearly implied in the dialogue
2. For symptoms, distinguish between what the patient HAS (present) vs DENIES having (absent)
3. Pay attention to negations: "no chest pain" means chest_pain is ABSENT
4. Include pertinent negatives (things the doctor asked about that the patient denied)
5. Use standardized medical terminology where possible
6. If a field has no relevant information, use null or an empty array []
7. Be precise with durations, doses, and frequencies when mentioned
8. Return ONLY the JSON object, no additional text or explanation

STRICT FORMATTING REQUIREMENTS:
- MEDICATIONS: Use ONLY lowercase generic drug names. Dose format: 'number unit' (e.g., '10 mg'). Frequency MUST be one of the allowed values.
- ALLERGIES: Use ONLY the standardized allergen names provided. Lowercase with underscores.
- VITALS: Numbers only, no units (units are assumed based on vital type).
- ASSERTION: ONLY use 'present', 'absent', or 'possible' - no other values."""


def _format_dialogue_for_prompt(dialogue: List[Dict]) -> str:
    """Format dialogue turns into a readable transcript."""
    lines = []
    for turn in dialogue:
        role = turn.get("role", "UNKNOWN").upper()
        text = turn.get("text", "").strip()
        if text:
            lines.append(f"[{role}]: {text}")
    return "\n".join(lines)


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


def _safe_get(d: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
        else:
            return default
        if d is None:
            return default
    return d if d is not None else default


def _normalize_assertion(value: str) -> str:
    """
    Normalize assertion values to valid schema values.
    
    Valid values: 'present', 'absent', 'possible'
    Maps common LLM outputs to valid values.
    """
    if not value or not isinstance(value, str):
        return "present"
    
    value_lower = value.lower().strip()
    
    # Direct matches
    if value_lower in ("present", "absent", "possible"):
        return value_lower
    
    # Map common variations to valid values
    present_terms = {"yes", "true", "current", "active", "taking", "has", "confirmed", "positive", "recommended"}
    absent_terms = {"no", "false", "none", "not", "negative", "denied", "denies", "stopped", "discontinued"}
    possible_terms = {"maybe", "uncertain", "unclear", "possible", "suspected", "likely", "consider", "rule out"}
    
    if value_lower in present_terms:
        return "present"
    if value_lower in absent_terms:
        return "absent"
    if value_lower in possible_terms:
        return "possible"
    
    # Default to present for anything else
    return "present"


class LLMExtractor:
    """
    LLM-based full-dialogue clinical extractor.
    
    Processes the entire conversation at once for comprehensive,
    context-aware extraction of clinical information.
    """
    
    def __init__(self, client, model: str):
        """
        Initialize the LLM extractor.
        
        Args:
            client: LLM client with .chat() method
            model: Model name to use for extraction
        """
        self.client = client
        self.model = model
    
    def extract(self, dialogue: List[Dict], max_retries: int = 2) -> ExtractorJSON:
        """
        Extract clinical information from a dialogue.
        
        Args:
            dialogue: List of turn dicts with 'role' and 'text'
            max_retries: Number of retry attempts on parse failure
        
        Returns:
            ExtractorJSON with extracted clinical information
        """
        # Format dialogue for prompt
        dialogue_text = _format_dialogue_for_prompt(dialogue)
        
        user_prompt = f"""DOCTOR-PATIENT DIALOGUE:

{dialogue_text}

Extract all clinical information from this dialogue and return as JSON:"""
        
        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
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
                
                # Handle various response formats (str, dict, or list)
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
                return self._parse_llm_output(data, dialogue)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # Retry with stricter prompt
                    messages = [
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt + "\n\nReturn ONLY valid JSON, no other text."}
                    ]
                else:
                    print(f"LLM extraction JSON parse failed: {e}")
                    return ExtractorJSON()
            except Exception as e:
                if attempt >= max_retries:
                    print(f"LLM extraction failed: {e}")
                    return ExtractorJSON()
        
        return ExtractorJSON()
    
    def _parse_llm_output(self, data: dict, dialogue: List[Dict]) -> ExtractorJSON:
        """Convert LLM JSON output to ExtractorJSON schema."""
        ej = ExtractorJSON()
        
        # Demographics
        demographics_data = data.get("demographics", {})
        if isinstance(demographics_data, dict):
            # Age - normalize to string
            age = demographics_data.get("age")
            if age and str(age).lower() not in ("null", "none", "unknown", ""):
                # Extract numeric age if present
                age_str = str(age).strip()
                # Try to extract just the number if there's extra text
                import re
                age_match = re.search(r'(\d+)', age_str)
                if age_match:
                    ej.demographics["age"] = age_match.group(1)
            
            # Sex - normalize to lowercase
            sex = demographics_data.get("sex")
            if sex and str(sex).lower() not in ("null", "none", "unknown", ""):
                sex_lower = str(sex).lower().strip()
                # Normalize common variations
                if sex_lower in ("m", "male", "man"):
                    ej.demographics["sex"] = "male"
                elif sex_lower in ("f", "female", "woman"):
                    ej.demographics["sex"] = "female"
                else:
                    ej.demographics["sex"] = sex_lower
            
            # Race/Ethnicity - normalize to lowercase snake_case
            race = demographics_data.get("race_ethnicity")
            if race and str(race).lower() not in ("null", "none", "unknown", ""):
                race_norm = str(race).lower().strip().replace(" ", "_").replace("-", "_")
                ej.demographics["race_ethnicity"] = race_norm
            
            # Sexual Orientation - normalize to lowercase
            orientation = demographics_data.get("sexual_orientation")
            if orientation and str(orientation).lower() not in ("null", "none", "unknown", ""):
                orientation_norm = str(orientation).lower().strip().replace(" ", "_")
                ej.demographics["sexual_orientation"] = orientation_norm
        
        # Chief Complaint
        cc = data.get("chief_complaint")
        if cc and isinstance(cc, str):
            ej.chief_complaint = cc[:200]
        
        # If no chief complaint, try HPI
        if not ej.chief_complaint:
            hpi = data.get("history_of_present_illness")
            if hpi and isinstance(hpi, str):
                # Take first sentence of HPI as chief complaint
                first_sentence = hpi.split('.')[0].strip()
                if len(first_sentence) > 10:
                    ej.chief_complaint = first_sentence[:200]
        
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
                    severity=s.get("severity"),
                    onset=s.get("onset"),
                    modifiers=[
                        m for m in [
                            s.get("character"),
                            s.get("location"),
                            s.get("aggravating_factors"),
                            s.get("relieving_factors")
                        ] if m
                    ],
                    evidence=Evidence(utt_ids=[], spans=[{"text": name}])
                ))
        
        # Also extract from Review of Systems
        ros = data.get("review_of_systems", {})
        if isinstance(ros, dict):
            for system, findings in ros.items():
                if isinstance(findings, list):
                    for finding in findings:
                        if isinstance(finding, str) and finding:
                            # Check if it's a negative (often prefixed with "denies" or has "_absent")
                            assertion = "present"
                            name = finding
                            if finding.startswith("denies_") or finding.endswith("_absent"):
                                assertion = "absent"
                                name = finding.replace("denies_", "").replace("_absent", "")
                            
                            # Avoid duplicates with main symptoms
                            existing_names = {s.name_norm for s in ej.symptoms if s.name_norm}
                            if name.lower().replace(" ", "_") not in existing_names:
                                ej.symptoms.append(Symptom(
                                    name_surface=name.replace("_", " "),
                                    name_norm=name.lower().replace(" ", "_"),
                                    assertion=assertion,
                                    evidence=Evidence(utt_ids=[], spans=[])
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
                    route=m.get("route"),
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
        
        # Add social history items as risk factors
        social = data.get("social_history", {})
        if isinstance(social, dict):
            smoking = social.get("smoking", {})
            if isinstance(smoking, dict) and smoking.get("status") in ["current", "former"]:
                details = smoking.get("details", "")
                rf = f"smoking: {smoking['status']}"
                if details:
                    rf += f" ({details})"
                ej.risk_factors.append(rf)
            
            alcohol = social.get("alcohol", {})
            if isinstance(alcohol, dict) and alcohol.get("status") == "current":
                details = alcohol.get("details", "")
                rf = "alcohol use"
                if details:
                    rf += f" ({details})"
                ej.risk_factors.append(rf)
            
            drugs = social.get("drugs")
            if drugs and isinstance(drugs, str) and drugs.lower() not in ["none", "no", "denies"]:
                ej.risk_factors.append(f"drug use: {drugs}")
        
        # Add past medical history as QA extractions (or could be separate field)
        pmh = data.get("past_medical_history", [])
        if isinstance(pmh, list):
            for condition in pmh:
                if isinstance(condition, str) and condition:
                    ej.qa_extractions.append(QAExtracted(
                        concept=f"pmh_{condition.lower().replace(' ', '_')}",
                        value=condition,
                        assertion="present",
                        evidence=Evidence(utt_ids=[], spans=[])
                    ))
        
        # Add family history
        fhx = data.get("family_history", [])
        if isinstance(fhx, list):
            for item in fhx:
                if isinstance(item, str) and item:
                    ej.qa_extractions.append(QAExtracted(
                        concept=f"family_history",
                        value=item,
                        assertion="present",
                        evidence=Evidence(utt_ids=[], spans=[])
                    ))
        
        return ej


def create_llm_extractor(client, model: str) -> LLMExtractor:
    """Factory function to create an LLM extractor."""
    return LLMExtractor(client, model)

