# diagnoser/engine.py
# ==============================
# LLM-powered Diagnoser and Consultant engines
# - DiagnoserEngine: generates structured clinical plan
# - ConsultantEngine: critiques for safety and guideline alignment
# ==============================

import json
from typing import Optional, List

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_clients import BaseLLMClient
from core.json_llm_utils import extract_json_block, clean_json_string
from diagnoser.schema import (
    DiagnoserInput, DiagnoserOutput, ConsultantCritique,
    EvidenceChunk, DifferentialDx, RedFlag, WorkupItem,
    ManagementPlan, NonPharmRec, PharmRec,
    ContraindicationOrInteraction, CodingAndLabels,
    CritiqueItem
)


# ============================================================
# Value Normalizers (handle LLM returning non-schema values)
# ============================================================

def _normalize_likelihood(value: str) -> str:
    """Normalize likelihood to: high, moderate, low"""
    if not value or not isinstance(value, str):
        return "moderate"
    v = value.lower().strip()
    if v in ("high", "moderate", "low"):
        return v
    if v in ("very high", "highly likely", "probable", "likely"):
        return "high"
    if v in ("very low", "unlikely", "rare", "minimal"):
        return "low"
    return "moderate"


def _normalize_risk_level(value: str) -> str:
    """Normalize risk_level to: critical, high, moderate (RedFlag only allows these)"""
    if not value or not isinstance(value, str):
        return "moderate"
    v = value.lower().strip()
    if v in ("critical", "high", "moderate"):
        return v
    if v in ("severe", "life-threatening", "emergent", "urgent"):
        return "critical"
    if v in ("significant", "important", "elevated"):
        return "high"
    # Map "low" to "moderate" since our schema doesn't allow "low" for red flags
    if v in ("low", "minimal", "minor"):
        return "moderate"
    return "moderate"


def _normalize_severity(value: str) -> str:
    """Normalize severity to: critical, high, moderate, low"""
    if not value or not isinstance(value, str):
        return "moderate"
    v = value.lower().strip()
    if v in ("critical", "high", "moderate", "low"):
        return v
    if v in ("severe", "life-threatening", "emergent"):
        return "critical"
    if v in ("significant", "important"):
        return "high"
    if v in ("mild", "minor", "minimal"):
        return "low"
    return "moderate"


def _normalize_priority(value: str) -> str:
    """Normalize priority to: urgent, routine, optional"""
    if not value or not isinstance(value, str):
        return "routine"
    v = value.lower().strip()
    if v in ("urgent", "routine", "optional"):
        return v
    if v in ("emergent", "stat", "immediate", "critical", "high"):
        return "urgent"
    if v in ("low", "elective", "if needed", "consider"):
        return "optional"
    return "routine"


def _normalize_safety_rating(value: str) -> str:
    """Normalize safety rating to: safe, needs_review, unsafe"""
    if not value or not isinstance(value, str):
        return "needs_review"
    v = value.lower().strip().replace(" ", "_")
    if v in ("safe", "needs_review", "unsafe"):
        return v
    if v in ("approved", "acceptable", "ok", "good"):
        return "safe"
    if v in ("dangerous", "harmful", "contraindicated", "reject"):
        return "unsafe"
    return "needs_review"


def _normalize_critique_kind(value: str) -> str:
    """Normalize critique kind to valid values."""
    valid = {"missing_differential", "unsafe_recommendation", "guideline_conflict", 
             "safety_gap", "dosing_error", "interaction_missed", "other"}
    if not value or not isinstance(value, str):
        return "other"
    v = value.lower().strip().replace(" ", "_").replace("-", "_")
    if v in valid:
        return v
    return "other"


def _extract_content_from_response(response) -> str:
    """
    Extract text content from various LLM response formats.
    
    Handles:
    - str: Return as-is
    - Ollama response object: Get message.content
    - dict with 'message': Get message.content or message['content']
    - dict with 'content': Get 'content' directly
    - list: Get first element's content
    """
    if isinstance(response, str):
        return response
    
    # Handle Ollama response object (has .message.content)
    if hasattr(response, 'message'):
        msg = response.message
        if hasattr(msg, 'content'):
            return msg.content
        elif isinstance(msg, dict):
            return msg.get("content", "")
    
    # Handle dict responses
    if isinstance(response, dict):
        # Try message.content first (Ollama format)
        if "message" in response:
            msg = response["message"]
            if isinstance(msg, dict):
                return msg.get("content", "")
            elif hasattr(msg, 'content'):
                return msg.content
        # Try direct content key
        return response.get("content", "")
    
    # Handle list responses
    if isinstance(response, list) and response:
        first = response[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    
    return str(response) if response else ""


# ============================================================
# System Prompts
# ============================================================

DIAGNOSER_SYSTEM_PROMPT = """You are an expert clinical diagnostic assistant. Your task is to generate a comprehensive clinical assessment and management plan based on the provided patient information and medical evidence.

You MUST return your response as a single valid JSON object conforming to this EXACT schema:

{
  "differential": [
    {
      "condition": "string (diagnosis name)",
      "likelihood": "high|moderate|low",
      "rationale": "string (clinical reasoning)",
      "evidence_ids": ["string (IDs from provided evidence)"]
    }
  ],
  "red_flags": [
    {
      "description": "string",
      "risk_level": "critical|high|moderate",
      "evidence_ids": ["string"]
    }
  ],
  "initial_workup": [
    {
      "test_name": "string",
      "priority": "urgent|routine|optional",
      "rationale": "string",
      "evidence_ids": ["string"]
    }
  ],
  "management_plan": {
    "non_pharm": [
      {
        "intervention": "string",
        "rationale": "string",
        "evidence_ids": ["string"]
      }
    ],
    "pharm": [
      {
        "drug": "string (generic name)",
        "route": "string",
        "dose": "string",
        "frequency": "string",
        "duration": "string",
        "indication": "string",
        "evidence_ids": ["string"]
      }
    ]
  },
  "contraindications_and_interactions": [
    {
      "issue": "string",
      "related_meds": ["string"],
      "severity": "critical|high|moderate|low",
      "evidence_ids": ["string"]
    }
  ],
  "coding_and_labels": {
    "icd10_codes": ["string"],
    "snomed_codes": ["string"],
    "labels": ["string"]
  },
  "overall_uncertainty": 0.0-1.0,
  "groundedness_score": 0.0-1.0,
  "notes_on_missing_info": "string or null"
}

CRITICAL REQUIREMENTS:
1. Every clinical claim MUST include evidence_ids referencing the provided evidence chunks.
2. Use the EXACT evidence_id strings provided (e.g., "guidelines:123", "merck:456").
3. Be conservative with recommendations - prefer well-established guidelines.
4. Clearly indicate uncertainty when evidence is limited.
5. Return ONLY the JSON object, no other text.
"""

CONSULTANT_SYSTEM_PROMPT = """You are a senior clinical consultant reviewing a colleague's diagnostic plan. Your role is to identify safety issues, guideline conflicts, medical errors, and areas for improvement.

You MUST return your critique as a single valid JSON object conforming to this EXACT schema:

{
  "issues": [
    {
      "kind": "missing_differential|unsafe_recommendation|guideline_conflict|safety_gap|dosing_error|interaction_missed|other",
      "description": "string (detailed description of the issue)",
      "severity": "critical|high|moderate|low",
      "target_path": "string (JSON pointer to problematic element, e.g., '/management_plan/pharm/0')",
      "evidence_ids": ["string (evidence supporting this critique)"],
      "suggested_fix": "string or null"
    }
  ],
  "overall_assessment": "string (summary of your review)",
  "overall_safety_rating": "safe|needs_review|unsafe"
}

REVIEW CRITERIA:
1. SAFETY: Check for dangerous drug interactions, contraindications, and dosing errors.
2. GUIDELINE ALIGNMENT: Verify recommendations match current clinical guidelines.
3. COMPLETENESS: Identify important missing differential diagnoses or workup items.
4. EVIDENCE GROUNDING: Flag claims that lack supporting evidence_ids.

SEVERITY LEVELS:
- critical: Immediate patient harm risk, must not proceed
- high: Significant safety concern, requires addressing before implementation
- moderate: Should be fixed but not blocking
- low: Minor improvement opportunity

Return ONLY the JSON object, no other text.
"""

REVISION_SYSTEM_PROMPT = """You are a senior clinical physician revising your initial assessment based on a colleague's critique.

CONTEXT:
- You generated an initial clinical plan
- A consulting physician reviewed it and identified issues
- You must now revise your plan to address their concerns

RULES:
1. Address EVERY issue raised in the critique
2. If the critique suggests a missing differential, add it with proper rationale
3. If the critique identifies an unsafe recommendation, remove or modify it
4. If the critique notes a guideline conflict, align with the guideline
5. If the critique identifies a dosing error, correct it with evidence
6. Preserve parts of your original plan that were NOT criticized
7. Maintain evidence_id citations for all recommendations

OUTPUT FORMAT:
Return the COMPLETE revised clinical plan as a single JSON object with the same schema as before:
{
  "differential": [...],
  "red_flags": [...],
  "initial_workup": [...],
  "management_plan": {"non_pharm": [...], "pharm": [...]},
  "contraindications_and_interactions": [...],
  "coding_and_labels": {...},
  "overall_uncertainty": <float 0-1>,
  "groundedness_score": <float 0-1>,
  "revision_notes": "<brief summary of changes made>"
}

Return ONLY valid JSON, no additional text.
"""


def _format_evidence_for_prompt(chunks: List[EvidenceChunk], max_chars: int = 1500) -> str:
    """Format evidence chunks for inclusion in the LLM prompt."""
    lines = ["AVAILABLE EVIDENCE:"]
    for chunk in chunks:
        lines.append(f"\n[{chunk.evidence_id}] Source: {chunk.source}")
        lines.append(f"Title: {chunk.title}")
        if chunk.heading_path:
            lines.append(f"Section: {chunk.heading_path}")
        # Truncate long text
        text = chunk.text[:max_chars] if len(chunk.text) > max_chars else chunk.text
        lines.append(f"Content: {text}")
        lines.append("-" * 40)
    return "\n".join(lines)


def _format_evidence_brief(chunks: List[EvidenceChunk]) -> str:
    """Format evidence chunks briefly (IDs and titles only) for Consultant."""
    if not chunks:
        return "AVAILABLE EVIDENCE IDs: None"
    lines = ["AVAILABLE EVIDENCE IDs:"]
    for chunk in chunks:
        lines.append(f"  [{chunk.evidence_id}] {chunk.title[:80]}")
    return "\n".join(lines)


def _format_patient_facts(input_data: DiagnoserInput) -> str:
    """Format patient facts for the prompt."""
    pf = input_data.patient_facts
    
    lines = ["PATIENT FACTS:"]
    
    if pf.chief_complaint:
        lines.append(f"Chief Complaint: {pf.chief_complaint}")
    
    if pf.symptoms:
        symptom_strs = []
        for s in pf.symptoms:
            name = s.name_norm or s.name_surface
            parts = [name]
            if s.assertion != "present":
                parts.append(f"({s.assertion})")
            if s.severity:
                parts.append(f"severity={s.severity}")
            if s.duration:
                parts.append(f"duration={s.duration}")
            symptom_strs.append(" ".join(parts))
        lines.append(f"Symptoms: {'; '.join(symptom_strs)}")
    
    if pf.meds:
        med_strs = []
        for m in pf.meds:
            name = m.name_norm or m.name_surface
            parts = [name]
            if m.dose:
                parts.append(m.dose)
            if m.freq:
                parts.append(m.freq)
            if m.assertion != "present":
                parts.append(f"({m.assertion})")
            med_strs.append(" ".join(parts))
        lines.append(f"Current Medications: {'; '.join(med_strs)}")
    
    if pf.allergies:
        allergy_strs = [a.substance_surface for a in pf.allergies if a.assertion == "present"]
        if allergy_strs:
            lines.append(f"Allergies: {', '.join(allergy_strs)}")
    
    if pf.vitals:
        vital_strs = [f"{v.kind}={v.value}" for v in pf.vitals]
        lines.append(f"Vitals: {', '.join(vital_strs)}")
    
    if pf.risk_factors:
        lines.append(f"Risk Factors: {', '.join(pf.risk_factors)}")
    
    return "\n".join(lines)


class DiagnoserEngine:
    """
    LLM engine that generates a structured clinical diagnostic plan.
    """
    
    def __init__(self, client: BaseLLMClient, model: str):
        """
        Initialize the Diagnoser engine.
        
        Args:
            client: LLM client instance.
            model: Model name to use.
        """
        self.client = client
        self.model = model
    
    def run(self, input_data: DiagnoserInput, max_retries: int = 2) -> DiagnoserOutput:
        """
        Generate a diagnostic plan from the input.
        
        Args:
            input_data: DiagnoserInput with patient facts and evidence.
            max_retries: Number of retry attempts on parse failure.
        
        Returns:
            DiagnoserOutput with the structured clinical plan.
        """
        # Build user message
        user_parts = [
            _format_patient_facts(input_data),
            "",
            f"DIALOGUE SUMMARY:\n{input_data.dialogue_brief}" if input_data.dialogue_brief else "",
            "",
            _format_evidence_for_prompt(input_data.evidence_chunks),
            "",
            "Based on the above patient information and evidence, generate a clinical assessment.",
            "",
            "IMPORTANT: Your response MUST be a JSON object with these exact top-level keys:",
            '  "differential": [array of diagnosis objects with condition, likelihood, rationale, evidence_ids]',
            '  "red_flags": [array of red flag objects]',
            '  "initial_workup": [array of test objects]',
            '  "management_plan": {object with non_pharm and pharm arrays}',
            '  "overall_uncertainty": (number 0-1)',
            '  "groundedness_score": (number 0-1)',
            "",
            "Return ONLY the JSON object, starting with { and ending with }:"
        ]
        user_message = "\n".join(user_parts)
        
        messages = [
            {"role": "system", "content": DIAGNOSER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Try to get valid JSON
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )
                
                content = _extract_content_from_response(response)
                
                # Extract and parse JSON
                json_str = extract_json_block(content)
                if not json_str:
                    json_str = content.strip()
                
                json_str = clean_json_string(json_str)
                data = json.loads(json_str)
                
                # Handle LLM returning array instead of object
                # Keep unwrapping until we get a dict or give up
                max_unwrap = 3
                for _ in range(max_unwrap):
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    else:
                        break
                
                # Final check - must be a dict
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict but got {type(data).__name__}")
                
                return self._parse_output(data)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # Retry with stricter prompt
                    messages = [
                        {"role": "system", "content": DIAGNOSER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message + "\n\nReturn ONLY valid minified JSON, no other text."}
                    ]
                else:
                    # Return default output with error note
                    return DiagnoserOutput(
                        notes_on_missing_info=f"Failed to parse LLM response: {str(e)}",
                        overall_uncertainty=1.0,
                        groundedness_score=0.0
                    )
            except Exception as e:
                if attempt >= max_retries:
                    return DiagnoserOutput(
                        notes_on_missing_info=f"Error generating diagnosis: {str(e)}",
                        overall_uncertainty=1.0,
                        groundedness_score=0.0
                    )
        
        return DiagnoserOutput(
            notes_on_missing_info="Failed to generate diagnosis after retries",
            overall_uncertainty=1.0,
            groundedness_score=0.0
        )
    
    def _parse_output(self, data: dict) -> DiagnoserOutput:
        """Parse raw JSON dict into DiagnoserOutput."""
        # Helper to safely get dict items (handles malformed LLM output)
        def safe_get(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            return default
        
        # Parse differential
        differential = []
        for d in data.get("differential", []):
            if not isinstance(d, dict):
                continue  # Skip malformed items
            differential.append(DifferentialDx(
                condition=safe_get(d, "condition", "Unknown"),
                likelihood=_normalize_likelihood(safe_get(d, "likelihood", "moderate")),
                rationale=safe_get(d, "rationale", ""),
                evidence_ids=safe_get(d, "evidence_ids", [])
            ))
        
        # Parse red flags
        red_flags = []
        for rf in data.get("red_flags", []):
            if not isinstance(rf, dict):
                continue  # Skip malformed items
            red_flags.append(RedFlag(
                description=safe_get(rf, "description", ""),
                risk_level=_normalize_risk_level(safe_get(rf, "risk_level", "high")),
                evidence_ids=safe_get(rf, "evidence_ids", [])
            ))
        
        # Parse workup
        initial_workup = []
        for w in data.get("initial_workup", []):
            if not isinstance(w, dict):
                continue  # Skip malformed items
            initial_workup.append(WorkupItem(
                test_name=safe_get(w, "test_name", ""),
                priority=_normalize_priority(safe_get(w, "priority", "routine")),
                rationale=safe_get(w, "rationale", ""),
                evidence_ids=safe_get(w, "evidence_ids", [])
            ))
        
        # Parse management plan
        mp_data = data.get("management_plan", {})
        if not isinstance(mp_data, dict):
            mp_data = {}
        non_pharm = []
        for np_item in mp_data.get("non_pharm", []):
            if not isinstance(np_item, dict):
                continue  # Skip malformed items
            non_pharm.append(NonPharmRec(
                intervention=safe_get(np_item, "intervention", ""),
                rationale=safe_get(np_item, "rationale", ""),
                evidence_ids=safe_get(np_item, "evidence_ids", [])
            ))
        
        pharm = []
        for p in mp_data.get("pharm", []):
            if not isinstance(p, dict):
                continue  # Skip malformed items
            pharm.append(PharmRec(
                drug=safe_get(p, "drug", ""),
                route=safe_get(p, "route", "oral"),
                dose=safe_get(p, "dose", ""),
                frequency=safe_get(p, "frequency", ""),
                duration=safe_get(p, "duration", ""),
                indication=safe_get(p, "indication", ""),
                evidence_ids=safe_get(p, "evidence_ids", [])
            ))
        
        management_plan = ManagementPlan(non_pharm=non_pharm, pharm=pharm)
        
        # Parse contraindications
        contras = []
        for c in data.get("contraindications_and_interactions", []):
            if not isinstance(c, dict):
                continue  # Skip malformed items
            contras.append(ContraindicationOrInteraction(
                issue=safe_get(c, "issue", ""),
                related_meds=safe_get(c, "related_meds", []),
                severity=_normalize_severity(safe_get(c, "severity", "moderate")),
                evidence_ids=safe_get(c, "evidence_ids", [])
            ))
        
        # Parse coding
        coding_data = data.get("coding_and_labels", {})
        if not isinstance(coding_data, dict):
            coding_data = {}
        coding = CodingAndLabels(
            icd10_codes=coding_data.get("icd10_codes", []),
            snomed_codes=coding_data.get("snomed_codes", []),
            labels=coding_data.get("labels", [])
        )
        
        return DiagnoserOutput(
            differential=differential,
            red_flags=red_flags,
            initial_workup=initial_workup,
            management_plan=management_plan,
            contraindications_and_interactions=contras,
            coding_and_labels=coding,
            overall_uncertainty=data.get("overall_uncertainty", 0.5),
            groundedness_score=data.get("groundedness_score", 0.5),
            notes_on_missing_info=data.get("notes_on_missing_info")
        )
    
    def revise(
        self,
        input_data: DiagnoserInput,
        original_output: DiagnoserOutput,
        critique: ConsultantCritique,
        max_retries: int = 2
    ) -> DiagnoserOutput:
        """
        Revise the clinical plan based on Consultant's critique.
        
        Args:
            input_data: Original DiagnoserInput.
            original_output: The initial DiagnoserOutput.
            critique: ConsultantCritique with identified issues.
            max_retries: Number of retry attempts.
        
        Returns:
            Revised DiagnoserOutput addressing the critique.
        """
        # Format the critique for the prompt
        critique_lines = []
        for i, issue in enumerate(critique.issues, 1):
            critique_lines.append(
                f"{i}. [{issue.severity.upper()}] {issue.kind}: {issue.description}"
            )
            if issue.suggested_fix:
                critique_lines.append(f"   Suggested fix: {issue.suggested_fix}")
        
        critique_text = "\n".join(critique_lines) if critique_lines else "No specific issues identified."
        
        # Build user message
        user_parts = [
            "=== ORIGINAL PATIENT DATA ===",
            _format_patient_facts(input_data),
            "",
            "=== AVAILABLE EVIDENCE ===",
            _format_evidence_for_prompt(input_data.evidence_chunks),
            "",
            "=== YOUR ORIGINAL PLAN ===",
            json.dumps(original_output.model_dump(), indent=2),
            "",
            "=== CONSULTANT'S CRITIQUE ===",
            critique_text,
            f"\nOverall Assessment: {critique.overall_assessment}",
            f"Safety Rating: {critique.overall_safety_rating}",
            "",
            "Please revise your clinical plan to address ALL issues raised above:"
        ]
        user_message = "\n".join(user_parts)
        
        messages = [
            {"role": "system", "content": REVISION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Try to get valid JSON
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=0.1  # Slightly higher for revision creativity
                )
                
                content = _extract_content_from_response(response)
                
                # Extract and parse JSON
                json_str = extract_json_block(content)
                if not json_str:
                    json_str = content.strip()
                
                json_str = clean_json_string(json_str)
                data = json.loads(json_str)

                # Handle LLM returning array instead of object
                # Keep unwrapping until we get a dict or give up
                for _ in range(3):
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    else:
                        break
                
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict but got {type(data).__name__}")

                # Add revision notes if present
                revised = self._parse_output(data)
                revision_notes = data.get("revision_notes", "")
                if revision_notes:
                    existing_notes = revised.notes_on_missing_info or ""
                    revised.notes_on_missing_info = (
                        f"Revision notes: {revision_notes}" +
                        (f"\n\n{existing_notes}" if existing_notes else "")
                    )
                
                return revised
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    messages[-1]["content"] += "\n\nReturn ONLY valid minified JSON."
                else:
                    # Return original with error note
                    original_output.notes_on_missing_info = (
                        (original_output.notes_on_missing_info or "") +
                        f"\n\nRevision failed: {str(e)}"
                    )
                    return original_output
            except Exception as e:
                if attempt >= max_retries:
                    original_output.notes_on_missing_info = (
                        (original_output.notes_on_missing_info or "") +
                        f"\n\nRevision error: {str(e)}"
                    )
                    return original_output
        
        return original_output


class ConsultantEngine:
    """
    LLM engine that critiques a diagnostic plan for safety and guideline alignment.
    """
    
    def __init__(self, client: BaseLLMClient, model: str):
        """
        Initialize the Consultant engine.
        
        Args:
            client: LLM client instance.
            model: Model name to use.
        """
        self.client = client
        self.model = model
    
    def run(
        self,
        input_data: DiagnoserInput,
        diagnoser_output: DiagnoserOutput,
        max_retries: int = 2
    ) -> ConsultantCritique:
        """
        Critique a diagnostic plan.
        
        Args:
            input_data: Original DiagnoserInput (for context).
            diagnoser_output: The plan to critique.
            max_retries: Number of retry attempts on parse failure.
        
        Returns:
            ConsultantCritique with identified issues.
        """
        # Build user message (compact version for faster processing)
        # Truncate dialogue brief for Consultant
        dialogue_brief = input_data.dialogue_brief
        if dialogue_brief and len(dialogue_brief) > 500:
            dialogue_brief = dialogue_brief[:500] + "..."
        
        # Use compact JSON (no indent) for diagnoser output
        diagnoser_json = json.dumps(diagnoser_output.model_dump(), separators=(',', ':'))
        
        user_parts = [
            _format_patient_facts(input_data),
            "",
            f"DIALOGUE SUMMARY:\n{dialogue_brief}" if dialogue_brief else "",
            "",
            _format_evidence_brief(input_data.evidence_chunks),  # Just IDs, not full text
            "",
            "DIAGNOSTIC PLAN TO REVIEW:",
            diagnoser_json,
            "",
            "Provide your critique as JSON:"
        ]
        user_message = "\n".join(user_parts)
        
        messages = [
            {"role": "system", "content": CONSULTANT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        # Try to get valid JSON
        for attempt in range(max_retries + 1):
            try:
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=0.0
                )
                
                content = _extract_content_from_response(response)
                
                # Extract and parse JSON
                json_str = extract_json_block(content)
                if not json_str:
                    json_str = content.strip()
                
                json_str = clean_json_string(json_str)
                data = json.loads(json_str)

                # Handle LLM returning array instead of object
                # Keep unwrapping until we get a dict or give up
                for _ in range(3):
                    if isinstance(data, list):
                        data = data[0] if data else {}
                    else:
                        break
                
                if not isinstance(data, dict):
                    raise ValueError(f"Expected dict but got {type(data).__name__}")

                return self._parse_critique(data)
                
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    messages = [
                        {"role": "system", "content": CONSULTANT_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message + "\n\nReturn ONLY valid minified JSON, no other text."}
                    ]
                else:
                    return ConsultantCritique(
                        issues=[],
                        overall_assessment=f"Failed to parse critique: {str(e)}",
                        overall_safety_rating="needs_review"
                    )
            except Exception as e:
                if attempt >= max_retries:
                    return ConsultantCritique(
                        issues=[],
                        overall_assessment=f"Error generating critique: {str(e)}",
                        overall_safety_rating="needs_review"
                    )
        
        return ConsultantCritique(
            issues=[],
            overall_assessment="Failed to generate critique after retries",
            overall_safety_rating="needs_review"
        )
    
    def _parse_critique(self, data: dict) -> ConsultantCritique:
        """Parse raw JSON dict into ConsultantCritique."""
        # Helper to safely get dict items (handles malformed LLM output)
        def safe_get(item, key, default=None):
            if isinstance(item, dict):
                return item.get(key, default)
            return default
        
        issues = []
        for issue in data.get("issues", []):
            if not isinstance(issue, dict):
                continue  # Skip malformed items
            issues.append(CritiqueItem(
                kind=_normalize_critique_kind(safe_get(issue, "kind", "other")),
                description=safe_get(issue, "description", ""),
                severity=_normalize_severity(safe_get(issue, "severity", "moderate")),
                target_path=safe_get(issue, "target_path", ""),
                evidence_ids=safe_get(issue, "evidence_ids", []),
                suggested_fix=safe_get(issue, "suggested_fix")
            ))

        return ConsultantCritique(
            issues=issues,
            overall_assessment=data.get("overall_assessment", ""),
            overall_safety_rating=_normalize_safety_rating(data.get("overall_safety_rating", "needs_review"))
        )

