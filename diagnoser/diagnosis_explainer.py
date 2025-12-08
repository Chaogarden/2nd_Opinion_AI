# diagnoser/diagnosis_explainer.py
# ==============================
# LLM-powered Natural Language Explanation of Diagnosis
# Translates structured diagnostic output into patient-friendly explanations
# ==============================

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.schema import DiagnoserOutput, EvidenceChunk
from extractor.schema import ExtractorJSON
from core.json_llm_utils import clean_json_string


# ============================================================
# Explanation Output Structure
# ============================================================

@dataclass
class DiagnosisExplanation:
    """Natural language explanation of the AI's diagnosis."""
    
    # Overall assessment
    overall_summary: str
    confidence_level: str  # "high", "moderate", "low"
    
    # Main diagnosis explanation
    primary_diagnosis: str
    primary_rationale: str
    
    # Alternative diagnoses to consider
    alternative_diagnoses: str
    
    # What symptoms/findings led to this conclusion
    key_findings: str
    
    # Red flags and urgent considerations
    urgent_considerations: str
    
    # Recommended next steps
    recommended_workup: str
    treatment_overview: str
    
    # Important caveats
    limitations_and_caveats: str
    
    # Full narrative (combined)
    full_narrative: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_summary": self.overall_summary,
            "confidence_level": self.confidence_level,
            "primary_diagnosis": self.primary_diagnosis,
            "primary_rationale": self.primary_rationale,
            "alternative_diagnoses": self.alternative_diagnoses,
            "key_findings": self.key_findings,
            "urgent_considerations": self.urgent_considerations,
            "recommended_workup": self.recommended_workup,
            "treatment_overview": self.treatment_overview,
            "limitations_and_caveats": self.limitations_and_caveats,
            "full_narrative": self.full_narrative
        }


# ============================================================
# LLM Prompt for Diagnosis Explanation
# ============================================================

DIAGNOSIS_SYSTEM_PROMPT = """You are a compassionate medical AI assistant for 2nd Opinion AI, helping patients understand their potential diagnosis. Your role is to explain complex medical findings in clear, empathetic, and reassuring language while being honest about uncertainties.

Guidelines for your explanation:
1. Be warm and reassuring, but honest about what the AI can and cannot determine
2. Use plain language that a patient without medical training can understand
3. Explain medical terms when you use them
4. Present diagnoses as possibilities to discuss with their doctor, NOT definitive conclusions
5. Highlight any urgent red flags that require immediate medical attention
6. Always emphasize this is a decision-support tool, not a replacement for professional medical care
7. Reference the clinical evidence and guidelines that support the analysis
8. Be clear about the confidence level and any limitations

Confidence levels:
- "high" = Strong evidence alignment, consistent symptoms, clear clinical picture
- "moderate" = Some supporting evidence but additional workup needed
- "low" = Limited information, multiple possibilities, significant uncertainty

IMPORTANT DISCLAIMERS TO INCLUDE:
- This is an AI-assisted analysis and should be discussed with a healthcare provider
- This is not a definitive diagnosis
- Seek immediate medical attention for any red flag symptoms mentioned"""


DIAGNOSIS_USER_PROMPT = """Based on the following clinical analysis, provide a clear, patient-friendly explanation of the findings.

PATIENT INFORMATION:
- Chief Complaint: {chief_complaint}
- Key Symptoms: {symptoms}
- Current Medications: {medications}
- Relevant Medical History: {medical_history}

DIAGNOSTIC ANALYSIS:
{diagnosis_json}

CONFIDENCE METRICS:
- Overall Uncertainty: {uncertainty:.0%} (lower is more confident)
- Groundedness Score: {groundedness:.0%} (how well-supported by evidence)

DIFFERENTIAL DIAGNOSES:
{differential_list}

RED FLAGS IDENTIFIED:
{red_flags}

RECOMMENDED WORKUP:
{workup_list}

MANAGEMENT PLAN SUMMARY:
{management_summary}

SUPPORTING EVIDENCE USED:
{evidence_citations}

Respond with a JSON object using empathetic, patient-friendly language:
{{
    "overall_summary": "2-3 sentence warm, clear summary of what the analysis found. Start with acknowledgment of their concerns.",
    "confidence_level": "high|moderate|low",
    "primary_diagnosis": "The most likely diagnosis explained in plain language with context",
    "primary_rationale": "Why this diagnosis fits their symptoms, explained simply",
    "alternative_diagnoses": "Other possibilities to consider and why they're less likely",
    "key_findings": "What symptoms and findings led to this assessment, in patient-friendly terms",
    "urgent_considerations": "Any red flags or urgent concerns they should know about. Say 'None identified' if none",
    "recommended_workup": "What tests or evaluations are recommended, explained in simple terms",
    "treatment_overview": "Overview of treatment approaches being considered",
    "limitations_and_caveats": "Important limitations of this AI analysis and why seeing a doctor is essential",
    "full_narrative": "Complete 3-4 paragraph patient-friendly explanation that ties everything together. Should read like a caring doctor explaining findings to a patient. Include appropriate caveats about AI limitations."
}}

Return ONLY the JSON object, no other text."""


# ============================================================
# Diagnosis Explainer
# ============================================================

class DiagnosisExplainer:
    """
    Generates natural language explanations of diagnostic results.
    
    Uses an LLM to translate structured diagnosis data into
    patient-friendly explanations.
    """
    
    def __init__(self, client, model: str):
        """
        Initialize the explainer.
        
        Args:
            client: LLM client with .chat() method
            model: Model name for explanation generation
        """
        self.client = client
        self.model = model
    
    def explain(
        self, 
        diagnoser_output: DiagnoserOutput,
        extracted_facts: ExtractorJSON,
        evidence_chunks: Optional[List[EvidenceChunk]] = None
    ) -> DiagnosisExplanation:
        """
        Generate a natural language explanation of the diagnosis.
        
        Args:
            diagnoser_output: DiagnoserOutput object to explain
            extracted_facts: ExtractorJSON with patient information
            evidence_chunks: Optional list of evidence chunks used
        
        Returns:
            DiagnosisExplanation with patient-friendly content
        """
        # Format the data for the prompt
        prompt_data = self._format_for_prompt(diagnoser_output, extracted_facts, evidence_chunks)
        
        # Build the prompt
        user_prompt = DIAGNOSIS_USER_PROMPT.format(**prompt_data)
        
        messages = [
            {"role": "system", "content": DIAGNOSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=0.3  # Slightly creative but mostly consistent
            )
            
            content = self._extract_content(response)
            data = self._parse_json(content)
            
            return DiagnosisExplanation(
                overall_summary=self._ensure_string(data.get("overall_summary", "Unable to generate summary.")),
                confidence_level=self._normalize_confidence(data.get("confidence_level", "moderate")),
                primary_diagnosis=self._ensure_string(data.get("primary_diagnosis", "Unable to determine.")),
                primary_rationale=self._ensure_string(data.get("primary_rationale", "Not available.")),
                alternative_diagnoses=self._ensure_string(data.get("alternative_diagnoses", "Not available.")),
                key_findings=self._ensure_string(data.get("key_findings", "Not available.")),
                urgent_considerations=self._ensure_string(data.get("urgent_considerations", "None identified.")),
                recommended_workup=self._ensure_string(data.get("recommended_workup", "Discuss with your healthcare provider.")),
                treatment_overview=self._ensure_string(data.get("treatment_overview", "To be discussed with your healthcare provider.")),
                limitations_and_caveats=self._ensure_string(data.get("limitations_and_caveats", "This is an AI analysis and should be reviewed by a healthcare professional.")),
                full_narrative=self._ensure_string(data.get("full_narrative", data.get("overall_summary", "Unable to generate explanation.")))
            )
            
        except Exception as e:
            print(f"Diagnosis explanation generation failed: {e}")
            return self._generate_fallback_explanation(diagnoser_output, extracted_facts)
    
    def _format_for_prompt(
        self,
        diagnoser_output: DiagnoserOutput,
        extracted_facts: ExtractorJSON,
        evidence_chunks: Optional[List[EvidenceChunk]] = None
    ) -> Dict[str, Any]:
        """Format data for the prompt template."""
        
        # Chief complaint
        chief_complaint = extracted_facts.chief_complaint or "Not specified"
        
        # Symptoms
        present_symptoms = [
            s.name_surface or s.name_norm 
            for s in extracted_facts.symptoms 
            if s.assertion == "present"
        ]
        symptoms = ", ".join(present_symptoms[:10]) if present_symptoms else "None documented"
        
        # Medications
        current_meds = [
            m.name_surface or m.name_norm
            for m in extracted_facts.meds
            if m.assertion == "present"
        ]
        medications = ", ".join(current_meds[:10]) if current_meds else "None documented"
        
        # Medical history (from risk factors and demographics)
        history_items = []
        if extracted_facts.risk_factors:
            history_items.extend(extracted_facts.risk_factors[:5])
        if extracted_facts.demographics:
            for key, val in extracted_facts.demographics.items():
                if val and key not in ("age", "sex", "gender"):
                    history_items.append(f"{key}: {val}")
        medical_history = ", ".join(history_items) if history_items else "Not documented"
        
        # Differential diagnoses
        diff_lines = []
        for i, dx in enumerate(diagnoser_output.differential[:5], 1):
            diff_lines.append(f"{i}. {dx.condition} ({dx.likelihood} likelihood): {dx.rationale}")
        differential_list = "\n".join(diff_lines) if diff_lines else "No differential diagnoses generated"
        
        # Red flags
        rf_lines = []
        for rf in diagnoser_output.red_flags:
            rf_lines.append(f"- [{rf.risk_level.upper()}] {rf.description}")
        red_flags = "\n".join(rf_lines) if rf_lines else "None identified"
        
        # Workup
        workup_lines = []
        for w in diagnoser_output.initial_workup[:5]:
            workup_lines.append(f"- {w.test_name} ({w.priority}): {w.rationale}")
        workup_list = "\n".join(workup_lines) if workup_lines else "No specific workup recommended"
        
        # Management summary
        mgmt_parts = []
        if diagnoser_output.management_plan.non_pharm:
            for np in diagnoser_output.management_plan.non_pharm[:3]:
                mgmt_parts.append(f"- {np.intervention}")
        if diagnoser_output.management_plan.pharm:
            for p in diagnoser_output.management_plan.pharm[:3]:
                dose_str = f"{p.dose} {p.route} {p.frequency}".strip()
                mgmt_parts.append(f"- {p.drug}: {dose_str} for {p.indication}")
        management_summary = "\n".join(mgmt_parts) if mgmt_parts else "Management plan pending further evaluation"
        
        # Evidence citations
        evidence_lines = []
        if evidence_chunks:
            for chunk in evidence_chunks[:5]:
                evidence_lines.append(f"- {chunk.evidence_id}: {chunk.heading_path or chunk.title}")
        evidence_citations = "\n".join(evidence_lines) if evidence_lines else "Clinical guidelines and medical references"
        
        # Simplified diagnosis JSON
        simplified_diagnosis = {
            "differential_count": len(diagnoser_output.differential),
            "red_flag_count": len(diagnoser_output.red_flags),
            "workup_items": len(diagnoser_output.initial_workup),
            "medications_recommended": len(diagnoser_output.management_plan.pharm),
            "non_pharm_recommendations": len(diagnoser_output.management_plan.non_pharm),
            "contraindications_flagged": len(diagnoser_output.contraindications_and_interactions)
        }
        
        return {
            "chief_complaint": chief_complaint,
            "symptoms": symptoms,
            "medications": medications,
            "medical_history": medical_history,
            "diagnosis_json": json.dumps(simplified_diagnosis, indent=2),
            "uncertainty": diagnoser_output.overall_uncertainty,
            "groundedness": diagnoser_output.groundedness_score,
            "differential_list": differential_list,
            "red_flags": red_flags,
            "workup_list": workup_list,
            "management_summary": management_summary,
            "evidence_citations": evidence_citations
        }
    
    def _extract_content(self, response) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content
        elif isinstance(response, dict):
            if "choices" in response:
                return response["choices"][0]["message"]["content"]
            elif "message" in response:
                return response["message"].get("content", "")
            elif "content" in response:
                return response["content"]
        elif isinstance(response, str):
            return response
        return str(response)
    
    def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown blocks."""
        content = content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        # Clean control characters and other JSON issues
        content = clean_json_string(content)
        
        return json.loads(content)
    
    def _ensure_string(self, value) -> str:
        """Ensure value is a string."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            items = []
            for item in value:
                if isinstance(item, str):
                    items.append(f"• {item}")
                elif isinstance(item, dict):
                    text = item.get("point") or item.get("text") or str(item)
                    items.append(f"• {text}")
                else:
                    items.append(f"• {str(item)}")
            return "\n".join(items)
        return str(value)
    
    def _normalize_confidence(self, value: str) -> str:
        """Normalize confidence level to valid values."""
        if not value:
            return "moderate"
        value_lower = value.lower().strip()
        if value_lower in ("high", "moderate", "low"):
            return value_lower
        return "moderate"
    
    def _generate_fallback_explanation(
        self,
        diagnoser_output: DiagnoserOutput,
        extracted_facts: ExtractorJSON
    ) -> DiagnosisExplanation:
        """Generate a basic explanation without LLM if the call fails."""
        
        # Determine confidence level
        uncertainty = diagnoser_output.overall_uncertainty
        if uncertainty < 0.3:
            confidence = "high"
        elif uncertainty < 0.6:
            confidence = "moderate"
        else:
            confidence = "low"
        
        # Build primary diagnosis
        primary_dx = "Unable to determine"
        primary_rationale = "Please consult with a healthcare provider for a proper evaluation."
        if diagnoser_output.differential:
            top_dx = diagnoser_output.differential[0]
            primary_dx = top_dx.condition
            primary_rationale = top_dx.rationale or "Based on the symptoms and findings provided."
        
        # Build alternative diagnoses
        alt_dx_parts = []
        for dx in diagnoser_output.differential[1:4]:
            alt_dx_parts.append(f"• {dx.condition} ({dx.likelihood} likelihood)")
        alternative_diagnoses = "\n".join(alt_dx_parts) if alt_dx_parts else "No alternatives identified."
        
        # Key findings
        present_symptoms = [
            s.name_surface or s.name_norm 
            for s in extracted_facts.symptoms 
            if s.assertion == "present"
        ]
        key_findings = f"Based on reported symptoms: {', '.join(present_symptoms[:5])}" if present_symptoms else "Limited symptom information available."
        
        # Red flags
        urgent_parts = []
        for rf in diagnoser_output.red_flags:
            urgent_parts.append(f"• {rf.description} ({rf.risk_level})")
        urgent_considerations = "\n".join(urgent_parts) if urgent_parts else "None identified."
        
        # Workup
        workup_parts = []
        for w in diagnoser_output.initial_workup[:5]:
            workup_parts.append(f"• {w.test_name} ({w.priority})")
        recommended_workup = "\n".join(workup_parts) if workup_parts else "Discuss with your healthcare provider."
        
        # Treatment
        treatment_parts = []
        for np in diagnoser_output.management_plan.non_pharm[:3]:
            treatment_parts.append(f"• {np.intervention}")
        for p in diagnoser_output.management_plan.pharm[:3]:
            treatment_parts.append(f"• {p.drug} for {p.indication}")
        treatment_overview = "\n".join(treatment_parts) if treatment_parts else "To be determined by your healthcare provider."
        
        # Build full narrative
        chief = extracted_facts.chief_complaint or "your symptoms"
        full_narrative = f"""Based on the information provided about {chief}, our AI analysis has generated some insights to discuss with your healthcare provider.

The analysis suggests {primary_dx} as the most likely explanation for your symptoms. {primary_rationale}

{"There are some urgent considerations that warrant prompt medical attention. " + urgent_considerations if diagnoser_output.red_flags else ""}

Please remember that this is an AI-assisted analysis tool and is not a substitute for professional medical evaluation. We recommend discussing these findings with your healthcare provider, who can perform a proper examination and order any necessary tests.

Your health is important, and a qualified healthcare professional should make any diagnostic or treatment decisions."""
        
        return DiagnosisExplanation(
            overall_summary=f"Based on your symptoms of {chief}, the AI analysis suggests {primary_dx} as a possibility to discuss with your doctor.",
            confidence_level=confidence,
            primary_diagnosis=primary_dx,
            primary_rationale=primary_rationale,
            alternative_diagnoses=alternative_diagnoses,
            key_findings=key_findings,
            urgent_considerations=urgent_considerations,
            recommended_workup=recommended_workup,
            treatment_overview=treatment_overview,
            limitations_and_caveats="This is an AI-assisted analysis and should be reviewed by a healthcare professional. It is not a definitive diagnosis.",
            full_narrative=full_narrative.strip()
        )


# ============================================================
# Convenience Functions
# ============================================================

def explain_diagnosis(
    diagnoser_output: DiagnoserOutput,
    extracted_facts: ExtractorJSON,
    llm_client,
    llm_model: str,
    evidence_chunks: Optional[List[EvidenceChunk]] = None
) -> DiagnosisExplanation:
    """
    Generate a natural language explanation of a diagnosis.
    
    Args:
        diagnoser_output: DiagnoserOutput to explain
        extracted_facts: ExtractorJSON with patient info
        llm_client: LLM client with .chat() method
        llm_model: Model name
        evidence_chunks: Optional evidence used in analysis
    
    Returns:
        DiagnosisExplanation with patient-friendly content
    """
    explainer = DiagnosisExplainer(llm_client, llm_model)
    return explainer.explain(diagnoser_output, extracted_facts, evidence_chunks)


def format_diagnosis_explanation_text(explanation: DiagnosisExplanation) -> str:
    """
    Format a diagnosis explanation as readable text.
    
    Args:
        explanation: DiagnosisExplanation object
    
    Returns:
        Formatted string for display
    """
    def _safe_str(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)
    
    lines = []
    
    lines.append("=" * 70)
    lines.append("2ND OPINION AI: DIAGNOSTIC ANALYSIS")
    lines.append("=" * 70)
    lines.append("")
    
    # Confidence badge
    confidence = _safe_str(explanation.confidence_level)
    confidence_emoji = {
        "high": "🟢",
        "moderate": "🟡", 
        "low": "🔴"
    }.get(confidence.lower(), "⚪")
    
    lines.append(f"{confidence_emoji} CONFIDENCE: {confidence.upper()}")
    lines.append("")
    lines.append(_safe_str(explanation.overall_summary))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("🩺 PRIMARY ASSESSMENT:")
    lines.append("-" * 70)
    lines.append(f"**{_safe_str(explanation.primary_diagnosis)}**")
    lines.append("")
    lines.append(_safe_str(explanation.primary_rationale))
    lines.append("")
    
    alt_dx = _safe_str(explanation.alternative_diagnoses)
    if alt_dx and alt_dx.lower() not in ("not available.", "no alternatives identified."):
        lines.append("-" * 70)
        lines.append("🔍 ALTERNATIVE POSSIBILITIES:")
        lines.append("-" * 70)
        lines.append(alt_dx)
        lines.append("")
    
    urgent = _safe_str(explanation.urgent_considerations)
    if urgent and urgent.lower() not in ("none identified.", "none identified"):
        lines.append("-" * 70)
        lines.append("🚨 URGENT CONSIDERATIONS:")
        lines.append("-" * 70)
        lines.append(urgent)
        lines.append("")
    
    lines.append("-" * 70)
    lines.append("📋 KEY FINDINGS:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.key_findings))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("🧪 RECOMMENDED NEXT STEPS:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.recommended_workup))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("💊 TREATMENT OVERVIEW:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.treatment_overview))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("⚠️ IMPORTANT LIMITATIONS:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.limitations_and_caveats))
    lines.append("")
    
    lines.append("=" * 70)
    lines.append("FULL SUMMARY:")
    lines.append("=" * 70)
    lines.append(_safe_str(explanation.full_narrative))
    lines.append("")
    
    return "\n".join(lines)


# ============================================================
# CLI Test
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test diagnosis explainer")
    parser.add_argument("--mode", "-m", type=str, default="test",
                        choices=["test", "prod"],
                        help="LLM mode")
    
    args = parser.parse_args()
    
    print("Diagnosis Explainer CLI - Run with actual pipeline output for testing")
    print(f"Mode: {args.mode}")
