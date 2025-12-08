# diagnoser/critique_explainer.py
# ==============================
# LLM-powered Natural Language Explanation of Note Critiques
# Translates structured critique into patient-friendly explanations
# ==============================

import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.critic_schema import NoteCritique
from core.json_llm_utils import clean_json_string


# ============================================================
# Explanation Output Structure
# ============================================================

@dataclass
class CritiqueExplanation:
    """Natural language explanation of a critique."""
    
    # Overall assessment
    overall_summary: str
    overall_rating: str  # "excellent", "good", "fair", "needs_improvement", "concerning"
    
    # What the doctor did well
    strengths: str
    
    # Areas of concern
    concerns: str
    
    # Specific findings
    documentation_quality: str
    diagnostic_reasoning: str
    treatment_plan: str
    safety_assessment: str
    
    # Actionable recommendations
    recommendations: str
    
    # Full narrative (combined)
    full_narrative: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_summary": self.overall_summary,
            "overall_rating": self.overall_rating,
            "strengths": self.strengths,
            "concerns": self.concerns,
            "documentation_quality": self.documentation_quality,
            "diagnostic_reasoning": self.diagnostic_reasoning,
            "treatment_plan": self.treatment_plan,
            "safety_assessment": self.safety_assessment,
            "recommendations": self.recommendations,
            "full_narrative": self.full_narrative
        }


# ============================================================
# LLM Prompt for Explanation
# ============================================================

EXPLANATION_SYSTEM_PROMPT = """You are a medical quality reviewer for 2nd Opinion AI, helping patients understand how well their doctor documented and managed their visit. Your role is to explain technical medical critique findings in clear, reassuring, but honest language.

Guidelines for your explanation:
1. Be balanced - acknowledge what the doctor did well before discussing concerns
2. Use plain language a patient can understand
3. ALWAYS CITE EVIDENCE - Reference specific scores, percentages, and evidence IDs when making claims
   - Example: "Your doctor's treatment plan aligned with clinical guidelines 85% of the time (see: guidelines:42, guidelines:78)"
   - Example: "The note accurately captured 90% of your reported symptoms"
4. Frame concerns constructively, not alarmingly
5. When issues are identified, recommend using 2nd Opinion AI's diagnostic analysis feature for a comprehensive second opinion
6. Be concise but thorough

Rating scale to use:
- "excellent" = Very thorough, no significant concerns
- "good" = Solid documentation with minor gaps
- "fair" = Adequate but has notable areas for improvement  
- "needs_improvement" = Several concerns that should be addressed
- "concerning" = Significant issues that warrant attention

IMPORTANT: When recommending next steps for concerning findings, suggest:
'Consider running the 2nd Opinion AI diagnostic analysis on your case for a comprehensive second opinion that cross-references your symptoms with clinical guidelines.'"""


EXPLANATION_USER_PROMPT = """Based on the following critique of a doctor's clinical note, provide a clear explanation for the patient. CITE SPECIFIC EVIDENCE for all claims.

CRITIQUE DATA:
{critique_json}

KEY METRICS (cite these percentages in your response):
- Overall Score: {overall_score:.0%}
- Overall Verdict: {overall_verdict}
- Safety Rating: {overall_safety}

CONSISTENCY WITH CONVERSATION:
- Support Score: {support_score:.0%} of note claims supported by conversation
- Hallucination Rate: {hallucination_rate:.0%} of claims had no supporting evidence
- Contradictions Found: {contradictions}

COMPLETENESS (cite these recall percentages):
- Symptom Coverage: {symptom_recall:.0%}
- Medication Coverage: {med_recall:.0%}
- Allergy Coverage: {allergy_recall:.0%}
- Critical Omissions: {critical_omissions}

DIAGNOSTIC QUALITY:
- Diagnosis Coherence Score: {dx_coherence:.0%}
- Differential Breadth Score: {dx_breadth:.0%}
- Missed Red-Flag Diagnoses: {red_flag_misses}

TREATMENT PLAN & GUIDELINES:
- Guideline Adherence Score: {guideline_adherence:.0%}
- Safety Score: {safety_score:.0%}
- Safety Check Failures: {safety_failures}
- Critical Safety Issues: {critical_issues}

EVIDENCE CITATIONS AVAILABLE:
{evidence_citations}

Respond with a JSON object. ALWAYS include specific percentages and evidence IDs when making claims:
{{
    "overall_summary": "2-3 sentence summary citing key scores (e.g., 'achieved 85% guideline adherence')",
    "overall_rating": "excellent|good|fair|needs_improvement|concerning",
    "strengths": "What the doctor did well WITH EVIDENCE (e.g., 'captured 95% of symptoms discussed')",
    "concerns": "Areas of concern WITH EVIDENCE (e.g., 'only 60% guideline adherence - see guidelines:42'). Say 'None identified' if excellent",
    "documentation_quality": "Assessment with coverage percentages (e.g., 'captured 90% of key findings')",
    "diagnostic_reasoning": "Assessment citing coherence score and any missed diagnoses",
    "treatment_plan": "Assessment citing guideline adherence score and specific guideline IDs",
    "safety_assessment": "Safety findings with specific check results",
    "recommendations": "If issues found, MUST include: 'Run 2nd Opinion AI diagnostic analysis for a comprehensive second opinion.' Plus other actionable items.",
    "full_narrative": "Complete 2-3 paragraph explanation with embedded evidence citations and scores throughout"
}}

CRITICAL: If overall_rating is 'needs_improvement' or 'concerning', recommendations MUST suggest running the 2nd Opinion AI diagnostic analysis.

Return ONLY the JSON object, no other text."""


# ============================================================
# Critique Explainer
# ============================================================

class CritiqueExplainer:
    """
    Generates natural language explanations of critique results.
    
    Uses an LLM to translate structured critique data into
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
    
    def explain(self, critique: NoteCritique) -> CritiqueExplanation:
        """
        Generate a natural language explanation of a critique.
        
        Args:
            critique: NoteCritique object to explain
        
        Returns:
            CritiqueExplanation with human-readable content
        """
        # Format the critique data for the prompt
        prompt_data = self._format_critique_for_prompt(critique)
        
        # Build the prompt
        user_prompt = EXPLANATION_USER_PROMPT.format(**prompt_data)
        
        messages = [
            {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
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
            
            return CritiqueExplanation(
                overall_summary=self._ensure_string(data.get("overall_summary", "Unable to generate summary.")),
                overall_rating=self._ensure_string(data.get("overall_rating", "fair")),
                strengths=self._ensure_string(data.get("strengths", "Not available.")),
                concerns=self._ensure_string(data.get("concerns", "Not available.")),
                documentation_quality=self._ensure_string(data.get("documentation_quality", "Not available.")),
                diagnostic_reasoning=self._ensure_string(data.get("diagnostic_reasoning", "Not available.")),
                treatment_plan=self._ensure_string(data.get("treatment_plan", "Not available.")),
                safety_assessment=self._ensure_string(data.get("safety_assessment", "Not available.")),
                recommendations=self._ensure_string(data.get("recommendations", "Please discuss any concerns with your healthcare provider.")),
                full_narrative=self._ensure_string(data.get("full_narrative", data.get("overall_summary", "Unable to generate explanation.")))
            )
            
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return self._generate_fallback_explanation(critique)
    
    def _format_critique_for_prompt(self, critique: NoteCritique) -> Dict[str, Any]:
        """Format critique data for the prompt template."""
        # Simplify the critique JSON (remove verbose nested structures)
        simplified = {
            "encounter_id": critique.encounter_id,
            "overall_score": critique.overall_score,
            "overall_verdict": critique.overall_verdict,
            "overall_safety": critique.overall_safety,
            "consistency_verdict": critique.source_note_consistency.verdict,
            "coverage_verdict": critique.coverage_of_salient_findings.verdict,
            "assessment_verdict": critique.assessment_quality.verdict,
            "plan_verdict": critique.plan_quality_and_safety.verdict
        }
        
        # Add omissions if any
        omissions = critique.coverage_of_salient_findings.omissions
        if omissions:
            simplified["notable_omissions"] = [
                {"type": o.finding_type, "finding": o.finding_text, "importance": o.importance}
                for o in omissions[:5]  # Limit to top 5
            ]
        
        # Add missed differentials if any
        missed_dx = critique.assessment_quality.missed_differentials
        if missed_dx:
            simplified["missed_diagnoses"] = [
                {"condition": m.condition, "is_red_flag": m.is_red_flag}
                for m in missed_dx[:3]
            ]
        
        # Add safety issues if any
        safety_issues = [
            s for s in critique.plan_quality_and_safety.safety_checks
            if s.status in ("warning", "fail")
        ]
        if safety_issues:
            simplified["safety_concerns"] = [
                {"type": s.check_type, "description": s.description, "severity": s.severity}
                for s in safety_issues[:3]
            ]
        
        # Build evidence citations section
        evidence_lines = []
        
        # Guideline citations from plan checks
        guideline_ids = set()
        for check in critique.plan_quality_and_safety.guideline_checks:
            for citation in check.guideline_citations:
                if citation.evidence_id not in guideline_ids:
                    guideline_ids.add(citation.evidence_id)
                    evidence_lines.append(
                        f"- {citation.evidence_id}: {citation.guideline_name} - {citation.section}"
                    )
        
        # Guideline citations from diagnosis evaluations
        for dx_eval in critique.assessment_quality.diagnosis_evaluations:
            for citation in dx_eval.guideline_support:
                if citation.evidence_id not in guideline_ids:
                    guideline_ids.add(citation.evidence_id)
                    evidence_lines.append(
                        f"- {citation.evidence_id}: {citation.guideline_name} - {citation.section}"
                    )
        
        # Dialogue evidence from claim judgments (summarize)
        entailed_claims = [j for j in critique.source_note_consistency.claim_judgments if j.verdict == "entailed"]
        contradicted_claims = [j for j in critique.source_note_consistency.claim_judgments if j.verdict == "contradicted"]
        unsupported_claims = [j for j in critique.source_note_consistency.claim_judgments if j.verdict == "unsupported"]
        
        if entailed_claims:
            evidence_lines.append(f"- {len(entailed_claims)} claims supported by dialogue (dialogue evidence available)")
        if contradicted_claims:
            evidence_lines.append(f"- {len(contradicted_claims)} claims CONTRADICTED by dialogue")
            for claim in contradicted_claims[:2]:
                evidence_lines.append(f"  * '{claim.claim_text[:50]}...' contradicts dialogue")
        if unsupported_claims:
            evidence_lines.append(f"- {len(unsupported_claims)} claims with NO supporting evidence in dialogue")
        
        # Missed differentials evidence
        for missed in critique.assessment_quality.missed_differentials:
            flag_text = " [RED FLAG]" if missed.is_red_flag else ""
            evidence_lines.append(f"- Missed diagnosis{flag_text}: {missed.condition} - {missed.reasoning[:60]}")
        
        # Safety check results
        for check in critique.plan_quality_and_safety.safety_checks:
            if check.status != "pass":
                evidence_lines.append(f"- Safety {check.status.upper()}: {check.check_type} - {check.description}")
        
        evidence_citations = "\n".join(evidence_lines) if evidence_lines else "No specific evidence citations available."
        
        return {
            "critique_json": json.dumps(simplified, indent=2),
            "overall_score": critique.overall_score,
            "overall_verdict": critique.overall_verdict,
            "overall_safety": critique.overall_safety,
            "support_score": critique.source_note_consistency.support_score,
            "hallucination_rate": critique.source_note_consistency.hallucination_density,
            "contradictions": f"{len(contradicted_claims)} found" if contradicted_claims else "None found",
            "symptom_recall": critique.coverage_of_salient_findings.symptom_recall,
            "med_recall": critique.coverage_of_salient_findings.medication_recall,
            "allergy_recall": critique.coverage_of_salient_findings.allergy_recall,
            "critical_omissions": critique.coverage_of_salient_findings.critical_omissions,
            "dx_coherence": critique.assessment_quality.dx_coherence_score,
            "dx_breadth": critique.assessment_quality.differential_breadth_score,
            "red_flag_misses": critique.assessment_quality.red_flag_misses,
            "guideline_adherence": critique.plan_quality_and_safety.guideline_adherence_score,
            "safety_score": critique.plan_quality_and_safety.safety_score,
            "safety_failures": critique.plan_quality_and_safety.safety_failures,
            "critical_issues": critique.plan_quality_and_safety.critical_safety_issues,
            "evidence_citations": evidence_citations
        }
    
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
    
    def _ensure_string(self, value) -> str:
        """Ensure a value is a string, converting lists/dicts as needed."""
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Join list items with newlines, handling nested items
            items = []
            for item in value:
                if isinstance(item, str):
                    items.append(f"• {item}")
                elif isinstance(item, dict):
                    # Handle dict items (e.g., {"point": "text"})
                    text = item.get("point") or item.get("text") or str(item)
                    items.append(f"• {text}")
                else:
                    items.append(f"• {str(item)}")
            return "\n".join(items)
        if isinstance(value, dict):
            return str(value)
        return str(value)
    
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
        
        # Clean control characters and other JSON issues
        content = clean_json_string(content)
        
        return json.loads(content)
    
    def _generate_fallback_explanation(self, critique: NoteCritique) -> CritiqueExplanation:
        """Generate a basic explanation without LLM if the call fails."""
        # Determine rating based on scores
        score = critique.overall_score
        if score >= 0.85:
            rating = "excellent"
        elif score >= 0.7:
            rating = "good"
        elif score >= 0.55:
            rating = "fair"
        elif score >= 0.4:
            rating = "needs_improvement"
        else:
            rating = "concerning"
        
        # Build basic explanations with evidence
        support_pct = critique.source_note_consistency.support_score
        coverage_pct = critique.coverage_of_salient_findings.completeness_score
        dx_pct = critique.assessment_quality.dx_coherence_score
        guideline_pct = critique.plan_quality_and_safety.guideline_adherence_score
        safety_pct = critique.plan_quality_and_safety.safety_score
        
        summary = (
            f"The clinical documentation received an overall score of {score:.0%}, "
            f"with {guideline_pct:.0%} guideline adherence and {support_pct:.0%} consistency with your conversation."
        )
        
        strengths = []
        concerns = []
        
        if support_pct >= 0.7:
            strengths.append(f"The note accurately reflects {support_pct:.0%} of what was discussed in your conversation.")
        else:
            concerns.append(f"Only {support_pct:.0%} of the note content was clearly supported by the conversation.")
        
        if coverage_pct >= 0.7:
            strengths.append(f"Important findings were well documented ({coverage_pct:.0%} coverage).")
        else:
            concerns.append(f"Documentation completeness was {coverage_pct:.0%} - some details may have been missed.")
        
        if dx_pct >= 0.7:
            strengths.append(f"Diagnostic reasoning showed {dx_pct:.0%} coherence with your symptoms.")
        else:
            concerns.append(f"Diagnostic coherence score was {dx_pct:.0%} - reasoning could be more thorough.")
        
        if safety_pct >= 0.8:
            strengths.append(f"The treatment plan achieved a {safety_pct:.0%} safety score.")
        else:
            concerns.append(f"Safety score was {safety_pct:.0%} - there may be considerations to discuss.")
        
        strengths_text = " ".join(strengths) if strengths else "The documentation covers the basic elements of your visit."
        concerns_text = " ".join(concerns) if concerns else "No significant concerns were identified."
        
        # Safety assessment with evidence
        if critique.overall_safety == "safe":
            safety = f"No safety concerns identified. Safety score: {safety_pct:.0%}."
        elif critique.overall_safety == "needs_review":
            safety = f"Safety score: {safety_pct:.0%}. Some aspects may benefit from additional review."
        else:
            safety = f"Safety score: {safety_pct:.0%}. There are safety considerations that should be addressed."
        
        # Recommendations with 2nd Opinion AI promotion if needed
        if rating in ("needs_improvement", "concerning"):
            recommendations = (
                "• Run 2nd Opinion AI diagnostic analysis for a comprehensive second opinion\n"
                "• Discuss the identified concerns with your healthcare provider\n"
                "• Ask your doctor about the diagnostic alternatives mentioned"
            )
        elif rating == "fair":
            recommendations = (
                "• Consider running 2nd Opinion AI diagnostic analysis to explore alternatives\n"
                "• Discuss any questions with your healthcare provider"
            )
        else:
            recommendations = "• Your documentation looks good. Discuss any remaining questions with your healthcare provider."
        
        full_narrative = (
            f"{summary} "
            f"{strengths_text} "
            f"{concerns_text} "
            f"{safety} "
            f"{'We recommend running the 2nd Opinion AI diagnostic analysis to get a comprehensive second opinion on your case.' if rating in ('needs_improvement', 'concerning') else ''}"
        )
        
        return CritiqueExplanation(
            overall_summary=summary,
            overall_rating=rating,
            strengths=strengths_text,
            concerns=concerns_text,
            documentation_quality=f"Documentation captured {coverage_pct:.0%} of key findings from your conversation.",
            diagnostic_reasoning=f"Diagnostic coherence score: {dx_pct:.0%}. Differential breadth: {critique.assessment_quality.differential_breadth_score:.0%}.",
            treatment_plan=f"Guideline adherence: {guideline_pct:.0%}. {critique.plan_quality_and_safety.guideline_deviations} deviations noted.",
            safety_assessment=safety,
            recommendations=recommendations,
            full_narrative=full_narrative.strip()
        )


# ============================================================
# Convenience Function
# ============================================================

def explain_critique(
    critique: NoteCritique,
    llm_client,
    llm_model: str
) -> CritiqueExplanation:
    """
    Generate a natural language explanation of a critique.
    
    Args:
        critique: NoteCritique to explain
        llm_client: LLM client
        llm_model: Model name
    
    Returns:
        CritiqueExplanation with human-readable content
    """
    explainer = CritiqueExplainer(llm_client, llm_model)
    return explainer.explain(critique)


def _safe_str(value) -> str:
    """Safely convert any value to string for formatting."""
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


def format_explanation_text(explanation: CritiqueExplanation) -> str:
    """
    Format an explanation as readable text.
    
    Args:
        explanation: CritiqueExplanation object
    
    Returns:
        Formatted string for display
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append("SECOND OPINION: CLINICAL DOCUMENTATION REVIEW")
    lines.append("=" * 70)
    lines.append("")
    
    # Overall rating badge
    rating = _safe_str(explanation.overall_rating)
    rating_emoji = {
        "excellent": "🌟",
        "good": "✅",
        "fair": "⚠️",
        "needs_improvement": "🔶",
        "concerning": "🔴"
    }.get(rating.lower(), "•")
    
    lines.append(f"{rating_emoji} OVERALL RATING: {rating.upper()}")
    lines.append("")
    lines.append(_safe_str(explanation.overall_summary))
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("WHAT YOUR DOCTOR DID WELL:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.strengths))
    lines.append("")
    
    concerns = _safe_str(explanation.concerns)
    if concerns and concerns.lower() != "none identified":
        lines.append("-" * 70)
        lines.append("AREAS FOR ATTENTION:")
        lines.append("-" * 70)
        lines.append(concerns)
        lines.append("")
    
    lines.append("-" * 70)
    lines.append("DETAILED ASSESSMENT:")
    lines.append("-" * 70)
    lines.append(f"📋 Documentation: {_safe_str(explanation.documentation_quality)}")
    lines.append(f"🔍 Diagnosis: {_safe_str(explanation.diagnostic_reasoning)}")
    lines.append(f"💊 Treatment Plan: {_safe_str(explanation.treatment_plan)}")
    lines.append(f"🛡️ Safety: {_safe_str(explanation.safety_assessment)}")
    lines.append("")
    
    lines.append("-" * 70)
    lines.append("RECOMMENDATIONS:")
    lines.append("-" * 70)
    lines.append(_safe_str(explanation.recommendations))
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
    
    parser = argparse.ArgumentParser(description="Explain a critique file")
    parser.add_argument("--critique", "-c", type=str, required=True,
                        help="Path to critique JSON file")
    parser.add_argument("--mode", "-m", type=str, default="test",
                        choices=["test", "prod"],
                        help="LLM mode")
    
    args = parser.parse_args()
    
    # Load critique
    critique_path = Path(args.critique)
    if not critique_path.exists():
        print(f"Critique file not found: {critique_path}")
        sys.exit(1)
    
    with critique_path.open("r") as f:
        critique_data = json.load(f)
    
    critique = NoteCritique(**critique_data)
    
    # Set up LLM
    try:
        from core.llm_config import LLMConfig
        config = LLMConfig(mode=args.mode)
        print(f"Using LLM mode: {args.mode}")
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        sys.exit(1)
    
    # Generate explanation
    print("\nGenerating explanation...")
    explanation = explain_critique(critique, config.client, config.diagnoser_model)
    
    # Print formatted
    print(format_explanation_text(explanation))
