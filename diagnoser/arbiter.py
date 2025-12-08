# diagnoser/arbiter.py
# ==============================
# Rule-based Arbiter that applies safe patches from Consultant critique
# ==============================

import copy
from typing import List, Tuple

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnoser.schema import (
    DiagnoserOutput, ConsultantCritique, CritiqueItem, ArbiterResult
)


def _parse_json_pointer(pointer: str) -> List[str]:
    """
    Parse a JSON pointer string into path segments.
    
    Args:
        pointer: JSON pointer like '/management_plan/pharm/0'
    
    Returns:
        List of path segments ['management_plan', 'pharm', '0']
    """
    if not pointer:
        return []
    
    # Remove leading slash if present
    if pointer.startswith('/'):
        pointer = pointer[1:]
    
    if not pointer:
        return []
    
    return pointer.split('/')


def _get_by_pointer(obj: dict, pointer: str):
    """
    Get a value from a dict using a JSON pointer.
    
    Args:
        obj: The dictionary to traverse.
        pointer: JSON pointer string.
    
    Returns:
        The value at the pointer location, or None if not found.
    """
    segments = _parse_json_pointer(pointer)
    current = obj
    
    for seg in segments:
        if isinstance(current, dict):
            current = current.get(seg)
        elif isinstance(current, list):
            try:
                idx = int(seg)
                current = current[idx] if 0 <= idx < len(current) else None
            except (ValueError, IndexError):
                return None
        else:
            return None
        
        if current is None:
            return None
    
    return current


def _remove_by_pointer(obj: dict, pointer: str) -> bool:
    """
    Remove an item from a dict/list using a JSON pointer.
    
    Args:
        obj: The dictionary to modify.
        pointer: JSON pointer string.
    
    Returns:
        True if removal was successful, False otherwise.
    """
    segments = _parse_json_pointer(pointer)
    if not segments:
        return False
    
    # Navigate to parent
    current = obj
    for seg in segments[:-1]:
        if isinstance(current, dict):
            current = current.get(seg)
        elif isinstance(current, list):
            try:
                idx = int(seg)
                current = current[idx]
            except (ValueError, IndexError):
                return False
        else:
            return False
        
        if current is None:
            return False
    
    # Remove the final element
    final_seg = segments[-1]
    if isinstance(current, dict):
        if final_seg in current:
            del current[final_seg]
            return True
    elif isinstance(current, list):
        try:
            idx = int(final_seg)
            if 0 <= idx < len(current):
                current.pop(idx)
                return True
        except (ValueError, IndexError):
            pass
    
    return False


def apply_critique(
    diagnoser: DiagnoserOutput,
    critique: ConsultantCritique
) -> ArbiterResult:
    """
    Apply Consultant's critique to the Diagnoser's output.
    
    Rules:
    1. Any CritiqueItem with severity="critical" -> abstain entirely
    2. Items with kind in (unsafe_recommendation, guideline_conflict, dosing_error, 
       interaction_missed) and severity >= "high" -> remove or flag the entry
    3. Other items -> log as unresolved but don't block
    
    Args:
        diagnoser: Original DiagnoserOutput from the Diagnoser.
        critique: ConsultantCritique with identified issues.
    
    Returns:
        ArbiterResult with patched plan or abstention.
    """
    # Check for critical issues first
    critical_issues = [
        issue for issue in critique.issues 
        if issue.severity == "critical"
    ]
    
    if critical_issues:
        # Abstain - cannot safely proceed
        critical_descriptions = [f"- {issue.description}" for issue in critical_issues]
        notes = (
            f"Cannot generate safe clinical plan due to {len(critical_issues)} critical issue(s):\n" +
            "\n".join(critical_descriptions) +
            "\n\nHuman review required."
        )
        return ArbiterResult(
            final_plan=None,
            abstained=True,
            patches_applied=[],
            notes_on_missing_info=notes,
            unresolved_issues=[issue.description for issue in critical_issues]
        )
    
    # If overall safety rating is "unsafe", also abstain
    if critique.overall_safety_rating == "unsafe":
        return ArbiterResult(
            final_plan=None,
            abstained=True,
            patches_applied=[],
            notes_on_missing_info=f"Consultant rated plan as unsafe: {critique.overall_assessment}",
            unresolved_issues=[issue.description for issue in critique.issues]
        )
    
    # Make a copy of the diagnoser output to modify
    plan_dict = diagnoser.model_dump()
    
    patches_applied = []
    unresolved_issues = []
    
    # Categorize issues
    actionable_kinds = {
        "unsafe_recommendation",
        "guideline_conflict", 
        "dosing_error",
        "interaction_missed"
    }
    
    for issue in critique.issues:
        if issue.severity in ("high", "critical") and issue.kind in actionable_kinds:
            # Try to remove the problematic entry
            if issue.target_path:
                removed = _remove_by_pointer(plan_dict, issue.target_path)
                if removed:
                    patches_applied.append(
                        f"Removed {issue.target_path}: {issue.description}"
                    )
                else:
                    # Couldn't remove, add to unresolved
                    unresolved_issues.append(
                        f"Could not auto-fix: {issue.description} at {issue.target_path}"
                    )
            else:
                # No target path, can't auto-fix
                unresolved_issues.append(
                    f"No target path for: {issue.description}"
                )
        elif issue.severity == "high":
            # High severity but not an actionable kind
            unresolved_issues.append(
                f"Needs manual review ({issue.kind}): {issue.description}"
            )
        else:
            # Moderate or low severity - just note it
            unresolved_issues.append(
                f"Minor issue ({issue.severity}): {issue.description}"
            )
    
    # Reconstruct the DiagnoserOutput from the modified dict
    try:
        final_plan = DiagnoserOutput.model_validate(plan_dict)
    except Exception as e:
        # If reconstruction fails, abstain
        return ArbiterResult(
            final_plan=None,
            abstained=True,
            patches_applied=patches_applied,
            notes_on_missing_info=f"Failed to reconstruct plan after patches: {str(e)}",
            unresolved_issues=unresolved_issues
        )
    
    # Add arbiter's notes about what was modified
    if patches_applied:
        existing_notes = final_plan.notes_on_missing_info or ""
        arbiter_notes = "Arbiter modifications:\n" + "\n".join(f"- {p}" for p in patches_applied)
        final_plan.notes_on_missing_info = (
            (existing_notes + "\n\n" if existing_notes else "") + arbiter_notes
        )
    
    return ArbiterResult(
        final_plan=final_plan,
        abstained=False,
        patches_applied=patches_applied,
        notes_on_missing_info=None,
        unresolved_issues=unresolved_issues
    )


def summarize_arbiter_result(result: ArbiterResult) -> str:
    """
    Generate a human-readable summary of the Arbiter result.
    
    Args:
        result: ArbiterResult to summarize.
    
    Returns:
        Formatted string summary.
    """
    lines = []
    
    if result.abstained:
        lines.append("ARBITER ABSTAINED - Human review required")
        if result.notes_on_missing_info:
            lines.append(f"\nReason: {result.notes_on_missing_info}")
    else:
        lines.append("Clinical plan approved")
        
        if result.patches_applied:
            lines.append(f"\nPatches applied ({len(result.patches_applied)}):")
            for patch in result.patches_applied:
                lines.append(f"  - {patch}")
    
    if result.unresolved_issues:
        lines.append(f"\nUnresolved issues ({len(result.unresolved_issues)}):")
        for issue in result.unresolved_issues:
            lines.append(f"  - {issue}")
    
    return "\n".join(lines)

