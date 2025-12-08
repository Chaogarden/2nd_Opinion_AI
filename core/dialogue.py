# core/dialogue.py
# ==============================
# Dialogue normalization and role-mapping utilities
# Moved from app/main.py for reuse across the pipeline
# ==============================

import re
from collections import defaultdict
from typing import List, Dict, Optional

# ---------------------------
# Constants for role heuristics
# ---------------------------
DOC_KEYWORDS = [
    "assessment", "plan", "follow up", "follow-up", "differential",
    "start you on", "i recommend", "i'll prescribe", "i will prescribe",
    "prescribe", "dose", "mg", "b.i.d", "bid", "t.i.d", "tid", "q.d", "qd",
    "referral", "imaging", "x-ray", "ct", "mri", "ultrasound",
    "labs", "cbc", "cmp", "a1c", "lipid", "troponin",
    "blood pressure", "bp", "heart rate", "hr", "respiratory rate", "rr", "spo2", "oxygen",
    "physical exam", "on exam", "exam", "auscultation", "palpation",
    "contraindicated", "indicated", "guideline", "screening", "uspstf", "nice"
]
DOC_IMPERATIVES = [
    "take", "start", "stop", "begin", "increase", "decrease", "schedule",
    "follow", "avoid", "return", "monitor", "check", "continue"
]
DOSAGE_RE = re.compile(r"\b\d+(\.\d+)?\s*mg\b", re.IGNORECASE)


def score_doctorish(text: str) -> int:
    """Score how likely a text segment is from a doctor."""
    t = text.lower()
    score = 0
    for kw in DOC_KEYWORDS:
        if kw in t:
            score += 2
    for imp in DOC_IMPERATIVES:
        if re.search(rf"(?:^|[\.!\?]\s+)({imp})\b", t):
            score += 2
    if DOSAGE_RE.search(t):
        score += 3
    if "we'll" in t or "let's" in t or "i recommend" in t:
        score += 1
    return score


def infer_role_mapping(segments: List[Dict]) -> Dict[str, str]:
    """
    Returns a dict: { 'SPEAKER_00': 'DOCTOR', 'SPEAKER_01': 'PATIENT', ... }
    Heuristic: speaker with highest 'doctorish' score -> DOCTOR. Others -> PATIENT/OTHER_i.
    """
    per_spk_score = defaultdict(int)
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        per_spk_score[spk] += score_doctorish(seg.get("text", ""))

    if not per_spk_score:
        return {}

    ranked = sorted(per_spk_score.items(), key=lambda x: x[1], reverse=True)
    mapping = {}
    if ranked:
        mapping[ranked[0][0]] = "DOCTOR"
        others = [spk for spk, _ in ranked[1:]]
        if others:
            mapping[others[0]] = "PATIENT"
            for i, spk in enumerate(others[1:], start=1):
                mapping[spk] = f"OTHER_{i}"
    return mapping


def apply_role_mapping(segments: List[Dict], mapping: Dict[str, str]) -> List[Dict]:
    """Append 'role' to segments; if unknown, keep original speaker id."""
    out = []
    for seg in segments:
        spk = seg.get("speaker")
        role = mapping.get(spk, spk or "UNKNOWN")
        out.append({**seg, "role": role})
    return out


def swap_doctor_patient(mapping: Dict[str, str]) -> Dict[str, str]:
    """Swap DOCTOR and PATIENT roles in an existing mapping."""
    swapped = {}
    for k, v in mapping.items():
        if v == "DOCTOR":
            swapped[k] = "PATIENT"
        elif v == "PATIENT":
            swapped[k] = "DOCTOR"
        else:
            swapped[k] = v
    return swapped


def build_dialogue(
    segments: List[Dict],
    mapping: Optional[Dict[str, str]] = None,
    auto_infer_roles: bool = True
) -> List[Dict]:
    """
    Convert ASR segments into a dialogue format with sequential utt_ids.
    
    Each turn in the output will have:
        - utt_id: int (sequential, starting at 0)
        - role: str ("DOCTOR", "PATIENT", or other)
        - speaker: str (original speaker ID from diarization)
        - start: float (timestamp)
        - end: float (timestamp)
        - text: str
    
    Args:
        segments: List of ASR segment dicts with at least 'text' and optionally
                  'speaker', 'start', 'end'.
        mapping: Optional pre-computed speaker->role mapping. If None and 
                 auto_infer_roles is True, we infer it.
        auto_infer_roles: If True and no mapping is provided, infer roles from
                          the segments using heuristics.
    
    Returns:
        List of dialogue turn dicts with utt_id assigned.
    """
    if not segments:
        return []
    
    # Infer mapping if needed
    if mapping is None and auto_infer_roles:
        mapping = infer_role_mapping(segments)
    elif mapping is None:
        mapping = {}
    
    dialogue = []
    for i, seg in enumerate(segments):
        speaker = seg.get("speaker", "UNKNOWN")
        role = mapping.get(speaker, speaker)
        
        turn = {
            "utt_id": i,
            "role": role,
            "speaker": speaker,
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        }
        dialogue.append(turn)
    
    return dialogue


def build_dialogue_from_text(text: str, default_role: str = "PATIENT") -> List[Dict]:
    """
    Create a dialogue from free-form text input.
    Used when the user pastes text instead of uploading audio.
    
    Parses text with role labels like:
    - "Doctor: ..." / "Patient: ..."
    - "[doctor] ..." / "[patient] ..."
    
    Handles both line-based and inline labels.
    If no labels are found, treats entire text as a single turn with default_role.
    
    Args:
        text: The free-form text input.
        default_role: The role to assign if no labels found.
    
    Returns:
        List of dialogue turn dicts.
    """
    if not text.strip():
        return []
    
    import re
    
    # Pattern 1: Bracket format [doctor] or [patient] - can be anywhere
    bracket_pattern = re.compile(r'\[(?P<role>doctor|patient)\]', re.IGNORECASE)
    
    # Pattern 2: Colon format "Doctor:" or "Patient:" - typically at line start or after punctuation
    colon_pattern = re.compile(r'(?:^|(?<=\n)|(?<=\.)\s*|(?<=\?)\s*|(?<=!)\s*)(?P<role>Doctor|Patient)\s*:', re.IGNORECASE | re.MULTILINE)
    
    # Try bracket format first (ACI-Bench style)
    bracket_matches = list(bracket_pattern.finditer(text))
    if bracket_matches:
        dialogue = []
        for i, match in enumerate(bracket_matches):
            role = match.group("role").upper()
            
            # Get text between this match and the next
            start_pos = match.end()
            end_pos = bracket_matches[i + 1].start() if i + 1 < len(bracket_matches) else len(text)
            
            turn_text = text[start_pos:end_pos].strip()
            
            if turn_text:
                dialogue.append({
                    "utt_id": len(dialogue),
                    "role": role,
                    "speaker": role,
                    "start": 0.0,
                    "end": 0.0,
                    "text": turn_text,
                })
        
        if dialogue:
            return dialogue
    
    # Try colon format (Doctor: / Patient:)
    colon_matches = list(colon_pattern.finditer(text))
    if colon_matches:
        dialogue = []
        for i, match in enumerate(colon_matches):
            role = match.group("role").upper()
            
            start_pos = match.end()
            end_pos = colon_matches[i + 1].start() if i + 1 < len(colon_matches) else len(text)
            
            turn_text = text[start_pos:end_pos].strip()
            
            if turn_text:
                dialogue.append({
                    "utt_id": len(dialogue),
                    "role": role,
                    "speaker": role,
                    "start": 0.0,
                    "end": 0.0,
                    "text": turn_text,
                })
        
        if dialogue:
            return dialogue
    
    # No labels found - treat as single turn
    return [{
        "utt_id": 0,
        "role": default_role,
        "speaker": default_role,
        "start": 0.0,
        "end": 0.0,
        "text": text.strip(),
    }]


def build_dialogue_brief(dialogue: List[Dict], max_chars: int = 2000) -> str:
    """
    Create a concise dialogue summary for the LLM context.
    
    Concatenates key DOCTOR/PATIENT turns into a ~1–2 paragraph summary,
    prioritizing:
    - First few patient turns (chief complaint area)
    - Doctor assessment-like segments
    - Key exchanges
    
    Args:
        dialogue: List of dialogue turns with 'role' and 'text'.
        max_chars: Maximum character count for the brief.
    
    Returns:
        A string summary of the dialogue.
    """
    if not dialogue:
        return ""
    
    lines = []
    char_count = 0
    
    # Prioritize first patient turn (chief complaint)
    patient_turns = [t for t in dialogue if t.get("role", "").upper() == "PATIENT"]
    doctor_turns = [t for t in dialogue if t.get("role", "").upper() == "DOCTOR"]
    
    # Add first patient turn if available
    if patient_turns:
        first_patient = f"PATIENT: {patient_turns[0]['text']}"
        lines.append(first_patient)
        char_count += len(first_patient)
    
    # Interleave remaining turns
    remaining_budget = max_chars - char_count
    for turn in dialogue[1:]:
        role = turn.get("role", "UNKNOWN").upper()
        text = turn.get("text", "").strip()
        if not text:
            continue
        
        line = f"{role}: {text}"
        if char_count + len(line) + 2 > max_chars:
            # Truncate if needed
            available = max_chars - char_count - 10
            if available > 50:
                lines.append(f"{role}: {text[:available]}...")
            break
        
        lines.append(line)
        char_count += len(line) + 2
    
    return "\n".join(lines)

