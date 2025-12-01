# extractor/qa_pairing.py (drop-in replacement for the detectors at top)
from typing import List, Tuple, Dict, Optional

QUESTION_MARK = "?"
QUESTION_CUES = ["do you","are you","have you","when","where","how","why","did you","does it","is there","can you","would you","any "]
BACKCHANNELS = {"uh","uh-huh","mm","mm-hmm","okay","ok","alright","right","got it","yep","yeah","yup","hmm","huh","mmm","sure"}

def _sentence_split(text: str) -> List[str]:
    # naive split on '?' keeping order; good enough for ASR-punctuated lines
    parts = []
    buf = ""
    for ch in text:
        buf += ch
        if ch == "?":
            parts.append(buf.strip())
            buf = ""
    if buf.strip():
        parts.append(buf.strip())
    return parts

def _is_question(text: str) -> bool:
    t = text.lower()
    if QUESTION_MARK in text:
        return True
    return any(c in t for c in QUESTION_CUES)

def _is_backchannel(text: str) -> bool:
    t = text.strip().lower().strip(".!,?")
    return (t in BACKCHANNELS) or (len(t.split()) <= 2 and t in {"yes","no"})

def pair_questions_and_answers(dialogue: List[Dict], answer_window: int = 3) -> List[Tuple[Dict, Dict]]:
    """
    Uses '?' to identify questions; splits multi-question turns; pairs with next meaningful PATIENT reply
    within `answer_window` patient turns; allows 'yes'/'no' as valid short answers, skips other fillers.
    """
    # Expand doctor turns with multiple '?' into pseudo-turns so each Q pairs cleanly
    expanded = []
    for turn in dialogue:
        if turn["role"].upper() == "DOCTOR" and _is_question(turn["text"]):
            for q in _sentence_split(turn["text"]):
                if _is_question(q):
                    expanded.append({"utt_id": turn["utt_id"], "role": "DOCTOR", "text": q})
        else:
            expanded.append(turn)

    pairs: List[Tuple[Dict, Dict]] = []
    n = len(expanded)
    for i, t in enumerate(expanded):
        if t["role"].upper() != "DOCTOR" or not _is_question(t["text"]):
            continue
        # look ahead for patient answer
        found: Optional[Dict] = None
        hops = 0
        j = i + 1
        while j < n and hops < answer_window:
            if expanded[j]["role"].upper() == "PATIENT":
                ans = expanded[j]["text"].strip()
                # keep yes/no; skip other short fillers
                if _is_backchannel(ans) and ans.lower() not in {"yes","no"}:
                    hops += 1; j += 1; continue
                found = expanded[j]; break
            j += 1
        if found:
            pairs.append((t, found))
    return pairs
