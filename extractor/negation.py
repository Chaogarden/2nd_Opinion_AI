from typing import Optional

try:
    import medspacy
    _MSP_AVAILABLE = True
except Exception:
    _MSP_AVAILABLE = False

from .patterns import NEG_CUES, UNCERT_CUES


class AssertionDetector:
    def __init__(self):
        self.mode = "medspacy" if _MSP_AVAILABLE else "fallback"
        if _MSP_AVAILABLE:
            self.nlp = medspacy.load()
        else:
            self.nlp = None

    def classify(self, text: str, span_text: Optional[str] = None) -> str:
        """Return 'present' | 'absent' | 'possible'.
        text is assumed to be sentence-scoped now.
        """
        if self.mode == "medspacy":
            doc = self.nlp(text)
            if span_text:
                for ent in doc.ents:
                    if ent.text.lower() == span_text.lower():
                        if getattr(ent._, "is_negated", False):
                            return "absent"
                        if getattr(ent._, "is_uncertain", False):
                            return "possible"
                        return "present"

        # fallback: local window only
        t = text.lower()
        if span_text:
            idx = t.find(span_text.lower())
            if idx != -1:
                window = t[max(0, idx - 25): idx + len(span_text) + 25]
                if any(cue in window for cue in NEG_CUES):
                    return "absent"
                if any(cue in window for cue in UNCERT_CUES):
                    return "possible"
                return "present"

        # ultra fallback
        if any(cue in t for cue in NEG_CUES):
            return "absent"
        if any(cue in t for cue in UNCERT_CUES):
            return "possible"
        return "present"