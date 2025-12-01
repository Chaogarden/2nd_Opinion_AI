from typing import List, Dict

import spacy
from ..schema import Symptom, Evidence
from ..patterns import DURATION_RE, SEVERITY_RE, ONSET_RE
from ..negation import AssertionDetector

# Try loading medspaCy to add ConText to THIS pipeline
try:
    import medspacy
    _MSP_OK = True
except Exception:
    _MSP_OK = False

SYMPTOM_LABELS = {"DISEASE"}  # from en_ner_bc5cdr_md


class SciSpaCySymptomExtractor:
    def __init__(self, model: str = "en_ner_bc5cdr_md"):
        self.nlp = spacy.load(model)

        # ---- CRITICAL FIX: Add medSpaCy ConText to THIS pipeline ----
        self._has_context = False
        if _MSP_OK:
            try:
                if "medspacy_context" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("medspacy_context", last=True)
                self._has_context = True
            except Exception:
                self._has_context = False

        # fallback assertion engine
        self.assertion = AssertionDetector()

    def __call__(self, patient_turns: List[Dict]) -> List[Symptom]:
        out: List[Symptom] = []

        for u in patient_turns:
            text = u["text"]
            lower = text.lower()

            # ignore allergy sentences
            if "allergic to" in lower or lower.strip().startswith("allergic"):
                continue

            doc = self.nlp(text)

            for ent in doc.ents:
                if ent.label_ not in SYMPTOM_LABELS:
                    continue

                sent_text = ent.sent.text  # sentence-level scope

                # duration / severity / onset found ONLY in this sentence
                duration = (DURATION_RE.search(sent_text) or [None, None])[1]
                severity = (SEVERITY_RE.search(sent_text) or [None, None])[1]
                onset = (ONSET_RE.search(sent_text) or [None, None])[1]

                # ---- assertion detection ----
                if self._has_context:
                    if getattr(ent._, "is_negated", False):
                        assertion = "absent"
                    elif getattr(ent._, "is_uncertain", False):
                        assertion = "possible"
                    else:
                        assertion = "present"
                else:
                    # fallback: classify within this sentence only
                    assertion = self.assertion.classify(sent_text, ent.text)

                # If absent, duration and severity shouldn't carry over
                if assertion == "absent":
                    duration = None
                    severity = None
                    onset = None

                out.append(
                    Symptom(
                        name_surface=ent.text,
                        name_norm=None,
                        cui=None,
                        assertion=assertion,
                        duration=duration,
                        severity=severity,
                        onset=onset,
                        evidence=Evidence(
                            utt_ids=[u["utt_id"]],
                            spans=[{"text": ent.text}],
                        ),
                    )
                )

        return out

    def extract(self, patient_turns: List[Dict]) -> List[Symptom]:
        return self.__call__(patient_turns)
