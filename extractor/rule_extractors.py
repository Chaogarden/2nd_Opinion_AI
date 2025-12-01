from typing import List, Dict
from .patterns import VITALS, RISK_FACTORS
from .schema import Vital, Evidence, Allergy
from .negation import AssertionDetector
import re

ALLERGY_HINT = re.compile(r"allerg(?:y|ic)\s*to\s+([a-zA-Z0-9\- ]+)", re.I)
REACTION_HINT = re.compile(r"\b(rash|hives|swelling|anaphylaxis|itching|SOB)\b", re.I)

class RuleExtras:
    def __init__(self):
        self.assertion = AssertionDetector()

    def extract_vitals(self, patient_turns: List[Dict]) -> List[Vital]:
        out: List[Vital] = []
        for u in patient_turns:
            text = u["text"]
            for kind, rx in VITALS.items():
                m = rx.search(text)
                if m:
                    out.append(Vital(kind=kind, value=m.group(1), evidence=Evidence(utt_ids=[u["utt_id"]])))
        return out

    def extract_allergies(self, patient_turns: List[Dict]) -> List[Allergy]:
        outs: List[Allergy] = []
        for u in patient_turns:
            text = u["text"]
            m = ALLERGY_HINT.search(text)
            if m:
                substance = m.group(1).strip().rstrip(". ,;")
                reaction = (REACTION_HINT.search(text) or [None, None])[1]
                assertion = self.assertion.classify(text, substance)
                outs.append(Allergy(
                    substance_surface=substance,
                    reaction=reaction,
                    assertion=assertion,
                    evidence=Evidence(utt_ids=[u["utt_id"]])
                ))
        return outs

    def extract_risk_factors(self, patient_turns: List[Dict]) -> List[str]:
        found = set()
        for u in patient_turns:
            t = u["text"].lower()
            for rf in RISK_FACTORS:
                if rf in t:
                    found.add(rf)
        return sorted(found)