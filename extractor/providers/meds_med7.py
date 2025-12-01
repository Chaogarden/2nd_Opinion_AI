from typing import List, Dict

from ..normalizers.rxnorm import RxNormNormalizer

import spacy as _sp
from ..schema import Medication, Evidence
from ..patterns import DOSE_RE, FREQ_RE, ROUTE_RE, FORM_RE, PRN_RE
from ..negation import AssertionDetector

# Try loading medspaCy to add ConText to THIS pipeline
try:
    import medspacy
    _MSP_OK = True
except Exception:
    _MSP_OK = False

# Cue packs for med state changes (simple heuristics)
_STOP_CUES = {"stopped", "no longer", "quit", "discontinued", "ran out"}
_START_CUES = {"started", "began", "restarted", "refilled"}
_DOSE_UP = {"increased", "upped", "higher dose"}
_DOSE_DOWN = {"decreased", "lowered", "reduced"}

class Med7MedicationExtractor:
    def __init__(self, model: str = "en_core_med7_lg", rxnorm: RxNormNormalizer | None = None):
        self.nlp = _sp.load(model)
        self._has_context = False
        if _MSP_OK:
            try:
                if "medspacy_context" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("medspacy_context", last=True)
                self._has_context = True
            except Exception:
                self._has_context = False
        self.assertion = AssertionDetector()
        self.rxnorm = rxnorm

    def __call__(self, patient_turns: List[Dict]) -> List[Medication]:
        meds: List[Medication] = []
        for u in patient_turns:
            text = u["text"]
            lower = text.lower()
            # Skip allergy declarations in med extractor
            if "allergic to" in lower or lower.strip().startswith("allergic"):
                continue

            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_.lower() not in {"drug", "medication", "brand", "generic", "dosage"}:
                    continue

                sent_text = ent.sent.text

                # Attributes from the sentence only
                dose_m = DOSE_RE.search(sent_text)
                freq_m = FREQ_RE.search(sent_text)
                route_m = ROUTE_RE.search(sent_text)
                form_m = FORM_RE.search(sent_text)

                route = route_m.group(0) if route_m else ("po" if "by mouth" in sent_text.lower() else None)
                dose = dose_m.group(1) if dose_m else None
                freq = freq_m.group(0) if freq_m else None
                form = form_m.group(0) if form_m else None
                prn = True if PRN_RE.search(sent_text) else None

                # Assertion via ConText if available; else sentence fallback
                if self._has_context:
                    if getattr(ent._, "is_negated", False):
                        assertion = "absent"
                    elif getattr(ent._, "is_uncertain", False):
                        assertion = "possible"
                    else:
                        assertion = "present"
                else:
                    assertion = self.assertion.classify(sent_text, ent.text)

                # State-change heuristics into modifiers
                modifiers = []
                l = sent_text.lower()
                if any(c in l for c in _STOP_CUES):
                    assertion = "absent"
                    modifiers.append("stopped")
                    dose = None; freq = None; route = None; form = None
                if any(c in l for c in _START_CUES):
                    modifiers.append("started")
                if any(c in l for c in _DOSE_UP):
                    modifiers.append("dose_increase")
                if any(c in l for c in _DOSE_DOWN):
                    modifiers.append("dose_decrease")

                name_norm = None
                rxcui = None
                if self.rxnorm:
                    rx, gen, score = self.rxnorm.normalize(ent.text)
                    if rx and gen:
                        rxcui = rx
                        name_norm = gen

                m = Medication(
                    name_surface=ent.text,
                    name_norm=name_norm,
                    rxcui=rxcui,
                    dose=dose,
                    freq=freq,
                    route=route,
                    form=form,
                    prn=prn,
                    assertion=assertion,
                    evidence=Evidence(utt_ids=[u["utt_id"]], spans=[{"text": ent.text}]),
                )
                # Store modifiers into Medication.modifiers if you add the field; otherwise ignore
                meds.append(m)
        return meds

    def extract(self, patient_turns: List[Dict]) -> List[Medication]:
        return self.__call__(patient_turns)