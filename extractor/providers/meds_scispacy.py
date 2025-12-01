from typing import List, Dict

from ..normalizers.rxnorm import RxNormNormalizer 
import spacy as _sp2
from ..schema import Medication, Evidence
from ..patterns import DOSE_RE, FREQ_RE, ROUTE_RE, FORM_RE, PRN_RE
from ..negation import AssertionDetector

# Try loading medspaCy to add ConText to THIS pipeline
try:
    import medspacy
    _MSP_OK = True
except Exception:
    _MSP_OK = False

class SciSpaCyMedicationExtractor:
    def __init__(self, model: str = "en_ner_bc5cdr_md", rxnorm: RxNormNormalizer | None = None):
        self.nlp = _sp2.load(model)
        # Add ConText to THIS pipeline so assertion lives on ent._
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
            # If this utterance declares an allergy, skip meds here
            if "allergic to" in lower or lower.strip().startswith("allergic"):
                continue

            doc = self.nlp(text)
            for ent in doc.ents:
                # CHEMICAL works as a broad proxy for meds in bc5cdr
                if ent.label_ not in {"CHEMICAL"}:
                    continue

                sent_text = ent.sent.text # sentence scope
                # Attribute extraction within the sentence only
                # (use regex over the sentence, not cross-sentence windows)
                dose_m = DOSE_RE.search(sent_text)
                freq_m = FREQ_RE.search(sent_text)
                route_m = ROUTE_RE.search(sent_text)
                form_m = FORM_RE.search(sent_text)

                # Map common phrasing → route
                route = route_m.group(0) if route_m else ("po" if "by mouth" in sent_text.lower() else None)
                dose = dose_m.group(1) if dose_m else None
                freq = freq_m.group(0) if freq_m else None
                form = form_m.group(0) if form_m else None
                prn = True if PRN_RE.search(sent_text) else None

                # Assertion via ConText on SAME doc/entity if available; else fallback on sentence
                if self._has_context:
                    if getattr(ent._, "is_negated", False):
                        assertion = "absent"
                    elif getattr(ent._, "is_uncertain", False):
                        assertion = "possible"
                    else:
                        assertion = "present"
                else:
                    assertion = self.assertion.classify(sent_text, ent.text)

                # If absent ("not taking X"/"no longer on X"), drop dose/freq unless explicitly stated
                if assertion == "absent":
                    dose = dose if "last took" in sent_text.lower() else None
                    freq = None
                    route = None
                    form = None

                name_norm = None
                rxcui = None
                if self.rxnorm:
                    rx, gen, score = self.rxnorm.normalize(ent.text)
                    if rx and gen:
                        rxcui = rx
                        name_norm = gen

                meds.append(
                    Medication(
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
                )
        return meds

    def extract(self, patient_turns: List[Dict]) -> List[Medication]:
        return self.__call__(patient_turns)
