from typing import List, Iterable
from .schema import Symptom, Medication, Allergy, Vital, QAExtracted

def _dedupe(items: Iterable, keyfunc):
    seen = set(); out = []
    for it in items:
        k = keyfunc(it)
        if k in seen:
            # merge evidence
            for o in out:
                if keyfunc(o) == k:
                    o.evidence.utt_ids = sorted(set(o.evidence.utt_ids + it.evidence.utt_ids))
                    break
        else:
            seen.add(k); out.append(it)
    return out

class Merger:
    def __call__(self, ej, symptoms: List[Symptom], meds: List[Medication], vitals: List[Vital], allergies: List[Allergy], qa: List[QAExtracted]):
        # remove meds that are actually allergy substances
        allergy_names = {a.substance_norm or a.substance_surface for a in allergies}
        meds = [m for m in meds if (m.name_norm or m.name_surface) not in allergy_names]
        ej.symptoms = _dedupe(symptoms, lambda s: (s.name_norm or s.name_surface, s.assertion, s.duration))
        ej.meds = _dedupe(meds, lambda m: (m.name_norm or m.name_surface, m.dose, m.freq, m.route))
        ej.vitals = _dedupe(vitals, lambda v: (v.kind, v.value))
        ej.allergies= _dedupe(allergies,lambda a: (a.substance_norm or a.substance_surface, a.reaction))
        ej.qa_extractions = _dedupe(qa, lambda q: (q.concept, q.value, q.assertion))
        return ej