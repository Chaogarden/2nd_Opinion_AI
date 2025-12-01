from typing import List, Dict

from ..schema import Symptom, Evidence
from ..patterns import DURATION_RE, SEVERITY_RE, ONSET_RE

try:
    from medcat.cdb import CDB
    from medcat.cat import CAT
    _MEDCAT_OK = True
except Exception:
    _MEDCAT_OK = False

class MedCATSymptomExtractor:
    def __init__(self, cdb_path: str):
        if not _MEDCAT_OK:
            raise RuntimeError("MedCAT not installed. pip install medcat and provide a CDB model.")
        self.cdb = CDB.load(cdb_path)
        self.cat = CAT(cdb=self.cdb)

def __call__(self, patient_turns: List[Dict]) -> List[Symptom]:
    out: List[Symptom] = []
    for u in patient_turns:
        text = u["text"]
        lower = text.lower()
        if "allergic to" in lower or lower.strip().startswith("allergic"):
            continue

        doc = self.cat(text)
        # MedCAT already sets ent.negated; use sentence for attributes
        for ent in getattr(doc, "ents", []):
            sent_text = ent.sent.text if hasattr(ent, "sent") else text
            duration = (DURATION_RE.search(sent_text) or [None, None])[1]
            severity = (SEVERITY_RE.search(sent_text) or [None, None])[1]
            onset = (ONSET_RE.search(sent_text) or [None, None])[1]
            assertion = "absent" if getattr(ent, "negated", False) else "present"
            if assertion == "absent":
                duration = None; severity = None; onset = None

            out.append(
                Symptom(
                    name_surface=ent.text,
                    name_norm=getattr(ent, "cui_name", None),
                    cui=getattr(ent, "cui", None),
                    assertion=assertion,
                    duration=duration,
                    severity=severity,
                    onset=onset,
                    evidence=Evidence(utt_ids=[u["utt_id"]], spans=[{"text": ent.text}]),
                )
            )
    return out

def extract(self, patient_turns: List[Dict]) -> List[Symptom]:
    return self.__call__(patient_turns)
