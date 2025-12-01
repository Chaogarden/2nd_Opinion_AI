from typing import Dict, List, Optional
from .normalizers.rxnorm import RxNormNormalizer 
from .schema import ExtractorJSON
from .providers.scispacy_symptoms import SciSpaCySymptomExtractor
from .providers.meds_med7 import Med7MedicationExtractor
from .providers.meds_scispacy import SciSpaCyMedicationExtractor
from .rule_extractors import RuleExtras
from .qa_pairing import pair_questions_and_answers
from .llm_assist import QAInterpreter
from .merge import Merger

class HybridExtractor:
    def __init__(self,
                symptom_backend: str = "scispacy", # or "medcat" (requires CDB)
                med_backend: str = "med7", # or "scispacy"
                scispacy_symptom_model: str = "en_ner_bc5cdr_md",
                med7_model: str = "en_core_med7_lg",
                scispacy_med_model: str = "en_ner_bc5cdr_md",
                medcat_cdb_path: Optional[str] = None,
                rxnorm_tsv_path: Optional[str] = None,
                llm_client=None):
        
        # normalizer
        self.rxnorm = RxNormNormalizer.from_tsv(rxnorm_tsv_path) if rxnorm_tsv_path else None

        # Symptoms
        if symptom_backend == "scispacy":
            self.symptoms = SciSpaCySymptomExtractor(scispacy_symptom_model)
        elif symptom_backend == "medcat":
            from .provider_experimental.medcat_symptoms import MedCATSymptomExtractor
            self.symptoms = MedCATSymptomExtractor(pack_dir="models/medcat/medcat_pack")
        else:
            raise ValueError("Unknown symptom backend")
        # Meds
        if med_backend == "med7":
            self.meds = Med7MedicationExtractor(med7_model, rxnorm=self.rxnorm)   
        elif med_backend == "scispacy":
            self.meds = SciSpaCyMedicationExtractor(scispacy_med_model, rxnorm=self.rxnorm)
        else:
            raise ValueError("Unknown med backend")
        # Rules
        self.rules = RuleExtras()
        # QA LLM
        self.qa = QAInterpreter(llm_client) if llm_client is not None else None
        self.merge = Merger()


    def extract(self, dialogue: List[Dict]) -> ExtractorJSON:
        ej = ExtractorJSON()
        patient_turns = [u for u in dialogue if u["role"].upper() == "PATIENT"]
        if patient_turns:
            ej.chief_complaint = patient_turns[0]["text"].strip().split("\n")[0][:160]
        # Run providers
        sym = self.symptoms(patient_turns)
        meds = self.meds(patient_turns)
        vitals = self.rules.extract_vitals(patient_turns)
        allergies = self.rules.extract_allergies(patient_turns)
        ej.risk_factors = self.rules.extract_risk_factors(patient_turns)
        # QA pairs
        qa_pairs = pair_questions_and_answers(dialogue)
        qa_out = []
        if self.qa is not None:
            for dturn, pturn in qa_pairs:
                q = self.qa.extract(dturn, pturn)
                if q: qa_out.append(q)
        # Merge
        ej = self.merge(ej, sym, meds, vitals, allergies, qa_out)
        return ej