# extractor/hybrid_extractor.py
# ==============================
# Enhanced Hybrid Medical Information Extractor
# LLM-first approach with NER enrichment
# ==============================

from typing import Dict, List, Optional
from .normalizers.rxnorm import RxNormNormalizer 
from .schema import ExtractorJSON, Symptom, Medication, Evidence
from .providers.scispacy_symptoms import SciSpaCySymptomExtractor
from .providers.meds_med7 import Med7MedicationExtractor
from .providers.meds_scispacy import SciSpaCyMedicationExtractor
from .rule_extractors import RuleExtras
from .merge import Merger
from .llm_extractor import LLMExtractor
from .enhanced_patterns import (
    extract_chief_complaint,
    extract_symptoms_by_pattern,
    extract_medications_by_pattern,
    is_greeting_or_filler,
)


class HybridExtractor:
    """
    Enhanced hybrid medical information extractor.
    
    Architecture (LLM-first):
    1. PRIMARY: LLM processes entire dialogue for comprehensive extraction
    2. ENRICHMENT: NER/patterns add medical codes (RxCUI, etc.)
    3. FALLBACK: If LLM fails, fall back to NER + patterns
    """
    
    def __init__(self,
                symptom_backend: str = "scispacy",
                med_backend: str = "scispacy",
                scispacy_symptom_model: str = "en_ner_bc5cdr_md",
                med7_model: str = "en_core_med7_lg",
                scispacy_med_model: str = "en_ner_bc5cdr_md",
                medcat_cdb_path: Optional[str] = None,
                rxnorm_tsv_path: Optional[str] = None,
                llm_client=None,
                qa_model_name: Optional[str] = None,
                use_llm_extraction: bool = True,
                use_ner_enrichment: bool = True):
        """
        Initialize the hybrid extractor.
        
        Args:
            symptom_backend: 'scispacy' or 'medcat' (for NER enrichment)
            med_backend: 'scispacy' or 'med7' (for NER enrichment)
            scispacy_symptom_model: Model name for scispacy symptoms
            med7_model: Model name for Med7
            scispacy_med_model: Model name for scispacy medications
            medcat_cdb_path: Path to MedCAT CDB (if using medcat)
            rxnorm_tsv_path: Path to RxNorm TSV for medication normalization
            llm_client: LLM client for extraction (required for LLM mode)
            qa_model_name: Model name for LLM extraction
            use_llm_extraction: Whether to use LLM as primary extractor
            use_ner_enrichment: Whether to enrich with NER codes
        """
        self.use_llm_extraction = use_llm_extraction
        self.use_ner_enrichment = use_ner_enrichment
        
        # RxNorm normalizer for medication codes
        self.rxnorm = RxNormNormalizer.from_tsv(rxnorm_tsv_path) if rxnorm_tsv_path else None
        
        # LLM Extractor (primary)
        if llm_client is not None and qa_model_name is not None:
            self.llm_extractor = LLMExtractor(llm_client, qa_model_name)
        else:
            self.llm_extractor = None
            if use_llm_extraction:
                print("Warning: LLM extraction requested but no client/model provided. Falling back to NER.")
                self.use_llm_extraction = False
        
        # NER extractors (for enrichment or fallback)
        # Only load if we need NER enrichment or don't have LLM extraction
        self.symptoms_ner = None
        self.meds_ner = None
        self._symptom_backend = symptom_backend
        self._med_backend = med_backend
        self._scispacy_symptom_model = scispacy_symptom_model
        self._scispacy_med_model = scispacy_med_model
        self._med7_model = med7_model
        
        needs_ner = use_ner_enrichment or not self.use_llm_extraction
        if needs_ner:
            self._load_ner_models()
        
        # Rule-based extractors
        self.rules = RuleExtras()
        self.merge = Merger()

    def _load_ner_models(self):
        """Lazily load NER models when needed."""
        if self.symptoms_ner is not None and self.meds_ner is not None:
            return  # Already loaded
        
        try:
            if self._symptom_backend == "scispacy":
                self.symptoms_ner = SciSpaCySymptomExtractor(self._scispacy_symptom_model)
            elif self._symptom_backend == "medcat":
                from .provider_experimental.medcat_symptoms import MedCATSymptomExtractor
                self.symptoms_ner = MedCATSymptomExtractor(pack_dir="models/medcat/medcat_pack")
            
            if self._med_backend == "med7":
                self.meds_ner = Med7MedicationExtractor(self._med7_model, rxnorm=self.rxnorm)   
            elif self._med_backend == "scispacy":
                self.meds_ner = SciSpaCyMedicationExtractor(self._scispacy_med_model, rxnorm=self.rxnorm)
        except Exception as e:
            print(f"Warning: Could not load NER models: {e}")
            print("         NER enrichment will be disabled.")
            self.use_ner_enrichment = False

    def extract(self, dialogue: List[Dict]) -> ExtractorJSON:
        """
        Extract medical information from a dialogue.
        
        Args:
            dialogue: List of turn dicts with 'utt_id', 'role', and 'text'.
        
        Returns:
            ExtractorJSON with extracted clinical information.
        """
        patient_turns = [u for u in dialogue if u["role"].upper() == "PATIENT"]
        
        # ================================================================
        # PRIMARY: LLM Extraction (if enabled)
        # ================================================================
        if self.use_llm_extraction and self.llm_extractor is not None:
            try:
                ej = self.llm_extractor.extract(dialogue)
                
                # Check if extraction was successful (has meaningful content)
                if ej.chief_complaint or ej.symptoms or ej.meds:
                    # Enrich with codes if enabled
                    if self.use_ner_enrichment:
                        ej = self._enrich_with_codes(ej, patient_turns)
                    return ej
                else:
                    print("LLM extraction returned empty results, falling back to NER")
            except Exception as e:
                print(f"LLM extraction failed: {e}, falling back to NER")
        
        # ================================================================
        # FALLBACK: NER + Pattern Extraction
        # ================================================================
        return self._extract_with_ner_patterns(dialogue, patient_turns)
    
    def _enrich_with_codes(self, ej: ExtractorJSON, patient_turns: List[Dict]) -> ExtractorJSON:
        """
        Enrich LLM extraction results with medical codes from NER.
        
        Adds:
        - RxCUI codes for medications
        - CUI codes for symptoms (if available)
        - Any medications/symptoms missed by LLM
        """
        # Enrich medications with RxNorm codes
        if self.rxnorm:
            for med in ej.meds:
                if not med.rxcui:
                    rxcui, generic, score = self.rxnorm.normalize(med.name_surface)
                    if rxcui:
                        med.rxcui = rxcui
                        if generic and not med.name_norm:
                            med.name_norm = generic
        
        # Run NER to find any missed medications (if NER is available)
        if self.meds_ner is not None:
            try:
                ner_meds = self.meds_ner(patient_turns)
                existing_med_names = {
                    (m.name_norm or m.name_surface).lower() 
                    for m in ej.meds
                }
                
                for ner_med in ner_meds:
                    name_key = (ner_med.name_norm or ner_med.name_surface).lower()
                    if name_key not in existing_med_names:
                        ej.meds.append(ner_med)
                        existing_med_names.add(name_key)
            except Exception as e:
                print(f"NER medication enrichment failed: {e}")
        
        # Run NER to find any missed symptoms (if NER is available)
        if self.symptoms_ner is not None:
            try:
                ner_symptoms = self.symptoms_ner(patient_turns)
                existing_symptom_names = {
                    (s.name_norm or s.name_surface).lower() 
                    for s in ej.symptoms
                }
                
                for ner_sym in ner_symptoms:
                    name_key = (ner_sym.name_norm or ner_sym.name_surface).lower()
                    if name_key not in existing_symptom_names:
                        ej.symptoms.append(ner_sym)
                        existing_symptom_names.add(name_key)
            except Exception as e:
                print(f"NER symptom enrichment failed: {e}")
        
        # Extract vitals with rules (LLM might miss specific values)
        try:
            rule_vitals = self.rules.extract_vitals(patient_turns)
            existing_vitals = {(v.kind, v.value) for v in ej.vitals}
            for rv in rule_vitals:
                if (rv.kind, rv.value) not in existing_vitals:
                    ej.vitals.append(rv)
        except Exception:
            pass
        
        # Extract allergies with rules
        try:
            rule_allergies = self.rules.extract_allergies(patient_turns)
            existing_allergies = {
                (a.substance_norm or a.substance_surface).lower() 
                for a in ej.allergies
            }
            for ra in rule_allergies:
                name_key = (ra.substance_norm or ra.substance_surface).lower()
                if name_key not in existing_allergies:
                    ej.allergies.append(ra)
        except Exception:
            pass
        
        # Extract risk factors with rules
        try:
            rule_rf = self.rules.extract_risk_factors(patient_turns)
            existing_rf = set(ej.risk_factors)
            for rf in rule_rf:
                if rf not in existing_rf:
                    ej.risk_factors.append(rf)
        except Exception:
            pass
        
        return ej
    
    def _extract_with_ner_patterns(
        self, 
        dialogue: List[Dict], 
        patient_turns: List[Dict]
    ) -> ExtractorJSON:
        """
        Fallback extraction using NER and pattern matching.
        Used when LLM extraction is disabled or fails.
        """
        # Ensure NER models are loaded for fallback
        if self.symptoms_ner is None or self.meds_ner is None:
            self._load_ner_models()
        
        ej = ExtractorJSON()
        
        # Chief Complaint
        if patient_turns:
            ej.chief_complaint = extract_chief_complaint(patient_turns)
            
            # Fallback to first substantive turn
            if not ej.chief_complaint:
                for turn in patient_turns:
                    text = turn["text"].strip()
                    if not is_greeting_or_filler(text) and len(text) > 15:
                        ej.chief_complaint = text.split("\n")[0][:160]
                        break
        
        # Symptoms from NER (if available)
        symptoms_ner = []
        if self.symptoms_ner is not None:
            try:
                symptoms_ner = self.symptoms_ner(patient_turns)
            except Exception as e:
                print(f"NER symptom extraction failed: {e}")
        
        # Symptoms from patterns
        symptoms_pattern = self._extract_symptoms_pattern(patient_turns)
        
        # Medications from NER (if available)
        meds_ner = []
        if self.meds_ner is not None:
            try:
                meds_ner = self.meds_ner(patient_turns)
            except Exception as e:
                print(f"NER medication extraction failed: {e}")
        
        # Medications from patterns
        meds_pattern = self._extract_meds_pattern(patient_turns)
        
        # Vitals, allergies, risk factors from rules
        vitals = self.rules.extract_vitals(patient_turns)
        allergies = self.rules.extract_allergies(patient_turns)
        ej.risk_factors = self.rules.extract_risk_factors(patient_turns)
        
        # Merge everything
        all_symptoms = symptoms_ner + symptoms_pattern
        all_meds = meds_ner + meds_pattern
        
        ej = self.merge(ej, all_symptoms, all_meds, vitals, allergies, [])
        
        return ej
    
    def _extract_symptoms_pattern(self, patient_turns: List[Dict]) -> List[Symptom]:
        """Extract symptoms using pattern matching."""
        symptoms = []
        seen = set()
        
        for turn in patient_turns:
            text = turn["text"]
            utt_id = turn["utt_id"]
            
            if is_greeting_or_filler(text):
                continue
            
            matches = extract_symptoms_by_pattern(text)
            for surface, normalized, start, end in matches:
                key = (normalized.lower(), utt_id)
                if key in seen:
                    continue
                seen.add(key)
                
                symptoms.append(Symptom(
                    name_surface=surface,
                    name_norm=normalized,
                    assertion="present",
                    evidence=Evidence(
                        utt_ids=[utt_id],
                        spans=[{"text": surface, "start": start, "end": end}]
                    )
                ))
        
        return symptoms
    
    def _extract_meds_pattern(self, patient_turns: List[Dict]) -> List[Medication]:
        """Extract medications using pattern matching."""
        meds = []
        seen = set()
        
        for turn in patient_turns:
            text = turn["text"]
            utt_id = turn["utt_id"]
            lower_text = text.lower()
            
            if "allergic to" in lower_text:
                continue
            
            matches = extract_medications_by_pattern(text)
            for med_name, start, end in matches:
                key = (med_name.lower(), utt_id)
                if key in seen:
                    continue
                seen.add(key)
                
                assertion = "present"
                if any(cue in lower_text for cue in ["stopped", "quit", "no longer", "used to"]):
                    assertion = "absent"
                
                name_norm = None
                rxcui = None
                if self.rxnorm:
                    rx, gen, score = self.rxnorm.normalize(med_name)
                    if rx and gen:
                        rxcui = rx
                        name_norm = gen
                
                meds.append(Medication(
                    name_surface=med_name,
                    name_norm=name_norm,
                    rxcui=rxcui,
                    assertion=assertion,
                    evidence=Evidence(
                        utt_ids=[utt_id],
                        spans=[{"text": med_name, "start": start, "end": end}]
                    )
                ))
        
        return meds
