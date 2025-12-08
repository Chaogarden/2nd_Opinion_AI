# extractor/merge.py
# ==============================
# Merging and deduplication of extracted medical information
# ==============================

from typing import List, Iterable, Set
from .schema import Symptom, Medication, Allergy, Vital, QAExtracted


def _normalize_name(name: str) -> str:
    """Normalize a name for deduplication comparison."""
    if not name:
        return ""
    return name.lower().strip()


def _dedupe_symptoms(symptoms: List[Symptom]) -> List[Symptom]:
    """
    Deduplicate symptoms, preferring entries with more information.
    """
    # Group by normalized name
    groups = {}
    for s in symptoms:
        key = _normalize_name(s.name_norm or s.name_surface)
        if key not in groups:
            groups[key] = []
        groups[key].append(s)
    
    # For each group, pick the best entry and merge evidence
    results = []
    for key, items in groups.items():
        if not items:
            continue
        
        # Sort by information completeness (prefer items with more fields filled)
        def completeness(s):
            score = 0
            if s.name_norm:
                score += 2
            if s.cui:
                score += 2
            if s.duration:
                score += 1
            if s.severity:
                score += 1
            if s.onset:
                score += 1
            if s.assertion == "present":
                score += 1  # Prefer present over absent for same symptom
            return score
        
        items.sort(key=completeness, reverse=True)
        best = items[0]
        
        # Merge evidence from all occurrences
        all_utt_ids = set()
        all_spans = []
        for item in items:
            all_utt_ids.update(item.evidence.utt_ids)
            all_spans.extend(item.evidence.spans)
        
        best.evidence.utt_ids = sorted(all_utt_ids)
        # Dedupe spans by text
        seen_spans = set()
        unique_spans = []
        for span in all_spans:
            span_key = span.get("text", "").lower()
            if span_key not in seen_spans:
                seen_spans.add(span_key)
                unique_spans.append(span)
        best.evidence.spans = unique_spans
        
        results.append(best)
    
    return results


def _dedupe_medications(meds: List[Medication]) -> List[Medication]:
    """
    Deduplicate medications, preferring entries with more information.
    """
    # Group by normalized name (or surface form)
    groups = {}
    for m in meds:
        key = _normalize_name(m.name_norm or m.name_surface)
        if key not in groups:
            groups[key] = []
        groups[key].append(m)
    
    results = []
    for key, items in groups.items():
        if not items:
            continue
        
        # Sort by information completeness
        def completeness(m):
            score = 0
            if m.name_norm:
                score += 2
            if m.rxcui:
                score += 2
            if m.dose:
                score += 1
            if m.freq:
                score += 1
            if m.route:
                score += 1
            if m.form:
                score += 1
            return score
        
        items.sort(key=completeness, reverse=True)
        best = items[0]
        
        # Merge evidence
        all_utt_ids = set()
        for item in items:
            all_utt_ids.update(item.evidence.utt_ids)
        best.evidence.utt_ids = sorted(all_utt_ids)
        
        results.append(best)
    
    return results


def _dedupe_vitals(vitals: List[Vital]) -> List[Vital]:
    """Deduplicate vitals by kind and value."""
    seen = {}
    for v in vitals:
        key = (v.kind, v.value)
        if key not in seen:
            seen[key] = v
        else:
            # Merge evidence
            seen[key].evidence.utt_ids = sorted(
                set(seen[key].evidence.utt_ids + v.evidence.utt_ids)
            )
    return list(seen.values())


def _dedupe_allergies(allergies: List[Allergy]) -> List[Allergy]:
    """Deduplicate allergies by substance."""
    groups = {}
    for a in allergies:
        key = _normalize_name(a.substance_norm or a.substance_surface)
        if key not in groups:
            groups[key] = a
        else:
            # Merge evidence and prefer entries with reaction info
            existing = groups[key]
            existing.evidence.utt_ids = sorted(
                set(existing.evidence.utt_ids + a.evidence.utt_ids)
            )
            if a.reaction and not existing.reaction:
                existing.reaction = a.reaction
    
    return list(groups.values())


def _dedupe_qa(qa_items: List[QAExtracted]) -> List[QAExtracted]:
    """Deduplicate QA extractions by concept."""
    groups = {}
    for q in qa_items:
        key = (_normalize_name(q.concept), q.assertion)
        if key not in groups:
            groups[key] = q
        else:
            # Merge evidence
            existing = groups[key]
            existing.evidence.utt_ids = sorted(
                set(existing.evidence.utt_ids + q.evidence.utt_ids)
            )
    
    return list(groups.values())


class Merger:
    """Merge and deduplicate extracted medical information."""
    
    def __call__(
        self, 
        ej, 
        symptoms: List[Symptom], 
        meds: List[Medication], 
        vitals: List[Vital], 
        allergies: List[Allergy], 
        qa: List[QAExtracted]
    ):
        """
        Merge all extracted information into the ExtractorJSON.
        
        - Removes medications that are actually allergy substances
        - Deduplicates all lists
        - Prefers entries with more complete information
        """
        # Remove meds that are actually allergy substances
        allergy_names = {
            _normalize_name(a.substance_norm or a.substance_surface) 
            for a in allergies
        }
        meds = [
            m for m in meds 
            if _normalize_name(m.name_norm or m.name_surface) not in allergy_names
        ]
        
        # Filter out likely non-symptom entries from NER
        # (e.g., diseases mentioned as diagnoses rather than symptoms)
        filtered_symptoms = []
        for s in symptoms:
            name = _normalize_name(s.name_norm or s.name_surface)
            # Skip very short names (likely noise)
            if len(name) < 3:
                continue
            # Skip certain patterns that are likely diagnoses not symptoms
            skip_patterns = ["cancer", "disease", "syndrome", "disorder", "infection"]
            if any(pat in name for pat in skip_patterns) and s.assertion == "present":
                # Only include if it seems like they're experiencing it as a symptom
                # This is a heuristic - we keep it if they say they "have" it
                pass  # Keep for now, could filter more aggressively
            filtered_symptoms.append(s)
        
        # Deduplicate
        ej.symptoms = _dedupe_symptoms(filtered_symptoms)
        ej.meds = _dedupe_medications(meds)
        ej.vitals = _dedupe_vitals(vitals)
        ej.allergies = _dedupe_allergies(allergies)
        ej.qa_extractions = _dedupe_qa(qa)
        
        return ej
