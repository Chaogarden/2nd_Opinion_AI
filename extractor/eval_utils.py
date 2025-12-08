# extractor/eval_utils.py
# ==============================
# Normalization and comparison utilities for micro-F1 evaluation
# Converts GOLD JSON and ExtractorJSON to comparable atomic item sets
# ==============================

import re
import json
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .schema import ExtractorJSON, Symptom, Medication, Allergy, Vital, QAExtracted


# ============================================================
# Normalization Functions
# ============================================================

def normalize_name(name: Optional[str]) -> str:
    """
    Normalize an entity name to lowercase snake_case.
    
    Args:
        name: Raw entity name (e.g., "Chest Pain", "chest-pain", "CHEST_PAIN")
    
    Returns:
        Normalized name (e.g., "chest_pain")
    """
    if not name:
        return ""
    
    # Lowercase
    name = name.lower().strip()
    
    # Replace common separators with underscore
    name = re.sub(r'[\s\-/]+', '_', name)
    
    # Remove non-alphanumeric characters except underscore
    name = re.sub(r'[^a-z0-9_]', '', name)
    
    # Collapse multiple underscores
    name = re.sub(r'_+', '_', name)
    
    # Strip leading/trailing underscores
    name = name.strip('_')
    
    return name


def normalize_dose(dose: Optional[str]) -> str:
    """
    Normalize a medication dose to a canonical form.
    
    Args:
        dose: Raw dose string (e.g., "10 mg", "10mg", "10 MG")
    
    Returns:
        Normalized dose (e.g., "10_mg")
    """
    if not dose:
        return ""
    
    dose = dose.lower().strip()
    
    # Extract numeric value and unit
    match = re.match(r'([\d.]+)\s*([a-z%]+)?', dose)
    if match:
        value = match.group(1)
        unit = match.group(2) or ""
        
        # Normalize common unit variations
        unit_map = {
            "milligram": "mg", "milligrams": "mg",
            "microgram": "mcg", "micrograms": "mcg", "ug": "mcg",
            "gram": "g", "grams": "g",
            "milliliter": "ml", "milliliters": "ml", "cc": "ml",
            "unit": "units", "iu": "units",
            "percent": "%",
        }
        unit = unit_map.get(unit, unit)
        
        return f"{value}_{unit}" if unit else value
    
    return normalize_name(dose)


def normalize_frequency(freq: Optional[str]) -> str:
    """
    Normalize a medication frequency to a canonical form.
    
    Args:
        freq: Raw frequency string (e.g., "Once daily", "QD", "every day")
    
    Returns:
        Normalized frequency (e.g., "once_daily")
    """
    if not freq:
        return ""
    
    freq = freq.lower().strip()
    
    # Map common variations to canonical forms
    freq_map = {
        # Once daily
        "qd": "once_daily", "od": "once_daily", "q24h": "once_daily",
        "every day": "once_daily", "daily": "once_daily",
        "once a day": "once_daily", "1x daily": "once_daily",
        "once daily": "once_daily", "one time daily": "once_daily",
        "every 24 hours": "once_daily",
        
        # Twice daily
        "bid": "twice_daily", "q12h": "twice_daily",
        "twice a day": "twice_daily", "2x daily": "twice_daily",
        "twice daily": "twice_daily", "two times daily": "twice_daily",
        "every 12 hours": "every_12_hours",
        
        # Three times daily
        "tid": "three_times_daily",
        "three times a day": "three_times_daily", "3x daily": "three_times_daily",
        "three times daily": "three_times_daily",
        
        # Four times daily
        "qid": "four_times_daily",
        "four times a day": "four_times_daily", "4x daily": "four_times_daily",
        "four times daily": "four_times_daily",
        
        # Hourly intervals
        "q4h": "every_4_hours", "every 4 hours": "every_4_hours", "every_4_hours": "every_4_hours",
        "q6h": "every_6_hours", "every 6 hours": "every_6_hours", "every_6_hours": "every_6_hours",
        "q8h": "every_8_hours", "every 8 hours": "every_8_hours", "every_8_hours": "every_8_hours",
        "q12h": "every_12_hours", "every 12 hours": "every_12_hours", "every_12_hours": "every_12_hours",
        
        # As needed
        "prn": "as_needed", "as needed": "as_needed", "as-needed": "as_needed",
        "when needed": "as_needed", "on an as needed basis": "as_needed",
        "as_needed": "as_needed", "as necessary": "as_needed",
        
        # Time-based
        "qhs": "at_bedtime", "at bedtime": "at_bedtime", "at night": "at_bedtime",
        "before bed": "at_bedtime", "nightly": "at_bedtime", "at_bedtime": "at_bedtime",
        "bedtime": "at_bedtime", "hs": "at_bedtime",
        
        "qam": "in_morning", "in the morning": "in_morning", "morning": "in_morning",
        "in_morning": "in_morning", "every morning": "in_morning",
        
        # Weekly
        "weekly": "once_weekly", "once weekly": "once_weekly", "once a week": "once_weekly",
        "once_weekly": "once_weekly", "every week": "once_weekly",
        "twice weekly": "twice_weekly", "twice a week": "twice_weekly",
        "twice_weekly": "twice_weekly", "2x weekly": "twice_weekly",
    }
    
    if freq in freq_map:
        return freq_map[freq]
    
    # Also try with underscores replaced by spaces
    freq_normalized = freq.replace('_', ' ')
    if freq_normalized in freq_map:
        return freq_map[freq_normalized]
    
    return normalize_name(freq)


def normalize_duration(duration: Optional[str]) -> str:
    """
    Normalize a symptom duration to a canonical form.
    
    Args:
        duration: Raw duration string (e.g., "2 weeks", "3 days", "since yesterday")
    
    Returns:
        Normalized duration (e.g., "2_weeks")
    """
    if not duration:
        return ""
    
    duration = duration.lower().strip()
    
    # Try to extract number + unit pattern
    match = re.match(r'([\d.]+)\s*(day|days|week|weeks|month|months|year|years|hour|hours)', duration)
    if match:
        value = match.group(1)
        unit = match.group(2)
        
        # Normalize to singular
        unit_map = {
            "days": "day", "weeks": "week", "months": "month", 
            "years": "year", "hours": "hour"
        }
        unit = unit_map.get(unit, unit)
        
        return f"{value}_{unit}"
    
    # Handle relative terms
    relative_map = {
        "yesterday": "1_day",
        "since yesterday": "1_day",
        "today": "1_day",
        "a few days": "few_days",
        "couple days": "few_days",
        "a few weeks": "few_weeks",
        "couple weeks": "few_weeks",
    }
    
    if duration in relative_map:
        return relative_map[duration]
    
    return normalize_name(duration)


def normalize_assertion(assertion: Optional[str]) -> str:
    """
    Normalize an assertion value.
    
    Args:
        assertion: Raw assertion (e.g., "Present", "ABSENT", "yes")
    
    Returns:
        Normalized assertion ("present", "absent", or "possible")
    """
    if not assertion:
        return "present"
    
    assertion = assertion.lower().strip()
    
    if assertion in ("present", "yes", "true", "current", "active", "confirmed"):
        return "present"
    elif assertion in ("absent", "no", "false", "denied", "denies", "negative"):
        return "absent"
    elif assertion in ("possible", "maybe", "uncertain", "suspected", "likely"):
        return "possible"
    
    return "present"


def normalize_vital_value(kind: str, value: str) -> str:
    """
    Normalize a vital sign value.
    
    Args:
        kind: Vital type (temp, hr, bp, rr, spo2)
        value: Raw value string
    
    Returns:
        Normalized value string (numbers only, no units)
    """
    if not value:
        return ""
    
    value = str(value).strip()
    
    # Remove common units and symbols
    value = re.sub(r'\s*(bpm|beats per minute|\/min|per minute|°[fc]|degrees|%|mmhg|f|c)$', '', value, flags=re.IGNORECASE)
    value = value.strip()
    
    # For blood pressure, ensure format is systolic/diastolic
    if kind == "bp" and "/" in value:
        parts = value.split("/")
        if len(parts) == 2:
            sys_val = re.sub(r'[^\d]', '', parts[0])
            dia_val = re.sub(r'[^\d]', '', parts[1])
            if sys_val and dia_val:
                return f"{sys_val}/{dia_val}"
    
    # For other vitals, extract just the number
    if kind in ("hr", "rr", "spo2"):
        match = re.search(r'(\d+)', value)
        if match:
            return match.group(1)
    
    # For temperature, keep decimal
    if kind == "temp":
        match = re.search(r'(\d+\.?\d*)', value)
        if match:
            return match.group(1)
    
    return value


# ============================================================
# Standardized Allergen Vocabulary
# ============================================================

STANDARD_ALLERGENS = {
    # Drug allergies - antibiotics
    "penicillin", "amoxicillin", "ampicillin", "penicillins",
    "sulfa", "sulfamethoxazole", "sulfasalazine", "sulfonamides",
    "erythromycin", "azithromycin", "clarithromycin", "macrolides",
    "tetracycline", "doxycycline", "minocycline",
    "cephalosporins", "cephalexin", "ceftriaxone", "cefdinir",
    "fluoroquinolones", "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "vancomycin", "metronidazole", "trimethoprim", "nitrofurantoin",
    "clindamycin", "gentamicin",
    
    # Drug allergies - pain/nsaids
    "aspirin", "nsaids", "ibuprofen", "naproxen", "celecoxib",
    "codeine", "morphine", "hydrocodone", "oxycodone", "tramadol", "fentanyl", "opioids",
    "acetaminophen", "tylenol",
    
    # Drug allergies - other
    "ace_inhibitors", "lisinopril", "enalapril",
    "statins", "atorvastatin", "simvastatin",
    "contrast_dye", "iodine", "iodinated_contrast",
    "latex", "adhesive_tape", "bandage_adhesive",
    "lidocaine", "novocaine", "local_anesthetics",
    "general_anesthesia", "propofol",
    "insulin", "heparin", "warfarin",
    
    # Food allergies
    "shellfish", "shrimp", "crab", "lobster",
    "peanuts", "peanut",
    "tree_nuts", "almonds", "walnuts", "cashews", "pecans",
    "eggs", "egg",
    "milk", "dairy", "lactose",
    "soy", "soybean",
    "wheat", "gluten",
    "fish", "salmon", "tuna",
    "sesame",
    
    # Environmental
    "pollen", "grass_pollen", "tree_pollen", "ragweed",
    "dust", "dust_mites",
    "mold", "mildew",
    "pet_dander", "cat_dander", "dog_dander", "cats", "dogs",
    "bee_stings", "wasp_stings", "insect_stings", "bee_venom",
    
    # Other
    "nickel", "jewelry",
}

# Mapping of common variations to standard names
ALLERGEN_NORMALIZATION_MAP = {
    # Penicillins
    "pen": "penicillin", "pcn": "penicillin", "pen vk": "penicillin",
    "amox": "amoxicillin", "augmentin": "amoxicillin",
    
    # Sulfa drugs
    "sulfa drugs": "sulfa", "sulfur": "sulfa", "bactrim": "sulfamethoxazole",
    "septra": "sulfamethoxazole",
    
    # NSAIDs
    "advil": "ibuprofen", "motrin": "ibuprofen",
    "aleve": "naproxen",
    "celebrex": "celecoxib",
    "non-steroidal": "nsaids", "nonsteroidal": "nsaids",
    
    # Opioids
    "vicodin": "hydrocodone", "norco": "hydrocodone",
    "percocet": "oxycodone", "oxycontin": "oxycodone",
    "ultram": "tramadol",
    
    # Antibiotics
    "z-pack": "azithromycin", "zithromax": "azithromycin", "zpak": "azithromycin",
    "cipro": "ciprofloxacin", "levaquin": "levofloxacin",
    "keflex": "cephalexin", "rocephin": "ceftriaxone",
    "flagyl": "metronidazole",
    
    # Contrast/Imaging
    "iv contrast": "contrast_dye", "ct contrast": "contrast_dye",
    "mri contrast": "contrast_dye", "gadolinium": "contrast_dye",
    
    # Food
    "peanut butter": "peanuts",
    "shellfish allergy": "shellfish",
    "lactose intolerant": "lactose", "lactose intolerance": "lactose",
    
    # Environmental
    "hay fever": "pollen",
    "cat": "cat_dander", "cats": "cat_dander",
    "dog": "dog_dander", "dogs": "dog_dander",
    "bee sting": "bee_stings", "bees": "bee_stings",
    "wasp": "wasp_stings", "wasps": "wasp_stings",
}


def normalize_allergen(allergen: Optional[str]) -> str:
    """
    Normalize an allergen name to the standardized vocabulary.
    
    Args:
        allergen: Raw allergen name
    
    Returns:
        Normalized allergen name from standard vocabulary
    """
    if not allergen:
        return ""
    
    allergen = str(allergen).lower().strip()
    
    # Replace common separators
    allergen = re.sub(r'[\s\-/]+', '_', allergen)
    allergen = re.sub(r'[^a-z0-9_]', '', allergen)
    allergen = re.sub(r'_+', '_', allergen)
    allergen = allergen.strip('_')
    
    # Check normalization map first
    original = allergen.replace('_', ' ')
    if original in ALLERGEN_NORMALIZATION_MAP:
        return ALLERGEN_NORMALIZATION_MAP[original]
    
    # Check if already a standard allergen
    if allergen in STANDARD_ALLERGENS:
        return allergen
    
    # Try to match partial names
    for std_allergen in STANDARD_ALLERGENS:
        if std_allergen in allergen or allergen in std_allergen:
            return std_allergen
    
    # Return normalized form even if not in standard list
    return allergen


def normalize_medication_name(name: Optional[str]) -> str:
    """
    Normalize a medication name to lowercase generic form.
    
    Args:
        name: Raw medication name
    
    Returns:
        Normalized lowercase generic drug name
    """
    if not name:
        return ""
    
    name = str(name).lower().strip()
    
    # Remove common suffixes/prefixes
    name = re.sub(r'\s*(tablet|capsule|cap|tab|oral|injection|solution|suspension|cream|ointment|gel|patch|inhaler|spray)s?\s*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^\s*(generic|brand)\s*', '', name, flags=re.IGNORECASE)
    
    # Replace separators with nothing (medications are typically one word)
    name = re.sub(r'[\s\-]+', '', name)
    
    # Remove non-alphanumeric
    name = re.sub(r'[^a-z0-9]', '', name)
    
    return name


def normalize_demographic_value(key: str, value: Optional[str]) -> str:
    """
    Normalize a demographic value for exact matching.
    
    Args:
        key: Demographic field name (age, sex, race_ethnicity, sexual_orientation)
        value: Raw value string
    
    Returns:
        Normalized value string (lowercase, snake_case)
    """
    if not value:
        return ""
    
    value = str(value).lower().strip()
    
    # Skip null-like values
    if value in ("null", "none", "unknown", "not specified", "not stated", ""):
        return ""
    
    if key == "age":
        # Extract just the numeric value for age
        match = re.search(r'(\d+)', value)
        if match:
            return match.group(1)
        return ""
    
    elif key == "sex":
        # Normalize sex values
        if value in ("m", "male", "man", "boy"):
            return "male"
        elif value in ("f", "female", "woman", "girl"):
            return "female"
        else:
            return value.replace(" ", "_").replace("-", "_")
    
    elif key == "race_ethnicity":
        # Normalize race/ethnicity to snake_case
        # Common normalizations
        race_map = {
            "african american": "african_american",
            "african-american": "african_american",
            "black": "african_american",
            "white": "caucasian",
            "caucasian": "caucasian",
            "hispanic": "hispanic",
            "latino": "hispanic",
            "latina": "hispanic",
            "asian": "asian",
            "pacific islander": "pacific_islander",
            "native american": "native_american",
            "american indian": "native_american",
            "mixed": "mixed",
            "multiracial": "mixed",
        }
        normalized = value.replace("-", " ").replace("_", " ").strip()
        if normalized in race_map:
            return race_map[normalized]
        return value.replace(" ", "_").replace("-", "_")
    
    elif key == "sexual_orientation":
        # Normalize sexual orientation
        orientation_map = {
            "straight": "heterosexual",
            "heterosexual": "heterosexual",
            "gay": "homosexual",
            "lesbian": "homosexual",
            "homosexual": "homosexual",
            "bisexual": "bisexual",
            "bi": "bisexual",
            "queer": "queer",
            "pansexual": "pansexual",
            "asexual": "asexual",
        }
        normalized = value.replace("-", " ").replace("_", " ").strip()
        if normalized in orientation_map:
            return orientation_map[normalized]
        return value.replace(" ", "_").replace("-", "_")
    
    # Default: normalize to snake_case
    return value.replace(" ", "_").replace("-", "_")


# ============================================================
# Atomic Item Set Conversion
# ============================================================

@dataclass
class AtomicItemSet:
    """
    Collection of atomic items for comparison.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    """
    # Entity-level items: (type, name_norm, assertion) - med, allergy only
    entities: Set[Tuple[str, str, str]] = field(default_factory=set)
    
    # Attribute-level items: (type, name_norm, attr_name, attr_value) - dose, freq, reaction
    attributes: Set[Tuple[str, str, str, str]] = field(default_factory=set)
    
    # Vitals: (kind, value)
    vitals: Set[Tuple[str, str]] = field(default_factory=set)
    
    # Demographics: (key, value) tuples for exact matching
    # Keys: age, sex, race_ethnicity, sexual_orientation
    demographics: Set[Tuple[str, str]] = field(default_factory=set)


def gold_to_atomic_items(gold: Dict[str, Any]) -> AtomicItemSet:
    """
    Convert GOLD JSON payload to atomic item set.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    
    Args:
        gold: The "gold" dict from a GOLD JSONL entry
    
    Returns:
        AtomicItemSet with normalized items
    """
    items = AtomicItemSet()
    
    # Demographics - exact match scoring
    demographics_data = gold.get("demographics", {})
    if isinstance(demographics_data, dict):
        for key in ("age", "sex", "race_ethnicity", "sexual_orientation"):
            value = demographics_data.get(key)
            normalized = normalize_demographic_value(key, value)
            if normalized:
                items.demographics.add((key, normalized))
    
    # Medications - use strict medication name normalization
    for m in gold.get("meds", []):
        name = normalize_medication_name(m.get("name_norm") or m.get("name"))
        assertion = normalize_assertion(m.get("assertion"))
        if name:
            items.entities.add(("med", name, assertion))
            
            # Dose attribute
            dose = normalize_dose(m.get("dose"))
            if dose:
                items.attributes.add(("med", name, "dose", dose))
            
            # Frequency attribute
            freq = normalize_frequency(m.get("freq") or m.get("frequency"))
            if freq:
                items.attributes.add(("med", name, "freq", freq))
    
    # Allergies - use strict allergen normalization
    for a in gold.get("allergies", []):
        name = normalize_allergen(a.get("substance_norm") or a.get("substance"))
        assertion = normalize_assertion(a.get("assertion"))
        if name:
            items.entities.add(("allergy", name, assertion))
            
            # Reaction attribute
            reaction = normalize_name(a.get("reaction"))
            if reaction:
                items.attributes.add(("allergy", name, "reaction", reaction))
    
    # Vitals
    for v in gold.get("vitals", []):
        kind = v.get("kind", "").lower()
        value = normalize_vital_value(kind, str(v.get("value", "")))
        if kind and value:
            items.vitals.add((kind, value))
    
    return items


def extractor_json_to_atomic_items(ej: ExtractorJSON) -> AtomicItemSet:
    """
    Convert ExtractorJSON prediction to atomic item set.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    
    Args:
        ej: ExtractorJSON from the extractor
    
    Returns:
        AtomicItemSet with normalized items
    """
    items = AtomicItemSet()
    
    # Demographics - exact match scoring
    if ej.demographics:
        for key in ("age", "sex", "race_ethnicity", "sexual_orientation"):
            value = ej.demographics.get(key)
            normalized = normalize_demographic_value(key, value)
            if normalized:
                items.demographics.add((key, normalized))
    
    # Medications - use strict medication name normalization
    for m in ej.meds:
        name = normalize_medication_name(m.name_norm or m.name_surface)
        assertion = normalize_assertion(m.assertion)
        if name:
            items.entities.add(("med", name, assertion))
            
            # Dose attribute
            dose = normalize_dose(m.dose)
            if dose:
                items.attributes.add(("med", name, "dose", dose))
            
            # Frequency attribute
            freq = normalize_frequency(m.freq)
            if freq:
                items.attributes.add(("med", name, "freq", freq))
    
    # Allergies - use strict allergen normalization
    for a in ej.allergies:
        name = normalize_allergen(a.substance_norm or a.substance_surface)
        assertion = normalize_assertion(a.assertion)
        if name:
            items.entities.add(("allergy", name, assertion))
            
            # Reaction attribute
            reaction = normalize_name(a.reaction)
            if reaction:
                items.attributes.add(("allergy", name, "reaction", reaction))
    
    # Vitals
    for v in ej.vitals:
        kind = v.kind.lower() if v.kind else ""
        value = normalize_vital_value(kind, v.value)
        if kind and value:
            items.vitals.add((kind, value))
    
    return items


# ============================================================
# Micro-F1 Computation
# ============================================================

@dataclass
class MetricCounts:
    """Counts for computing precision, recall, F1."""
    tp: int = 0  # True positives
    fp: int = 0  # False positives
    fn: int = 0  # False negatives
    
    def precision(self) -> float:
        """Compute precision."""
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)
    
    def recall(self) -> float:
        """Compute recall."""
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)
    
    def f1(self) -> float:
        """Compute F1 score."""
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    def add(self, other: 'MetricCounts') -> 'MetricCounts':
        """Add counts from another MetricCounts."""
        return MetricCounts(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn
        )


def compute_set_metrics(gold_set: Set, pred_set: Set) -> MetricCounts:
    """
    Compute TP, FP, FN for two sets.
    
    Args:
        gold_set: Set of gold items
        pred_set: Set of predicted items
    
    Returns:
        MetricCounts with TP, FP, FN
    """
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return MetricCounts(tp=tp, fp=fp, fn=fn)


@dataclass
class EvalResult:
    """
    Complete evaluation result with per-class breakdown.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    """
    overall: MetricCounts = field(default_factory=MetricCounts)
    by_entity_type: Dict[str, MetricCounts] = field(default_factory=dict)  # med, allergy
    by_attribute: Dict[str, MetricCounts] = field(default_factory=dict)    # dose, freq, reaction
    vitals: MetricCounts = field(default_factory=MetricCounts)
    demographics: MetricCounts = field(default_factory=MetricCounts)
    by_demographic: Dict[str, MetricCounts] = field(default_factory=dict)
    
    # Error analysis
    false_negatives: List[Tuple[str, Any]] = field(default_factory=list)
    false_positives: List[Tuple[str, Any]] = field(default_factory=list)


def evaluate_single(gold_items: AtomicItemSet, pred_items: AtomicItemSet) -> EvalResult:
    """
    Evaluate a single gold/pred pair.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    
    Args:
        gold_items: GOLD atomic items
        pred_items: Predicted atomic items
    
    Returns:
        EvalResult with metrics and error analysis
    """
    result = EvalResult()
    
    # Entity-level metrics (overall and by type) - ONLY med and allergy
    entity_metrics = compute_set_metrics(gold_items.entities, pred_items.entities)
    result.overall = entity_metrics
    
    # Break down by entity type - ONLY med and allergy (crucial categories)
    for entity_type in ("med", "allergy"):
        gold_type = {e for e in gold_items.entities if e[0] == entity_type}
        pred_type = {e for e in pred_items.entities if e[0] == entity_type}
        result.by_entity_type[entity_type] = compute_set_metrics(gold_type, pred_type)
    
    # Attribute-level metrics
    attr_metrics = compute_set_metrics(gold_items.attributes, pred_items.attributes)
    result.by_attribute["all"] = attr_metrics
    
    # Break down by attribute type - ONLY dose, freq, reaction (crucial)
    for attr_name in ("dose", "freq", "reaction"):
        gold_attr = {a for a in gold_items.attributes if a[2] == attr_name}
        pred_attr = {a for a in pred_items.attributes if a[2] == attr_name}
        if gold_attr or pred_attr:
            result.by_attribute[attr_name] = compute_set_metrics(gold_attr, pred_attr)
    
    # Vitals
    result.vitals = compute_set_metrics(gold_items.vitals, pred_items.vitals)
    
    # Demographics - overall and by field
    result.demographics = compute_set_metrics(gold_items.demographics, pred_items.demographics)
    
    # Break down by demographic field
    for demo_field in ("age", "sex", "race_ethnicity", "sexual_orientation"):
        gold_demo = {d for d in gold_items.demographics if d[0] == demo_field}
        pred_demo = {d for d in pred_items.demographics if d[0] == demo_field}
        if gold_demo or pred_demo:
            result.by_demographic[demo_field] = compute_set_metrics(gold_demo, pred_demo)
    
    # Error analysis - ONLY crucial categories
    for item in gold_items.entities - pred_items.entities:
        result.false_negatives.append(("entity", item))
    for item in gold_items.attributes - pred_items.attributes:
        result.false_negatives.append(("attribute", item))
    for item in gold_items.vitals - pred_items.vitals:
        result.false_negatives.append(("vital", item))
    for item in gold_items.demographics - pred_items.demographics:
        result.false_negatives.append(("demographic", item))
    
    for item in pred_items.entities - gold_items.entities:
        result.false_positives.append(("entity", item))
    for item in pred_items.attributes - gold_items.attributes:
        result.false_positives.append(("attribute", item))
    for item in pred_items.vitals - gold_items.vitals:
        result.false_positives.append(("vital", item))
    for item in pred_items.demographics - gold_items.demographics:
        result.false_positives.append(("demographic", item))
    
    return result


def aggregate_results(results: List[EvalResult]) -> EvalResult:
    """
    Aggregate multiple evaluation results (micro-averaging).
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    
    Args:
        results: List of per-example EvalResult
    
    Returns:
        Aggregated EvalResult with combined counts
    """
    agg = EvalResult()
    
    for r in results:
        agg.overall = agg.overall.add(r.overall)
        agg.vitals = agg.vitals.add(r.vitals)
        agg.demographics = agg.demographics.add(r.demographics)
        
        for key, counts in r.by_entity_type.items():
            if key not in agg.by_entity_type:
                agg.by_entity_type[key] = MetricCounts()
            agg.by_entity_type[key] = agg.by_entity_type[key].add(counts)
        
        for key, counts in r.by_attribute.items():
            if key not in agg.by_attribute:
                agg.by_attribute[key] = MetricCounts()
            agg.by_attribute[key] = agg.by_attribute[key].add(counts)
        
        for key, counts in r.by_demographic.items():
            if key not in agg.by_demographic:
                agg.by_demographic[key] = MetricCounts()
            agg.by_demographic[key] = agg.by_demographic[key].add(counts)
        
        agg.false_negatives.extend(r.false_negatives)
        agg.false_positives.extend(r.false_positives)
    
    return agg


def format_metrics_table(result: EvalResult) -> str:
    """
    Format evaluation result as a human-readable table.
    
    CRUCIAL CATEGORIES ONLY: Demographics, Medications, Allergies, Vitals
    
    Args:
        result: EvalResult to format
    
    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 70)
    lines.append("CRUCIAL INFORMATION MICRO-F1 EVALUATION")
    lines.append("=" * 70)
    lines.append("")
    
    lines.append(f"{'Category':<20} {'Prec':>8} {'Rec':>8} {'F1':>8} {'TP':>6} {'FP':>6} {'FN':>6}")
    lines.append("-" * 70)
    
    # Demographics
    m = result.demographics
    lines.append(f"{'Demographics':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Demographics breakdown by field
    for demo_field in ("age", "sex", "race_ethnicity", "sexual_orientation"):
        if demo_field in result.by_demographic:
            m = result.by_demographic[demo_field]
            label = f"  {demo_field}"
            lines.append(f"{label:<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Medications
    if "med" in result.by_entity_type:
        m = result.by_entity_type["med"]
        lines.append(f"{'Medications':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Allergies
    if "allergy" in result.by_entity_type:
        m = result.by_entity_type["allergy"]
        lines.append(f"{'Allergies':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Vitals
    m = result.vitals
    lines.append(f"{'Vitals':<20} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    # Attribute-level (dose, freq, reaction)
    lines.append("")
    lines.append("ATTRIBUTES (dose, freq, reaction)")
    lines.append("-" * 70)
    
    for attr_name in ("all", "dose", "freq", "reaction"):
        if attr_name in result.by_attribute:
            m = result.by_attribute[attr_name]
            label = f"attr_{attr_name}" if attr_name != "all" else "All Attributes"
            lines.append(f"  {label:<18} {m.precision():>8.3f} {m.recall():>8.3f} {m.f1():>8.3f} {m.tp:>6} {m.fp:>6} {m.fn:>6}")
    
    lines.append("")
    lines.append("=" * 70)
    
    return "\n".join(lines)


def format_error_analysis(result: EvalResult, max_errors: int = 10) -> str:
    """
    Format error analysis showing top false negatives and positives.
    
    Args:
        result: EvalResult with error lists
        max_errors: Maximum errors to show per category
    
    Returns:
        Formatted string
    """
    lines = []
    lines.append("")
    lines.append("ERROR ANALYSIS")
    lines.append("=" * 70)
    
    lines.append("")
    lines.append(f"Top False Negatives (missed in prediction) - showing {min(max_errors, len(result.false_negatives))} of {len(result.false_negatives)}")
    lines.append("-" * 50)
    for item_type, item in result.false_negatives[:max_errors]:
        lines.append(f"  [{item_type}] {item}")
    
    lines.append("")
    lines.append(f"Top False Positives (over-predicted) - showing {min(max_errors, len(result.false_positives))} of {len(result.false_positives)}")
    lines.append("-" * 50)
    for item_type, item in result.false_positives[:max_errors]:
        lines.append(f"  [{item_type}] {item}")
    
    return "\n".join(lines)


# ============================================================
# GOLD File Loading
# ============================================================

def load_gold_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load GOLD JSONL file.
    
    Args:
        path: Path to the GOLD JSONL file
    
    Returns:
        List of GOLD entries
    """
    path = Path(path)
    entries = []
    
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
    
    return entries


# ============================================================
# Convenience wrapper
# ============================================================

def evaluate_extraction(
    gold_json: Dict[str, Any],
    pred_ej: ExtractorJSON
) -> EvalResult:
    """
    Evaluate a single extraction against gold.
    
    Args:
        gold_json: The "gold" payload from GOLD JSONL
        pred_ej: ExtractorJSON prediction
    
    Returns:
        EvalResult for this example
    """
    gold_items = gold_to_atomic_items(gold_json)
    pred_items = extractor_json_to_atomic_items(pred_ej)
    return evaluate_single(gold_items, pred_items)
