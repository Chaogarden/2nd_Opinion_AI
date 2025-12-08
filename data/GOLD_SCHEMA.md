# GOLD Extraction Schema Documentation

This document describes the schema for the 15-example GOLD evaluation dataset used to compute micro-F1 scores for the clinical information extractor.

## Evaluation Focus

The evaluator tests **4 crucial information categories** only:
- **Demographics** (exact match)
- **Medications** (strict format)
- **Allergies** (standardized vocabulary)
- **Vitals** (fixed types + units)

## File Format

The GOLD data is stored in JSON format at `data/aci_extractor_gold_15.jsonl`. When running evaluation, convert to JSONL (one JSON object per line).

## Schema Definition

Each entry contains:

```json
{
  "encounter_id": "string",
  "source_type": "dialogue" | "note" | "augmented_note",
  "gold": {
    "demographics": {
      "age": "patient age in years as string (e.g., '45', '72')",
      "sex": "male | female",
      "race_ethnicity": "lowercase_snake_case",
      "sexual_orientation": "lowercase"
    },
    "meds": [
      {
        "name_norm": "lowercase generic drug name",
        "assertion": "present" | "absent" | "possible",
        "dose": "string (e.g., '10 mg', '500 mg')",
        "freq": "canonical frequency value"
      }
    ],
    "allergies": [
      {
        "substance_norm": "standardized allergen name",
        "assertion": "present" | "absent" | "possible",
        "reaction": "standardized reaction"
      }
    ],
    "vitals": [
      {
        "kind": "temp" | "hr" | "bp" | "rr" | "spo2",
        "value": "numeric string (no units)"
      }
    ]
  }
}
```

---

## Normalization Conventions

### Demographics (EXACT MATCH)

Demographics are scored with **exact matching**. Only include demographics that are explicitly mentioned in the dialogue.

- `age`: Patient's age in years as a string containing only digits
  - Examples: `"45"`, `"72"`, `"28"`
  - NOT: `"45 years old"`, `"forty-five"`
  
- `sex`: Normalized to:
  - `"male"` (includes: m, man, boy)
  - `"female"` (includes: f, woman, girl)
  
- `race_ethnicity`: Lowercase snake_case:
  - `"african_american"`, `"caucasian"`, `"hispanic"`, `"asian"`, `"native_american"`, `"pacific_islander"`, `"mixed"`
  
- `sexual_orientation`: Lowercase:
  - `"heterosexual"`, `"homosexual"`, `"bisexual"`, `"queer"`, `"pansexual"`, `"asexual"`

**Scoring:**
- If present in gold but absent in prediction → **False Negative**
- If absent in gold but present in prediction → **False Positive**
- If absent in both → **No penalty**

---

### Medications (STRICT FORMAT)

- `name_norm`: **LOWERCASE generic drug name only**
  - ✅ `"lisinopril"`, `"metformin"`, `"aspirin"`, `"atorvastatin"`
  - ❌ `"Lipitor"` (use `"atorvastatin"`), `"Tylenol"` (use `"acetaminophen"`)

- `dose`: **"number unit" format with space**
  - ✅ `"10 mg"`, `"500 mg"`, `"81 mg"`, `"2.5 mg"`, `"100 mcg"`
  - ❌ `"10mg"`, `"ten milligrams"`

- `freq`: **USE ONLY THESE CANONICAL VALUES:**

  | Canonical Value | Aliases |
  |-----------------|---------|
  | `"once_daily"` | daily, qd, q24h |
  | `"twice_daily"` | bid, 2x daily |
  | `"three_times_daily"` | tid, 3x daily |
  | `"four_times_daily"` | qid, 4x daily |
  | `"every_4_hours"` | q4h |
  | `"every_6_hours"` | q6h |
  | `"every_8_hours"` | q8h |
  | `"every_12_hours"` | q12h |
  | `"as_needed"` | prn |
  | `"at_bedtime"` | qhs, nightly |
  | `"in_morning"` | qam |
  | `"once_weekly"` | weekly |
  | `"twice_weekly"` | 2x weekly |

- `assertion`: `"present"`, `"absent"`, or `"possible"`

---

### Allergies (STANDARDIZED VOCABULARY)

- `substance_norm`: **USE STANDARDIZED NAMES:**

  **Drug Allergies:**
  - Antibiotics: `"penicillin"`, `"amoxicillin"`, `"sulfa"`, `"sulfamethoxazole"`, `"erythromycin"`, `"azithromycin"`, `"cephalosporins"`, `"ciprofloxacin"`, `"vancomycin"`, `"clindamycin"`
  - Pain/NSAIDs: `"aspirin"`, `"nsaids"`, `"ibuprofen"`, `"naproxen"`, `"codeine"`, `"morphine"`, `"hydrocodone"`, `"oxycodone"`, `"acetaminophen"`
  - Other: `"ace_inhibitors"`, `"lisinopril"`, `"statins"`, `"contrast_dye"`, `"iodine"`, `"latex"`, `"lidocaine"`, `"insulin"`, `"heparin"`

  **Food Allergies:**
  - `"shellfish"`, `"peanuts"`, `"tree_nuts"`, `"eggs"`, `"milk"`, `"dairy"`, `"soy"`, `"wheat"`, `"gluten"`, `"fish"`, `"sesame"`

  **Environmental:**
  - `"pollen"`, `"dust"`, `"dust_mites"`, `"mold"`, `"pet_dander"`, `"cat_dander"`, `"dog_dander"`, `"bee_stings"`, `"insect_stings"`

- `reaction`: **STANDARDIZED REACTIONS:**
  - `"rash"`, `"hives"`, `"itching"`, `"swelling"`, `"anaphylaxis"`, `"shortness_of_breath"`, `"nausea"`, `"vomiting"`, `"diarrhea"`, `"throat_swelling"`, `"difficulty_breathing"`, `"hypotension"`, `"unknown"`

- `assertion`: `"present"`, `"absent"`, or `"possible"`

---

### Vitals (STRICT UNIT CONVENTIONS)

- `kind`: **MUST be one of:**
  - `"temp"` - Temperature
  - `"hr"` - Heart rate
  - `"bp"` - Blood pressure
  - `"rr"` - Respiratory rate
  - `"spo2"` - Oxygen saturation

- `value`: **NUMBERS ONLY, NO UNITS**

  | Kind | Format | Examples | Implied Unit |
  |------|--------|----------|--------------|
  | `temp` | Decimal | `"98.6"`, `"101.2"` | °F |
  | `hr` | Integer | `"72"`, `"88"` | bpm |
  | `bp` | sys/dia | `"120/80"`, `"140/90"` | mmHg |
  | `rr` | Integer | `"16"`, `"20"` | /min |
  | `spo2` | Integer | `"98"`, `"95"` | % |

  ❌ `"98.6°F"`, `"72 bpm"`, `"98%"` (no units!)

---

## Example Entry

```json
{
  "encounter_id": "D2N068",
  "source_type": "dialogue",
  "gold": {
    "demographics": {
      "age": "58",
      "sex": "male",
      "race_ethnicity": "african_american",
      "sexual_orientation": null
    },
    "meds": [
      {"name_norm": "lisinopril", "assertion": "present", "dose": "10 mg", "freq": "once_daily"},
      {"name_norm": "metformin", "assertion": "present", "dose": "500 mg", "freq": "twice_daily"}
    ],
    "allergies": [
      {"substance_norm": "penicillin", "assertion": "present", "reaction": "rash"},
      {"substance_norm": "shellfish", "assertion": "present", "reaction": "anaphylaxis"}
    ],
    "vitals": [
      {"kind": "bp", "value": "140/90"},
      {"kind": "hr", "value": "88"},
      {"kind": "temp", "value": "98.6"}
    ]
  }
}
```

---

## Scoring Summary

| Category | Match Type | Description |
|----------|------------|-------------|
| **Demographics** | Exact | Field-by-field exact match after normalization |
| **Medications** | Strict | `name_norm` + `assertion` for entity; `dose`, `freq` for attributes |
| **Allergies** | Strict | `substance_norm` + `assertion` for entity; `reaction` for attribute |
| **Vitals** | Exact | `kind` + `value` (numbers only) |

## Converting for Evaluation

After annotating in JSON format, convert to JSONL:

```python
import json

with open("data/aci_extractor_gold_15.jsonl") as f:
    entries = json.load(f)

with open("data/aci_extractor_gold_15.jsonl", "w") as f:
    for entry in entries:
        f.write(json.dumps(entry) + "\n")
```
