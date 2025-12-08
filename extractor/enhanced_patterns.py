# extractor/enhanced_patterns.py
# ==============================
# Enhanced pattern matching for clinical information extraction
# Addresses gaps in NER-based extraction for conversational clinical text
# ==============================

import re
from typing import List, Tuple, Optional, Set

# ============================================================
# Chief Complaint Detection
# ============================================================

# Phrases that indicate the patient is stating their chief complaint
CHIEF_COMPLAINT_CUES = [
    r"i(?:'m| am) here (?:because|for|to)",
    r"i came in (?:because|for|to|due to)",
    r"i(?:'ve| have) been having",
    r"i(?:'ve| have) been experiencing",
    r"i(?:'ve| have) got",
    r"my (?:main |chief )?(?:complaint|concern|problem|issue) is",
    r"what(?:'s| is) (?:bothering|troubling) me is",
    r"i(?:'m| am) (?:worried|concerned) about",
    r"i need help with",
    r"i(?:'m| am) having (?:trouble|problems?) with",
    r"i(?:'ve| have) noticed",
    r"recently i(?:'ve| have)",
    r"for the past",
    r"i woke up with",
    r"started (?:having|feeling|experiencing)",
]

# Social/greeting phrases to skip when looking for chief complaint
GREETING_PATTERNS = [
    r"^(?:hi|hello|hey|good (?:morning|afternoon|evening))[\s,\.!]*",
    r"^(?:nice|good|great) to (?:see|meet) you",
    r"^how(?:'re| are) you",
    r"^i(?:'m| am) (?:doing )?(?:good|well|fine|okay|ok)",
    r"^thank(?:s| you)",
    r"^yes[\s,\.]*(?:hi|hello)?",
    r"^(?:okay|ok|alright|sure)[\s,\.]*$",
]

GREETING_RE = re.compile("|".join(GREETING_PATTERNS), re.IGNORECASE)
CHIEF_COMPLAINT_CUE_RE = re.compile("|".join(CHIEF_COMPLAINT_CUES), re.IGNORECASE)


def is_greeting_or_filler(text: str) -> bool:
    """Check if text is just a greeting or filler phrase."""
    text = text.strip()
    if len(text) < 5:
        return True
    if GREETING_RE.search(text):
        # Check if there's substantial content after the greeting
        cleaned = GREETING_RE.sub("", text).strip()
        if len(cleaned) < 15:
            return True
    return False


def extract_chief_complaint(patient_turns: List[dict], doctor_turns: List[dict] = None) -> Optional[str]:
    """
    Extract the chief complaint from patient utterances.
    
    Strategy:
    1. Skip greetings and filler responses
    2. Look for explicit chief complaint cues
    3. Look for responses to "what brings you in" type questions
    4. Fall back to first substantive patient statement about symptoms
    """
    # Look for explicit chief complaint cues first
    for turn in patient_turns:
        text = turn["text"].strip()
        if is_greeting_or_filler(text):
            continue
        
        match = CHIEF_COMPLAINT_CUE_RE.search(text)
        if match:
            # Extract from the cue onwards, up to 200 chars
            start = match.start()
            complaint = text[start:start + 200]
            # Clean up and return first sentence-ish chunk
            complaint = re.split(r'[.!?]', complaint)[0].strip()
            if len(complaint) > 10:
                return complaint[:160]
    
    # Fall back: find first substantive patient turn with symptom-like content
    symptom_indicators = [
        r"\b(?:pain|ache|hurt|sore|tender)\b",
        r"\b(?:cough|sneez|wheez|breath)\b",
        r"\b(?:nausea|vomit|dizz|faint)\b",
        r"\b(?:tired|fatigue|weak|exhaust)\b",
        r"\b(?:fever|chill|sweat)\b",
        r"\b(?:itch|rash|swell|bump)\b",
        r"\b(?:can(?:'t|not) (?:sleep|eat|breathe|walk))\b",
        r"\b(?:trouble|difficulty|problem)\b",
        r"\b(?:worse|better|constant|intermittent)\b",
    ]
    symptom_re = re.compile("|".join(symptom_indicators), re.IGNORECASE)
    
    for turn in patient_turns:
        text = turn["text"].strip()
        if is_greeting_or_filler(text):
            continue
        if len(text) < 15:
            continue
        if symptom_re.search(text):
            # Found symptom-related content
            first_sentence = re.split(r'[.!?]', text)[0].strip()
            if len(first_sentence) > 10:
                return first_sentence[:160]
    
    # Ultimate fallback: first non-greeting patient turn
    for turn in patient_turns:
        text = turn["text"].strip()
        if not is_greeting_or_filler(text) and len(text) > 15:
            first_sentence = re.split(r'[.!?]', text)[0].strip()
            return first_sentence[:160]
    
    return None


# ============================================================
# Symptom Extraction (Rule-based supplement)
# ============================================================

# Common symptoms with their normalized names
SYMPTOM_PATTERNS = {
    # Pain symptoms
    r"\b(?:chest\s+)?pain\b": "chest pain",
    r"\bheadache\b": "headache",
    r"\b(?:head\s+)?(?:hurt|hurts|hurting)\b": "headache",
    r"\bback\s*(?:pain|ache)\b": "back pain",
    r"\babdominal?\s*(?:pain|cramp)s?\b": "abdominal pain",
    r"\bstomach\s*(?:pain|ache|cramp)s?\b": "abdominal pain",
    r"\bjoint\s*(?:pain|ache)s?\b": "joint pain",
    r"\bmuscle\s*(?:pain|ache)s?\b": "muscle pain",
    r"\bsore\s+throat\b": "sore throat",
    
    # Respiratory
    r"\b(?:short(?:ness)?|difficult[yi]?)\s*(?:of\s+)?breath(?:ing)?\b": "shortness of breath",
    r"\bcan(?:'t|not)\s+(?:catch\s+(?:my\s+)?breath|breathe)\b": "shortness of breath",
    r"\bcough(?:ing)?\b": "cough",
    r"\bwheez(?:e|ing)\b": "wheezing",
    r"\bcongestion\b": "congestion",
    r"\brunny\s+nose\b": "rhinorrhea",
    r"\bstuffy\s+nose\b": "nasal congestion",
    
    # GI symptoms
    r"\bnausea(?:ted)?\b": "nausea",
    r"\bvomit(?:ing|ed)?\b": "vomiting",
    r"\bthrow(?:ing)?\s+up\b": "vomiting",
    r"\bdiarrhea\b": "diarrhea",
    r"\bconstipat(?:ed|ion)\b": "constipation",
    r"\bheartburn\b": "heartburn",
    r"\bacid\s+reflux\b": "acid reflux",
    r"\bindigestion\b": "indigestion",
    r"\bbloat(?:ed|ing)\b": "bloating",
    r"\bloss\s+of\s+appetite\b": "decreased appetite",
    r"\bno\s+appetite\b": "decreased appetite",
    
    # Neurological
    r"\bdizz(?:y|iness)\b": "dizziness",
    r"\blightheaded(?:ness)?\b": "lightheadedness",
    r"\bfaint(?:ing|ed)?\b": "syncope",
    r"\bpass(?:ed|ing)?\s+out\b": "syncope",
    r"\bnumb(?:ness)?\b": "numbness",
    r"\btingl(?:e|ing)\b": "tingling",
    r"\btremor\b": "tremor",
    r"\bshak(?:y|ing)\b": "tremor",
    r"\bseizure\b": "seizure",
    r"\bconfus(?:ed|ion)\b": "confusion",
    r"\bmemory\s+(?:loss|problems?)\b": "memory impairment",
    
    # Constitutional
    r"\bfever\b": "fever",
    r"\bchills?\b": "chills",
    r"\bnight\s+sweats?\b": "night sweats",
    r"\bfatigue[d]?\b": "fatigue",
    r"\btired(?:ness)?\b": "fatigue",
    r"\bexhaust(?:ed|ion)\b": "fatigue",
    r"\bweak(?:ness)?\b": "weakness",
    r"\bweight\s+(?:loss|gain)\b": "weight change",
    r"\blost\s+(?:some\s+)?weight\b": "weight loss",
    r"\bgained?\s+(?:some\s+)?weight\b": "weight gain",
    
    # Sleep
    r"\binsomnia\b": "insomnia",
    r"\bcan(?:'t|not)\s+sleep\b": "insomnia",
    r"\btrouble\s+sleep(?:ing)?\b": "insomnia",
    r"\bsleep\s+(?:problem|difficult|issue)s?\b": "sleep disturbance",
    
    # Skin
    r"\brash\b": "rash",
    r"\bitch(?:y|ing)?\b": "pruritus",
    r"\bswell(?:ing|ed)?\b": "swelling",
    r"\bedema\b": "edema",
    r"\bruise[ds]?\b": "bruising",
    r"\bbleed(?:ing)?\b": "bleeding",
    
    # Cardiac
    r"\bpalpitation[s]?\b": "palpitations",
    r"\bheart\s+(?:racing|pounding|flutter)\b": "palpitations",
    r"\birregular\s+heart\s*beat\b": "irregular heartbeat",
    
    # Urinary
    r"\bfrequent\s+urinat(?:ion|ing)\b": "urinary frequency",
    r"\bburn(?:ing|s)?\s+(?:when|during)\s+(?:I\s+)?(?:pee|urinat)\b": "dysuria",
    r"\bpain(?:ful)?\s+urinat(?:ion|ing)\b": "dysuria",
    r"\bblood\s+in\s+(?:my\s+)?urine\b": "hematuria",
    
    # Mental health
    r"\banxi(?:ety|ous)\b": "anxiety",
    r"\bdepress(?:ed|ion)\b": "depression",
    r"\bstress(?:ed)?\b": "stress",
    r"\bpanic\s+attack\b": "panic attacks",
}

SYMPTOM_PATTERN_COMPILED = {
    re.compile(pattern, re.IGNORECASE): norm
    for pattern, norm in SYMPTOM_PATTERNS.items()
}


def extract_symptoms_by_pattern(text: str) -> List[Tuple[str, str, int, int]]:
    """
    Extract symptoms using pattern matching.
    
    Returns:
        List of (surface_form, normalized_name, start, end)
    """
    results = []
    for pattern, normalized in SYMPTOM_PATTERN_COMPILED.items():
        for match in pattern.finditer(text):
            results.append((match.group(0), normalized, match.start(), match.end()))
    return results


# ============================================================
# Medication Extraction (Rule-based supplement)
# ============================================================

# Common medication names (generic and brand)
COMMON_MEDICATIONS = {
    # Pain/Inflammation
    "ibuprofen", "advil", "motrin", "naproxen", "aleve", "aspirin", "tylenol",
    "acetaminophen", "excedrin", "celebrex", "meloxicam",
    
    # Cardiovascular
    "lisinopril", "losartan", "amlodipine", "metoprolol", "atenolol", "carvedilol",
    "hydrochlorothiazide", "hctz", "furosemide", "lasix", "spironolactone",
    "atorvastatin", "lipitor", "simvastatin", "zocor", "rosuvastatin", "crestor",
    "pravastatin", "warfarin", "coumadin", "eliquis", "apixaban", "xarelto",
    "rivaroxaban", "plavix", "clopidogrel", "nitroglycerin",
    
    # Diabetes
    "metformin", "glucophage", "glipizide", "glyburide", "glimepiride",
    "insulin", "lantus", "humalog", "novolog", "ozempic", "trulicity",
    "jardiance", "farxiga", "invokana",
    
    # Respiratory
    "albuterol", "proventil", "ventolin", "advair", "symbicort", "breo",
    "spiriva", "singulair", "montelukast", "prednisone", "fluticasone",
    "flonase", "cetirizine", "zyrtec", "loratadine", "claritin",
    "diphenhydramine", "benadryl", "fexofenadine", "allegra",
    
    # GI
    "omeprazole", "prilosec", "pantoprazole", "protonix", "nexium",
    "esomeprazole", "famotidine", "pepcid", "ranitidine", "zantac",
    "sucralfate", "carafate", "ondansetron", "zofran", "metoclopramide",
    "reglan", "docusate", "colace", "miralax", "polyethylene glycol",
    "loperamide", "imodium",
    
    # Psychiatric
    "sertraline", "zoloft", "fluoxetine", "prozac", "escitalopram", "lexapro",
    "citalopram", "celexa", "paroxetine", "paxil", "venlafaxine", "effexor",
    "duloxetine", "cymbalta", "bupropion", "wellbutrin", "trazodone",
    "mirtazapine", "remeron", "alprazolam", "xanax", "lorazepam", "ativan",
    "clonazepam", "klonopin", "diazepam", "valium", "buspirone", "buspar",
    "quetiapine", "seroquel", "aripiprazole", "abilify", "risperidone",
    "risperdal", "olanzapine", "zyprexa", "lithium",
    
    # Thyroid
    "levothyroxine", "synthroid", "armour thyroid", "methimazole", "tapazole",
    
    # Antibiotics
    "amoxicillin", "augmentin", "azithromycin", "zithromax", "z-pack",
    "ciprofloxacin", "cipro", "levofloxacin", "levaquin", "doxycycline",
    "metronidazole", "flagyl", "clindamycin", "sulfamethoxazole", "bactrim",
    "nitrofurantoin", "macrobid", "cephalexin", "keflex",
    
    # Sleep
    "zolpidem", "ambien", "eszopiclone", "lunesta", "melatonin",
    
    # Other common
    "gabapentin", "neurontin", "pregabalin", "lyrica", "tramadol",
    "cyclobenzaprine", "flexeril", "methocarbamol", "robaxin",
    "sumatriptan", "imitrex", "topiramate", "topamax",
    "finasteride", "propecia", "tamsulosin", "flomax",
    "sildenafil", "viagra", "tadalafil", "cialis",
    "vitamin d", "vitamin b12", "folic acid", "iron", "calcium",
    "fish oil", "omega-3", "multivitamin", "probiotics",
}

# Patterns for medication context
MED_CONTEXT_PATTERNS = [
    r"\b(?:tak(?:e|ing|en)|on|use|using|prescribed|started?|stopped?|quit)\b",
    r"\b(?:mg|milligram|mcg|microgram)\b",
    r"\b(?:daily|twice|once|every|morning|evening|night|bedtime)\b",
    r"\b(?:pill|tablet|capsule|medication|medicine|drug|prescription)\b",
]

MED_CONTEXT_RE = re.compile("|".join(MED_CONTEXT_PATTERNS), re.IGNORECASE)


def extract_medications_by_pattern(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract medications using pattern matching against known drug names.
    
    Returns:
        List of (medication_name, start, end)
    """
    results = []
    text_lower = text.lower()
    
    for med in COMMON_MEDICATIONS:
        # Create word-boundary pattern for each medication
        pattern = re.compile(r'\b' + re.escape(med) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            results.append((match.group(0), match.start(), match.end()))
    
    return results


def sentence_has_med_context(sentence: str) -> bool:
    """Check if a sentence has medication-related context."""
    return bool(MED_CONTEXT_RE.search(sentence))


# ============================================================
# Clinical Question Filtering for QA Extraction
# ============================================================

# Questions that are clinically relevant and worth extracting answers for
CLINICAL_QUESTION_PATTERNS = [
    # Symptom questions
    r"(?:do|does|are|is|have|has)\s+(?:you|it|the)\s+(?:have|had|having|feel|felt|feeling|experience|experienced)",
    r"(?:how|what)\s+(?:long|often|much|many|severe)",
    r"(?:when|where)\s+(?:did|does|do)\s+(?:it|you|the|this)",
    r"(?:any|some)\s+(?:pain|symptoms?|problems?|issues?|trouble|difficulty)",
    r"(?:does|do)\s+(?:anything|it)\s+(?:make|help|worsen|improve)",
    r"(?:what|which)\s+(?:makes?|triggers?|causes?|helps?)",
    
    # Medical history
    r"(?:have|has)\s+(?:you|anyone)\s+(?:ever|been|had)",
    r"(?:any|family)\s+(?:history|medical)",
    r"(?:previous|past|prior)\s+(?:surgery|surgeries|procedures?|hospitalizations?)",
    r"(?:diagnosed|told)\s+(?:you\s+)?(?:have|had|with)",
    
    # Medication questions
    r"(?:what|which)\s+(?:medications?|medicines?|drugs?|pills?)",
    r"(?:are|were)\s+(?:you|they)\s+(?:taking|on|prescribed)",
    r"(?:any|some)\s+(?:medications?|allergies)",
    r"(?:do|did)\s+(?:you|it)\s+(?:take|try|use)",
    
    # Lifestyle
    r"(?:do|did)\s+you\s+(?:smoke|drink|exercise|use)",
    r"(?:how\s+)?(?:much|many|often)\s+(?:do|did)\s+you",
    
    # Functional status
    r"(?:can|could|able)\s+(?:you|to)\s+(?:walk|sleep|eat|work|function)",
    r"(?:affect|impact|interfere)\s+(?:your|with|daily)",
]

CLINICAL_QUESTION_RE = re.compile("|".join(CLINICAL_QUESTION_PATTERNS), re.IGNORECASE)

# Questions to skip (non-clinical)
SKIP_QUESTION_PATTERNS = [
    r"(?:how|what)\s+(?:is|are)\s+(?:your\s+)?(?:name|age|address|phone|insurance)",
    r"(?:can|could)\s+(?:you|I)\s+(?:see|have|get)\s+(?:your|an|a)\s+(?:id|card|insurance)",
    r"(?:date\s+of\s+birth|birthday|dob)",
    r"(?:emergency\s+contact|next\s+of\s+kin)",
    r"(?:pharmacy|pharmacist)",
    r"(?:appointment|schedule|follow.?up)",
    r"(?:questions?\s+for\s+me|anything\s+else)",
]

SKIP_QUESTION_RE = re.compile("|".join(SKIP_QUESTION_PATTERNS), re.IGNORECASE)


def is_clinically_relevant_question(question_text: str) -> bool:
    """
    Determine if a doctor's question is clinically relevant
    and worth extracting the patient's answer.
    """
    # Skip administrative questions
    if SKIP_QUESTION_RE.search(question_text):
        return False
    
    # Accept clinically relevant questions
    if CLINICAL_QUESTION_RE.search(question_text):
        return True
    
    # Accept questions about specific body parts or symptoms
    body_part_symptom_re = re.compile(
        r"\b(?:chest|head|stomach|back|leg|arm|heart|lung|kidney|liver|"
        r"pain|ache|swell|numb|tingle|bleed|burn|itch)\b",
        re.IGNORECASE
    )
    if body_part_symptom_re.search(question_text):
        return True
    
    return False

