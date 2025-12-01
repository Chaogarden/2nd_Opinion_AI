import re

# Duration, severity, onset
DURATION_RE = re.compile(r"\b(?:for|x)\s*(\d+\s*(?:d(?:ays?)?|w(?:eeks?)?|mo(?:nths?)?|y(?:ears?)?))\b", re.I)
SEVERITY_RE = re.compile(r"\b(mild|moderate|severe)\b", re.I)
ONSET_RE = re.compile(r"\b(sudden|gradual|acute|chronic|subacute)\b", re.I)

# Vitals
VITALS = {
    "temp": re.compile(r"\b(?:t(?:emp(?:erature)?)?)\s*[:=]?\s*(\d{2}(?:\.\d)?)\s*°?\s*([CF])?\b", re.I),
    "hr": re.compile(r"\b(?:hr|heart\s*rate)\s*[:=]?\s*(\d{2,3})\b", re.I),
    "bp": re.compile(r"\b(?:bp|blood\s*pressure)\s*[:=]?\s*(\d{2,3}\s*/\s*\d{2,3})\b", re.I),
    "rr": re.compile(r"\b(?:rr|resp(?:iratory)?\s*rate)\s*[:=]?\s*(\d{2})\b", re.I),
    "spo2": re.compile(r"\b(?:spo2|o2\s*saturation)\s*[:=]?\s*(\d{2})%\b", re.I),
}

# Medication attributes
DOSE_RE = re.compile(r"\b(\d+\.?\d*\s*(?:mg|mcg|g|ml|units))\b", re.I)
FREQ_RE = re.compile(r"\b(q\d+h|daily|qd|bid|tid|qid|qhs|prn|once daily|twice daily|every\s*\d+\s*(?:h|hours?))\b", re.I)
ROUTE_RE = re.compile(r"\b(po|oral|inh|inhal(?:er)?|iv|im|subcut|sq|sc|topical|sl)\b", re.I)
FORM_RE = re.compile(r"\b(tab(?:let)?|cap(?:sule)?|inhaler|syrup|suspension|cream|ointment|solution|spray)\b", re.I)
PRN_RE = re.compile(r"\bprn\b", re.I)

# Negation / uncertainty cues (fallback; prefer medspaCy ConText)
NEG_CUES = ["no","not","denies","without","never"]
UNCERT_CUES = ["possible","consider","?","likely","unlikely","rule out","r/o"]

# Simple risk factors keyword list (expand later)
RISK_FACTORS = ["smoker","smoking","vape","alcohol","pregnant","obese","overweight","diabetic"]