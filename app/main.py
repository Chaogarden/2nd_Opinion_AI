# main.py
# ==============================
# Streamlit UI for 2nd Opinion AI
# - Audio upload -> Faster-Whisper ASR
# - Optional speaker diarization (pyannote)
# - Auto-map speakers to DOCTOR / PATIENT (heuristic)
# - Placeholders for Extraction / RAG / Reasoning / Output
# ==============================

import sys
from pathlib import Path
import tempfile
import re
from collections import defaultdict

import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Faster-Whisper transcription (already updated to accept diarization flags)
from core.audio.transcribe import transcribe_file


# ---------------------------
# Global page settings + CSS
# ---------------------------
st.set_page_config(page_title="2nd Opinion AI", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #1a1a1a; }
    [data-testid="stSidebar"] { background-color: #2d2d2d; }
    h1, h2, h3 { color: #ffffff !important; }
    .stMarkdown, p, label { color: #e0e0e0 !important; }
    .stTextArea textarea, .stTextInput input {
        background-color: #3a3a3a !important; color: #ffffff !important;
        border: 1px solid #4a4a4a !important;
    }
    [data-testid="stFileUploader"] { background-color: #3a3a3a; border-radius: 8px; padding: 10px; }
    .stButton button {
        background-color: #6366f1 !important; color: white !important; border: none !important;
        border-radius: 8px !important; padding: 0.5rem 1rem !important; font-weight: 500 !important;
    }
    .stButton button:hover { background-color: #4f46e5 !important; }
    .stAlert {
        background-color: #3a3a3a !important; color: #22d3ee !important; border-left: 4px solid #22d3ee !important;
    }
    .stJson { background-color: #2d2d2d !important; border-radius: 8px !important; padding: 15px !important; }
    .stMarkdown h3 { color: #ffffff !important; border-bottom: 2px solid #6366f1; padding-bottom: 8px; margin-top: 20px; }
    .stCaption { color: #9ca3af !important; }
    [data-testid="stSidebar"] h2 { color: #22d3ee !important; }
    hr { border-color: #4a4a4a !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------
# Title + subtitle
# ---------------------------
st.title("2nd Opinion AI — MVP")
st.caption("Local ASR/OCR → Extraction → RAG → Reasoning → SOAP → PDF/TTS (skeleton)")


# ---------------------------
# Heuristic role mapper
# ---------------------------
DOC_KEYWORDS = [
    "assessment", "plan", "follow up", "follow-up", "differential",
    "start you on", "i recommend", "i'll prescribe", "i will prescribe",
    "prescribe", "dose", "mg", "b.i.d", "bid", "t.i.d", "tid", "q.d", "qd",
    "referral", "imaging", "x-ray", "ct", "mri", "ultrasound",
    "labs", "cbc", "cmp", "a1c", "lipid", "troponin",
    "blood pressure", "bp", "heart rate", "hr", "respiratory rate", "rr", "spo2", "oxygen",
    "physical exam", "on exam", "exam", "auscultation", "palpation",
    "contraindicated", "indicated", "guideline", "screening", "uspstf", "nice"
]
DOC_IMPERATIVES = [
    "take", "start", "stop", "begin", "increase", "decrease", "schedule",
    "follow", "avoid", "return", "monitor", "check", "continue"
]
DOSAGE_RE = re.compile(r"\b\d+(\.\d+)?\s*mg\b", re.IGNORECASE)

def score_doctorish(text: str) -> int:
    t = text.lower()
    score = 0
    # keyword hits
    for kw in DOC_KEYWORDS:
        if kw in t:
            score += 2
    # imperatives at beginning of sentence phrases (very rough)
    for imp in DOC_IMPERATIVES:
        if re.search(rf"(?:^|[\.!\?]\s+)({imp})\b", t):
            score += 2
    # dosage patterns
    if DOSAGE_RE.search(t):
        score += 3
    # “we’ll / let's / i recommend” style
    if "we'll" in t or "let's" in t or "i recommend" in t:
        score += 1
    return score

def infer_role_mapping(segments):
    """
    Returns a dict: { 'SPEAKER_00': 'DOCTOR', 'SPEAKER_01': 'PATIENT', ... }
    Heuristic: speaker with highest 'doctorish' score -> DOCTOR. Others -> PATIENT/OTHER_i.
    """
    per_spk_score = defaultdict(int)
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        per_spk_score[spk] += score_doctorish(seg.get("text", ""))

    # If no speakers or all zero, bail to default order
    if not per_spk_score:
        return {}

    # Rank speakers by score (desc)
    ranked = sorted(per_spk_score.items(), key=lambda x: x[1], reverse=True)
    mapping = {}
    if ranked:
        mapping[ranked[0][0]] = "DOCTOR"
        # Everyone else as PATIENT/OTHER_i
        others = [spk for spk, _ in ranked[1:]]
        if others:
            # pick the first other as PATIENT
            mapping[others[0]] = "PATIENT"
            # remaining as OTHER_i
            for i, spk in enumerate(others[1:], start=1):
                mapping[spk] = f"OTHER_{i}"
    return mapping

def apply_role_mapping(segments, mapping):
    """Append 'role' to segments; if unknown, keep original speaker id."""
    out = []
    for seg in segments:
        spk = seg.get("speaker")
        role = mapping.get(spk, spk or "UNKNOWN")
        out.append({**seg, "role": role})
    return out

def swap_doctor_patient(mapping):
    """Swap DOCTOR and PATIENT roles in an existing mapping."""
    swapped = {}
    for k, v in mapping.items():
        if v == "DOCTOR": swapped[k] = "PATIENT"
        elif v == "PATIENT": swapped[k] = "DOCTOR"
        else: swapped[k] = v
    return swapped


# ---------------------------
# Sidebar inputs
# ---------------------------
with st.sidebar:
    st.header("Inputs")

    # Audio
    audio_file = st.file_uploader("Upload audio (WAV/MP3/M4A)", type=["wav", "mp3", "m4a"])

    # Optional labs image/PDF (future OCR)
    labs_img = st.file_uploader("Upload lab image/PDF (optional)", type=["png", "jpg", "jpeg", "pdf"])

    # Free-text alternative
    free_text = st.text_area(
        "Or paste a short patient summary",
        height=120,
        placeholder="e.g., 45M chest pain on exertion, diabetic, smoker..."
    )

    st.divider()

    # Diarization controls
    diarize_on = st.checkbox("Diarize speakers (pyannote)", value=True,
                             help="Requires pyannote.audio + HF token. Uses segment-level speaker tags.")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        min_spk = st.number_input("Min speakers", value=2, min_value=1, max_value=8, step=1,
                                  help="Set equal to Max for fixed speaker count.")
    with col_d2:
        max_spk = st.number_input("Max speakers", value=2, min_value=1, max_value=8, step=1)

    st.divider()
    auto_role = st.checkbox("Auto-label DOCTOR / PATIENT", value=True,
                            help="Heuristic based on medical keywords, dosage patterns, and clinician phrasing.")


# ---------------------------
# Transcript (ASR wired here)
# ---------------------------
st.subheader("Transcript")

# Persist across reruns
if "transcript_result" not in st.session_state:
    st.session_state["transcript_result"] = None
if "role_mapping" not in st.session_state:
    st.session_state["role_mapping"] = {}

# 1) If an audio file is uploaded, transcribe it
if audio_file is not None:
    suffix = Path(audio_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    st.info(f"Transcribing: {audio_file.name}")
    with st.spinner("Running Faster-Whisper..."):
        result = transcribe_file(
            tmp_path,
            with_diarization=diarize_on,
            min_speakers=int(min_spk),
            max_speakers=int(max_spk),
            hf_token="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",
        )
        st.session_state["transcript_result"] = result
        # reset mapping on new transcript
        st.session_state["role_mapping"] = {}

# 2) If no audio but free text is provided, treat it as the transcript
elif free_text.strip():
    st.session_state["transcript_result"] = {
        "text": free_text.strip(),
        "segments": [],
        "info": {"language": "en", "duration": None},
    }
    st.session_state["role_mapping"] = {}

# 3) Display transcript (or prompt user)
if st.session_state["transcript_result"] is not None:
    tr = st.session_state["transcript_result"]
    transcript_text = tr.get("text", "")
    st.text_area("ASR Transcript", value=transcript_text, height=200)

    # Colors
    speaker_palette = {
        "DOCTOR": "#22d3ee",   # cyan
        "PATIENT": "#a78bfa",  # violet
    }
    # generic palette for other speakers
    generic_palette = [
        "#f59e0b", "#34d399", "#f472b6", "#60a5fa", "#f87171", "#c084fc", "#10b981"
    ]

    # Segments (with speaker/role tags)
    with st.expander("Show segments (diarization / roles)"):
        segments = tr.get("segments", [])
        if segments:
            # Build or use existing role mapping
            if diarize_on and auto_role:
                if not st.session_state["role_mapping"]:
                    st.session_state["role_mapping"] = infer_role_mapping(segments)

                # Render a swap button
                swap_cols = st.columns(3)
                if swap_cols[0].button("Swap DOCTOR ↔ PATIENT"):
                    st.session_state["role_mapping"] = swap_doctor_patient(st.session_state["role_mapping"])

                mapping = st.session_state["role_mapping"]
                segs_with_roles = apply_role_mapping(segments, mapping)
            else:
                mapping = {}
                segs_with_roles = [{**s, "role": s.get("speaker", "UNKNOWN")} for s in segments]

            # Build legend
            roles_present = list(dict.fromkeys(s["role"] for s in segs_with_roles))  # preserve order
            legend_cols = st.columns(max(1, min(6, len(roles_present))))
            other_color_iter = iter(generic_palette)
            role_color = {}
            for i, role in enumerate(roles_present):
                if role in ("DOCTOR", "PATIENT"):
                    color = speaker_palette[role]
                else:
                    color = role_color.get(role) or next(other_color_iter, "#9ca3af")
                    role_color[role] = color
                legend_cols[i].markdown(f"<div style='color:{color}; font-weight:600'>{role}</div>", unsafe_allow_html=True)

            # Print each segment
            for seg in segs_with_roles:
                start = seg.get("start", 0.0)
                end = seg.get("end", 0.0)
                text = seg.get("text", "").strip()
                role = seg.get("role", "UNKNOWN")
                color = speaker_palette.get(role, role_color.get(role, "#9ca3af"))
                st.markdown(
                    f"<div style='color:{color}'>[{start:.1f}–{end:.1f}s] ({role}) {text}</div>",
                    unsafe_allow_html=True
                )
        else:
            st.write("No segments (free-text mode or minimal output).")
else:
    st.write("Upload audio or paste text to see a transcript.")

# ---------------------------
# Structured slots placeholder
# ---------------------------
st.subheader("Structured slots (FHIR-lite)")
slots = {
    "patient": {"age": None, "sex": None},
    "chief_complaint": "",
    "hpi": {"onset": "", "duration": "", "modifiers": [], "associated_symptoms": []},
    "meds": [],
    "allergies": [],
    "vitals": {"temp": None, "hr": None, "bp": None, "rr": None, "spo2": None},
    "labs": [],
    "exam_findings": [],
}
st.json(slots)

# ---------------------------
# RAG evidence placeholder
# ---------------------------
st.subheader("RAG evidence")
st.write("_Top-k guideline snippets will appear here in a future commit._")

# ---------------------------
# Reasoner output placeholder
# ---------------------------
st.subheader("Draft differential + plan (with citations)")
st.write("_Reasoner output placeholder_")

# ---------------------------
# Action buttons
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Export SOAP as PDF"):
        st.info("PDF export to be implemented.")
with col2:
    if st.button("Speak summary (TTS)"):
        st.info("TTS to be implemented.")