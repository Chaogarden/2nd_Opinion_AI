# app/main.py
# ==============================
# 2nd Opinion AI - ChatGPT-style Streamlit UI
# 
# Features:
# - Dark glossy theme with purple accents
# - ChatGPT-style chat interface
# - Multiple input modes: mic, .wav upload, free text, ACI-Bench
# - Settings in sidebar
# - Narrative response with typing animation
# - Expandable "Full Analysis" section
# ==============================

import sys
import time
from pathlib import Path
import tempfile
import json

import streamlit as st

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.audio.transcribe import transcribe_file
from core.dialogue import (
    score_doctorish, infer_role_mapping, apply_role_mapping, swap_doctor_patient
)
from core.llm_config import (
    LLMConfig, get_llm_mode, get_default_models, get_available_ollama_models
)
from core.pipeline import run_clinical_pipeline, clear_caches
from core.soap_sum_small import generate_soap_note
from diagnoser.arbiter import summarize_arbiter_result
from diagnoser.diagnosis_explainer import DiagnosisExplanation

# ACI-Bench dataset integration
from data.load_ACI_dataset import (
    make_transcript_result_from_aci,
    get_prebuilt_role_mapping_from_aci,
    extract_aci_dialogues
)


# ============================================================
# Page Config & Global CSS (Dark Glossy Theme + Purple Accents)
# ============================================================

st.set_page_config(
    page_title="2nd Opinion AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark glossy background */
    .stApp {
        background: linear-gradient(145deg, #2a2a2a 0%, #1e1e1e 10%, #141414 50%, #0c0c0c 90%, #080808 100%);
        min-height: 100vh;
    }
    
    /* Header/toolbar bar - match background */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stAppHeader, .stToolbar {
        background: transparent !important;
    }
    
    /* Sidebar styling - slightly darker than main with gloss */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #0f0f0f 15%, #0f0f0f 85%, #0a0a0a 100%);
        border-right: 1px solid rgba(139, 92, 246, 0.15);
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #a78bfa !important;
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #c0c0d0 !important;
    }
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    
    /* Text colors */
    h1, h2, h3, h4 { color: #ffffff !important; }
    p, label, .stMarkdown { color: #e0e0e0 !important; }
    
    /* App title */
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        text-align: center;
    }
    .app-subtitle {
        color: #7070a0 !important;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* Input mode selector - radio buttons */
    .stRadio > div {
        display: flex;
        justify-content: center;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .stRadio label {
        background: rgba(20, 20, 22, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem !important;
        color: #c0c0d0 !important;
        transition: all 0.3s ease !important;
    }
    .stRadio label:hover {
        border-color: #8b5cf6 !important;
        background: rgba(139, 92, 246, 0.15) !important;
    }
    .stRadio label[data-checked="true"] {
        background: rgba(139, 92, 246, 0.25) !important;
        border-color: #8b5cf6 !important;
        color: #ffffff !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(18, 18, 20, 0.8);
        border: 2px dashed rgba(139, 92, 246, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
        background: rgba(139, 92, 246, 0.1);
    }
    [data-testid="stFileUploader"] label {
        color: #c0c0d0 !important;
    }
    
    /* Text area */
    .stTextArea textarea {
        background: rgba(18, 18, 20, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
        font-size: 0.95rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.3) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: rgba(18, 18, 20, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div {
        background: rgba(139, 92, 246, 0.3) !important;
    }
    .stSlider > div > div > div > div {
        background: #8b5cf6 !important;
    }
    
    /* Number inputs */
    .stNumberInput input {
        background: rgba(18, 18, 20, 0.9) !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
    }
    
    /* Checkboxes */
    .stCheckbox label {
        color: #c0c0d0 !important;
    }
    
    /* Primary button (Run Analysis) */
    .stButton > button[kind="primary"], 
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.6) !important;
    }
    
    /* Secondary buttons */
    .stButton > button {
        background: rgba(18, 18, 20, 0.9) !important;
        color: #c0c0d0 !important;
        border: 1px solid rgba(139, 92, 246, 0.3) !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        border-color: #8b5cf6 !important;
        background: rgba(139, 92, 246, 0.15) !important;
        color: #ffffff !important;
    }
    
    /* Chat message - Assistant */
    .chat-message {
        background: linear-gradient(145deg, rgba(20, 20, 24, 0.95) 0%, rgba(16, 16, 18, 0.98) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    .chat-message p, .chat-message li {
        color: #e8e8f0 !important;
        line-height: 1.7;
    }
    .chat-message h2 {
        color: #a78bfa !important;
        font-size: 1.4rem;
        margin-bottom: 1rem;
    }
    .chat-message h3 {
        color: #a78bfa !important;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .chat-message strong {
        color: #c4b5fd;
    }
    
    /* Full Analysis expander bar */
    .streamlit-expanderHeader {
        background: transparent !important;
        color: #a78bfa !important;
        font-weight: 500 !important;
    }
    .streamlit-expanderHeader:hover {
        color: #c4b5fd !important;
    }
    .streamlit-expanderContent {
        background: rgba(14, 14, 16, 0.8) !important;
        border-top: 1px solid rgba(139, 92, 246, 0.2) !important;
    }
    
    /* JSON viewer */
    .stJson {
        background: rgba(12, 12, 14, 0.9) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(139, 92, 246, 0.2) !important;
    }
    
    /* Metrics */
    [data-testid="stMetric"] {
        background: rgba(18, 18, 20, 0.8);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }
    [data-testid="stMetricValue"] {
        color: #a78bfa !important;
    }
    [data-testid="stMetricLabel"] {
        color: #9090a0 !important;
    }
    
    /* Alerts/Info boxes */
    .stAlert {
        background: rgba(18, 18, 20, 0.9) !important;
        border-radius: 10px !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #8b5cf6 !important;
    }
    
    /* Divider */
    hr {
        border-color: rgba(139, 92, 246, 0.2) !important;
        margin: 1.5rem 0 !important;
    }
    
    /* Evidence chunks */
    .evidence-chunk {
        background: rgba(18, 18, 22, 0.9);
        border-left: 3px solid #8b5cf6;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Red flags */
    .red-flag {
        background: rgba(80, 20, 20, 0.5);
        border-left: 3px solid #ef4444;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Differential diagnosis */
    .differential {
        background: rgba(20, 40, 50, 0.6);
        border-left: 3px solid #22d3ee;
        padding: 12px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Caption text */
    .stCaption {
        color: #7070a0 !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State Initialization
# ============================================================

def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        # Input/Output state
        "transcript_result": None,
        "role_mapping": {},
        "pipeline_result": None,
        "last_run_id": None,
        
        # ASR settings
        "model_size": "base",
        "diarize_on": True,
        "min_speakers": 2,
        "max_speakers": 2,
        "hf_token": "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",
        "auto_role": True,
        
        # LLM settings
        "llm_mode": get_llm_mode(),
        "diag_model": None,
        "consult_model": None,
        
        # RAG settings
        "k_guidelines": 5,
        "k_merck": 5,
        "skip_consultant": False,
        
        # ACI-Bench settings
        "aci_split": "test",
        "aci_example_idx": 0,
        
        # Input mode
        "input_mode": "Upload .wav"
    }
    
    for key, default_val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_val
    
    # Set default models based on mode
    if st.session_state["diag_model"] is None or st.session_state["consult_model"] is None:
        default_models = get_default_models(st.session_state["llm_mode"])
        if st.session_state["diag_model"] is None:
            st.session_state["diag_model"] = default_models["diagnoser"]
        if st.session_state["consult_model"] is None:
            st.session_state["consult_model"] = default_models["consultant"]


init_session_state()


# ============================================================
# Sidebar Settings
# ============================================================

def render_sidebar():
    """Render the settings sidebar."""
    
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        # ASR Settings
        st.markdown("### Audio Transcription")
        
        st.session_state["model_size"] = st.selectbox(
            "Whisper model size",
            options=["tiny", "base", "small", "medium", "large-v2"],
            index=["tiny", "base", "small", "medium", "large-v2"].index(st.session_state["model_size"]),
            help="Larger models are more accurate but slower."
        )
        
        st.session_state["diarize_on"] = st.checkbox(
            "Enable speaker diarization",
            value=st.session_state["diarize_on"],
            help="Identifies different speakers in audio."
        )
        
        if st.session_state["diarize_on"]:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.session_state["min_speakers"] = st.number_input(
                    "Min speakers", value=st.session_state["min_speakers"],
                    min_value=1, max_value=8, step=1
                )
            with col_d2:
                st.session_state["max_speakers"] = st.number_input(
                    "Max speakers", value=st.session_state["max_speakers"],
                    min_value=1, max_value=8, step=1
                )
            
            st.session_state["hf_token"] = st.text_input(
                "HuggingFace Token",
                type="password",
                value=st.session_state["hf_token"],
                help="Required for speaker diarization."
            )
            
            st.session_state["auto_role"] = st.checkbox(
                "Auto-label DOCTOR / PATIENT",
                value=st.session_state["auto_role"],
                help="Heuristic based on medical keywords."
            )
        
        st.divider()
        
        # LLM Settings
        st.markdown("### Reasoning Models")
        
        llm_mode = st.selectbox(
            "LLM Mode",
            options=["test", "prod"],
            index=0 if st.session_state["llm_mode"] == "test" else 1,
            format_func=lambda x: "Test / Local (Ollama)" if x == "test" else "Production",
            help="Test mode uses local Ollama models."
        )
        
        if llm_mode != st.session_state["llm_mode"]:
            st.session_state["llm_mode"] = llm_mode
            default_models = get_default_models(llm_mode)
            st.session_state["diag_model"] = default_models["diagnoser"]
            st.session_state["consult_model"] = default_models["consultant"]
        
        if st.session_state["llm_mode"] == "test":
            available_models = get_available_ollama_models()
            if available_models:
                diag_idx = available_models.index(st.session_state["diag_model"]) if st.session_state["diag_model"] in available_models else 0
                st.session_state["diag_model"] = st.selectbox(
                    "Diagnoser Model", options=available_models, index=diag_idx
                )
                consult_idx = available_models.index(st.session_state["consult_model"]) if st.session_state["consult_model"] in available_models else 0
                st.session_state["consult_model"] = st.selectbox(
                    "Consultant Model", options=available_models, index=consult_idx
                )
            else:
                st.warning("⚠️ Ollama not running")
                st.session_state["diag_model"] = st.text_input(
                    "Diagnoser Model", value=st.session_state["diag_model"]
                )
                st.session_state["consult_model"] = st.text_input(
                    "Consultant Model", value=st.session_state["consult_model"]
                )
        else:
            st.session_state["diag_model"] = st.text_input(
                "Diagnoser Model", value=st.session_state["diag_model"]
            )
            st.session_state["consult_model"] = st.text_input(
                "Consultant Model", value=st.session_state["consult_model"]
            )
        
        st.divider()
        
        # RAG Settings
        st.markdown("### RAG Settings")
        
        st.session_state["k_guidelines"] = st.slider(
            "Guidelines chunks (k)", 1, 10, st.session_state["k_guidelines"]
        )
        st.session_state["k_merck"] = st.slider(
            "Merck chunks (k)", 1, 10, st.session_state["k_merck"]
        )
        
        st.session_state["skip_consultant"] = st.checkbox(
            "Skip Consultant review",
            value=st.session_state["skip_consultant"],
            help="Faster but less safe - skips critique step."
        )
        
        st.divider()
        
        # ACI-Bench Settings
        st.markdown("### ACI-Bench Testing")
        
        st.session_state["aci_split"] = st.selectbox(
            "Dataset split",
            options=["train", "validation", "test"],
            index=["train", "validation", "test"].index(st.session_state["aci_split"])
        )
        
        max_idx = {"train": 176, "validation": 9, "test": 19}.get(st.session_state["aci_split"], 19)
        st.session_state["aci_example_idx"] = st.number_input(
            "Example index",
            min_value=0,
            max_value=max_idx,
            value=min(st.session_state["aci_example_idx"], max_idx)
        )
        
        st.divider()
        
        # Clear button
        if st.button("🔄 Clear All Results", use_container_width=True):
            st.session_state["transcript_result"] = None
            st.session_state["pipeline_result"] = None
            st.session_state["role_mapping"] = {}
            st.session_state.pop("recorded_audio", None)
            clear_caches()
            st.rerun()


# ============================================================
# Mic Recording Component
# ============================================================

def render_mic_input():
    """Render microphone recording interface."""
    try:
        from streamlit_mic_recorder import mic_recorder
        
        audio_data = mic_recorder(
            start_prompt="🎤 Start Recording",
            stop_prompt="⏹️ Stop Recording",
            just_once=False,
            use_container_width=True,
            format="wav",
            key="mic_recorder"
        )
        
        if audio_data:
            st.audio(audio_data["bytes"], format="audio/wav")
            st.session_state["recorded_audio"] = audio_data["bytes"]
        
    except ImportError:
        st.warning("Mic recording requires `streamlit-mic-recorder`. Install with: `pip install streamlit-mic-recorder`")
        st.session_state["recorded_audio"] = None


# ============================================================
# File Upload Component
# ============================================================

def render_upload_input():
    """Render file upload interface."""
    uploaded_file = st.file_uploader(
        "Upload audio file (WAV, MP3, M4A)",
        type=["wav", "mp3", "m4a"],
        key="audio_upload"
    )
    
    if uploaded_file:
        st.audio(uploaded_file)
    
    return uploaded_file


# ============================================================
# Free Text Input Component
# ============================================================

def render_freetext_input():
    """Render free text input interface."""
    free_text = st.text_area(
        "Enter patient summary or dialogue",
        height=120,
        placeholder="e.g., 45M presenting with chest pain on exertion for 2 weeks. History of diabetes and hypertension...",
        key="free_text_input"
    )
    
    return free_text


# ============================================================
# ACI-Bench Input Component
# ============================================================

def render_aci_input():
    """Render ACI-Bench dataset selection interface."""
    st.caption(f"Split: {st.session_state['aci_split']} | Example: {st.session_state['aci_example_idx']}")
    
    try:
        dialogues = extract_aci_dialogues(
            split=st.session_state["aci_split"],
            max_examples=st.session_state["aci_example_idx"] + 1,
            include_notes=True  # Include doctor's notes for critique
        )
        
        if dialogues and len(dialogues) > st.session_state["aci_example_idx"]:
            dialogue_data = dialogues[st.session_state["aci_example_idx"]]
            
            st.markdown(f"**Encounter ID:** `{dialogue_data['encounter_id']}`")
            
            with st.expander("Preview dialogue"):
                preview = dialogue_data["dialogue"][:500]
                if len(dialogue_data["dialogue"]) > 500:
                    preview += "..."
                st.text(preview)
            
            return dialogue_data
        else:
            st.error("Could not load the selected example.")
            return None
            
    except Exception as e:
        st.error(f"Error loading ACI dataset: {str(e)}")
        return None


# ============================================================
# Build Narrative Output (Critique + Diagnosis Explainers)
# ============================================================

def build_narrative_output(result) -> str:
    """Build the main narrative output using both critique and diagnosis explanations."""
    
    if result.error:
        return f"<strong>Analysis Error</strong><br><br>An issue was encountered while processing: {result.error}"
    
    parts = []
    
    # =========================================
    # SECTION 1: Analysis of Primary Care Doctor (Critique)
    # =========================================
    if result.critique_explanation:
        crit_exp = result.critique_explanation
        
        parts.append("<h2 style='color: #a78bfa; margin-bottom: 0.5rem;'>Analysis of Primary Care Doctor</h2>")
        
        # Overall rating
        rating = crit_exp.get("overall_rating", "fair")
        rating_color = {
            "excellent": "#22c55e",
            "good": "#22c55e", 
            "fair": "#f59e0b",
            "needs_improvement": "#f59e0b",
            "concerning": "#ef4444"
        }.get(rating.lower(), "#6b7280")
        parts.append(f"<p><strong>Overall Rating:</strong> <span style='color: {rating_color}; font-weight: bold;'>{rating.upper()}</span></p>")
        
        # Overall summary
        overall_summary = crit_exp.get("overall_summary", "")
        if overall_summary:
            parts.append(f"<p>{overall_summary}</p>")
        
        # Strengths
        strengths = crit_exp.get("strengths", "")
        if strengths:
            parts.append("<h4 style='margin-top: 1rem;'>Strengths</h4>")
            parts.append(f"<p>{strengths}</p>")
        
        # Concerns
        concerns = crit_exp.get("concerns", "")
        if concerns and concerns.lower() not in ("none identified", "none identified.", "none"):
            parts.append("<h4 style='margin-top: 1rem;'>Areas of Concern</h4>")
            parts.append(f"<p>{concerns}</p>")
        
        # Key assessments
        doc_quality = crit_exp.get("documentation_quality", "")
        diag_reasoning = crit_exp.get("diagnostic_reasoning", "")
        treatment = crit_exp.get("treatment_plan", "")
        safety = crit_exp.get("safety_assessment", "")
        
        if any([doc_quality, diag_reasoning, treatment, safety]):
            parts.append("<h4 style='margin-top: 1rem;'>Detailed Assessment</h4>")
            if doc_quality:
                parts.append(f"<p><strong>Documentation:</strong> {doc_quality}</p>")
            if diag_reasoning:
                parts.append(f"<p><strong>Diagnostic Reasoning:</strong> {diag_reasoning}</p>")
            if treatment:
                parts.append(f"<p><strong>Treatment Plan:</strong> {treatment}</p>")
            if safety:
                parts.append(f"<p><strong>Safety:</strong> {safety}</p>")
        
        # Recommendations
        recommendations = crit_exp.get("recommendations", "")
        if recommendations:
            parts.append("<h4 style='margin-top: 1rem;'>Recommendations</h4>")
            parts.append(f"<p>{recommendations}</p>")
        
        parts.append("<hr style='margin: 1.5rem 0; border-color: rgba(139, 92, 246, 0.3);'>")
    
    # =========================================
    # SECTION 2: AI Powered Second Opinion (Diagnosis)
    # =========================================
    parts.append("<h2 style='color: #a78bfa; margin-bottom: 0.5rem;'>AI Powered Second Opinion</h2>")
    
    if result.diagnosis_explanation:
        dx_exp = result.diagnosis_explanation
        
        # Confidence level
        confidence = dx_exp.get("confidence_level", "moderate")
        conf_color = {"high": "#22c55e", "moderate": "#f59e0b", "low": "#ef4444"}.get(confidence.lower(), "#6b7280")
        parts.append(f"<p><strong>Confidence Level:</strong> <span style='color: {conf_color}; font-weight: bold;'>{confidence.upper()}</span></p>")
        
        # Overall summary
        overall_summary = dx_exp.get("overall_summary", "")
        if overall_summary:
            parts.append(f"<p>{overall_summary}</p>")
        
        # Primary diagnosis
        primary_dx = dx_exp.get("primary_diagnosis", "")
        if primary_dx:
            parts.append("<h4 style='margin-top: 1rem;'>Primary Assessment</h4>")
            parts.append(f"<p><strong>{primary_dx}</strong></p>")
            
            rationale = dx_exp.get("primary_rationale", "")
            if rationale:
                parts.append(f"<p>{rationale}</p>")
        
        # Urgent considerations (red flags)
        urgent = dx_exp.get("urgent_considerations", "")
        if urgent and urgent.lower() not in ("none identified.", "none identified", "none"):
            parts.append("<h4 style='margin-top: 1rem; color: #ef4444;'>Urgent Considerations</h4>")
            parts.append(f"<p>{urgent}</p>")
        
        # Alternative diagnoses
        alt_dx = dx_exp.get("alternative_diagnoses", "")
        if alt_dx and alt_dx.lower() not in ("not available.", "no alternatives identified."):
            parts.append("<h4 style='margin-top: 1rem;'>Differential Diagnoses</h4>")
            parts.append(f"<p>{alt_dx}</p>")
        
        # Key findings
        key_findings = dx_exp.get("key_findings", "")
        if key_findings:
            parts.append("<h4 style='margin-top: 1rem;'>Key Findings</h4>")
            parts.append(f"<p>{key_findings}</p>")
        
        # Recommended next steps
        workup = dx_exp.get("recommended_workup", "")
        if workup:
            parts.append("<h4 style='margin-top: 1rem;'>Recommended Workup</h4>")
            parts.append(f"<p>{workup}</p>")
        
        # Treatment overview
        treatment = dx_exp.get("treatment_overview", "")
        if treatment:
            parts.append("<h4 style='margin-top: 1rem;'>Treatment Considerations</h4>")
            parts.append(f"<p>{treatment}</p>")
        
        # Limitations/caveats
        limitations = dx_exp.get("limitations_and_caveats", "")
        if limitations:
            parts.append("<hr style='margin: 1rem 0; border-color: rgba(139, 92, 246, 0.3);'>")
            parts.append(f"<p style='color: #9ca3af; font-style: italic;'><strong>Important:</strong> {limitations}</p>")
    
    else:
        # Fallback to basic output if no diagnosis explanation available
        if result.extracted_facts and result.extracted_facts.chief_complaint:
            parts.append(f"<p><strong>Chief Complaint:</strong> {result.extracted_facts.chief_complaint}</p>")
        
        if result.diagnoser_output and result.diagnoser_output.differential:
            parts.append("<h4>Differential Diagnosis</h4>")
            for i, dx in enumerate(result.diagnoser_output.differential[:3], 1):
                parts.append(f"<p>{i}. <strong>{dx.condition}</strong> ({dx.likelihood})</p>")
        
        parts.append("<p><em>Detailed explanation not available. Please see Full Analysis for complete results.</em></p>")
    
    return "".join(parts)


# ============================================================
# Typing Animation Effect
# ============================================================

def display_with_typing(text: str, container, delay: float = 0.01):
    """Display text with typing animation effect."""
    placeholder = container.empty()
    displayed = ""
    
    words = text.split(" ")
    
    for i, word in enumerate(words):
        displayed += word + " "
        placeholder.markdown(f'<div class="chat-message">{displayed}▌</div>', unsafe_allow_html=True)
        time.sleep(delay)
    
    placeholder.markdown(f'<div class="chat-message">{text}</div>', unsafe_allow_html=True)


# ============================================================
# Full Analysis Panel (Detailed Results)
# ============================================================

def render_full_analysis(result):
    """Render the detailed analysis in the expandable section."""
    
    # Extracted Facts
    st.markdown("#### Extracted Medical Facts")
    if result.extracted_facts:
        facts = result.extracted_facts
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Symptoms", len(facts.symptoms))
        with col2:
            st.metric("Medications", len(facts.meds))
        with col3:
            st.metric("Allergies", len(facts.allergies))
        with col4:
            st.metric("Risk Factors", len(facts.risk_factors))
        
        with st.expander("View Full Extracted Facts JSON"):
            st.json(facts.model_dump())
    
    st.divider()
    
    # RAG Evidence
    st.markdown("#### RAG Evidence")
    if result.evidence_chunks:
        guidelines_chunks = [c for c in result.evidence_chunks if c.source == "guidelines"]
        merck_chunks = [c for c in result.evidence_chunks if c.source == "merck"]
        
        col_g, col_m = st.columns(2)
        
        with col_g:
            st.markdown("**Clinical Guidelines**")
            for chunk in guidelines_chunks:
                title_short = chunk.title[:50] + "..." if len(chunk.title) > 50 else chunk.title
                with st.expander(f"[{chunk.evidence_id}] {title_short}"):
                    st.markdown(f"**Score:** {chunk.score:.3f}")
                    st.markdown(f"**Section:** {chunk.heading_path}")
                    st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
        
        with col_m:
            st.markdown("**Merck Manual**")
            for chunk in merck_chunks:
                heading_short = chunk.heading_path[:50] + "..." if len(chunk.heading_path) > 50 else chunk.heading_path
                with st.expander(f"[{chunk.evidence_id}] {heading_short}"):
                    st.markdown(f"**Score:** {chunk.score:.3f}")
                    st.text(chunk.text[:500] + ("..." if len(chunk.text) > 500 else ""))
    else:
        st.info("No evidence chunks retrieved.")
    
    st.divider()
    
    # Diagnoser Output
    st.markdown("#### Diagnoser Output")
    if result.diagnoser_output:
        diag = result.diagnoser_output
        
        if diag.red_flags:
            st.markdown("**Red Flags**")
            for rf in diag.red_flags:
                st.markdown(
                    f"""<div class="red-flag">
                    <strong>{rf.risk_level.upper()}</strong>: {rf.description}<br/>
                    <small>Evidence: {', '.join(rf.evidence_ids) if rf.evidence_ids else 'None cited'}</small>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        if diag.differential:
            st.markdown("**Full Differential**")
            for dx in diag.differential:
                likelihood_color = {"high": "#22c55e", "moderate": "#f59e0b", "low": "#6b7280"}.get(dx.likelihood, "#6b7280")
                st.markdown(
                    f"""<div class="differential">
                    <strong>{dx.condition}</strong> 
                    <span style="color:{likelihood_color}; font-weight:bold;">({dx.likelihood})</span><br/>
                    {dx.rationale}<br/>
                    <small>Evidence: {', '.join(dx.evidence_ids) if dx.evidence_ids else 'None cited'}</small>
                    </div>""",
                    unsafe_allow_html=True
                )
        
        if diag.contraindications_and_interactions:
            st.markdown("**Contraindications & Interactions**")
            for ci in diag.contraindications_and_interactions:
                st.warning(f"**{ci.severity.upper()}**: {ci.issue} (Meds: {', '.join(ci.related_meds)})")
        
        with st.expander("View Full Diagnoser JSON"):
            st.json(diag.model_dump())
    
    st.divider()
    
    # Consultant Critique
    if result.consultant_critique:
        st.markdown("#### Consultant Critique")
        critique = result.consultant_critique
        
        safety_colors = {"safe": "#22c55e", "needs_review": "#f59e0b", "unsafe": "#ef4444"}
        safety_color = safety_colors.get(critique.overall_safety_rating, "#6b7280")
        st.markdown(
            f"**Safety Rating:** <span style='color:{safety_color}; font-weight:bold;'>{critique.overall_safety_rating.upper()}</span>",
            unsafe_allow_html=True
        )
        
        if critique.overall_assessment:
            st.markdown(f"**Assessment:** {critique.overall_assessment}")
        
        if critique.issues:
            st.markdown("**Issues:**")
            for issue in critique.issues:
                st.markdown(f"[{issue.severity.upper()}] **{issue.kind}**: {issue.description}")
        
        with st.expander("View Full Critique JSON"):
            st.json(critique.model_dump())
        
        st.divider()
    
    # Arbiter Result
    if result.arbiter_result:
        st.markdown("#### Arbiter Decision")
        arbiter = result.arbiter_result
        
        summary = summarize_arbiter_result(arbiter)
        st.text(summary)
        
        if arbiter.abstained:
            st.error("Arbiter abstained - human review required")
        else:
            if arbiter.patches_applied:
                st.info(f"{len(arbiter.patches_applied)} modification(s) applied")
            else:
                st.success("Plan approved without modifications")
        
        with st.expander("View Full Arbiter JSON"):
            st.json(arbiter.model_dump())
    
    st.divider()
    
    # Critique Explanation (Analysis of Primary Care Doctor)
    if result.critique_explanation:
        st.markdown("#### Critique Explanation (Analysis of Primary Care Doctor)")
        crit_exp = result.critique_explanation
        
        # Show key fields
        rating = crit_exp.get("overall_rating", "unknown")
        st.markdown(f"**Overall Rating:** {rating.upper()}")
        
        if crit_exp.get("overall_summary"):
            st.markdown(f"**Summary:** {crit_exp['overall_summary']}")
        
        if crit_exp.get("concerns") and crit_exp["concerns"].lower() not in ("none identified.", "none identified", "none"):
            st.warning(f"**Concerns:** {crit_exp['concerns']}")
        
        with st.expander("View Full Critique Explanation JSON"):
            st.json(crit_exp)
    
    st.divider()
    
    # Diagnosis Explanation (AI Powered Second Opinion)
    if result.diagnosis_explanation:
        st.markdown("#### Diagnosis Explanation (AI Powered Second Opinion)")
        dx_exp = result.diagnosis_explanation
        
        # Show key fields
        confidence = dx_exp.get("confidence_level", "unknown")
        st.markdown(f"**Confidence:** {confidence.upper()}")
        
        if dx_exp.get("primary_diagnosis"):
            st.markdown(f"**Primary Assessment:** {dx_exp['primary_diagnosis']}")
        
        if dx_exp.get("urgent_considerations") and dx_exp["urgent_considerations"].lower() not in ("none identified.", "none identified", "none"):
            st.warning(f"**Urgent:** {dx_exp['urgent_considerations']}")
        
        with st.expander("View Full Diagnosis Explanation JSON"):
            st.json(dx_exp)
    
    st.divider()
    
    # Dialogue Brief
    if result.dialogue_brief:
        st.markdown("#### Dialogue Brief")
        st.markdown("*Summary of the doctor-patient conversation used for analysis:*")
        with st.expander("View Dialogue Brief"):
            st.text(result.dialogue_brief)
    
    st.divider()
    
    # Generated SOAP Note (for non-ACI inputs)
    doctor_note = st.session_state.get("doctor_note", None)
    if doctor_note:
        st.markdown("#### Generated SOAP Note")
        input_mode = st.session_state.get("input_mode", "")
        if input_mode != "ACI-Bench":
            st.markdown("*SOAP note generated by omi-health/sum-small from the dialogue:*")
        else:
            st.markdown("*Doctor's SOAP note from ACI-Bench dataset:*")
        st.text_area(
            "SOAP Note",
            value=doctor_note,
            height=300,
            disabled=True,
            key="full_analysis_soap_note"
        )


# ============================================================
# Main Application
# ============================================================

def main():
    # Render sidebar with settings
    render_sidebar()
    
    # ---------------------------
    # Main Content Area
    # ---------------------------
    
    # Title
    st.markdown('<h1 class="app-title">2nd Opinion AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="app-subtitle">AI-powered clinical decision support</p>', unsafe_allow_html=True)
    
    # ---------------------------
    # Input Mode Selector + Run Button (compact row)
    # ---------------------------
    col_input, col_run = st.columns([4, 1])
    
    with col_input:
        input_mode = st.radio(
            "Select input method",
            options=["🎤 Record Mic", "📁 Upload .wav", "📝 Free Text", "🧪 ACI-Bench"],
            horizontal=True,
            label_visibility="collapsed",
            key="input_mode_radio"
        )
    
    with col_run:
        run_button = st.button("Run", type="primary", use_container_width=True)
    
    mode_map = {
        "🎤 Record Mic": "Record Mic",
        "📁 Upload .wav": "Upload .wav",
        "📝 Free Text": "Free Text",
        "🧪 ACI-Bench": "ACI-Bench"
    }
    st.session_state["input_mode"] = mode_map.get(input_mode, "Upload .wav")
    
    # ---------------------------
    # Render Input Based on Mode (in expander to keep it compact)
    # ---------------------------
    uploaded_file = None
    free_text = ""
    aci_dialogue = None
    
    with st.expander("📥 Input Details", expanded=st.session_state["pipeline_result"] is None):
        if st.session_state["input_mode"] == "Record Mic":
            render_mic_input()
        elif st.session_state["input_mode"] == "Upload .wav":
            uploaded_file = render_upload_input()
        elif st.session_state["input_mode"] == "Free Text":
            free_text = render_freetext_input()
        elif st.session_state["input_mode"] == "ACI-Bench":
            aci_dialogue = render_aci_input()
    
    # ---------------------------
    # Display Results - Diagnosis Narrative
    # ---------------------------
    if st.session_state["pipeline_result"] is not None:
        result = st.session_state["pipeline_result"]
        
        # Build diagnosis explanation narrative
        summary = build_narrative_output(result)
        
        # Chat container for diagnosis
        chat_container = st.container()
        
        # Check if this is a fresh run (for typing animation)
        current_run_id = st.session_state.get("last_run_id")
        shown_run_id = st.session_state.get("shown_run_id")
        
        if current_run_id != shown_run_id:
            display_with_typing(summary, chat_container, delay=0.005)
            st.session_state["shown_run_id"] = current_run_id
        else:
            chat_container.markdown(f'<div class="chat-message">{summary}</div>', unsafe_allow_html=True)
        
        # Full Narratives from both explanations (if available)
        has_critique_narrative = result.critique_explanation and result.critique_explanation.get("full_narrative")
        has_diagnosis_narrative = result.diagnosis_explanation and result.diagnosis_explanation.get("full_narrative")
        
        if has_critique_narrative or has_diagnosis_narrative:
            with st.expander("Full Explanations"):
                if has_critique_narrative:
                    st.markdown("### Analysis of Primary Care Doctor")
                    st.markdown(result.critique_explanation["full_narrative"])
                    st.markdown("---")
                
                if has_diagnosis_narrative:
                    st.markdown("### AI Powered Second Opinion")
                    st.markdown(result.diagnosis_explanation["full_narrative"])
        
        # Full Analysis Expander
        with st.expander("Full Analysis"):
            render_full_analysis(result)
        
        # Transcript viewer
        if st.session_state["transcript_result"]:
            with st.expander("View Transcript"):
                tr = st.session_state["transcript_result"]
                st.text_area(
                    "Transcript",
                    value=tr.get("text", ""),
                    height=150,
                    disabled=True
                )
                
                segments = tr.get("segments", [])
                if segments:
                    st.markdown("**Diarized Segments:**")
                    for seg in segments[:10]:
                        speaker = seg.get("speaker", "UNKNOWN")
                        text = seg.get("text", "").strip()
                        st.markdown(f"**{speaker}:** {text}")
    
    # ---------------------------
    # Process Input & Run Pipeline
    # ---------------------------
    if run_button:
        transcript_result = None
        role_mapping = None
        
        if st.session_state["input_mode"] == "Record Mic":
            if "recorded_audio" in st.session_state and st.session_state["recorded_audio"]:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(st.session_state["recorded_audio"])
                    tmp_path = tmp.name
                
                with st.spinner("Transcribing recorded audio..."):
                    try:
                        transcript_result = transcribe_file(
                            tmp_path,
                            model_size=st.session_state["model_size"],
                            with_diarization=st.session_state["diarize_on"],
                            min_speakers=st.session_state["min_speakers"],
                            max_speakers=st.session_state["max_speakers"],
                            hf_token=st.session_state["hf_token"] if st.session_state["hf_token"] else None,
                        )
                        if st.session_state["auto_role"] and transcript_result.get("segments"):
                            role_mapping = infer_role_mapping(transcript_result["segments"])
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                
                # Generate SOAP note from the diarized dialogue
                if transcript_result:
                    with st.spinner("Generating SOAP note from dialogue..."):
                        try:
                            soap_note = generate_soap_note(transcript_result, role_mapping)
                            st.session_state["doctor_note"] = soap_note
                            print(f"  Generated SOAP note: {len(soap_note)} chars")
                        except Exception as e:
                            st.warning(f"SOAP generation failed: {str(e)}. Proceeding without doctor's note.")
                            st.session_state["doctor_note"] = None
            else:
                st.session_state["doctor_note"] = None
                st.warning("Please record audio first.")
        
        elif st.session_state["input_mode"] == "Upload .wav":
            if uploaded_file:
                suffix = Path(uploaded_file.name).suffix or ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                with st.spinner("Transcribing uploaded audio..."):
                    try:
                        transcript_result = transcribe_file(
                            tmp_path,
                            model_size=st.session_state["model_size"],
                            with_diarization=st.session_state["diarize_on"],
                            min_speakers=st.session_state["min_speakers"],
                            max_speakers=st.session_state["max_speakers"],
                            hf_token=st.session_state["hf_token"] if st.session_state["hf_token"] else None,
                        )
                        if st.session_state["auto_role"] and transcript_result.get("segments"):
                            role_mapping = infer_role_mapping(transcript_result["segments"])
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                
                # Generate SOAP note from the diarized dialogue
                if transcript_result:
                    with st.spinner("Generating SOAP note from dialogue..."):
                        try:
                            soap_note = generate_soap_note(transcript_result, role_mapping)
                            st.session_state["doctor_note"] = soap_note
                            print(f"  Generated SOAP note: {len(soap_note)} chars")
                        except Exception as e:
                            st.warning(f"SOAP generation failed: {str(e)}. Proceeding without doctor's note.")
                            st.session_state["doctor_note"] = None
            else:
                st.session_state["doctor_note"] = None
                st.warning("Please upload an audio file first.")
        
        elif st.session_state["input_mode"] == "Free Text":
            if free_text.strip():
                transcript_result = {
                    "text": free_text.strip(),
                    "segments": [],
                    "info": {"language": "en", "duration": None},
                }
                role_mapping = {}
                
                # Generate SOAP note from the free text input
                with st.spinner("Generating SOAP note from text..."):
                    try:
                        soap_note = generate_soap_note(transcript_result, role_mapping)
                        st.session_state["doctor_note"] = soap_note
                        print(f"  Generated SOAP note: {len(soap_note)} chars")
                    except Exception as e:
                        st.warning(f"SOAP generation failed: {str(e)}. Proceeding without doctor's note.")
                        st.session_state["doctor_note"] = None
            else:
                st.session_state["doctor_note"] = None
                st.warning("Please enter some text first.")
        
        elif st.session_state["input_mode"] == "ACI-Bench":
            if aci_dialogue:
                with st.spinner("Loading ACI-Bench dialogue..."):
                    transcript_result = make_transcript_result_from_aci(aci_dialogue["dialogue"])
                    role_mapping = get_prebuilt_role_mapping_from_aci()
                    # Store the doctor's note for critique
                    doctor_note_text = aci_dialogue.get("augmented_note") or aci_dialogue.get("note", "")
                    st.session_state["doctor_note"] = doctor_note_text
                    print(f"  ACI-Bench note captured: {bool(doctor_note_text)}, length: {len(doctor_note_text) if doctor_note_text else 0}")
            else:
                st.warning("Please select a valid ACI-Bench example.")
        
        # Run the pipeline if we have transcript
        if transcript_result:
            st.session_state["transcript_result"] = transcript_result
            st.session_state["role_mapping"] = role_mapping or {}
            
            # Get doctor's note if available (from ACI-Bench or generated SOAP)
            doctor_note = st.session_state.get("doctor_note", None)
            # Detect SOAP format - check for SUBJECTIVE header or S: prefix
            is_soap = False
            if doctor_note:
                upper_note = doctor_note.upper()
                is_soap = (
                    "SUBJECTIVE" in upper_note or 
                    "S:" in doctor_note[:50].upper() or
                    upper_note.strip().startswith("S:")
                )
            note_type = "soap" if is_soap else "prose"
            
            with st.spinner("Running clinical reasoning pipeline..."):
                try:
                    llm_config = LLMConfig(
                        mode=st.session_state["llm_mode"],
                        diagnoser_model=st.session_state["diag_model"],
                        consultant_model=st.session_state["consult_model"],
                        qa_model=st.session_state["diag_model"]
                    )
                    
                    result = run_clinical_pipeline(
                        transcript_result=transcript_result,
                        role_mapping=st.session_state["role_mapping"],
                        llm_config=llm_config,
                        k_guidelines=st.session_state["k_guidelines"],
                        k_merck=st.session_state["k_merck"],
                        skip_consultant=st.session_state["skip_consultant"],
                        doctor_note=doctor_note,
                        note_type=note_type
                    )
                    
                    st.session_state["pipeline_result"] = result
                    st.session_state["last_run_id"] = time.time()
                    st.rerun()  # Rerun to show results at top
                    
                except Exception as e:
                    st.error(f"Pipeline failed: {str(e)}")
                    import traceback
                    st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
