# core/soap_sum_small.py
# ==============================
# SOAP Note Generation using omi-health/sum-small
# 
# Converts doctor-patient dialogues into structured SOAP notes
# using the omi-health/sum-small model from Hugging Face.
# Optimized for Mac M4 (MPS/CPU, no CUDA/bitsandbytes).
# ==============================

import warnings
from typing import Optional, Dict, Any, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress unnecessary warnings during model loading
warnings.filterwarnings('ignore', category=UserWarning)

# Model configuration
MODEL_NAME = "omi-health/sum-small"

# Cached model/tokenizer to avoid reloading
_cached_model = None
_cached_tokenizer = None
_cached_device = None


# ============================================================
# Device Selection (Mac M4 Compatible)
# ============================================================

def get_soap_device() -> str:
    """
    Determine the best available device for SOAP model inference.
    
    Note: We use CPU by default because MPS has compatibility issues
    with the Phi-3 based sum-small model (generates truncated output).
    CPU performance is still acceptable for SOAP generation.
    
    Returns:
        Device string: 'cpu' (MPS disabled due to compatibility issues)
    """
    # MPS has issues with Phi-3 based models - outputs get truncated
    # Forcing CPU which works reliably
    return "cpu"


def get_soap_dtype(device: str) -> torch.dtype:
    """
    Get optimal dtype for the device.
    
    Args:
        device: Device string ('mps' or 'cpu')
    
    Returns:
        torch.dtype: float16 for MPS, float32 for CPU
    """
    if device == "mps":
        # MPS works well with float16
        return torch.float16
    # CPU needs float32 for compatibility
    return torch.float32


# ============================================================
# Model Loading
# ============================================================

def load_sum_small_model(
    force_reload: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """
    Load the omi-health/sum-small model and tokenizer.
    
    Uses caching to avoid reloading the model on every call.
    Automatically selects MPS on Mac M4, falls back to CPU.
    
    Args:
        force_reload: If True, reload the model even if cached.
    
    Returns:
        Tuple of (model, tokenizer, device)
    """
    global _cached_model, _cached_tokenizer, _cached_device
    
    if not force_reload and _cached_model is not None:
        return _cached_model, _cached_tokenizer, _cached_device
    
    print(f"Loading {MODEL_NAME} model...")
    
    device = get_soap_device()
    dtype = get_soap_dtype(device)
    
    print(f"  Device: {device}, dtype: {dtype}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with appropriate settings for Mac
    # Note: We avoid bitsandbytes (load_in_8bit) since it's not supported on Mac
    # trust_remote_code=False to use transformers' native implementation
    # (avoids DynamicCache 'seen_tokens' compatibility issues)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    
    # Move to device
    model = model.to(device)
    model.eval()
    
    # Cache for reuse
    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_device = device
    
    print(f"  Model loaded successfully on {device}")
    
    return model, tokenizer, device


# ============================================================
# Dialogue Building Utilities
# ============================================================

def build_doctor_patient_dialogue_text(
    transcript_result: Dict[str, Any],
    role_mapping: Optional[Dict[str, str]] = None
) -> str:
    """
    Convert transcript result into a Doctor/Patient dialogue string.
    
    This function takes the output from WhisperX ASR (with diarization)
    and formats it as a clean dialogue suitable for the SOAP model.
    
    Args:
        transcript_result: Dict with 'text' and optionally 'segments' from ASR
        role_mapping: Optional speaker->role mapping from diarization
    
    Returns:
        Formatted dialogue string like:
        "Doctor: How are you feeling?
         Patient: I have a headache..."
    """
    # Import dialogue utilities
    from core.dialogue import build_dialogue, build_dialogue_from_text, infer_role_mapping
    
    segments = transcript_result.get("segments", [])
    full_text = transcript_result.get("text", "")
    
    if segments:
        # We have diarized segments
        if role_mapping is None:
            role_mapping = infer_role_mapping(segments)
        
        dialogue_turns = build_dialogue(segments, mapping=role_mapping)
    elif full_text:
        # Free-text input - treat as single patient turn
        dialogue_turns = build_dialogue_from_text(full_text, default_role="PATIENT")
    else:
        return ""
    
    # Convert to Doctor/Patient format for the SOAP model
    lines = []
    for turn in dialogue_turns:
        role = turn.get("role", "UNKNOWN").upper()
        text = turn.get("text", "").strip()
        
        if not text:
            continue
        
        # Map roles to labels the model expects
        if role == "DOCTOR":
            label = "Doctor"
        elif role == "PATIENT":
            label = "Patient"
        else:
            # Handle other roles (e.g., OTHER_1, UNKNOWN)
            label = "Other"
        
        lines.append(f"{label}: {text}")
    
    return "\n".join(lines)


# ============================================================
# SOAP Note Generation
# ============================================================

def generate_soap_note(
    transcript_result: Dict[str, Any],
    role_mapping: Optional[Dict[str, str]] = None,
    max_new_tokens: int = 800,
    temperature: float = 0.0,
) -> str:
    """
    Generate a SOAP note from a doctor-patient dialogue.
    
    This is the main entry point for SOAP generation. It takes the
    transcript result from WhisperX (or free text) and produces a
    structured SOAP note using the omi-health/sum-small model.
    
    Args:
        transcript_result: Dict with 'text' and optionally 'segments' from ASR
        role_mapping: Optional speaker->role mapping from diarization
        max_new_tokens: Maximum tokens to generate (default 800)
        temperature: Generation temperature (0.0 = deterministic)
    
    Returns:
        SOAP note string with SUBJECTIVE:, OBJECTIVE:, ASSESSMENT:, PLAN: sections
    """
    # Load model (uses cache if already loaded)
    model, tokenizer, device = load_sum_small_model()
    
    # Build dialogue text from transcript
    dialogue = build_doctor_patient_dialogue_text(transcript_result, role_mapping)
    
    if not dialogue.strip():
        return "Unable to generate SOAP note: No dialogue content available."
    
    # Construct prompt using Phi-3 chat format
    # The model expects a specific format for best results
    messages = [
        {
            "role": "system",
            "content": "You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting."
        },
        {
            "role": "user", 
            "content": f"Create a medical SOAP summary of this dialogue:\n\n{dialogue}"
        }
    ]
    
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback to simple format
        prompt = f"""<|system|>
You are an expert medical professor assisting in the creation of medically accurate SOAP summaries. Please ensure the response follows the structured format: S:, O:, A:, P: without using markdown or special formatting.<|end|>
<|user|>
Create a medical SOAP summary of this dialogue:

{dialogue}<|end|>
<|assistant|>
"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
        )
    
    # Decode only the new tokens (skip input tokens)
    input_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_length:]
    soap_note = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    # Debug: print raw output for troubleshooting
    print(f"  Raw SOAP output length: {len(soap_note)} chars")
    
    # Clean up any remaining special tokens or markers
    # Remove Phi-3 end tokens if present
    for end_marker in ["<|end|>", "<|assistant|>", "<|endoftext|>"]:
        soap_note = soap_note.replace(end_marker, "")
    soap_note = soap_note.strip()
    
    # If output is reasonable, return it
    if len(soap_note) >= 50:
        print(f"  SOAP note generated successfully: {len(soap_note)} chars")
        return soap_note
    
    # Something went wrong with the new token extraction
    # Fall back to decoding full output and extracting after assistant marker
    print(f"  Raw output too short, trying full decode...")
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Full output length: {len(full_output)} chars")
    
    # For Phi-3 models, look for content after the assistant turn starts
    assistant_markers = ["<|assistant|>", "### Your SOAP Summary:", "SOAP Summary:"]
    for marker in assistant_markers:
        idx = full_output.find(marker)
        if idx >= 0:
            soap_note = full_output[idx + len(marker):].strip()
            # Clean up end markers
            for end_marker in ["<|end|>", "<|endoftext|>"]:
                soap_note = soap_note.replace(end_marker, "")
            soap_note = soap_note.strip()
            if len(soap_note) >= 20:
                print(f"  Extracted SOAP after '{marker}': {len(soap_note)} chars")
                return soap_note
    
    # Last resort: look for S: or SUBJECTIVE in the generated portion only
    # (not in the prompt - check from halfway through)
    midpoint = len(full_output) // 2
    second_half = full_output[midpoint:]
    upper_second = second_half.upper()
    
    for marker in ["S:", "SUBJECTIVE"]:
        idx = upper_second.find(marker)
        if idx >= 0:
            soap_note = second_half[idx:].strip()
            print(f"  Found '{marker}' in second half at index {idx}: {len(soap_note)} chars")
            return soap_note
    
    # If all else fails, return whatever we have
    print(f"  Warning: Could not extract clean SOAP note, returning raw output")
    return soap_note if soap_note else full_output


# ============================================================
# Cleanup
# ============================================================

def clear_soap_model_cache():
    """
    Clear the cached SOAP model to free memory.
    
    Useful when switching contexts or to force a model reload.
    """
    global _cached_model, _cached_tokenizer, _cached_device
    
    if _cached_model is not None:
        del _cached_model
        _cached_model = None
    
    if _cached_tokenizer is not None:
        del _cached_tokenizer
        _cached_tokenizer = None
    
    _cached_device = None
    
    # Clear CUDA/MPS cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ============================================================
# Testing / CLI
# ============================================================

if __name__ == "__main__":
    # Simple test with a sample dialogue
    sample_dialogue = """Doctor: Hello, can you please tell me about your past medical history?
Patient: Hi, I don't have any past medical history.
Doctor: Okay. What brings you in today?
Patient: I've been experiencing headaches for the past week, along with some dizziness.
Doctor: How severe are the headaches on a scale of 1 to 10?
Patient: About a 6 or 7. They're worse in the morning.
Doctor: Any nausea or visual changes?
Patient: Some nausea, but no visual changes.
Doctor: I see. Let me check your vitals. Your blood pressure is 145/92, which is elevated."""
    
    # Create a mock transcript result
    test_transcript = {
        "text": sample_dialogue,
        "segments": [],  # Empty segments = treat as free text
    }
    
    print("Testing SOAP generation...")
    print("=" * 60)
    print("Input dialogue:")
    print(sample_dialogue)
    print("=" * 60)
    
    soap_note = generate_soap_note(test_transcript)
    
    print("\nGenerated SOAP Note:")
    print("=" * 60)
    print(soap_note)
    print("=" * 60)
