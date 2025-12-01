# core/audio/transcribe.py
# ==============================
# WhisperX ASR + Diarization
# - Uses whisperx with faster-whisper backend
# - Speaker diarization via pyannote (built into whisperx)
# - Optimized for CPU (M4 Mac compatible)
# ==============================

import whisperx
import torch
from pathlib import Path
from typing import Optional


def get_device() -> str:
    """Determine the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS can be buggy with whisperx, fallback to CPU for stability
        return "cpu"
    return "cpu"


def get_compute_type(device: str) -> str:
    """Get optimal compute type for the device."""
    if device == "cuda":
        return "float16"
    # CPU works best with float32 for compatibility (int8 requires specific builds)
    return "float32"


def transcribe_file(
    audio_path: str,
    model_size: str = "base",
    language: Optional[str] = None,
    with_diarization: bool = True,
    min_speakers: int = 2,
    max_speakers: int = 2,
    hf_token: Optional[str] = None,
) -> dict:
    """
    Transcribe an audio file using WhisperX with optional speaker diarization.
    
    Args:
        audio_path: Path to the audio file (WAV, MP3, M4A, etc.)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2')
        language: Language code (e.g., 'en'). If None, auto-detect.
        with_diarization: Whether to perform speaker diarization
        min_speakers: Minimum number of speakers for diarization
        max_speakers: Maximum number of speakers for diarization
        hf_token: HuggingFace token for pyannote diarization models
        
    Returns:
        dict with keys:
            - text: Full transcript text
            - segments: List of segment dicts with start, end, text, and optionally speaker
            - info: Metadata dict with language and duration
    """
    device = get_device()
    compute_type = get_compute_type(device)
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    
    # Load model and transcribe
    model = whisperx.load_model(
        model_size,
        device=device,
        compute_type=compute_type,
        language=language,
    )
    
    result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)
    detected_language = result.get("language", language or "en")
    
    # Align whisper output for word-level timestamps
    model_a, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )
    
    # Speaker diarization
    if with_diarization and hf_token:
        from whisperx.diarize import DiarizationPipeline
        
        diarize_model = DiarizationPipeline(
            use_auth_token=hf_token,
            device=device,
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
    
    # Build output
    segments = []
    for seg in result.get("segments", []):
        segment_data = {
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        }
        if "speaker" in seg:
            segment_data["speaker"] = seg["speaker"]
        segments.append(segment_data)
    
    # Combine all text
    full_text = " ".join(seg["text"] for seg in segments if seg["text"])
    
    # Get duration from audio
    duration = len(audio) / 16000  # whisperx loads at 16kHz
    
    return {
        "text": full_text,
        "segments": segments,
        "info": {
            "language": detected_language,
            "duration": duration,
        },
    }


# Convenience function for quick transcription without diarization
def transcribe_simple(audio_path: str, model_size: str = "base") -> str:
    """Quick transcription returning just the text."""
    result = transcribe_file(
        audio_path,
        model_size=model_size,
        with_diarization=False,
    )
    return result["text"]