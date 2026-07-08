# 2nd Opinion AI

2nd Opinion AI is a medical AI project designed to analyze doctor–patient conversations and generate a structured, evidence-based second opinion.

The goal of the project is not to replace medical professionals. Instead, it is intended to help patients better understand their visit, identify important questions to ask, and review whether any symptoms, risks, or concerns may require further discussion with a healthcare provider.

## Planned Features

* Convert recorded medical conversations into text
* Separate doctor and patient speakers
* Extract symptoms, medications, allergies, and medical history
* Generate a structured SOAP note
* Identify red flags and missing clinical questions
* Compare the doctor’s assessment with trusted medical sources
* Generate a patient-friendly summary
* Suggest follow-up questions for the patient to ask their doctor

## Project Pipeline

```text
Medical Conversation
        ↓
Audio Transcription
        ↓
Speaker Diarization
        ↓
Medical Information Extraction
        ↓
SOAP Note Generation
        ↓
Evidence Retrieval
        ↓
Clinical Reasoning
        ↓
Second-Opinion Report
```

## Technologies

The project may use:

* Python
* Llama 3
* Faster-Whisper
* pyannote.audio
* medspaCy
* Retrieval-Augmented Generation
* LoRA fine-tuning
* Streamlit

## Current Status

This project is currently under development.

The initial development stages focus on:

1. Medical conversation transcription
2. Structured medical information extraction
3. SOAP note generation
4. Evaluation using medical conversation and question-answering datasets

## Safety

2nd Opinion AI is an educational and research project.

It is not a medical device and should not be used to diagnose, treat, or prevent any medical condition. All generated information should be reviewed by a qualified healthcare professional.

In a medical emergency, contact emergency services immediately.

## Future Goals

Future versions may include:

* Transcript-grounded medical claims
* Source citations for medical recommendations
* Hallucination detection
* Clinical note quality checks
* Support for multiple medical specialties
* Improved patient-friendly explanations

## License

A license has not yet been selected for this project.
