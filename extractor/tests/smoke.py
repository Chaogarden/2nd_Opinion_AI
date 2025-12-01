# smoke.py
import os
from extractor.hybrid_extractor import HybridExtractor
from extractor.llm_assist import QAInterpreter, OllamaAdapter

# --- Demo dialogue (covers symptoms, negation, meds, allergy, and Q/A) ---
dialogue = [
    {"utt_id":1,"role":"PATIENT","text":"I've had a cough for 3 days and mild fever. No chest pain."},
    {"utt_id":2,"role":"DOCTOR","text":"Are you exercising a lot this week?"},
    {"utt_id":3,"role":"PATIENT","text":"Not really."},
    {"utt_id":4,"role":"DOCTOR","text":"Any shortness of breath when lying down?"},
    {"utt_id":5,"role":"PATIENT","text":"A little bit at night."},
    {"utt_id":6,"role":"PATIENT","text":"I take Advil 200 mg twice daily by mouth."},
    {"utt_id":7,"role":"PATIENT","text":"Allergic to penicillin, had a rash."},
]

# --- Optional: Q/A via Ollama (set OLLAMA_MODEL to override) ---
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"  # or "llama3.1:8b-instruct" if you pulled it
ollama_model = (os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)).strip()

qa = None
try:
    adapter = OllamaAdapter("http://localhost:11434")
    qa = QAInterpreter(client=adapter, model=ollama_model)
except Exception as e:
    print(f"[WARN] QA disabled (Ollama not available or adapter init failed): {e}")

# --- Build the extractor (uses scispaCy providers + RxNorm TSV if present) ---
rxnorm_tsv = "models/rxnorm_names.tsv" if os.path.exists("models/rxnorm_names.tsv") else None

hx = HybridExtractor(
    symptom_backend="scispacy",
    med_backend="scispacy",
    rxnorm_tsv_path=rxnorm_tsv,
    llm_client=None
)
if qa:
    hx.qa = qa  # enable Q/A pairing + LLM interpretation

# --- Run extraction ---
ej = hx.extract(dialogue)

# --- Pretty output (sectioned) ---
print("\n=== SUMMARY ===")
print("Chief complaint:", ej.chief_complaint or "<none>")
print("\nSymptoms:")
for s in ej.symptoms:
    attrs = []
    if s.duration: attrs.append(f"duration={s.duration}")
    if s.severity: attrs.append(f"severity={s.severity}")
    if s.onset:    attrs.append(f"onset={s.onset}")
    print(f" - {s.name_surface} [{s.assertion}]" + (f" ({', '.join(attrs)})" if attrs else ""))

print("\nMedications:")
for m in ej.meds:
    norm = f" → {m.name_norm}" if m.name_norm and m.name_norm.lower()!=m.name_surface.lower() else ""
    rxcui = f" (RxCUI {m.rxcui})" if m.rxcui else ""
    bits = [b for b in [m.dose, m.freq, m.route, m.form] if b]
    tail = f" | {' '.join(bits)}" if bits else ""
    print(f" - {m.name_surface}{norm}{rxcui} [{m.assertion}]{tail}")

print("\nAllergies:")
for a in ej.allergies:
    react = f" → {a.reaction}" if a.reaction else ""
    print(f" - {a.substance_surface}{react} [{a.assertion}]")

print("\nQ/A Extractions:")
if ej.qa_extractions:
    for q in ej.qa_extractions:
        print(f" - {q.concept}: {q.value} [{q.assertion}] (evidence utt_ids={q.evidence.utt_ids})")
else:
    print(" - <none> (QA disabled or no pairs found)")

print("\n=== FULL JSON ===")
print(ej.model_dump_json(indent=2))