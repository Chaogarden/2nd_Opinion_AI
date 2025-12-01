# qa_smoke.py
import os
from extractor.hybrid_extractor import HybridExtractor
from extractor.llm_assist import QAInterpreter, OllamaAdapter

dialogue = [
    {"utt_id":1,"role":"PATIENT","text":"I've had a cough for 3 days and mild fever. No chest pain."},
    {"utt_id":2,"role":"DOCTOR","text":"Are you exercising a lot this week?"},
    {"utt_id":3,"role":"PATIENT","text":"Not really."},
    {"utt_id":4,"role":"DOCTOR","text":"Any shortness of breath when lying down?"},
    {"utt_id":5,"role":"PATIENT","text":"A little bit at night."},
    {"utt_id":6,"role":"PATIENT","text":"I take ibuprofen 200 mg twice daily by mouth."},
    {"utt_id":7,"role":"PATIENT","text":"Allergic to penicillin, had a rash."},
]

MODEL_NAME = "llama3.1:8b"  # good Ollama default; or "llama3.1:8b-instruct"

model = (os.environ.get("OLLAMA_MODEL", MODEL_NAME)).strip()

adapter = OllamaAdapter("http://localhost:11434")

qa = QAInterpreter(client=adapter, model=model)

hx = HybridExtractor(
    symptom_backend="scispacy",
    med_backend="scispacy",
    rxnorm_tsv_path="models/rxnorm_names.tsv",
    llm_client=None  # we'll attach qa manually for now
)
# Patch in the QA interpreter
hx.qa = qa

ej = hx.extract(dialogue)
print("Q/A extracted:")
for q in ej.qa_extractions:
    print(q.model_dump())

print("\nFULL JSON:")
print(ej.model_dump_json(indent=2))
