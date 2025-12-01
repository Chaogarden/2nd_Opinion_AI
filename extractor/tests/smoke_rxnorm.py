# smoke_rxnorm.py (repo root)
from extractor.hybrid_extractor import HybridExtractor

dialogue = [
    {"utt_id":1,"role":"PATIENT","text":"I’ve been taking Advil 200 mg twice daily by mouth."},
]

hx = HybridExtractor(
    symptom_backend="scispacy",
    med_backend="scispacy",
    rxnorm_tsv_path="models/rxnorm_names.tsv",
    llm_client=None
)

ej = hx.extract(dialogue)
print(ej.meds[0].name_surface, "→", ej.meds[0].name_norm, ej.meds[0].rxcui)
print(ej.model_dump_json(indent=2))
