import pytest
from extractor.hybrid_extractor import HybridExtractor

@pytest.mark.skip(reason="requires models installed")
def test_smoke_extraction():
    dlg = [
        {"utt_id":1, "role":"PATIENT", "text":"I've had a cough for 3 days and mild fever. No chest pain."},
        {"utt_id":2, "role":"DOCTOR", "text":"Are you exercising a lot this week?"},
        {"utt_id":3, "role":"PATIENT", "text":"Not really."},
        {"utt_id":4, "role":"PATIENT", "text":"I take ibuprofen 200 mg twice daily by mouth."},
        {"utt_id":5, "role":"PATIENT", "text":"Allergic to penicillin, had a rash."},
    ]
    hx = HybridExtractor(symptom_backend="scispacy", med_backend="scispacy", llm_client=None)
    ej = hx.extract(dlg)
    assert ej.chief_complaint
    assert len(ej.symptoms) >= 1
    assert any(m.dose for m in ej.meds)
    assert ej.allergies