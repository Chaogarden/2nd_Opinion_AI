# extractor/llm_assist.py
from typing import Optional, Dict
import json
import re
from .schema import QAExtracted, Evidence

SYS = (
    "You are a clinical fact extractor.\n"
    "Given a doctor's question and a patient's answer, extract ONE clinically meaningful fact.\n"
    "Use 'present' if the patient affirms, 'absent' if the patient denies, 'possible' if uncertain.\n"
    "Return STRICT JSON ONLY with keys: concept, value, assertion, evidence.\n"
    "Where evidence = {\"utt_ids\": [<doctor_utt_id>, <patient_utt_id>]}.\n"
    "If no fact is extractable, return null.\n"
)

USR_TMPL = (
    "Doctor ({d_id}): {d_txt}\n"
    "Patient ({p_id}): {p_txt}\n"
    "JSON schema: {{\"concept\": str, \"value\": str|null, \"assertion\": \"present|absent|possible\", \"evidence\": {{\"utt_ids\": [int,int]}}}}"
)

def _extract_json_block(text: str) -> Optional[str]:
    """
    Return the first balanced {...} block in `text`.
    Handles optional ```json ... ``` wrappers and ignores stray braces.
    """
    if not text:
        return None

    # strip code fences if present
    t = text.strip()
    if t.startswith("```"):
        # e.g. ```json\n{...}\n```
        parts = t.split("```")
        # collect anything between the first and second fence
        if len(parts) >= 3:
            t = parts[1] if "{" in parts[1] else parts[2]

    # scan for balanced braces
    start = None
    depth = 0
    for i, ch in enumerate(t):
        if ch == "{":
            if start is None:
                start = i
                depth = 1
            else:
                depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    return t[start:i+1]
    return None


class QAInterpreter:
    """
    client must expose a .chat(model, messages, temperature) -> str | dict
    See adapters below.
    """
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    def extract(self, doctor_turn: Dict, patient_turn: Dict) -> Optional[QAExtracted]:
        prompt = USR_TMPL.format(
            d_id=doctor_turn["utt_id"], d_txt=doctor_turn["text"],
            p_id=patient_turn["utt_id"], p_txt=patient_turn["text"]
        )

        # 1st attempt
        out = self.client.chat(
            model=self.model,
            messages=[{"role":"system","content":SYS},{"role":"user","content":prompt}],
            temperature=0
        )
        content = out if isinstance(out, str) else out.get("content", "")
        blob = _extract_json_block(content) or content.strip()

        # retry once if parsing fails
        for _ in range(2):
            try:
                if blob == "null":
                    return None
                data = json.loads(blob)
                qa = QAExtracted(
                    concept=data["concept"],
                    value=data.get("value"),
                    assertion=data.get("assertion","present"),
                    evidence=Evidence(utt_ids=data.get("evidence",{}).get("utt_ids", [doctor_turn["utt_id"], patient_turn["utt_id"]]))
                )
                return qa
            except Exception:
                # second try: ask for stricter JSON
                out = self.client.chat(
                    model=self.model,
                    messages=[
                        {"role":"system","content":SYS},
                        {"role":"user","content":prompt + "\nReturn ONLY valid minified JSON."}
                    ],
                    temperature=0
                )
                content = out if isinstance(out, str) else out.get("content", "")
                blob = _extract_json_block(content) or content.strip()

        return None

# -----------------------
# Adapters (pick one)
# -----------------------

class OllamaAdapter:
    """Calls local Ollama chat API (HTTP)."""
    def __init__(self, host: str = "http://localhost:11434"):
        import requests
        self.requests = requests
        self.host = host.rstrip("/")

    def chat(self, model: str, messages, temperature: float):
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,  # IMPORTANT: avoid streaming for simpler parsing
            "options": {"temperature": temperature}
        }
        r = self.requests.post(self.host + "/api/chat", json=payload, timeout=180)
        # If it's a 400, print the server message to help debugging
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = {"error_text": r.text}
            raise RuntimeError(f"Ollama error {r.status_code}: {err}")
        data = r.json()
        # Non-stream response shape: {"message":{"role":"assistant","content":"..."}}
        msg = data.get("message", {})
        return {"content": msg.get("content", "")}

