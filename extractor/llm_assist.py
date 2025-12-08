# extractor/llm_assist.py
# ==============================
# LLM-assisted clinical fact extraction from Q/A pairs
# ==============================

from typing import Optional, Dict, Any
import json
import re
from .schema import QAExtracted, Evidence


def _extract_content(response: Any) -> str:
    """Extract text content from various LLM response formats."""
    if isinstance(response, str):
        return response
    elif isinstance(response, dict):
        return response.get("content", "")
    elif isinstance(response, list) and response:
        first = response[0]
        return first.get("content", "") if isinstance(first, dict) else str(first)
    return str(response) if response else ""


# Improved system prompt with examples and clearer constraints
SYS = """You are a clinical fact extractor. Extract ONE specific, medically relevant fact from a doctor-patient exchange.

RULES:
1. Extract ONLY facts about symptoms, conditions, medications, lifestyle, or medical history
2. Use standardized medical concepts (e.g., "chest_pain" not "it hurts here")
3. Return null if the exchange is just social conversation, greetings, or administrative
4. Return null if no clear medical fact can be extracted

ASSERTION VALUES:
- "present": Patient confirms/affirms (yes, I do, I have)
- "absent": Patient denies (no, I don't, never)
- "possible": Patient is uncertain (maybe, sometimes, not sure)

EXAMPLES:
Doctor: "Do you smoke?" Patient: "Yes, about a pack a day"
→ {"concept": "tobacco_use", "value": "1 pack per day", "assertion": "present", "evidence": {"utt_ids": [1, 2]}}

Doctor: "Any chest pain?" Patient: "No, nothing like that"
→ {"concept": "chest_pain", "value": null, "assertion": "absent", "evidence": {"utt_ids": [3, 4]}}

Doctor: "How's your appetite?" Patient: "Not great, I haven't been eating much"
→ {"concept": "decreased_appetite", "value": null, "assertion": "present", "evidence": {"utt_ids": [5, 6]}}

Doctor: "Any allergies?" Patient: "Yeah, penicillin gives me hives"
→ {"concept": "penicillin_allergy", "value": "hives", "assertion": "present", "evidence": {"utt_ids": [7, 8]}}

Doctor: "How are you today?" Patient: "Good, thanks"
→ null (social greeting, no medical fact)

Return ONLY valid JSON or null. No explanations."""

USR_TMPL = """Doctor (utt_id={d_id}): {d_txt}
Patient (utt_id={p_id}): {p_txt}

Extract the clinical fact as JSON:"""


def _extract_json_block(text: str) -> Optional[str]:
    """
    Return the first balanced {...} block in `text`.
    Handles optional ```json ... ``` wrappers and ignores stray braces.
    """
    if not text:
        return None

    # Strip code fences if present
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            t = parts[1] if "{" in parts[1] else parts[2]
        elif len(parts) >= 2 and "{" in parts[1]:
            t = parts[1]
    
    # Remove language tag if present
    if t.startswith("json"):
        t = t[4:].lstrip()

    # Scan for balanced braces
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


def _validate_qa_extraction(data: dict) -> bool:
    """Validate that the extracted QA data is meaningful."""
    if not data:
        return False
    
    concept = data.get("concept", "")
    if not concept or len(concept) < 3:
        return False
    
    # Skip generic/noise concepts
    noise_concepts = {
        "greeting", "acknowledgment", "confirmation", "response",
        "question", "answer", "statement", "conversation",
        "thanks", "thank_you", "okay", "yes", "no",
        "understanding", "agreement", "compliance"
    }
    if concept.lower().strip("_- ") in noise_concepts:
        return False
    
    # Check assertion is valid
    assertion = data.get("assertion", "")
    if assertion not in {"present", "absent", "possible"}:
        return False
    
    return True


class QAInterpreter:
    """
    LLM-based interpreter for doctor-patient Q/A pairs.
    Extracts structured clinical facts from conversational exchanges.
    """
    
    def __init__(self, client, model: str):
        """
        Initialize the QA interpreter.
        
        Args:
            client: LLM client with .chat() method
            model: Model name to use
        """
        self.client = client
        self.model = model
    
    def extract(self, doctor_turn: Dict, patient_turn: Dict) -> Optional[QAExtracted]:
        """
        Extract a clinical fact from a Q/A pair.
        
        Args:
            doctor_turn: Dict with 'utt_id' and 'text'
            patient_turn: Dict with 'utt_id' and 'text'
        
        Returns:
            QAExtracted if a valid fact was extracted, None otherwise.
        """
        prompt = USR_TMPL.format(
            d_id=doctor_turn["utt_id"], 
            d_txt=doctor_turn["text"],
            p_id=patient_turn["utt_id"], 
            p_txt=patient_turn["text"]
        )

        # First attempt
        try:
            out = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYS},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
        except Exception as e:
            # If LLM call fails, return None gracefully
            return None
        
        content = _extract_content(out)
        blob = _extract_json_block(content) or content.strip()

        # Try to parse
        for attempt in range(2):
            try:
                # Handle explicit null response
                if blob.lower().strip() in ("null", "none", "{}"):
                    return None
                
                data = json.loads(blob)
                
                # Validate the extraction
                if not _validate_qa_extraction(data):
                    return None
                
                qa = QAExtracted(
                    concept=data["concept"],
                    value=data.get("value"),
                    assertion=data.get("assertion", "present"),
                    evidence=Evidence(
                        utt_ids=data.get("evidence", {}).get(
                            "utt_ids", 
                            [doctor_turn["utt_id"], patient_turn["utt_id"]]
                        )
                    )
                )
                return qa
                
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == 0:
                    # Retry with stricter prompt
                    try:
                        out = self.client.chat(
                            model=self.model,
                            messages=[
                                {"role": "system", "content": SYS},
                                {"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON or null."}
                            ],
                            temperature=0
                        )
                        content = _extract_content(out)
                        blob = _extract_json_block(content) or content.strip()
                    except Exception:
                        return None

        return None


# -----------------------
# Adapters (for standalone use)
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
            "stream": False,
            "options": {"temperature": temperature}
        }
        r = self.requests.post(self.host + "/api/chat", json=payload, timeout=180)
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = {"error_text": r.text}
            raise RuntimeError(f"Ollama error {r.status_code}: {err}")
        data = r.json()
        msg = data.get("message", {})
        return {"content": msg.get("content", "")}
