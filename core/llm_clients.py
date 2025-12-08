# core/llm_clients.py
# ==============================
# LLM client abstractions for test (Ollama) and production endpoints
# ==============================

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> Union[Dict[str, Any], str]:
        """
        Send a chat completion request.
        
        Args:
            model: The model name/identifier.
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0 = deterministic).
        
        Returns:
            Either a dict with 'content' key or a raw string.
        """
        pass


class OllamaClient(BaseLLMClient):
    """
    Client for local Ollama HTTP API.
    Default endpoint: http://localhost:11434
    """
    
    def __init__(self, host: Optional[str] = None):
        """
        Initialize Ollama client.
        
        Args:
            host: Ollama API host URL. Defaults to OLLAMA_HOST env var
                  or http://localhost:11434.
        """
        import requests
        self._requests = requests
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Send a chat request to Ollama."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature}
        }
        
        try:
            r = self._requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=300  # 5 minutes for large models
            )
        except self._requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.host}. "
                "Make sure Ollama is running (ollama serve)."
            )
        
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = {"error_text": r.text}
            raise RuntimeError(f"Ollama error {r.status_code}: {err}")
        
        data = r.json()
        msg = data.get("message", {})
        return {"content": msg.get("content", "")}
    
    def list_models(self) -> List[str]:
        """List available models in Ollama."""
        try:
            r = self._requests.get(f"{self.host}/api/tags", timeout=10)
            if r.ok:
                data = r.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []


class OpenAICompatibleClient(BaseLLMClient):
    """
    Client for OpenAI-compatible API endpoints.
    Works with OpenAI, Azure OpenAI, vLLM, and other compatible services.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI-compatible client.
        
        Args:
            base_url: API base URL. Defaults to PROD_LLM_BASE_URL env var
                      or https://api.openai.com/v1.
            api_key: API key. Defaults to PROD_LLM_API_KEY env var.
            organization: Optional organization ID. Defaults to PROD_LLM_ORG env var.
        """
        import requests
        self._requests = requests
        
        self.base_url = (
            base_url or 
            os.environ.get("PROD_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("PROD_LLM_API_KEY", "")
        self.organization = organization or os.environ.get("PROD_LLM_ORG", "")
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Send a chat completion request to an OpenAI-compatible endpoint."""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        try:
            r = self._requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=300
            )
        except self._requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Could not connect to LLM endpoint at {self.base_url}: {e}")
        
        if not r.ok:
            try:
                err = r.json()
            except Exception:
                err = {"error_text": r.text}
            raise RuntimeError(f"LLM API error {r.status_code}: {err}")
        
        data = r.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            return {"content": msg.get("content", "")}
        return {"content": ""}


class MockLLMClient(BaseLLMClient):
    """
    Mock client for testing purposes.
    Returns pre-configured responses or echoes input.
    """
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        """
        Initialize mock client.
        
        Args:
            responses: Optional dict mapping model names to canned responses.
        """
        self.responses = responses or {}
        self.call_history: List[Dict] = []
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0
    ) -> Dict[str, Any]:
        """Return a mock response."""
        self.call_history.append({
            "model": model,
            "messages": messages,
            "temperature": temperature
        })
        
        if model in self.responses:
            return {"content": self.responses[model]}
        
        # Default: echo the last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return {"content": f"Mock response for: {msg.get('content', '')[:100]}..."}
        
        return {"content": "Mock response"}

