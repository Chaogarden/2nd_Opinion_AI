# core/llm_config.py
# ==============================
# LLM configuration helpers with environment-driven defaults
# Supports hybrid mode: env vars as source of truth, UI overrides at runtime
# ==============================

import os
from typing import Literal, Tuple, Dict, Any
from .llm_clients import BaseLLMClient, OllamaClient, OpenAICompatibleClient

LLMMode = Literal["test", "prod"]

# Default model names for each mode
DEFAULT_TEST_MODELS = {
    # Use different models in test mode to add diversity:
    # - Diagnoser / QA: llama3.1:8b
    # - Consultant: mistral:7b (assumes this Ollama model is pulled)
    "diagnoser": "llama3.1:8b",
    "consultant": "mistral:7b",
    "qa": "llama3.1:8b",
}

DEFAULT_PROD_MODELS = {
    "diagnoser": "OpenBioLLM-70B",
    "consultant": "Meditron-70B",
    "qa": "OpenBioLLM-70B",
}


def get_llm_mode() -> LLMMode:
    """
    Get the current LLM mode from environment.
    
    Returns:
        'test' or 'prod' based on LLM_MODE env var (default: 'test').
    """
    mode = os.environ.get("LLM_MODE", "test").lower()
    if mode in ("prod", "production"):
        return "prod"
    return "test"


def get_default_models(mode: LLMMode = None) -> Dict[str, str]:
    """
    Get default model names for the given mode.
    
    Args:
        mode: 'test' or 'prod'. If None, uses get_llm_mode().
    
    Returns:
        Dict with 'diagnoser', 'consultant', and 'qa' model names.
    """
    if mode is None:
        mode = get_llm_mode()
    
    if mode == "prod":
        return {
            "diagnoser": os.environ.get("LLM_PROD_DIAGNOSER_MODEL", DEFAULT_PROD_MODELS["diagnoser"]),
            "consultant": os.environ.get("LLM_PROD_CONSULTANT_MODEL", DEFAULT_PROD_MODELS["consultant"]),
            "qa": os.environ.get("LLM_PROD_QA_MODEL", DEFAULT_PROD_MODELS["qa"]),
        }
    else:
        return {
            "diagnoser": os.environ.get("LLM_TEST_DIAGNOSER_MODEL", DEFAULT_TEST_MODELS["diagnoser"]),
            "consultant": os.environ.get("LLM_TEST_CONSULTANT_MODEL", DEFAULT_TEST_MODELS["consultant"]),
            "qa": os.environ.get("LLM_TEST_QA_MODEL", DEFAULT_TEST_MODELS["qa"]),
        }


def build_client(mode: LLMMode = None) -> BaseLLMClient:
    """
    Build an LLM client for the given mode.
    
    Args:
        mode: 'test' or 'prod'. If None, uses get_llm_mode().
    
    Returns:
        An LLM client instance (OllamaClient for test, OpenAICompatibleClient for prod).
    """
    if mode is None:
        mode = get_llm_mode()
    
    if mode == "prod":
        return OpenAICompatibleClient()
    else:
        return OllamaClient()


def build_clients(
    mode: LLMMode = None,
    diag_model_override: str = None,
    consult_model_override: str = None,
    qa_model_override: str = None
) -> Tuple[BaseLLMClient, BaseLLMClient, str, str, str]:
    """
    Build LLM clients and get model names for the pipeline.
    
    This is the main entry point for getting configured clients.
    Supports UI overrides for model names while using environment-configured clients.
    
    Args:
        mode: 'test' or 'prod'. If None, uses get_llm_mode().
        diag_model_override: Optional override for diagnoser model name.
        consult_model_override: Optional override for consultant model name.
        qa_model_override: Optional override for QA model name.
    
    Returns:
        Tuple of (client, diagnoser_model, consultant_model, qa_model).
        Note: In the current design, all roles use the same client but may
        use different models.
    """
    if mode is None:
        mode = get_llm_mode()
    
    client = build_client(mode)
    defaults = get_default_models(mode)
    
    diag_model = diag_model_override or defaults["diagnoser"]
    consult_model = consult_model_override or defaults["consultant"]
    qa_model = qa_model_override or defaults["qa"]
    
    return client, diag_model, consult_model, qa_model


def get_available_ollama_models() -> list:
    """
    Get list of available models from Ollama.
    Returns empty list if Ollama is not running.
    """
    try:
        client = OllamaClient()
        return client.list_models()
    except Exception:
        return []


class LLMConfig:
    """
    Configuration container for LLM settings.
    Can be passed around and modified by UI components.
    """
    
    def __init__(
        self,
        mode: LLMMode = None,
        diagnoser_model: str = None,
        consultant_model: str = None,
        qa_model: str = None
    ):
        """
        Initialize LLM configuration.
        
        Args:
            mode: 'test' or 'prod'. If None, uses environment default.
            diagnoser_model: Model name for diagnoser. If None, uses default.
            consultant_model: Model name for consultant. If None, uses default.
            qa_model: Model name for QA extraction. If None, uses default.
        """
        self.mode = mode or get_llm_mode()
        defaults = get_default_models(self.mode)
        
        self.diagnoser_model = diagnoser_model or defaults["diagnoser"]
        self.consultant_model = consultant_model or defaults["consultant"]
        self.qa_model = qa_model or defaults["qa"]
        
        self._client: BaseLLMClient = None
    
    @property
    def client(self) -> BaseLLMClient:
        """Get or create the LLM client."""
        if self._client is None:
            self._client = build_client(self.mode)
        return self._client
    
    @client.setter
    def client(self, value: BaseLLMClient):
        """Set a custom LLM client."""
        self._client = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "mode": self.mode,
            "diagnoser_model": self.diagnoser_model,
            "consultant_model": self.consultant_model,
            "qa_model": self.qa_model,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        """Create config from dictionary."""
        return cls(
            mode=data.get("mode"),
            diagnoser_model=data.get("diagnoser_model"),
            consultant_model=data.get("consultant_model"),
            qa_model=data.get("qa_model"),
        )

