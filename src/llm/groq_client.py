"""
STAGE 3 — LLM Access (Groq, free tier)

Goal: a thin wrapper around the Groq API (via langchain-groq) so the
rest of the code doesn't need to know provider details. This makes it
easy to swap Groq for Ollama later (config.yaml -> llm.provider).

Key function to build:

    get_llm(model_name: str = None, temperature: float = None) -> ChatGroq

Remember: GROQ_API_KEY must be set in your .env file (never hardcoded).
"""
import os
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)
cfg = load_config()


def get_llm(model_name: str = None, temperature: float = None):
    """
    TODO (build this together):
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=model_name or cfg["llm"]["model_name"],
        temperature=temperature if temperature is not None else cfg["llm"]["temperature"],
        max_tokens=cfg["llm"]["max_tokens"],
        api_key=os.environ["GROQ_API_KEY"],
    )
    """
    raise NotImplementedError("Stage 3: implement Groq LLM client wrapper")
