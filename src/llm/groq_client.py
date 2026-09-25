"""
Stage 3 — LLM Client

Thin wrapper around Groq's API via langchain-groq. Nothing else in the
codebase should import ChatGroq directly — everything goes through
here, so swapping providers later (e.g. to Ollama) only touches this
one file.

Supports two roles, each with its own model (set in config.yaml):
  - "generation"   -> the higher-quality model used to produce the
                       actual answer shown to the user
  - "verification" -> a cheaper/faster model used in Stage 4 to fact-
                       check the generated answer against the retrieved
                       source chunks

Groq's free-tier model lineup changes fairly often (several models were
deprecated in 2026) — if you hit a `model_not_found` error, run
`scripts/check_groq_models.py` to see what's currently live for your
API key, then update the model names in config.yaml accordingly. No
code changes are needed here when that happens.
"""

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import RateLimitError

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.utils.config import load_config
from src.utils.logger import get_logger

load_dotenv()  # reads .env into os.environ

logger = get_logger(__name__)


@lru_cache(maxsize=2)
def get_llm(role: str = "generation") -> ChatGroq:
    """
    Load (once per role) and cache a Groq LLM client.

    Args:
        role: "generation" or "verification" — determines which model
            key from config.yaml is used (llm.generation_model or
            llm.verification_model). Defaults to "generation" for
            Stage 3, where there's no verification step yet.

    Returns:
        A cached ChatGroq instance configured with the model,
        temperature, and max_tokens from config.yaml.

    Raises:
        ValueError: if GROQ_API_KEY is missing from the environment,
            or if `role` doesn't match a known config key.

    Note on caching: lru_cache keys on the function's arguments
    automatically, so get_llm("generation") and get_llm("verification")
    are cached as two separate client instances — each role's client
    is only ever built once per process, not once per call.
    """
    config = load_config()
    llm_config = config["llm"]

    if role not in ("generation", "verification"):
        raise ValueError(f"Unknown role '{role}' — expected 'generation' or 'verification'")

    model_key = "generation_model" if role == "generation" else "verification_model"
    model_name = llm_config[model_key]

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file — "
            "get a free key at https://console.groq.com/keys"
        )

    logger.info(f"Initializing Groq LLM ({role}): {model_name}")
    return ChatGroq(
        model=model_name,
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        api_key=api_key,
    )
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
)
def invoke_with_retry(llm, prompt: str):
    """
    Wraps llm.invoke() with exponential backoff specifically for Groq's
    rate-limit errors (429s). Waits 2s, then 4s, then 8s... up to 4
    total attempts before giving up. This matters because Groq's free
    tier has real per-minute/per-day request caps — without this, a
    demo could crash mid-session just from normal testing traffic.
    """
    return llm.invoke(prompt)