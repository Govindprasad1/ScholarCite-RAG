"""
Diagnostic — runs ONLY the ResponseRelevancy metric on a single sample,
without Ragas' error-swallowing, to see the real underlying exception.
Run with: uv run python notebooks/diagnose_relevancy.py
"""
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings
from ragas import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ResponseRelevancy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llm.groq_client import get_llm
from src.utils.config import load_config

config = load_config()
judge_llm = LangchainLLMWrapper(get_llm(role="generation"))
judge_embeddings = LangchainEmbeddingsWrapper(
    HuggingFaceEmbeddings(model_name=config["embeddings"]["model_name"])
)

sample = SingleTurnSample(
    user_input="How many attention heads are used in the base Transformer model?",
    response="The base Transformer model uses 8 parallel attention heads.",
    retrieved_contexts=["In this work we employ h = 8 parallel attention layers, or heads."],
)

scorer = ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings, strictness=1)

try:
    score = asyncio.run(scorer.single_turn_ascore(sample))
    print(f"Response relevancy: {score:.3f}")
except Exception as e:
    import traceback
    print("FULL ERROR TRACEBACK:")
    traceback.print_exc()