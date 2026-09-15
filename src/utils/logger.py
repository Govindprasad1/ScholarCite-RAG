"""
Structured logging setup.

Why this exists (see architecture doc, MLOps section):
Using plain print() makes it impossible to filter, aggregate, or monitor
what's happening in the pipeline (retrieval latency, LLM latency, token
usage, verification pass/fail). This gives every module a consistent,
timestamped logger instead.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Retrieved %d chunks in %.2fs", len(chunks), elapsed)
"""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
