"""
Loads config.yaml into a simple, typed, dot-accessible object.

Usage:
    from src.utils.config import load_config
    cfg = load_config()
    print(cfg["chunking"]["chunk_size"])
"""
import yaml
from pathlib import Path
from functools import lru_cache

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
