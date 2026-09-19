"""
Run this occasionally (or before a demo) to confirm your configured
models are still live on Groq, since their free-tier lineup changes
without much warning.
Run with: uv run python scripts/check_groq_models.py
"""
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.environ["GROQ_API_KEY"])

models = client.models.list()
print("Currently available Groq models:\n")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")