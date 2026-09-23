# Run this once to check
from sentence_transformers import CrossEncoder
model = CrossEncoder("BAAI/bge-reranker-base")
scores = model.predict([("What is attention?", "The attention mechanism allows..."),
                         ("What is attention?", "Unrelated text about cooking recipes.")])
print(scores)