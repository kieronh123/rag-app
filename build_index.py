import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Example dummy docs; in real use, load from PDFs or text files
documents = [
    "The mitochondria is the powerhouse of the cell.",
    "Paris is the capital of France.",
    "Transformers are state-of-the-art models for NLP tasks."
]

# Save for later retrieval
with open("docs.pkl", "wb") as f:
    pickle.dump(documents, f)

# Embed the documents
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Save embeddings
np.save("doc_embeddings.npy", doc_embeddings)

# Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
faiss.write_index(index, "faiss.index")

print("✅ FAISS index built and saved.")
