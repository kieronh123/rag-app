import os
import gradio as gr
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import pickle

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents and FAISS index
with open("docs.pkl", "rb") as f:
    documents = pickle.load(f)

index = faiss.read_index("faiss.index")
doc_embeddings = np.load("doc_embeddings.npy")

# Load OpenAI or Hugging Face LLM pipeline
generator = pipeline("text-generation", model="gpt2")  # or use OpenAI API

def search_and_generate(query):
    query_vec = embedder.encode([query])
    top_k = 3
    _, indices = index.search(query_vec, top_k)
    context = "\n".join([documents[i] for i in indices[0]])
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_new_tokens=100, do_sample=True)[0]["generated_text"]
    return response.strip()

iface = gr.Interface(fn=search_and_generate,
                     inputs=gr.Textbox(lines=2, placeholder="Ask a question..."),
                     outputs="text",
                     title="Mini RAG Demo",
                     description="Ask questions about local documents using semantic search + GPT.")

if __name__ == "__main__":
    iface.launch()
