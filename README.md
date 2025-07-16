# Mini RAG: Retrieval-Augmented Generation Demo

This project is a minimal implementation of a **Retrieval-Augmented Generation (RAG)** system using open-source tools. It allows users to ask natural language questions, and the system responds with answers generated from relevant documents using semantic search and a language model.

## What This Project Does

1. **Document Ingestion and Embedding**
   - A small set of example documents is embedded using a SentenceTransformer (`all-MiniLM-L6-v2`).
   - These embeddings are stored in a FAISS vector index for efficient similarity search.

2. **Retrieval**
   - When a user submits a question, it is also embedded using the same SentenceTransformer model.
   - The question embedding is compared against the stored document embeddings using FAISS to retrieve the top-k most relevant documents.

3. **Augmented Generation**
   - The retrieved document snippets are concatenated into a context prompt.
   - This context, along with the user's question, is passed into a GPT-2 language model using Hugging Face’s `pipeline`.
   - The model generates a natural language answer based on the context.

4. **Web Interface**
   - The application uses Gradio to provide a simple web-based interface.
   - Users type a question and receive a generated response that draws from the indexed documents.

## Folder Contents

- `app.py`: Main Gradio interface for querying the model.
- `build_index.py`: Script that builds document embeddings and creates a FAISS index.
- `requirements.txt`: All Python dependencies required to run the app.
- `docs.pkl`: Serialized list of input documents (created by `build_index.py`).
- `doc_embeddings.npy`: Numpy array of document embeddings.
- `faiss.index`: Binary FAISS index used for vector search.

## How to Run (PowerShell)

1. **Set up virtual environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

2. **Install dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Build FAISS index**
   ```powershell
   python build_index.py
   ```

4. **Start the app**
   ```powershell
   python app.py
   ```

The app will launch locally at `http://127.0.0.1:7860` where you can interact with the system.

## Use Cases

- Proof of concept for semantic search + generation applications
- Academic or technical document Q&A
- Building blocks for chatbot or assistant interfaces

## Notes

- This is a minimal and educational implementation using only 3 hardcoded documents.
- It can be easily extended to index PDFs, CSVs, or text scraped from websites.
- The generator currently uses `gpt2`, but can be swapped with a more powerful model or an OpenAI endpoint.

## License

MIT License
