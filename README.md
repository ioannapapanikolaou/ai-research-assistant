## AI Research Assistant (RAG-based Knowledge Retrieval)

**Goal**: A small RAG pipeline that ingests PDFs/text reports, builds a FAISS index with OpenAI embeddings, retrieves relevant chunks, and generates grounded answers via an LLM. Exposed as a FastAPI app or a simple Streamlit UI.

### Tech Stack
- **LangChain**, **FAISS**, **OpenAI** (Chat + Embeddings)
- **FastAPI**/**Streamlit**, **Uvicorn**
- **Docker**

### Repository Layout
```
ai-research-assistant/
  src/
    loaders/
    retriever/
    pipeline/
    api/
    config/
  tests/
  data/
  notebooks/
  Dockerfile
  requirements.txt
  .env (git-ignored)
  README.md
  main.py
```

### Quickstart (Local)
1) Create and activate a virtualenv, then install deps:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Configure provider and environment:
```bash
# Create .env and choose providers
cat > .env <<'ENV'
# LLM provider: ollama or openai
LLM_PROVIDER=ollama

# Embeddings provider: hf (HuggingFace), ollama, or openai
EMBEDDINGS_PROVIDER=hf

# OpenAI (paid)
# OPENAI_API_KEY=
# MODEL_NAME=gpt-4o-mini
# EMBEDDING_MODEL=text-embedding-3-small

# Ollama (local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3:8b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text

# HuggingFace (free/local embeddings)
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Paths
INDEX_DIR=vector_index
DATA_DIR=data
ENV
```

3) Put a couple PDFs or .txt files under `data/`.

4) Build the vector index:
```bash
python main.py build-index
```

5) Ask a question from CLI:
```bash
python main.py ask --q "Summarize the main findings"
```

6) Run FastAPI (ensure venv is active; use module form to avoid system Python):
```bash
python -m uvicorn src.api.app:app --reload
```
Visit `http://localhost:8000/docs` or `http://localhost:8000/ask?query=Your+question`.

### Using Ollama (free/local)
Install and run Ollama, then pull models:
```bash
brew install ollama
ollama serve &
ollama pull llama3:8b-instruct
ollama pull nomic-embed-text
```

Set `LLM_PROVIDER=ollama` and prefer `EMBEDDINGS_PROVIDER=hf` for stability. Then build the index and run the API as usual.

### Docker
```bash
docker build -t ai-research-assistant .
docker run -p 8000:8000 --env-file .env ai-research-assistant
```

### Tests
```bash
pytest -q
```

### Notes
- Index path defaults to `vector_index/` and can be changed via `.env`.
- Uses modern LangChain packages: `langchain-openai`, `langchain-community`.
- If you prefer Streamlit, see `src/api/app.py` for FastAPI and extend with Streamlit per the README plan.

### Future improvements
- Dynamic uploads, caching, evaluation notebook, cloud deployment.

