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

2) Copy env template and add your key:
```bash
cp .env.example .env
# edit .env with OPENAI_API_KEY=...
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

6) Run FastAPI:
```bash
uvicorn src.api.app:app --reload
```
Visit `http://localhost:8000/docs` or `http://localhost:8000/ask?query=Your+question`.

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

