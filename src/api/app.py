from fastapi import FastAPI, Query
from src.pipeline.rag_pipeline import make_rag_pipeline

app = FastAPI(title="AI Research Assistant")
qa = make_rag_pipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ask")
def ask(query: str = Query(..., description="User question")):
    answer = qa.invoke({"query": query}) if hasattr(qa, "invoke") else qa.run(query)
    return {"query": query, "answer": answer}


