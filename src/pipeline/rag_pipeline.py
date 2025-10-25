from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from src.config.config import MODEL_NAME, LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.retriever.vector_store import load_index


def make_rag_pipeline():
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    if (LLM_PROVIDER or "ollama").lower() == "ollama":
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
    else:
        llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

    return RetrievalQA.from_chain_type(llm, retriever=retriever)


