from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from src.config.config import MODEL_NAME
from src.retriever.vector_store import load_index


def make_rag_pipeline():
    db = load_index()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0)
    # Newer LC uses invoke(); keep .run for compatibility
    return RetrievalQA.from_chain_type(llm, retriever=retriever)


