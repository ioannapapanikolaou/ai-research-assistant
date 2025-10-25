import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings

from src.config.config import (
    EMBEDDING_MODEL,
    INDEX_DIR,
    EMBEDDINGS_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    HF_EMBEDDING_MODEL,
)


def _get_embedder():
    provider = (EMBEDDINGS_PROVIDER or "hf").lower()
    if provider == "openai":
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if provider == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    # default: HuggingFace
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)


def build_faiss_index(docs: List[Document], index_dir: Optional[str] = None) -> FAISS:
    embedder = _get_embedder()
    db = FAISS.from_documents(docs, embedder)
    save_dir = index_dir or INDEX_DIR
    os.makedirs(save_dir, exist_ok=True)
    db.save_local(save_dir)
    return db


def load_index(index_dir: Optional[str] = None) -> FAISS:
    embedder = _get_embedder()
    load_dir = index_dir or INDEX_DIR
    return FAISS.load_local(load_dir, embedder, allow_dangerous_deserialization=True)


