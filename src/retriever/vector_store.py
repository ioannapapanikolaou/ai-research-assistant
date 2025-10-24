import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config.config import EMBEDDING_MODEL, INDEX_DIR


def build_faiss_index(docs: List[Document], index_dir: Optional[str] = None) -> FAISS:
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    db = FAISS.from_documents(docs, embedder)
    save_dir = index_dir or INDEX_DIR
    os.makedirs(save_dir, exist_ok=True)
    db.save_local(save_dir)
    return db


def load_index(index_dir: Optional[str] = None) -> FAISS:
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    load_dir = index_dir or INDEX_DIR
    return FAISS.load_local(load_dir, embedder, allow_dangerous_deserialization=True)


