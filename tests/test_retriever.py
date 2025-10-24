from langchain_core.documents import Document

from src.retriever.vector_store import build_faiss_index, load_index


def test_build_and_load_index(tmp_path, monkeypatch):
    tmp_dir = tmp_path / "index"
    docs = [Document(page_content="Hello world."), Document(page_content="Another doc.")]

    build_faiss_index(docs, index_dir=str(tmp_dir))
    db = load_index(index_dir=str(tmp_dir))

    retriever = db.as_retriever(search_kwargs={"k": 1})
    results = retriever.get_relevant_documents("Hello")
    assert len(results) >= 1

