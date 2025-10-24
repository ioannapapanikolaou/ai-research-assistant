import argparse
import glob
import logging
import os
from typing import List

from dotenv import load_dotenv

from src.config.config import DATA_DIR, INDEX_DIR
from src.loaders.pdf_loader import load_and_split
from src.retriever.vector_store import build_faiss_index


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
load_dotenv()


def find_documents(data_dir: str) -> List[str]:
    patterns = [
        os.path.join(data_dir, "**", "*.pdf"),
        os.path.join(data_dir, "**", "*.txt"),
    ]
    files: List[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def cmd_build_index() -> None:
    files = find_documents(DATA_DIR)
    if not files:
        logging.warning("No PDF or TXT files found under %s", DATA_DIR)
        return

    all_chunks = []
    for path in files:
        logging.info("Loading and splitting: %s", path)
        try:
            chunks = load_and_split(path)
            all_chunks.extend(chunks)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to process %s: %s", path, exc)

    if not all_chunks:
        logging.warning("No chunks were produced. Aborting index build.")
        return

    logging.info("Building FAISS index with %d chunks...", len(all_chunks))
    build_faiss_index(all_chunks, index_dir=INDEX_DIR)
    logging.info("Index saved to %s", INDEX_DIR)


def cmd_ask(question: str) -> None:
    from src.pipeline.rag_pipeline import make_rag_pipeline  # lazy import

    qa = make_rag_pipeline()
    answer = qa.invoke({"query": question}) if hasattr(qa, "invoke") else qa.run(question)
    print(answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Research Assistant CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("build-index", help="Build FAISS index from documents in data/")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the index")
    ask_parser.add_argument("--q", "--query", dest="query", required=True)

    args = parser.parse_args()
    if args.command == "build-index":
        cmd_build_index()
    elif args.command == "ask":
        cmd_ask(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


