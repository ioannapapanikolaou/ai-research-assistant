import os
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


INDEX_DIR = _get_env("INDEX_DIR", "vector_index")
DATA_DIR = _get_env("DATA_DIR", "data")

CHUNK_SIZE = int(_get_env("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get_env("CHUNK_OVERLAP", "100"))

# Back-compat single provider var
PROVIDER = _get_env("PROVIDER", "ollama")

# Separate providers (embeddings default to HF for stability; LLM inherits PROVIDER)
EMBEDDINGS_PROVIDER = _get_env("EMBEDDINGS_PROVIDER", "hf")
LLM_PROVIDER = _get_env("LLM_PROVIDER", PROVIDER)

# OpenAI (paid)
MODEL_NAME = _get_env("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")

# Ollama (local)
OLLAMA_BASE_URL = _get_env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _get_env("OLLAMA_MODEL", "llama3:8b-instruct")
OLLAMA_EMBED_MODEL = _get_env("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# HuggingFace (free/local)
HF_EMBEDDING_MODEL = _get_env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

