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

MODEL_NAME = _get_env("MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")

