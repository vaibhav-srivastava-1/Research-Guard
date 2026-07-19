import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
USER_DATA_DIR = DATA_DIR / "users"
APP_DB_PATH = DATA_DIR / "researchguard.db"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Config
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Retrieval Config
TOP_K_BM25 = 60
TOP_K_DENSE = 60
RRF_K = 60
FINAL_TOP_K = 5

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "roberta-large-mnli"
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")

# Agent Config
MAX_CRITIC_RETRIES = 2
