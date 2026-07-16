import os
from pathlib import Path
from src.utils import setup_logger

logger = setup_logger(__name__)

def load_documents(directory: Path | str) -> list[dict]:
    """
    Reads all text files in the given directory and returns a list of document dicts.
    """
    directory = Path(directory)
    docs = []
    
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist.")
        return docs

    for file_path in directory.rglob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                docs.append({
                    "doc_id": file_path.stem,
                    "text": text,
                    "metadata": {"source": str(file_path)}
                })
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            
    logger.info(f"Loaded {len(docs)} documents from {directory}")
    return docs
