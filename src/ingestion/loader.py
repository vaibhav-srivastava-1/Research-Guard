from pathlib import Path
from src.utils import setup_logger

logger = setup_logger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


def _load_text_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf_file(file_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("Install pypdf to read PDF uploads.")
        return ""

    reader = PdfReader(str(file_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _load_docx_file(file_path: Path) -> str:
    try:
        from docx import Document
    except ImportError:
        logger.warning("Install python-docx to read DOCX uploads.")
        return ""

    document = Document(str(file_path))
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def _read_supported_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return _load_text_file(file_path)
    if suffix == ".pdf":
        return _load_pdf_file(file_path)
    if suffix == ".docx":
        return _load_docx_file(file_path)
    return ""


def load_documents(directory: Path | str) -> list[dict]:
    """
    Reads supported document files in the given directory and returns document dicts.
    """
    directory = Path(directory)
    docs = []
    
    if not directory.exists():
        logger.warning(f"Directory {directory} does not exist.")
        return docs

    for file_path in directory.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = _read_supported_document(file_path).strip()
            if not text:
                logger.warning(f"No readable text found in {file_path}.")
                continue
            docs.append({
                "doc_id": file_path.stem,
                "text": text,
                "metadata": {"source": str(file_path)}
            })
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            
    logger.info(f"Loaded {len(docs)} documents from {directory}")
    return docs
