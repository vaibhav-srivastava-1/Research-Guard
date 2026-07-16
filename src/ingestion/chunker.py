import re
from src.utils import setup_logger
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = setup_logger(__name__)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into sentences and chunks them with the given size and overlap
    while respecting sentence boundaries.
    """
    raw_sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw_sentences:
        if not s.strip():
            continue
        words = s.split()
        if len(words) > chunk_size:
            step = overlap if overlap > 0 else chunk_size
            if step <= 0:
                step = 1
            i = 0
            while i < len(words):
                sentences.append(" ".join(words[i:i+step]))
                i += step
        else:
            sentences.append(s)
            
    chunks = []
    
    if not sentences:
        return chunks
        
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_words = sentence.split()
        sentence_len = len(sentence_words)
        
        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Keep the last few sentences as overlap (based on word count)
            overlap_length = 0
            overlap_chunk = []
            for s in reversed(current_chunk):
                s_len = len(s.split())
                if overlap_length + s_len > overlap and overlap_chunk:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += s_len
                
            current_chunk = overlap_chunk
            current_length = overlap_length
            
        current_chunk.append(sentence)
        current_length += sentence_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Takes a list of document dicts and returns a list of chunk dicts.
    """
    chunked_data = []
    
    for doc in documents:
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, chunk_text_val in enumerate(chunks):
            chunked_data.append({
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "text": chunk_text_val,
                "metadata": doc.get("metadata", {})
            })
            
    logger.info(f"Generated {len(chunked_data)} chunks from {len(documents)} documents.")
    return chunked_data

