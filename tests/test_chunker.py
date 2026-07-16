from src.ingestion.chunker import chunk_text, chunk_documents

def test_chunk_text():
    text = "word " * 100
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    
    # 100 words, chunks of 30, overlap 10 -> stride 20
    # chunk 0: 0-30
    # chunk 1: 20-50
    # chunk 2: 40-70
    # chunk 3: 60-90
    # chunk 4: 80-100 (20 words)
    assert len(chunks) == 5
    assert len(chunks[0].split()) == 30
    assert len(chunks[-1].split()) == 20

def test_chunk_documents():
    docs = [
        {"doc_id": "doc1", "text": "word " * 40, "metadata": {}},
        {"doc_id": "doc2", "text": "word " * 10, "metadata": {}}
    ]
    
    # doc1: chunk 0 (0-30), chunk 1 (20-40) -> 2 chunks
    # doc2: chunk 0 (0-10) -> 1 chunk
    
    chunked = chunk_documents(docs, chunk_size=30, overlap=10)
    assert len(chunked) == 3
    assert chunked[0]["chunk_id"] == "doc1_0"
    assert chunked[1]["chunk_id"] == "doc1_1"
    assert chunked[2]["chunk_id"] == "doc2_0"
