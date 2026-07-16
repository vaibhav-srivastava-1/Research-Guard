import pytest
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker

@pytest.fixture
def sample_chunks():
    return [
        {"chunk_id": "c1", "text": "The quick brown fox jumps over the lazy dog."},
        {"chunk_id": "c2", "text": "A fast brown fox leaped over a sleepy dog."},
        {"chunk_id": "c3", "text": "Apples are delicious and healthy fruits."},
        {"chunk_id": "c4", "text": "The financial crisis of 2008 was devastating."}
    ]

def test_bm25_retrieval(sample_chunks):
    retriever = BM25Retriever(sample_chunks)
    results = retriever.retrieve("fox dog", top_k=2)
    assert len(results) == 2
    assert "bm25_score" in results[0]
    assert results[0]["chunk_id"] in ["c1", "c2"] # Lexical match

def test_dense_retrieval(sample_chunks):
    retriever = DenseRetriever(sample_chunks)
    results = retriever.retrieve("economic collapse", top_k=2)
    assert len(results) == 2
    assert "dense_score" in results[0]
    # Semantic match for economic collapse -> financial crisis
    assert results[0]["chunk_id"] == "c4"

def test_fusion_and_reranker(sample_chunks):
    bm25_ret = BM25Retriever(sample_chunks)
    dense_ret = DenseRetriever(sample_chunks)
    
    query = "swift fox"
    bm25_res = bm25_ret.retrieve(query, top_k=2)
    dense_res = dense_ret.retrieve(query, top_k=2)
    
    fused = reciprocal_rank_fusion(bm25_res, dense_res)
    assert len(fused) > 0
    assert "rrf_score" in fused[0]
    
    reranker = Reranker()
    reranked = reranker.rerank(query, fused, top_k=1)
    
    assert len(reranked) == 1
    assert "final_rank" in reranked[0]
    assert "rerank_score" in reranked[0]
