from src.config import RRF_K

def reciprocal_rank_fusion(bm25_results: list[dict], dense_results: list[dict], k: int = RRF_K) -> list[dict]:
    """
    Combines ranks from BM25 and Dense retrievers using RRF.
    Score = 1 / (k + rank)
    """
    scores = {}
    chunk_map = {}
    
    # Process BM25
    for item in bm25_results:
        chunk_id = item["chunk_id"]
        rank = item["bm25_rank"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        chunk_map[chunk_id] = item
        
    # Process Dense
    for item in dense_results:
        chunk_id = item["chunk_id"]
        rank = item["dense_rank"]
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
        
        # Merge if it exists in both, else add
        if chunk_id in chunk_map:
            chunk_map[chunk_id]["dense_score"] = item.get("dense_score")
            chunk_map[chunk_id]["dense_rank"] = item.get("dense_rank")
        else:
            chunk_map[chunk_id] = item
            
    # Sort by RRF score
    sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for rank, (chunk_id, score) in enumerate(sorted_chunks):
        item = chunk_map[chunk_id].copy()
        item["rrf_score"] = score
        item["rrf_rank"] = rank + 1
        final_results.append(item)
        
    return final_results
