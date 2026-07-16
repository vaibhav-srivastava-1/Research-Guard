from rank_bm25 import BM25Okapi
import numpy as np
import re

class BM25Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokenized_corpus = [self._preprocess(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        # Clamp IDF to positive values to avoid ignoring matching terms in small corpora
        for term, idf_val in self.bm25.idf.items():
            if idf_val <= 0:
                self.bm25.idf[term] = 1e-4

    def _preprocess(self, text: str) -> list[str]:
        # Lowercase and remove punctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()

    def retrieve(self, query: str, top_k: int = 60) -> list[dict]:
        tokenized_query = self._preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] > 0:
                result = self.chunks[idx].copy()
                result["bm25_score"] = float(scores[idx])
                result["bm25_rank"] = rank + 1
                results.append(result)
        return results

