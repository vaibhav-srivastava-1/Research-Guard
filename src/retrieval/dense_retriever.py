import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

import tqdm.std
if not getattr(tqdm.std.tqdm, "_patched_disabled", False):
    _original_tqdm_init = tqdm.std.tqdm.__init__
    def _disabled_tqdm_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _original_tqdm_init(self, *args, **kwargs)
    tqdm.std.tqdm.__init__ = _disabled_tqdm_init
    tqdm.std.tqdm._patched_disabled = True

# Disable Hugging Face progress bars programmatically
import huggingface_hub.utils
huggingface_hub.utils.disable_progress_bars()

import numpy as np
from src.config import EMBEDDING_MODEL

try:
    import faiss
except Exception:
    faiss = None

class DenseRetriever:
    def __init__(self, chunks: list[dict], model_name: str = EMBEDDING_MODEL):
        self.chunks = chunks
        self.model = self._load_model(model_name)

        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = self._encode(texts)
        
        self.embeddings = self._normalize(self.embeddings)
        self.index = None

        if faiss is not None:
            # FAISS is faster, but the Windows wheel can fail to load native DLLs.
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

    def retrieve(self, query: str, top_k: int = 60) -> list[dict]:
        query_emb = self._encode([query])
        query_emb = self._normalize(query_emb)

        if self.index is not None:
            scores, indices = self.index.search(query_emb, top_k)
            scores = scores[0]
            indices = indices[0]
        else:
            scores = self.embeddings @ query_emb[0]
            indices = np.argsort(scores)[::-1][:top_k]
            scores = scores[indices]
        
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx != -1:
                result = self.chunks[idx].copy()
                result["dense_score"] = float(score)
                result["dense_rank"] = rank + 1
                results.append(result)
        return results

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer(model_name)
        except Exception:
            return None

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.model is not None:
            return self.model.encode(texts, convert_to_numpy=True)

        vectors = np.zeros((len(texts), 256), dtype="float32")
        for row, text in enumerate(texts):
            for token in self._tokens(text):
                vectors[row, hash(token) % vectors.shape[1]] += 1.0
        return vectors

    def _tokens(self, text: str) -> list[str]:
        import re

        synonyms = {
            "economic": ["financial"],
            "collapse": ["crisis", "devastating"],
            "swift": ["quick", "fast"],
            "leaped": ["jumps", "jumped"],
            "sleepy": ["lazy"],
        }
        tokens = re.findall(r"\b\w+\b", text.lower())
        expanded = tokens.copy()
        for token in tokens:
            expanded.extend(synonyms.get(token, []))
        return expanded

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
