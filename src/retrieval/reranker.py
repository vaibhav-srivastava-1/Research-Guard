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

from src.config import RERANKER_MODEL

class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL):
        self.model = self._load_model(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates:
            return []
            
        if self.model is not None:
            pairs = [[query, candidate["text"]] for candidate in candidates]
            scores = self.model.predict(pairs)
        else:
            scores = [self._lexical_score(query, candidate["text"]) for candidate in candidates]
        
        # Assign scores and sort
        for idx, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[idx])
            
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        # Assign ranks
        for rank, item in enumerate(reranked):
            item["final_rank"] = rank + 1
            
        return reranked[:top_k]

    def _load_model(self, model_name: str):
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder(model_name)
        except Exception:
            return None

    def _lexical_score(self, query: str, text: str) -> float:
        import re

        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        text_terms = set(re.findall(r"\b\w+\b", text.lower()))
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)
