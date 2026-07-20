import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

# Force every tqdm progress bar to be disabled at the class level, permanently.
import tqdm.std
if not getattr(tqdm.std.tqdm, "_patched_disabled", False):
    _original_tqdm_init = tqdm.std.tqdm.__init__
    def _disabled_tqdm_init(self, *args, **kwargs):
        kwargs["disable"] = True
        _original_tqdm_init(self, *args, **kwargs)
    tqdm.std.tqdm.__init__ = _disabled_tqdm_init
    tqdm.std.tqdm._patched_disabled = True

# Disable Hugging Face progress bars programmatically when available.
try:
    import huggingface_hub.utils
    huggingface_hub.utils.disable_progress_bars()
except Exception:
    pass

from src.config import NLI_MODEL
from src.utils import setup_logger

logger = setup_logger(__name__)

class CriticAgent:
    def __init__(self, model_name: str = NLI_MODEL):
        logger.info(f"Loading Critic NLI model: {model_name}...")
        self.classifier = self._load_classifier(model_name)
        if self.classifier is None:
            logger.warning("Critic NLI model unavailable; using lexical entailment fallback.")
        else:
            logger.info("Critic NLI model loaded.")

    def check_entailment(self, premise: str, hypothesis: str) -> str:
        """
        Uses an NLI model to check if the premise (retrieved chunk) entails the hypothesis (drafted claim).
        Returns 'entailment', 'neutral', or 'contradiction'.
        Note: Roberta-MNLI labels are typically: 'ENTAILMENT', 'NEUTRAL', 'CONTRADICTION'.
        """
        if self.classifier is None:
            return self._fallback_entailment(premise, hypothesis)

        result = self.classifier(
            f"{premise} </s></s> {hypothesis}",
            truncation=True,
            max_length=512,
        )[0]
        label = result['label'].lower()
        score = result['score']

        logger.debug(f"Critic Evaluation: {label} (Score: {score:.3f}) | Claim: {hypothesis}")

        return label

    def _load_classifier(self, model_name: str):
        try:
            from transformers import pipeline
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            return pipeline(
                "text-classification",
                model=model_name,
                return_all_scores=False,
                truncation=True,
                max_length=512,
            )
        except Exception as exc:
            logger.warning(f"Failed to load Critic NLI model: {exc}")
            return None

    def _fallback_entailment(self, premise: str, hypothesis: str) -> str:
        import re

        premise_terms = set(re.findall(r"\b\w{4,}\b", premise.lower()))
        hypothesis_terms = set(re.findall(r"\b\w{4,}\b", hypothesis.lower()))
        if not hypothesis_terms:
            return "neutral"
        overlap = len(premise_terms & hypothesis_terms) / len(hypothesis_terms)
        return "entailment" if overlap >= 0.35 else "neutral"
