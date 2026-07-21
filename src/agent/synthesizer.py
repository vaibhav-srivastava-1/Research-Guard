import os
import re
from pathlib import Path
from openai import OpenAI
from src.config import GENERATOR_MODEL
from src.utils import setup_logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
logger = setup_logger(__name__)

class SynthesizerAgent:
    def __init__(self, model_name: str = GENERATOR_MODEL):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY") or "dummy_key"
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def synthesize(self, query: str, context_chunks: list[dict]) -> str:
        """
        Drafts an answer to the query based on the retrieved context chunks.
        Strictly enforces inline citations.
        """
        
        # Format context for the prompt
        context_text = ""
        for chunk in context_chunks:
            context_text += f"\n--- [Chunk ID: {chunk['chunk_id']}] ---\n{chunk['text']}\n"

        system_prompt = (
            "You are an expert Research Synthesizer. You will be provided with a user query and a set of retrieved text chunks.\n"
            "Your job is to write a comprehensive, factual answer.\n\n"
            "CRITICAL RULES:\n"
            "1. Every factual sentence you write MUST end with an inline citation indicating exactly which chunk supports it.\n"
            "2. Citation format must be exactly: (source: CHUNK_ID)\n"
            "3. Example: The sky is blue (source: doc1_0).\n"
            "4. If multiple chunks support a claim, cite them like: (source: doc1_0, doc2_3).\n"
            "5. Use the chunk ID from the same chunk that contains the specific fact in that sentence.\n"
            "6. Do not write uncited setup, transition, summary, or conclusion sentences.\n"
            "7. Keep each sentence to ONE atomic, independently-verifiable fact. Do not combine facts from "
            "different chunks into a single compound sentence (e.g. do not join a cause from one chunk with an "
            "effect from another chunk in the same sentence). If a full explanation needs multiple facts, write "
            "them as separate cited sentences instead of one long sentence.\n"
            "8. Do not repeat the same claim in multiple sentences, even if it appears in multiple chunks.\n"
            "9. Prefer a compact answer that covers distinct points once.\n"
            "10. If the provided context chunks DO NOT contain the answer, respond with exactly this sentence and no citation: "
            "I do not have enough information in the provided document set to answer this question."
        )
        
        user_prompt = f"User Query: {query}\n\nRetrieved Context:\n{context_text}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2
            )
            
            answer = response.choices[0].message.content
            logger.info("Drafted answer successfully.")
            return self._deduplicate_sentences(answer)
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            if context_chunks:
                logger.info("Generating query-aware local heuristic fallback response from context chunks...")
                return self._heuristic_fallback(query, context_chunks)
            return "I do not have enough information in the provided document set to answer this question."

    def _heuristic_fallback(self, query: str, context_chunks: list[dict]) -> str:
        import re

        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "who", "whom", "this", "that", "these", "those", "am", "is",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "how", "why", "where", "when", "can", "could",
            "should", "would", "about", "for", "with", "from", "into", "of", "to",
            "in", "on", "at", "by", "it", "its", "they", "them", "their"
        }

        query_words = [
            w.lower()
            for w in re.findall(r"\b[a-zA-Z0-9]{2,}\b", query)
            if w.lower() not in stopwords
        ]

        if not context_chunks or not query_words:
            return "I do not have enough information in the provided document set to answer this question."

        candidate_sentences = []
        seen_sentences = set()

        for chunk in context_chunks:
            chunk_id = chunk["chunk_id"]
            raw_sents = re.split(r"(?<=[.!?])\s+", chunk["text"])
            for sent in raw_sents:
                s_clean = sent.strip()
                if not s_clean or s_clean in seen_sentences:
                    continue
                seen_sentences.add(s_clean)

                sent_words = set(
                    w.lower() for w in re.findall(r"\b[a-zA-Z0-9]{2,}\b", s_clean)
                )
                matches = sum(1 for q_term in query_words if q_term in sent_words)

                if matches > 0:
                    punctuation = ""
                    if s_clean[-1] in ".!?":
                        punctuation = s_clean[-1]
                        s_clean = s_clean[:-1].strip()
                    formatted = f"{s_clean} (source: {chunk_id}){punctuation}"
                    candidate_sentences.append((matches, formatted))

        if not candidate_sentences:
            return "I do not have enough information in the provided document set to answer this question."

        # Sort candidate sentences by relevance match score descending
        candidate_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [item[1] for item in candidate_sentences[:4]]

        return self._deduplicate_sentences(" ".join(top_sentences))

    @classmethod
    def _deduplicate_sentences(cls, text: str, threshold: float = 0.5) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        kept_sentences = []
        seen_signatures = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            signature = cls._sentence_signature(sentence)
            if signature and any(cls._jaccard(signature, seen) >= threshold for seen in seen_signatures):
                continue

            kept_sentences.append(sentence)
            if signature:
                seen_signatures.append(signature)

        return " ".join(kept_sentences)

    @staticmethod
    def _sentence_signature(sentence: str) -> set[str]:
        sentence = re.sub(r'\(source:\s*[^\)]+\)', '', sentence)
        stopwords = {
            "about",
            "after",
            "also",
            "because",
            "been",
            "being",
            "from",
            "have",
            "including",
            "into",
            "that",
            "their",
            "there",
            "these",
            "this",
            "through",
            "were",
            "which",
            "with",
        }
        terms = re.findall(r"\b[a-z0-9]{4,}\b", sentence.lower())
        return {SynthesizerAgent._stem_token(term) for term in terms} - stopwords

    @staticmethod
    def _stem_token(term: str) -> str:
        for suffix in ("ingly", "edly", "ments", "ment", "tion", "ions", "ing", "ed", "es", "s"):
            if len(term) > len(suffix) + 3 and term.endswith(suffix):
                return term[: -len(suffix)]
        return term

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
