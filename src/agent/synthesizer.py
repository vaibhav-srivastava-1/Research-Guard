import os
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
            "7. If the provided context chunks DO NOT contain the answer, you MUST state that you do not have enough information. Do not invent facts.\n"
            "8. Keep each sentence to ONE atomic, independently-verifiable fact. Do not combine facts from "
            "different chunks into a single compound sentence (e.g. do not join a cause from one chunk with an "
            "effect from another chunk in the same sentence). If a full explanation needs multiple facts, write "
            "them as separate cited sentences instead of one long sentence."
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
            return answer
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            if context_chunks:
                logger.info("Generating local heuristic fallback response from context chunks...")
                import re
                fallback_sentences = []
                for chunk in context_chunks[:3]:
                    sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
                    for sent in sentences[:2]:
                        if sent.strip():
                            s_clean = sent.strip()
                            punctuation = ""
                            if s_clean[-1] in ".!?":
                                punctuation = s_clean[-1]
                                s_clean = s_clean[:-1].strip()
                            fallback_sentences.append(f"{s_clean} (source: {chunk['chunk_id']}){punctuation}")
                if fallback_sentences:
                    return " ".join(fallback_sentences)
            return "Error generating response."
