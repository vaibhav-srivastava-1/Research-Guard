import json
import os
import re
from pathlib import Path
from openai import OpenAI
from src.config import GENERATOR_MODEL
from src.utils import setup_logger
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
logger = setup_logger(__name__)

class PlannerAgent:
    def __init__(self, model_name: str = GENERATOR_MODEL):
        self.model_name = model_name
        api_key = os.getenv("OPENAI_API_KEY") or "dummy_key"
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )

    def decompose(self, query: str) -> list[dict]:
        """
        Takes a complex user query and breaks it down into 2-5 sub-questions.
        If the query is atomic (simple), it returns a single sub-question.
        """
        system_prompt = (
            "You are a Research Planner Agent. Your job is to break down a complex user query "
            "into 2-5 distinct, answerable sub-questions to guide a retrieval system.\n"
            "If the query is simple and atomic, do not over-decompose; just return a single sub-question.\n"
            "Respond in JSON format with a key 'sub_questions', which is a list of objects. "
            "Each object must have 'sub_question', 'rationale', and 'expected_evidence_type'."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_content, re.DOTALL)
                if match:
                    raw_content = match.group(1).strip()
            
            try:
                result = json.loads(raw_content)
            except json.JSONDecodeError:
                # Regex search fallback to find any JSON object
                match = re.search(r"\{.*\}", raw_content, re.DOTALL)
                if match:
                    result = json.loads(match.group(0))
                else:
                    raise
            
            sub_qs = result.get("sub_questions", [])
            logger.info(f"Decomposed query into {len(sub_qs)} sub-questions.")
            return sub_qs
        except Exception as e:
            logger.error(f"Planner failed: {e}")
            # Fallback to the original query if API fails
            return [{"sub_question": query, "rationale": "Fallback due to API error", "expected_evidence_type": "text"}]
