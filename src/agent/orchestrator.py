import re
from src.agent.planner import PlannerAgent
from src.agent.synthesizer import SynthesizerAgent
from src.agent.critic import CriticAgent
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.utils import setup_logger

logger = setup_logger(__name__)

class ResearchOrchestrator:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}
        
        logger.info("Initializing components...")
        self.planner = PlannerAgent()
        self.synthesizer = SynthesizerAgent()
        self.critic = CriticAgent()
        
        self.bm25 = BM25Retriever(chunks)
        self.dense = DenseRetriever(chunks)
        self.reranker = Reranker()
        logger.info("Orchestrator ready.")

    def run(self, query: str) -> str:
        from src.config import MAX_CRITIC_RETRIES
        
        # Step 1: Plan
        sub_questions = self.planner.decompose(query)
        
        retries = 0
        while retries <= MAX_CRITIC_RETRIES:
            # Step 2: Retrieve
            all_top_chunks = {}
            for sq in sub_questions:
                sq_text = sq.get("sub_question", query)
                logger.info(f"Retrieving for sub-question: {sq_text}")
                
                b_res = self.bm25.retrieve(sq_text, top_k=20)
                d_res = self.dense.retrieve(sq_text, top_k=20)
                
                fused = reciprocal_rank_fusion(b_res, d_res)
                reranked = self.reranker.rerank(sq_text, fused, top_k=3)
                
                for chunk in reranked:
                    all_top_chunks[chunk["chunk_id"]] = chunk
                    
            context_chunks = list(all_top_chunks.values())
            
            # Step 3: Synthesize
            logger.info(f"Drafting response (Retry {retries})...")
            draft_answer = self.synthesizer.synthesize(query, context_chunks)
            
            # Step 4: Critic Loop (Parsing Citations and Verifying)
            logger.info("Starting Critic Verification...")
            verified_answer, unsupported_claims = self._verify_and_revise(draft_answer, context_chunks)
            
            if not unsupported_claims or retries >= MAX_CRITIC_RETRIES:
                return verified_answer
                
            logger.info(f"Found {len(unsupported_claims)} unsupported claims. Re-retrieving...")
            
            # Append unsupported claims as new sub-questions for the next iteration
            for claim in unsupported_claims:
                sub_questions.append({"sub_question": claim})
                
            retries += 1
            
        return verified_answer

    def _verify_and_revise(self, draft: str, context_chunks: list[dict]) -> tuple[str, list[str]]:
        """
        Parses sentences with (source: chunk_id) and runs entailment checks.
        If a sentence fails, we mark it as [UNSUPPORTED] in the text for this baseline implementation.
        Returns the verified answer and a list of unsupported claims.
        """
        sentences = re.split(r'(?<=[.!?]) +', draft)
        verified_sentences = []
        unsupported_claims = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Extract citations like (source: c1, c2)
            match = re.search(r'\(source:\s*([^\)]+)\)', sentence)
            if match:
                citations = [c.strip() for c in match.group(1).split(',')]
                claim = re.sub(r'\(source:\s*[^\)]+\)', '', sentence).strip()
                claim = re.sub(r'\s+([.!?])', r'\1', claim)
                
                # Check entailment against cited chunks
                entailed = False
                for cid in citations:
                    # Allow fuzzy matching if models generate slight variations like doc1_0
                    clean_cid = cid.replace("chunk_id:", "").strip()
                    if clean_cid in self.chunk_map:
                        premise = self.chunk_map[clean_cid]["text"]
                        label = self.critic.check_entailment(premise, claim)
                        if label == 'entailment':
                            entailed = True
                            break # One supporting chunk is enough
                
                if entailed:
                    verified_sentences.append(sentence)
                else:
                    logger.warning(f"Failed verification: {claim}")
                    verified_sentences.append(f"{claim} [WARNING: UNSUPPORTED BY CITATION]")
                    unsupported_claims.append(claim)
            else:
                # If a factual claim has no citation, we could flag it. For now, just pass it through.
                verified_sentences.append(sentence)
                
        return " ".join(verified_sentences), unsupported_claims
