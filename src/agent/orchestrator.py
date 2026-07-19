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
        # Preserve document order so we can build multi-chunk premise windows.
        # This assumes `chunks` is passed in the order produced by the chunker
        # (i.e. sequential chunks from the same document are adjacent in the list).
        self.chunk_order = {chunk["chunk_id"]: i for i, chunk in enumerate(chunks)}
        
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
        If a sentence is missing a citation or cites the wrong chunk, first try
        to find support in another retrieved chunk, then in the full indexed
        document set. This handles near-miss citations from the generator, such
        as citing a neighboring overlap chunk or omitting a citation. If no chunk
        supports it, preserve the sentence and add a warning.
        Returns the verified answer and a list of unsupported claims.
        """
        sentences = re.split(r'(?<=[.!?]) +', draft)
        verified_sentences = []
        unsupported_claims = []
        context_map = {chunk["chunk_id"]: chunk for chunk in context_chunks}
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Extract citations like (source: c1, c2)
            match = re.search(r'\(source:\s*([^\)]+)\)', sentence)
            if match:
                citations = [c.strip() for c in match.group(1).split(',')]
                claim = self._claim_text(sentence)
                
                # Check entailment against cited chunks
                entailed = False
                for cid in citations:
                    # Allow fuzzy matching if models generate slight variations like doc1_0
                    clean_cid = cid.replace("chunk_id:", "").strip()
                    if clean_cid in self.chunk_map:
                        # Try the cited chunk alone first (cheapest, most precise).
                        premise = self.chunk_map[clean_cid]["text"]
                        label = self.critic.check_entailment(premise, claim)
                        if label == 'entailment':
                            entailed = True
                            break # One supporting chunk is enough

                        # The claim may combine a fact from this chunk with a
                        # fact from an adjacent chunk (the chunker can split a
                        # single paragraph mid-thought). Retry with a window
                        # that includes the neighboring chunks before giving up.
                        windowed_premise = self._windowed_premise(clean_cid)
                        if windowed_premise != premise:
                            label = self.critic.check_entailment(windowed_premise, claim)
                            if label == 'entailment':
                                entailed = True
                                break
                
                if entailed:
                    verified_sentences.append(sentence)
                else:
                    recovered_citation = self._find_supporting_citation(
                        claim,
                        context_map,
                    )
                    if recovered_citation:
                        logger.info(f"Recovered support for claim from chunk: {recovered_citation}")
                        verified_sentences.append(
                            self._replace_or_append_citation(sentence, recovered_citation)
                        )
                    else:
                        logger.warning(f"Failed verification: {claim}")
                        verified_sentences.append(
                            f"{sentence} [WARNING: UNSUPPORTED BY CITATION]"
                        )
                        unsupported_claims.append(claim)
            else:
                claim = self._claim_text(sentence)
                if not claim:
                    continue
                recovered_citation = self._find_supporting_citation(
                    claim,
                    context_map,
                )
                if recovered_citation:
                    logger.info(f"Added missing citation for claim from chunk: {recovered_citation}")
                    verified_sentences.append(
                        self._replace_or_append_citation(claim, recovered_citation)
                    )
                else:
                    logger.warning(f"Failed verification for uncited claim: {claim}")
                    verified_sentences.append(
                        f"{claim} [WARNING: UNSUPPORTED BY CITATION]"
                    )
                    unsupported_claims.append(claim)
                
        return " ".join(verified_sentences), unsupported_claims

    def _windowed_premise(self, chunk_id: str, window: int = 1) -> str:
        """
        Builds a premise from `chunk_id` plus its `window` nearest neighbors on
        each side (in document order), concatenated in order. This lets the
        critic verify claims whose facts were split across a chunk boundary by
        the chunker, instead of only checking a single 400-char chunk in
        isolation. Falls back to the single chunk's text if ordering info or
        neighbors aren't available.
        """
        chunk = self.chunk_map.get(chunk_id)
        if chunk is None:
            return ""
        position = self.chunk_order.get(chunk_id)
        if position is None:
            return chunk["text"]

        # Only stitch together chunks that belong to the same source document,
        # so we don't accidentally merge unrelated documents at a boundary.
        doc_id = chunk.get("doc_id")
        ids_by_position = {v: k for k, v in self.chunk_order.items()}

        pieces = []
        for offset in range(-window, window + 1):
            neighbor_id = ids_by_position.get(position + offset)
            if neighbor_id is None:
                continue
            neighbor = self.chunk_map[neighbor_id]
            if doc_id is not None and neighbor.get("doc_id") != doc_id:
                continue
            pieces.append(neighbor["text"])

        return " ".join(pieces) if pieces else chunk["text"]

    def _find_supporting_citation(self, claim: str, chunks: dict[str, dict]) -> str | None:
        chunk_sets = [chunks]
        if chunks.keys() != self.chunk_map.keys():
            chunk_sets.append(self.chunk_map)

        for chunk_set in chunk_sets:
            entailed_chunk_id = self._find_entailed_chunk(claim, chunk_set)
            if entailed_chunk_id:
                return entailed_chunk_id

        return self._find_lexically_supported_chunk(claim, chunk_sets)

    def _find_entailed_chunk(self, claim: str, chunks: dict[str, dict]) -> str | None:
        for chunk_id, chunk in chunks.items():
            label = self.critic.check_entailment(chunk["text"], claim)
            if label == "entailment":
                return chunk_id

            # Same boundary-fragmentation retry as in the primary citation
            # check: a claim can legitimately draw on a fact that spills into
            # the neighboring chunk.
            windowed_premise = self._windowed_premise(chunk_id)
            if windowed_premise != chunk["text"]:
                label = self.critic.check_entailment(windowed_premise, claim)
                if label == "entailment":
                    return chunk_id
        return None

    def _find_lexically_supported_chunk(
        self,
        claim: str,
        chunk_sets: list[dict[str, dict]],
    ) -> str | None:
        best_chunk_id = None
        best_score = 0.0
        seen = set()
        for chunks in chunk_sets:
            for chunk_id, chunk in chunks.items():
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                score = self._lexical_support_score(chunk["text"], claim)
                if score > best_score:
                    best_chunk_id = chunk_id
                    best_score = score
        if best_chunk_id and best_score >= 0.28:
            return best_chunk_id
        return None

    @staticmethod
    def _replace_or_append_citation(sentence: str, chunk_id: str) -> str:
        citation = f"(source: {chunk_id})"
        if re.search(r'\(source:\s*[^\)]+\)', sentence):
            return re.sub(r'\(source:\s*[^\)]+\)', citation, sentence)

        stripped = sentence.rstrip()
        if stripped and stripped[-1] in ".!?":
            return f"{stripped[:-1].rstrip()} {citation}{stripped[-1]}"
        return f"{stripped} {citation}"

    @staticmethod
    def _claim_text(sentence: str) -> str:
        claim = re.sub(r'\(source:\s*[^\)]+\)', '', sentence)
        claim = claim.replace("**[WARNING: UNSUPPORTED BY CITATION]**", "")
        claim = claim.replace("[WARNING: UNSUPPORTED BY CITATION]", "")
        claim = claim.strip()
        return re.sub(r'\s+([.!?])', r'\1', claim)

    @staticmethod
    def _lexical_support_score(premise: str, claim: str) -> float:
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
        premise_terms = set(re.findall(r"\b[a-z0-9]{4,}\b", premise.lower())) - stopwords
        claim_terms = set(re.findall(r"\b[a-z0-9]{4,}\b", claim.lower())) - stopwords
        if not claim_terms:
            return 0.0
        return len(premise_terms & claim_terms) / len(claim_terms)
