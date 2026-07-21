import re
from src.agent.planner import PlannerAgent
from src.agent.synthesizer import SynthesizerAgent
from src.agent.critic import CriticAgent
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.utils import setup_logger
from src.config import FINAL_TOP_K

logger = setup_logger(__name__)

INSUFFICIENT_INFO_PATTERNS = [
    re.compile(r"\bdo(?:es)?\s+not\s+(?:have|contain|provide)\s+(?:enough\s+)?information\b", re.IGNORECASE),
    re.compile(r"\bnot\s+enough\s+information\b", re.IGNORECASE),
    re.compile(r"\binsufficient\s+information\b", re.IGNORECASE),
    re.compile(r"\bprovided\s+context\s+does\s+not\s+contain\s+information\b", re.IGNORECASE),
    re.compile(r"\bcontext\s+does\s+not\s+contain\s+information\b", re.IGNORECASE),
]


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
                    
            context_chunks = self._diversify_chunks(list(all_top_chunks.values()), max_chunks=FINAL_TOP_K)
            
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

            if self._is_insufficient_info(sentence):
                verified_sentences.append(self._strip_citations(sentence))
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
                
        verified_sentences = self._deduplicate_answer_sentences(verified_sentences)
        unsupported_claims = list(dict.fromkeys(unsupported_claims))
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
        chunk_order = getattr(self, "chunk_order", {})
        position = chunk_order.get(chunk_id)
        if position is None:
            return chunk["text"]

        # Only stitch together chunks that belong to the same source document,
        # so we don't accidentally merge unrelated documents at a boundary.
        doc_id = chunk.get("doc_id")
        ids_by_position = {v: k for k, v in chunk_order.items()}

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
    def _is_insufficient_info(sentence: str) -> bool:
        return any(pattern.search(sentence) for pattern in INSUFFICIENT_INFO_PATTERNS)

    @staticmethod
    def _strip_citations(sentence: str) -> str:
        stripped = re.sub(r'\s*\(source:\s*[^\)]+\)', '', sentence).strip()
        return re.sub(r'\s+([.!?])', r'\1', stripped)

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
        premise_terms = ResearchOrchestrator._term_signature(premise) - stopwords
        claim_terms = ResearchOrchestrator._term_signature(claim) - stopwords
        if not claim_terms:
            return 0.0
        return len(premise_terms & claim_terms) / len(claim_terms)

    @classmethod
    def _diversify_chunks(
        cls,
        chunks: list[dict],
        max_chunks: int = FINAL_TOP_K,
        similarity_threshold: float = 0.88,
    ) -> list[dict]:
        selected = []
        selected_signatures = []

        for chunk in chunks:
            signature = cls._term_signature(chunk.get("text", ""))
            if signature and any(
                cls._jaccard_similarity(signature, selected_signature) >= similarity_threshold
                for selected_signature in selected_signatures
            ):
                continue

            selected.append(chunk)
            selected_signatures.append(signature)
            if len(selected) >= max_chunks:
                return selected

        if selected:
            return selected
        return chunks[:max_chunks]

    @classmethod
    def _deduplicate_answer_sentences(
        cls,
        sentences: list[str],
        similarity_threshold: float = 0.5,
    ) -> list[str]:
        deduplicated = []
        signatures = []

        for sentence in sentences:
            signature = cls._term_signature(cls._claim_text(sentence))
            if signature and any(
                cls._jaccard_similarity(signature, existing) >= similarity_threshold
                for existing in signatures
            ):
                continue
            deduplicated.append(sentence)
            signatures.append(signature)

        return deduplicated

    @staticmethod
    def _term_signature(text: str) -> set[str]:
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
        terms = re.findall(r"\b[a-z0-9]{4,}\b", text.lower())
        return {ResearchOrchestrator._stem_token(term) for term in terms} - stopwords

    @staticmethod
    def _stem_token(term: str) -> str:
        for suffix in ("ingly", "edly", "ments", "ment", "tion", "ions", "ing", "ed", "es", "s"):
            if len(term) > len(suffix) + 3 and term.endswith(suffix):
                return term[: -len(suffix)]
        return term

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
