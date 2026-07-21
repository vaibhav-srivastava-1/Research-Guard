from src.agent.orchestrator import ResearchOrchestrator


class FakeCritic:
    def check_entailment(self, premise: str, hypothesis: str) -> str:
        del hypothesis
        if "housing bubble" in premise:
            return "entailment"
        return "neutral"


def test_verifier_recovers_support_from_retrieved_chunk():
    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.critic = FakeCritic()
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
        {
            "chunk_id": "doc_1",
            "text": "Dodd-Frank introduced new financial regulations after the crisis.",
        },
    ]
    orchestrator.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}

    draft = "The crisis was caused by the bursting of the United States housing bubble (source: doc_1)."
    verified, unsupported = orchestrator._verify_and_revise(draft, chunks)

    assert unsupported == []
    assert "(source: doc_0)" in verified
    assert "UNSUPPORTED" not in verified


def test_verifier_adds_missing_citation_for_supported_sentence():
    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.critic = FakeCritic()
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
    ]
    orchestrator.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}

    draft = (
        "The crisis was caused by the bursting of the United States housing bubble. "
        "**[WARNING: UNSUPPORTED BY CITATION]**"
    )
    verified, unsupported = orchestrator._verify_and_revise(draft, chunks)

    assert unsupported == []
    assert "(source: doc_0)" in verified
    assert "UNSUPPORTED" not in verified


def test_critic_agent_long_premise_truncation():
    from src.agent.critic import CriticAgent

    critic = CriticAgent()
    long_premise = "The 2008 financial crisis was triggered by subprime mortgage defaults. " * 80
    hypothesis = "The crisis was triggered by subprime default risks."
    # Should complete without throwing IndexOutOfBounds for position embeddings (size 514)
    label = critic.check_entailment(long_premise, hypothesis)
    assert label in ["entailment", "neutral", "contradiction"]


def test_synthesizer_query_aware_fallback():
    from src.agent.synthesizer import SynthesizerAgent

    synthesizer = SynthesizerAgent()
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by defaults on subprime mortgages. Banks stopped lending to each other.",
        }
    ]

    # Query 1: Banks lending
    ans1 = synthesizer._heuristic_fallback("Why did banks stop lending?", chunks)
    assert "Banks stopped lending" in ans1

    # Query 2: Unmentioned topic (COVID)
    ans2 = synthesizer._heuristic_fallback("How does it compare to COVID?", chunks)
    assert "do not have enough information" in ans2.lower()
    assert ans1 != ans2


def test_verifier_falls_back_to_all_chunks_when_context_misses_support():
    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.critic = FakeCritic()
    all_chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
        {
            "chunk_id": "doc_1",
            "text": "Dodd-Frank introduced new financial regulations after the crisis.",
        },
    ]
    context_chunks = [all_chunks[1]]
    orchestrator.chunk_map = {chunk["chunk_id"]: chunk for chunk in all_chunks}

    draft = "The crisis was caused by the bursting of the United States housing bubble."
    verified, unsupported = orchestrator._verify_and_revise(draft, context_chunks)

    assert unsupported == []
    assert "(source: doc_0)" in verified
    assert "UNSUPPORTED" not in verified


def test_verifier_removes_duplicate_supported_claims():
    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.critic = FakeCritic()
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
    ]
    orchestrator.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}

    draft = (
        "The crisis was caused by the bursting of the United States housing bubble (source: doc_0). "
        "The United States housing bubble caused the crisis (source: doc_0)."
    )
    verified, unsupported = orchestrator._verify_and_revise(draft, chunks)

    assert unsupported == []
    assert verified.count("(source: doc_0)") == 1


def test_verifier_accepts_insufficient_info_refusal_without_warning():
    orchestrator = ResearchOrchestrator.__new__(ResearchOrchestrator)
    orchestrator.critic = FakeCritic()
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The 2008 financial crisis involved mortgage-backed securities.",
        },
    ]
    orchestrator.chunk_map = {chunk["chunk_id"]: chunk for chunk in chunks}

    draft = (
        "The provided context does not contain information about the 2020 COVID-19 economic downturn "
        "(source: doc_0)."
    )
    verified, unsupported = orchestrator._verify_and_revise(draft, chunks)

    assert unsupported == []
    assert verified == "The provided context does not contain information about the 2020 COVID-19 economic downturn."
    assert "(source:" not in verified
    assert "UNSUPPORTED" not in verified


def test_diversify_chunks_skips_near_duplicate_context():
    chunks = [
        {
            "chunk_id": "doc_0",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
        {
            "chunk_id": "doc_1",
            "text": "The crisis was caused by the bursting of the United States housing bubble.",
        },
        {
            "chunk_id": "doc_2",
            "text": "Dodd-Frank introduced new financial regulations after the crisis.",
        },
    ]

    diversified = ResearchOrchestrator._diversify_chunks(chunks, max_chunks=3)

    assert [chunk["chunk_id"] for chunk in diversified] == ["doc_0", "doc_2"]
