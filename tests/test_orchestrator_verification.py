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


