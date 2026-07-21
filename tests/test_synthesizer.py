from src.agent.synthesizer import SynthesizerAgent


def test_synthesizer_deduplicates_repeated_claims_with_different_citations():
    text = (
        "The crisis was caused by the bursting of the United States housing bubble (source: doc_0). "
        "The United States housing bubble caused the crisis (source: doc_1). "
        "Dodd-Frank introduced new financial regulations after the crisis (source: doc_2)."
    )

    deduplicated = SynthesizerAgent._deduplicate_sentences(text)

    assert deduplicated.count("housing bubble") == 1
    assert "Dodd-Frank" in deduplicated
