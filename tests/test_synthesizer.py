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


def test_synthesizer_deduplicates_lightly_rephrased_crisis_claims():
    text = (
        "The value of MBS and CDOs plummeted, causing massive losses for financial institutions globally "
        "(source: doc_0). "
        "The decline in mortgage payments caused the value of MBS and CDOs to plummet, producing massive "
        "losses for financial institutions globally (source: doc_1). "
        "The collapse of Lehman Brothers marked the climax of the crisis (source: doc_2)."
    )

    deduplicated = SynthesizerAgent._deduplicate_sentences(text)

    assert deduplicated.count("MBS and CDOs") == 1
    assert "Lehman Brothers" in deduplicated
