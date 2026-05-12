from __future__ import annotations

from consciousness.dream_artifacts import ArtifactBuffer, ReflectiveValidator, create_artifact


def test_informational_artifact_with_sources_is_held() -> None:
    validator = ReflectiveValidator(ArtifactBuffer(), remember_fn=lambda *_args, **_kwargs: None)
    artifact = create_artifact(
        artifact_type="waking_question",
        source_memory_ids=["mem_a", "mem_b"],
        content="What memory pattern should I revisit while awake?",
        confidence=0.30,
        cluster_coherence=0.25,
    )

    outcome = validator._evaluate(artifact)

    assert outcome.state == "held"
    assert "informational artifact" in outcome.notes


def test_bridge_candidate_with_sources_is_not_discarded_for_missing_provenance() -> None:
    validator = ReflectiveValidator(ArtifactBuffer(), remember_fn=lambda *_args, **_kwargs: None)
    artifact = create_artifact(
        artifact_type="bridge_candidate",
        source_memory_ids=["mem_a", "mem_b", "mem_c"],
        content="Connection: user stress rises after repeated interruption topics.",
        confidence=0.45,
        cluster_coherence=0.55,
    )

    outcome = validator._evaluate(artifact)

    assert outcome.state in {"held", "promoted"}
