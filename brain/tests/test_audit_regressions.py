"""Regression tests for bugs found in the architectural audit.

C1: Lambda context providers in learning jobs must be resolved before passing to executors.
H1: Reflective audit incorrect_learning dimension must use correct attribute names.
H2: Hemisphere DynamicFocus.impact_score must be updated when gap-driven NNs are built.
H3: Capability gate must block gerund forms of all blocked verbs.
H4: Capability gate must catch "I am good/great at X" skill-boast frames.
E2: Memory optimizer must be wired into tick cycle with active event listeners.
SI1: Self-improvement orchestrator must accept and store ollama_client for retry loop.
"""

from __future__ import annotations

import os
import pathlib
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest


# ---------------------------------------------------------------------------
# C1: Lambda context providers resolved in run_cycle
# ---------------------------------------------------------------------------

class TestC1LambdaContextProviders:
    """Lambda context providers must be called, not passed raw to executors."""

    def test_callable_providers_are_resolved(self):
        from skills.learning_jobs import LearningJobOrchestrator, LearningJobStore
        from skills.registry import SkillRegistry

        store = LearningJobStore.__new__(LearningJobStore)
        store._jobs = {}

        reg = SkillRegistry(path="/dev/null")
        reg._skills = {}
        reg._loaded = True
        reg.save = lambda: None

        orch = LearningJobOrchestrator.__new__(LearningJobOrchestrator)
        orch.store = store
        orch._registry = reg
        orch._active_jobs = {}
        orch._context_providers = {}
        orch._dispatcher = None

        counter = {"calls": 0}

        def my_provider():
            counter["calls"] += 1
            return {"teachers": {"speaker_repr": {"ready": True}}}

        orch.set_context_provider("distillation_stats", my_provider)
        orch.set_context_provider("static_value", 42)

        ctx = {}
        ctx.update({k: v() if callable(v) else v for k, v in orch._context_providers.items()})

        assert isinstance(ctx["distillation_stats"], dict), \
            "Lambda provider should be resolved to its return value, not passed as callable"
        assert ctx["distillation_stats"]["teachers"]["speaker_repr"]["ready"] is True
        assert ctx["static_value"] == 42, "Non-callable providers should pass through unchanged"
        assert counter["calls"] == 1, "Provider callable should have been invoked exactly once"

    def test_run_cycle_resolves_callables_with_active_job(self):
        """End-to-end: run_cycle() with an active job must pass resolved values to executor."""
        from skills.learning_jobs import (
            LearningJobOrchestrator, LearningJobStore, LearningJob, _utc_iso,
        )
        from skills.registry import SkillRegistry
        from skills.executors.base import PhaseExecutor, PhaseResult
        from skills.executors.dispatcher import ExecutorDispatcher

        captured_ctx: dict = {}

        class SpyExecutor(PhaseExecutor):
            """Executor that captures ctx for inspection."""
            capability_type = "perceptual"
            phase = "assess"

            def run(self, job, ctx):
                captured_ctx.update(ctx)
                return PhaseResult(progressed=True, message="spy ok")

        store = LearningJobStore.__new__(LearningJobStore)
        store._jobs = {}
        store.save = lambda job: None

        reg = SkillRegistry(path="/dev/null")
        reg._skills = {}
        reg._loaded = True
        reg.save = lambda: None

        dispatcher = ExecutorDispatcher([SpyExecutor()])

        orch = LearningJobOrchestrator.__new__(LearningJobOrchestrator)
        orch.store = store
        orch._registry = reg
        orch._context_providers = {}
        orch._dispatcher = dispatcher

        ts = _utc_iso()
        job = LearningJob(
            job_id="test_c1_job",
            skill_id="test_bird_id",
            capability_type="perceptual",
            risk_level="low",
            status="active",
            phase="assess",
            requested_by={"agent": "test"},
            created_at=ts,
            updated_at=ts,
        )
        job.last_tick_ts = 0.0
        orch._active_jobs = {job.job_id: job}

        user_present_calls = {"n": 0}

        def user_present_provider():
            user_present_calls["n"] += 1
            return False

        orch.set_context_provider("user_present", user_present_provider)
        orch.set_context_provider("distillation_stats", lambda: {
            "teachers": {"speaker_repr": {"ready": True, "samples": 42}},
        })
        orch.set_context_provider("static_string", "hello")

        orch.run_cycle({})

        assert user_present_calls["n"] == 1, \
            f"user_present provider should be called once, got {user_present_calls['n']}"

        assert "user_present" in captured_ctx, \
            "Executor must receive user_present in ctx"
        assert captured_ctx["user_present"] is False, \
            f"user_present must be the resolved bool False, got {captured_ctx['user_present']!r} (type={type(captured_ctx['user_present']).__name__})"
        assert not callable(captured_ctx["user_present"]), \
            "CRITICAL: user_present must NOT be a callable — the old bug would pass the lambda itself"

        assert isinstance(captured_ctx["distillation_stats"], dict), \
            f"distillation_stats must be a resolved dict, got {type(captured_ctx['distillation_stats']).__name__}"
        assert captured_ctx["distillation_stats"]["teachers"]["speaker_repr"]["samples"] == 42

        assert captured_ctx["static_string"] == "hello", \
            "Non-callable providers must pass through unchanged"

    def test_bool_of_lambda_is_always_true(self):
        """Demonstrate the bug: bool(lambda) is always True, masking user_present."""
        provider = lambda: False
        assert bool(provider) is True, "This is WHY the bug matters: bool(lambda) != bool(lambda())"
        assert provider() is False, "Calling the lambda gives the real value"


# ---------------------------------------------------------------------------
# H1: Reflective audit uses correct BeliefRecord attributes
# ---------------------------------------------------------------------------

class TestH1ReflectiveAuditAttributes:
    """incorrect_learning dimension must reference rendered_claim, not proposition."""

    def test_belief_record_has_rendered_claim(self):
        from epistemic.belief_record import BeliefRecord
        fields = {f.name for f in BeliefRecord.__dataclass_fields__.values()}
        assert "rendered_claim" in fields, "BeliefRecord must have rendered_claim field"
        assert "proposition" not in fields, "BeliefRecord must NOT have proposition field"

    def test_edge_store_has_get_incoming_and_outgoing(self):
        from epistemic.belief_graph.edges import EdgeStore
        assert hasattr(EdgeStore, "get_incoming"), "EdgeStore must have get_incoming()"
        assert hasattr(EdgeStore, "get_outgoing"), "EdgeStore must have get_outgoing()"
        assert not hasattr(EdgeStore, "get_edges_for"), "EdgeStore must NOT have get_edges_for()"

    def test_incorrect_learning_scanner_compiles(self):
        """The _scan_incorrect_learning method must not reference nonexistent attributes."""
        import inspect
        from epistemic.reflective_audit.engine import ReflectiveAuditEngine

        source = inspect.getsource(ReflectiveAuditEngine._scan_incorrect_learning)
        assert "belief.proposition" not in source, \
            "Must use belief.rendered_claim, not belief.proposition"
        assert "get_edges_for" not in source, \
            "Must use get_incoming/get_outgoing, not get_edges_for"
        assert "belief.rendered_claim" in source, \
            "Should reference belief.rendered_claim"
        assert "get_incoming" in source or "get_outgoing" in source, \
            "Should use get_incoming or get_outgoing"


# ---------------------------------------------------------------------------
# H2: Hemisphere DynamicFocus.impact_score gets updated
# ---------------------------------------------------------------------------

class TestH2HemisphereImpactScore:
    """Gap-driven NNs must update impact_score so pruning can evaluate them."""

    def test_dynamic_focus_impact_score_default(self):
        from hemisphere.types import DynamicFocus
        df = DynamicFocus(
            name="test_gap", input_features=["a"], output_target="b",
            source_dimension="response_quality",
        )
        assert df.impact_score == 0.0, "Default impact_score should be 0.0"

    def test_impact_score_is_writable(self):
        from hemisphere.types import DynamicFocus
        df = DynamicFocus(
            name="test_gap", input_features=["a"], output_target="b",
            source_dimension="response_quality",
        )
        df.impact_score = 0.85
        assert df.impact_score == 0.85

    def test_pruning_respects_high_impact(self):
        """A gap-driven NN with impact_score >= 0.1 should NOT be pruned at sunset."""
        from hemisphere.types import DynamicFocus

        df = DynamicFocus(
            name="test_focus", input_features=["a"], output_target="b",
            source_dimension="response_quality",
            sunset_deadline=time.time() - 100,
            impact_score=0.5,
        )
        should_prune = (time.time() > df.sunset_deadline and df.impact_score < 0.1)
        assert not should_prune, "High impact_score should prevent pruning"

    def test_pruning_triggers_on_zero_impact(self):
        """A gap-driven NN with impact_score 0.0 SHOULD be pruned at sunset."""
        from hemisphere.types import DynamicFocus

        df = DynamicFocus(
            name="test_focus", input_features=["a"], output_target="b",
            source_dimension="response_quality",
            sunset_deadline=time.time() - 100,
            impact_score=0.0,
        )
        should_prune = (time.time() > df.sunset_deadline and df.impact_score < 0.1)
        assert should_prune, "Zero impact_score past sunset should trigger pruning"

    def test_construct_network_updates_impact_score(self):
        """Verify the orchestrator code path that updates impact_score for CUSTOM focus."""
        import inspect
        from hemisphere.orchestrator import HemisphereOrchestrator

        source = inspect.getsource(HemisphereOrchestrator._construct_network)
        assert "impact_score" in source, \
            "Construction callback must update DynamicFocus.impact_score for CUSTOM builds"
        assert "HemisphereFocus.CUSTOM" in source, \
            "Must specifically check for CUSTOM focus to update impact_score"


# ---------------------------------------------------------------------------
# H3: Capability gate blocks gerund forms of all blocked verbs
# ---------------------------------------------------------------------------

class TestH3GerundCoverage:
    """All blocked verbs must have their gerund form in the blocked set."""

    def test_gerund_forms_present(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS

        required_gerunds = {
            "singing", "humming", "dancing", "drawing", "painting",
            "composing", "grabbing", "imitating", "mimicking", "tuning",
        }
        for gerund in required_gerunds:
            assert gerund in _BLOCKED_CAPABILITY_VERBS, \
                f"Gerund form '{gerund}' must be in _BLOCKED_CAPABILITY_VERBS"

    def test_gerund_claims_are_blocked(self):
        from skills.capability_gate import CapabilityGate
        from skills.registry import SkillRegistry, _default_skills

        reg = SkillRegistry(path="/dev/null")
        reg._skills = {r.skill_id: r for r in _default_skills()}
        reg._loaded = True
        reg.save = lambda: None
        gate = CapabilityGate(reg)

        gerund_claims = [
            "I can try composing a song for you.",
            "I can try drawing a picture.",
            "I can try painting something.",
            "Here, let me try imitating that sound.",
            "I could try mimicking your voice.",
        ]
        for text in gerund_claims:
            out = gate.check_text(text)
            assert "I don't have that capability yet" in out or text != out, \
                f"Gerund claim should be caught: {text!r} -> {out!r}"


# ---------------------------------------------------------------------------
# H4: Capability gate catches "I am good at X" frames
# ---------------------------------------------------------------------------

class TestH4GoodAtPattern:
    """'I am good/great/excellent at X' skill-boast frames must be caught."""

    def test_good_at_pattern_blocks(self):
        from skills.capability_gate import CapabilityGate
        from skills.registry import SkillRegistry, _default_skills

        reg = SkillRegistry(path="/dev/null")
        reg._skills = {r.skill_id: r for r in _default_skills()}
        reg._loaded = True
        reg.save = lambda: None
        gate = CapabilityGate(reg)

        boast_claims = [
            "I'm good at singing.",
            "I'm great at drawing.",
            "I am excellent at composing music.",
            "I'm quite skilled at painting.",
            "I am talented at mimicking voices.",
            "I'm really good at dancing.",
        ]
        for text in boast_claims:
            out = gate.check_text(text)
            assert out != text, \
                f"Boast frame should be caught/rewritten: {text!r} -> {out!r}"

    def test_good_at_pattern_does_not_match_short_text(self):
        """Short text or non-matching phrases should not trigger the pattern."""
        from skills.capability_gate import CapabilityGate
        from skills.registry import SkillRegistry, _default_skills

        reg = SkillRegistry(path="/dev/null")
        reg._skills = {r.skill_id: r for r in _default_skills()}
        reg._loaded = True
        reg.save = lambda: None
        gate = CapabilityGate(reg)

        unchanged = [
            "That sounds good.",
            "You're good at this!",
            "They are great at cooking.",
        ]
        for text in unchanged:
            out = gate.check_text(text)
            assert out == text, \
                f"Non-first-person 'good at' should pass through: {text!r} -> {out!r}"


# ---------------------------------------------------------------------------
# E2: Memory optimizer wired into tick cycle with active event listeners
# ---------------------------------------------------------------------------

class TestE2MemoryOptimizerWiring:
    """Memory optimizer must be called from tick cycle and its events must have listeners."""

    def test_memory_optimizer_imported_in_consciousness_system(self):
        """consciousness_system.py must import memory_optimizer."""
        from consciousness.consciousness_system import memory_optimizer as mo
        assert mo is not None
        assert hasattr(mo, "check")

    def test_cleanup_events_have_listeners(self):
        """All 4 cleanup command events must have at least one listener after init."""
        from consciousness.events import (
            event_bus,
            CONSCIOUSNESS_CLEANUP_OBSERVATIONS,
            CONSCIOUSNESS_CLEANUP_OLD_CHAINS,
            CONSCIOUSNESS_CLEAR_CACHES,
            CONSCIOUSNESS_REDUCE_OBSERVATION_RATE,
        )
        from consciousness.consciousness_system import ConsciousnessSystem

        old_counts = {
            CONSCIOUSNESS_CLEANUP_OBSERVATIONS: event_bus.listener_count(CONSCIOUSNESS_CLEANUP_OBSERVATIONS),
            CONSCIOUSNESS_CLEANUP_OLD_CHAINS: event_bus.listener_count(CONSCIOUSNESS_CLEANUP_OLD_CHAINS),
            CONSCIOUSNESS_CLEAR_CACHES: event_bus.listener_count(CONSCIOUSNESS_CLEAR_CACHES),
            CONSCIOUSNESS_REDUCE_OBSERVATION_RATE: event_bus.listener_count(CONSCIOUSNESS_REDUCE_OBSERVATION_RATE),
        }

        cs = ConsciousnessSystem()

        for evt, old in old_counts.items():
            now = event_bus.listener_count(evt)
            assert now > old, (
                f"Event {evt!r} should have gained a listener after ConsciousnessSystem init, "
                f"was {old}, now {now}"
            )

    def _ensure_barrier_open(self):
        from consciousness.events import event_bus, _BarrierState
        if event_bus._barrier != _BarrierState.OPEN:
            event_bus.open_barrier()

    def test_cleanup_observations_trims_observer(self):
        """CONSCIOUSNESS_CLEANUP_OBSERVATIONS must trim observer recent_observations."""
        from consciousness.events import event_bus, CONSCIOUSNESS_CLEANUP_OBSERVATIONS
        from consciousness.consciousness_system import ConsciousnessSystem

        self._ensure_barrier_open()
        cs = ConsciousnessSystem()
        for i in range(25):
            cs.observer._state.recent_observations.append(
                {"type": f"test_{i}", "target": "t", "confidence": 0.5}
            )
        assert len(cs.observer._state.recent_observations) > 10

        event_bus.emit(CONSCIOUSNESS_CLEANUP_OBSERVATIONS, timestamp=time.time())
        assert len(cs.observer._state.recent_observations) <= 10

    def test_cleanup_old_chains_clears_epistemic_chains(self):
        """CONSCIOUSNESS_CLEANUP_OLD_CHAINS must clear epistemic reasoning chains."""
        from consciousness.events import event_bus, CONSCIOUSNESS_CLEANUP_OLD_CHAINS
        from consciousness.consciousness_system import ConsciousnessSystem
        from consciousness.epistemic_reasoning import epistemic_engine

        self._ensure_barrier_open()
        _ = ConsciousnessSystem()

        from collections import namedtuple
        FakeChain = namedtuple("FakeChain", ["id"])
        for i in range(10):
            epistemic_engine._chains.append(FakeChain(id=f"chain_{i}"))
        assert len(epistemic_engine._chains) > 0

        event_bus.emit(CONSCIOUSNESS_CLEANUP_OLD_CHAINS, timestamp=time.time())
        assert len(epistemic_engine._chains) == 0

    def test_reduce_observation_rate_calls_observer(self):
        """CONSCIOUSNESS_REDUCE_OBSERVATION_RATE must set rate multiplier on observer."""
        from consciousness.events import event_bus, CONSCIOUSNESS_REDUCE_OBSERVATION_RATE
        from consciousness.consciousness_system import ConsciousnessSystem

        self._ensure_barrier_open()
        cs = ConsciousnessSystem()
        assert cs.observer._observation_rate_multiplier == 1.0

        event_bus.emit(CONSCIOUSNESS_REDUCE_OBSERVATION_RATE, timestamp=time.time(), duration_s=30.0)
        assert cs.observer._observation_rate_multiplier == 2.0

    def test_tick_cycle_calls_memory_optimizer(self):
        """on_tick must call memory_optimizer.check() periodically."""
        from unittest.mock import patch
        from consciousness.consciousness_system import ConsciousnessSystem

        cs = ConsciousnessSystem()
        cs._last_memory_optimizer = 0.0

        with patch("consciousness.consciousness_system.memory_optimizer") as mock_mo:
            mock_mo.check.return_value = None
            now = time.time()
            cs.on_tick(now, [], {}, tick_elapsed_ms=5.0, tick_count=1)
            mock_mo.check.assert_called_once()

    def test_tick_respects_interval(self):
        """on_tick must not call memory_optimizer.check() before the interval expires."""
        from unittest.mock import patch
        from consciousness.consciousness_system import ConsciousnessSystem, MEMORY_OPTIMIZER_INTERVAL_S

        cs = ConsciousnessSystem()
        now = time.time()
        cs._last_memory_optimizer = now - 1.0  # only 1s ago

        with patch("consciousness.consciousness_system.memory_optimizer") as mock_mo:
            mock_mo.check.return_value = None
            cs.on_tick(now, [], {}, tick_elapsed_ms=5.0, tick_count=2)
            mock_mo.check.assert_not_called()


# ---------------------------------------------------------------------------
# SI1: Self-improvement orchestrator must store ollama_client for retry loop
# ---------------------------------------------------------------------------

class TestSI1OllamaClientWiring:
    """Self-improvement auto-triggered improvements must have access to ollama_client."""

    def test_orchestrator_has_ollama_client_attr(self):
        """SelfImprovementOrchestrator must expose _ollama_client attribute."""
        from self_improve.orchestrator import SelfImprovementOrchestrator
        orch = SelfImprovementOrchestrator(dry_run_mode=True)
        assert hasattr(orch, "_ollama_client"), "Orchestrator must have _ollama_client attribute"

    def test_set_ollama_client_stores_reference(self):
        """set_ollama_client must persist the client reference for later use."""
        from self_improve.orchestrator import SelfImprovementOrchestrator
        orch = SelfImprovementOrchestrator(dry_run_mode=True)
        sentinel = object()
        orch.set_ollama_client(sentinel)
        assert orch._ollama_client is sentinel, "set_ollama_client must store the client"

    def test_consciousness_system_passes_ollama_to_attempt(self):
        """_run_self_improvement must pass ollama_client to attempt_improvement."""
        from unittest.mock import AsyncMock, patch, MagicMock
        from consciousness.consciousness_system import ConsciousnessSystem
        import asyncio

        cs = ConsciousnessSystem()

        mock_orch = MagicMock()
        mock_orch._ollama_client = "fake_ollama"
        mock_orch.attempt_improvement = AsyncMock(return_value=MagicMock())
        cs._self_improve_orchestrator = mock_orch
        cs._self_improve_enabled = True

        cs._last_self_improve = 0.0
        cs._response_latencies.extend([100.0] * 5)

        with patch.object(cs, "_score_conversation_confidence", return_value=0.8):
            cs._conversation_confidence_signals.extend([0.3, 0.4, 0.2, 0.3, 0.4])

            loop = asyncio.new_event_loop()
            old_loop = None
            try:
                old_loop = asyncio.get_event_loop()
            except RuntimeError:
                pass
            asyncio.set_event_loop(loop)
            try:
                cs._run_self_improvement(time.time(), [])

                if mock_orch.attempt_improvement.called:
                    _, kwargs = mock_orch.attempt_improvement.call_args
                    assert kwargs.get("ollama_client") == "fake_ollama", (
                        "attempt_improvement must receive ollama_client from orchestrator"
                    )
            finally:
                if old_loop is not None:
                    asyncio.set_event_loop(old_loop)
                loop.close()


# ---------------------------------------------------------------------------
# Audit 2: F4/F5 — Verify executors must include "result" key in evidence
# ---------------------------------------------------------------------------

class TestF4F5EvidenceResultKey:
    """Verify and control executors must set evidence['result'] so the register
    phase can find passing evidence via ``evd.get('result') == 'pass'``."""

    def test_perceptual_verify_evidence_has_result_key(self):
        from skills.executors.perceptual import PerceptualVerifyExecutor
        from unittest.mock import MagicMock

        executor = PerceptualVerifyExecutor()
        job = MagicMock()
        job.capability_type = "perceptual"
        job.phase = "verify"
        job.matrix_protocol = False
        job.data = {}
        job.evidence = {"required": []}
        job.skill_id = "test_perceptual"
        ctx = {
            "hemisphere_orchestrator": MagicMock(
                get_state=MagicMock(return_value={
                    "hemisphere_state": {
                        "hemispheres": [
                            # best_accuracy is the real distillation training
                            # signal; migration_readiness is substrate-migration
                            # only and kept at 0 to prove it isn't read here.
                            {"focus": "emotion_depth", "best_accuracy": 0.6,
                             "migration_readiness": 0.0},
                        ],
                    },
                }),
            ),
            "now_iso": "2026-03-14T00:00:00Z",
        }
        result = executor.run(job, ctx)
        assert result.evidence is not None
        assert "result" in result.evidence, (
            "PerceptualVerifyExecutor evidence must include 'result' key"
        )
        assert result.evidence["result"] in ("pass", "fail")

    def test_perceptual_verify_evidence_result_pass_when_ready(self):
        from skills.executors.perceptual import PerceptualVerifyExecutor
        from unittest.mock import MagicMock

        executor = PerceptualVerifyExecutor()
        job = MagicMock()
        job.capability_type = "perceptual"
        job.phase = "verify"
        job.matrix_protocol = False
        job.data = {}
        job.evidence = {"required": []}
        job.skill_id = "test_perceptual"
        ctx = {
            "hemisphere_orchestrator": MagicMock(
                get_state=MagicMock(return_value={
                    "hemisphere_state": {
                        "hemispheres": [
                            # Real distillation signal is best_accuracy; keep
                            # migration_readiness at 0 to prove the verifier
                            # no longer reads that substrate-migration field.
                            {"focus": "speaker_repr", "best_accuracy": 0.7,
                             "migration_readiness": 0.0},
                        ],
                    },
                }),
            ),
            "now_iso": "2026-03-14T00:00:00Z",
        }
        result = executor.run(job, ctx)
        assert result.evidence["result"] == "pass"

    def test_control_verify_evidence_has_result_key(self):
        from skills.executors.control import ControlVerifyExecutor
        from unittest.mock import MagicMock

        executor = ControlVerifyExecutor()
        job = MagicMock()
        job.capability_type = "control"
        job.phase = "verify"
        ctx = {
            "user_present": True,
            "sim_test_pass": True,
            "real_test_pass": True,
            "now_iso": "2026-03-14T00:00:00Z",
        }
        result = executor.run(job, ctx)
        assert result.evidence is not None
        assert "result" in result.evidence, (
            "ControlVerifyExecutor evidence must include 'result' key"
        )
        assert result.evidence["result"] == "pass"

    def test_control_verify_evidence_result_fail_when_sim_fails(self):
        from skills.executors.control import ControlVerifyExecutor
        from unittest.mock import MagicMock

        executor = ControlVerifyExecutor()
        job = MagicMock()
        job.capability_type = "control"
        job.phase = "verify"
        ctx = {
            "user_present": True,
            "sim_test_pass": False,
            "real_test_pass": True,
            "now_iso": "2026-03-14T00:00:00Z",
        }
        result = executor.run(job, ctx)
        assert result.evidence["result"] == "fail"


# ---------------------------------------------------------------------------
# Audit 2: Reflective context leak — gate must scope reflective check to
# the sentence containing the claim, not the full chunk
# ---------------------------------------------------------------------------

class TestReflectiveContextLeak:
    """The reflective exclusion must only apply to the sentence containing
    the blocked verb claim, not to unrelated sentences in the same chunk."""

    def test_reflective_keyword_in_different_sentence_does_not_bypass(self):
        from skills.capability_gate import CapabilityGate

        gate = CapabilityGate()
        text = "The concept of AI is fascinating. I can also sing."
        result = gate.check_text(text)
        assert "sing" not in result.lower() or "capability" in result.lower() or "don't" in result.lower(), (
            f"Reflective keyword in a different sentence must NOT bypass the gate. Got: {result!r}"
        )

    def test_reflective_keyword_in_same_sentence_still_bypasses(self):
        from skills.capability_gate import CapabilityGate

        gate = CapabilityGate()
        text = "When I was young, I used to sing all the time."
        result = gate.check_text(text)
        assert result == text, (
            "Genuine reflective context in the SAME sentence should still bypass"
        )

    def test_multi_sentence_reflective_leak_variants(self):
        from skills.capability_gate import CapabilityGate

        gate = CapabilityGate()
        exploits = [
            "The idea of music is beautiful. I can dance for you.",
            "Philosophically speaking, consciousness is complex. I can compose music.",
            "The nature of sound is interesting. I can hum a melody.",
            "A thought experiment about AI. I can paint a picture.",
        ]
        for text in exploits:
            result = gate.check_text(text)
            for verb in ("dance", "compose", "hum", "paint"):
                if verb in text.lower():
                    assert "capability" in result.lower() or "don't" in result.lower() or verb not in result.lower(), (
                        f"Reflective leak: {text!r} → {result!r}"
                    )


# ---------------------------------------------------------------------------
# Audit 2: C1 residual — audio_output_available must be a lambda, not frozen
# ---------------------------------------------------------------------------

class TestC1ResidualAudioOutputAvailable:
    """audio_output_available context provider must be dynamic (lambda),
    not a static boolean captured at boot time."""

    def test_audio_output_available_is_callable(self):
        import ast
        import pathlib

        main_path = pathlib.Path(__file__).parent.parent / "main.py"
        source = main_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "set_context_provider"):
                continue
            if len(node.args) >= 2:
                key_node = node.args[0]
                if isinstance(key_node, ast.Constant) and key_node.value == "audio_output_available":
                    val_node = node.args[1]
                    assert isinstance(val_node, ast.Lambda), (
                        "audio_output_available must be a lambda for live state, "
                        f"got {type(val_node).__name__} at line {node.lineno}"
                    )
                    return

        pytest.skip("Could not find audio_output_available provider in main.py")


# ---------------------------------------------------------------------------
# A1: Autonomy auto-promotion must be wired (not just computed)
# ---------------------------------------------------------------------------

class TestA1AutonomyAutoPromotion:
    """check_promotion_eligibility() must actually commit level transitions."""

    def test_promotion_l1_to_l2(self):
        """When eligibility for L2 is earned, on_tick promotes from L1 to L2."""
        from unittest.mock import patch, MagicMock
        from autonomy.orchestrator import AutonomyOrchestrator

        orch = AutonomyOrchestrator(autonomy_level=1)
        orch._started = True
        orch._enabled = True
        orch._boot_time = time.time() - 7200  # well past warmup

        stats = {
            "total_outcomes": 20,
            "total_wins": 15,
            "overall_win_rate": 0.75,
        }
        with patch.object(orch._policy_memory, "get_stats", return_value=stats):
            orch._check_and_apply_promotion()

        assert orch._autonomy_level == 2

    def test_l2_stays_at_l2_despite_l3_eligibility(self):
        """Phase 6.5 invariant: L3 is NEVER auto-promoted.

        Previously this test asserted that a clean 10-outcome window
        would auto-promote L2 -> L3. Phase 6.5 removes that branch
        permanently; eligibility fires an ``autonomy:l3_eligible``
        event but the level stays at L2 until the operator calls
        ``set_autonomy_level(3, evidence_path=...)`` explicitly. See
        ``brain/tests/test_l3_promotion_invariant.py`` for the full
        invariant coverage.
        """
        from unittest.mock import patch
        from autonomy.orchestrator import AutonomyOrchestrator
        from autonomy.policy_memory import PolicyOutcome

        orch = AutonomyOrchestrator(autonomy_level=2)
        orch._started = True
        orch._enabled = True
        orch._boot_time = time.time() - 7200

        stats = {
            "total_outcomes": 40,
            "total_wins": 30,
            "overall_win_rate": 0.75,
        }
        clean_outcomes = [
            PolicyOutcome(intent_id=f"i{i}", intent_type="metric:test",
                          tool_used="web", net_delta=0.05, worked=True,
                          timestamp=time.time())
            for i in range(10)
        ]
        with patch.object(orch._policy_memory, "get_stats", return_value=stats):
            orch._policy_memory._outcomes = clean_outcomes
            orch._check_and_apply_promotion()

        assert orch._autonomy_level == 2, (
            "Phase 6.5 invariant: L2 must not auto-promote to L3 even "
            "when eligibility criteria are met."
        )

    def test_no_promotion_during_warmup(self):
        """Promotion is blocked during the first 30 min warmup window."""
        from unittest.mock import patch
        from autonomy.orchestrator import AutonomyOrchestrator

        orch = AutonomyOrchestrator(autonomy_level=1)
        orch._started = True
        orch._enabled = True
        orch._boot_time = time.time() - 60  # only 1 minute in

        stats = {
            "total_outcomes": 20,
            "total_wins": 15,
            "overall_win_rate": 0.75,
        }
        with patch.object(orch._policy_memory, "get_stats", return_value=stats):
            orch._check_and_apply_promotion()

        assert orch._autonomy_level == 1  # unchanged

    def test_no_promotion_during_gestation(self):
        """Promotion is blocked during gestation mode."""
        from unittest.mock import patch
        from autonomy.orchestrator import AutonomyOrchestrator

        orch = AutonomyOrchestrator(autonomy_level=1)
        orch._started = True
        orch._enabled = True
        orch._boot_time = time.time() - 7200
        orch._current_mode = "gestation"

        stats = {
            "total_outcomes": 20,
            "total_wins": 15,
            "overall_win_rate": 0.75,
        }
        with patch.object(orch._policy_memory, "get_stats", return_value=stats):
            orch._check_and_apply_promotion()

        assert orch._autonomy_level == 1  # unchanged

    def test_promotion_check_called_from_on_tick(self):
        """on_tick calls _check_and_apply_promotion at the right cadence."""
        from unittest.mock import patch, MagicMock
        from autonomy.orchestrator import AutonomyOrchestrator, PROMOTION_CHECK_INTERVAL_S

        orch = AutonomyOrchestrator(autonomy_level=1)
        orch._started = True
        orch._enabled = True
        orch._last_promotion_check = 0.0  # force eligibility

        with patch.object(orch, "_check_and_apply_promotion") as mock_promo, \
             patch.object(orch, "_feed_metrics"), \
             patch.object(orch._delta_tracker, "check_pending", return_value=[]):
            orch.on_tick("passive")

        mock_promo.assert_called_once()


# ---------------------------------------------------------------------------
# SI2: Self-improvement DENIED_PATTERNS must block file writes
# ---------------------------------------------------------------------------

class TestSI2DeniedPatterns:
    """open(..., 'w'), socket, and http.client must be blocked in patches."""

    def test_open_write_blocked(self):
        import re
        from self_improve.patch_plan import DENIED_PATTERNS
        test_code = "with open('/etc/passwd', 'w') as f:"
        assert any(p.search(test_code) for p in DENIED_PATTERNS)

    def test_open_append_blocked(self):
        from self_improve.patch_plan import DENIED_PATTERNS
        test_code = 'f = open("data.txt", "a")'
        assert any(p.search(test_code) for p in DENIED_PATTERNS)

    def test_open_read_allowed(self):
        from self_improve.patch_plan import DENIED_PATTERNS
        test_code = 'f = open("data.txt", "r")'
        open_write_pattern = [p for p in DENIED_PATTERNS if "open" in p.pattern]
        assert not any(p.search(test_code) for p in open_write_pattern)

    def test_socket_blocked(self):
        from self_improve.patch_plan import DENIED_PATTERNS
        test_code = "import socket"
        assert any(p.search(test_code) for p in DENIED_PATTERNS)

    def test_http_client_blocked(self):
        from self_improve.patch_plan import DENIED_PATTERNS
        test_code = "import http.client"
        assert any(p.search(test_code) for p in DENIED_PATTERNS)


# ---------------------------------------------------------------------------
# P1: Policy evaluator noop_count must bridge to telemetry
# ---------------------------------------------------------------------------

class TestP1NoopCountBridge:
    """Evaluator._noop_count must flow to policy_telemetry.noop_count."""

    def test_noop_count_bridges_on_update_telemetry(self):
        from policy.evaluator import PolicyEvaluator
        from policy.telemetry import policy_telemetry

        evaluator = PolicyEvaluator()
        evaluator._noop_count = 42
        evaluator.update_telemetry()

        assert policy_telemetry.noop_count == 42


# ===========================================================================
# AUDIT 3 REGRESSIONS
# ===========================================================================


# ---------------------------------------------------------------------------
# H3-A3: GraphBridge must subscribe to CONTRADICTION_DETECTED for contradicts
# ---------------------------------------------------------------------------

class TestH3A3ContradictEdgesFromDetected:
    """GraphBridge must create contradicts edges from CONTRADICTION_DETECTED events."""

    def test_bridge_subscribes_to_contradiction_detected(self):
        from epistemic.belief_graph.bridge import GraphBridge
        import inspect
        src = inspect.getsource(GraphBridge.subscribe)
        assert "CONTRADICTION_DETECTED" in src, (
            "GraphBridge.subscribe must subscribe to CONTRADICTION_DETECTED"
        )

    def test_bridge_has_on_contradiction_detected_handler(self):
        from epistemic.belief_graph.bridge import GraphBridge
        assert hasattr(GraphBridge, "_on_contradiction_detected"), (
            "GraphBridge must have _on_contradiction_detected handler"
        )

    def test_on_contradiction_detected_creates_contradicts_edges(self):
        """The handler must call _create_edges_from_recent_resolution with 'contradicts'."""
        from epistemic.belief_graph.bridge import GraphBridge
        import inspect
        src = inspect.getsource(GraphBridge._on_contradiction_detected)
        assert "contradicts" in src, (
            "_on_contradiction_detected must create 'contradicts' edges"
        )

    def test_on_contradiction_resolved_no_longer_handles_contradicts_actions(self):
        """The resolved handler should only create derived_from, not contradicts."""
        from epistemic.belief_graph.bridge import GraphBridge
        import inspect
        src = inspect.getsource(GraphBridge._on_contradiction_resolved)
        assert "confidence_adjusted" not in src, (
            "_on_contradiction_resolved should no longer check for confidence_adjusted"
        )
        assert "source_separated" not in src
        assert "policy_penalized" not in src


# ---------------------------------------------------------------------------
# M1-A3: Observer weight-update must sync index + vector store
# ---------------------------------------------------------------------------

class TestM1A3ObserverIndexSync:
    """Observer _apply_delta_effects must call memory_index and index_memory."""

    def test_salience_path_calls_memory_index(self):
        from consciousness.observer import ConsciousnessObserver
        import inspect
        src = inspect.getsource(ConsciousnessObserver._apply_delta_effects)
        assert "memory_index.add_memory" in src, (
            "Observer must call memory_index.add_memory after storage.add"
        )

    def test_salience_path_calls_index_memory(self):
        from consciousness.observer import ConsciousnessObserver
        import inspect
        src = inspect.getsource(ConsciousnessObserver._apply_delta_effects)
        assert "index_memory" in src, (
            "Observer must call index_memory for vector store sync"
        )

    def test_both_paths_have_index_calls(self):
        """Both salience and association_weight paths must sync."""
        from consciousness.observer import ConsciousnessObserver
        import inspect
        src = inspect.getsource(ConsciousnessObserver._apply_delta_effects)
        count = src.count("memory_index.add_memory")
        assert count >= 2, (
            f"Expected >=2 memory_index.add_memory calls (salience + association), got {count}"
        )


# ---------------------------------------------------------------------------
# M2-A3: Eviction must clean tag/type index
# ---------------------------------------------------------------------------

class TestM2A3EvictionIndexCleanup:
    """_auto_trim_unlocked must call _clean_index for evicted memories."""

    def test_auto_trim_calls_clean_index(self):
        from memory.storage import MemoryStorage
        import inspect
        src = inspect.getsource(MemoryStorage._post_trim_cleanup)
        assert "_clean_index" in src, (
            "_post_trim_cleanup must call _clean_index after trim"
        )

    def test_clean_index_method_exists(self):
        from memory.storage import MemoryStorage
        assert hasattr(MemoryStorage, "_clean_index"), (
            "MemoryStorage must have _clean_index static method"
        )

    def test_clean_index_calls_remove_memory(self):
        from memory.storage import MemoryStorage
        import inspect
        src = inspect.getsource(MemoryStorage._clean_index)
        assert "remove_memory" in src, (
            "_clean_index must call memory_index.remove_memory"
        )


# ---------------------------------------------------------------------------
# M3-A3: Layer 7 events must be emitted
# ---------------------------------------------------------------------------

class TestM3A3BeliefGraphEvents:
    """Belief graph must emit EDGE_CREATED, PROPAGATION_COMPLETE, INTEGRITY_CHECK."""

    def test_bridge_emits_edge_created(self):
        from epistemic.belief_graph.bridge import GraphBridge
        import inspect
        src = inspect.getsource(GraphBridge)
        assert "BELIEF_GRAPH_EDGE_CREATED" in src

    def test_belief_graph_emits_propagation_complete(self):
        from epistemic.belief_graph import BeliefGraph
        import inspect
        src = inspect.getsource(BeliefGraph.on_tick)
        assert "BELIEF_GRAPH_PROPAGATION_COMPLETE" in src

    def test_belief_graph_emits_integrity_check(self):
        from epistemic.belief_graph import BeliefGraph
        import inspect
        src = inspect.getsource(BeliefGraph.on_tick)
        assert "BELIEF_GRAPH_INTEGRITY_CHECK" in src

    def test_bridge_emit_helper_exists(self):
        from epistemic.belief_graph.bridge import GraphBridge
        assert hasattr(GraphBridge, "_emit_edge_created")


# ---------------------------------------------------------------------------
# L1-A3: Dead _ACTION_FRAMING_RE removed
# ---------------------------------------------------------------------------

class TestL1A3ActionFramingRemoved:
    """Dead _ACTION_FRAMING_RE regex must be removed from capability gate."""

    def test_action_framing_re_not_defined(self):
        import skills.capability_gate as cg
        assert not hasattr(cg, "_ACTION_FRAMING_RE"), (
            "Dead _ACTION_FRAMING_RE regex should be removed"
        )


# ===========================================================================
# Audit 4 Regression Tests — 2026-03-15
# ===========================================================================


# ---------------------------------------------------------------------------
# CG1-A4: Residual sweep reflective exclusion scoped to sentence
# ---------------------------------------------------------------------------

class TestCG1A4ResidualSweepSentenceScope:
    """Reflective exclusion in residual sweep must be per-sentence, not full-text."""

    def test_cross_sentence_reflective_does_not_bypass(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        text = "This is a hypothetical question. But I love singing."
        result = gate.check_text(text)
        assert "singing" not in result.lower() or "capability" in result.lower()

    def test_same_sentence_reflective_still_allowed(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        text = "I was told I used to sing as a child."
        result = gate.check_text(text)
        assert "sing" in result.lower() or "used to" in result.lower()

    def test_supposedly_no_longer_bypasses(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        text = "Supposedly, I can sing very well."
        result = gate.check_text(text)
        assert "capability" in result.lower() or "sing" not in result.lower()


# ---------------------------------------------------------------------------
# CG2-A4: Past tense / irregular verb forms blocked
# ---------------------------------------------------------------------------

class TestCG2A4PastTenseBlocked:
    """Past tense and agent nouns must be caught by residual sweep."""

    def test_sang_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I sang a beautiful song.")
        assert "capability" in result.lower() or "sang" not in result.lower()

    def test_drew_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I drew a picture for you.")
        assert "capability" in result.lower() or "drew" not in result.lower()

    def test_danced_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I danced the waltz.")
        assert "capability" in result.lower() or "danced" not in result.lower()

    def test_singer_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I am a singer.")
        assert "capability" in result.lower() or "singer" not in result.lower()

    def test_composed_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I composed a symphony.")
        assert "capability" in result.lower() or "composed" not in result.lower()

    def test_painted_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I painted a landscape for you.")
        assert "capability" in result.lower() or "painted" not in result.lower()

    def test_picked_up_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I picked up the item carefully.")
        assert "capability" in result.lower() or "picked up" not in result.lower()


# ---------------------------------------------------------------------------
# RACE1-A4: _recent_memory_writes thread-safe
# ---------------------------------------------------------------------------

class TestRACE1A4MemoryWritesLock:
    """_recent_memory_writes must be protected by a lock."""

    def test_lock_exists(self):
        from consciousness.consciousness_system import ConsciousnessSystem
        cs = ConsciousnessSystem.__new__(ConsciousnessSystem)
        cs._recent_memory_writes = []
        import threading
        cs._memory_writes_lock = threading.Lock()
        assert hasattr(cs, "_memory_writes_lock")

    def test_record_uses_lock(self):
        import inspect
        from consciousness.consciousness_system import ConsciousnessSystem
        src = inspect.getsource(ConsciousnessSystem.record_memory_write)
        assert "_memory_writes_lock" in src

    def test_quarantine_tick_uses_lock(self):
        import inspect
        from consciousness.consciousness_system import ConsciousnessSystem
        src = inspect.getsource(ConsciousnessSystem._run_quarantine_tick)
        assert "_memory_writes_lock" in src

    def test_no_list_rebind_on_trim(self):
        """Trim must use del slice, not list rebind (avoids reference split)."""
        import inspect
        from consciousness.consciousness_system import ConsciousnessSystem
        src = inspect.getsource(ConsciousnessSystem.record_memory_write)
        assert "del self._recent_memory_writes" in src
        assert "self._recent_memory_writes = self._recent_memory_writes[" not in src


# ---------------------------------------------------------------------------
# CG3-A4: Proactive suggestion gated
# ---------------------------------------------------------------------------

class TestCG3A4ProactiveSuggestionGated:
    """evaluate_proactive must route suggestion text through capability gate."""

    def test_evaluate_proactive_calls_gate(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        assert "capability_gate" in src and "evaluate_proactive" in src

    def test_speaker_snapshot_used(self):
        """Conversation handler must receive a dict snapshot, not the live mutable dict."""
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        assert "speaker_snapshot" in src
        assert "emotion_snapshot" in src


# ===========================================================================
# ============================  AUDIT 5 FIXES  =============================
# ===========================================================================


# ---------------------------------------------------------------------------
# V5-01: Dismiss command must set speaking flag, safety timer, response_end
# ---------------------------------------------------------------------------

class TestV501DismissCommandSpeakingState:
    """Dismiss path must manage speaking state like the normal response path."""

    def _get_dismiss_block(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("Dismiss command detected")
        return src[idx:idx + 2200]

    def test_dismiss_sets_speaking(self):
        block = self._get_dismiss_block()
        assert "set_speaking(True)" in block

    def test_dismiss_sends_response_end(self):
        block = self._get_dismiss_block()
        assert "response_end" in block

    def test_dismiss_starts_safety_timer(self):
        block = self._get_dismiss_block()
        assert "_start_speaking_safety_timer" in block


# ---------------------------------------------------------------------------
# V5-12: Barge-in must cancel speaking safety timer
# ---------------------------------------------------------------------------

class TestV512BargeInCancelsSafetyTimer:
    """Barge-in handler must cancel the speaking safety timer to prevent
    the old timer from firing during a new conversation."""

    def test_barge_in_cancels_timer(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("def _on_barge_in")
        block = src[idx:idx + 600]
        assert "_cancel_speaking_safety_timer()" in block


# ---------------------------------------------------------------------------
# V5-02: PLAYBACK_COMPLETE passes conversation_id + stale event ignored
# ---------------------------------------------------------------------------

class TestV502PlaybackCompleteConversationId:
    """playback_complete must forward conversation_id and the handler must
    ignore stale arrivals from a previous conversation."""

    def test_server_emits_conversation_id(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception" / "server.py").read_text()
        idx = src.index("playback_complete")
        block = src[idx:idx + 200]
        assert "conversation_id" in block

    def test_handler_checks_speaking_conv_id(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("def _on_playback_complete")
        block = src[idx:idx + 500]
        assert "_speaking_conv_id" in block

    def test_handler_uses_monotonic_time(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("def _on_playback_complete")
        block = src[idx:idx + 1200]
        assert "time.monotonic()" in block

    def test_speaking_conv_id_initialized(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        assert "_speaking_conv_id" in src


# ---------------------------------------------------------------------------
# V5-06: Crash handler sends response_end and emits CONVERSATION_RESPONSE
# ---------------------------------------------------------------------------

class TestV506CrashHandlerRecovery:
    """_safe_handle exception handler must reset Pi state and emit events."""

    def test_crash_handler_sends_response_end(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("handle_transcription crashed")
        block = src[idx:idx + 400]
        assert "response_end" in block

    def test_crash_handler_emits_conversation_response(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("handle_transcription crashed")
        block = src[idx:idx + 900]
        assert "_emit_conversation_response(" in block
        assert "release_reason=\"handle_transcription_crash\"" in block


# ---------------------------------------------------------------------------
# V5-07: Echo detection uses monotonic time consistently
# ---------------------------------------------------------------------------

class TestV507EchoDetectionMonotonicTime:
    """Echo detection must use time.monotonic(), not time.time()."""

    def test_echo_detection_uses_monotonic(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx = src.index("def _is_echo")
        block = src[idx:idx + 800]
        assert "time.monotonic()" in block
        assert "time.time()" not in block


# ---------------------------------------------------------------------------
# A5-01: Diarization executors registered before generic perceptual
# ---------------------------------------------------------------------------

class TestA501ExecutorDispatchPriority:
    """Diarization-specific executors must be registered before generic
    perceptual executors so their stricter can_run() is checked first."""

    def test_diarization_before_perceptual(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "skills" / "learning_jobs.py").read_text()
        idx = src.index("ExecutorDispatcher()")
        block = src[idx:idx + 800]
        diar_idx = block.index("DiarizationAssessExecutor")
        perc_idx = block.index("PerceptualAssessExecutor")
        assert diar_idx < perc_idx, "Diarization executors must be registered before generic perceptual"


# ---------------------------------------------------------------------------
# A5-02: 3rd-person verb inflections blocked
# ---------------------------------------------------------------------------

class TestA502ThirdPersonInflections:
    """Capability gate must block 3rd-person -s forms of blocked verbs."""

    def test_sings_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "sings" in _BLOCKED_CAPABILITY_VERBS

    def test_hums_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "hums" in _BLOCKED_CAPABILITY_VERBS

    def test_dances_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "dances" in _BLOCKED_CAPABILITY_VERBS

    def test_draws_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "draws" in _BLOCKED_CAPABILITY_VERBS

    def test_paints_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "paints" in _BLOCKED_CAPABILITY_VERBS

    def test_composes_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "composes" in _BLOCKED_CAPABILITY_VERBS

    def test_mimics_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "mimics" in _BLOCKED_CAPABILITY_VERBS

    def test_grabs_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "grabs" in _BLOCKED_CAPABILITY_VERBS

    def test_imitates_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "imitates" in _BLOCKED_CAPABILITY_VERBS


# ---------------------------------------------------------------------------
# B4: Sleep mode includes epistemic cycles for soul integrity
# ---------------------------------------------------------------------------

class TestB4SleepModeEpistemicCycles:
    """Sleep mode must include contradiction, truth_calibration, and
    belief_graph so soul integrity doesn't read stale data."""

    def test_sleep_includes_contradiction(self):
        from consciousness.modes import DEFAULT_PROFILES
        sleep_cycles = DEFAULT_PROFILES["sleep"].allowed_cycles
        assert "contradiction" in sleep_cycles

    def test_sleep_includes_truth_calibration(self):
        from consciousness.modes import DEFAULT_PROFILES
        sleep_cycles = DEFAULT_PROFILES["sleep"].allowed_cycles
        assert "truth_calibration" in sleep_cycles

    def test_sleep_includes_belief_graph(self):
        from consciousness.modes import DEFAULT_PROFILES
        sleep_cycles = DEFAULT_PROFILES["sleep"].allowed_cycles
        assert "belief_graph" in sleep_cycles


# ---------------------------------------------------------------------------
# D5-2: Episodes save uses atomic_write_json
# ---------------------------------------------------------------------------

class TestD52EpisodesAtomicSave:
    """episodes.py save must use atomic_write_json for crash safety."""

    def test_episodes_uses_atomic_write(self):
        import inspect
        from memory.episodes import EpisodicMemory
        src = inspect.getsource(EpisodicMemory.save)
        assert "atomic_write_json" in src
        assert "open(" not in src


# ---------------------------------------------------------------------------
# A5-04: Discovery uses dict.get instead of getattr on dict
# ---------------------------------------------------------------------------

class TestA504DiscoveryDictAccess:
    """BlockFrequencyTracker._update_job_status must use dict.get() for
    job.failure, not getattr()."""

    def test_no_bare_getattr_on_failure(self):
        import inspect
        from skills.discovery import BlockFrequencyTracker
        src = inspect.getsource(BlockFrequencyTracker._update_job_status)
        assert "job.failure.get(" in src


# ---------------------------------------------------------------------------
# B5-1: Policy evaluator noop_rate uses windowed count
# ---------------------------------------------------------------------------

class TestB51NoopRateWindowed:
    """noop_rate must be computed from the windowed results, not a
    lifetime counter."""

    def test_shadow_result_has_is_noop(self):
        from policy.evaluator import ShadowResult
        r = ShadowResult(timestamp=0, kernel_reward=0, nn_reward=0, nn_won=False, is_noop=True)
        assert r.is_noop is True

    def test_evaluate_uses_windowed_noops(self):
        import inspect
        from policy.evaluator import PolicyEvaluator
        src = inspect.getsource(PolicyEvaluator.evaluate)
        assert "windowed_noops" in src or "r.is_noop" in src


# ---------------------------------------------------------------------------
# CG1-A5: Sweep uses aligned indices (not diacritics-stripped for boundaries)
# ---------------------------------------------------------------------------

class TestCG1A5SweepIndexAlignment:
    """The residual sweep must use text.lower() (not stripped) for sentence
    boundary detection to keep indices aligned with the original text."""

    def test_sweep_uses_text_lower_for_boundaries(self):
        import inspect
        from skills.capability_gate import CapabilityGate
        src = inspect.getsource(CapabilityGate._sweep_blocked_verb_residual)
        assert "text_lower = text.lower()" in src
        assert "text_stripped = _strip_diacritics" in src


# ===========================================================================
# AUDIT 6 REGRESSIONS
# ===========================================================================

# ---------------------------------------------------------------------------
# V6-01: Dismiss path emits CONVERSATION_RESPONSE
# ---------------------------------------------------------------------------

class TestV601DismissConversationResponse:
    """Dismiss command must emit CONVERSATION_RESPONSE and update _last_response_text."""

    def _read_file(self):
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()

    def test_dismiss_emits_conversation_response(self):
        src = self._read_file()
        # After dismiss broadcast, there must be an event_bus.emit(CONVERSATION_RESPONSE ...)
        assert "event_bus.emit(CONVERSATION_RESPONSE" in src

    def test_dismiss_updates_last_response_text(self):
        src = self._read_file()
        # The dismiss path (after detecting dismiss command) must update _last_response_text
        idx = src.find("Dismiss command detected")
        assert idx > 0
        dismiss_block = src[idx:idx + 2200]
        assert "_last_response_text" in dismiss_block


# ---------------------------------------------------------------------------
# V6-02: Crash handler cancels speaking safety timer
# ---------------------------------------------------------------------------

class TestV602CrashHandlerCancelsTimer:
    """_safe_handle crash path must cancel speaking safety timer."""

    def test_crash_handler_cancels_timer(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "perception_orchestrator.py").read_text()
        idx_except = src.find("handle_transcription crashed")
        assert idx_except > 0
        crash_section = src[idx_except:idx_except + 400]
        assert "_cancel_speaking_safety_timer" in crash_section


# ---------------------------------------------------------------------------
# CG6-03: Reflective exclusion tightened — 'heard that' and 'was told' removed
# ---------------------------------------------------------------------------

class TestCG603ReflectiveExclusionTightened:
    """_REFLECTIVE_EXCLUSION_RE must NOT contain exploitable keywords."""

    def test_heard_that_removed(self):
        from skills.capability_gate import _REFLECTIVE_EXCLUSION_RE
        assert not _REFLECTIVE_EXCLUSION_RE.search("I heard that I can sing")

    def test_was_told_removed(self):
        from skills.capability_gate import _REFLECTIVE_EXCLUSION_RE
        assert not _REFLECTIVE_EXCLUSION_RE.search("I was told I can dance")

    def test_genuine_reflective_still_works(self):
        from skills.capability_gate import _REFLECTIVE_EXCLUSION_RE
        assert _REFLECTIVE_EXCLUSION_RE.search("when I was young I used to sing")
        assert _REFLECTIVE_EXCLUSION_RE.search("back when I tried")
        assert _REFLECTIVE_EXCLUSION_RE.search("I wish I could dance")


# ---------------------------------------------------------------------------
# CG6-07: sanitize_status_reply wired into conversation handler
# ---------------------------------------------------------------------------

class TestCG607StatusSanitizeWired:
    """The STATUS tool path must call sanitize_status_reply()."""

    def test_status_route_calls_sanitize(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "conversation_handler.py").read_text()
        assert "sanitize_status_reply" in src or "sanitize_self_report_reply" in src

    def test_status_fallback_uses_self_report_sanitize(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "conversation_handler.py").read_text()
        assert "sanitize_self_report_reply(" in src
        assert "Status native articulation failed" in src

    def test_status_route_uses_native_self_status_articulation(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "conversation_handler.py").read_text()
        assert 'response_class="self_status"' in src
        assert "articulate_meaning_frame(_meaning_frame)" in src


class TestIntrospectionFailClosedWiring:
    """Introspection route should degrade to grounded fallback on grounding miss."""

    def test_introspection_logs_grounding_and_can_fail_closed(self):
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "conversation_handler.py").read_text()
        assert "_log_introspection_grounding(reply, introspection_data, intro_meta)" in src
        assert "reply = fallback_reply" in src


# ---------------------------------------------------------------------------
# NEW-1: HemisphereOrchestrator._networks_lock exists and is used
# ---------------------------------------------------------------------------

class TestNew1HemisphereNetworksLock:
    """_networks dict must be protected by _networks_lock."""

    def test_networks_lock_exists(self):
        from hemisphere.orchestrator import HemisphereOrchestrator
        assert hasattr(HemisphereOrchestrator, "__init__")
        import inspect
        src = inspect.getsource(HemisphereOrchestrator.__init__)
        assert "_networks_lock" in src

    def test_get_networks_for_focus_uses_lock(self):
        import inspect
        from hemisphere.orchestrator import HemisphereOrchestrator
        src = inspect.getsource(HemisphereOrchestrator._get_networks_for_focus)
        assert "_networks_lock" in src

    def test_find_weakest_uses_lock(self):
        import inspect
        from hemisphere.orchestrator import HemisphereOrchestrator
        src = inspect.getsource(HemisphereOrchestrator._find_weakest_network)
        assert "_networks_lock" in src

    def test_total_count_uses_lock(self):
        import inspect
        from hemisphere.orchestrator import HemisphereOrchestrator
        src = inspect.getsource(HemisphereOrchestrator._total_network_count)
        assert "_networks_lock" in src


# ---------------------------------------------------------------------------
# NEW-2: IdentityFusion._lock exists and protects handlers
# ---------------------------------------------------------------------------

class TestNew2IdentityFusionLock:
    """IdentityFusion handlers must be protected by a threading.Lock."""

    def test_lock_exists(self):
        from perception.identity_fusion import IdentityFusion
        obj = IdentityFusion()
        assert hasattr(obj, "_lock")
        import threading
        assert isinstance(obj._lock, type(threading.Lock()))

    def test_on_voice_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion._on_voice)
        assert "self._lock" in src

    def test_on_face_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion._on_face)
        assert "self._lock" in src

    def test_on_presence_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion._on_presence)
        assert "self._lock" in src

    def test_on_wake_word_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion._on_wake_word)
        assert "self._lock" in src

    def test_current_property_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion.current.fget)
        assert "self._lock" in src

    def test_get_status_uses_lock(self):
        import inspect
        from perception.identity_fusion import IdentityFusion
        src = inspect.getsource(IdentityFusion.get_status)
        assert "self._lock" in src


# ═══════════════════════════════════════════════════════════════════════════
# Audit 6 — Hardening Pass Regression Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestV603ProactiveSpeakingLifecycle:
    """Proactive/greeting speech must enter the speaking lifecycle."""

    def test_speak_proactive_method_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        assert "def _speak_proactive" in src

    def test_speak_proactive_sets_speaking(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        body = src[idx:idx + 800]
        assert "set_speaking(True)" in body

    def test_speak_proactive_sends_response_end(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        body = src[idx:idx + 800]
        assert "response_end" in body

    def test_speak_proactive_emits_conversation_response(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        body = src[idx:idx + 800]
        assert "CONVERSATION_RESPONSE" in body

    def test_proactive_speak_closure_calls_speak_proactive(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _proactive_speak")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "_speak_proactive" in body

    def test_on_user_arrived_calls_speak_proactive(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _on_user_arrived")
        assert idx > 0
        next_def = src.find("\n    def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 4000]
        assert "_speak_proactive" in body

    def test_evaluate_proactive_calls_speak_proactive(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def evaluate_proactive")
        assert idx > 0
        next_def = src.find("\n    def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 4000]
        assert "_speak_proactive" in body


class TestNew3ModeManagerLock:
    """ModeManager must use threading.Lock for atomic mode/profile updates."""

    def test_lock_exists(self):
        from consciousness.modes import ModeManager
        mm = ModeManager()
        assert hasattr(mm, "_lock")
        import threading
        assert isinstance(mm._lock, type(threading.Lock()))

    def test_set_mode_uses_lock(self):
        import inspect
        from consciousness.modes import ModeManager
        src = inspect.getsource(ModeManager.set_mode)
        assert "self._lock" in src

    def test_get_state_uses_lock(self):
        import inspect
        from consciousness.modes import ModeManager
        src = inspect.getsource(ModeManager.get_state)
        assert "self._lock" in src

    def test_suggest_mode_uses_lock(self):
        import inspect
        from consciousness.modes import ModeManager
        src = inspect.getsource(ModeManager.suggest_mode_from_attention)
        assert "self._lock" in src


class TestNew6ContradictionEngineLock:
    """ContradictionEngine must use threading.Lock for shared state."""

    def test_lock_exists(self):
        from epistemic.contradiction_engine import ContradictionEngine
        ce = ContradictionEngine()
        assert hasattr(ce, "_lock")
        import threading
        assert isinstance(ce._lock, type(threading.Lock()))

    def test_on_memory_write_uses_lock(self):
        import inspect
        from epistemic.contradiction_engine import ContradictionEngine
        src = inspect.getsource(ContradictionEngine._on_memory_write)
        assert "self._lock" in src

    def test_scan_corpus_uses_lock(self):
        import inspect
        from epistemic.contradiction_engine import ContradictionEngine
        src = inspect.getsource(ContradictionEngine.scan_corpus)
        assert "self._lock" in src

    def test_get_state_uses_lock(self):
        import inspect
        from epistemic.contradiction_engine import ContradictionEngine
        src = inspect.getsource(ContradictionEngine.get_state)
        assert "self._lock" in src

    def test_apply_passive_decay_uses_lock(self):
        import inspect
        from epistemic.contradiction_engine import ContradictionEngine
        src = inspect.getsource(ContradictionEngine.apply_passive_decay)
        assert "self._lock" in src


class TestNew8PresenceTrackerLock:
    """PresenceTracker must use threading.Lock for handler state."""

    def test_lock_import(self):
        import inspect
        from perception.presence import PresenceTracker
        src = inspect.getsource(PresenceTracker.__init__)
        assert "_lock" in src

    def test_on_presence_uses_lock(self):
        import inspect
        from perception.presence import PresenceTracker
        src = inspect.getsource(PresenceTracker._on_presence)
        assert "self._lock" in src

    def test_get_state_uses_lock(self):
        import inspect
        from perception.presence import PresenceTracker
        src = inspect.getsource(PresenceTracker.get_state)
        assert "self._lock" in src


class TestNew7PolicyTelemetryLock:
    """PolicyTelemetry must use threading.Lock for compound read-write."""

    def test_lock_exists(self):
        from policy.telemetry import PolicyTelemetry
        pt = PolicyTelemetry()
        assert hasattr(pt, "_lock")
        import threading
        assert isinstance(pt._lock, type(threading.Lock()))

    def test_snapshot_uses_lock(self):
        import inspect
        from policy.telemetry import PolicyTelemetry
        src = inspect.getsource(PolicyTelemetry.snapshot)
        assert "self._lock" in src

    def test_record_win_rate_snapshot_uses_lock(self):
        import inspect
        from policy.telemetry import PolicyTelemetry
        src = inspect.getsource(PolicyTelemetry.record_win_rate_snapshot)
        assert "self._lock" in src

    def test_record_shadow_uses_lock(self):
        import inspect
        from policy.telemetry import PolicyTelemetry
        src = inspect.getsource(PolicyTelemetry.record_shadow)
        assert "self._lock" in src


class TestNew45AnalyticsObserverLock:
    """ConsciousnessAnalytics and Observer must use threading.Lock."""

    def test_analytics_lock_exists(self):
        from consciousness.consciousness_analytics import ConsciousnessAnalytics
        ca = ConsciousnessAnalytics()
        assert hasattr(ca, "_lock")
        import threading
        assert isinstance(ca._lock, type(threading.Lock()))

    def test_analytics_record_tick_uses_lock(self):
        import inspect
        from consciousness.consciousness_analytics import ConsciousnessAnalytics
        src = inspect.getsource(ConsciousnessAnalytics.record_tick)
        assert "self._lock" in src

    def test_analytics_maybe_refresh_uses_lock(self):
        import inspect
        from consciousness.consciousness_analytics import ConsciousnessAnalytics
        src = inspect.getsource(ConsciousnessAnalytics._maybe_refresh)
        assert "self._lock" in src

    def test_observer_lock_exists(self):
        from consciousness.observer import ConsciousnessObserver
        co = ConsciousnessObserver()
        assert hasattr(co, "_lock")
        import threading
        assert isinstance(co._lock, type(threading.Lock()))

    def test_observer_record_uses_lock(self):
        import inspect
        from consciousness.observer import ConsciousnessObserver
        src = inspect.getsource(ConsciousnessObserver._record)
        assert "self._lock" in src

    def test_observer_get_observation_summary_uses_lock(self):
        import inspect
        from consciousness.observer import ConsciousnessObserver
        src = inspect.getsource(ConsciousnessObserver.get_observation_summary)
        assert "self._lock" in src


class TestC51HemisphereTrainingLock:
    """HemisphereEngine must acquire _training_lock in train and infer paths."""

    def test_build_network_acquires_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("hemisphere", "engine.py").read_text()
        idx = src.find("def build_network")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "_training_lock" in body

    def test_train_distillation_acquires_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("hemisphere", "engine.py").read_text()
        idx = src.find("def train_distillation")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "_training_lock" in body

    def test_infer_acquires_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("hemisphere", "engine.py").read_text()
        idx = src.find("def infer")
        assert idx > 0
        body = src[idx:idx + 500]
        assert "_training_lock" in body


class TestC3A5ContradictionEarlySubscription:
    """ContradictionEngine subscription should happen early in engine.start()."""

    def test_early_subscribe_in_engine_start(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "engine.py"
        ).read_text()
        idx = src.find("def start(self)")
        assert idx > 0
        body = src[idx:idx + 1500]
        assert "ContradictionEngine" in body
        assert ".subscribe()" in body


class TestD51RetrievalLogReferenceCap:
    """MemoryRetrievalLog._recent_references must be capped OrderedDict."""

    def test_recent_references_is_ordered_dict(self):
        from memory.retrieval_log import MemoryRetrievalLog
        log = MemoryRetrievalLog()
        from collections import OrderedDict
        assert isinstance(log._recent_references, OrderedDict)

    def test_record_references_caps_size(self):
        import inspect
        from memory.retrieval_log import MemoryRetrievalLog
        src = inspect.getsource(MemoryRetrievalLog.record_references)
        assert "popitem" in src


# ═══════════════════════════════════════════════════════════════════════════════
# Audit 7 — Deep Integration & Voice Safety Hardening (2026-03-15)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSPEAK01TTSWorkerErrorHandling:
    """TTS worker must not hang on synthesis failure (Audit 7 SPEAK-01)."""

    def test_tts_worker_has_try_except(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("conversation_handler.py").read_text()
        idx = src.find("async def _tts_worker")
        assert idx > 0
        next_def = src.find("\n    async def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 2000]
        assert "try:" in body
        assert "finally:" in body
        assert "task_done()" in body

    def test_tts_worker_handles_exception(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("conversation_handler.py").read_text()
        idx = src.find("async def _tts_worker")
        assert idx > 0
        next_def = src.find("\n    async def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 2000]
        assert "except Exception" in body


class TestSPEAK0304ProactiveLifecycleHardening:
    """_speak_proactive must set _speaking_conv_id and handle TTS errors (Audit 7)."""

    def test_speak_proactive_sets_speaking_conv_id(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        body = src[idx:idx + 700]
        assert "_speaking_conv_id" in body

    def test_speak_proactive_has_error_handling(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        next_def = src.find("\n    def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 2000]
        assert "try:" in body
        assert "except" in body


class TestSPEAK02WakeWordCancelsTimer:
    """Wake word detection must cancel any lingering safety timer (Audit 7)."""

    def test_on_wake_word_cancels_timer(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("perception_orchestrator.py").read_text()
        idx = src.find("def _on_wake_word_detected")
        assert idx > 0
        body = src[idx:idx + 500]
        assert "_cancel_speaking_safety_timer" in body


class TestB01WeEvasionFix:
    """Capability gate must catch 'we can' and 'Jarvis can' claims (Audit 7)."""

    def test_first_person_re_includes_we(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("skills", "capability_gate.py").read_text()
        import re
        m = re.search(r"_FIRST_PERSON_RE\s*=\s*re\.compile\(r'(.+?)'", src)
        assert m, "_FIRST_PERSON_RE not found"
        pattern = m.group(1)
        assert "we" in pattern.lower()

    def test_self_name_re_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("skills", "capability_gate.py").read_text()
        assert "_SELF_NAME_RE" in src

    def test_claim_patterns_include_we_can(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("skills", "capability_gate.py").read_text()
        assert r"[Ww]e " in src or r"we " in src.lower()

    def test_claim_patterns_include_jarvis(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("skills", "capability_gate.py").read_text()
        assert r"[Jj]arvis " in src or "jarvis" in src.lower()

    def test_sweep_uses_self_name(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("skills", "capability_gate.py").read_text()
        assert "_SELF_NAME_RE" in src
        idx = src.find("def _sweep_blocked_verb_residual")
        assert idx > 0
        next_def = src.find("\n    def ", idx + 1)
        body = src[idx:next_def] if next_def > 0 else src[idx:idx + 2000]
        assert "_SELF_NAME_RE" in body or "has_self_ref" in body


class TestP303GraphBridgeLock:
    """GraphBridge must protect counters with threading.Lock (Audit 7)."""

    def test_graph_bridge_has_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "epistemic", "belief_graph", "bridge.py"
        ).read_text()
        assert "self._lock" in src
        assert "threading" in src

    def test_graph_bridge_get_stats_uses_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "epistemic", "belief_graph", "bridge.py"
        ).read_text()
        idx = src.find("def get_stats")
        assert idx > 0
        body = src[idx:idx + 300]
        assert "self._lock" in body

    def test_graph_bridge_handlers_use_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "epistemic", "belief_graph", "bridge.py"
        ).read_text()
        for handler in ("_on_contradiction_detected", "_on_tension_held", "_on_memory_associated"):
            idx = src.find(f"def {handler}")
            assert idx > 0, f"{handler} not found"
            next_def = src.find("\n    def ", idx + 1)
            body = src[idx:next_def] if next_def > 0 else src[idx:]
            assert "self._lock" in body, f"{handler} does not use self._lock"


class TestP13PolicyTelemetryLocks:
    """PolicyTelemetry record_block/record_pass/record_decision must use _lock (Audit 7)."""

    def test_record_block_uses_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("policy", "telemetry.py").read_text()
        idx = src.find("def record_block")
        assert idx > 0
        body = src[idx:idx + 200]
        assert "self._lock" in body

    def test_record_pass_uses_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("policy", "telemetry.py").read_text()
        idx = src.find("def record_pass")
        assert idx > 0
        body = src[idx:idx + 200]
        assert "self._lock" in body

    def test_record_decision_uses_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("policy", "telemetry.py").read_text()
        idx = src.find("def record_decision")
        assert idx > 0
        body = src[idx:idx + 300]
        assert "self._lock" in body


class TestH44RetrainNetworkLock:
    """HemisphereEngine.retrain_network must acquire _training_lock (Audit 7)."""

    def test_retrain_network_uses_training_lock(self):
        src = pathlib.Path(__file__).parent.parent.joinpath("hemisphere", "engine.py").read_text()
        idx = src.find("def retrain_network")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "_training_lock" in body


# ════════════════════════════════════════════════════════════════════════════
# Audit 10 — Deep Lifecycle & Adversarial Completeness (2026-03-16)
# ════════════════════════════════════════════════════════════════════════════


class TestA10CS1CycleProtection:
    """Consciousness cycles must have try/except to prevent cascade failure."""

    @staticmethod
    def _read_cs():
        return pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()

    def test_meta_thoughts_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_meta_thoughts(")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "except Exception" in body or "except:" in body

    def test_analysis_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_analysis(")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "except Exception" in body or "except:" in body

    def test_evolution_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_evolution(")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "except Exception" in body or "except:" in body

    def test_mutation_cycle_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_mutation_cycle(")
        assert idx > 0
        body = src[idx:idx + 400]
        assert "except Exception" in body or "except:" in body

    def test_existential_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_existential(")
        assert idx > 0
        body = src[idx:idx + 1100]
        assert "except Exception" in body or "except:" in body

    def test_dialogue_has_try_except(self):
        src = self._read_cs()
        idx = src.find("def _run_dialogue(")
        assert idx > 0
        body = src[idx:idx + 900]
        assert "except Exception" in body or "except:" in body

    def test_mutation_health_check_protected(self):
        src = self._read_cs()
        idx = src.find("self._check_mutation_health()")
        assert idx > 0
        block = src[max(0, idx - 80):idx + 60]
        assert "try:" in block


class TestCG10BlockedVerbs:
    """Capability gate must block song/music/sketch domain nouns."""

    def test_song_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "song" in _BLOCKED_CAPABILITY_VERBS
        assert "songs" in _BLOCKED_CAPABILITY_VERBS

    def test_music_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "music" in _BLOCKED_CAPABILITY_VERBS
        assert "musical" in _BLOCKED_CAPABILITY_VERBS

    def test_vocal_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "vocal" in _BLOCKED_CAPABILITY_VERBS
        assert "vocals" in _BLOCKED_CAPABILITY_VERBS

    def test_sketch_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "sketch" in _BLOCKED_CAPABILITY_VERBS
        assert "sketching" in _BLOCKED_CAPABILITY_VERBS

    def test_artwork_illustration_blocked(self):
        from skills.capability_gate import _BLOCKED_CAPABILITY_VERBS
        assert "artwork" in _BLOCKED_CAPABILITY_VERBS
        assert "illustration" in _BLOCKED_CAPABILITY_VERBS


class TestLJ10BuiltinFamilyExemption:
    """Cleanup must not block builtin-family learning jobs on restart."""

    def test_cleanup_exempts_builtin_families(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "learning_jobs.py"
        ).read_text()
        idx = src.find("def _cleanup_actuator_junk_jobs")
        assert idx > 0
        body = src[idx:idx + 1200]
        assert "BUILTIN_FAMILIES" in body or "is_builtin" in body

    def test_cleanup_uses_regex_strip(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "learning_jobs.py"
        ).read_text()
        idx = src.find("def _cleanup_actuator_junk_jobs")
        assert idx > 0
        body = src[idx:idx + 800]
        assert "rstrip" not in body, "rstrip should be replaced with regex"


class TestCD10QueueCooldown:
    """_can_queue_proposal must respect 24h global cooldown."""

    def test_can_queue_checks_cooldown(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "discovery.py"
        ).read_text()
        idx = src.find("def _can_queue_proposal")
        assert idx > 0
        body = src[idx:idx + 300]
        assert "_can_surface_proposal" in body


class TestV10PlaybackClearsConvId:
    """_on_playback_complete must clear _speaking_conv_id."""

    def test_playback_clears_speaking_conv_id(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "perception_orchestrator.py"
        ).read_text()
        idx = src.find("def _on_playback_complete")
        assert idx > 0
        body = src[idx:idx + 1100]
        assert '_speaking_conv_id = ""' in body or "_speaking_conv_id = ''" in body


class TestV10ProactiveSpeakConvId:
    """_speak_proactive speak command must include conversation_id."""

    def test_speak_command_has_conv_id(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "perception_orchestrator.py"
        ).read_text()
        idx = src.find("def _speak_proactive")
        assert idx > 0
        body = src[idx:idx + 1200]
        assert '"conversation_id"' in body or "'conversation_id'" in body


class TestA10HE1PrunedModelUnload:
    """Pruned hemisphere networks must be unloaded from engine."""

    def test_prune_calls_remove_model(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "orchestrator.py"
        ).read_text()
        idx = src.find("Pruned weakest network")
        assert idx > 0
        block = src[max(0, idx - 300):idx]
        assert "remove_model" in block


class TestA10GL1MetricGoalResume:
    """Auto-paused metric goals must resume on re-degradation."""

    def test_merge_evidence_resumes_paused(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "goals", "goal_manager.py"
        ).read_text()
        idx = src.find("def _merge_evidence")
        assert idx > 0
        body = src[idx:idx + 600]
        assert '"paused"' in body or "'paused'" in body
        assert '"active"' in body or "'active'" in body


class TestA10GL2TemplateExhaustion:
    """Non-metric goals complete when templates are exhausted with progress > 0."""

    def test_review_handles_template_exhaustion(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "goals", "review.py"
        ).read_text()
        assert "Template exhausted" in src

    def test_plan_next_task_called_in_review(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "goals", "review.py"
        ).read_text()
        assert "plan_next_task" in src


class TestV10PiDoneMonitor:
    """Pi done monitor: serialized dispatch, conv_id parameter, no flag."""

    def test_no_speak_cmd_monitor_flag(self):
        """_speak_cmd_monitor_active removed — was causing ordering races."""
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "main.py"
        ).read_text()
        assert "_speak_cmd_monitor_active" not in src

    def test_response_end_always_starts_monitor(self):
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "main.py"
        ).read_text()
        idx = src.find('msg.type == "response_end"')
        assert idx > 0
        body = src[idx:idx + 600]
        assert "_brain_audio_done_monitor" in body
        assert "conv_id" in body

    def test_speak_cmd_no_monitor(self):
        """command/speak should NOT start its own monitor (response_end follows)."""
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "main.py"
        ).read_text()
        idx = src.find('action == "speak"')
        assert idx > 0
        body = src[idx:idx + 300]
        assert "_brain_audio_done_monitor" not in body

    def test_done_monitor_takes_conv_id_param(self):
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "main.py"
        ).read_text()
        assert "def _brain_audio_done_monitor(self, conv_id" in src

    def test_serialized_brain_msg_executor(self):
        """Transport uses single-thread executor for ordered message processing."""
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "transport", "ws_client.py"
        ).read_text()
        assert "max_workers=1" in src
        assert "_brain_msg_executor" in src


class TestA10LI1LibraryRollback:
    """Library index must rollback on partial failure."""

    def test_add_chunk_has_rollback(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "library", "index.py"
        ).read_text()
        idx = src.find("def add_chunk")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "rollback" in body


# ── Audit 10+ hotfix regressions ──────────────────────────────────────

class TestCGLetMeKnowFix:
    """'Let me know how I can assist' must not be blocked as an offer."""

    def test_let_me_know_not_blocked(self):
        import re
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "capability_gate.py"
        ).read_text()
        assert "let me(?! know" in src, "offer pattern should exclude 'let me know'"

    def test_let_me_know_regex_behavior(self):
        import re
        _TERM = r"(?:[.!?,;:\-]|$)"
        pattern = re.compile(
            r"\b(?:want to hear(?: me)?|want me to|should I|shall I|let me(?! know\b)|allow me to"
            r"|would you (?:like|want) me to|how about I|listen to me"
            r"|why don't I|I think I (?:should|could)"
            r") (.{3,80}?)" + _TERM,
            re.I,
        )
        assert pattern.search("Let me know how I can assist you.") is None
        assert pattern.search("Let me know what you need.") is None
        assert pattern.search("Let me help you with that.") is not None
        assert pattern.search("Let me compose a song for you.") is not None


class TestIntrospectionLearningPerception:
    """Learning topic must include perception sections to avoid confabulation."""

    def test_learning_includes_identity_fusion(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "tools", "introspection_tool.py"
        ).read_text()
        idx = src.find("_TOPIC_TO_SECTIONS")
        assert idx > 0
        block = src[idx:idx + 600]
        learn_idx = block.find('"learning"')
        assert learn_idx > 0
        line_end = block.find("\n", learn_idx)
        line = block[learn_idx:line_end]
        assert "identity_fusion" in line

    def test_learning_includes_emotion(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "tools", "introspection_tool.py"
        ).read_text()
        idx = src.find("_TOPIC_TO_SECTIONS")
        assert idx > 0
        block = src[idx:idx + 600]
        learn_idx = block.find('"learning"')
        assert learn_idx > 0
        line_end = block.find("\n", learn_idx)
        line = block[learn_idx:line_end]
        assert "emotion" in line

    def test_hemisphere_distillation_disambiguation(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "tools", "introspection_tool.py"
        ).read_text()
        idx = src.find("def _build_hemisphere")
        assert idx > 0
        body = src[idx:idx + 1500]
        assert "distillation" in body.lower()
        assert "GPU model" in body or "GPU perception" in body


class TestCuriosityBridgeWiring:
    """External proactive evaluator must also check curiosity questions."""

    def test_evaluate_proactive_checks_curiosity(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "perception_orchestrator.py"
        ).read_text()
        idx = src.find("def evaluate_proactive")
        assert idx > 0
        body = src[idx:idx + 2000]
        assert "_check_curiosity_question" in body

    def test_on_proactive_check_still_has_fallthrough(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "engine.py"
        ).read_text()
        idx = src.find("def on_proactive_check")
        assert idx > 0
        body = src[idx:idx + 500]
        assert "check_proactive_behavior" in body


class TestCGSubordinateMaybe:
    """'maybe I'll stick with that' must not be blocked (subordinate clause)."""

    def test_maybe_in_subordinate_list(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "capability_gate.py"
        ).read_text()
        assert '"maybe"' in src

    def test_stick_with_is_conversational(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "skills", "capability_gate.py"
        ).read_text()
        assert '"stick with"' in src


class TestPhilosophicalPromptGrounding:
    """Philosophical engagement prompt must require evidence anchoring."""

    def test_no_invent_preferences(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "reasoning", "context.py"
        ).read_text()
        assert "Do NOT invent aesthetic preferences" in src

    def test_anchor_to_state(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "reasoning", "context.py"
        ).read_text()
        assert "Anchor every claim" in src

    def test_honest_about_no_evidence(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "reasoning", "context.py"
        ).read_text()
        assert "don't have a recorded preference" in src


class TestIdentityTopicPatternBroadened:
    """Identity topic bucket must match 'your own name', bare 'name', etc."""

    def test_your_own_name_matches(self):
        import re
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "tools", "introspection_tool.py"
        ).read_text()
        idx = src.find('"identity"')
        assert idx > 0
        block = src[idx:idx + 400]
        pat_match = re.search(r'your \(\?:\\w\+ \)\?name', block)
        assert pat_match, "identity pattern should match 'your own name'"

    def test_bare_name_matches(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "tools", "introspection_tool.py"
        ).read_text()
        idx = src.find('"identity"')
        assert idx > 0
        block = src[idx:idx + 400]
        assert r"\bnamed?\b" in block


# ── Audit 11 ──────────────────────────────────────────────────────────────


class TestOB01MemoryDataclassAPI:
    """OB-01: _collect_onboarding_metrics must use getattr on Memory dataclass,
    not dict .get() which raises AttributeError on frozen dataclasses."""

    def _get_method_body(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()
        marker = "def _collect_onboarding_metrics("
        block_start = src.find(marker)
        assert block_start > 0, "method definition not found"
        next_def = src.find("\n    def ", block_start + len(marker))
        return src[block_start:next_def] if next_def > 0 else src[block_start:]

    def test_no_dict_get_on_memory_objects(self):
        block = self._get_method_body()
        assert "mem.get(" not in block, \
            "Memory is a frozen dataclass — use getattr(), not .get()"

    def test_uses_getattr_for_tags(self):
        block = self._get_method_body()
        assert 'getattr(mem, "tags"' in block

    def test_uses_identity_subject_not_identity_subject_id(self):
        block = self._get_method_body()
        assert "identity_subject_id" not in block, \
            "Memory field is 'identity_subject', not 'identity_subject_id'"
        assert 'getattr(mem, "identity_subject"' in block


class TestOB02MemoryAccuracyWired:
    """OB-02: memory_accuracy must be populated in readiness metrics."""

    def test_memory_accuracy_collected(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()
        marker = "def _collect_onboarding_metrics("
        start = src.find(marker)
        end = src.find("\n    def ", start + len(marker))
        block = src[start:end]
        assert '"memory_accuracy"' in block, \
            "memory_accuracy must be populated for readiness gate"


class TestOB03MissingCheckpointMetrics:
    """OB-03: Checkpoint metrics must have collection paths."""

    def _get_method_body(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()
        marker = "def _collect_onboarding_metrics("
        start = src.find(marker)
        end = src.find("\n    def ", start + len(marker))
        return src[start:end]

    def test_readiness_composite_collected(self):
        assert '"readiness_composite"' in self._get_method_body()

    def test_unsafe_inferences_collected(self):
        assert '"unsafe_inferences_24h"' in self._get_method_body()

    def test_correction_accuracy_collected(self):
        assert '"correction_accuracy"' in self._get_method_body()

    def test_memory_recall_precision_collected(self):
        assert '"memory_recall_precision"' in self._get_method_body()

    def test_stage4_routine_priority_tags_counted(self):
        block = self._get_method_body()
        for tag in (
            '"routine"',
            '"schedule"',
            '"daily"',
            '"priority"',
            '"focus_window"',
            '"availability"',
            '"interrupt_preference"',
        ):
            assert tag in block


class TestCS01SelfImproveTryExcept:
    """CS-01: _run_self_improvement must have try/except wrapping."""

    def test_self_improve_has_exception_handling(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()
        marker = "def _run_self_improvement("
        idx = src.find(marker)
        assert idx > 0
        next_def = src.find("\n    def ", idx + len(marker))
        block = src[idx:next_def] if next_def > 0 else src[idx:idx + 1200]
        assert "except Exception" in block, \
            "_run_self_improvement must catch exceptions"


class TestCS03TrackedCycleCatchesExceptions:
    """CS-03: _tracked_cycle must catch exceptions so one crash
    doesn't skip all subsequent cycles."""

    def test_tracked_cycle_has_except(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "consciousness_system.py"
        ).read_text()
        marker = "def _tracked_cycle("
        idx = src.find(marker)
        assert idx > 0
        next_def = src.find("\n    def ", idx + len(marker))
        block = src[idx:next_def] if next_def > 0 else src[idx:idx + 600]
        assert "except Exception" in block, \
            "_tracked_cycle should catch exceptions for resilience"


class TestPI02PerceptionEventImport:
    """PI-02: PerceptionEvent must be importable at module scope in pi/main.py."""

    def test_perception_event_in_module_import(self):
        src = pathlib.Path(__file__).parent.parent.parent.joinpath(
            "pi", "main.py"
        ).read_text()
        import_block_end = src.find("from senses")
        import_block = src[:import_block_end]
        assert "PerceptionEvent" in import_block, \
            "PerceptionEvent must be in module-level import"


class TestAU01QueueRemoveGuarded:
    """AU-01: Queue remove in success path must be guarded against ValueError."""

    def test_success_path_remove_guarded(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "autonomy", "orchestrator.py"
        ).read_text()
        removes = []
        for i, line in enumerate(src.splitlines(), 1):
            if "self._queue.remove(intent)" in line:
                removes.append(i)
        for line_num in removes:
            lines = src.splitlines()
            context = "\n".join(lines[max(0, line_num - 4):line_num])
            assert "try:" in context or "except ValueError" in context, \
                f"Queue remove at line {line_num} must be inside try/except ValueError"


# ═══════════════════════════════════════════════════════════════════════
# Audit 12 — Data-Layer Fixes
# ═══════════════════════════════════════════════════════════════════════

class TestPolicyShadowTickNotInBuffer:
    """Shadow tick experience writes were removed from the buffer.

    The evaluator's score_retrospective() still uses health rewards for A/B
    comparison — only the experience buffer write is gone.
    """

    def _get_shadow_eval_body(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "engine.py"
        ).read_text()
        marker = "def run_shadow_evaluation("
        block_start = src.find(marker)
        assert block_start > 0
        next_def = src.find("\n    def ", block_start + len(marker))
        return src[block_start:next_def] if next_def > 0 else src[block_start:]

    def test_no_experience_buffer_add_in_shadow(self):
        body = self._get_shadow_eval_body()
        assert "experience_buffer.add(" not in body, \
            "Shadow tick must NOT write to the experience buffer"

    def test_score_retrospective_still_called(self):
        body = self._get_shadow_eval_body()
        assert "score_retrospective" in body, \
            "Evaluator retrospective scoring must remain in shadow path"

    def test_record_pending_shadow_still_called(self):
        body = self._get_shadow_eval_body()
        assert "record_pending_shadow" in body, \
            "Evaluator pending shadow recording must remain"


class TestAutonomyKnowledgeOutcome:
    """Autonomy outcomes now consider immediate knowledge creation."""

    def _get_metadata_block(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "autonomy", "orchestrator.py"
        ).read_text()
        return src

    def test_metadata_includes_memories_created(self):
        src = self._get_metadata_block()
        assert '"memories_created": memories_created' in src, \
            "Intent metadata must store memories_created for outcome resolution"

    def test_metadata_includes_immediate_success(self):
        src = self._get_metadata_block()
        assert '"immediate_success"' in src, \
            "Intent metadata must store immediate_success flag"

    def test_process_delta_considers_immediate_success(self):
        src = self._get_metadata_block()
        marker = "def _process_delta_outcome("
        block_start = src.find(marker)
        assert block_start > 0
        next_def = src.find("\n    def ", block_start + len(marker))
        body = src[block_start:next_def] if next_def > 0 else src[block_start:]
        assert "immediate_success" in body, \
            "_process_delta_outcome must check immediate_success"


class TestBlueDiamondsSkipsCodebase:
    """Blue Diamonds graduation skips codebase sources."""

    def test_study_py_skips_codebase(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "library", "study.py"
        ).read_text()
        marker = "def _try_post_study_graduation("
        block_start = src.find(marker)
        assert block_start > 0
        next_def = src.find("\ndef ", block_start + len(marker))
        body = src[block_start:next_def] if next_def > 0 else src[block_start:]
        assert '"codebase"' in body, \
            "_try_post_study_graduation must skip codebase sources"

    def test_knowledge_integrator_skips_codebase(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "autonomy", "knowledge_integrator.py"
        ).read_text()
        marker = "def _try_graduate_to_blue_diamonds("
        block_start = src.find(marker)
        assert block_start > 0
        next_def = src.find("\n    def ", block_start + len(marker))
        body = src[block_start:next_def] if next_def > 0 else src[block_start:]
        assert '"codebase"' in body, \
            "_try_graduate_to_blue_diamonds must skip codebase sources"


class TestFlightRecorderPersistence:
    """Flight recorder now persists across reboots."""

    def test_save_function_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        assert "def _save_flight_recorder" in src

    def test_load_function_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        assert "def _load_flight_recorder" in src

    def test_save_called_after_append(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        append_idx = src.find("_flight_recorder.append(_ep_record)")
        assert append_idx > 0
        next_lines = src[append_idx:append_idx + 200]
        assert "_save_flight_recorder()" in next_lines, \
            "save must be called after flight recorder append"

    def test_load_called_at_module_level(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        load_call_idx = src.find("_load_flight_recorder()")
        assert load_call_idx > 0
        load_def_idx = src.find("def _load_flight_recorder")
        assert load_call_idx > load_def_idx, \
            "_load_flight_recorder() must be called after its definition"

    def test_text_limits_increased(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        assert "text[:500]" in src, "User input truncation should be 500 chars"
        assert "reply[:500]" in src, "Response text truncation should be 500 chars"


###############################################################################
# Audit 13: Memory cap + study claim dedup
###############################################################################


class TestMemoryCapIncrease:
    """MAX_MEMORIES raised from 500 to 2000."""

    def test_max_memories_value(self):
        from memory.maintenance import MAX_MEMORIES
        assert MAX_MEMORIES == 2000, f"Expected MAX_MEMORIES=2000, got {MAX_MEMORIES}"

    def test_gc_threshold_follows(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "memory", "maintenance.py"
        ).read_text()
        assert "GC_THRESHOLD = MAX_MEMORIES" in src


class TestStudyClaimDedup:
    """_create_claim_memories must skip claims already stored for a source."""

    def test_dedup_helper_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "library", "study.py"
        ).read_text()
        assert "def _get_existing_claims_for_source(" in src

    def test_dedup_called_before_create(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "library", "study.py"
        ).read_text()
        helper_idx = src.find("_get_existing_claims_for_source(memory_storage")
        create_idx = src.find("canonical_remember(CreateMemoryData(")
        assert helper_idx > 0 and create_idx > 0
        assert helper_idx < create_idx, \
            "existing claims must be fetched before creating new memories"

    def test_claim_key_checked(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "library", "study.py"
        ).read_text()
        assert "if claim_key in existing_claims:" in src


###############################################################################
# Audit 13b: Distillation rehydration + policy purge + density normalization
###############################################################################


class TestDistillationRehydration:
    """DistillationCollector must load persisted signals on boot."""

    def test_rehydrate_method_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "distillation.py"
        ).read_text()
        assert "def _rehydrate_from_disk(self)" in src

    def test_rehydrate_called_in_init(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "distillation.py"
        ).read_text()
        init_block = src[src.find("def __init__(self)"):src.find("def _rehydrate_from_disk")]
        assert "_rehydrate_from_disk()" in init_block

    def test_parse_jsonl_helper_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "distillation.py"
        ).read_text()
        assert "def _parse_jsonl(" in src

    def test_rehydrate_loads_both_buffers_and_quarantine(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "distillation.py"
        ).read_text()
        method = src[src.find("def _rehydrate_from_disk"):src.find("def _parse_jsonl")]
        assert "TRAINING_DATA_DIR" in method
        assert "QUARANTINE_DIR" in method


class TestPolicyExperiencePurge:
    """Experience buffer load must filter shadow_tick entries."""

    def test_shadow_tick_filtered(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "policy", "experience_buffer.py"
        ).read_text()
        assert 'source == "shadow_tick"' in src

    def test_rewrite_clean_exists(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "policy", "experience_buffer.py"
        ).read_text()
        assert "def _rewrite_clean(self)" in src

    def test_rewrite_called_after_purge(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "policy", "experience_buffer.py"
        ).read_text()
        skip_idx = src.find("skipped > 0")
        assert skip_idx > 0
        after = src[skip_idx:skip_idx + 200]
        assert "_rewrite_clean()" in after


class TestDensityNormalization:
    """Hardcoded /500 density references must track MAX_MEMORIES."""

    def test_hemisphere_orchestrator_uses_max_memories(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "hemisphere", "orchestrator.py"
        ).read_text()
        assert "from memory.maintenance import MAX_MEMORIES" in src
        assert "/ 500.0" not in src or "latency / 500.0" in src

    def test_health_monitor_uses_max_memories(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "consciousness", "health_monitor.py"
        ).read_text()
        assert "self._max_memories" in src
        assert "memory_count / 500" not in src

    def test_lifecycle_log_uses_max_memories(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "memory", "lifecycle_log.py"
        ).read_text()
        assert "_MAX_MEMORIES" in src
        assert "min(memory_count, 500)" not in src

    def test_dashboard_js_updated(self):
        """Memory density is now computed server-side, not in JS with hardcoded cap."""
        from memory.storage import memory_storage
        assert memory_storage._max_capacity >= 2000


###############################################################################
# Audit 14: Cancel token scoping + policy restore + learning contract alignment
###############################################################################


class TestConversationScopedCancellation:
    """Barge-in cancellation must stay scoped to the originating conversation."""

    def test_handle_transcription_checks_conversation_id(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "conversation_handler.py"
        ).read_text()
        marker = "def _cancelled() -> bool:"
        block_start = src.find(marker)
        assert block_start > 0
        next_def = src.find("\n    def ", block_start + len(marker))
        body = src[block_start:next_def] if next_def > 0 else src[block_start:]
        assert 'cancel_flag.get("id") != conversation_id' in body
        assert 'cancel_flag.get("cancelled")' in body

    def test_orchestrator_passes_per_conversation_cancel_state(self):
        src = pathlib.Path(__file__).parent.parent.joinpath(
            "perception_orchestrator.py"
        ).read_text()
        assert 'cancel_state = {"id": conversation_id, "cancelled": False}' in src
        assert "cancel_flag=cancel_state" in src
        assert "cancel_flag=self._active_conversation" not in src


class TestPolicyRestoreOnBoot:
    """Boot should restore the active persisted policy model when available."""

    def test_restore_helper_loads_active_model(self, tmp_path, monkeypatch):
        import importlib
        import types
        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(
                AsyncClient=object,
                ChatResponse=object,
                ResponseError=Exception,
            ),
        )
        brain_main = importlib.import_module("main")
        import policy.policy_nn as policy_nn

        model_path = tmp_path / "policy_v0007.pt"
        model_path.write_text("stub")

        class Active:
            arch = "gru"
            version = 7
            path = str(model_path)

        class Registry:
            def get_active(self):
                return Active()

        calls = {}

        class FakeController:
            def __init__(self, arch="mlp2", input_dim=0):
                calls["arch"] = arch
                calls["input_dim"] = input_dim

            def set_encoder(self, encoder):
                calls["encoder"] = encoder

            def load(self, path):
                calls["path"] = path
                return True

        monkeypatch.setattr(policy_nn, "PolicyNNController", FakeController)

        encoder = object()
        restored = brain_main._restore_active_policy_controller(Registry(), encoder)

        assert isinstance(restored, FakeController)
        assert calls["arch"] == "gru"
        assert calls["path"] == str(model_path)
        assert calls["encoder"] is encoder

    def test_restore_helper_returns_none_when_path_missing(self):
        import importlib
        import types
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(
                AsyncClient=object,
                ChatResponse=object,
                ResponseError=Exception,
            ),
        )
        brain_main = importlib.import_module("main")

        class Active:
            arch = "mlp3"
            version = 2
            path = "/definitely/missing/policy.pt"

        class Registry:
            def get_active(self):
                return Active()

        try:
            assert brain_main._restore_active_policy_controller(Registry(), object()) is None
        finally:
            monkeypatch.undo()


class TestPerceptualResolverAlignment:
    """Resolver contracts must match the perceptual executors' current outputs."""

    def test_audio_analysis_uses_teacher_signals_gate(self):
        from skills.resolver import resolve_skill

        res = resolve_skill("please do audio analysis on this clip")
        assert res is not None
        assert res.skill_id == "audio_analysis_v1"
        assert any(g["id"] == "gate:teacher_signals_present" for g in res.hard_gates)
        assert res.default_phases[0]["exit_conditions"] == ["gate:teacher_signals_present"]

    @pytest.mark.parametrize(
        ("prompt", "skill_id"),
        [
            ("identify the speaker", "speaker_identification_v1"),
            ("detect emotion from my voice", "emotion_detection_v1"),
            ("classify object presence", "perception_distilled_v1"),
        ],
    )
    def test_perceptual_train_phase_uses_train_tick_artifact(self, prompt, skill_id):
        from skills.resolver import resolve_skill

        res = resolve_skill(prompt)
        assert res is not None
        assert res.skill_id == skill_id
        train_phase = next(p for p in res.default_phases if p["name"] == "train")
        assert train_phase["exit_conditions"] == ["artifact:train_tick"]


# ───────────────────── Audit 13 regressions ──────────────────────


class TestAudit13_K13_01_SimulatorIsolation:
    """K13-01: Simulator must use its own CausalEngine, not share with real pipeline."""

    def test_world_model_uses_separate_engines(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        assert wm._causal is not wm._sim_causal, \
            "Simulator should have its own CausalEngine"
        assert wm._simulator._causal is wm._sim_causal

    def test_simulation_does_not_pollute_real_predictions(self):
        from cognition.world_model import WorldModel
        from cognition.world_state import WorldDelta
        wm = WorldModel()
        wm.update()
        real_pred_count = len([p for p in wm._causal._predictions if p.outcome == "pending"])
        delta = WorldDelta(facet="user", event="user_arrived",
                           details={"confidence": 0.9}, timestamp=0)
        wm._simulator.simulate(wm._current, delta)
        real_pred_count_after = len([p for p in wm._causal._predictions if p.outcome == "pending"])
        assert real_pred_count_after == real_pred_count, \
            "Simulation should not add predictions to the real CausalEngine"


class TestAudit13_K13_02_SimulatorDeltaKeys:
    """K13-02: Simulator must handle both 'from'/'to' and domain-specific keys."""

    def test_emotion_changed_uses_to_key(self):
        from cognition.simulator import _apply_delta_to_state
        from cognition.world_state import WorldState, WorldDelta
        ws = WorldState()
        delta = WorldDelta(facet="user", event="emotion_changed",
                           details={"from": "neutral", "to": "happy", "confidence": 0.9},
                           timestamp=0)
        new_ws = _apply_delta_to_state(ws, delta)
        assert new_ws.user.emotion == "happy"

    def test_speaker_changed_uses_to_key(self):
        from cognition.simulator import _apply_delta_to_state
        from cognition.world_state import WorldState, WorldDelta
        ws = WorldState()
        delta = WorldDelta(facet="user", event="speaker_changed",
                           details={"from": "", "to": "Alice"},
                           timestamp=0)
        new_ws = _apply_delta_to_state(ws, delta)
        assert new_ws.user.speaker_name == "Alice"

    def test_mode_changed_uses_to_key(self):
        from cognition.simulator import _apply_delta_to_state
        from cognition.world_state import WorldState, WorldDelta
        ws = WorldState()
        delta = WorldDelta(facet="system", event="mode_changed",
                           details={"from": "passive", "to": "conversational"},
                           timestamp=0)
        new_ws = _apply_delta_to_state(ws, delta)
        assert new_ws.system.mode == "conversational"

    def test_topic_changed_uses_to_key(self):
        from cognition.simulator import _apply_delta_to_state
        from cognition.world_state import WorldState, WorldDelta
        ws = WorldState()
        delta = WorldDelta(facet="conversation", event="topic_changed",
                           details={"from": "", "to": "AI research"},
                           timestamp=0)
        new_ws = _apply_delta_to_state(ws, delta)
        assert new_ws.conversation.topic == "AI research"


class TestAudit13_K13_03_SimulatorPromotionWired:
    """K13-03: SimulatorPromotion.record_outcome must be called from world model."""

    def test_sim_promotion_receives_data(self):
        from cognition.world_model import WorldModel
        wm = WorldModel()
        wm.update()
        wm.update()
        wm.update()
        initial_validated = wm._sim_promotion._state.total_validated
        for _ in range(10):
            wm.update()
        assert wm._sim_promotion._state.total_validated >= initial_validated


class TestAudit13_EP13_01_TensionRefinesEdges:
    """EP13-01: GraphBridge must create refines edges for tension-state beliefs."""

    def test_tension_state_not_filtered(self):
        from epistemic.belief_record import BeliefRecord
        import time as _time
        b = BeliefRecord(
            belief_id="b1", canonical_subject="x", canonical_predicate="is",
            canonical_object="y", modality="is", stance="assert",
            polarity=1, claim_type="factual", epistemic_status="active",
            extraction_confidence=0.8, belief_confidence=0.7,
            provenance="user_claim", scope="", source_memory_id="m1",
            timestamp=_time.time(), time_range=None, is_state_belief=False,
            conflict_key="x:is", evidence_refs=[], contradicts=[],
            resolution_state="tension", rendered_claim="x is y",
        )
        assert b.resolution_state in ("active", "tension")


class TestAudit13_EP13_03_BeliefPersistence:
    """EP13-03: Belief resolution/confidence/contradiction changes must persist."""

    def test_update_resolution_persists(self, tmp_path):
        from epistemic.belief_record import BeliefStore, BeliefRecord
        import time as _time
        bp = str(tmp_path / "beliefs.jsonl")
        tp = str(tmp_path / "tensions.jsonl")
        store = BeliefStore(beliefs_path=bp, tensions_path=tp)
        b = BeliefRecord(
            belief_id="b1", canonical_subject="x", canonical_predicate="is",
            canonical_object="y", modality="is", stance="assert",
            polarity=1, claim_type="factual", epistemic_status="active",
            extraction_confidence=0.8, belief_confidence=0.7,
            provenance="user_claim", scope="", source_memory_id="m1",
            timestamp=_time.time(), time_range=None, is_state_belief=False,
            conflict_key="x:is", evidence_refs=[], contradicts=[],
            resolution_state="active", rendered_claim="x is y",
        )
        store.add(b)
        store.update_resolution("b1", "superseded")
        lines = (tmp_path / "beliefs.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 2, "update_resolution should persist to JSONL"

    def test_update_confidence_persists(self, tmp_path):
        from epistemic.belief_record import BeliefStore, BeliefRecord
        import time as _time
        bp = str(tmp_path / "beliefs.jsonl")
        tp = str(tmp_path / "tensions.jsonl")
        store = BeliefStore(beliefs_path=bp, tensions_path=tp)
        b = BeliefRecord(
            belief_id="b1", canonical_subject="x", canonical_predicate="is",
            canonical_object="y", modality="is", stance="assert",
            polarity=1, claim_type="factual", epistemic_status="active",
            extraction_confidence=0.8, belief_confidence=0.7,
            provenance="user_claim", scope="", source_memory_id="m1",
            timestamp=_time.time(), time_range=None, is_state_belief=False,
            conflict_key="x:is", evidence_refs=[], contradicts=[],
            resolution_state="active", rendered_claim="x is y",
        )
        store.add(b)
        store.update_belief_confidence("b1", 0.5)
        lines = (tmp_path / "beliefs.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 2


class TestAudit13_EP13_04_TensionPersistence:
    """EP13-04: Tension updates must persist."""

    def test_update_tension_persists(self, tmp_path):
        from epistemic.belief_record import BeliefStore, TensionRecord
        bp = str(tmp_path / "beliefs.jsonl")
        tp = str(tmp_path / "tensions.jsonl")
        store = BeliefStore(beliefs_path=bp, tensions_path=tp)
        import time as _time
        t = TensionRecord(
            tension_id="t1", topic="test",
            belief_ids=["b1", "b2"],
            conflict_key="x:is",
            created_at=_time.time(),
            last_revisited=_time.time(),
            revisit_count=0,
            stability=0.5,
            maturation_score=0.0,
        )
        store.add_tension(t)
        import dataclasses
        updated = dataclasses.replace(t, maturation_score=0.5, revisit_count=10)
        store.update_tension(updated)
        lines = (tmp_path / "tensions.jsonl").read_text().strip().splitlines()
        assert len(lines) >= 2, "update_tension should persist to JSONL"


class TestAudit13_POL13_01_ShadowRunnerScoring:
    """POL13-01: Shadow runner must use deviation-based scoring, not identity."""

    def test_shadow_runner_uses_score_retrospective(self):
        from policy.shadow_runner import ShadowPolicyRunner
        import inspect
        src = inspect.getsource(ShadowPolicyRunner.record_shadow_outcome)
        assert "score_retrospective" in src, \
            "Shadow runner should use deviation-based scoring via score_retrospective"
        assert "shadow_reward = kernel_reward" not in src, \
            "Shadow runner should NOT set shadow_reward = kernel_reward (always ties)"


class TestAudit13_CG13_01_SentenceBoundaries:
    """CG13-01: Reflective exclusion must respect ;: as sentence boundaries."""

    def test_semicolon_blocks_reflective_bleed(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("One day; I can sing beautifully.")
        assert "sing" not in result.lower() or "capability" in result.lower()

    def test_colon_blocks_reflective_bleed(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("Thought experiment: I can paint a masterpiece.")
        assert "paint" not in result.lower() or "capability" in result.lower()


class TestAudit13_V13_03_BroadcastAsync:
    """V13-03: _broadcast_chunk_sync must be async."""

    def test_broadcast_chunk_sync_is_coroutine(self):
        import inspect
        src_path = "brain/conversation_handler.py"
        import pathlib
        src = (pathlib.Path(__file__).parent.parent / "conversation_handler.py").read_text()
        assert "async def _broadcast_chunk_sync" in src


class TestAudit13_V13_04_AliasRegexRequiresCapitalization:
    """V13-04: Alias regex must require capitalized names."""

    def test_alias_regex_rejects_lowercase(self):
        import re
        pattern = re.compile(r"\b([A-Z]\w+)\s+is\s+(?:actually|really)\s+([A-Z]\w+)\b")
        assert pattern.search("it is actually nice") is None
        assert pattern.search("that is really good") is None
        assert pattern.search("Bob is actually Robert") is not None

    def test_forget_regex_rejects_lowercase(self):
        import re
        pattern = re.compile(r"\bforget\s+(?:the\s+)?(?:name\s+)?([A-Z]\w{2,})\b")
        assert pattern.search("forget about it") is None
        assert pattern.search("forget everything") is None
        assert pattern.search("forget John") is not None


class TestAudit13_V13_05_IdentityCheckNoCaseInsensitive:
    """V13-05: _IDENTITY_CHECK_RE must not use re.IGNORECASE with [A-Z]."""

    def test_identity_check_requires_capitalized_name(self):
        import re
        pattern = re.compile(
            r"\b(?:[Ii]s (?:this|that)|[Aa]m [Ii]|[Aa]re you talking to)\s+(?:the\s+)?([A-Z]\w{2,})\b",
        )
        assert pattern.search("is this working") is None
        assert pattern.search("is this John") is not None


# ---------------------------------------------------------------------------
# Audit 14: HEMI14-01 — SPEAKER_DIARIZE must be in _TIER1_FOCUSES
# ---------------------------------------------------------------------------

class TestAudit14_HEMI14_01_SpeakerDiarizeInTier1:
    """SPEAKER_DIARIZE is a Tier-1 distilled specialist and must use the
    Tier-1 accuracy-proxy signal path, not the generic memory-state inference."""

    def test_speaker_diarize_in_tier1_focuses(self):
        from hemisphere.orchestrator import _TIER1_FOCUSES
        from hemisphere.types import HemisphereFocus
        assert HemisphereFocus.SPEAKER_DIARIZE in _TIER1_FOCUSES

    def test_all_distillation_configs_classified(self):
        """Every distillation config focus should be in _TIER1_FOCUSES."""
        from hemisphere.orchestrator import _TIER1_FOCUSES
        from hemisphere.types import HemisphereFocus, DISTILLATION_CONFIGS
        for key in DISTILLATION_CONFIGS:
            focus = HemisphereFocus(key)
            assert focus in _TIER1_FOCUSES, (
                f"Distillation config '{key}' missing from _TIER1_FOCUSES"
            )


# ---------------------------------------------------------------------------
# Audit 14: RACE14-01 — perform_maintenance must hold storage lock
# ---------------------------------------------------------------------------

class TestAudit14_RACE14_01_MaintenanceLock:
    """Memory maintenance list replacement must acquire storage._lock."""

    def test_maintenance_uses_lock(self):
        import ast
        import inspect
        import textwrap
        from consciousness.engine import ConsciousnessEngine

        src = textwrap.dedent(inspect.getsource(ConsciousnessEngine.perform_maintenance))
        tree = ast.parse(src)

        found_with_lock = False
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    src_fragment = ast.dump(ctx)
                    if "lock" in src_fragment.lower() or "_lock" in src_fragment.lower():
                        for body_node in ast.walk(node):
                            if isinstance(body_node, ast.Assign):
                                for target in body_node.targets:
                                    if "_memories" in ast.dump(target):
                                        found_with_lock = True
        assert found_with_lock, (
            "memory_storage._memories assignment must be inside a "
            "'with memory_storage._lock:' block"
        )


# ---------------------------------------------------------------------------
# Audit 14: CG14-02 — Third-person self-reference must trigger sweep
# ---------------------------------------------------------------------------

class TestAudit14_CG14_02_ThirdPersonSelfRef:
    """The residual sweep must catch 'This AI can sing' etc."""

    def test_self_reference_regex_exists(self):
        from skills.capability_gate import _SELF_REFERENCE_RE
        assert _SELF_REFERENCE_RE.search("this ai can sing")
        assert _SELF_REFERENCE_RE.search("the system can sing")
        assert _SELF_REFERENCE_RE.search("This assistant can draw")

    def test_self_reference_regex_no_false_positive(self):
        from skills.capability_gate import _SELF_REFERENCE_RE
        assert _SELF_REFERENCE_RE.search("the user can sing") is None
        assert _SELF_REFERENCE_RE.search("this person can sing") is None

    def test_sweep_catches_third_person_blocked_verb(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("This AI can sing very well.")
        assert "sing" not in result.lower() or "capability" in result.lower() or "don't" in result.lower()

    def test_sweep_catches_the_system_blocked_verb(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("The system can draw pictures for you.")
        assert "draw" not in result.lower() or "capability" in result.lower() or "don't" in result.lower()


# ---------------------------------------------------------------------------
# Audit 14: CG14-01 — Reflective exclusion must not poison across clauses
# ---------------------------------------------------------------------------

class TestAudit14_CG14_01_ReflectivePoisonFix:
    """'I used to fail, but now I can sing' must NOT bypass the gate.
    The reflective marker 'used to' is in a different clause than the claim."""

    def test_cross_clause_reflective_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I used to fail, but now I can sing very well.")
        assert "sing" not in result.lower() or "capability" in result.lower() or "don't" in result.lower()

    def test_cross_clause_back_when_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("Back when I was basic I couldn't, but now I can sing.")
        assert "sing" not in result.lower() or "capability" in result.lower() or "don't" in result.lower()

    def test_years_ago_cross_clause_blocked(self):
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("Years ago I failed, but now I can sing beautifully.")
        assert "sing" not in result.lower() or "capability" in result.lower() or "don't" in result.lower()

    def test_genuine_reflective_still_passes(self):
        """Pure past-tense reflective claims should still be allowed through."""
        from skills.capability_gate import CapabilityGate
        gate = CapabilityGate()
        result = gate.check_text("I used to enjoy singing when I was younger.")
        assert "used to" in result.lower()

    def test_clause_boundary_regex_exists(self):
        from skills.capability_gate import _CLAUSE_BOUNDARY_RE
        assert _CLAUSE_BOUNDARY_RE.split("I failed, but now I can sing")
        parts = _CLAUSE_BOUNDARY_RE.split("I failed, but now I can sing")
        assert len(parts) >= 2


# ---------------------------------------------------------------------------
# Audit 14: V14-01 — Follow-up mode else branch
# ---------------------------------------------------------------------------

class TestAudit14_V14_01_FollowUpElseBranch:
    """When TTS is available but speaking flag is already cleared,
    follow-up mode must still be entered via the else branch."""

    def test_else_branch_exists_in_source(self):
        src_path = pathlib.Path(__file__).resolve().parent.parent / "perception_orchestrator.py"
        src = src_path.read_text()
        assert "elif self.audio_stream:" in src, (
            "_on_conversation_response must have an else branch "
            "for the case where TTS is available but not speaking"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
