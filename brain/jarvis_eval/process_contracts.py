"""Process contract definitions for the Process Verification Layer.

Each contract declares what the architecture promises should happen at runtime.
Contracts are evaluated against live event/snapshot data by the ProcessVerifier.

Contract types:
  - event: Did event X fire at least once in the evaluation window?
  - snapshot: Is metric Y in snapshot source S non-zero / above threshold?
  - compound: Both an event condition AND a snapshot condition must hold.

Contracts carry mode prerequisites: a contract is only applicable when the
system is (or has been) in one of its required modes. Inapplicable contracts
are marked 'not_applicable' rather than 'fail'.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ContractStatus = Literal["pass", "fail", "not_applicable", "awaiting"]
VerifyMethod = Literal["event", "snapshot", "compound"]


@dataclass(frozen=True)
class ProcessContract:
    """A single architectural process contract."""

    contract_id: str
    group: str
    label: str
    method: VerifyMethod
    event_type: str | None = None
    snapshot_source: str | None = None
    snapshot_key: str | None = None
    snapshot_min: float = 0.0
    snapshot_max: float | None = None
    required_modes: frozenset[str] = field(default_factory=frozenset)
    excluded_modes: frozenset[str] = field(default_factory=frozenset)
    session_once: bool = False
    training_stage: int | None = None
    description: str = ""
    missing_event_status: ContractStatus = "fail"

    @property
    def playbook_day(self) -> int | None:
        """Backward-compatible alias for older callers/tests."""
        return self.training_stage


# ── Helpers ─────────────────────────────────────────────────────────

_ALL_ACTIVE = frozenset({"conversational", "reflective", "focused", "passive"})
_POST_GESTATION = frozenset({
    "conversational", "reflective", "focused", "passive",
    "sleep", "dreaming", "deep_learning",
})
_GESTATION = frozenset({"gestation"})
_ALL_MODES = frozenset({
    "gestation", "passive", "conversational", "reflective",
    "focused", "sleep", "dreaming", "deep_learning",
})
_BACKGROUND = frozenset({
    "passive", "conversational", "reflective", "focused",
    "deep_learning", "dreaming", "gestation",
})
_LEARNING = frozenset({"deep_learning", "dreaming", "gestation"})


def _evt(
    cid: str,
    group: str,
    label: str,
    event_type: str,
    *,
    modes: frozenset[str] = _ALL_MODES,
    excluded: frozenset[str] = frozenset(),
    once: bool = False,
    stage: int | None = None,
    desc: str = "",
    missing_event_status: ContractStatus = "fail",
) -> ProcessContract:
    return ProcessContract(
        contract_id=cid, group=group, label=label, method="event",
        event_type=event_type, required_modes=modes, excluded_modes=excluded,
        session_once=once, training_stage=stage, description=desc,
        missing_event_status=missing_event_status,
    )


def _snap(
    cid: str,
    group: str,
    label: str,
    source: str,
    key: str,
    *,
    min_val: float = 0.0,
    max_val: float | None = None,
    modes: frozenset[str] = _ALL_MODES,
    excluded: frozenset[str] = frozenset(),
    stage: int | None = None,
    desc: str = "",
) -> ProcessContract:
    return ProcessContract(
        contract_id=cid, group=group, label=label, method="snapshot",
        snapshot_source=source, snapshot_key=key, snapshot_min=min_val,
        snapshot_max=max_val,
        required_modes=modes, excluded_modes=excluded, training_stage=stage,
        description=desc,
    )


def _compound(
    cid: str,
    group: str,
    label: str,
    event_type: str,
    source: str,
    key: str,
    *,
    min_val: float = 0.0,
    modes: frozenset[str] = _ALL_MODES,
    excluded: frozenset[str] = frozenset(),
    stage: int | None = None,
    desc: str = "",
) -> ProcessContract:
    return ProcessContract(
        contract_id=cid, group=group, label=label, method="compound",
        event_type=event_type, snapshot_source=source, snapshot_key=key,
        snapshot_min=min_val, required_modes=modes, excluded_modes=excluded,
        training_stage=stage, description=desc,
    )


# ── Contract Registry ───────────────────────────────────────────────

ALL_CONTRACTS: list[ProcessContract] = []

# -- Voice Pipeline ------------------------------------------------
_G = "voice_pipeline"
ALL_CONTRACTS += [
    _evt("wake_word_detected", _G, "Wake Word Detected",
         "perception:wake_word", modes=_POST_GESTATION, stage=1,
         desc="openWakeWord fires at least once when user is present"),
    _evt("stt_transcribed", _G, "STT Transcription",
         "perception:transcription", modes=_POST_GESTATION, stage=1,
         desc="faster-whisper produced a transcription"),
    _evt("user_message_received", _G, "User Message Received",
         "conversation:user_message", modes=_POST_GESTATION, stage=1,
         desc="A user message was dispatched to conversation handler"),
    _evt("conversation_completed", _G, "Conversation Completed",
         "conversation:response", modes=_POST_GESTATION, stage=1,
         desc="Brain generated and sent a response"),
]

# -- Identity Pipeline ---------------------------------------------
_G = "identity_pipeline"
ALL_CONTRACTS += [
    _evt("speaker_identified", _G, "Speaker Identified",
         "perception:speaker_identified", modes=_POST_GESTATION, stage=1,
         desc="ECAPA-TDNN speaker ID ran after STT"),
    _evt("face_identified", _G, "Face Identified",
         "perception:face_identified", modes=_POST_GESTATION, stage=1,
         desc="MobileFaceNet face ID matched a crop"),
    _evt("identity_fused", _G, "Identity Fused",
         "perception:identity_resolved", modes=_POST_GESTATION, stage=1,
         desc="IdentityFusion produced a canonical identity"),
    _evt("identity_scoped", _G, "Identity Scoped",
         "identity:scope_assigned", modes=_POST_GESTATION, stage=3,
         desc="IdentityBoundaryEngine assigned scope to a memory"),
]

# -- Memory Pipeline -----------------------------------------------
_G = "memory_pipeline"
ALL_CONTRACTS += [
    _evt("memory_written", _G, "Memory Written",
         "memory:write", stage=2,
         desc="At least one memory was created"),
    _evt("memory_associated", _G, "Memory Associated",
         "memory:associated", modes=_BACKGROUND, stage=6,
         desc="Two memories were linked via association"),
    _snap("cortex_ranker_data", _G, "Cortex Ranker Has Data",
          "cortex", "ranker.train_count", min_val=1.0, modes=_LEARNING, stage=6,
          desc="Memory ranker has completed at least one training run"),
    _snap("salience_model_data", _G, "Salience Model Has Data",
          "cortex", "salience.train_count", min_val=1.0, modes=_LEARNING, stage=6,
          desc="Salience model has completed at least one training run"),
    _snap("memory_count_growing", _G, "Memory Count > 0",
          "memory", "total", min_val=1.0, stage=2,
          desc="Memory storage has at least one memory"),
]

# -- Study Pipeline ------------------------------------------------
_G = "study_pipeline"
ALL_CONTRACTS += [
    _snap("source_studied", _G, "Sources Studied",
          "library", "studied", min_val=1.0,
          desc="At least one library source has been studied"),
    _snap("llm_extraction_used", _G, "LLM Extraction Used",
          "study_telemetry", "cumulative_studied", min_val=1.0,
          desc="Study pipeline has studied at least one source (LLM or regex extraction)"),
    _snap("claims_extracted", _G, "Claims Extracted",
          "study_telemetry", "cumulative_studied", min_val=1.0,
          desc="Study pipeline has completed extraction on at least one source"),
    _snap("source_ingested", _G, "Sources Ingested",
          "library", "total", min_val=1.0,
          desc="At least one source exists in the library"),
]

# -- Epistemic System ----------------------------------------------
_G = "epistemic_system"
ALL_CONTRACTS += [
    _evt("contradiction_scanned", _G, "Contradiction Scan",
         "contradiction:detected", modes=_BACKGROUND,
         missing_event_status="awaiting",
         desc="Layer 5 detected at least one contradiction"),
    _evt("calibration_ticked", _G, "Calibration Ticked",
         "calibration:updated", modes=_BACKGROUND, stage=5,
         desc="Layer 6 truth calibration ran a tick"),
    _evt("quarantine_ticked", _G, "Quarantine Ticked",
         "quarantine:tick_complete", modes=_ALL_MODES,
         desc="Layer 8 quarantine scorer completed a tick"),
    _evt("audit_completed", _G, "Audit Completed",
         "audit:completed", modes=_ALL_MODES,
         desc="Layer 9 reflective audit produced a report"),
    _evt("integrity_computed", _G, "Integrity Computed",
         "soul_integrity:updated", modes=_ALL_MODES, stage=7,
         desc="Layer 10 soul integrity index was computed"),
    _evt("graph_edge_created", _G, "Belief Graph Edge",
         "belief_graph:edge_created", modes=_BACKGROUND,
         desc="Layer 7 belief graph created an evidence edge"),
    _evt("prediction_validated", _G, "Prediction Validated",
         "calibration:prediction_validated", modes=_BACKGROUND,
         missing_event_status="awaiting",
         desc="A typed prediction was validated against reality"),
]

# -- Hemisphere / Distillation -------------------------------------
_G = "hemisphere_distillation"
ALL_CONTRACTS += [
    _evt("hemisphere_trained", _G, "Hemisphere Trained",
         "hemisphere:training_progress", modes=_BACKGROUND,
         desc="A hemisphere NN completed training progress"),
    _snap("hemisphere_ready", _G, "Hemisphere Ready",
          "hemisphere", "total_networks", min_val=1.0, modes=_BACKGROUND,
          desc="At least one hemisphere NN is active and ready for inference"),
    _evt("distillation_stats", _G, "Distillation Stats",
         "hemisphere:distillation_stats", modes=_BACKGROUND,
         desc="Distillation cycle completed with stats"),
    _snap("distillation_signals", _G, "Distillation Signals Collected",
          "hemisphere", "distillation.total_signals", min_val=1.0, modes=_BACKGROUND,
          desc="DistillationCollector has captured at least one teacher signal"),
    _snap("broadcast_slots_filled", _G, "Broadcast Slots Filled",
          "hemisphere", "broadcast_slots_count", min_val=1.0, modes=_BACKGROUND,
          desc="At least one hemisphere signal occupies a broadcast slot"),
]

# -- Policy Pipeline -----------------------------------------------
_G = "policy_pipeline"
ALL_CONTRACTS += [
    _snap("policy_decisions", _G, "Policy Decisions Made",
          "policy_telemetry", "decisions_total", min_val=1.0,
          modes=_BACKGROUND, stage=4,
          desc="Neural policy made at least one decision"),
    _snap("shadow_ab_evaluated", _G, "Shadow A/B Evaluated",
          "policy_telemetry", "shadow_ab_total", min_val=1.0,
          modes=_BACKGROUND, stage=4,
          desc="Shadow A/B evaluation has run at least once"),
    _snap("experience_logged", _G, "Experience Logged",
          "policy_telemetry", "train_runs_total", min_val=1.0,
          modes=_BACKGROUND,
          desc="Policy trainer has run at least once"),
]

# -- Autonomy Pipeline ---------------------------------------------
_G = "autonomy_pipeline"
ALL_CONTRACTS += [
    _evt("intent_queued", _G, "Intent Queued",
         "autonomy:intent_queued", modes=_BACKGROUND, stage=7,
         desc="An autonomy research intent was queued"),
    _evt("research_started", _G, "Research Started",
         "autonomy:research_started", modes=_BACKGROUND, stage=7,
         desc="Autonomy research execution began"),
    _evt("research_completed", _G, "Research Completed",
         "autonomy:research_completed", modes=_BACKGROUND, stage=7,
         desc="Autonomy research completed successfully"),
    _evt("delta_measured", _G, "Delta Measured",
         "autonomy:delta_measured", modes=_BACKGROUND, stage=7,
         desc="DeltaTracker measured before/after attribution"),
]

# -- Gestation -----------------------------------------------------
_G = "gestation"
ALL_CONTRACTS += [
    _evt("gestation_started", _G, "Gestation Started",
         "gestation:started", modes=_GESTATION, once=True,
         desc="Birth protocol initiated for a fresh brain"),
    _evt("gestation_phase_advanced", _G, "Phase Advanced",
         "gestation:phase_advanced", modes=_GESTATION,
         desc="Gestation advanced to a new phase"),
    _evt("gestation_directive_completed", _G, "Directive Completed",
         "gestation:directive_completed", modes=_GESTATION,
         desc="A gestation directive was completed"),
    _evt("gestation_readiness_updated", _G, "Readiness Updated",
         "gestation:readiness_update", modes=_GESTATION,
         desc="Gestation readiness scores were recomputed"),
    _evt("gestation_graduated", _G, "Gestation Graduated",
         "gestation:complete", modes=_GESTATION, once=True,
         desc="Gestation completed — brain graduated"),
]

# -- Skill Learning ------------------------------------------------
_G = "skill_learning"
ALL_CONTRACTS += [
    _snap("skill_registered", _G, "Skill Registered",
          "skills", "total", min_val=1.0, modes=_BACKGROUND,
          desc="At least one skill exists in persistent Skill Registry storage"),
    _snap("learning_job_started", _G, "Learning Job Started",
          "learning_jobs", "total_count", min_val=1.0, modes=_BACKGROUND,
          desc="At least one learning job exists in persistent Learning Job storage"),
    _snap("job_phase_advanced", _G, "Job Phase Advanced",
          "learning_jobs", "phase_transition_count", min_val=1.0, modes=_BACKGROUND,
          desc="At least one learning job recorded a persisted phase transition"),
    _snap("skill_learning_completed", _G, "Learning Completed",
          "learning_jobs", "completed_count", min_val=1.0, modes=_BACKGROUND,
          desc="At least one learning job completed a full lifecycle"),
]

# -- Mutation Pipeline ---------------------------------------------
_G = "mutation_pipeline"
ALL_CONTRACTS += [
    _evt("mutation_proposed", _G, "Mutation Proposed",
         "consciousness:mutation_proposed", modes=_BACKGROUND,
         excluded=frozenset({"sleep"}),
         desc="MutationProposer generated a mutation proposal"),
    _evt("mutation_governed", _G, "Mutation Governed",
         "mutation:applied", modes=_BACKGROUND,
         excluded=frozenset({"sleep"}),
         desc="MutationGovernor applied a mutation"),
]

# -- Consciousness Tick --------------------------------------------
_G = "consciousness_tick"
ALL_CONTRACTS += [
    _evt("mode_managed", _G, "Mode Managed",
         "mode:change",
         desc="ModeManager transitioned to a new mode"),
    _evt("meta_thoughts_generated", _G, "Meta Thoughts",
         "meta:thought_generated", modes=_BACKGROUND,
         desc="MetaCognitiveThoughts generated a thought"),
    _snap("evolution_analyzed", _G, "Evolution Analyzed",
          "consciousness", "emergent_behavior_count", min_val=1.0, modes=_BACKGROUND,
          desc="Consciousness evolution subsystem has produced persisted emergent analysis"),
    _snap("consciousness_stage", _G, "Consciousness Active",
          "consciousness", "stage",
          desc="Consciousness system reports a stage"),
    _evt("consciousness_analysis_ran", _G, "Analysis Ran",
         "consciousness:analysis",
         desc="Consciousness analysis event was emitted"),
]

# -- Capability Gate -----------------------------------------------
_G = "capability_gate"
ALL_CONTRACTS += [
    _snap("gate_claims_checked", _G, "Claims Checked",
          "capability_gate", "claims_passed", modes=_POST_GESTATION, stage=2,
          desc="CapabilityGate has checked at least one claim"),
    _snap("gate_stats_nonzero", _G, "Gate Active",
          "capability_gate", "claims_blocked",
          desc="CapabilityGate has blocked at least one claim"),
]

# -- World Model ---------------------------------------------------
_G = "world_model"
ALL_CONTRACTS += [
    _evt("world_model_ticked", _G, "World Model Ticked",
         "world_model:update", modes=_BACKGROUND,
         desc="WorldModel ran a tick and updated state"),
    _evt("world_model_prediction_validated", _G, "Prediction Validated",
         "world_model:prediction_validated", modes=_BACKGROUND,
         desc="A world model prediction was validated"),
]


# -- Mental Simulator (Phase 3) ------------------------------------
_G = "mental_simulator"
ALL_CONTRACTS += [
    _snap("simulator_running", _G, "Simulator Running",
          "simulator_promotion", "total_validated", min_val=1.0, modes=_BACKGROUND,
          desc="Mental simulator has validated at least 1 trace"),
    _snap("simulator_traces_10", _G, "Simulator Traces >= 10",
          "simulator_promotion", "total_validated", min_val=10.0, modes=_BACKGROUND,
          desc="Mental simulator has validated >= 10 traces"),
    _snap("simulator_accuracy_healthy", _G, "Simulator Accuracy >= 70%",
          "simulator_promotion", "rolling_accuracy", min_val=0.7, modes=_BACKGROUND,
          desc="Simulator rolling accuracy is >= 70%"),
]

# -- Curiosity Bridge (Phase 1) ------------------------------------
_G = "curiosity_bridge"
ALL_CONTRACTS += [
    _evt("curiosity_question_generated", _G, "Question Generated",
         "curiosity:question_generated", modes=_POST_GESTATION,
         desc="A curiosity question was generated from subsystem state"),
    _evt("curiosity_question_asked", _G, "Question Asked",
         "curiosity:question_asked", modes=_POST_GESTATION,
         desc="A curiosity question was spoken to the user"),
    _evt("curiosity_answer_processed", _G, "Answer Processed",
         "curiosity:answer_processed", modes=_POST_GESTATION,
         desc="User's answer to a curiosity question was routed back"),
]

# -- Roadmap Maturity Gates ----------------------------------------
_G = "roadmap_maturity"
ALL_CONTRACTS += [
    # Phase 1 unlock gates
    _snap("maturity_identity_enrolled", _G, "Identity Enrolled (Phase 1)",
          "speakers", "enrolled_count", min_val=1.0, modes=_POST_GESTATION,
          desc="At least 1 speaker profile enrolled (Phase 1: identity curiosity unlock)"),
    _snap("maturity_scene_observations", _G, "Scene Observations >= 50 (Phase 1)",
          "scene", "update_count", min_val=50.0, modes=_POST_GESTATION,
          desc="Scene tracker has >= 50 observation cycles (Phase 1: scene curiosity unlock)"),
    _snap("maturity_research_completed_20", _G, "Research Episodes >= 20 (Phase 1)",
          "autonomy", "completed_total", min_val=20.0, modes=_BACKGROUND,
          desc="Autonomy completed >= 20 research episodes (Phase 1: research curiosity unlock)"),
    _snap("maturity_world_model_level_1", _G, "World Model Level >= 1 (Phase 1/3)",
          "world_model_promotion", "level", min_val=1.0, modes=_BACKGROUND,
          desc="World model promoted to Level 1+ advisory (Phase 1: world curiosity unlock, Phase 3 prerequisite)"),

    # Phase 3 gates
    _snap("maturity_wm_predictions_50", _G, "WM Predictions >= 50 (Phase 3)",
          "world_model_promotion", "total_validated", min_val=50.0, modes=_BACKGROUND,
          desc="World model has >= 50 validated predictions (Phase 3: mental simulator prerequisite)"),
    _snap("maturity_simulator_validated_100", _G, "Simulator Validated >= 100 (Phase 3)",
          "simulator_promotion", "total_validated", min_val=100.0, modes=_BACKGROUND,
          desc="Mental simulator has >= 100 validated traces (Phase 3: advisory promotion prerequisite)"),

    # Phase 5 gates
    _snap("maturity_autonomy_wins_10", _G, "Autonomy Wins >= 10 (Phase 5)",
          "autonomy", "total_wins", min_val=10.0, modes=_BACKGROUND,
          desc="Autonomy has >= 10 positive attributions (Phase 5: L2 execution prerequisite)"),

    # Phase 8 gates
    _snap("maturity_dream_artifacts_reviewed", _G, "Dream Artifacts Reviewed (Phase 8)",
          "dream_artifacts", "buffer.total_created", min_val=500.0, modes=_LEARNING,
          desc="500+ dream artifacts created (Phase 8: dream observer NN prerequisite)"),
    _snap("maturity_dream_promoted_100", _G, "Dream Promoted >= 100 (Phase 8)",
          "dream_artifacts", "buffer.total_promoted", min_val=100.0, modes=_LEARNING,
          desc="100+ dream artifacts promoted (Phase 8: dream observer NN prerequisite)"),

    # Phase 9 gates
    _snap("maturity_conversation_outcomes_200", _G, "Conversation Outcomes >= 200 (Phase 9)",
          "policy_telemetry", "decisions_total", min_val=200.0, modes=_BACKGROUND,
          desc="200+ policy decisions with outcomes (Phase 9: counterfactual engine prerequisite)"),
    _snap("maturity_experience_buffer_500", _G, "Experience Buffer >= 500 (Phase 9)",
          "experience_buffer", "size", min_val=500.0, modes=_BACKGROUND,
          desc="Policy experience buffer has >= 500 entries (Phase 9: counterfactual engine prerequisite)"),

    # Cross-phase health gates
    _snap("maturity_face_enrolled", _G, "Face Profile Enrolled",
          "faces", "enrolled_count", min_val=1.0, modes=_POST_GESTATION,
          desc="At least 1 face profile enrolled (identity fusion readiness)"),
    _snap("maturity_hemisphere_networks", _G, "Hemisphere NNs Active",
          "hemisphere", "total_networks", min_val=1.0, modes=_BACKGROUND,
          desc="At least 1 hemisphere neural network is active"),
    _snap("maturity_autonomy_level_2", _G, "Autonomy Level >= 2 (Phase 5/6)",
          "autonomy", "autonomy_level", min_val=2.0, modes=_BACKGROUND,
          desc="Autonomy promoted to Level 2+ (Phase 5: L2 execution active)"),
]

# -- Quality Baselines (value-range contracts) -------------------------
_G = "quality_baselines"
ALL_CONTRACTS += [
    _snap("quality_soul_integrity", _G, "Soul Integrity >= Yellow",
          "soul_integrity", "current_index", min_val=0.50, modes=_ALL_MODES,
          desc="Soul integrity index is above yellow threshold (>= 0.50)"),
    _snap("quality_contradiction_debt", _G, "Contradiction Debt <= Yellow",
          "contradiction", "contradiction_debt", max_val=0.15, modes=_BACKGROUND,
          desc="Contradiction debt is within yellow threshold (<= 0.15)"),
    _snap("quality_quarantine_pressure", _G, "Quarantine Pressure <= Yellow",
          "quarantine", "composite", max_val=0.40, modes=_ALL_MODES,
          desc="Quarantine pressure is within yellow threshold (<= 0.40)"),
    _snap("quality_audit_score", _G, "Audit Score >= Yellow",
          "reflective_audit", "latest_score", min_val=0.45, modes=_ALL_MODES,
          desc="Reflective audit score is above yellow threshold (>= 0.45)"),
    _snap("quality_memory_weight", _G, "Avg Memory Weight >= Yellow",
          "memory", "avg_weight", min_val=0.40, modes=_ALL_MODES,
          desc="Average memory weight is above yellow threshold (>= 0.40)"),
    _snap("quality_dream_promotion", _G, "Dream Promotion Rate <= Yellow",
          "dream_artifacts", "promotion_rate", max_val=0.60, modes=_LEARNING,
          desc="Dream promotion rate is within yellow threshold (<= 0.60)"),
    _snap("quality_library_content", _G, "Substantive Content >= Yellow",
          "library", "substantive_ratio", min_val=0.30, modes=_ALL_MODES,
          desc="Library substantive content ratio is above yellow threshold (>= 0.30)"),
]

# -- NN Quality (training health contracts) ----------------------------
_G = "nn_quality"
ALL_CONTRACTS += [
    _snap("quality_hemisphere_loss_valid", _G, "Hemisphere Losses Valid",
          "hemisphere", "all_losses_valid", min_val=1.0, modes=_BACKGROUND,
          desc="No hemisphere NN has NaN or infinite loss"),
    _snap("quality_policy_reward_positive", _G, "Policy Avg Reward > 0",
          "experience_buffer", "avg_reward", min_val=0.01, modes=_BACKGROUND,
          desc="Policy experience buffer average reward is positive"),
    _snap("quality_cortex_ranker_enabled", _G, "Cortex Ranker Not Disabled",
          "cortex", "ranker.enabled", min_val=1.0, modes=_LEARNING,
          desc="Memory ranker has not been auto-disabled by flap guard"),
    _snap("quality_autonomy_win_rate", _G, "Autonomy Win Rate >= 20%",
          "autonomy", "overall_win_rate", min_val=0.20, modes=_BACKGROUND,
          desc="Autonomy research win rate is above 20% (when sufficient outcomes exist)"),
]

# -- Matrix Protocol -------------------------------------------------------
_G = "matrix_protocol"
ALL_CONTRACTS += [
    _evt("matrix_dl_requested", _G, "Deep Learning Mode Requested",
         "matrix:deep_learning_requested", modes=_POST_GESTATION,
         missing_event_status="awaiting",
         desc="A Matrix Protocol learning request triggered deep_learning mode"),
    _evt("matrix_expansion_triggered", _G, "Expansion Triggered",
         "matrix:expansion_triggered", modes=_POST_GESTATION,
         missing_event_status="awaiting",
         desc="M6 broadcast/policy expansion was triggered by promoted specialists"),
    _snap("matrix_active_jobs", _G, "Matrix Jobs Observed",
          "matrix", "matrix_jobs_observed", min_val=1.0, modes=_POST_GESTATION,
          desc="At least one Matrix Protocol job is active or has completed"),
    _snap("matrix_specialists_exist", _G, "Specialists Born",
          "matrix", "specialist_count", min_val=1.0, modes=frozenset({"deep_learning"}),
          desc="At least one specialist NN has been born via Matrix Protocol"),
]

# -- System upgrades (self-improvement truth lane) ---------------------------
_G = "system_upgrades"
ALL_CONTRACTS += [
    _evt("si_improvement_started", _G, "Improvement Started",
         "improvement:started", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Self-improvement pipeline accepted a tracked upgrade intent"),
    _evt("si_sandbox_passed", _G, "Sandbox Passed",
         "improvement:sandbox_passed", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Sandbox validation completed successfully for a patch"),
    _evt("si_sandbox_failed_observed", _G, "Sandbox Failure Observable",
         "improvement:sandbox_failed", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Sandbox or generation failure was recorded (anti silent-success lane)"),
    _evt("si_dry_run_persisted", _G, "Dry Run Persisted",
         "improvement:dry_run", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Dry-run completion was recorded to proposals/history"),
    _evt("si_needs_approval_when_gated", _G, "Approval Gate Emitted",
         "improvement:needs_approval", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Human-approval stage emitted a needs-approval event when required"),
    _evt("si_promoted", _G, "Improvement Promoted",
         "improvement:promoted", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="A patch was promoted to disk after sandbox + gates"),
    _compound(
        "si_structured_report_with_sandbox_pass",
        _G, "Structured Report When Sandbox Passes",
        "improvement:sandbox_passed", "system_upgrades", "upgrade_reports_total",
        min_val=1.0, modes=_ALL_MODES,
        desc="Sandbox-pass event is paired with at least one on-disk structured report",
    ),
    _evt("si_post_restart_verified", _G, "Post-Restart Verification Recorded",
         "improvement:post_restart_verified", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="Post-restart verification outcome was stamped after pending verification"),
    _evt("si_rollback_observed", _G, "Rollback Observed",
         "improvement:rolled_back", modes=_ALL_MODES,
         missing_event_status="awaiting",
         desc="A rollback or regression-driven rollback event was observed"),
    _snap("si_reports_dir_ready", _G, "Upgrade Reports Dir Ready",
          "system_upgrades", "truth_lane_ready", min_val=1.0, modes=_ALL_MODES,
          desc="Structured upgrade report storage path is present"),
]

# -- Language Eval Gates (Phase D) -----------------------------------------
_G = "language_eval"
ALL_CONTRACTS += [
    _snap("lang_corpus_volume", _G, "Language Corpus >= 30 Examples",
          "language", "total_examples", min_val=30.0, modes=_POST_GESTATION,
          desc="Language corpus has enough examples for eval (>= 30)"),
    _snap("lang_native_usage_healthy", _G, "Native Usage >= 70%",
          "language", "native_usage_rate", min_val=0.70, modes=_POST_GESTATION,
          desc="Bounded articulator used >= 70% when eligible"),
    _snap("lang_fail_closed_low", _G, "Fail-Closed Rate <= 25%",
          "language", "fail_closed_rate", max_val=0.25, modes=_POST_GESTATION,
          desc="Fail-closed rate is within acceptable range (<= 25%)"),
    _snap("lang_provenance_coverage", _G, "Provenance Coverage >= 90%",
          "language", "provenance_coverage", min_val=0.90, modes=_POST_GESTATION,
          desc="At least 90% of corpus examples have grounded provenance verdicts"),
    _snap("lang_gate_not_red", _G, "Gate Color Non-Red",
          "language", "gate_color_code", min_val=1.0, modes=_POST_GESTATION,
          desc="Composite language gate is not red (yellow/green)"),
    _snap("lang_promotion_evals_recorded", _G, "Promotion Evals Recorded",
          "language", "promotion_total_evaluations", min_val=7.0, modes=_POST_GESTATION,
          desc="Promotion governor persisted at least one full class evaluation pass"),
    _snap("lang_red_pressure_controlled", _G, "Red Gate Pressure Controlled",
          "language", "promotion_red_quality_classes", max_val=2.0, modes=_POST_GESTATION,
          desc="At most two response classes are quality-risk red (sample-limited reds tracked separately)"),
    _snap("lang_runtime_unpromoted_live_zero", _G, "Runtime Unpromoted Live Attempts = 0",
          "language", "runtime_unpromoted_live_attempts", max_val=0.0, modes=_POST_GESTATION,
          desc="Runtime guard prevented unpromoted classes from consuming live native output"),
    _snap("lang_runtime_live_red_zero", _G, "Runtime Live Red Classes = 0",
          "language", "runtime_live_red_classes", max_val=0.0, modes=_POST_GESTATION,
          desc="No class with red gate diagnostics consumed live native output"),
]

# -- Intention Truth Layer (Stage 0) ---------------------------------------
_G = "intention_truth"
ALL_CONTRACTS += [
    _snap("intention_registry_loaded", _G, "Intention Registry Loaded",
          "intentions", "loaded", min_val=1.0, modes=_ALL_MODES,
          desc="Intention registry was loaded (or initialized) on boot"),
    _snap("intention_no_error_accumulation", _G, "Registry Errors Bounded",
          "intentions", "errors", max_val=5.0, modes=_ALL_MODES,
          desc="Intention registry has not accumulated excessive persistence errors"),
    _snap("intention_no_chronic_stale_backlog", _G, "Stale 7d Backlog Bounded",
          "intentions", "stale_7d", max_val=25.0, modes=_POST_GESTATION,
          desc="Stale intentions in the last 7 days are within the bounded housekeeping envelope"),
    _snap("intention_graduation_readiness_reported", _G,
          "Stage-1 Graduation Readiness Reported",
          "intentions", "graduation_gates_reported",
          min_val=1.0, modes=_ALL_MODES,
          desc=("IntentionRegistry.get_graduation_status() produced a "
                "non-empty gate checklist. Observability only — PVL never "
                "gates runtime behavior. See docs/INTENTION_STAGE_1_DESIGN.md")),
]

# ── Group metadata ──────────────────────────────────────────────────

PROCESS_GROUPS: dict[str, dict[str, Any]] = {
    "voice_pipeline":          {"label": "Voice Pipeline",           "order": 0},
    "identity_pipeline":       {"label": "Identity Pipeline",        "order": 1},
    "memory_pipeline":         {"label": "Memory Pipeline",          "order": 2},
    "study_pipeline":          {"label": "Study Pipeline",           "order": 3},
    "epistemic_system":        {"label": "Epistemic System",         "order": 4},
    "hemisphere_distillation": {"label": "Hemisphere / Distillation","order": 5},
    "policy_pipeline":         {"label": "Policy Pipeline",          "order": 6},
    "autonomy_pipeline":       {"label": "Autonomy Pipeline",        "order": 7},
    "gestation":               {"label": "Gestation",                "order": 8},
    "skill_learning":          {"label": "Skill Learning",           "order": 9},
    "mutation_pipeline":       {"label": "Mutation Pipeline",        "order": 10},
    "consciousness_tick":      {"label": "Consciousness Tick",       "order": 11},
    "capability_gate":         {"label": "Capability Gate",          "order": 12},
    "world_model":             {"label": "World Model",              "order": 13},
    "mental_simulator":        {"label": "Mental Simulator",         "order": 14},
    "curiosity_bridge":        {"label": "Curiosity Bridge",          "order": 15},
    "roadmap_maturity":        {"label": "Roadmap Maturity Gates",   "order": 16},
    "quality_baselines":       {"label": "Quality Baselines",        "order": 17},
    "nn_quality":              {"label": "NN Quality",               "order": 18},
    "matrix_protocol":         {"label": "Matrix Protocol",          "order": 19},
    "language_eval":            {"label": "Language Eval Gates",      "order": 20},
    "system_upgrades":          {"label": "System Upgrades",          "order": 21},
    "intention_truth":          {"label": "Intention Truth (Stage 0)", "order": 22},
}

TRAINING_STAGE_MAP: dict[int, list[str]] = {
    1: ["voice_pipeline", "identity_pipeline"],
    2: ["memory_pipeline", "capability_gate"],
    3: ["identity_pipeline"],
    4: ["policy_pipeline", "consciousness_tick"],
    5: ["epistemic_system"],
    6: ["memory_pipeline"],
    7: ["autonomy_pipeline", "epistemic_system"],
}

PLAYBOOK_DAY_MAP = TRAINING_STAGE_MAP


def get_contracts_by_group() -> dict[str, list[ProcessContract]]:
    """Return contracts organized by group."""
    groups: dict[str, list[ProcessContract]] = {}
    for c in ALL_CONTRACTS:
        groups.setdefault(c.group, []).append(c)
    return groups


def get_contracts_for_training_stage(stage: int) -> list[ProcessContract]:
    """Return contracts relevant to a companion-training stage."""
    target_groups = TRAINING_STAGE_MAP.get(stage, [])
    return [c for c in ALL_CONTRACTS if c.group in target_groups]


def get_contracts_for_playbook_day(day: int) -> list[ProcessContract]:
    """Backward-compatible wrapper for older callers."""
    return get_contracts_for_training_stage(day)
