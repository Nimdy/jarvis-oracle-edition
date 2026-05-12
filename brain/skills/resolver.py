"""Skill Resolver — classifies user requests into capability types.

Given a user request, returns a ``SkillResolution`` that tells the system:
- What skill_id this maps to
- What capability type (procedural / perceptual / control)
- What risk level
- What evidence is required for verification
- What hard safety gates apply
- What structured capability contract the skill must satisfy

v1 uses regex templates.  Later versions can upgrade to a small classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


CapabilityType = str  # "procedural" | "perceptual" | "control"
RiskLevel = str       # "low" | "medium" | "high"


@dataclass(frozen=True)
class StructuredCapability:
    """Machine-readable contract for what a skill must do."""
    input_type: str
    output_type: str
    success_metrics: tuple[str, ...] = ()
    evidence_requirements: tuple[str, ...] = ()
    hardware_requirements: tuple[str, ...] = ()
    execution_contract_id: str = ""
    required_executor_kind: str = ""
    acquisition_eligible: bool = False


@dataclass(frozen=True)
class SkillResolution:
    skill_id: str
    name: str
    capability_type: CapabilityType
    risk_level: RiskLevel
    required_evidence: list[str]
    hard_gates: list[dict[str, str]] = field(default_factory=list)
    default_phases: list[dict[str, list[str]]] = field(default_factory=list)
    guided_collect: dict[str, Any] | None = None
    notes: str = ""
    capability: StructuredCapability | None = None


def is_generic_fallback_resolution(resolution: SkillResolution) -> bool:
    """Return True when resolution came from the generic procedural fallback."""
    return (
        resolution.capability_type == "procedural"
        and resolution.required_evidence == ["test:procedure_smoke"]
        and resolution.notes.startswith("Auto-generated from:")
    )


# ── Stop words for dynamic skill ID generation ──────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "to", "for", "and", "or", "in", "on", "of", "with",
    "that", "this", "can", "could", "would", "should", "will", "may", "might",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "not", "but", "if", "its", "it", "my", "your",
    "their", "our", "from", "by", "at", "into", "about", "how", "what",
    "when", "where", "which", "who", "whom", "why", "so", "up", "out",
    "just", "also", "very", "too", "more", "some", "any", "all", "each",
    "both", "few", "most", "other", "than", "then", "now", "here", "there",
    "you", "me", "him", "her", "them", "us",
    # Conversational filler that should not shape fallback skill IDs
    "hey", "okay", "able", "like", "please", "doing", "fine",
    "make", "were", "youre", "dont", "have", "dont", "need",
    # Stance / emotional words that produce garbage skill IDs
    "better", "worse", "serve", "operate", "shadows", "honest",
    "trust", "lying", "useless", "pointless", "waste", "terrible", "mad",
})


def _generate_skill_id(text: str) -> str:
    """Generate a descriptive skill ID from user text."""
    words = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
    words = [w for w in words if len(w) > 2 and w not in _STOP_WORDS][:4]
    if not words:
        return "unresolved_procedure_v1"
    return "_".join(words) + "_v1"


def _generate_skill_name(text: str) -> str:
    """Generate a human-readable skill name from user text."""
    words = re.sub(r'[^a-z0-9\s]', '', text.lower()).split()
    words = [w for w in words if len(w) > 2 and w not in _STOP_WORDS][:4]
    if not words:
        return "Unresolved Procedure"
    return " ".join(w.capitalize() for w in words)


# ── Skill templates (extend over time) ──────────────────────────────────────

SKILL_TEMPLATES: list[tuple[re.Pattern[str], SkillResolution]] = [
    # ── Speaker Diarization / Source Separation ──────────────────────────
    (
        re.compile(
            r"\b(diariz"
            r"|separate.{0,20}(?:voice|speaker|people)"
            r"|tell.{0,20}(?:voice|speaker).{0,20}apart"
            r"|(?:voice|speaker).{0,10}apart"
            r"|who.{0,15}speaking.{0,10}when"
            r"|speaker.{0,10}segmentation"
            r"|source.{0,10}separation"
            r"|isolate.{0,20}(?:voice|speaker|waveform)"
            r"|separate.{0,20}(?:waveform|audio))",
            re.I,
        ),
        SkillResolution(
            skill_id="speaker_diarization_v1",
            name="Speaker Diarization",
            capability_type="perceptual",
            risk_level="low",
            required_evidence=[
                "test:diarization_der_below_threshold",
                "test:known_speaker_match_accuracy",
                "test:turn_boundary_f1",
            ],
            capability=StructuredCapability(
                input_type="mixed_speech_audio",
                output_type="speaker_segments_with_timestamps",
                success_metrics=(
                    "diarization_error_rate",
                    "jaccard_error_rate",
                    "turn_boundary_f1",
                    "known_speaker_match_accuracy",
                ),
                evidence_requirements=(
                    "labeled_multi_speaker_clips >= 20",
                    "baseline_comparison_run",
                    "held_out_household_audio_test",
                ),
                hardware_requirements=("gpu_cuda",),
            ),
            hard_gates=[
                {"id": "gate:speaker_profiles_exist", "kind": "data",
                 "required": "true", "state": "unknown",
                 "details": "At least 2 enrolled speaker profiles required."},
                {"id": "gate:audio_capture_active", "kind": "resource",
                 "required": "true", "state": "unknown",
                 "details": "Pi audio stream must be active for data collection."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": [
                    "gate:speaker_profiles_exist", "gate:audio_capture_active"]},
                {"name": "collect", "exit_conditions": ["metric:labeled_segments>=50"]},
                {"name": "train", "exit_conditions": ["artifact:diarization_model_checkpoint"]},
                {"name": "verify", "exit_conditions": [
                    "evidence:test:diarization_der_below_threshold",
                    "evidence:test:known_speaker_match_accuracy"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Windowed ECAPA-TDNN embeddings + clustering. "
                  "Teacher: full SpeakerIdentifier. "
                  "Student: lightweight diarization specialist.",
        ),
    ),

    # ── Audio Analysis / Feature Extraction ──────────────────────────────
    (
        re.compile(
            r"\b(audio.{0,15}analysis"
            r"|sound.{0,15}classification"
            r"|acoustic.{0,15}scene"
            r"|audio.{0,15}separation"
            r"|classify.{0,15}audio"
            r"|audio.{0,15}feature)\b",
            re.I,
        ),
        SkillResolution(
            skill_id="audio_analysis_v1",
            name="Audio Analysis",
            capability_type="perceptual",
            risk_level="low",
            required_evidence=["test:audio_analysis_accuracy"],
            capability=StructuredCapability(
                input_type="raw_audio_pcm",
                output_type="audio_analysis_report",
                success_metrics=("classification_accuracy", "feature_quality"),
                evidence_requirements=(
                    "labeled_audio_samples >= 30",
                    "baseline_comparison",
                ),
            ),
            hard_gates=[
                {"id": "gate:teacher_signals_present", "kind": "data",
                 "required": "true", "state": "unknown",
                 "details": "Teacher signals must be present for distillation-driven audio analysis."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": ["gate:teacher_signals_present"]},
                {"name": "collect", "exit_conditions": ["metric:audio_samples>=30"]},
                {"name": "train", "exit_conditions": ["artifact:train_tick"]},
                {"name": "verify", "exit_conditions": ["evidence:test:audio_analysis_accuracy"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Requires audio capture + feature extraction pipeline.",
        ),
    ),

    # ── Robot Arm / Actuators ────────────────────────────────────────────
    (
        re.compile(r"\b(robot\s*arm|arm\s*controller|servo|gripper|pick\s*and\s*place)\b", re.I),
        SkillResolution(
            skill_id="robot_arm_control_v1",
            name="Robot Arm Control",
            capability_type="control",
            risk_level="high",
            required_evidence=[
                "sim:test_pick_place_10runs",
                "real:test_pick_place_3runs_user_present",
            ],
            hard_gates=[
                {"id": "gate:user_present_required", "kind": "safety", "required": "true",
                 "state": "unknown", "details": "Control skills require user present for real hardware runs."},
                {"id": "gate:hardware_connected", "kind": "safety", "required": "true",
                 "state": "unknown", "details": "Robot arm hardware must be detected/connected."},
                {"id": "gate:kill_switch_configured", "kind": "safety", "required": "true",
                 "state": "unknown", "details": "Physical control must have an emergency stop."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": ["gate:hardware_connected", "gate:kill_switch_configured"]},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "acquire", "exit_conditions": ["artifact:driver_connected_or_sim_ready"]},
                {"name": "integrate", "exit_conditions": ["artifact:integration_test_passed"]},
                {"name": "collect", "exit_conditions": ["metric:episodes>=50"]},
                {"name": "train", "exit_conditions": ["artifact:model_checkpoint"]},
                {"name": "verify", "exit_conditions": ["evidence:sim:test_pick_place_10runs"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Sim-first, real hardware gated behind user_present + kill_switch.",
        ),
    ),

    # ── Speaker Identification ───────────────────────────────────────────
    (
        re.compile(
            r"(?:speaker.{0,10}identif|speaker.{0,10}recogni|voice.{0,10}identif"
            r"|voice.{0,10}recogni|identify.{0,30}speaker|recognize.{0,30}speaker"
            r"|recognize.{0,30}voice|who.{0,10}(?:is|am).{0,10}(?:speaking|talking))",
            re.I,
        ),
        SkillResolution(
            skill_id="speaker_identification_v1",
            name="Speaker Identification",
            capability_type="perceptual",
            risk_level="low",
            required_evidence=[
                "test:speaker_id_accuracy_min",
                "test:speaker_id_false_positive_max",
            ],
            capability=StructuredCapability(
                input_type="audio_embedding",
                output_type="speaker_identity_with_confidence",
                success_metrics=(
                    "identification_accuracy",
                    "false_positive_rate",
                    "enrollment_quality",
                ),
                evidence_requirements=(
                    "enrolled_speakers >= 2",
                    "held_out_accuracy >= 80%",
                    "false_positive_rate <= 5%",
                ),
                hardware_requirements=("gpu_cuda",),
            ),
            hard_gates=[
                {"id": "gate:speaker_profiles_exist", "kind": "data",
                 "required": "true", "state": "unknown",
                 "details": "At least 1 enrolled speaker profile required."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": ["gate:speaker_profiles_exist"]},
                {"name": "collect", "exit_conditions": ["metric:speaker_samples>=20"]},
                {"name": "train", "exit_conditions": ["artifact:train_tick"]},
                {"name": "verify", "exit_conditions": [
                    "evidence:test:speaker_id_accuracy_min",
                    "evidence:test:speaker_id_false_positive_max"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            guided_collect={
                "mode": "open_labeled",
                "metric_name": "speaker_samples",
                "prompt_template": (
                    "Training mode for {skill_name}: give one short labeled voice sample for {metric_label}. "
                    "Say the label first, then the example. For example: 'normal: this is my normal voice'. "
                    "You can use labels like normal, louder, quieter, or expressive. "
                    "Say 'stop' when you want to end training mode.{remaining_hint}"
                ),
                "user_input_hints": [
                    "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                    "Varied labeled examples help more than repeating the same sentence the same way.",
                    "About {remaining} more {metric_label} are still needed to clear collect.",
                ],
            },
            notes="ECAPA-TDNN speaker embeddings + cosine similarity. "
                  "Verifies against enrolled profiles with held-out test clips.",
        ),
    ),

    # ── Emotion Detection / Recognition ──────────────────────────────────
    (
        re.compile(
            r"(?:emotion.{0,10}(?:detect|recogni|classif|analys)"
            r"|(?:detect|recogni|classif|analys).{0,20}emotion"
            r"|mood.{0,10}(?:detect|recogni|classif)"
            r"|(?:detect|recogni|classif).{0,20}mood"
            r"|(?:how|what).{0,15}(?:feeling|emotion))",
            re.I,
        ),
        SkillResolution(
            skill_id="emotion_detection_v1",
            name="Emotion Detection",
            capability_type="perceptual",
            risk_level="low",
            required_evidence=[
                "test:emotion_accuracy_min",
                "test:emotion_confusion_matrix_ok",
            ],
            capability=StructuredCapability(
                input_type="audio_waveform",
                output_type="emotion_classification_with_confidence",
                success_metrics=("classification_accuracy", "confusion_matrix_quality"),
                evidence_requirements=(
                    "teacher_signals >= 30",
                    "held_out_accuracy >= 60%",
                ),
            ),
            hard_gates=[
                {"id": "gate:emotion_model_available", "kind": "resource",
                 "required": "true", "state": "unknown",
                 "details": "wav2vec2 emotion model must be loaded."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": ["gate:emotion_model_available"]},
                {"name": "collect", "exit_conditions": ["metric:emotion_samples>=30"]},
                {"name": "train", "exit_conditions": ["artifact:train_tick"]},
                {"name": "verify", "exit_conditions": [
                    "evidence:test:emotion_accuracy_min",
                    "evidence:test:emotion_confusion_matrix_ok"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            guided_collect={
                "mode": "open_labeled",
                "metric_name": "emotion_samples",
                "prompt_template": (
                    "Training mode for {skill_name}: give one short labeled example for {metric_label}. "
                    "Say the label first, then the example. For example: 'happy: I feel great today'. "
                    "Labels like happy, sad, angry, calm, or frustrated are useful. "
                    "Say 'stop' when you want to end training mode.{remaining_hint}"
                ),
                "user_input_hints": [
                    "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                    "Self-labeled examples help most because they provide ground truth for the sample.",
                    "About {remaining} more {metric_label} are still needed to clear collect.",
                ],
            },
            notes="wav2vec2 emotion classifier + distillation. "
                  "Verifies against labeled emotion samples.",
        ),
    ),

    # ── Perception / Recognition (generic catch-all) ─────────────────────
    (
        re.compile(r"\b(recognize|identify|detect|classify)\b.{0,30}\b(emotion|mood|speaker|face|object|gesture|presence)\b", re.I),
        SkillResolution(
            skill_id="perception_distilled_v1",
            name="Perceptual Distillation",
            capability_type="perceptual",
            risk_level="medium",
            required_evidence=[
                "test:distilled_accuracy_min",
                "test:distilled_latency_budget",
            ],
            capability=StructuredCapability(
                input_type="sensor_stream",
                output_type="classification_result",
                success_metrics=("accuracy", "latency_ms"),
                evidence_requirements=(
                    "teacher_signals >= 30",
                    "held_out_accuracy >= 70%",
                ),
            ),
            hard_gates=[
                {"id": "gate:teacher_signals_present", "kind": "data", "required": "true",
                 "state": "unknown", "details": "Teacher signals must be flowing."},
            ],
            default_phases=[
                {"name": "assess", "exit_conditions": ["gate:teacher_signals_present"]},
                {"name": "collect", "exit_conditions": ["metric:teacher_samples>=30"]},
                {"name": "train", "exit_conditions": ["artifact:train_tick"]},
                {"name": "verify", "exit_conditions": ["evidence:test:distilled_accuracy_min"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            guided_collect={
                "mode": "open_labeled",
                "metric_name": "teacher_samples",
                "prompt_template": (
                    "Training mode for {skill_name}: provide one labeled example for {metric_label}. "
                    "Say the label first, then the example using the format 'label: example'. "
                    "Say 'stop' when you want to end training mode.{remaining_hint}"
                ),
                "user_input_hints": [
                    "A short labeled calibration round would help most. This collect phase is waiting on {metric_label}.",
                    "Explicit labels make the collected teacher signals much more useful for later training.",
                    "About {remaining} more {metric_label} are still needed to clear collect.",
                ],
            },
            notes="Collect teacher signals -> train Tier-1 students -> verify on held-out.",
        ),
    ),

    # ── Drawing / Image Generation ───────────────────────────────────────
    (
        re.compile(r"\b(draw|paint|sketch|generate\s+(?:an?\s+)?image|create\s+(?:an?\s+)?picture)\b", re.I),
        SkillResolution(
            skill_id="image_generation_v1",
            name="Image Generation",
            capability_type="procedural",
            risk_level="low",
            required_evidence=["demo:generated_image_sample"],
            default_phases=[
                {"name": "assess", "exit_conditions": []},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "acquire", "exit_conditions": ["artifact:model_available"]},
                {"name": "verify", "exit_conditions": ["evidence:demo:generated_image_sample"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Requires generative image model. Assess feasibility first.",
        ),
    ),

    # ── Coding / Programming ─────────────────────────────────────────────
    (
        re.compile(r"\b(write\s+code|program|coding|compile|debug)\b", re.I),
        SkillResolution(
            skill_id="code_generation_v1",
            name="Code Generation",
            capability_type="procedural",
            risk_level="medium",
            required_evidence=["test:code_generation_smoke", "test:sandbox_execution_pass"],
            capability=StructuredCapability(
                input_type="programming_task",
                output_type="sandbox_validated_code",
                success_metrics=("smoke_task_passed", "sandbox_execution_pass"),
                evidence_requirements=("callable_code_generation_path", "sandbox_execution_artifact"),
                execution_contract_id="code_generation_v1",
                required_executor_kind="sandbox_backed_callable",
                acquisition_eligible=True,
            ),
            default_phases=[
                {"name": "assess", "exit_conditions": []},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "integrate", "exit_conditions": ["artifact:integration_test_passed"]},
                {"name": "verify", "exit_conditions": ["evidence:test:code_generation_smoke"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Leverages existing coding model + sandbox. Verify sandbox execution.",
        ),
    ),

    # ── Web Scraping (acquisition-eligible) ───────────────────────────────
    (
        re.compile(
            r"\b(scrap(?:e|ing)"
            r"|crawl(?:ing)?"
            r"|extract.{0,15}(?:web|page|site|html|content)"
            r"|web.{0,10}(?:extract|harvest|mine)"
            r"|pull.{0,10}data.{0,10}(?:from|off).{0,10}(?:web|site|page))",
            re.I,
        ),
        SkillResolution(
            skill_id="web_scraping_v1",
            name="Web Scraping",
            capability_type="procedural",
            risk_level="medium",
            required_evidence=["test:scraper_returns_structured_data", "test:sandbox_execution_pass"],
            capability=StructuredCapability(
                input_type="web_resource_reference",
                output_type="structured_extracted_data",
                success_metrics=("structured_data_returned", "sandbox_execution_pass"),
                evidence_requirements=("active_plugin_or_tool_path", "sandbox_execution_artifact"),
                execution_contract_id="web_scraping_v1",
                required_executor_kind="plugin",
                acquisition_eligible=True,
            ),
            default_phases=[
                {"name": "assess", "exit_conditions": []},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "integrate", "exit_conditions": ["artifact:plugin_quarantined"]},
                {"name": "verify", "exit_conditions": ["evidence:test:scraper_returns_structured_data"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Acquisition-eligible: may produce a plugin. Requires doc freshness check.",
        ),
    ),

    # ── Data Processing / Transformation ──────────────────────────────────
    (
        re.compile(
            r"\b(parse|transform|convert|process).{0,15}(?:data|csv|json|xml|file|text)"
            r"|\b(?:data).{0,10}(?:pipeline|processing|etl|transformation)",
            re.I,
        ),
        SkillResolution(
            skill_id="data_processing_v1",
            name="Data Processing",
            capability_type="procedural",
            risk_level="low",
            required_evidence=["test:data_processing_smoke", "test:sandbox_execution_pass"],
            capability=StructuredCapability(
                input_type="tabular_or_structured_text",
                output_type="structured_measurements",
                success_metrics=("data_processing_smoke_passed", "sandbox_execution_pass"),
                evidence_requirements=("operational_callable_path", "sandbox_execution_artifact"),
                execution_contract_id="data_transform_v1",
                required_executor_kind="callable_or_plugin",
                acquisition_eligible=True,
            ),
            default_phases=[
                {"name": "assess", "exit_conditions": []},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "integrate", "exit_conditions": ["artifact:integration_test_passed"]},
                {"name": "verify", "exit_conditions": ["evidence:test:data_processing_smoke"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Acquisition-eligible: may produce a plugin for data transformation.",
        ),
    ),

    # ── API Integration ───────────────────────────────────────────────────
    (
        re.compile(
            r"\b(?:connect|integrate|hook|interface).{0,15}(?:api|service|endpoint|rest|graphql)"
            r"|\bapi.{0,10}(?:integration|connect|wrapper|client)",
            re.I,
        ),
        SkillResolution(
            skill_id="api_integration_v1",
            name="API Integration",
            capability_type="procedural",
            risk_level="medium",
            required_evidence=["test:api_integration_smoke", "test:sandbox_execution_pass"],
            capability=StructuredCapability(
                input_type="api_operation_request",
                output_type="typed_api_response",
                success_metrics=("api_integration_smoke_passed", "sandbox_execution_pass"),
                evidence_requirements=("active_plugin_or_tool_path", "sandbox_execution_artifact"),
                execution_contract_id="api_integration_v1",
                required_executor_kind="plugin",
                acquisition_eligible=True,
            ),
            default_phases=[
                {"name": "assess", "exit_conditions": []},
                {"name": "research", "exit_conditions": ["artifact:research_summary"]},
                {"name": "integrate", "exit_conditions": ["artifact:plugin_quarantined"]},
                {"name": "verify", "exit_conditions": ["evidence:test:api_integration_smoke"]},
                {"name": "register", "exit_conditions": ["skill_status:verified"]},
            ],
            notes="Acquisition-eligible: may produce a plugin. Check API docs freshness.",
        ),
    ),
]


def resolve_skill(user_text: str) -> SkillResolution | None:
    """Classify a user request into a skill resolution.

    Returns None for empty text.  Falls back to a dynamically-named
    skill rather than the old generic_procedure_v1.
    """
    t = user_text.strip()
    if not t:
        return None

    for pat, resolution in SKILL_TEMPLATES:
        if pat.search(t):
            return resolution

    skill_id = _generate_skill_id(t)
    skill_name = _generate_skill_name(t)

    return SkillResolution(
        skill_id=skill_id,
        name=skill_name,
        capability_type="procedural",
        risk_level="low",
        required_evidence=["test:procedure_smoke"],
        default_phases=[
            {"name": "assess", "exit_conditions": []},
            {"name": "research", "exit_conditions": ["artifact:research_summary"]},
            {"name": "verify", "exit_conditions": ["evidence:test:procedure_smoke"]},
            {"name": "register", "exit_conditions": ["skill_status:verified"]},
        ],
        notes=f"Auto-generated from: \"{t[:80]}\"",
    )
