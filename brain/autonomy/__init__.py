"""Autonomy — autonomous research and learning for Jarvis.

Components:
  - CuriosityDetector: extracts actionable questions from internal events
  - InternalQueryInterface: headless tool runtime (web, codebase, memory)
  - ResearchGovernor: rate limits, mode gating, topic-cluster cooldowns, budget caps
  - KnowledgeIntegrator: stores findings as memories with provenance
  - AutonomyEventBridge: wires consciousness events → curiosity detection
  - OpportunityScorer: ranks intents by Impact × Evidence × Confidence − Risk − Cost
                       + policy memory adjustment − diminishing returns − action rate penalty
  - MetricTriggers: drives research from sustained system metric deficits
  - DeltaTracker: before/after + counterfactual baselines for credit assignment
  - AutonomyPolicyMemory: persists what worked/regressed, feeds future scoring
  - EpisodeRecorder: records trace episodes for offline replay and policy comparison
  - AutonomyOrchestrator: queue management, coordinates all components, L2 graduation
"""

from autonomy.orchestrator import AutonomyOrchestrator
from autonomy.research_intent import ResearchIntent, ResearchResult, ResearchFinding

__all__ = [
    "AutonomyOrchestrator",
    "ResearchIntent",
    "ResearchResult",
    "ResearchFinding",
]
