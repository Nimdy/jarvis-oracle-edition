"""SelfViewSynthesizer — assembles the Operational Self-View from existing readouts.

P0 substrate: a DETERMINISTIC, READ-ONLY, NO-LLM synthesizer. ``synthesize(sources)`` is
a pure function of its input dict — it never touches a subsystem, never writes canonical
state, never calls an LLM. Live wiring (reading subsystems into ``sources``) lives in
``gather.py``; keeping synthesis pure makes the read-only guarantee structural and the
whole thing trivially testable.

Every reported value is a provenance-tagged :class:`Fact`. Missing/unreadable source data
degrades to a GAP or UNKNOWN fact — never a fabricated default. Gaps are first-class.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cognition.self_view.provenance import Fact, Provenance, gap, unknown

SCHEMA_VERSION = "osv-0.1.0"


@dataclass
class SelfModel:
    """JARVIS's fused self-model (P0). Provenance-honest; gaps explicit."""

    generated_at: float
    schema_version: str
    structural: dict[str, Any]
    performance: dict[str, Any]
    maturity: dict[str, Any]
    belief: dict[str, Any]
    change: dict[str, Any]
    subsystems: dict[str, Any]
    gaps: list[dict[str, Any]]
    coverage: dict[str, Any]
    architecture: dict[str, Any] | None = None  # P-A: code-grounded full structural map (manifest)
    live_activity: dict[str, Any] | None = None  # P-C: current NN-substrate activity (bounded, no history)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": round(self.generated_at, 3),
            "structural": _facts_to_dict(self.structural),
            "performance": _facts_to_dict(self.performance),
            "maturity": _facts_to_dict(self.maturity),
            "belief": _facts_to_dict(self.belief),
            "change": _facts_to_dict(self.change),
            "subsystems": _facts_to_dict(self.subsystems),
            "gaps": self.gaps,
            "coverage": self.coverage,
            "architecture": _facts_to_dict(self.architecture) if self.architecture else {},
            "live_activity": _facts_to_dict(self.live_activity) if self.live_activity else {},
        }


def _facts_to_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, Fact):
            out[k] = v.to_dict()
        elif isinstance(v, dict):
            out[k] = _facts_to_dict(v)
        elif isinstance(v, list):
            out[k] = [x.to_dict() if isinstance(x, Fact) else x for x in v]
        else:
            out[k] = v
    return out


def _num(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


class SelfViewSynthesizer:
    """Pure, deterministic assembler of the SelfModel from a sources dict."""

    def synthesize(self, sources: dict[str, Any] | None, now: float) -> SelfModel:
        sources = sources or {}
        structural = self._structural(sources)
        performance = self._performance(sources)
        maturity = self._maturity(sources)
        belief = self._belief(sources)
        change = self._change(sources)
        subsystems = self._subsystems(sources)
        architecture = sources.get("architecture") or {}
        live_activity = sources.get("live_activity") or {}
        gaps = self._gaps(sources, performance, belief)
        coverage = self._coverage(performance, subsystems, gaps)
        return SelfModel(
            generated_at=now,
            schema_version=SCHEMA_VERSION,
            structural=structural,
            performance=performance,
            maturity=maturity,
            belief=belief,
            change=change,
            subsystems=subsystems,
            gaps=gaps,
            coverage=coverage,
            architecture=architecture,
            live_activity=live_activity,
        )

    # -- Subsystems: the live subsystem surface (P0.5, from build_cache) ----

    def _subsystems(self, s: dict[str, Any]) -> dict[str, Any]:
        inv = s.get("subsystems")
        if not isinstance(inv, dict) or not inv:
            return {"_meta": {"lifecycle": gap("no subsystem inventory (snapshot unavailable)")}}
        # inv is {name: {field: Fact}} from the bespoke adapters; pass through.
        return dict(inv)

    # -- Structural: "how I'm built" ---------------------------------------

    def _structural(self, s: dict[str, Any]) -> dict[str, Any]:
        skills = s.get("skills")
        if not isinstance(skills, dict):
            return {"capabilities": gap("skills registry unavailable", "skills.registry")}
        by_status = skills.get("by_status") or {}
        skill_list = skills.get("skills") or []
        earned = []
        bootstrap = []
        for sk in skill_list:
            if not isinstance(sk, dict):
                continue
            row = {"id": sk.get("skill_id"), "status": sk.get("status")}
            if sk.get("learning_job_id"):
                earned.append(row)
            else:
                bootstrap.append(row)
        return {
            "verified_counts": Fact(by_status, Provenance.MEASURED, source="skills.registry"),
            "earned_capabilities": Fact(
                earned, Provenance.MEASURED,
                note="acquired end-to-end through the learning pipeline", source="skills.registry"),
            "bootstrap_capabilities": Fact(
                bootstrap, Provenance.INTERNALLY_SCORED,
                note="seeded/re-registered, not earned through the live pipeline",
                source="skills.registry"),
        }

    # -- Performance: "how well I'm doing" (provenance-critical) ------------

    def _performance(self, s: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        # scoreboard composite (#9): measured only when really covered
        sb = s.get("scoreboard")
        if isinstance(sb, dict):
            if sb.get("composite_enabled") and sb.get("composite") is not None:
                cov = sb.get("coverage") or {}
                out["scoreboard_composite"] = Fact(
                    sb.get("composite"), Provenance.MEASURED,
                    note=f"measured composite over {cov.get('measured')}/{cov.get('total')} categories",
                    source="jarvis_eval.scoreboard")
            else:
                out["scoreboard_composite"] = gap(
                    "composite not enabled — insufficient measured coverage",
                    "jarvis_eval.scoreboard")
        else:
            out["scoreboard_composite"] = gap("scoreboard unavailable", "jarvis_eval.scoreboard")

        # causal foresight (#9 provenance flows straight through)
        causal = s.get("causal")
        if isinstance(causal, dict) and causal.get("predictive_total"):
            out["predictive_accuracy"] = Fact(
                _num(causal.get("predictive_accuracy")), Provenance.MEASURED,
                note="validated foresight (event-triggered transitions)",
                source="cognition.causal_engine")
            out["persistence_accuracy"] = Fact(
                _num(causal.get("persistence_accuracy")), Provenance.INTERNALLY_SCORED,
                note="near-tautological steady-state continuation — not foresight",
                source="cognition.causal_engine")
        else:
            out["predictive_accuracy"] = gap(
                "no validated predictions yet (cold or pre-validation)",
                "cognition.causal_engine")

        # simulator confidence — internally scored (its own confidence)
        sim = s.get("simulator")
        if isinstance(sim, dict) and sim.get("avg_confidence") is not None:
            out["simulator_avg_confidence"] = Fact(
                _num(sim.get("avg_confidence")), Provenance.INTERNALLY_SCORED,
                note="the simulator's own projected confidence, not validated; shadow-gated",
                source="cognition.simulator")

        # policy NN win-rate — shadow only
        pol = s.get("policy")
        if isinstance(pol, dict) and pol.get("nn_win_rate") is not None:
            out["policy_nn_win_rate"] = Fact(
                _num(pol.get("nn_win_rate")), Provenance.SHADOW_ONLY,
                note="shadow NN-vs-kernel comparison; advisory/non-authoritative until promotion",
                source="policy.evaluator")

        return out

    # -- Maturity: "earned vs gated" ---------------------------------------

    def _maturity(self, s: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}

        for key, src in (("simulator_promotion", "cognition.promotion"),
                         ("world_model_promotion", "cognition.promotion")):
            p = s.get(key)
            if isinstance(p, dict) and p.get("level_name"):
                out[key] = Fact(
                    {"level": p.get("level"), "level_name": p.get("level_name"),
                     "total_validated": p.get("total_validated")},
                    Provenance.MEASURED,
                    note="accuracy-gated promotion state (validated outcomes)", source=src)
            else:
                out[key] = unknown(f"{key} not reported", src)

        # dormant-by-gate capabilities render as DORMANT, never "available"
        for key, src in (("cognitive_planner", "cognition.planner"),
                         ("counterfactual", "epistemic.counterfactual")):
            c = s.get(key)
            if isinstance(c, dict):
                active = bool(c.get("active"))
                out[key] = Fact(
                    "active" if active else "dormant",
                    Provenance.MEASURED if active else Provenance.DORMANT,
                    note=c.get("reason", ""), source=src)
            else:
                out[key] = unknown(f"{key} not reported", src)

        return out

    # -- Belief: "what I believe about myself" -----------------------------

    def _belief(self, s: dict[str, Any]) -> dict[str, Any]:
        beliefs = s.get("beliefs")
        if not isinstance(beliefs, dict):
            return {"self_beliefs": gap("belief graph unavailable", "epistemic.belief_graph")}
        items = beliefs.get("self_beliefs") or []
        rendered: list[Fact] = []
        for b in items:
            if not isinstance(b, dict):
                continue
            prov_field = (b.get("provenance") or "").lower()
            status = (b.get("epistemic_status") or "inferred").lower()
            # A self-belief is NEVER a measurement. The gather only emits operator-stated beliefs
            # (provenance==user_claim) → INTERNALLY_SCORED (the operator said it; it was NOT externally
            # validated). Anything else → SELF_SCORED. Nothing here reaches ADVISORY/MEASURED — that
            # requires external grounding, not this path. (The old epistemic_status branch was a dead
            # wire: gather never populated these beliefs, so the map never ran.)
            if prov_field == "user_claim":
                prov = Provenance.INTERNALLY_SCORED
            else:
                prov = Provenance.SELF_SCORED
            rendered.append(Fact(
                b.get("statement") or b.get("id"), prov,
                note=f"provenance={prov_field} status={status}", source="epistemic.belief_graph"))
        if not rendered:
            return {"self_beliefs": gap("no self-referential beliefs yet", "epistemic.belief_graph")}
        return {"self_beliefs": rendered}

    # -- Change: "what recently changed in me" -----------------------------

    def _change(self, s: dict[str, Any]) -> dict[str, Any]:
        rc = s.get("recent_changes")
        if not isinstance(rc, list) or not rc:
            return {"recent": gap("no recent-change record available",
                                  "build_history/attribution_ledger")}
        return {"recent": Fact(rc, Provenance.MEASURED,
                               note="recorded changes (skills/commits/learning-jobs)",
                               source="build_history/attribution_ledger")}

    # -- Gaps: first-class -------------------------------------------------

    def _gaps(self, s: dict[str, Any], performance: dict[str, Any],
              belief: dict[str, Any]) -> list[dict[str, Any]]:
        gaps: list[dict[str, Any]] = []

        # comparator-less scoreboard categories
        sb = s.get("scoreboard")
        if isinstance(sb, dict):
            for bar in sb.get("bars") or []:
                if isinstance(bar, dict) and not bar.get("measured"):
                    gaps.append({
                        "area": f"scoreboard:{bar.get('category')}",
                        "reason": "no genuine external comparator / insufficient samples",
                        "source": "jarvis_eval.scoreboard",
                    })

        # any performance fact that came back absent (gap/unknown/stale)
        for name, fact in performance.items():
            if isinstance(fact, Fact) and fact.is_absent:
                gaps.append({"area": f"performance:{name}", "reason": fact.note,
                             "source": fact.source})

        # belief gap
        sb_self = belief.get("self_beliefs")
        if isinstance(sb_self, Fact) and sb_self.is_absent:
            gaps.append({"area": "belief:self_beliefs", "reason": sb_self.note,
                         "source": sb_self.source})

        # architecture-manifest gaps (code-grounded; first-class curiosity targets)
        arch = s.get("architecture")
        if isinstance(arch, dict):
            for g in (arch.get("gaps") or []):
                gaps.append({"area": "architecture", "reason": str(g)[:300],
                             "source": "architecture_manifest"})

        return gaps

    def _coverage(self, performance: dict[str, Any], subsystems: dict[str, Any],
                  gaps: list[dict[str, Any]]) -> dict[str, Any]:
        measured = sum(1 for f in performance.values()
                       if isinstance(f, Fact) and f.is_measurement)
        # subsystem inventory tally by lifecycle provenance — the honest "what can you do"
        by_prov: dict[str, int] = {}
        sub_count = 0
        for name, entry in subsystems.items():
            if name.startswith("_") or not isinstance(entry, dict):
                continue
            life = entry.get("lifecycle")
            if not isinstance(life, Fact):
                continue
            sub_count += 1
            by_prov[life.provenance] = by_prov.get(life.provenance, 0) + 1
        return {
            "measured_performance_facts": measured,
            "total_performance_facts": len(performance),
            "subsystem_count": sub_count,
            "subsystems_by_provenance": by_prov,
            "gap_count": len(gaps),
        }
