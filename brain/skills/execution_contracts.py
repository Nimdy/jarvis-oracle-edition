"""Reusable operational verification contracts for learned skills.

Contracts define what a skill must *do* to become operationally verified.
They are not implementations: a contract fixture must execute through a
declared callable path, plugin/tool path, or sandbox-backed generated code.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class SkillSmokeFixture:
    """A deterministic fixture used to verify one operational behavior."""

    name: str
    input_type: str
    input: Any
    expected: dict[str, Any]


@dataclass(frozen=True)
class SkillExecutionContract:
    """Operational contract for a family of learned skills."""

    contract_id: str
    family: str
    skill_ids: tuple[str, ...]
    required_executor_kind: str
    smoke_test_name: str
    smoke_fixtures: tuple[SkillSmokeFixture, ...] = ()
    requires_sandbox: bool = False
    acquisition_eligible: bool = False


@dataclass
class SkillSmokeResult:
    """Result of one contract verification check."""

    name: str
    passed: bool
    details: str
    expected: Any = None
    actual: Any = None
    artifact_refs: list[dict[str, Any]] = field(default_factory=list)

    def to_test(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "expected": self.expected,
            "actual": self.actual,
            "artifact_refs": self.artifact_refs,
        }


_DATA_TRANSFORM_CONTRACT = SkillExecutionContract(
    contract_id="data_transform_v1",
    family="data_transform",
    skill_ids=("data_processing_v1",),
    required_executor_kind="callable_or_plugin",
    smoke_test_name="test:data_processing_smoke",
    requires_sandbox=True,
    acquisition_eligible=True,
    smoke_fixtures=(
        SkillSmokeFixture(
            name="csv_basic_totals",
            input_type="csv_text",
            input="item,quantity,price\napple,2,3\nbanana,4,5\n",
            expected={
                "row_count": 2,
                "columns": ["item", "quantity", "price"],
                "numeric_sums": {"quantity": 6, "price": 8},
                "computed_metrics": {"quantity_x_price": 26},
            },
        ),
    ),
)

_CODE_GENERATION_CONTRACT = SkillExecutionContract(
    contract_id="code_generation_v1",
    family="code_generation",
    skill_ids=("code_generation_v1",),
    required_executor_kind="sandbox_backed_callable",
    smoke_test_name="test:code_generation_smoke",
    requires_sandbox=True,
    acquisition_eligible=True,
    smoke_fixtures=(),
)

_WEB_SCRAPING_CONTRACT = SkillExecutionContract(
    contract_id="web_scraping_v1",
    family="web_scraping",
    skill_ids=("web_scraping_v1",),
    required_executor_kind="plugin",
    smoke_test_name="test:scraper_returns_structured_data",
    requires_sandbox=True,
    acquisition_eligible=True,
    smoke_fixtures=(),
)

_API_INTEGRATION_CONTRACT = SkillExecutionContract(
    contract_id="api_integration_v1",
    family="api_integration",
    skill_ids=("api_integration_v1",),
    required_executor_kind="plugin",
    smoke_test_name="test:api_integration_smoke",
    requires_sandbox=True,
    acquisition_eligible=True,
    smoke_fixtures=(),
)

_CONTRACTS: tuple[SkillExecutionContract, ...] = (
    _DATA_TRANSFORM_CONTRACT,
    _CODE_GENERATION_CONTRACT,
    _WEB_SCRAPING_CONTRACT,
    _API_INTEGRATION_CONTRACT,
)


def get_contract(skill_id: str) -> SkillExecutionContract | None:
    """Return the operational contract for a skill, if one exists."""

    for contract in _CONTRACTS:
        if skill_id in contract.skill_ids:
            return contract
    return None


def requires_contract_test(test_name: str) -> bool:
    """Return True for named tests that must come from contract proof."""

    return any(test_name == contract.smoke_test_name for contract in _CONTRACTS)


def run_contract_smoke(
    job: Any,
    ctx: dict[str, Any],
    artifact_dir: str,
) -> tuple[list[SkillSmokeResult], dict[str, Any]]:
    """Execute a skill contract through its declared callable path.

    The verifier/fixture never acts as the implementation. If the runtime has
    no production callable path, the operational smoke test fails.
    """

    contract = get_contract(job.skill_id)
    if contract is None:
        return [], {}

    results: list[SkillSmokeResult] = []
    artifact_refs: list[dict[str, Any]] = []
    callable_path = _resolve_callable_path(job.skill_id, contract, ctx)

    if callable_path is None:
        results.append(SkillSmokeResult(
            name=contract.smoke_test_name,
            passed=False,
            details="no_operational_callable_path",
            expected={"required_executor_kind": contract.required_executor_kind},
            actual=None,
        ))
    elif not contract.smoke_fixtures:
        results.append(SkillSmokeResult(
            name=contract.smoke_test_name,
            passed=False,
            details="no_smoke_fixture_defined",
            expected={"contract_id": contract.contract_id},
            actual=None,
        ))
    else:
        fixture_results = [
            _run_fixture(contract, fixture, callable_path)
            for fixture in contract.smoke_fixtures
        ]
        passed = all(r.passed for r in fixture_results)
        results.append(SkillSmokeResult(
            name=contract.smoke_test_name,
            passed=passed,
            details="; ".join(r.details for r in fixture_results),
            expected={r.details: r.expected for r in fixture_results},
            actual={r.details: r.actual for r in fixture_results},
        ))

    if contract.requires_sandbox:
        sandbox_ref = _find_sandbox_artifact(job)
        results.append(SkillSmokeResult(
            name="test:sandbox_execution_pass",
            passed=sandbox_ref is not None,
            details="sandbox artifact present" if sandbox_ref else "missing_sandbox_execution_artifact",
            expected={"sandbox_execution_pass": True},
            actual={"sandbox_execution_pass": sandbox_ref is not None},
            artifact_refs=[sandbox_ref] if sandbox_ref else [],
        ))
        if sandbox_ref:
            artifact_refs.append(sandbox_ref)

    payload = {
        "contract_id": contract.contract_id,
        "family": contract.family,
        "skill_id": job.skill_id,
        "created_at": _utc_iso(),
        "results": [asdict(r) for r in results],
        "artifact_refs": artifact_refs,
    }
    path = os.path.join(artifact_dir, "smoke_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    artifact = {
        "id": "contract_smoke_result",
        "type": "contract_smoke_result",
        "path": path,
        "details": {
            "contract_id": contract.contract_id,
            "passed": all(r.passed for r in results),
        },
    }
    return results, artifact


def _resolve_callable_path(
    skill_id: str,
    contract: SkillExecutionContract,
    ctx: dict[str, Any],
) -> Callable[..., Any] | None:
    callables = ctx.get("skill_execution_callables", {}) or {}
    if callable(callables):
        return callables
    for key in (skill_id, contract.family, contract.contract_id):
        candidate = callables.get(key) if isinstance(callables, dict) else None
        if callable(candidate):
            return candidate
    return None


def _run_fixture(
    contract: SkillExecutionContract,
    fixture: SkillSmokeFixture,
    callable_path: Callable[..., Any],
) -> SkillSmokeResult:
    try:
        try:
            actual = callable_path(fixture.input, fixture)
        except TypeError:
            actual = callable_path(fixture.input)
    except Exception as exc:
        return SkillSmokeResult(
            name=contract.smoke_test_name,
            passed=False,
            details=f"{fixture.name}: raised {type(exc).__name__}: {exc}",
            expected=fixture.expected,
            actual=None,
        )

    passed, mismatch = _expected_subset(fixture.expected, actual)
    return SkillSmokeResult(
        name=contract.smoke_test_name,
        passed=passed,
        details=f"{fixture.name}: {'pass' if passed else mismatch}",
        expected=fixture.expected,
        actual=actual,
    )


def _expected_subset(expected: Any, actual: Any, path: str = "") -> tuple[bool, str]:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False, f"{path or 'value'} expected dict got {type(actual).__name__}"
        for key, exp_val in expected.items():
            if key not in actual:
                return False, f"{path + '.' if path else ''}{key} missing"
            ok, reason = _expected_subset(exp_val, actual[key], f"{path + '.' if path else ''}{key}")
            if not ok:
                return False, reason
        return True, ""
    if isinstance(expected, list):
        if actual != expected:
            return False, f"{path or 'value'} expected {expected!r} got {actual!r}"
        return True, ""
    if actual != expected:
        return False, f"{path or 'value'} expected {expected!r} got {actual!r}"
    return True, ""


def _find_sandbox_artifact(job: Any) -> dict[str, Any] | None:
    sandbox_types = {
        "sandbox_execution_pass",
        "sandbox_execution_passed",
        "sandbox_pass",
        "sandbox_result",
    }
    for artifact in reversed(getattr(job, "artifacts", []) or []):
        artifact_id = artifact.get("id", "")
        artifact_type = artifact.get("type", "")
        if artifact_id in sandbox_types or artifact_type in sandbox_types:
            return artifact
    return None


def _utc_iso() -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(time.time(), dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
