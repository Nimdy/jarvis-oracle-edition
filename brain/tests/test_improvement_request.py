"""Tests for self_improve/improvement_request.py — data contract validation.

Covers:
  - ImprovementRequest field defaults and generation
  - to_dict() serialization and JSON roundtrip
  - scope_paths property (delegates to ALLOWED_PATHS)
  - Type literal values
  - Field completeness
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from self_improve.improvement_request import ImprovementRequest


class TestDefaults:
    def test_id_generated(self):
        req = ImprovementRequest()
        assert req.id.startswith("imp_")
        assert len(req.id) > 4

    def test_unique_ids(self):
        a = ImprovementRequest()
        b = ImprovementRequest()
        assert a.id != b.id

    def test_default_type(self):
        req = ImprovementRequest()
        assert req.type == "consciousness_enhancement"

    def test_default_priority(self):
        req = ImprovementRequest()
        assert req.priority == 0.5

    def test_default_golden_status(self):
        req = ImprovementRequest()
        assert req.golden_status == "none"

    def test_default_empty_fields(self):
        req = ImprovementRequest()
        assert req.target_module == ""
        assert req.description == ""
        assert req.evidence == []
        assert req.constraints == {}
        assert req.fingerprint == ""
        assert req.evidence_detail == {}

    def test_timestamp_set(self):
        req = ImprovementRequest()
        assert req.timestamp > 0


class TestToDict:
    def test_all_fields_present(self):
        req = ImprovementRequest(
            target_module="consciousness",
            description="test improvement",
            evidence=["metric_a degraded"],
            priority=0.8,
            fingerprint="abc123",
        )
        d = req.to_dict()
        expected_keys = {
            "id", "timestamp", "type", "target_module", "description",
            "evidence", "priority", "constraints", "requires_approval",
            "golden_trace_id", "golden_command_id", "golden_authority_class",
            "golden_status", "fingerprint", "evidence_detail",
            "declared_scope",
        }
        assert expected_keys == set(d.keys())

    def test_values_match(self):
        req = ImprovementRequest(
            target_module="memory",
            description="optimize search",
            priority=0.9,
        )
        d = req.to_dict()
        assert d["target_module"] == "memory"
        assert d["description"] == "optimize search"
        assert d["priority"] == 0.9

    def test_json_serializable(self):
        req = ImprovementRequest()
        serialized = json.dumps(req.to_dict())
        parsed = json.loads(serialized)
        assert parsed["id"] == req.id


class TestScopePaths:
    def test_returns_allowed_paths(self):
        req = ImprovementRequest()
        paths = req.scope_paths
        assert isinstance(paths, list)
        assert len(paths) >= 11
        assert any("consciousness" in p for p in paths)
        assert any("memory" in p for p in paths)

    def test_returns_copy(self):
        req = ImprovementRequest()
        paths = req.scope_paths
        paths.append("brain/hacked/")
        assert "brain/hacked/" not in req.scope_paths


class TestTypeValues:
    def test_all_valid_types(self):
        valid = [
            "performance_optimization",
            "policy_model_upgrade",
            "consciousness_enhancement",
            "bug_fix",
            "architecture_improvement",
        ]
        for t in valid:
            req = ImprovementRequest(type=t)
            assert req.type == t
