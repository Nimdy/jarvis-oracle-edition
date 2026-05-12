"""Backward-compatibility re-export — Sandbox now lives in codegen/."""

from codegen.sandbox import (  # noqa: F401
    Sandbox,
    SandboxDiagnostic,
    SIM_TICKS,
    LINT_TIMEOUT_S,
    TEST_TIMEOUT_S,
    COPIED_SUBDIRS,
    _MODULE_TO_TESTS,
)
