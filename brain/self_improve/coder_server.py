"""Backward-compatibility re-export — CoderServer now lives in codegen/."""

from codegen.coder_server import (  # noqa: F401
    CoderServer,
    MAX_PARSE_RETRIES,
    JSON_REPAIR_PROMPT,
)
