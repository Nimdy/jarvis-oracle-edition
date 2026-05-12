# Plugin Process Isolation Design — Tier 2+

**Status**: Design document (no implementation yet)
**Scope**: Runtime isolation for Tier 2+ plugins that perform network I/O, file access, or long-running computation
**Informed by**: Dogfooding campaign observations, existing sandbox and coder_server subprocess patterns

---

## Problem Statement

Tier 0-1 plugins run in-process via `importlib` with import allowlists and `CapabilityGate` output wrapping. This is acceptable for safe, stateless, compute-bound plugins. Tier 2+ plugins may require:

- Network access (HTTP requests, API calls)
- File system access (read/write data, config)
- Long-running computation (model inference, data processing)
- External process communication (subprocess calls)

Running these in-process creates unacceptable blast radius: a misbehaving plugin can leak file descriptors, exhaust memory, block the event loop, or corrupt shared state. Process isolation contains these failure modes.

---

## Existing Subprocess Patterns

Two patterns already exist in the codebase. The isolation design builds on both.

### Pattern A: Sandbox (batch validation)

**Source**: `codegen/sandbox.py`

- Creates a tempdir copy of the project
- Runs `asyncio.create_subprocess_exec` with modified `PYTHONPATH`
- Fixed timeout per phase (lint: 30s, tests: 180s)
- Captures stdout/stderr as structured `TestResult`
- Tempdir cleaned up on completion or failure
- No persistent state between invocations

**Good for**: One-shot plugin invocations (request → response)

### Pattern B: CoderServer (persistent process)

**Source**: `codegen/coder_server.py`

- Spawns `llama-server` as a long-running child process via `subprocess.Popen`
- Communicates over HTTP (OpenAI-compatible API on localhost)
- Process group management (`os.setpgrp`) for clean kill
- `atexit` handler for orphan cleanup
- Health polling with timeout (180s startup, 2s poll interval)
- Graceful shutdown (SIGTERM → grace period → SIGKILL)

**Good for**: Plugins that maintain state between invocations (caches, connections, model loading)

---

## Execution Model

### Decision: Hybrid (subprocess-per-invocation default, persistent server optional)

| Plugin Characteristic | Execution Model | Rationale |
|---|---|---|
| Stateless, fast (<5s) | Subprocess per invocation | Clean isolation, no orphan risk |
| Stateful, reusable connection | Persistent server process | Avoids cold-start on every call |
| Long-running computation (>30s) | Subprocess with extended timeout | Contains resource usage |
| Risk tier 3+ | Subprocess per invocation (mandatory) | Maximum isolation |

### Subprocess per invocation (default)

```
orchestrator → spawn subprocess → JSON stdin → plugin executes → JSON stdout → parse result → kill process
```

1. Create tempdir with plugin code files
2. Write `PluginRequest` as JSON to a runner script's stdin
3. Runner script imports and calls plugin handler
4. Plugin writes `PluginResponse` JSON to stdout
5. Parent reads stdout, parses response
6. Kill process, clean tempdir

### Persistent server (opt-in, tier 2 only)

```
orchestrator → ensure server running → HTTP POST JSON → parse response → keep server alive
```

1. Spawn plugin server process (similar to CoderServer pattern)
2. Health-check via HTTP GET `/health`
3. Invocations via HTTP POST `/invoke`
4. Idle timeout: shut down after 5 minutes of no invocations
5. Process group management + atexit cleanup

---

## IPC Protocol

### Subprocess mode: JSON over stdin/stdout

```python
# Runner script (executed in subprocess)
import sys, json

request = json.loads(sys.stdin.read())
# ... import and call plugin handler ...
result = {"status": "ok", "result": response_data}
sys.stdout.write(json.dumps(result))
sys.stdout.flush()
```

Advantages:
- No network stack required
- No port allocation
- Clean EOF semantics
- Captures stderr separately for error reporting

### Server mode: HTTP localhost

```
POST http://localhost:{port}/invoke
Content-Type: application/json

{"user_text": "...", "context": {...}}
```

Response:
```json
{"status": "ok", "result": {...}, "latency_ms": 42}
```

Port allocation: dynamic (OS-assigned), stored in plugin record.

---

## Resource Limits

| Resource | Limit | Enforcement |
|---|---|---|
| Memory | 512 MB (tier 2), 256 MB (tier 3+) | `resource.setrlimit(RLIMIT_AS)` in subprocess |
| CPU time | 30s (per invocation) | `asyncio.wait_for` timeout + SIGKILL |
| Wall clock | 60s (per invocation) | Parent-side alarm + process group kill |
| File descriptors | 64 | `resource.setrlimit(RLIMIT_NOFILE)` |
| Disk write | 50 MB total | Tempdir with size monitoring |
| Child processes | 0 (no subprocess spawning) | `resource.setrlimit(RLIMIT_NPROC, 0)` |

### Implementation

```python
# Applied in subprocess before plugin import
import resource

resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
```

---

## Filesystem Isolation

### Subprocess working directory

Each invocation gets a fresh tempdir containing:
- Plugin code files (from `PluginCodeBundle`)
- A `_data/` subdirectory (writable, size-capped at 50MB)
- A read-only `_lib/` mount point (if the plugin declares dependencies)

### Restrictions

- No access to `~/.jarvis/` (memories, beliefs, identity data)
- No access to brain source code
- No access to other plugins' directories
- `PYTHONPATH` set to tempdir only (no brain modules importable)

### Persistence between invocations (server mode only)

- Plugin-specific data dir: `~/.jarvis/plugin_data/{plugin_name}/`
- Size-capped (100MB per plugin)
- Explicit cleanup on plugin disable/removal

---

## Network Policy

| Risk Tier | Network Policy | Implementation |
|---|---|---|
| Tier 1 | HTTP/HTTPS GET allowed (outbound only) | Import allowlist includes `requests`, `urllib3` |
| Tier 2 | Restricted endpoints (allowlist) | Plugin manifest declares `allowed_hosts` |
| Tier 3+ | No network | `RLIMIT_NOFILE` + no socket modules in allowlist |

### Tier 2 host allowlist

Plugin manifest declares allowed hosts:
```json
{
  "name": "weather_api",
  "allowed_hosts": ["api.openweathermap.org", "api.weatherapi.com"],
  "network_policy": "restricted"
}
```

Enforcement: DNS-level filtering is impractical in subprocess isolation. Instead, the runner script patches `socket.create_connection` to check the target host against the allowlist before connecting.

---

## Failure Modes

| Failure | Detection | Recovery |
|---|---|---|
| OOM kill | Process exit code 137 (SIGKILL) | Log, increment failure counter, circuit breaker |
| Timeout | `asyncio.wait_for` raises `TimeoutError` | Kill process group, log, circuit breaker |
| Orphan process | `atexit` handler + periodic process audit | `os.killpg` on all tracked PIDs |
| Corrupt output | JSON parse failure on stdout | Return error response, increment failure counter |
| Import violation | Import hook raises `ImportError` | Logged in audit trail, invocation fails |
| Excessive disk write | Periodic `du` check on tempdir | Kill process, clean tempdir |
| Process hangs (no output) | Wall clock timeout (60s) | SIGTERM → 5s grace → SIGKILL |

### Circuit breaker integration

The existing `PluginRegistry` circuit breaker (3 failures in 300s → auto-disable) applies to isolated plugins identically. Process-level failures (OOM, timeout, orphan) count as failures.

---

## Runner Script Template

```python
#!/usr/bin/env python3
"""Plugin subprocess runner — loaded into isolated subprocess."""

import json
import os
import resource
import sys
import time

def _apply_limits(memory_mb, fd_limit, no_subproc):
    mem = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    resource.setrlimit(resource.RLIMIT_NOFILE, (fd_limit, fd_limit))
    if no_subproc:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))

def main():
    config = json.loads(os.environ.get("PLUGIN_CONFIG", "{}"))
    _apply_limits(
        memory_mb=config.get("memory_mb", 256),
        fd_limit=config.get("fd_limit", 64),
        no_subproc=config.get("no_subproc", True),
    )

    request = json.loads(sys.stdin.read())

    plugin_dir = os.environ.get("PLUGIN_DIR", ".")
    sys.path.insert(0, plugin_dir)

    try:
        import importlib
        mod = importlib.import_module("__init__")
        handler = getattr(mod, "handle", None)
        if handler is None:
            print(json.dumps({"status": "error", "error": "No handle() function found"}))
            return

        import asyncio
        if asyncio.iscoroutinefunction(handler):
            result = asyncio.run(handler(request["user_text"], request.get("context", {})))
        else:
            result = handler(request["user_text"], request.get("context", {}))

        print(json.dumps({"status": "ok", "result": result}))
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)[:500]}))

if __name__ == "__main__":
    main()
```

---

## Migration Path

| Risk Tier | Current | Target | When |
|---|---|---|---|
| Tier 0 | In-process | In-process (unchanged) | N/A |
| Tier 1 | In-process | In-process with strict import allowlist | Already implemented |
| Tier 2 | In-process | Subprocess per invocation (default) or persistent server (opt-in) | After dogfood campaign validates lifecycle |
| Tier 3+ | Not allowed | Subprocess per invocation (mandatory) | After Tier 2 isolation is stable |

### Migration sequence

1. Implement the runner script and subprocess invocation path in `PluginRegistry.invoke()`
2. Add `isolation_mode: str` field to `PluginManifest` (values: `in_process`, `subprocess`, `server`)
3. Default all Tier 2+ plugins to `subprocess` isolation
4. Add `/api/plugins/{name}/test-isolation` endpoint for dry-run testing
5. Monitor subprocess invocation latency overhead (target: <100ms for cold start)
6. Enable persistent server mode for plugins that demonstrate >500ms cold-start penalty

### What NOT to implement now

- Container-level isolation (Docker/podman) — overkill for current plugin scope
- Network namespace isolation — impractical without root or container runtime
- Seccomp/AppArmor profiles — adds complexity without proportional safety benefit for Tier 2
- Plugin marketplace / code signing — premature until plugin ecosystem exists

---

## Dashboard Observability

Add to the plugin detail view:
- `isolation_mode` badge (in-process / subprocess / server)
- Process start/stop events in audit trail
- Resource usage per invocation (memory peak, CPU time, wall clock)
- Subprocess failure categorization (OOM / timeout / crash / import violation)

Add to the scheduler status:
- Active isolated plugin processes count
- Total subprocess spawns / kills in the last hour
- Average cold-start latency by isolation mode

---

## Open Questions

1. **Shared library access**: Should Tier 2 plugins access `numpy`, `pandas`, etc. from the brain's venv, or should each plugin bundle its own dependencies?
   - **Recommendation**: Share read-only access to brain venv site-packages via `PYTHONPATH` appending, with explicit allowlist. Bundled deps only for version-sensitive plugins.

2. **State persistence for subprocess mode**: If a plugin needs to cache data between invocations (e.g., API tokens, session state), where does it go?
   - **Recommendation**: `~/.jarvis/plugin_data/{name}/` with 100MB cap. Plugin runner reads/writes this directory. Cleaned on plugin removal.

3. **GPU access**: Should Tier 2 plugins be allowed to use CUDA?
   - **Recommendation**: Not initially. GPU-requiring plugins should be Tier 3+ and go through the Matrix Protocol pathway. Subprocess isolation with CUDA is complex (device sharing, memory contention with active models).
