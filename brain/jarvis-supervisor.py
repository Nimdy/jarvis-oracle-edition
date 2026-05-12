#!/usr/bin/env python3
"""Jarvis Brain Process Supervisor.

Lightweight process manager that spawns, monitors, and restarts the brain.
Sits between systemd and main.py — systemd manages this supervisor,
the supervisor manages the brain.

Features:
- Exit code protocol: 0=shutdown, 10=intentional restart, other=crash
- Atomic intent file for restart context
- Exponential backoff on crashes (5s → 60s cap)
- Patch rollback on rapid crash with pending verification
- Signal forwarding (SIGTERM/SIGINT → brain subprocess)
- Crash-loop ceiling (5 in 300s → give up)
- Debug mode via JARVIS_SUPERVISOR_DEBUG=1

Zero imports from the brain codebase — stdlib only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import fcntl
from pathlib import Path

EXIT_CLEAN = 0
EXIT_RESTART = 10

JARVIS_DIR = Path.home() / ".jarvis"
INTENT_FILE = JARVIS_DIR / "restart_intent.json"
PENDING_FILE = JARVIS_DIR / "pending_verification.json"
LOCK_FILE = JARVIS_DIR / "jarvis_supervisor.lock"

RAPID_CRASH_WINDOW_S = 30
BACKOFF_SCHEDULE = [5, 10, 20, 40, 60]
CRASH_LOOP_MAX = 5
CRASH_LOOP_WINDOW_S = 300
HEALTHY_RUN_RESET_S = 120
INTENT_STALENESS_S = 120
STALE_PEER_TERM_GRACE_S = 5.0

DEBUG = os.environ.get("JARVIS_SUPERVISOR_DEBUG", "").strip() in ("1", "true", "yes")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [supervisor] {msg}", file=sys.stderr, flush=True)


def acquire_supervisor_lock(script_dir: str):
    """Hold a per-user supervisor lock for the lifetime of this process."""
    JARVIS_DIR.mkdir(parents=True, exist_ok=True)
    lock_fh = LOCK_FILE.open("a+")
    try:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_fh.seek(0)
        owner = lock_fh.read().strip() or "unknown owner"
        log(f"Another supervisor is already active ({owner}) — exiting")
        lock_fh.close()
        return None

    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(json.dumps({
        "pid": os.getpid(),
        "script_dir": script_dir,
        "started_at": time.time(),
    }))
    lock_fh.flush()
    return lock_fh


def _proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "ignore")
    except OSError:
        return ""


def _proc_cwd(pid: int) -> str:
    try:
        return os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return ""


def retire_stale_peer_supervisors(script_dir: str) -> None:
    """Terminate old manual supervisors for this brain directory.

    This is intentionally narrow: same script directory, same supervisor script,
    not this process. It prevents a leftover terminal-launched supervisor from
    racing systemd for main.py and ports 9100/9200.
    """
    this_pid = os.getpid()
    target_cwd = os.path.realpath(script_dir)
    peers: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == this_pid:
            continue
        cmdline = _proc_cmdline(pid)
        if "jarvis-supervisor.py" not in cmdline:
            continue
        if _proc_cwd(pid) != target_cwd:
            continue
        peers.append(pid)

    if not peers:
        return

    log(f"Retiring stale peer supervisor(s): {', '.join(map(str, peers))}")
    for pid in peers:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError as exc:
            log(f"Failed to signal stale supervisor {pid}: {exc}")

    deadline = time.monotonic() + STALE_PEER_TERM_GRACE_S
    while time.monotonic() < deadline:
        remaining = []
        for pid in peers:
            try:
                os.kill(pid, 0)
                remaining.append(pid)
            except ProcessLookupError:
                pass
            except OSError:
                pass
        if not remaining:
            return
        peers = remaining
        time.sleep(0.1)

    for pid in peers:
        try:
            log(f"Force-killing stale supervisor {pid}")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            log(f"Failed to kill stale supervisor {pid}: {exc}")


def read_and_clear_intent() -> dict | None:
    """Read the restart intent file, validate it, delete it, and return contents.

    Returns None if file is missing, malformed, or stale.
    """
    if not INTENT_FILE.exists():
        return None

    try:
        raw = INTENT_FILE.read_text()
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError) as exc:
        log(f"Malformed intent file, ignoring: {exc}")
        _safe_delete(INTENT_FILE)
        return None

    if not isinstance(data, dict) or "reason" not in data:
        log("Intent file missing 'reason' key, ignoring")
        _safe_delete(INTENT_FILE)
        return None

    requested_at = data.get("requested_at", 0)
    age = time.time() - requested_at
    if age > INTENT_STALENESS_S:
        log(f"Stale intent file ({age:.0f}s old), ignoring")
        _safe_delete(INTENT_FILE)
        return None

    _safe_delete(INTENT_FILE)
    return data


def _safe_delete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def has_pending_verification() -> bool:
    return PENDING_FILE.exists()


def rollback_patch(venv_python: str, script_dir: str) -> bool:
    """Run patch rollback via a subprocess into the brain codebase.

    Idempotent — safe to call even if there's nothing to roll back.
    Returns True if rollback succeeded or nothing to do, False on failure.
    """
    log("Attempting patch rollback...")
    rollback_script = """
import sys, json
sys.path.insert(0, '.')
from self_improve.verification import read_pending, clear_pending
from self_improve.orchestrator import SelfImprovementOrchestrator
p = read_pending()
if not p:
    print('[supervisor] No pending verification — nothing to rollback', file=sys.stderr)
    sys.exit(0)
print(f'[supervisor] Rolling back patch {p.patch_id} from {p.snapshot_path}', file=sys.stderr)
try:
    ok = SelfImprovementOrchestrator.restore_snapshot_static(p.snapshot_path)
except Exception as e:
    print(f'[supervisor] Rollback exception: {e}', file=sys.stderr)
    ok = False
if ok:
    clear_pending()
    print('[supervisor] Rollback OK', file=sys.stderr)
    sys.exit(0)
else:
    print('[supervisor] Rollback FAILED', file=sys.stderr)
    sys.exit(1)
"""
    try:
        result = subprocess.run(
            [venv_python, "-c", rollback_script],
            cwd=script_dir,
            timeout=30,
            capture_output=False,
        )
        if result.returncode == 0:
            log("Patch rollback succeeded")
            return True
        else:
            log(f"Patch rollback failed (exit {result.returncode})")
            return False
    except subprocess.TimeoutExpired:
        log("Patch rollback timed out (30s)")
        return False
    except Exception as exc:
        log(f"Patch rollback error: {exc}")
        return False


def get_backoff_delay(crash_count: int) -> float:
    if DEBUG:
        return 0.0
    idx = min(crash_count - 1, len(BACKOFF_SCHEDULE) - 1)
    return BACKOFF_SCHEDULE[max(0, idx)]


def main() -> int:
    script_dir = os.environ.get("JARVIS_BRAIN_DIR") or os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    retire_stale_peer_supervisors(script_dir)
    lock_fh = acquire_supervisor_lock(script_dir)
    if lock_fh is None:
        return 0

    venv_python = os.path.join(script_dir, ".venv", "bin", "python")
    if not os.path.isfile(venv_python):
        log(f"ERROR: venv Python not found at {venv_python}")
        log("Run ./setup.sh first")
        return 1

    extra_args = sys.argv[1:]

    log(f"Starting (pid={os.getpid()}, debug={'ON' if DEBUG else 'off'})")
    if DEBUG:
        log("Debug mode: backoff disabled, verbose logging")

    crash_times: list[float] = []
    brain_proc: subprocess.Popen | None = None
    shutdown_requested = False

    def _forward_signal(signum, _frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        if brain_proc and brain_proc.poll() is None:
            sig_name = signal.Signals(signum).name
            log(f"Forwarding {sig_name} to brain (pid={brain_proc.pid})")
            try:
                brain_proc.send_signal(signum)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _forward_signal)
    signal.signal(signal.SIGINT, _forward_signal)

    while True:
        launch_time = time.monotonic()
        log(f"Launching brain: {venv_python} -u main.py {' '.join(extra_args)}".strip())

        brain_proc = subprocess.Popen(
            [venv_python, "-u", "main.py"] + extra_args,
            cwd=script_dir,
        )

        exit_code = brain_proc.wait()
        elapsed = time.monotonic() - launch_time
        brain_proc = None

        if shutdown_requested:
            log(f"Brain exited {exit_code} after supervisor shutdown request | action: stop")
            return 0

        intent = read_and_clear_intent()
        reason = intent.get("reason", "none") if intent else "none"

        # -- Decide action based on exit code + intent --

        if exit_code == EXIT_CLEAN and not intent:
            log(f"Brain exited {exit_code} | reason: clean_shutdown | action: stop")
            return 0

        if exit_code == EXIT_RESTART:
            log(f"Brain exited {exit_code} | reason: {reason} | action: restart (immediate)")

            crash_times.clear()

            delay = 0.0
            if intent and intent.get("delay_s"):
                delay = float(intent["delay_s"])
            if delay > 0:
                log(f"Waiting {delay:.1f}s before restart (requested by brain)")
                time.sleep(delay)
            continue

        # -- Crash path --

        was_signaled = exit_code < 0
        sig_info = ""
        if was_signaled:
            try:
                sig_info = f" ({signal.Signals(-exit_code).name})"
            except (ValueError, AttributeError):
                sig_info = f" (signal {-exit_code})"

        log(f"Brain exited {exit_code}{sig_info} | reason: crash | elapsed: {elapsed:.1f}s | action: evaluate")

        if elapsed < RAPID_CRASH_WINDOW_S and has_pending_verification():
            log(f"Rapid crash ({elapsed:.0f}s) with pending verification — rolling back")
            ok = rollback_patch(venv_python, script_dir)
            if not ok:
                log("CRITICAL: Rollback failed — exiting to let systemd handle it")
                return 1

        now = time.monotonic()
        crash_times.append(now)

        recent = [t for t in crash_times if now - t < CRASH_LOOP_WINDOW_S]
        crash_times[:] = recent

        if len(recent) >= CRASH_LOOP_MAX:
            log(f"Crash loop detected: {len(recent)} crashes in {CRASH_LOOP_WINDOW_S}s — giving up")
            return 1

        if elapsed >= HEALTHY_RUN_RESET_S:
            crash_times.clear()
            crash_times.append(now)
            if DEBUG:
                log(f"Brain ran for {elapsed:.0f}s (healthy) — crash counter reset")

        delay = get_backoff_delay(len(crash_times))
        if delay > 0:
            log(f"Backing off {delay:.0f}s before restart (crash #{len(crash_times)})")
            time.sleep(delay)


if __name__ == "__main__":
    sys.exit(main())
