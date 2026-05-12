#!/usr/bin/env python3
"""Generic child wrapper for process-isolated plugin execution.

This script is invoked by PluginProcessManager as a subprocess.
It reads JSON requests from stdin (one per line), imports the plugin's
handle() function, and writes JSON responses to stdout.

CRITICAL: This file must have ZERO imports from the brain package.
It runs inside the plugin's own venv with no brain on PYTHONPATH.
Only stdlib modules are allowed here.
"""

import importlib.util
import json
import os
import signal
import sys
import traceback

_shutdown_requested = False


def _handle_sigterm(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


signal.signal(signal.SIGTERM, _handle_sigterm)


def _load_plugin_handler(plugin_dir: str):
    """Load the plugin's handle() function from its __init__.py."""
    init_path = os.path.join(plugin_dir, "__init__.py")
    if not os.path.exists(init_path):
        return None, f"__init__.py not found in {plugin_dir}"

    try:
        spec = importlib.util.spec_from_file_location(
            "_isolated_plugin",
            init_path,
            submodule_search_locations=[plugin_dir],
        )
        if spec is None or spec.loader is None:
            return None, f"Could not create module spec from {init_path}"

        mod = importlib.util.module_from_spec(spec)
        sys.modules["_isolated_plugin"] = mod
        spec.loader.exec_module(mod)

        handler = getattr(mod, "handle", None)
        if handler is None:
            return None, "Plugin module has no handle() function"

        return handler, None
    except Exception:
        return None, traceback.format_exc()


def _invoke_handler(handler, request: dict) -> dict:
    """Call handler and return a structured response dict."""
    request_id = request.get("request_id", "")
    user_text = request.get("user_text", "")
    context = request.get("context", {})

    try:
        import asyncio
        if asyncio.iscoroutinefunction(handler):
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(handler(user_text, context))
            finally:
                loop.close()
        else:
            result = handler(user_text, context)

        if not isinstance(result, dict):
            result = {"output": str(result)}

        return {
            "request_id": request_id,
            "success": True,
            "result": result,
            "error": None,
        }
    except Exception:
        return {
            "request_id": request_id,
            "success": False,
            "result": None,
            "error": traceback.format_exc(),
        }


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: plugin_runner_child.py <plugin_dir>\n")
        sys.exit(1)

    plugin_dir = sys.argv[1]
    if not os.path.isdir(plugin_dir):
        sys.stderr.write(f"Plugin directory does not exist: {plugin_dir}\n")
        sys.exit(1)

    if plugin_dir not in sys.path:
        sys.path.insert(0, plugin_dir)

    handler, load_error = _load_plugin_handler(plugin_dir)
    if handler is None:
        err_resp = json.dumps({
            "request_id": "",
            "success": False,
            "result": None,
            "error": f"Failed to load plugin handler: {load_error}",
        })
        sys.stdout.write(err_resp + "\n")
        sys.stdout.flush()
        sys.exit(1)

    while not _shutdown_requested:
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            break

        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = {
                "request_id": "",
                "success": False,
                "result": None,
                "error": f"Invalid JSON request: {exc}",
            }
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
            continue

        if request.get("action") == "shutdown":
            break

        resp = _invoke_handler(handler, request)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
