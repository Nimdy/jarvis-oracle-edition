"""Codebase self-awareness tool -- AST indexer, symbol table, import graph,
context budgeter, and write boundaries.

Gives Jarvis the ability to read, understand, and reason about its own source
code.  The index is built at startup and rebuilt after self-improvement patches.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # brain/

WRITE_BOUNDARIES: dict[str, list[str]] = {
    "self_improve":   ["brain/self_improve/", "brain/tools/", "brain/reasoning/"],
    "skill_plugin":   ["brain/tools/plugins/"],
    "hemisphere":     ["brain/hemisphere/"],
    "consciousness":  ["brain/consciousness/", "brain/personality/"],
    "policy":         ["brain/policy/"],
    "memory":         ["brain/memory/"],
    "perception":     ["brain/perception/"],
}

_APPROX_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CodeSymbol:
    fqn: str
    kind: str               # "module" | "class" | "function" | "method" | "constant"
    file: str               # relative to project root, e.g. "consciousness/engine.py"
    line: int
    end_line: int
    signature: str
    docstring: str
    imports: list[str] = field(default_factory=list)
    imported_by: list[str] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Per-file metadata extracted from AST."""
    file: str
    module_fqn: str
    docstring: str
    imports: list[str] = field(default_factory=list)
    symbols: list[CodeSymbol] = field(default_factory=list)
    line_count: int = 0


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------


def _signature_from_func(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Reconstruct a human-readable function signature from an AST node."""
    args_parts: list[str] = []
    fa = node.args

    # positional args
    n_defaults = len(fa.defaults)
    n_args = len(fa.args)
    for i, arg in enumerate(fa.args):
        ann = ast.unparse(arg.annotation) if arg.annotation else ""
        name = arg.arg
        s = f"{name}: {ann}" if ann else name
        default_idx = i - (n_args - n_defaults)
        if default_idx >= 0:
            s += f" = {ast.unparse(fa.defaults[default_idx])}"
        args_parts.append(s)

    if fa.vararg:
        v = fa.vararg
        ann = f": {ast.unparse(v.annotation)}" if v.annotation else ""
        args_parts.append(f"*{v.arg}{ann}")

    for i, arg in enumerate(fa.kwonlyargs):
        ann = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default = fa.kw_defaults[i]
        d = f" = {ast.unparse(default)}" if default else ""
        args_parts.append(f"{arg.arg}{ann}{d}")

    if fa.kwarg:
        k = fa.kwarg
        ann = f": {ast.unparse(k.annotation)}" if k.annotation else ""
        args_parts.append(f"**{k.arg}{ann}")

    ret = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{prefix} {node.name}({', '.join(args_parts)}){ret}"


def _first_docstring(body: list[ast.stmt]) -> str:
    """Extract the docstring from a body, truncated to 200 chars."""
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        ds = body[0].value.value.strip()
        return ds[:200] if len(ds) > 200 else ds
    return ""


def _extract_imports(tree: ast.Module) -> list[str]:
    """Return list of imported module names from a module AST."""
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return sorted(set(names))


def _end_line(node: ast.AST) -> int:
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


# ---------------------------------------------------------------------------
# CodebaseIndex -- the main singleton
# ---------------------------------------------------------------------------


class CodebaseIndex:
    """Indexes the brain/ source tree and provides querying + context budgeting."""

    _HASH_STORE_PATH = Path.home() / ".jarvis" / "code_index_hashes.json"

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or _PROJECT_ROOT
        self._modules: dict[str, ModuleInfo] = {}       # module_fqn -> ModuleInfo
        self._symbols: dict[str, CodeSymbol] = {}        # fqn -> CodeSymbol
        self._import_graph: dict[str, set[str]] = {}     # module_fqn -> set of imported module_fqns
        self._imported_by: dict[str, set[str]] = {}      # module_fqn -> set of modules that import it
        self._last_indexed: float = 0.0
        self._file_hashes: dict[str, str] = {}           # rel_path -> sha256[:32]
        self._changed_files: list[dict[str, str]] = []   # detected changes since last index

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Walk the source tree and build the full index."""
        t0 = time.monotonic()
        self._modules.clear()
        self._symbols.clear()
        self._import_graph.clear()
        self._imported_by.clear()

        old_hashes = self._load_hashes()
        new_hashes: dict[str, str] = {}

        py_files = sorted(self._root.rglob("*.py"))
        for fpath in py_files:
            rel = fpath.relative_to(self._root)
            if any(p.startswith(".") or p == "__pycache__" or p == ".venv" for p in rel.parts):
                continue
            self._index_file(fpath, rel)
            try:
                content = fpath.read_bytes()
                new_hashes[str(rel)] = hashlib.sha256(content).hexdigest()[:32]
            except OSError:
                pass

        self._resolve_import_graph()
        self._last_indexed = time.time()

        project_root = self._root.parent
        for doc_rel in ("AGENTS.md", "ARCHITECTURE.md", "docs/SYSTEM_OVERVIEW.md"):
            doc_path = project_root / doc_rel
            if doc_path.exists():
                try:
                    content = doc_path.read_bytes()
                    new_hashes[doc_rel] = hashlib.sha256(content).hexdigest()[:32]
                except OSError:
                    pass

        self._changed_files = self._compute_changes(old_hashes, new_hashes)
        self._file_hashes = new_hashes
        self._save_hashes(new_hashes)

        total_symbols = len(self._symbols)
        total_modules = len(self._modules)
        elapsed = (time.monotonic() - t0) * 1000
        change_note = ""
        if self._changed_files:
            change_note = f", {len(self._changed_files)} file(s) changed since last index"
        logger.info("Codebase indexed: %d modules, %d symbols in %.0fms%s",
                     total_modules, total_symbols, elapsed, change_note)

    def rebuild_file(self, rel_path: str) -> None:
        """Re-index a single file after a patch is applied."""
        fpath = self._root / rel_path
        if not fpath.exists():
            # file was deleted
            mod_fqn = self._path_to_module(Path(rel_path))
            self._remove_module(mod_fqn)
            return
        rel = Path(rel_path)
        self._remove_module(self._path_to_module(rel))
        self._index_file(fpath, rel)
        self._resolve_import_graph()

    def _index_file(self, fpath: Path, rel: Path) -> None:
        try:
            source = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        try:
            tree = ast.parse(source, filename=str(rel))
        except SyntaxError as exc:
            logger.debug("Syntax error in %s: %s", rel, exc)
            return

        mod_fqn = self._path_to_module(rel)
        line_count = source.count("\n") + 1
        imports = _extract_imports(tree)
        mod_doc = _first_docstring(tree.body)

        mod_info = ModuleInfo(
            file=str(rel),
            module_fqn=mod_fqn,
            docstring=mod_doc,
            imports=imports,
            line_count=line_count,
        )

        # Top-level functions and classes
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                sym = CodeSymbol(
                    fqn=f"{mod_fqn}.{node.name}",
                    kind="function",
                    file=str(rel),
                    line=node.lineno,
                    end_line=_end_line(node),
                    signature=_signature_from_func(node),
                    docstring=_first_docstring(node.body),
                )
                mod_info.symbols.append(sym)
                self._symbols[sym.fqn] = sym

            elif isinstance(node, ast.ClassDef):
                cls_fqn = f"{mod_fqn}.{node.name}"
                cls_sym = CodeSymbol(
                    fqn=cls_fqn,
                    kind="class",
                    file=str(rel),
                    line=node.lineno,
                    end_line=_end_line(node),
                    signature=f"class {node.name}",
                    docstring=_first_docstring(node.body),
                )
                mod_info.symbols.append(cls_sym)
                self._symbols[cls_sym.fqn] = cls_sym

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        meth_fqn = f"{cls_fqn}.{item.name}"
                        meth_sym = CodeSymbol(
                            fqn=meth_fqn,
                            kind="method",
                            file=str(rel),
                            line=item.lineno,
                            end_line=_end_line(item),
                            signature=_signature_from_func(item),
                            docstring=_first_docstring(item.body),
                        )
                        mod_info.symbols.append(meth_sym)
                        self._symbols[meth_sym.fqn] = meth_sym

            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        const_fqn = f"{mod_fqn}.{target.id}"
                        const_sym = CodeSymbol(
                            fqn=const_fqn,
                            kind="constant",
                            file=str(rel),
                            line=node.lineno,
                            end_line=_end_line(node),
                            signature=f"{target.id} = ...",
                            docstring="",
                        )
                        mod_info.symbols.append(const_sym)
                        self._symbols[const_sym.fqn] = const_sym

        self._modules[mod_fqn] = mod_info
        self._import_graph[mod_fqn] = set(imports)

    def _remove_module(self, mod_fqn: str) -> None:
        info = self._modules.pop(mod_fqn, None)
        if info:
            for sym in info.symbols:
                self._symbols.pop(sym.fqn, None)
        self._import_graph.pop(mod_fqn, None)
        self._imported_by.pop(mod_fqn, None)

    def _resolve_import_graph(self) -> None:
        """Build the reverse import graph (imported_by) and annotate symbols."""
        self._imported_by.clear()
        known = set(self._modules.keys())

        for mod_fqn, raw_imports in self._import_graph.items():
            resolved: set[str] = set()
            for imp in raw_imports:
                # try to match against known modules
                if imp in known:
                    resolved.add(imp)
                else:
                    # try prefix match (e.g., "consciousness.engine" for import "consciousness.engine")
                    for k in known:
                        if k == imp or k.endswith(f".{imp}") or imp.startswith(f"{k}."):
                            resolved.add(k)
                            break
            self._import_graph[mod_fqn] = resolved
            for dep in resolved:
                self._imported_by.setdefault(dep, set()).add(mod_fqn)

        # propagate to symbols
        for mod_fqn, info in self._modules.items():
            imp_list = sorted(self._import_graph.get(mod_fqn, set()))
            imp_by_list = sorted(self._imported_by.get(mod_fqn, set()))
            for sym in info.symbols:
                sym.imports = imp_list
                sym.imported_by = imp_by_list

    def _path_to_module(self, rel: Path) -> str:
        parts = list(rel.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def search(self, keyword: str, limit: int = 20) -> list[CodeSymbol]:
        """Search symbols by keyword in fqn, signature, or docstring."""
        kw = keyword.lower()
        results: list[CodeSymbol] = []
        for sym in self._symbols.values():
            if kw in sym.fqn.lower() or kw in sym.signature.lower() or kw in sym.docstring.lower():
                results.append(sym)
                if len(results) >= limit:
                    break
        return results

    def get_symbol(self, fqn: str) -> CodeSymbol | None:
        return self._symbols.get(fqn)

    def get_module(self, mod_fqn: str) -> ModuleInfo | None:
        return self._modules.get(mod_fqn)

    def get_imports_of(self, mod_fqn: str) -> list[str]:
        return sorted(self._import_graph.get(mod_fqn, set()))

    def get_importers_of(self, mod_fqn: str) -> list[str]:
        return sorted(self._imported_by.get(mod_fqn, set()))

    def get_module_symbols(self, mod_fqn: str) -> list[CodeSymbol]:
        info = self._modules.get(mod_fqn)
        return list(info.symbols) if info else []

    def list_modules(self) -> list[str]:
        return sorted(self._modules.keys())

    def read_file(self, rel_path: str) -> str | None:
        """Read a source file and return contents with line numbers."""
        fpath = self._root / rel_path
        if not fpath.exists():
            return None
        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            numbered = [f"{i + 1:4d}| {line}" for i, line in enumerate(lines)]
            return "\n".join(numbered)
        except OSError:
            return None

    def read_doc(self, rel_path: str, max_chars: int = 20000) -> str | None:
        """Read a documentation file (.md, .txt) from the project root.

        Unlike read_file, this searches from the project root (parent of brain/)
        to reach top-level docs like AGENTS.md, ARCHITECTURE.md.
        """
        project_root = self._root.parent
        fpath = project_root / rel_path
        if not fpath.exists():
            fpath = self._root / rel_path
        if not fpath.exists():
            return None
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
            return text[:max_chars] if len(text) > max_chars else text
        except OSError:
            return None

    def read_span(self, rel_path: str, start: int, end: int) -> str | None:
        """Read specific line range from a file."""
        fpath = self._root / rel_path
        if not fpath.exists():
            return None
        try:
            lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(0, start - 1)
            end = min(len(lines), end)
            numbered = [f"{i + start + 1:4d}| {lines[i + start]}" for i in range(end - start)]
            return "\n".join(numbered)
        except (OSError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Context Budgeter
    # ------------------------------------------------------------------

    def get_budgeted_context(
        self,
        target_files: list[str],
        referenced_symbols: list[str] | None = None,
        max_tokens: int = 4000,
    ) -> str:
        """Build context for an LLM prompt within a token budget.

        Fills greedily in priority order:
          1. Signatures + docstrings of target files (cheapest)
          2. Full spans for referenced symbols
          3. Signatures of direct import neighbors (one hop)
        """
        budget_chars = max_tokens * _APPROX_CHARS_PER_TOKEN
        parts: list[str] = []
        used = 0

        # Layer 1: signatures + docstrings of target modules
        for fpath in target_files:
            mod_fqn = self._path_to_module(Path(fpath))
            info = self._modules.get(mod_fqn)
            if not info:
                continue
            section = self._format_module_signatures(info)
            if used + len(section) > budget_chars:
                break
            parts.append(section)
            used += len(section)

        # Layer 2: full spans for referenced symbols
        if referenced_symbols:
            for sym_fqn in referenced_symbols:
                sym = self._symbols.get(sym_fqn)
                if not sym:
                    continue
                span = self.read_span(sym.file, sym.line, sym.end_line)
                if not span:
                    continue
                header = f"\n# --- {sym.fqn} ({sym.file}:{sym.line}-{sym.end_line}) ---\n"
                block = header + span
                if used + len(block) > budget_chars:
                    break
                parts.append(block)
                used += len(block)

        # Layer 3: signatures of direct import neighbors
        for fpath in target_files:
            mod_fqn = self._path_to_module(Path(fpath))
            neighbors = self.get_imports_of(mod_fqn) + self.get_importers_of(mod_fqn)
            for nbr in neighbors:
                info = self._modules.get(nbr)
                if not info:
                    continue
                section = self._format_module_signatures(info, header_prefix="[neighbor] ")
                if used + len(section) > budget_chars:
                    return "\n".join(parts)
                parts.append(section)
                used += len(section)

        return "\n".join(parts)

    def _format_module_signatures(self, info: ModuleInfo, header_prefix: str = "") -> str:
        lines: list[str] = []
        lines.append(f"\n# {header_prefix}{info.module_fqn} ({info.file}, {info.line_count} lines)")
        if info.docstring:
            lines.append(f'"""  {info.docstring}  """')
        lines.append(f"# imports: {', '.join(info.imports[:10])}")
        for sym in info.symbols:
            indent = "    " if sym.kind == "method" else ""
            lines.append(f"{indent}{sym.signature}")
            if sym.docstring:
                doc_preview = sym.docstring[:80]
                lines.append(f'{indent}    """{doc_preview}"""')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Write Boundary Enforcement
    # ------------------------------------------------------------------

    @staticmethod
    def check_write_boundaries(category: str, file_paths: list[str]) -> list[str]:
        """Return list of violations if any paths fall outside write boundaries.

        Args:
            category: one of WRITE_BOUNDARIES keys (e.g., "self_improve")
            file_paths: relative paths to check
        """
        allowed = WRITE_BOUNDARIES.get(category, [])
        if not allowed:
            return [f"Unknown write category: {category}"]

        violations: list[str] = []
        for fp in file_paths:
            normalized = fp if fp.startswith("brain/") else f"brain/{fp}"
            if not any(normalized.startswith(a) for a in allowed):
                violations.append(f"Write boundary violation ({category}): {fp}")
        return violations

    # ------------------------------------------------------------------
    # Telemetry / dashboard
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_modules": len(self._modules),
            "total_symbols": len(self._symbols),
            "total_lines": sum(m.line_count for m in self._modules.values()),
            "last_indexed": self._last_indexed,
            "symbol_kinds": self._count_by_kind(),
            "import_edges": sum(len(v) for v in self._import_graph.values()),
        }

    def _count_by_kind(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for sym in self._symbols.values():
            counts[sym.kind] = counts.get(sym.kind, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Code change detection
    # ------------------------------------------------------------------

    def _load_hashes(self) -> dict[str, str]:
        """Load stored file hashes from previous index."""
        try:
            if self._HASH_STORE_PATH.exists():
                data = json.loads(self._HASH_STORE_PATH.read_text())
                return data.get("hashes", {})
        except Exception:
            pass
        return {}

    def _save_hashes(self, hashes: dict[str, str]) -> None:
        """Persist file hashes for next startup comparison."""
        try:
            self._HASH_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {"hashes": hashes, "saved_at": time.time()}
            self._HASH_STORE_PATH.write_text(json.dumps(data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed to save code hashes: %s", exc)

    @staticmethod
    def _compute_changes(old: dict[str, str], new: dict[str, str]) -> list[dict[str, str]]:
        """Compare old and new hash maps to find changed/added/removed files."""
        changes: list[dict[str, str]] = []
        for path, new_hash in new.items():
            old_hash = old.get(path, "")
            if not old_hash:
                changes.append({"path": path, "type": "added", "old_hash": "", "new_hash": new_hash})
            elif old_hash != new_hash:
                changes.append({"path": path, "type": "modified", "old_hash": old_hash, "new_hash": new_hash})
        for path in old:
            if path not in new:
                changes.append({"path": path, "type": "removed", "old_hash": old.get(path, ""), "new_hash": ""})
        return changes

    def get_changes_since_last_index(self) -> list[dict[str, str]]:
        """Return files that changed since the previous build.

        Only meaningful after build() has been called. Returns modified
        and removed files (not added, which is everything on first run).
        """
        return [c for c in self._changed_files if c["type"] in ("modified", "removed")]

    def get_modified_files(self) -> list[str]:
        """Return paths of files modified (not just added) since last index."""
        return [c["path"] for c in self._changed_files if c["type"] == "modified"]

    # ------------------------------------------------------------------
    # Telemetry / dashboard
    # ------------------------------------------------------------------

    def persist(self, path: str = "~/.jarvis/code_index.json") -> None:
        """Save index summary to disk for dashboard display."""
        out_path = Path(path).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "stats": self.get_stats(),
            "modules": {
                fqn: {
                    "file": info.file,
                    "line_count": info.line_count,
                    "docstring": info.docstring[:100],
                    "symbol_count": len(info.symbols),
                    "imports": info.imports[:10],
                }
                for fqn, info in self._modules.items()
            },
        }
        try:
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to persist code index: %s", exc)

    # ------------------------------------------------------------------
    # Conversation tool interface
    # ------------------------------------------------------------------

    def answer_query(self, query: str) -> str:
        """Answer a natural-language question about the codebase.

        Used by ToolType.CODEBASE in the conversation handler and autonomy
        self-study queries. Handles both precise symbol lookups and broad
        natural-language questions.
        """
        q = query.lower()

        # "where is X defined"
        if "where" in q and "defined" in q:
            words = query.split()
            for w in words:
                if "." in w or w[0].isupper():
                    results = self.search(w, limit=5)
                    if results:
                        lines = [f"Found {len(results)} match(es) for '{w}':"]
                        for sym in results:
                            importers = ", ".join(sym.imported_by[:5]) if sym.imported_by else "none"
                            lines.append(f"  - {sym.fqn} ({sym.kind}) at {sym.file}:{sym.line}")
                            lines.append(f"    signature: {sym.signature}")
                            lines.append(f"    imported by: {importers}")
                        return "\n".join(lines)

        # "what calls X" / "who imports X"
        if "call" in q or "import" in q or "uses" in q:
            words = query.split()
            for w in words:
                if "." in w or (len(w) > 3 and w[0].isupper()):
                    results = self.search(w, limit=3)
                    if results:
                        sym = results[0]
                        importers = self.get_importers_of(sym.imports[0]) if sym.imports else []
                        lines = [f"'{sym.fqn}' is in module '{sym.file}'"]
                        if sym.imported_by:
                            lines.append(f"Imported by: {', '.join(sym.imported_by[:10])}")
                        return "\n".join(lines)

        # NL keyword extraction: split query into meaningful words and search each
        stop_words = {
            "what", "how", "does", "do", "is", "are", "my", "the", "a", "an",
            "and", "or", "in", "of", "to", "for", "with", "from", "about",
            "read", "summarize", "explain", "describe", "work", "have", "has",
            "can", "when", "where", "which", "its", "it", "this", "that",
            "overall", "current", "known", "key", "all", "each", "their",
        }
        keywords = [
            w.strip("?.,!:;()\"'")
            for w in query.split()
            if len(w.strip("?.,!:;()\"'")) > 2
            and w.strip("?.,!:;()\"'").lower() not in stop_words
        ]

        all_results: dict[str, CodeSymbol] = {}
        for kw in keywords:
            for sym in self.search(kw, limit=5):
                all_results[sym.fqn] = sym

        if all_results:
            syms = list(all_results.values())[:15]
            lines = [f"Found {len(syms)} symbol(s) related to query:"]
            for sym in syms:
                lines.append(f"  - {sym.fqn} ({sym.kind}) at {sym.file}:{sym.line}")
                lines.append(f"    {sym.signature}")
                if sym.docstring:
                    lines.append(f"    doc: {sym.docstring[:100]}")
            return "\n".join(lines)

        # Module-level search: match query keywords against module docstrings and paths
        module_matches: list[ModuleInfo] = []
        for kw in keywords:
            kw_lower = kw.lower()
            for info in self._modules.values():
                if kw_lower in info.module_fqn.lower() or kw_lower in info.docstring.lower():
                    if info not in module_matches:
                        module_matches.append(info)

        if module_matches:
            lines = [f"Found {len(module_matches)} module(s) related to query:"]
            for info in module_matches[:10]:
                lines.append(f"  - {info.module_fqn} ({info.file}, {info.line_count} lines, {len(info.symbols)} symbols)")
                if info.docstring:
                    lines.append(f"    doc: {info.docstring[:120]}")
                lines.append(f"    imports: {', '.join(info.imports[:8])}")
            return "\n".join(lines)

        return f"No symbols found matching '{query}'. Try a class or function name."


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

codebase_index = CodebaseIndex()
