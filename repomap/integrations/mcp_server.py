"""RepoMap MCP server — exposes repo mapping tools via the Model Context Protocol.

Tools:
    repomap_overview      — token-efficient repo overview
    repomap_around        — dependency neighborhood of a symbol
    repomap_query         — structural questions about the codebase
    repomap_data_model    — data model graph with read/write relationships
    repomap_entry_points  — detected entry points
    repomap_impact        — blast radius of modifying a symbol
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _get_engine(repo_root: str | None = None):
    """Load the RepoMap engine for the given (or detected) repo root."""
    from repomap.core.config import RepomapConfig
    from repomap.core.engine import RepomapEngine

    root = Path(repo_root) if repo_root else _detect_repo_root()
    config = RepomapConfig.load(root)
    return RepomapEngine(repo_root=root, config=config)


def _detect_repo_root() -> Path:
    """Walk up from cwd to find repo root (presence of .git, pyproject.toml, etc.)."""
    cwd = Path(os.getcwd()).resolve()
    markers = {".git", "pyproject.toml", "package.json", "Cargo.toml", "go.mod"}
    current = cwd
    while current != current.parent:
        if any((current / m).exists() for m in markers):
            return current
        current = current.parent
    return cwd


def _format_node_summary(node, edges: list, max_edges: int = 10) -> str:
    """Compact text summary of a single node for MCP responses."""
    out_edges = [e for e in edges if e.source_id == node.symbol_id][:max_edges]
    lines = [f"{node.kind} `{node.qualified_name}`"]
    if node.signature:
        lines.append(f"  sig: {node.signature[:100]}")
    lines.append(f"  file: {node.file_path}:{node.line_start}")
    if node.is_entry_point:
        lines.append("  ★ entry point")
    if node.data_model_framework:
        lines.append(f"  ⬡ data model ({node.data_model_framework})")
    for e in out_edges:
        tname = e.target_qualified_name.split(".")[-1]
        lines.append(f"  {e.display_arrow} {e.edge_type}: {tname}")
    return "\n".join(lines)


def create_mcp_server(
    repo_root: str | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> Any:
    """Create and return a configured FastMCP server instance."""
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP(
        name="repomap",
        instructions=(
            "RepoMap provides structural intelligence about a code repository. "
            "Use these tools to understand the codebase before making changes, "
            "find entry points, trace data model relationships, and assess impact."
        ),
        host=host,
        port=port,
    )

    # Lazy engine — shared across tool calls within one server session
    _engine_cache: dict[str, Any] = {}

    def get_engine():
        if "engine" not in _engine_cache:
            _engine_cache["engine"] = _get_engine(repo_root)
        return _engine_cache["engine"]

    # ── Tool 1: overview ──────────────────────────────────────────────────────
    @mcp.tool()
    def repomap_overview(max_tokens: int = 1000) -> str:
        """Get a token-efficient overview of the repository structure.

        Returns the most important functions, classes, and data models ranked
        by structural importance (PageRank), fitted to the token budget.
        """
        engine = get_engine()
        result = engine.generate(max_tokens=max_tokens, output_format="xml")
        return result.text

    # ── Tool 2: around ────────────────────────────────────────────────────────
    @mcp.tool()
    def repomap_around(target: str, depth: int = 2, max_tokens: int = 500) -> str:
        """Get the dependency neighborhood of a specific function or file.

        Args:
            target: Function name, class name, or file path to focus on.
            depth:  Number of hops to traverse (1-3).
            max_tokens: Token budget for the response.
        """
        depth = max(1, min(depth, 3))
        engine = get_engine()
        result = engine.generate(around=target, max_tokens=max_tokens, output_format="xml")
        if result.total_symbols == 0:
            return f"Symbol '{target}' not found in the repository index. Try running `repomap generate` first."
        return result.text

    # ── Tool 3: query ─────────────────────────────────────────────────────────
    @mcp.tool()
    def repomap_query(question: str) -> str:
        """Answer structural questions about the codebase.

        Examples:
            "what functions write to UserModel?"
            "which files import auth?"
            "what are the callers of create_session?"
            "show me all entry points"
        """
        engine = get_engine()
        q = question.lower()
        store = engine.store

        # Route to specialized handlers based on question patterns
        if any(x in q for x in ("write", "writes", "written", "mutate", "create")):
            model_name = _extract_model_name(question)
            return _query_writers(store, model_name)

        if any(x in q for x in ("read", "reads", "use", "uses")):
            model_name = _extract_model_name(question)
            return _query_readers(store, model_name)

        if any(x in q for x in ("import", "imports", "imported")):
            target = _extract_target(question)
            return _query_importers(store, target)

        if any(x in q for x in ("call", "calls", "caller")):
            target = _extract_target(question)
            return _query_callers(store, target)

        if any(x in q for x in ("entry", "entrypoint", "route", "endpoint")):
            return _query_entry_points(store)

        if any(x in q for x in ("data model", "model", "schema", "pydantic", "dataclass")):
            model_name = _extract_model_name(question)
            return _query_data_models(store, model_name)

        # Fallback: full-text search over symbol names
        return _query_search(store, question)

    # ── Tool 4: data_model ────────────────────────────────────────────────────
    @mcp.tool()
    def repomap_data_model(model_name: str = "") -> str:
        """Get data model definitions with their read/write relationships.

        Args:
            model_name: Specific model name (e.g. 'UserCreate'). Leave empty for all.
        """
        engine = get_engine()
        store = engine.store
        return _query_data_models(store, model_name or None)

    # ── Tool 5: entry_points ──────────────────────────────────────────────────
    @mcp.tool()
    def repomap_entry_points() -> str:
        """List all detected entry points: API routes, CLI commands, main functions."""
        engine = get_engine()
        store = engine.store
        return _query_entry_points(store)

    # ── Tool 6: impact ────────────────────────────────────────────────────────
    @mcp.tool()
    def repomap_impact(function: str, depth: int = 2) -> str:
        """Show the blast radius of modifying a specific function.

        Returns all callers (direct and transitive) that would be affected
        by a change to the given function.

        Args:
            function: Function or class name to analyze.
            depth:    How many hops to traverse upstream (default 2).
        """
        engine = get_engine()
        store = engine.store
        depth = max(1, min(depth, 4))

        # Find the target symbol
        rows = store.get_symbols_by_name(function)
        if not rows:
            row = store.get_symbol_by_qualified_name(function)
            rows = [row] if row else []
        if not rows:
            return f"Symbol '{function}' not found."

        target_ids = {r["id"] for r in rows}

        # BFS upstream: find all callers within depth hops
        visited: dict[int, int] = {tid: 0 for tid in target_ids}  # id → hop distance
        frontier = set(target_ids)

        all_edges = store.get_all_edges()
        # Build reverse adjacency: target_id → list of source_ids
        rev_adj: dict[int, list[int]] = {}
        for e in all_edges:
            if e["target_id"] is None:
                continue
            if e["target_id"] not in rev_adj:
                rev_adj[e["target_id"]] = []
            rev_adj[e["target_id"]].append(e["source_id"])

        for hop in range(depth):
            next_frontier: set[int] = set()
            for tid in frontier:
                for caller_id in rev_adj.get(tid, []):
                    if caller_id not in visited:
                        visited[caller_id] = hop + 1
                        next_frontier.add(caller_id)
            frontier = next_frontier
            if not frontier:
                break

        if len(visited) <= len(target_ids):
            return f"No callers found for '{function}' within {depth} hops."

        lines = [f"Impact analysis for `{function}` (depth={depth}):\n"]
        lines.append(f"Affected symbols: {len(visited) - len(target_ids)}\n")
        for sym_id, hop in sorted(visited.items(), key=lambda x: x[1]):
            if sym_id in target_ids:
                continue
            row = store.get_symbol_by_id(sym_id)
            if row:
                hop_label = "direct caller" if hop == 1 else f"{hop} hops"
                lines.append(
                    f"  [{hop_label}] {row['kind']} `{row['name']}` "
                    f"— {row['file_path']}:{row['line_start']}"
                )
        return "\n".join(lines)

    return mcp


# ── Query helpers ─────────────────────────────────────────────────────────────

def _extract_model_name(question: str) -> str | None:
    """Heuristically extract a model name from a question string."""
    # Look for CamelCase words
    import re
    camel = re.findall(r'\b[A-Z][a-zA-Z0-9]+\b', question)
    return camel[0] if camel else None


def _extract_target(question: str) -> str | None:
    """Extract a function/module name from a question."""
    import re
    # backtick-quoted names
    bt = re.findall(r'`([^`]+)`', question)
    if bt:
        return bt[0]
    # CamelCase or snake_case identifiers
    idents = re.findall(r'\b([a-z_][a-z0-9_]+|[A-Z][a-zA-Z0-9]+)\b', question)
    # Skip common English words
    stop = {'what','which','where','who','how','show','me','all','the','are','is','of','to','in','from','that','it'}
    for ident in idents:
        if ident.lower() not in stop:
            return ident
    return None


def _query_writers(store, model_name: str | None) -> str:
    rows = store.get_all_edges()
    results = []
    for e in rows:
        if e["edge_type"] != "writes":
            continue
        if model_name and model_name.lower() not in e["target_qualified_name"].lower():
            continue
        sym = store.get_symbol_by_id(e["source_id"])
        if sym:
            results.append(
                f"  `{sym['name']}` ({sym['kind']}) — {sym['file_path']}:{sym['line_start']}"
                f" writes → {e['target_qualified_name'].split('.')[-1]}"
            )
    if not results:
        return f"No write relationships found{' for ' + model_name if model_name else ''}."
    header = f"Functions that write{' to ' + model_name if model_name else ''}:\n"
    return header + "\n".join(results[:50])


def _query_readers(store, model_name: str | None) -> str:
    rows = store.get_all_edges()
    results = []
    for e in rows:
        if e["edge_type"] != "reads":
            continue
        if model_name and model_name.lower() not in e["target_qualified_name"].lower():
            continue
        sym = store.get_symbol_by_id(e["source_id"])
        if sym:
            results.append(
                f"  `{sym['name']}` ({sym['kind']}) — {sym['file_path']}:{sym['line_start']}"
                f" reads → {e['target_qualified_name'].split('.')[-1]}"
            )
    if not results:
        return f"No read relationships found{' for ' + model_name if model_name else ''}."
    header = f"Functions that read{' from ' + model_name if model_name else ''}:\n"
    return header + "\n".join(results[:50])


def _query_importers(store, target: str | None) -> str:
    rows = store.get_all_edges()
    results = []
    for e in rows:
        if e["edge_type"] != "imports":
            continue
        if target and target.lower() not in e["target_qualified_name"].lower():
            continue
        sym = store.get_symbol_by_id(e["source_id"])
        if sym:
            results.append(
                f"  `{sym['file_path']}` imports {e['target_qualified_name']}"
            )
    if not results:
        return f"No imports found{' for ' + target if target else ''}."
    return f"Files that import{' ' + target if target else ''}:\n" + "\n".join(results[:50])


def _query_callers(store, target: str | None) -> str:
    rows = store.get_all_edges()
    results = []
    for e in rows:
        if e["edge_type"] != "calls":
            continue
        if target and target.lower() not in e["target_qualified_name"].lower():
            continue
        sym = store.get_symbol_by_id(e["source_id"])
        if sym:
            results.append(
                f"  `{sym['name']}` — {sym['file_path']}:{sym['line_start']}"
                f" → calls {e['target_qualified_name'].split('.')[-1]}"
            )
    if not results:
        return f"No callers found{' for ' + target if target else ''}."
    return f"Callers{' of ' + target if target else ''}:\n" + "\n".join(results[:50])


def _query_entry_points(store) -> str:
    syms = [r for r in store.get_all_symbols() if r["is_entry_point"]]
    if not syms:
        return "No entry points detected. Entry points are auto-detected from decorators, main() functions, and CLI commands."
    lines = ["Detected entry points:\n"]
    for s in syms:
        lines.append(f"  ★ `{s['name']}` ({s['kind']}) — {s['file_path']}:{s['line_start']}")
        if s["signature"]:
            lines.append(f"    {s['signature'][:100]}")
    return "\n".join(lines)


def _query_data_models(store, model_name: str | None) -> str:
    models = store.get_all_data_models()
    if model_name:
        models = [m for m in models if model_name.lower() in m["name"].lower()]
    if not models:
        return f"No data models found{' matching ' + model_name if model_name else ''}."

    # Pre-fetch all edges once to avoid N+1 queries
    all_edges = store.get_all_edges()
    model_ids = {m["symbol_id"] for m in models[:20]}
    # Build per-model reader/writer lists
    model_readers: dict[int, list[str]] = {mid: [] for mid in model_ids}
    model_writers: dict[int, list[str]] = {mid: [] for mid in model_ids}
    for e in all_edges:
        tid = e["target_id"]
        if tid not in model_ids:
            continue
        sym = store.get_symbol_by_id(e["source_id"])
        if not sym:
            continue
        if e["edge_type"] == "reads":
            model_readers[tid].append(sym["name"])
        elif e["edge_type"] == "writes":
            model_writers[tid].append(sym["name"])

    lines = []
    for m in models[:20]:
        lines.append(f"\n### {m['name']} ({m['framework']})")
        lines.append(f"File: {m['file_path']}:{m['line_start']}")
        fields = json.loads(m["fields_json"] or "[]")
        if fields:
            for f in fields:
                opt = "?" if f.get("optional") else ""
                lines.append(f"  {f['name']}: {f['type']}{opt}")

        readers = model_readers.get(m["symbol_id"], [])
        writers = model_writers.get(m["symbol_id"], [])
        if readers:
            lines.append(f"Read by: {', '.join(readers[:10])}")
        if writers:
            lines.append(f"Written by: {', '.join(writers[:10])}")
    return "\n".join(lines)


def _query_search(store, query: str) -> str:
    q = query.lower()
    results = []
    for row in store.get_all_symbols():
        if q in row["name"].lower() or q in (row["qualified_name"] or "").lower():
            results.append(
                f"  `{row['name']}` ({row['kind']}) — {row['file_path']}:{row['line_start']}"
            )
    if not results:
        return f"No symbols matching '{query}'."
    return f"Symbols matching '{query}':\n" + "\n".join(results[:30])


# ── Entry point ───────────────────────────────────────────────────────────────

def run_stdio(repo_root: str | None = None) -> None:
    """Run the MCP server in STDIO mode."""
    import asyncio
    mcp = create_mcp_server(repo_root=repo_root)
    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 3847, repo_root: str | None = None) -> None:
    """Run the MCP server in HTTP/SSE mode."""
    mcp = create_mcp_server(repo_root=repo_root, host=host, port=port)
    mcp.run(transport="streamable-http")
