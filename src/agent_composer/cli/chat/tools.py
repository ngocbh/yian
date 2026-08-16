"""Flow-op tools for the `ac chat` composer assistant.

A CLI-HOST concern, not engine core (the core ships no domain tools). Each tool is a
`@register_tool`-decorated function; importing this module self-registers them. All
paths resolve against a workspace root set by `set_workspace` at `ac chat` startup and
escapes (absolute paths, `..`) are rejected.
"""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Annotated, Any, Iterator, Optional
from agent_composer.tools import register_tool
from agent_composer.compose.loader import load_flow
from agent_composer.compose.helpers.errors import LoadError
from agent_composer.run import _run_flow_controlled, run_flow as _run_flow

# The workspace root every tool path is confined to. Set at `ac chat` startup.
_WORKSPACE: Path = Path.cwd()
_CONTROLLER_STOP: ContextVar[Any | None] = ContextVar(
    "agent_composer_chat_controller_stop",
    default=None,
)


@contextmanager
def _bind_controller_stop(stop_event: Any) -> Iterator[None]:
    """Make the enclosing chat lifecycle visible to flow-op tools on this thread."""
    token = _CONTROLLER_STOP.set(stop_event)
    try:
        yield
    finally:
        _CONTROLLER_STOP.reset(token)


def set_workspace(root: Path) -> None:
    """Set the directory all flow-op tools are confined to."""
    global _WORKSPACE
    _WORKSPACE = Path(root).resolve()


def _resolve(rel: str) -> Path:
    """Resolve `rel` under the workspace, rejecting absolute paths and `..` escapes."""
    p = (_WORKSPACE / rel).resolve()
    if p != _WORKSPACE and _WORKSPACE not in p.parents:
        raise ValueError(f"path escapes the workspace: {rel!r}")
    return p


@register_tool("list_flows")
def list_flows(
    subdir: Annotated[
        str, "Optional workspace-relative folder to list within; empty lists the whole workspace."
    ] = "",
) -> str:
    """DISCOVER which flows exist. Use this FIRST when you don't already know a flow's
    filename. Lists every `*.yaml` flow file under the workspace (pass `subdir` to look in
    one folder). Returns the paths, one per line, relative to the workspace — feed one of
    them to `read_flow`, `validate_flow`, or `run_flow`."""
    base = _resolve(subdir) if subdir else _WORKSPACE
    names = sorted(str(p.relative_to(_WORKSPACE)) for p in base.rglob("*.yaml"))
    return "\n".join(names) if names else "(no flows found)"


@register_tool("read_flow")
def read_flow(
    path: Annotated[str, "Workspace-relative path to the flow .yaml file to read."],
) -> str:
    """READ a flow's YAML source. Use this to inspect how a flow is built before you
    explain it or edit it (always read before you overwrite with `write_flow`). `path` is a
    workspace-relative flow file. Returns the full YAML text."""
    return _resolve(path).read_text()


@register_tool("validate_flow")
def validate_flow(
    path: Annotated[str, "Workspace-relative path to the flow .yaml file to compile-check."],
) -> str:
    """CHECK that a flow compiles, WITHOUT running it — and discover the inputs it needs.
    Use this before `run_flow` (to learn what inputs to pass) and after `write_flow` edits.
    `path` is a workspace-relative flow file. Returns `OK: <name>; inputs: <names>` when it
    compiles, or `INVALID: <compile error>` when it does not."""
    source = _resolve(path)
    text = source.read_text()
    try:
        loaded = load_flow(
            text,
            search_paths=[_WORKSPACE],
            source_label=source.name,
        )
    except LoadError as err:
        return f"INVALID: {err}"
    names = ", ".join(d.name for d in loaded.input) or "(none)"
    return f"OK: {loaded.name}; inputs: {names}"


@register_tool("run_flow")
def run_flow(
    path: Annotated[str, "Workspace-relative path to the flow .yaml file to run."],
    inputs: Annotated[
        Optional[dict[str, Any]],
        'A JSON OBJECT mapping each of the flow\'s declared input names to a value, e.g. '
        '{"text": "Alice is 30"}. Pass an object, NOT a JSON string. Omit for a flow that '
        "takes no inputs.",
    ] = None,
) -> str:
    """EXECUTE a flow and get its output. `path` is a workspace-relative flow file.
    `inputs` is a JSON OBJECT mapping each of the flow's declared input names to a value
    (e.g. {"text": "Alice is 30"}) — pass it as an object, NOT a JSON string; omit it for a
    flow that takes no inputs. If unsure which inputs a flow needs, call `validate_flow`
    first. Returns `OUTPUT: <result>` on success or `FAILED: <error>` if the run errors.
    Blocking HUMAN_INPUT is served by the enclosing chat host, while authored WAIT remains
    live until its deadline."""
    source = _resolve(path)
    text = source.read_text()
    try:
        loaded = load_flow(
            text,
            search_paths=[_WORKSPACE],
            source_label=source.name,
        )
    except LoadError as err:
        return f"ERROR: {err}"
    parent_stop = _CONTROLLER_STOP.get()
    if parent_stop is None:
        result = _run_flow(loaded, inputs or {})  # direct Python use: private lifecycle
    else:
        # Outer host cancellation flows down; a nested NodeFailed sets only this
        # child, preserving the failure as a model-visible tool result.
        with parent_stop.linked_child() as child_stop:
            result = _run_flow_controlled(
                loaded,
                inputs or {},
                stop_event=child_stop,
            )
    if result.status == "succeeded":
        return f"OUTPUT: {result.output}"
    detail = result.failure.message if result.failure is not None else "(no detail)"
    return f"FAILED: {detail}"


@register_tool("write_flow")
def write_flow(
    path: Annotated[str, "Workspace-relative destination path for the flow .yaml file."],
    content: Annotated[str, "The complete flow YAML text to write."],
) -> str:
    """CREATE a new flow or OVERWRITE an existing one. Use this to author or edit flows
    (read the current file with `read_flow` first when editing). `path` is a
    workspace-relative destination; `content` is the complete flow YAML. The content is
    compile-checked BEFORE anything is written: returns `WROTE: <path>` on success, or
    `INVALID (not written): <compile error>` if it doesn't compile — so a bad edit never
    lands on disk."""
    dest = _resolve(path)                        # raises on escape
    try:
        load_flow(
            content,
            search_paths=[_WORKSPACE],
            source_label=dest.name,
        )
    except LoadError as err:
        return f"INVALID (not written): {err}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content)
    return f"WROTE: {dest.relative_to(_WORKSPACE)}"
