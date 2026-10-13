"""Flow-op tools for the `yian chat` assistant.

A CLI-HOST concern, not engine core (the core ships no domain tools). Each tool is a
`@register_tool`-decorated function; importing this module self-registers them. All
paths resolve against a workspace root set by `set_workspace` at `yian chat` startup and
escapes (absolute paths, `..`) are rejected.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from difflib import unified_diff
import errno
import os
from pathlib import Path
import sys
from typing import Annotated, Any, Iterator, Optional

from yian.compose.loader import load_flow
from yian.compose.helpers.errors import LoadError
from yian.run import _run_flow_controlled, run_flow as _run_flow
from yian.tools import ToolSet, register_tool, registered_tools

# The workspace root every tool path is confined to. Set at `yian chat` startup.
_WORKSPACE: Path = Path.cwd()
_CONTROLLER_STOP: ContextVar[Any | None] = ContextVar(
    "yian_chat_controller_stop",
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


_CONTROLLER_ASK_USER: ContextVar[Any | None] = ContextVar(
    "yian_chat_controller_ask_user",
    default=None,
)


@contextmanager
def _bind_ask_user(handler: Any) -> Iterator[None]:
    """Make the chat host's ``ask_user`` handler visible to flow-op tools.

    The engine runs pooled nodes in a copy of this context, so workers see it too.
    """
    token = _CONTROLLER_ASK_USER.set(handler)
    try:
        yield
    finally:
        _CONTROLLER_ASK_USER.reset(token)


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


def _approval_prompt_available() -> bool:
    """Return whether this CLI host has an interactive input terminal."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, OSError, ValueError):
        return False


def _secure_workspace_write_available() -> bool:
    """Return whether the host can anchor every path component to a directory fd."""
    required_dir_fd = {os.mkdir, os.open, os.stat, os.unlink}
    return (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and required_dir_fd.issubset(os.supports_dir_fd)
        and os.stat in os.supports_follow_symlinks
    )


@dataclass(frozen=True, slots=True)
class _TargetSnapshot:
    """Approval-time bytes and filesystem identity for one existing target."""

    content: bytes
    device: int
    inode: int
    link_count: int


def _target_snapshot(dest: Path) -> _TargetSnapshot | None:
    """Read target bytes and identity, using ``None`` only for a missing path."""
    relative = dest.relative_to(_WORKSPACE)
    try:
        opened_parent = _open_workspace_parent(relative, create=False)
    except OSError as exc:
        if exc.errno in _CHANGED_PATH_ERRNOS:
            return None
        raise
    if opened_parent is None:
        return None
    parent_fd, leaf = opened_parent
    descriptor = None
    try:
        try:
            descriptor = _open_no_follow(
                leaf,
                os.O_RDONLY | getattr(os, "O_BINARY", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if exc.errno in _CHANGED_PATH_ERRNOS:
                return None
            raise
        content = _read_fd(descriptor)
        identity = os.fstat(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    return _TargetSnapshot(
        content=content,
        device=identity.st_dev,
        inode=identity.st_ino,
        link_count=identity.st_nlink,
    )


def _target_is_unchanged(
    authored_path: str,
    dest: Path,
    snapshot: _TargetSnapshot | None,
) -> bool:
    """Re-confine the authored path and compare its exact current target bytes."""
    try:
        if _resolve(authored_path) != dest:
            return False
        return _target_snapshot(dest) == snapshot
    except (OSError, RuntimeError, ValueError):
        return False


def _open_no_follow(
    path: str | Path,
    flags: int,
    mode: int = 0o777,
    *,
    dir_fd: int | None = None,
) -> int:
    """Open one path component without following it when the platform supports it."""
    return os.open(
        path,
        flags | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=dir_fd,
    )


def _open_workspace_parent(relative: Path, *, create: bool) -> tuple[int, str] | None:
    """Open the target parent from an anchored workspace fd, never through symlinks."""
    parts = relative.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        return None

    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = _open_no_follow(_WORKSPACE, directory_flags)
    try:
        for part in parts[:-1]:
            try:
                child = _open_no_follow(part, directory_flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    os.close(current)
                    return None
                try:
                    os.mkdir(part, 0o777, dir_fd=current)
                except FileExistsError:
                    pass
                child = _open_no_follow(part, directory_flags, dir_fd=current)
            os.close(current)
            current = child
        return current, parts[-1]
    except BaseException:
        os.close(current)
        raise


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    """Return whether two stat records identify the same filesystem object."""
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_fd(descriptor: int) -> bytes:
    """Read one open regular file from its first byte."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(descriptor, 65536)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _replace_fd_contents(descriptor: int, content: bytes) -> None:
    """Replace one open regular file's bytes without resolving its path again."""
    os.lseek(descriptor, 0, os.SEEK_SET)
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written == 0:
            raise OSError("zero-byte write while committing approved flow")
        remaining = remaining[written:]
    os.ftruncate(descriptor, len(content))


def _target_binding_is_current(
    authored_path: str,
    dest: Path,
    parent_fd: int,
    leaf: str,
    leaf_fd: int,
) -> bool:
    """Verify that authored path, parent, and leaf still name the opened objects."""
    try:
        if _resolve(authored_path) != dest:
            return False
        if not _same_file(os.stat(dest.parent), os.fstat(parent_fd)):
            return False
        opened = os.fstat(leaf_fd)
        return opened.st_nlink == 1 and _same_file(
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False), opened
        )
    except (OSError, RuntimeError, ValueError):
        return False


def _snapshot_matches_fd(snapshot: _TargetSnapshot, descriptor: int) -> bool:
    """Match bytes, identity, and link count against the approval snapshot."""
    identity = os.fstat(descriptor)
    return (
        identity.st_dev == snapshot.device
        and identity.st_ino == snapshot.inode
        and identity.st_nlink == snapshot.link_count == 1
        and _read_fd(descriptor) == snapshot.content
    )


def _rollback_created_leaf(parent_fd: int, leaf: str, descriptor: int) -> None:
    """Restore a newly created inode to empty and unlink our still-current leaf."""
    _replace_fd_contents(descriptor, b"")
    opened = os.fstat(descriptor)
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if _same_file(opened, current):
        os.unlink(leaf, dir_fd=parent_fd)


_CHANGED_PATH_ERRNOS = frozenset(
    {
        errno.EISDIR,
        errno.ELOOP,
        errno.ENOENT,
        errno.ENOTDIR,
    }
)


def _commit_flow_write(
    authored_path: str,
    dest: Path,
    content: str,
    snapshot: _TargetSnapshot | None,
) -> bool:
    """Commit through workspace-anchored fds or roll back a changed binding."""
    relative = dest.relative_to(_WORKSPACE)
    try:
        opened_parent = _open_workspace_parent(relative, create=snapshot is None)
    except OSError as exc:
        if exc.errno in _CHANGED_PATH_ERRNOS:
            return False
        raise
    if opened_parent is None:
        return False
    parent_fd, leaf = opened_parent
    descriptor = None
    try:
        file_flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        if snapshot is None:
            file_flags |= os.O_CREAT | os.O_EXCL
        try:
            descriptor = _open_no_follow(
                leaf,
                file_flags,
                0o666,
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if isinstance(exc, FileExistsError) or exc.errno in _CHANGED_PATH_ERRNOS:
                return False
            raise

        if snapshot is None:
            if os.fstat(descriptor).st_nlink != 1:
                _rollback_created_leaf(parent_fd, leaf, descriptor)
                return False
        elif not _snapshot_matches_fd(snapshot, descriptor):
            return False
        _replace_fd_contents(descriptor, content.encode())
        if _target_binding_is_current(
            authored_path,
            dest,
            parent_fd,
            leaf,
            descriptor,
        ):
            return True

        if snapshot is None:
            _rollback_created_leaf(parent_fd, leaf, descriptor)
        else:
            _replace_fd_contents(descriptor, snapshot.content)
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _line_ending_labels(text: str) -> tuple[str, ...]:
    """Return one visible line-ending label for every logical source line."""
    labels = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\r\n"):
            labels.append("CRLF")
        elif line.endswith("\n"):
            labels.append("LF")
        elif line.endswith("\r"):
            labels.append("CR")
        else:
            labels.append("no newline")
    return tuple(labels)


def _diff_lines(text: str, *, show_endings: bool) -> list[str]:
    """Split source for a unified diff, optionally making line endings visible."""
    if not show_endings:
        return text.splitlines()

    lines = []
    for line, ending in zip(
        text.splitlines(keepends=True),
        _line_ending_labels(text),
    ):
        if ending == "CRLF":
            body = line[:-2]
        elif ending in {"LF", "CR"}:
            body = line[:-1]
        else:
            body = line
        lines.append(f"{body} [{ending}]")
    return lines


def _write_flow_preview(
    dest: Path,
    content: str,
    snapshot: _TargetSnapshot | None,
) -> str:
    """Return the target-labelled create content or overwrite diff for approval."""
    relative = dest.relative_to(_WORKSPACE)
    if snapshot is None:
        return f"Create {relative} with this validated content:\n\n{content}"

    current = snapshot.content.decode()
    show_endings = _line_ending_labels(current) != _line_ending_labels(content)
    diff = "\n".join(
        unified_diff(
            _diff_lines(current, show_endings=show_endings),
            _diff_lines(content, show_endings=show_endings),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
            lineterm="",
        )
    )
    if not diff:
        diff = "(content is unchanged)"
    elif show_endings:
        diff += "\n\nLine endings are shown as [LF], [CRLF], [CR], or [no newline]."
    return f"Overwrite {relative} with this validated diff:\n\n{diff}"


def _approve_flow_write(
    dest: Path,
    content: str,
    snapshot: _TargetSnapshot | None,
) -> bool | None:
    """Return exact host approval, or ``None`` when no host owns interaction."""
    handler = _CONTROLLER_ASK_USER.get()
    if handler is None:
        return None

    answer = ToolSet({"ask_user": handler}).invoke_trusted(
        "ask_user",
        {
            "question": _write_flow_preview(dest, content, snapshot),
            "questions": [
                {
                    "question": "Write this validated flow?",
                    "header": "Decision",
                    "options": [
                        {
                            "label": "Approve",
                            "description": "Commit the shown file change.",
                        },
                        {
                            "label": "Reject",
                            "description": "Leave the workspace unchanged.",
                        },
                    ],
                    "multi_select": False,
                }
            ],
            "title": "Flow write approval",
        },
    )
    return answer == {"Decision": "Approve"}


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
            search_paths=[source.parent],
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
            search_paths=[source.parent],
            source_label=source.name,
        )
    except LoadError as err:
        return f"ERROR: {err}"
    handler = _CONTROLLER_ASK_USER.get()
    # A tools mapping hides the registry, so keep the registered tools next to the handler.
    tools = None if handler is None else {**registered_tools(), "ask_user": handler}
    parent_stop = _CONTROLLER_STOP.get()
    try:
        if parent_stop is None:
            # Direct Python use: private lifecycle.
            result = _run_flow(loaded, inputs or {}, tools=tools)
        else:
            # Outer host cancellation flows down; a nested NodeFailed sets only this
            # child, preserving the failure as a model-visible tool result.
            with parent_stop.linked_child() as child_stop:
                result = _run_flow_controlled(
                    loaded,
                    inputs or {},
                    stop_event=child_stop,
                    tools=tools,
                )
    except (TypeError, ValueError) as exc:
        # Bad inputs or an unresolvable tool grant fail at run start, before any
        # record exists; report them to the model like a failed run.
        return f"FAILED: {exc}"
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
    compile-checked and shown to the human for explicit approval BEFORE anything is
    written. Returns `WROTE: <path>` on approval, `NOT WRITTEN: <path> (...)` when
    approval is non-interactive, rejected, unavailable, stale, multi-linked, or lacks a
    secure host commit primitive, or `INVALID (not written): <compile error>` if it
    doesn't compile."""
    dest = _resolve(path)  # raises on escape
    try:
        load_flow(
            content,
            search_paths=[dest.parent],
            source_label=dest.name,
        )
    except LoadError as err:
        return f"INVALID (not written): {err}"
    relative = dest.relative_to(_WORKSPACE)
    if not _approval_prompt_available():
        return f"NOT WRITTEN: {relative} (interactive approval requires a TTY)"
    if not _secure_workspace_write_available():
        return (
            f"NOT WRITTEN: {relative} "
            "(secure workspace writes are unsupported on this platform)"
        )
    snapshot = _target_snapshot(dest)
    if snapshot is not None and snapshot.link_count != 1:
        return f"NOT WRITTEN: {relative} (target has multiple hard links)"
    approved = _approve_flow_write(dest, content, snapshot)
    if approved is None:
        return f"NOT WRITTEN: {relative} (host approval is unavailable)"
    if not approved:
        return f"NOT WRITTEN: {relative} (approval was not granted)"
    if not _target_is_unchanged(path, dest, snapshot):
        return f"NOT WRITTEN: {relative} (target changed during approval)"
    if not _commit_flow_write(path, dest, content, snapshot):
        return f"NOT WRITTEN: {relative} (target changed during approval)"
    return f"WROTE: {relative}"
