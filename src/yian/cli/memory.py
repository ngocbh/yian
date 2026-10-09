"""Turn `yian run` memory flags into one host-built memory registry.

The CLI builds only built-in memory types. A flow with a custom type fails
here, before any run or resume, even a terminal ``--resume`` that would open
no memory; such flows run through the Python API.
"""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike

from yian.compile.memory_layout import resolve_memory_layout
from yian.compose.loader import ComposedFlow
from yian.memory.contracts import BUILTIN_MEMORY_CONFIG_KEYS, MemorySetupError
from yian.memory_host import MemoryRegistry, build_memories


class MemoryOptionError(ValueError):
    """A memory flag value is invalid; ``param_hint`` names the flag."""

    def __init__(self, message: str, *, param_hint: str) -> None:
        """Record the message and the flag it belongs to."""
        super().__init__(message)
        self.param_hint = param_hint


def parse_memory_storage(options: Iterable[str]) -> dict[str, str]:
    """Parse repeated ``KEY=PATH`` options into a key -> path mapping.

    Surrounding whitespace is stripped from KEY; PATH is kept as given.

    Raises:
        `MemoryOptionError`: An option is malformed or repeats a key.
    """
    paths: dict[str, str] = {}
    for option in options:
        key, separator, path = option.partition("=")
        key = key.strip()
        if not separator or not key or not path.strip():
            raise MemoryOptionError(
                f"expected KEY=PATH, got {option!r}",
                param_hint="--memory-storage",
            )
        if key in paths:
            raise MemoryOptionError(
                f"duplicate key {key!r}", param_hint="--memory-storage"
            )
        paths[key] = path
    return paths


def build_cli_memories(
    flow: ComposedFlow,
    *,
    root: str | PathLike[str] | None,
    storage_options: Iterable[str],
) -> MemoryRegistry:
    """Build every declared memory from the CLI flags.

    Args:
        flow (`ComposedFlow`): Loaded flow.
        root (`str | PathLike[str] | None`): ``--memory-root`` value.
        storage_options (`Iterable[str]`): Repeated ``--memory-storage`` values.

    Returns:
        `MemoryRegistry`: Registry to pass to the run and close afterwards.

    Raises:
        `MemoryOptionError`: A flag value is invalid for this flow.
        `MemorySetupError`: The flow declares a custom type, or building fails.
    """
    storage = parse_memory_storage(storage_options)
    types = {m.key: m.type for m in resolve_memory_layout(flow.compiled).memories}
    builtins = ", ".join(sorted(BUILTIN_MEMORY_CONFIG_KEYS))
    for key in storage:
        if key not in types:
            raise MemoryOptionError(
                f"key {key!r} is not a memory declared by this "
                f"flow (declared: {', '.join(sorted(types)) or 'none'})",
                param_hint="--memory-storage",
            )
        if types[key] not in BUILTIN_MEMORY_CONFIG_KEYS:
            raise MemoryOptionError(
                f"key {key!r} has custom type {types[key]!r}; "
                f"the CLI stores only built-in types ({builtins})",
                param_hint="--memory-storage",
            )
    custom = sorted(k for k, t in types.items() if t not in BUILTIN_MEMORY_CONFIG_KEYS)
    if custom:
        raise MemorySetupError(
            f"this flow declares custom memory type(s) for {custom}; yian run "
            f"builds only built-in types ({builtins}). Use the Python API: "
            "registry = build_memories(flow, factories=... or supplied=...), then "
            "run_flow(flow, inputs, memory=registry)."
        )
    if root is not None and flow.compiled.flow_id is None:
        raise MemoryOptionError(
            "--memory-root needs a flow with an id: field; memory lives under "
            "DIR/<flow id>/",
            param_hint="--memory-root",
        )
    return build_memories(flow, root=root, storage=storage)


__all__ = ["MemoryOptionError", "build_cli_memories", "parse_memory_storage"]
