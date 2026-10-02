"""Select and own lazy built-in memory backends for one CLI controller."""

from __future__ import annotations

from collections.abc import Iterable
from types import TracebackType

from agent_composer.compile.memory_layout import resolve_memory_layout
from agent_composer.compile.model import CompiledFlow
from agent_composer.memory.contracts import (
    JsonObject,
    Memory,
    MemorySession,
    MemorySetupError,
)

_BUILTINS = frozenset({"episodic", "reflective", "session"})


_CONFIG_KEYS = {
    "episodic": frozenset({"max_episodes", "recall_limit"}),
    "reflective": frozenset({"max_items", "recall_limit"}),
    "session": frozenset({"history_budget_chars", "recent_turns"}),
}
"""YAML config keys each CLI-built memory type accepts; learner models are excluded."""


def _create_backend(memory_type: str, storage: str | None, config: JsonObject) -> Memory:
    """Import and construct only the selected built-in memory type with its config.

    Raises:
        `MemorySetupError`: The type has no built-in CLI backend.
        `TypeError` / `ValueError`: The constructor rejects ``config``.
    """
    if memory_type == "episodic":
        from agent_composer.memory.episodic import EpisodicMemory as backend
    elif memory_type == "reflective":
        from agent_composer.memory.reflective import ReflectiveMemory as backend
    elif memory_type == "session":
        from agent_composer.memory.session import SessionMemory as backend
    else:
        raise MemorySetupError(
            f"Memory policy {memory_type!r} has no built-in CLI backend. "
            "Use the Python API with memory={policy_name: implementation}."
        )
    return backend(storage=storage, **config)


class _LazyMemory:
    """Defer backend effects until the existing run boundary opens a policy."""

    def __init__(
        self, key: str, memory_type: str, storage: str | None, config: JsonObject
    ) -> None:
        self._key = key
        self._type = memory_type
        self._storage = storage
        self._config = config
        self._backend: Memory | None = None
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise MemorySetupError("CLI memory host is closed")

    def open_run(self, *, snapshot: JsonObject | None = None) -> MemorySession:
        """Create the backend once and pass the exact requested opening pin.

        Raises:
            `MemorySetupError`: The type has no built-in CLI backend, or its
                config has a key outside the type's allowlist or a value the
                constructor rejects.
        """
        self._check_open()
        if self._backend is None:
            allowed = _CONFIG_KEYS.get(self._type)
            unknown = sorted(set(self._config) - allowed) if allowed is not None else []
            if unknown:
                raise MemorySetupError(
                    f"memory {self._key!r} config has unknown key(s) {unknown}; "
                    f"allowed: {', '.join(sorted(allowed))}"
                )
            try:
                self._backend = _create_backend(self._type, self._storage, self._config)
            except (TypeError, ValueError) as exc:
                raise MemorySetupError(
                    f"memory {self._key!r} config was rejected: {exc}"
                ) from exc
        return self._backend.open_run(snapshot=snapshot)

    def reset(self) -> None:
        """Reset an already constructed backend without opening unused policies."""
        self._check_open()
        if self._backend is not None:
            self._backend.reset()

    def close(self) -> None:
        """Invalidate the wrapper and close its concrete backend, if any."""
        if self._closed:
            return
        self._closed = True
        if self._backend is not None:
            self._backend.close()


class MemoryHost:
    """Own declared CLI memory policies without opening resources during setup.

    Args:
        compiled (`CompiledFlow`): Loaded authored graph used for pure policy discovery.
        storage (`Iterable[str]`): Repeated `POLICY=PATH` options; omitted policies
            remain process-local. Paths are passed unchanged to their backend.

    Raises:
        `ValueError`: A storage option is malformed, repeated, unknown, or undeclared.
        `MemorySetupError`: The compiled memory layout is invalid.
    """

    def __init__(self, compiled: CompiledFlow, storage: Iterable[str]) -> None:
        policies = {
            memory.key: (memory.type, memory.spec.config)
            for memory in resolve_memory_layout(compiled).memories
        }
        paths: dict[str, str] = {}
        for option in storage:
            policy, separator, path = option.partition("=")
            if not separator or not policy.strip() or not path.strip():
                raise ValueError("--memory-storage requires POLICY=PATH")
            if policy in paths:
                raise ValueError(f"duplicate --memory-storage policy {policy!r}")
            if policy not in _BUILTINS:
                raise ValueError(
                    f"--memory-storage policy {policy!r} is not a built-in "
                    "policy (episodic, reflective, session)"
                )
            if policy not in policies:
                raise ValueError(
                    f"--memory-storage policy {policy!r} is not declared in this flow"
                )
            paths[policy] = path
        self._policies = {
            key: _LazyMemory(key, memory_type, paths.get(key), config)
            for key, (memory_type, config) in policies.items()
        }
        self.registry: dict[str, Memory] | None = dict(self._policies) or None
        self._closed = False

    def __enter__(self) -> MemoryHost:
        if self._closed:
            raise MemorySetupError("CLI memory host is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"CLI memory cleanup failed: {type(cleanup).__name__}")

    def close(self) -> None:
        """Close every constructed backend, preserving the first cleanup failure."""
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for policy in reversed(tuple(self._policies.values())):
            try:
                policy.close()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                elif hasattr(first_error, "add_note"):
                    first_error.add_note(
                        f"Additional CLI memory cleanup failed: {type(exc).__name__}"
                    )
        if first_error is not None:
            raise first_error
