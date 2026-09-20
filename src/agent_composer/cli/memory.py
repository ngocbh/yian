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


def _create_backend(policy: str, storage: str | None) -> Memory:
    """Import and construct only the selected built-in policy."""
    if policy == "episodic":
        from agent_composer.memory.episodic import EpisodicMemory

        return EpisodicMemory(storage=storage)
    if policy == "reflective":
        from agent_composer.memory.reflective import ReflectiveMemory

        return ReflectiveMemory(storage=storage)
    if policy == "session":
        from agent_composer.memory.session import SessionMemory

        return SessionMemory(storage=storage)
    raise MemorySetupError(
        f"Memory policy {policy!r} has no built-in CLI backend. "
        "Use the Python API with memory={policy_name: implementation}."
    )


class _LazyMemory:
    """Defer backend effects until the existing run boundary opens a policy."""

    def __init__(self, policy: str, storage: str | None) -> None:
        self._policy = policy
        self._storage = storage
        self._backend: Memory | None = None
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise MemorySetupError("CLI memory host is closed")

    def open_run(self, *, snapshot: JsonObject | None = None) -> MemorySession:
        """Create the backend once and pass the exact requested opening pin."""
        self._check_open()
        if self._backend is None:
            self._backend = _create_backend(self._policy, self._storage)
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
        policies = dict.fromkeys(
            frame.spec.policy_name for frame in resolve_memory_layout(compiled).frames
        )
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
            policy: _LazyMemory(policy, paths.get(policy)) for policy in policies
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
