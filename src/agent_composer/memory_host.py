"""Build the memories a flow declares and own their lifetime for one host.

This module sits beside `run.py`, not under `memory/`, because it reads the
compiled memory table (`compile`) and `memory/` may import only lower memory
modules. `run_flow` never constructs memory; a host calls `build_memories` and
passes the returned registry as ``memory=``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType, TracebackType

from agent_composer.memory.contracts import Memory


def _close_all(objects: Iterable[object]) -> BaseException | None:
    """Close each object that has a callable ``close``; return the first failure.

    Later failures are attached to the first one as notes.
    """
    first: BaseException | None = None
    for item in objects:
        close = getattr(item, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            if first is None:
                first = exc
            elif hasattr(first, "add_note"):
                first.add_note(f"Additional memory cleanup failed: {type(exc).__name__}")
    return first


class MemoryRegistry(Mapping[str, Memory]):
    """Read-only memory-key -> backend mapping that owns what it constructed.

    Attributes:
        types (`Mapping[str, str]`): Declared type of each key, from the flow's
            memory table. `run_flow` compares it with the table of the flow it runs.
    """

    def __init__(
        self,
        memories: Mapping[str, Memory],
        types: Mapping[str, str],
        *,
        owned: tuple[Memory, ...] = (),
    ) -> None:
        """Wrap built memories; ``owned`` lists, in build order, what close() closes.

        Args:
            memories (`Mapping[str, Memory]`): Backend for each memory key.
            types (`Mapping[str, str]`): Declared type for each memory key.
            owned (`tuple[Memory, ...]`, *optional*, defaults to `()`): Objects
                this registry constructed. Supplied objects stay with the caller.
        """
        self._memories = dict(memories)
        self.types: Mapping[str, str] = MappingProxyType(dict(types))
        self._owned = tuple(owned)
        self._closed = False

    def __getitem__(self, key: str) -> Memory:
        """Return the backend for one memory key."""
        return self._memories[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate memory keys in table order."""
        return iter(self._memories)

    def __len__(self) -> int:
        """Return the number of memory keys."""
        return len(self._memories)

    def close(self) -> None:
        """Close owned backends in reverse build order; repeated calls do nothing.

        Raises:
            `BaseException`: The first close failure, with later ones as notes.
        """
        if self._closed:
            return
        self._closed = True
        first = _close_all(reversed(self._owned))
        if first is not None:
            raise first

    def __enter__(self) -> MemoryRegistry:
        """Return this registry; exit closes it."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the registry without replacing an exception already in flight."""
        try:
            self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            if hasattr(exc, "add_note"):
                exc.add_note(f"Memory cleanup failed: {type(cleanup).__name__}")


__all__ = ["MemoryRegistry"]
