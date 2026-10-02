"""Build the memories a flow declares and own their lifetime for one host.

This module sits beside `run.py`, not under `memory/`, because it reads the
compiled memory table (`compile`) and `memory/` may import only lower memory
modules. `run_flow` never constructs memory; a host calls `build_memories` and
passes the returned registry as ``memory=``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType, TracebackType

from agent_composer.compile.memory_layout import ResolvedMemory, resolve_memory_layout
from agent_composer.compose.loader import ComposedFlow
from agent_composer.memory.contracts import (
    BUILTIN_MEMORY_CONFIG_KEYS,
    Memory,
    MemoryOperationError,
    MemorySetupError,
)


def _note_cleanup(target: BaseException, cleanup: BaseException, label: str) -> None:
    """Attach one cleanup failure, and the notes it carries, to ``target``."""
    if hasattr(target, "add_note"):
        target.add_note(f"{label}: {type(cleanup).__name__}: {cleanup}")
        for note in getattr(cleanup, "__notes__", ()):
            target.add_note(note)


def _close_all(
    objects: Iterable[object], failed: list[object] | None = None
) -> BaseException | None:
    """Close each object that has a callable ``close``; return the first failure.

    Later failures are attached to the first one as notes. Objects whose close
    raised are appended to ``failed`` when it is given.
    """
    first: BaseException | None = None
    for item in objects:
        close = getattr(item, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except BaseException as exc:
            if failed is not None:
                failed.append(item)
            if first is None:
                first = exc
            else:
                _note_cleanup(first, exc, "Additional memory cleanup failed")
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
        """Close owned backends in reverse build order.

        A backend that closed is not closed again. One whose close raised stays
        owned, so a later call retries it; once all closed, calls do nothing.

        Raises:
            `BaseException`: The first close failure, with later ones as notes.
        """
        failed: list[object] = []
        first = _close_all(reversed(self._owned), failed)
        self._owned = tuple(reversed(failed))
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
            _note_cleanup(exc, cleanup, "Memory cleanup failed")


def _error(memory: ResolvedMemory, message: str) -> MemorySetupError:
    """Attribute one setup failure to the key's first declaration site."""
    return MemorySetupError(
        f"memory {memory.key!r}: {message}",
        origin=memory.origin,
        related=tuple(origin for origin in memory.origins[1:] if origin is not None),
    )


def _builtin(
    type_name: str, config: dict, path: str | None, create_parent: bool
) -> Memory:
    """Construct one built-in backend, importing only the selected module."""
    if type_name == "episodic":
        from agent_composer.memory.episodic import EpisodicMemory

        return EpisodicMemory(**config, storage=path, create_parent=create_parent)
    if type_name == "reflective":
        from agent_composer.memory.reflective import ReflectiveMemory

        return ReflectiveMemory(**config, storage=path, create_parent=create_parent)
    if type_name == "session":
        from agent_composer.memory.session import SessionMemory

        return SessionMemory(**config, storage=path, create_parent=create_parent)
    from agent_composer.memory.profile import ProfileMemory
    from agent_composer.memory.storage.markdown import (
        FileMarkdownStore,
        InMemoryMarkdownStore,
    )

    store = (
        InMemoryMarkdownStore()
        if path is None
        else FileMarkdownStore(path, create_parent=create_parent)
    )
    return ProfileMemory(store, **config)


def build_memories(
    flow: ComposedFlow,
    *,
    root: str | PathLike[str] | None = None,
    storage: Mapping[str, str | PathLike[str]] | None = None,
    supplied: Mapping[str, Memory] | None = None,
    factories: Mapping[str, Callable[..., Memory]] | None = None,
) -> MemoryRegistry:
    """Build one backend for every memory key the flow declares.

    For each key, in key order: ``supplied[key]`` wins; else a custom type is
    built by ``factories[type](**config)``; else the built-in type is built
    with ``config`` as keywords and storage from ``storage[key]``, else
    ``<root>/<flow id>/<scope...>/<name>.sqlite`` (``.md`` for profile, parent
    created on first use), else process RAM. Nothing touches the filesystem
    until a backend's first operation.

    Args:
        flow (`ComposedFlow`): Loaded flow whose memory table is read.
        root (`str | PathLike[str] | None`, *optional*): Directory holding
            per-flow memory files. Requires the flow to have a safe ``id``.
        storage (`Mapping[str, str | PathLike[str]] | None`, *optional*):
            Explicit file per built-in key. The parent must already exist.
        supplied (`Mapping[str, Memory] | None`, *optional*): Ready objects by
            key. The registry does not close them.
        factories (`Mapping[str, Callable[..., Memory]] | None`, *optional*):
            Constructors for custom types, called with the declared config.

    Returns:
        `MemoryRegistry`: Mapping from memory key to backend, plus declared types.

    Raises:
        `TypeError`: ``flow`` is not a `ComposedFlow`.
        `MemorySetupError`: The table is invalid, or a key cannot be built as
            asked (including a storage value that is not a str or
            ``PathLike[str]``, a path that is not a local file path, or two
            built-in keys whose storage paths are equal, ignoring case).
            Anything already built is closed before the error propagates.
    """
    if not isinstance(flow, ComposedFlow):
        raise TypeError("flow must be a ComposedFlow")
    base: Path | None = None
    if root is not None:
        flow_id = flow.compiled.flow_id
        if flow_id is None:
            raise MemorySetupError(
                "root= needs the flow to have an id: field; its memory lives "
                "under <root>/<flow id>/"
            )
        if flow_id in ("", ".", "..") or any(ch in flow_id for ch in "/\\\x00"):
            raise MemorySetupError(
                f"flow id {flow_id!r} is not one safe path component for root="
            )
        base = Path(os.fspath(root), flow_id)
    storage = dict(storage or {})
    supplied = dict(supplied or {})
    factories = dict(factories or {})
    table = resolve_memory_layout(flow.compiled).memories
    declared = {memory.key for memory in table}
    for name, given in (("storage", storage), ("supplied", supplied)):
        extra = sorted(set(given) - declared)
        if extra:
            raise MemorySetupError(
                f"{name} names memory key(s) {extra} not declared by this flow "
                f"(declared: {sorted(declared)})"
            )
    for key, path in storage.items():
        # Backends take text paths only. Checked here so other values are not
        # reported as a config error, and None does not silently mean RAM.
        if not isinstance(path, (str, PathLike)) or not isinstance(os.fspath(path), str):
            raise MemorySetupError(
                f"storage[{key!r}] must be a str or os.PathLike[str] path, "
                f"not {type(path).__name__}"
            )
    shadowed = sorted(set(factories) & set(BUILTIN_MEMORY_CONFIG_KEYS))
    if shadowed:
        raise MemorySetupError(
            f"factories may not name built-in memory type(s) {shadowed}; "
            "use supplied= to replace one built-in instance"
        )
    memories: dict[str, Memory] = {}
    owned: list[Memory] = []
    paths: dict[str, str] = {}
    try:
        for memory in table:
            memories[memory.key] = _build_one(
                memory, storage, supplied, factories, owned, base, paths
            )
    except BaseException as primary:
        # Construction is effect-free, so this cleanup does no I/O.
        cleanup = _close_all(reversed(owned))
        if cleanup is not None:
            _note_cleanup(primary, cleanup, "Memory cleanup failed")
        raise
    return MemoryRegistry(
        memories, {memory.key: memory.type for memory in table}, owned=tuple(owned)
    )


def _build_one(
    memory: ResolvedMemory,
    storage: dict,
    supplied: dict,
    factories: dict,
    owned: list[Memory],
    base: Path | None,
    paths: dict[str, str],
) -> Memory:
    """Return the backend for one key, appending it to ``owned`` if built here.

    ``paths`` maps each normalized built-in storage path seen so far to its key.
    """
    builtin = memory.type in BUILTIN_MEMORY_CONFIG_KEYS
    if memory.key in supplied:
        if memory.key in storage:
            raise _error(
                memory,
                "storage= applies only to built-ins this call constructs; this key is supplied",
            )
        if memory.config:
            raise _error(
                memory,
                "a supplied memory cannot take YAML config; "
                "remove config or build it with factories=",
            )
        value = supplied[memory.key]
        if not isinstance(value, Memory):
            raise _error(memory, f"supplied value is not a Memory ({type(value).__name__})")
        return value
    if not builtin:
        if memory.key in storage:
            raise _error(memory, f"storage= applies only to built-in types, not {memory.type!r}")
        factory = factories.get(memory.type)
        if factory is None:
            raise _error(memory, f"custom type {memory.type!r} has no factory or supplied value")
        try:
            value = factory(**memory.config)
        except (TypeError, ValueError) as exc:
            raise _error(memory, f"{memory.type} rejected its config: {exc}") from exc
        if not isinstance(value, Memory):
            error = _error(
                memory,
                f"factory for {memory.type!r} returned a {type(value).__name__}, not a Memory",
            )
            # The factory built it, so this call owns it even though it is rejected.
            cleanup = _close_all([value])
            if cleanup is not None:
                _note_cleanup(error, cleanup, "Memory cleanup failed")
            raise error
        owned.append(value)
        return value
    path, create_parent = storage.get(memory.key), False
    if path is None and base is not None:
        *scope, name = memory.key.split(".")
        suffix = ".md" if memory.type == "profile" else ".sqlite"
        path, create_parent = base.joinpath(*scope, name + suffix), True
    try:
        value = _builtin(
            memory.type,
            memory.config,
            None if path is None else os.fspath(path),
            create_parent=create_parent,
        )
    except MemoryOperationError as exc:
        raise _error(
            memory, f"storage {os.fspath(path)!r} is not a local file path"
        ) from exc
    except (TypeError, ValueError) as exc:
        raise _error(memory, f"{memory.type} rejected its config: {exc}") from exc
    owned.append(value)
    if path is not None:
        # Case-folded so two keys cannot share one file on a case-insensitive
        # filesystem; realpath reads links but opens nothing.
        normal = os.path.normcase(os.path.realpath(os.fspath(path))).casefold()
        other = paths.setdefault(normal, memory.key)
        if other != memory.key:
            raise _error(
                memory, f"storage {os.fspath(path)!r} is also used by memory {other!r}"
            )
    return value


__all__ = ["MemoryRegistry", "build_memories"]
