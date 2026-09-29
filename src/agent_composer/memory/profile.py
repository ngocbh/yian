"""Markdown profile memory: one versioned document per memory instance.

``ProfileMemory`` recalls one markdown document into every AGENT under its
outermost frame and, on backward, asks one model call to rewrite the whole
document from that run's feedback. Commits are compare-and-set writes to a
``ProfileStore`` and never call a model.
"""

from __future__ import annotations

import hashlib
import json
from threading import Lock
from types import TracebackType

from langchain_core.messages import HumanMessage, SystemMessage

from agent_composer.llm_clients import model_from_config
from agent_composer.llm_clients.connection import resolve_model_config
from agent_composer.memory.contracts import (
    JsonObject,
    JsonValue,
    LearnContext,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationError,
    RecallContext,
)
from agent_composer.memory.profile_store import ProfileStore, ProfileVersionMismatch
from agent_composer.memory.reflective import (
    _bounded_json_material,
    _canonical_json,
    _visible_interaction_json,
)

_SNAPSHOT_FORMAT = "agent_composer.profile_memory"
_SNAPSHOT_VERSION = 1
_SNAPSHOT_KEYS = frozenset({"format", "version", "profile_version", "text_sha256"})
_OPEN_TAG = "<profile>"
_CLOSE_TAG = "</profile>"
_SYSTEM = (
    "You maintain one markdown profile document. The user message is JSON with the "
    "current document, the older document the reviewed run saw when it differs, "
    "the feedback on that run, and recent agent episodes with their visible model "
    "and tool calls. Rewrite the complete document so later runs follow the "
    "feedback. Keep content the feedback does not contradict, including edits "
    "people made directly. The current document may already reflect this feedback, "
    "for example after an interrupted earlier attempt or a person's edit: never "
    "repeat what it already says, and return it unchanged if nothing needs to "
    "change. Stay under max_chars characters. Never call tools. Reply with the "
    f"complete new document between {_OPEN_TAG} and {_CLOSE_TAG} and nothing else."
)


def _sha256(text: str) -> str:
    """Return the hex SHA-256 of one UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_strict_prefix(prefix: tuple[str, ...], path: tuple[str, ...]) -> bool:
    """Return whether ``prefix`` is a proper leading part of ``path``."""
    return len(prefix) < len(path) and path[: len(prefix)] == prefix


def _error(stage: str, code: str, message: str, *, retryable: bool = False) -> MemoryOperationError:
    """Build one typed profile failure."""
    return MemoryOperationError(stage, code, message, retryable)


def _read_error(stage: str, exc: Exception) -> MemoryOperationError:
    """Map a store read failure: I/O is retryable, an unreadable document is not."""
    if isinstance(exc, OSError):
        return _error(stage, "profile.store_unavailable", "profile store could not be read",
                      retryable=True)
    return _error(stage, "profile.store_invalid", "profile store returned an unreadable document")


def _parse_reply(reply: object, current_text: str, max_chars: int) -> str:
    """Extract and bound the document the learner wrote between the profile tags.

    Args:
        reply (`object`): Model reply; its ``content`` is text or text blocks.
        current_text (`str`): Current document; a blank reply may not erase it.
        max_chars (`int`): Largest allowed document.

    Returns:
        `str`: The new document with one trailing newline, or ``""``.

    Raises:
        `MemoryOperationError`: ``profile.contract`` for a reply without tags or a
            blank reply over a non-empty document, ``profile.over_budget`` for a
            document longer than ``max_chars``.
    """
    content = getattr(reply, "content", reply)
    if isinstance(content, list):
        content = "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content if isinstance(block, (str, dict))
        )
    if type(content) is not str:
        raise _error("learn", "profile.contract", "profile learner returned no text")
    start, end = content.find(_OPEN_TAG), content.rfind(_CLOSE_TAG)
    if start < 0 or end < start:
        raise _error("learn", "profile.contract",
                     f"profile learner reply must wrap the document in {_OPEN_TAG} tags")
    text = content[start + len(_OPEN_TAG):end].strip()
    if not text and current_text.strip():
        raise _error("learn", "profile.contract", "profile learner returned an empty document")
    if len(text) > max_chars:
        raise _error("learn", "profile.over_budget",
                     f"profile learner returned {len(text)} characters; the limit is {max_chars}")
    return text + "\n" if text else ""


class ProfileMemory:
    """Policy backend that keeps one markdown profile in a versioned store.

    Inject one instance per profile under a host-chosen policy name, for example
    ``run_flow(flow, inputs, memory={"profile": ProfileMemory(store)})``. Only the
    outermost frame of that policy recalls and learns; nested frames of the same
    policy recall nothing, so the document appears once per AGENT prompt.
    """

    def __init__(
        self,
        store: ProfileStore,
        *,
        llm_config: JsonObject | None = None,
        max_chars: int = 32_000,
        input_budget_chars: int = 24_000,
    ) -> None:
        """Bind the backend to one store and learner model.

        Args:
            store (`ProfileStore`): Versioned document store, such as
                `FileProfileStore`.
            llm_config (`JsonObject | None`, *optional*, defaults to `None`):
                Learner model config. `None` uses the ambient defaults when a
                backward attempt learns.
            max_chars (`int`, *optional*, defaults to `32_000`): Largest document
                the learner may write (about 8k tokens). Host writes are not bounded.
            input_budget_chars (`int`, *optional*, defaults to `24_000`): Character
                budget for episode JSON sent to the learner. Feedback is always sent.

        Raises:
            `TypeError`: ``store`` lacks ``read``/``write`` or ``llm_config`` is not
                a dict.
            `ValueError`: A budget is not a positive integer.
        """
        if not isinstance(store, ProfileStore):
            raise TypeError("store must provide read() and write()")
        if llm_config is not None and type(llm_config) is not dict:
            raise TypeError("llm_config must be a dict or None")
        for name, value in (("max_chars", max_chars), ("input_budget_chars", input_budget_chars)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._store = store
        self._llm_config = dict(llm_config or {})
        self.max_chars = max_chars
        self.input_budget_chars = input_budget_chars

    def read(self, version: str | None = None) -> tuple[str, str]:
        """Return ``(text, version)`` from the store for host display or editing.

        Args:
            version (`str | None`, *optional*, defaults to `None`): Earlier version,
                or `None` for the current document.

        Returns:
            `tuple[str, str]`: Document text and version.

        Raises:
            `LookupError`: The named version is unknown.
        """
        return self._store.read(version)

    def write(self, text: str, *, base_version: str, provenance: JsonObject | None = None) -> str:
        """Write a host or human edit with compare-and-set, without a budget check.

        Args:
            text (`str`): Complete new document.
            base_version (`str`): Version the editor started from.
            provenance (`JsonObject | None`, *optional*, defaults to `None`):
                JSON describing the edit.

        Returns:
            `str`: The new version.

        Raises:
            `ProfileVersionMismatch`: ``base_version`` is not current.
        """
        return self._store.write(text, base_version=base_version, provenance=provenance)

    def reset(self) -> None:
        """Refuse engine resets; delete or rewrite the profile through the store.

        Raises:
            `MemoryOperationError`: Always, with code ``profile.reset_unsupported``.
        """
        raise _error("reset", "profile.reset_unsupported",
                     "profile memory is reset through its store, not the engine")

    def open_run(self, *, snapshot: JsonObject | None = None) -> _ProfileSession:
        """Pin the current document, or the recorded version when restoring.

        Args:
            snapshot (`JsonObject | None`, *optional*, defaults to `None`): A
                snapshot recorded by an earlier session of this backend.

        Returns:
            `_ProfileSession`: Session whose recall uses the pinned document.

        Raises:
            `MemoryOperationError`: ``profile.store_unavailable`` (retryable) when
                the store cannot be reached, ``profile.store_invalid`` when its
                document cannot be decoded, or ``profile.snapshot_unavailable``
                when the recorded version is malformed, unknown, or changed.
        """
        if snapshot is None:
            try:
                text, version = self._store.read(None)
            except (LookupError, OSError, ValueError) as exc:
                raise _read_error("open_run", exc) from exc
            return _ProfileSession(self, text, version)
        try:
            if type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_KEYS:
                raise ValueError("snapshot fields differ")
            if snapshot["format"] != _SNAPSHOT_FORMAT or snapshot["version"] != _SNAPSHOT_VERSION:
                raise ValueError("snapshot format differs")
            recorded = snapshot["profile_version"]
            if type(recorded) is not str:
                raise ValueError("profile_version must be a string")
            text, version = self._store.read(recorded)
            if version != recorded or _sha256(text) != snapshot["text_sha256"]:
                raise ValueError("stored text differs from the recorded version")
        except (LookupError, OSError, TypeError, ValueError) as exc:
            raise _error("open_run", "profile.snapshot_unavailable",
                         "recorded profile version cannot be restored") from exc
        return _ProfileSession(self, text, version)


class _ProfileSession:
    """One run's pinned view of the profile and its commit authority."""

    def __init__(self, memory: ProfileMemory, text: str, version: str) -> None:
        self._memory = memory
        self._text = text
        self._version = version
        self._frames: dict[MemoryFrameRef, _ProfileFrame] = {}
        self._applied: dict[str, tuple[MemoryMutation, ...]] = {}
        self._lock = Lock()
        self._closed = False

    @property
    def snapshot(self) -> JsonObject:
        """Return the pinned version and text hash, never the text itself."""
        return {
            "format": _SNAPSHOT_FORMAT,
            "version": _SNAPSHOT_VERSION,
            "profile_version": self._version,
            "text_sha256": _sha256(self._text),
        }

    def __enter__(self) -> _ProfileSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Mark the session closed; the store holds no per-session resources."""
        self._closed = True

    def _ensure_open(self, stage: str) -> None:
        """Reject use after close."""
        if self._closed:
            raise _error(stage, "profile.closed", "profile session is closed")

    def open_frame(self, spec: MemoryFrameSpec) -> _ProfileFrame:
        """Open one frame; all frames of one policy must share one outermost frame.

        Args:
            spec (`MemoryFrameSpec`): Frame declared by the compiled layout.

        Returns:
            `_ProfileFrame`: The opened frame, or the same frame for a repeated spec.

        Raises:
            `MemoryOperationError`: ``profile.invalid_config`` for non-empty frame
                config, ``profile.multiple_roots`` for a second unrelated top frame.
        """
        with self._lock:
            self._ensure_open("open_frame")
            if type(spec) is not MemoryFrameSpec:
                raise _error("open_frame", "profile.invalid_config", "spec must be a MemoryFrameSpec")
            existing = self._frames.get(spec.ref)
            if existing is not None:
                if existing.spec != spec:
                    raise _error("open_frame", "profile.invalid_config",
                                 "frame was already opened with a different spec")
                return existing
            if spec.config:
                raise _error("open_frame", "profile.invalid_config",
                             "profile frames take no config")
            path = spec.ref.static_path
            nested = any(_is_strict_prefix(ref.static_path, path) for ref in self._frames)
            if self._frames and not nested:
                raise _error("open_frame", "profile.multiple_roots",
                             "every frame of one profile policy must sit under one outermost frame")
            frame = _ProfileFrame(self, spec)
            self._frames[spec.ref] = frame
            return frame

    def _is_outermost(self, ref: MemoryFrameRef) -> bool:
        """Return whether no other opened frame encloses ``ref``."""
        return not any(_is_strict_prefix(other.static_path, ref.static_path) for other in self._frames)


class _ProfileFrame:
    """One opened frame; only the outermost frame recalls and learns."""

    def __init__(self, session: _ProfileSession, spec: MemoryFrameSpec) -> None:
        self._session = session
        self.spec = spec

    def recall(self, context: RecallContext | None = None) -> str:
        """Return the pinned document for the outermost frame, else ``""``.

        Args:
            context (`RecallContext | None`, *optional*, defaults to `None`): The
                AGENT call being prepared. The document does not depend on it.

        Returns:
            `str`: The full pinned document, never truncated, or ``""``.
        """
        session = self._session
        session._ensure_open("recall")
        if context is not None and type(context) is not RecallContext:
            raise _error("recall", "profile.invalid_context", "context must be a RecallContext")
        return session._text if session._is_outermost(self.spec.ref) else ""
