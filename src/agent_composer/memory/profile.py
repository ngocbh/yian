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
