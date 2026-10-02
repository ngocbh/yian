"""Markdown profile memory: one versioned document per memory instance.

``ProfileMemory`` recalls one markdown document into every AGENT whose stack
holds the instance and, on backward, asks one model call to rewrite the whole
document from that run's feedback. Commits are compare-and-set writes to a
``MarkdownStore`` and never call a model.
"""

from __future__ import annotations

import hashlib
import json
from threading import Lock
from types import TracebackType

from langchain_core.messages import HumanMessage, SystemMessage

from agent_composer.llm_clients import model_from_config
from agent_composer.llm_clients.config import pop_api_key
from agent_composer.llm_clients.connection import resolve_model_config
from agent_composer.llm_clients.redact import scrub_key
from agent_composer.memory.contracts import (
    JsonObject,
    JsonValue,
    LearnContext,
    MemoryMutation,
    MemoryOperationError,
    RecallContext,
)
from agent_composer.memory.storage import MarkdownStore, MarkdownVersionMismatch
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
    """Memory backend that keeps one markdown profile in a versioned store.

    Inject one instance under the memory key the flow declares, for example
    ``run_flow(flow, inputs, memory={"profile": ProfileMemory(store)})``. Every
    AGENT whose stack holds this key sees the whole document once.
    """

    def __init__(
        self,
        store: MarkdownStore,
        *,
        llm_config: JsonObject | None = None,
        max_chars: int = 32_000,
        input_budget_chars: int = 24_000,
    ) -> None:
        """Bind the backend to one store and learner model.

        Args:
            store (`MarkdownStore`): Versioned document store, such as
                `FileMarkdownStore`.
            llm_config (`JsonObject | None`, *optional*, defaults to `None`):
                Learner model config. `None` uses the ambient defaults when a
                backward attempt learns. An ``api_key`` entry needs ``provider``;
                it authenticates the learner, is kept as a `SecretStr`, and is
                never recorded.
            max_chars (`int`, *optional*, defaults to `32_000`): Largest document
                the learner may write (about 8k tokens). Host writes are not bounded.
            input_budget_chars (`int`, *optional*, defaults to `24_000`): Character
                budget for episode JSON sent to the learner. Feedback is always sent.

        Raises:
            `TypeError`: ``store`` lacks ``read``/``write`` or ``llm_config`` is not
                a dict.
            `ValueError`: A budget is not a positive integer, or ``api_key`` is
                empty or lacks ``provider``.
        """
        if not isinstance(store, MarkdownStore):
            raise TypeError("store must provide read() and write()")
        if llm_config is not None and type(llm_config) is not dict:
            raise TypeError("llm_config must be a dict or None")
        for name, value in (("max_chars", max_chars), ("input_budget_chars", input_budget_chars)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._store = store
        self._llm_config = dict(llm_config or {})
        self._api_key = pop_api_key(self._llm_config)
        self.max_chars = max_chars
        self.input_budget_chars = input_budget_chars
        self._closed = False

    def close(self) -> None:
        """Reject later use of this backend and its sessions; repeated calls do nothing.

        The store holds no handle, so closing does no I/O.
        """
        self._closed = True

    def _ensure_open(self, stage: str) -> None:
        """Reject use after close."""
        if self._closed:
            raise _error(stage, "profile.closed", "memory backend is closed")

    def read(self, version: str | None = None) -> tuple[str, str]:
        """Return ``(text, version)`` from the store for host display or editing.

        Args:
            version (`str | None`, *optional*, defaults to `None`): Earlier version,
                or `None` for the current document.

        Returns:
            `tuple[str, str]`: Document text and version.

        Raises:
            `LookupError`: The named version is unknown.
            `MemoryOperationError`: ``profile.closed`` after close.
        """
        self._ensure_open("read")
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
            `MarkdownVersionMismatch`: ``base_version`` is not current.
            `MemoryOperationError`: ``profile.closed`` after close.
        """
        self._ensure_open("write")
        return self._store.write(text, base_version=base_version, provenance=provenance)

    def reset(self) -> None:
        """Refuse engine resets; delete or rewrite the profile through the store.

        Raises:
            `MemoryOperationError`: Always: ``profile.closed`` after close, else
                ``profile.reset_unsupported``.
        """
        self._ensure_open("reset")
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
            `MemoryOperationError`: ``profile.closed`` after close,
                ``profile.store_unavailable`` (retryable) when the store cannot
                be reached, ``profile.store_invalid`` when its document cannot be
                decoded, or ``profile.snapshot_unavailable`` when the recorded
                version is malformed, unknown, or changed.
        """
        self._ensure_open("open_run")
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
        """Reject use after this session or its backend closes."""
        self._memory._ensure_open(stage)
        if self._closed:
            raise _error(stage, "profile.closed", "profile session is closed")

    def recall(self, context: RecallContext | None = None) -> str:
        """Return the full pinned document.

        Args:
            context (`RecallContext | None`, *optional*, defaults to `None`): The
                AGENT call being prepared. The document does not depend on it.

        Returns:
            `str`: The full pinned document, never truncated.
        """
        self._ensure_open("recall")
        if context is not None and type(context) is not RecallContext:
            raise _error("recall", "profile.invalid_context", "context must be a RecallContext")
        return self._text

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        """Propose one full-document rewrite from the run's feedback.

        Args:
            context (`LearnContext`): Episodes and feedback for this instance.

        Returns:
            `list[MemoryMutation]`: One document mutation, or an empty list for
                runs without feedback and unchanged documents.

        Raises:
            `MemoryOperationError`: ``profile.invalid_context``,
                ``profile.store_unavailable`` (retryable), ``profile.store_invalid``,
                ``profile.model_construction``, ``profile.provider`` (retryable),
                ``profile.contract``, or ``profile.over_budget``.
        """
        self._ensure_open("learn")
        if type(context) is not LearnContext:
            raise _error("learn", "profile.invalid_context", "context must be a LearnContext")
        if not context.feedback:
            return []
        memory = self._memory
        try:
            current_text, current_version = memory._store.read(None)
        except (LookupError, OSError, ValueError) as exc:
            raise _read_error("learn", exc) from exc
        messages = [SystemMessage(content=_SYSTEM),
                    HumanMessage(content=self._material(context, current_text, current_version))]
        try:
            config = dict(memory._llm_config)
            if memory._api_key is not None:
                config["api_key"] = memory._api_key
            model = model_from_config(resolve_model_config(config))
        except Exception as exc:
            scrub_key(exc, memory._api_key)
            raise _error("learn", "profile.model_construction",
                         "profile learner model construction failed") from exc
        try:
            reply = model.invoke(messages)
        except Exception as exc:
            # Providers may echo a rejected key; the chained cause must not carry it.
            scrub_key(exc, memory._api_key)
            raise _error("learn", "profile.provider", "profile learner invocation failed",
                         retryable=True) from exc
        text = _parse_reply(reply, current_text, memory.max_chars)
        if text.strip() == current_text.strip():
            return []
        feedback: list[JsonValue] = []
        for entry in context.feedback:
            value = entry.get("value") if isinstance(entry, dict) else entry
            if value not in feedback:
                feedback.append(value)
        payload: JsonObject = {
            "base_version": current_version,
            "text": text,
            "seen_version": self._version,
            "run_id": context.run_id,
            "feedback": feedback,
        }
        return [MemoryMutation("profile:" + _sha256(text), payload)]

    def _material(self, context: LearnContext, current_text: str, current_version: str) -> str:
        """Return the learner's JSON input with the newest episodes that fit the budget.

        The newest episode is always sent; if it alone is over the budget, it is
        cut to a prefix and a suffix the way ReflectiveMemory trims its material.
        Older episodes are added whole, newest first, until one does not fit.
        """
        feedback: list[JsonValue] = []
        for entry in context.feedback:
            if isinstance(entry, dict):
                feedback.append({key: entry.get(key) for key in
                                 ("channel", "target_name", "value", "owner_static_path")})
            else:
                feedback.append({"value": entry})
        budget = self._memory.input_budget_chars
        agent_episodes = [episode for episode in context.episodes if episode.base_system_prompt is not None]
        kept: list[JsonValue] = []
        used = 0
        for episode in reversed(agent_episodes):
            item: JsonObject = {
                "owner_static_path": list(episode.owner_static_path),
                "status": episode.status.value,
                "prompt": episode.prompt,
                "inputs": episode.inputs,
                "interactions": [_visible_interaction_json(interaction) for interaction in episode.interactions],
                "output": episode.output,
                "failure": episode.failure,
            }
            encoded = _canonical_json(item)
            if not kept:
                encoded = _bounded_json_material(item, budget)
            elif used + len(encoded) > budget:
                break
            used += len(encoded)
            kept.append(json.loads(encoded))
        kept.reverse()
        material: JsonObject = {
            "max_chars": self._memory.max_chars,
            "current_profile": current_text,
            "feedback": feedback,
            "episodes": kept,
            "omitted_episodes": len(agent_episodes) - len(kept),
        }
        if current_version != self._version:
            material["seen_profile"] = self._text
        return json.dumps(material, ensure_ascii=False, indent=2, sort_keys=True)

    def commit(self, commit_id: str, mutations: tuple[MemoryMutation, ...]) -> None:
        """Write the proposed document with compare-and-set; never call a model.

        Args:
            commit_id (`str`): Runtime commit id, unique per backward attempt and
                memory. A repeat within this session is a no-op.
            mutations (`tuple[MemoryMutation, ...]`): Empty, or the one mutation
                this session proposed.

        Raises:
            `MemoryOperationError`: ``profile.version_mismatch`` (retryable; nothing
                was written) when the document changed after the learner read it,
                ``profile.store_unavailable`` when the store failed during the
                write (it may have landed), ``profile.store_invalid`` when the
                store rejected it, or ``profile.invalid_batch`` for a malformed or
                reused batch.
        """
        with self._lock:
            self._ensure_open("commit")
            if type(commit_id) is not str or not commit_id or type(mutations) is not tuple:
                raise _error("commit", "profile.invalid_batch", "commit needs an id and a mutation tuple")
            if commit_id in self._applied:
                if self._applied[commit_id] != mutations:
                    raise _error("commit", "profile.invalid_batch", "commit id was reused for another batch")
                return
            if not mutations:
                self._applied[commit_id] = mutations
                return
            mutation = mutations[0] if len(mutations) == 1 else None
            payload = None if mutation is None else mutation.payload
            if (type(payload) is not dict or type(payload.get("text")) is not str
                    or type(payload.get("base_version")) is not str):
                raise _error("commit", "profile.invalid_batch",
                             "profile commits take at most one document mutation")
            provenance: JsonObject = {
                "commit_id": commit_id,
                "run_id": payload.get("run_id"),
                "seen_version": payload.get("seen_version"),
                "feedback": payload.get("feedback"),
            }
            try:
                self._memory._store.write(payload["text"], base_version=payload["base_version"],
                                          provenance=provenance)
            except MarkdownVersionMismatch as exc:
                raise _error("commit", "profile.version_mismatch",
                             "profile changed after the learner read it; nothing was written",
                             retryable=True) from exc
            except OSError as exc:
                raise _error("commit", "profile.store_unavailable",
                             "profile store write failed and may have landed; check the store "
                             "before retrying") from exc
            except ValueError as exc:
                raise _error("commit", "profile.store_invalid",
                             "profile store rejected the document") from exc
            self._applied[commit_id] = mutations
