"""Observable injected memory resources shared by runtime tests."""

from __future__ import annotations

import copy
from typing import Any

from agent_composer.memory import (
    LearnContext, MemoryFrameSpec, MemoryMutation, RecallContext,
)


class _Frame:
    """One fake frame backed by process-local test memory state."""

    def __init__(self, memory: "_Memory", spec: MemoryFrameSpec) -> None:
        self.memory = memory
        self.spec = spec

    def recall(self, context: RecallContext | None = None) -> str:
        self.memory.recall_contexts.append(context)
        return self.memory.recall_text

    def learn(self, context: LearnContext) -> list[MemoryMutation]:
        self.memory.learn_contexts.append(context)
        outcome = (
            self.memory.learn_outcomes.pop(0)
            if self.memory.learn_outcomes
            else []
        )
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(context)
        return outcome


class _Session:
    """One immutable-snapshot fake session."""

    def __init__(self, memory: "_Memory", snapshot: dict[str, Any]) -> None:
        self.memory = memory
        self._snapshot = copy.deepcopy(snapshot)
        self.frames: dict[object, _Frame] = {}
        self.closed = False

    @property
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._snapshot)

    def open_frame(self, spec: MemoryFrameSpec) -> _Frame:
        frame = _Frame(self.memory, spec)
        self.frames[spec.ref] = frame
        self.memory.opened_specs.append(spec)
        return frame

    def commit(
        self,
        commit_id: str,
        mutations: tuple[MemoryMutation, ...],
    ) -> None:
        self.memory.commit_calls.append((commit_id, mutations))
        outcome = (
            self.memory.commit_outcomes.pop(0)
            if self.memory.commit_outcomes
            else None
        )
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.memory.close_count += 1


class _Memory:
    """Observable structural Memory implementation for public-boundary tests."""

    def __init__(
        self,
        *,
        recall_text: str = "",
        learn_outcomes: list[object] | None = None,
        commit_outcomes: list[object] | None = None,
    ) -> None:
        self.recall_text = recall_text
        self.learn_outcomes = list(learn_outcomes or [])
        self.commit_outcomes = list(commit_outcomes or [])
        self.open_snapshots: list[dict[str, Any] | None] = []
        self.opened_specs: list[MemoryFrameSpec] = []
        self.recall_contexts: list[RecallContext | None] = []
        self.learn_contexts: list[LearnContext] = []
        self.commit_calls: list[tuple[str, tuple[MemoryMutation, ...]]] = []
        self.close_count = 0
        self.reset_count = 0

    def open_run(self, *, snapshot=None) -> _Session:
        restored = None if snapshot is None else copy.deepcopy(snapshot)
        self.open_snapshots.append(restored)
        effective = {"namespace": "test"} if restored is None else restored
        return _Session(self, effective)

    def reset(self) -> None:
        self.reset_count += 1
