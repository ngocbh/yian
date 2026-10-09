"""Observable injected memory sessions shared by runtime tests."""

from __future__ import annotations

import copy
from typing import Any

from yian.memory import LearnContext, MemoryMutation, RecallContext


class _Session:
    """One immutable-snapshot fake session."""

    def __init__(self, memory: "_Memory", snapshot: dict[str, Any]) -> None:
        self.memory = memory
        self._snapshot = copy.deepcopy(snapshot)
        self.closed = False

    @property
    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self._snapshot)

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

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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
        self.sessions: list[_Session] = []
        self.recall_contexts: list[RecallContext | None] = []
        self.learn_contexts: list[LearnContext] = []
        self.commit_calls: list[tuple[str, tuple[MemoryMutation, ...]]] = []
        self.close_count = 0
        self.reset_count = 0

    def open_run(self, *, snapshot=None) -> _Session:
        restored = None if snapshot is None else copy.deepcopy(snapshot)
        self.open_snapshots.append(restored)
        effective = {"namespace": "test"} if restored is None else restored
        session = _Session(self, effective)
        self.sessions.append(session)
        return session

    def reset(self) -> None:
        self.reset_count += 1
