"""Immutable backward outcomes shared by runtime records and their consumers.

These values depend only on lower node, memory, and diagnostic contracts. They
never import the runtime log, state, trace, driver, or public run boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from agent_composer.diagnostics import Failure, OriginRef
from agent_composer.memory.contracts import (
    LearningResult,
    MemoryFrameRef,
    MemoryOperationFailure,
)
from agent_composer.nodes.base import BackwardResult
from agent_composer.nodes.contracts import ExecutionScope


@dataclass(frozen=True, slots=True)
class BackwardFailure:
    """An attributed failure from one backward or memory operation.

    Attributes:
        stage (`str`): Non-empty operation stage, such as backward, learn, or commit.
        code (`str`): Stable, non-empty semantic failure code.
        message (`str`): Human-readable detail; it may be empty.
        scope (`ExecutionScope | None`): Responsible authored owner and child position.
        run_id (`str | None`): Recorded run identity, when available.
        node_id (`str | None`): Executing node's display ID, when applicable.
        node_started_seq (`int | None`): Positive accepted start sequence identifying
            a node occurrence together with its run ID.
        policy_name (`str | None`): Responsible memory policy, when applicable.
        frame_ref (`MemoryFrameRef | None`): Responsible owned memory frame, if any.
        cause (`Failure | MemoryOperationFailure | None`): Underlying source-neutral
            diagnostic, including a backend retryability hint when one exists.
            Failure origins retain only their canonical keys.
        cause_type (`str | None`): Python exception class name retained for debugging.
        traceback (`str | None`): Opaque debugging traceback text.
    """

    stage: str
    code: str
    message: str
    scope: ExecutionScope | None = None
    run_id: str | None = None
    node_id: str | None = None
    node_started_seq: int | None = None
    policy_name: str | None = None
    frame_ref: MemoryFrameRef | None = None
    cause: Failure | MemoryOperationFailure | None = None
    cause_type: str | None = None
    traceback: str | None = None

    def __post_init__(self) -> None:
        """Require stable attribution and data-only diagnostic causes."""
        for name, value in (("stage", self.stage), ("code", self.code)):
            if type(value) is not str:
                raise TypeError(f"backward failure {name} must be a string")
            if not value:
                raise ValueError(f"backward failure {name} must be non-empty")
        if type(self.message) is not str:
            raise TypeError("backward failure message must be a string")
        for name, value in (
            ("run_id", self.run_id),
            ("node_id", self.node_id),
            ("policy_name", self.policy_name),
            ("cause_type", self.cause_type),
            ("traceback", self.traceback),
        ):
            if value is not None and type(value) is not str:
                raise TypeError(f"backward failure {name} must be a string or None")
        if self.node_started_seq is not None:
            if type(self.node_started_seq) is not int:
                raise TypeError("node_started_seq must be an integer or None")
            if self.node_started_seq <= 0:
                raise ValueError("node_started_seq must be positive")
        if self.scope is not None:
            if type(self.scope) is not ExecutionScope:
                raise TypeError("scope must be an ExecutionScope or None")
            ExecutionScope.__post_init__(self.scope)
        if self.frame_ref is not None:
            if type(self.frame_ref) is not MemoryFrameRef:
                raise TypeError("frame_ref must be a MemoryFrameRef or None")
            MemoryFrameRef.__post_init__(self.frame_ref)
        if self.cause is not None:
            if type(self.cause) is Failure:
                Failure.__post_init__(self.cause)
                object.__setattr__(
                    self,
                    "cause",
                    replace(
                        self.cause,
                        trace=tuple(OriginRef(origin.key) for origin in self.cause.trace),
                    ),
                )
            elif type(self.cause) is MemoryOperationFailure:
                MemoryOperationFailure.__post_init__(self.cause)
            else:
                raise TypeError("cause must be a Failure, MemoryOperationFailure, or None")


@dataclass(frozen=True, slots=True)
class BackwardOutcome:
    """A backward attempt's disposition, separate from its forward run result.

    Attributes:
        status (`Literal["succeeded", "failed"]`): Status of this backward attempt.
        feedback (`BackwardResult | None`): Completed flow-level feedback, when
            available, including when a later learning operation fails.
        failure (`BackwardFailure | None`): The failed attempt's diagnostic;
            absent on success.
        learning (`LearningResult | None`): Memory learning disposition, or
            `None` when no memory policies participate.
    """

    status: Literal["succeeded", "failed"]
    feedback: BackwardResult | None = None
    failure: BackwardFailure | None = None
    learning: LearningResult | None = None

    def __post_init__(self) -> None:
        """Validate the disposition and detach the completed feedback result."""
        if type(self.status) is not str or self.status not in ("succeeded", "failed"):
            raise ValueError("backward status must be 'succeeded' or 'failed'")
        if self.feedback is not None:
            if type(self.feedback) is not BackwardResult:
                raise TypeError("feedback must be a BackwardResult or None")
            object.__setattr__(
                self,
                "feedback",
                BackwardResult(
                    self.feedback.input_feedback,
                    self.feedback.self_feedback,
                    self.feedback.variable_feedback,
                ),
            )
        if self.failure is not None:
            if type(self.failure) is not BackwardFailure:
                raise TypeError("failure must be a BackwardFailure or None")
            object.__setattr__(self, "failure", replace(self.failure))
        if self.status == "succeeded" and self.failure is not None:
            raise ValueError("succeeded backward cannot carry a failure")
        if self.status == "failed" and self.failure is None:
            raise ValueError("failed backward requires a failure")
        if self.learning is not None:
            if type(self.learning) is not LearningResult:
                raise TypeError("learning must be a LearningResult or None")
            LearningResult.__post_init__(self.learning)
            if self.status == "succeeded" and self.learning.status != "succeeded":
                raise ValueError("succeeded backward requires succeeded learning")


__all__ = ["BackwardFailure", "BackwardOutcome"]
