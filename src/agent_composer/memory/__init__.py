"""Source-neutral identities and injected contracts for workflow memory.

This package owns stable memory-frame identity, occurrence identity, learning
payloads, and the narrow protocols implemented by host-provided memory objects.
Concrete storage and reflection backends live in separate modules so importing
this facade remains dependency-light.

Knows about:   the standard library and agent_composer.diagnostics.
Never imports: nodes, compile, compose, runtime, run, CLI, LangChain, Pydantic,
               llm_clients, or concrete persistence backends.
"""

from agent_composer.memory.contracts import (
    JsonObject,
    JsonScalar,
    JsonValue,
    EvaluationEvidence,
    LearnContext,
    Memory,
    MemoryBoundary,
    MemoryBoundaryKind,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryFrame,
    MemoryFrameRef,
    MemoryFrameSpec,
    MemoryMutation,
    MemoryOperationFailure,
    MemoryOccurrenceKind,
    MemoryOccurrenceRef,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemoryOwner,
    MemoryPolicySpec,
    MemoryRecallError,
    MemorySection,
    MemorySnapshotEnvelope,
    MemoryScope,
    MemorySession,
    MemorySetupError,
    RecallContext,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
    VisibleInteraction,
)

__all__ = [
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "EvaluationEvidence",
    "LearnContext",
    "Memory",
    "MemoryBoundary",
    "MemoryBoundaryKind",
    "MemoryEpisode",
    "MemoryEpisodeStatus",
    "MemoryFrame",
    "MemoryFrameRef",
    "MemoryFrameSpec",
    "MemoryMutation",
    "MemoryOperationFailure",
    "MemoryOccurrenceKind",
    "MemoryOccurrenceRef",
    "MemoryOccurrenceSegment",
    "MemoryOperationError",
    "MemoryOwner",
    "MemoryPolicySpec",
    "MemoryRecallError",
    "MemorySection",
    "MemorySnapshotEnvelope",
    "MemoryScope",
    "MemorySession",
    "MemorySetupError",
    "RecallContext",
    "ModelTextInteraction",
    "ModelValueInteraction",
    "ToolCallInteraction",
    "ToolResultInteraction",
    "VisibleInteraction",
]
