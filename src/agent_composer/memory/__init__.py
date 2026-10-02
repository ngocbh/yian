"""Source-neutral identities and injected contracts for workflow memory.

A memory is one backend instance, identified by its memory key. Authors
declare memories on flows and AGENTs. Each declaration is a ``MemoryDecl``
stamped with the ``DeclSite`` (import scope, lexical owner, AGENT node) where
it was written. Compile resolves the declarations into a key table of
``MemorySpec`` entries and one ``MemoryStack`` of keys, outermost first, for
each node placement. The run log records both, so replay and backward read
keys instead of deriving them again. A session memory has one owning AGENT
and holds one conversation.

This package also owns occurrence paths, learning payloads, detached
conversation checkpoints, and the narrow protocols implemented by
host-provided memory objects. Structured transcript contributions extend
recall while retaining existing text-only memory implementations.
``contracts.py`` owns that vocabulary; ``conversation.py`` owns its pure fold,
strict codecs, and digests; ``session.py`` owns conversation views,
FIFO leases, atomic turn updates, and learned self-feedback notes.
``summary.py`` owns pure whole-turn selection and summary-cache proofs; the
backend invokes injected projection/model callbacks without importing providers.
The private ``storage.py`` adapter owns
single-host SQLite file locking and atomic finite-JSON state transactions below
the concrete backends; it owns no run sessions or provider behavior.
Concrete session, episodic, reflection, and profile
backends are imported from their own modules, never eagerly by this contracts-only
facade.

Knows about:   the standard library and agent_composer.diagnostics.
Never imports: nodes, compile, compose, runtime, run, CLI, LangChain, Pydantic,
               llm_clients, or third-party persistence frameworks.
"""

from agent_composer.memory.contracts import (
    BUILTIN_MEMORY_CONFIG_KEYS,
    CheckStopped,
    ConversationAdmitted,
    ConversationApplied,
    ConversationCheckpoint,
    ConversationClosed,
    ConversationCompletion,
    ConversationEffectKind,
    ConversationEffectObserved,
    ConversationEffectRequested,
    ConversationEffectState,
    ConversationEffectStatus,
    ConversationInvocation,
    ConversationItem,
    ConversationRequest,
    ConversationStamp,
    ConversationTurn,
    ConversationTurnStatus,
    ConversationView,
    DeclSite,
    JsonObject,
    JsonScalar,
    JsonValue,
    EvaluationEvidence,
    EvaluationTrace,
    LearnContext,
    LearningFailure,
    LearningResult,
    PolicyCommitOutcome,
    MEMORY_IDENTIFIER_PATTERN,
    Memory,
    MemoryContent,
    MemoryContext,
    MemoryContextRequest,
    MemoryDecl,
    MemoryEpisode,
    MemoryEpisodeStatus,
    MemoryMutation,
    MemoryOperationFailure,
    MemoryOccurrenceKind,
    MemoryOccurrenceSegment,
    MemoryOperationError,
    MemoryRecallError,
    MemorySection,
    MemorySnapshotEnvelope,
    MemorySpec,
    MemoryStack,
    MemoryScope,
    MemorySession,
    MemorySetupError,
    RecallContext,
    ModelTextInteraction,
    ModelValueInteraction,
    ToolCallInteraction,
    ToolResultInteraction,
    TranscriptSource,
    VisibleInteraction,
    check_memory_declaration,
)

__all__ = [
    "BUILTIN_MEMORY_CONFIG_KEYS",
    "CheckStopped",
    "ConversationAdmitted",
    "ConversationApplied",
    "ConversationCheckpoint",
    "ConversationClosed",
    "ConversationCompletion",
    "ConversationEffectKind",
    "ConversationEffectObserved",
    "ConversationEffectRequested",
    "ConversationEffectState",
    "ConversationEffectStatus",
    "ConversationInvocation",
    "ConversationItem",
    "ConversationRequest",
    "ConversationStamp",
    "ConversationTurn",
    "ConversationTurnStatus",
    "ConversationView",
    "DeclSite",
    "JsonObject",
    "JsonScalar",
    "JsonValue",
    "EvaluationEvidence",
    "EvaluationTrace",
    "LearnContext",
    "LearningFailure",
    "LearningResult",
    "PolicyCommitOutcome",
    "MEMORY_IDENTIFIER_PATTERN",
    "Memory",
    "MemoryContent",
    "MemoryContext",
    "MemoryContextRequest",
    "MemoryDecl",
    "MemoryEpisode",
    "MemoryEpisodeStatus",
    "MemoryMutation",
    "MemoryOperationFailure",
    "MemoryOccurrenceKind",
    "MemoryOccurrenceSegment",
    "MemoryOperationError",
    "MemoryRecallError",
    "MemorySection",
    "MemorySnapshotEnvelope",
    "MemorySpec",
    "MemoryStack",
    "MemoryScope",
    "MemorySession",
    "MemorySetupError",
    "RecallContext",
    "ModelTextInteraction",
    "ModelValueInteraction",
    "ToolCallInteraction",
    "ToolResultInteraction",
    "TranscriptSource",
    "VisibleInteraction",
    "check_memory_declaration",
]
