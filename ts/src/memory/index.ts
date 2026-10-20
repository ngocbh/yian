/**
 * Source-neutral identities and injected contracts for workflow memory, as Python's
 * `yian.memory`.
 *
 * A memory is one backend instance, identified by its memory key. Authors declare memories on
 * flows and AGENTs. Each declaration is a `MemoryDecl` stamped with the `DeclSite` (import scope,
 * lexical owner, AGENT node) where it was written. Compile resolves the declarations into a key
 * table of `MemorySpec` entries and one `MemoryStack` of keys, outermost first, for each node
 * placement. The run log records both, so replay and backward read keys instead of deriving them
 * again. A session memory has one owning AGENT and holds one conversation.
 *
 * `contracts.ts` owns that vocabulary: the memory JSON snapshot types, the per-declaration shape
 * check (`checkMemoryDeclaration`, with `MEMORY_IDENTIFIER_PATTERN` and
 * `BUILTIN_MEMORY_CONFIG_KEYS`), node scopes and their occurrence paths, the setup and operation
 * failures, recorded snapshots, the host memory objects (`Memory`, its run `MemorySession`,
 * and `isMemory`, which stands in for Python's `isinstance` check against the runtime-checkable
 * `Memory` protocol), and the learning results (`MemoryMutation`, `LearningFailure`,
 * `PolicyCommitOutcome`, `LearningResult`), the visible evidence of one node outcome
 * (`EvaluationEvidence`, `MemorySection`, and the `VisibleInteraction` carriers
 * `ModelTextInteraction`, `ModelValueInteraction`, `ToolCallInteraction`,
 * `ToolResultInteraction`), the learning inputs (`MemoryEpisode` with its
 * `MemoryEpisodeStatus`, and `LearnContext`), the recall carriers (`RecallContext`,
 * `MemoryContextRequest`, `MemoryContent`, `MemoryContext`, `MemoryRecallError`, and the engine's
 * `EvaluationTrace`), and the session conversation contracts: the frozen checkpoints and views
 * (`ConversationInvocation` through `ConversationClosed`, with `ConversationCheckpoint`) and the
 * host interfaces `TranscriptSource` and `ConversationTurn`. Python's `CheckStopped` has no
 * counterpart: `openTurn` takes an `AbortSignal`. The declaration origin traversal and rewrite,
 * the mutation batch check `validateMutations` and the feedback snapshot `freezeFeedback` aren't
 * re-exported (Python keeps them out of `__all__`), nor are the `same*` value comparisons; their
 * callers import `contracts.ts`.
 *
 * `conversation.ts` owns the pure conversation fold, strict codecs and digests; `summary.ts` owns
 * whole-turn selection and summary prefix proofs. `storage/sqlite.ts` and `storage/markdown.ts`
 * are the storage adapters below the backends. The facade re-exports none of these modules, as
 * Python's `memory/__init__.py` doesn't import them. `episodic.ts`, `reflective.ts` and `session.ts`
 * hold `EpisodicMemory`, `ReflectiveMemory` and `SessionMemory`; the profile backend is not ported yet.
 *
 * Knows about: `diagnostics` and `defaults`, through `contracts.ts`; `conversation.ts` and
 * `summary.ts` also know `typesys`; `storage/` also knows `platform` and `typesys`.
 * Never imports: `nodes`, `compile`, `compose`, `runtime`, `run`, the CLI, `llm`, or persistence
 * frameworks.
 */

export type {
  ConversationCheckpoint,
  ConversationEffectKind,
  ConversationEffectStatus,
  ConversationTurn,
  ConversationTurnStatus,
  EvaluationTrace,
  JsonObject,
  JsonScalar,
  JsonValue,
  LearningOperation,
  LearningStatus,
  Memory,
  MemorySession,
  PolicyCommitStatus,
  TranscriptSource,
  VisibleInteraction,
} from "./contracts.ts";
export {
  BUILTIN_MEMORY_CONFIG_KEYS,
  checkMemoryDeclaration,
  ConversationAdmitted,
  ConversationApplied,
  ConversationClosed,
  ConversationCompletion,
  ConversationEffectObserved,
  ConversationEffectRequested,
  ConversationEffectState,
  ConversationInvocation,
  ConversationItem,
  ConversationRequest,
  ConversationStamp,
  ConversationView,
  DeclSite,
  EvaluationEvidence,
  isMemory,
  LearnContext,
  LearningFailure,
  LearningResult,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryContent,
  MemoryContext,
  MemoryContextRequest,
  MemoryDecl,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryRecallError,
  MemoryScope,
  MemorySection,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
  ModelValueInteraction,
  PolicyCommitOutcome,
  RecallContext,
  ToolCallInteraction,
  ToolResultInteraction,
} from "./contracts.ts";
