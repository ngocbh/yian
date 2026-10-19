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
 * `ToolResultInteraction`), and the learning inputs (`MemoryEpisode` with its
 * `MemoryEpisodeStatus`, and `LearnContext`). The declaration origin traversal and rewrite, the
 * mutation batch check `validateMutations` and the feedback snapshot `freezeFeedback` aren't
 * re-exported (Python keeps them out of `__all__`), nor are the `same*` value comparisons; their
 * callers import `contracts.ts`. `storage/sqlite.ts` and `storage/markdown.ts` are ported: private
 * adapters the facade doesn't re-export, as Python's `memory/__init__.py` doesn't. The recall
 * context and the session conversation contracts, and the conversation, session, summary and
 * backend modules are not ported yet.
 *
 * Knows about: `diagnostics` and `defaults`, through `contracts.ts`; `storage/` also knows
 * `platform` and `typesys`.
 * Never imports: `nodes`, `compile`, `compose`, `runtime`, `run`, the CLI, `llm`, or persistence
 * frameworks.
 */

export type {
  JsonObject,
  JsonScalar,
  JsonValue,
  LearningOperation,
  LearningStatus,
  Memory,
  MemorySession,
  PolicyCommitStatus,
  VisibleInteraction,
} from "./contracts.ts";
export {
  BUILTIN_MEMORY_CONFIG_KEYS,
  checkMemoryDeclaration,
  DeclSite,
  EvaluationEvidence,
  isMemory,
  LearnContext,
  LearningFailure,
  LearningResult,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryDecl,
  MemoryEpisode,
  MemoryEpisodeStatus,
  MemoryMutation,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryScope,
  MemorySection,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
  ModelTextInteraction,
  ModelValueInteraction,
  PolicyCommitOutcome,
  ToolCallInteraction,
  ToolResultInteraction,
} from "./contracts.ts";
