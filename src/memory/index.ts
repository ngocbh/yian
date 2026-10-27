/**
 * Source-neutral identities and injected contracts for workflow memory, as Python's
 * `yian.memory`.
 *
 * A memory is one backend instance, identified by its memory key. Authors declare memories on
 * flows and AGENTs. Each declaration is a `MemoryDecl` stamped with the `DeclSite` (import scope,
 * lexical owner, AGENT node) where it was written. Compile resolves the declarations into a key
 * table of `MemorySpec` entries and one `MemoryStack` of keys, outermost first, for each node
 * placement. The run log records both, so replay and backward read keys instead of deriving them
 * again.
 *
 * `contracts.ts` owns that vocabulary: the memory JSON snapshot types, the per-declaration shape
 * check (`checkMemoryDeclaration`, with `MEMORY_IDENTIFIER_PATTERN` and
 * `BUILTIN_MEMORY_CONFIG_KEYS`), node scopes and their occurrence paths, the setup and operation
 * failures, recorded snapshots, the host memory objects (`Memory`, its run `MemorySession`, and
 * `isMemory`, which stands in for Python's `isinstance` check against the runtime-checkable
 * `Memory` protocol), and the learning results (`MemoryMutation`, `LearningFailure`,
 * `PolicyCommitOutcome`, `LearningResult`), what a learner sees of an AGENT invocation
 * (`MemorySection`, and the `VisibleInteraction` carriers `ModelTextInteraction`,
 * `ModelValueInteraction`, `ToolCallInteraction`, `ToolResultInteraction`), the learning inputs
 * (`MemoryEpisode` with its `MemoryEpisodeStatus`, and `LearnContext`), and the recall carriers
 * (`RecallContext`, `MemoryContent`, `MemoryContext`, `MemoryRecallError`).
 * TS yian has no session memory and no conversation contracts: chat history is a LOOP-carried
 * value. The declaration origin traversal and rewrite, the mutation batch check
 * `validateMutations` and the feedback snapshot `freezeFeedback` aren't re-exported (Python keeps
 * them out of `__all__`), nor are the `same*` value comparisons; their callers import
 * `contracts.ts`.
 *
 * `storage/sqlite.ts` and `storage/markdown.ts` are the storage adapters below the backends. The
 * facade re-exports none of these modules, as Python's `memory/__init__.py` doesn't import them.
 * `episodic.ts`, `reflective.ts` and `profile.ts` hold `EpisodicMemory`, `ReflectiveMemory` and
 * `ProfileMemory`.
 *
 * Knows about: `diagnostics` and `defaults`, through `contracts.ts`; `storage/` also knows
 * `platform` and `typesys`; the backends `episodic.ts`, `reflective.ts` and `profile.ts` also know
 * `typesys`; `reflective.ts` and `profile.ts` also know `llm` and `llm_clients`; `profile.ts` also
 * knows `reflective.ts`, `storage/markdown.ts` and `storage/sqlite.ts`.
 * Never imports (this facade and `contracts.ts`): `nodes`, `compile`, `compose`, `runtime`, `run`,
 * the CLI, `llm`, or persistence frameworks.
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
  isMemory,
  LearnContext,
  LearningFailure,
  LearningResult,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryContent,
  MemoryContext,
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
