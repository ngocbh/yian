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
 * failures, recorded snapshots, and the host memory objects (`Memory`, its run `MemorySession`,
 * and `isMemory`, Python's runtime-checkable `isinstance`). The declaration origin traversal and
 * rewrite aren't re-exported (Python keeps them out of `__all__`); their callers import
 * `contracts.ts`. The
 * recall, learning evidence, learning and session conversation contracts, and the conversation,
 * session, summary, storage and backend modules, are not ported yet.
 *
 * Knows about: `diagnostics` and `defaults`, through `contracts.ts`.
 * Never imports: `nodes`, `compile`, `compose`, `runtime`, `run`, the CLI, `llm`, or persistence
 * frameworks.
 */

export type { JsonObject, JsonScalar, JsonValue, Memory, MemorySession } from "./contracts.ts";
export {
  BUILTIN_MEMORY_CONFIG_KEYS,
  checkMemoryDeclaration,
  DeclSite,
  isMemory,
  MEMORY_IDENTIFIER_PATTERN,
  MemoryDecl,
  MemoryOccurrenceKind,
  MemoryOccurrenceSegment,
  MemoryOperationError,
  MemoryOperationFailure,
  MemoryScope,
  MemorySetupError,
  MemorySnapshotEnvelope,
  MemorySpec,
  MemoryStack,
} from "./contracts.ts";
