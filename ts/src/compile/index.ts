/**
 * Compiled flow IR, validation, origin traversal/rebasing, growth, and stable identity, as
 * Python's `yian.compile`.
 *
 * This package turns pure node-owned growth seeds into deterministic `Flow` fragments through
 * `expand.buildGrow`. It also owns canonical manifests and fingerprints, identity-safe
 * opaque-origin traversal and source-neutral rebasing, static execution scopes independent of
 * memory declarations, static memory-layout derivation, and the complete static-path LLM
 * configuration map used for replay. Execution layouts derive root-relative template locations
 * and initial LOOP drivers, validating every target before stamping. Growth builders add
 * structured occurrence positions, child boundary associations, and source-neutral materialized
 * seed recipes. The shared model validates those recipes; expansion digests cover their topology
 * without adding derived state to authored fingerprints.
 *
 * `model.ts` owns the graph, its growth-metadata check and `copyGraph`, `origins.ts` the origin
 * traversal and rewrites, `execution_layout.ts` the static execution scopes, `validation.ts` the
 * cycle, path and type checks, and `manifest.ts` the manifests, fingerprints and expansion
 * digests. This index re-exports Python's `__all__` and nothing else; callers of any other name
 * import the module that defines it. Expansion, memory layout and the LLM configuration map are
 * not ported yet.
 *
 * Knows about: diagnostics, nodes, memory contracts and typesys.
 * Never imports: compose, runtime, the package-root run boundary, the CLI, or provider SDKs.
 */

export { applyExecutionLayout, resolveExecutionLayout } from "./execution_layout.ts";
export { CompiledFlow, Edge, END_ID, START_ID } from "./model.ts";
export { iterFlowOrigins, rebaseFlowOrigins, rewriteFlowOrigins } from "./origins.ts";
