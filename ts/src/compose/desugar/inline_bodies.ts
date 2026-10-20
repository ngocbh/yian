/**
 * Occurrence-local extraction for inline composite child-flow bodies.
 *
 * Holds `INLINE_FLOW_PREFIX`, the callable-name prefix the extraction reserves; the extraction
 * itself (`InlineBody`, `extractInlineBodies`, `registerInlineBodies`) is not ported yet (T3-17).
 *
 * Knows about: nothing yet.
 * Never imports: compose/parse, edges, validate, build, the loader, or the runtime.
 */

/** Callable-name prefix reserved for compiler-generated inline child flows. */
export const INLINE_FLOW_PREFIX = "__inline_flow_";
