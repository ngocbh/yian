/**
 * CODE-node functions harvested cases reference: a TypeScript port of yian's
 * `tests/engine/_compose_codefns.py`. Functions are ported as harvested cases reference them.
 */

/** Return a multi-field object matching a declared `{rating, score}` record. */
export function make_plan(inputs: Record<string, any>): Record<string, unknown> {
  return { rating: `plan for ${inputs.topic}`, score: 0.9 };
}

/** Return a single numeric value (the case `on:`/`when:` producer in the coalesce flow). */
export function score(inputs: Record<string, any>): unknown {
  return inputs.seed;
}

/** Return the pro case for the bound `topic`. */
export function positive(inputs: Record<string, any>): string {
  return `pro case for ${inputs.topic}`;
}

/** Return the cautious note for the bound `topic`. */
export function cautious(inputs: Record<string, any>): string {
  return `cautious note for ${inputs.topic}`;
}

// --- CALL/MAP child-flow CODE fns (Ollama-free subflows) ---
/** A `{report, n}` child codomain (CALL re-exports it; MAP collects a list of it). */
export function make_report(inputs: Record<string, any>): Record<string, unknown> {
  const t = inputs.topic;
  // Python's len() counts code points, not UTF-16 units.
  return { report: `report for ${t}`, n: BigInt([...t].length) };
}

/** Return the bound `topic` unchanged (MAP order/identity checks). */
export function echo(inputs: Record<string, any>): unknown {
  return inputs.topic;
}

/** Return the bound `x` unchanged (the boundary-node _FLOW fixture). */
export function echo_x(inputs: Record<string, any>): unknown {
  return inputs.x;
}

// --- co-skip fns (Ollama-free branch+join) ---
/** Return the pro branch's claim for the bound `topic`. */
export function pick_pro(inputs: Record<string, any>): string {
  return `pro: ${inputs.topic}`;
}

/** Return the con branch's claim for the bound `topic`. */
export function pick_con(inputs: Record<string, any>): string {
  return `con: ${inputs.topic}`;
}

/** The `take(stance=...)` def body for then:/else: ${call} tests. */
export function took(inputs: Record<string, any>): string {
  return `took:${inputs.stance}`;
}

/** The `checked` def body for child-boundary-assert tests. */
export function double(inputs: Record<string, any>): bigint {
  return inputs.n * 2n;
}

// --- LOOP body CODE fns (Ollama-free while-loop e2e) ---
/**
 * A loop body 'a -> 'a over `{n, exited}`: increment n, set exited once n reaches 3.
 * Drives the while-loop e2e (3 iterations from a `{n:0, exited:false}` seed).
 */
export function loop_bump(inputs: Record<string, any>): Record<string, unknown> {
  const n = inputs.n + 1n;
  return { n, exited: n >= 3n };
}
