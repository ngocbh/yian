/** CODE-node functions the conformance cases reference by `./fixtures/compose_codefns.ts:<name>`. */

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
  // `n` counts code points, not UTF-16 units.
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

/** A child CODE node that fails makes the CALL/MAP run fail. */
export function boom(inputs: Record<string, any>): unknown {
  throw new Error("boom");
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

/** Return the detail for the bound `base`. */
export function detail_of(inputs: Record<string, any>): string {
  return `detail: ${inputs.base}`;
}

/** claim = the taken branch (ref-coalesce join); detail = pro_detail or null (`:-null`). */
export function assemble_join(inputs: Record<string, any>): string {
  return `${inputs.claim}|detail=${inputs.detail}`;
}

/** The `take(stance=...)` def body for then:/else: ${call} tests. */
export function took(inputs: Record<string, any>): string {
  return `took:${inputs.stance}`;
}

/** The `checked` def body for child-boundary-assert tests. */
export function double(inputs: Record<string, any>): number {
  return inputs.n * 2;
}

/** Surface the bound `${system.run_id}` so the run test can assert on it. */
export function echo_rid(inputs: Record<string, any>): string {
  return inputs.rid;
}

// --- LOOP body CODE fns (Ollama-free while-loop e2e) ---
/**
 * A loop body 'a -> 'a over `{n, exited}`: increment n, set exited once n reaches 3.
 * Drives the while-loop e2e (3 iterations from a `{n:0, exited:false}` seed).
 */
export function loop_bump(inputs: Record<string, any>): Record<string, unknown> {
  const n = inputs.n + 1;
  return { n, exited: n >= 3 };
}

/**
 * Decrement the carried n. Drives a loop whose `while:` predicate divides by n
 * (`10 / n > 0`), so the predicate RAISES (division by zero) once n reaches 0 on
 * a later iteration — the engine must convert that predicate error into a failed run,
 * not let it escape the run uncaught.
 */
export function loop_countdown(inputs: Record<string, any>): Record<string, unknown> {
  return { n: inputs.n - 1 };
}

// Tags `record` saw, in call order; never cleared (a case observes only the return value).
export const RECORDED: unknown[] = [];

/** A side-effect probe: note that its node ran, then echo its tag. */
export function record(inputs: Record<string, any>): string {
  RECORDED.push(inputs.tag);
  return inputs.tag;
}

/** A `{tree}` record whose tree is a chain of `n` nested records (the value nesting tests). */
export function deep_tree(inputs: Record<string, any>): Record<string, unknown> {
  let tree: unknown = "leaf";
  for (let i = 0; i < inputs.n; i++) tree = { a: tree };
  return { tree };
}
