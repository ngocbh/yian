/** CODE-node functions the seed cases reference: a TypeScript port of yian's `tests/seeds/fns.py`. */

// --- seed 01 (structured-agent) ---
/** `{rating: float, rationale: str}` -> a one-line verdict string. */
export function one_line_summary(inputs: Record<string, any>): string {
  const rating = "rating" in inputs ? inputs.rating : null;
  const rationale = "rationale" in inputs ? inputs.rationale : "";
  const lean = (rating || 0) >= 0 ? "positive" : "negative";
  return `${lean} (${(rating >= 0 ? "+" : "") + rating.toFixed(2)}): ${rationale}`.trim();
}

// --- seed 03 (research-one) ---
/** `{topic, as_of}` -> `{values: list[float], news: list[str]}`. */
export function fetch_facts(inputs: Record<string, any>): Record<string, any> {
  const topic = String("topic" in inputs ? inputs.topic : "").toUpperCase();
  return {
    values: [100.0, 101.5, 99.75, 102.25],
    news: [`${topic}: quarterly update`, `${topic}: reviewer note`],
  };
}

// --- seed 13 (types-objects) ---
/** Compose a nested plan matching seed 13's declared output. */
export function build_outline(inputs: Record<string, any>): Record<string, any> {
  const names: string[] = [...(("names" in inputs ? inputs.names : null) || [])];
  const quantity = ("quantity" in inputs ? inputs.quantity : null) || 0;
  const items = names.map((n) => `item ${n}`);
  const total = quantity;
  return {
    items,
    summary: {
      count: items.length,
      total,
      meta: {
        as_of: "as_of" in inputs ? inputs.as_of : null,
        dry_run: Boolean("dry_run" in inputs ? inputs.dry_run : false),
      },
    },
  };
}

// --- seed 12 (depends-on) ---
/** Side-effect node: warm a cache, return a `bool` nobody binds. */
export function prime_cache(inputs: Record<string, any>): boolean {
  return true;
}

// --- seed 17 (effects-human-wait) ---
/** Echo the confirmed action. */
export function confirm_action(inputs: Record<string, any>): string {
  return String("answer" in inputs ? inputs.answer : "cancel");
}

// --- errors/ negative-gallery CODE callables (deterministic, no IO) ---
/** Return the integer 1 (a deterministic numeric producer for assert fixtures). */
export function const_one(inputs: Record<string, any>): number {
  return 1;
}

/** Raise — a CODE node that always fails (runtime-failure fixture). */
export function fail_always(inputs: Record<string, any>): never {
  throw new Error("intentional CODE failure");
}

/** Return a str even where the node declares `int` — trips the write-boundary check. */
export function wrong_type(inputs: Record<string, any>): string {
  return "not-an-int";
}

/** Return the single bound input value (a passthrough leaf). */
export function echo_value(inputs: Record<string, any>): any {
  return Object.values(inputs)[0] ?? null;
}

// --- human_input questions ref source ---
/** `{seed}` -> a one-element `list[object]` of question records (a gate's `questions:` ref). */
export function questions_seed(inputs: Record<string, any>): Record<string, any>[] {
  return [{ question: inputs.seed, header: "H", options: [{ label: "A" }, { label: "B" }] }];
}

// --- seed 26 (human-questions) — route on the keyed answer record ---
/** `{ans: {Framework, Notes}}` -> the chosen framework label (routes on the answer). */
export function chosen_framework(inputs: Record<string, any>): string {
  return String(inputs.ans.Framework);
}

// --- seed 30 (reference-code) — the reference twin of seed 29's inline body ---
/** `{rating: float, label: str}` -> a verdict string. */
export function verdict_line(inputs: Record<string, any>): string {
  const lean = inputs.rating >= 0 ? "positive" : "negative";
  return `${inputs.label}: ${lean} (${(inputs.rating >= 0 ? "+" : "") + inputs.rating.toFixed(2)})`;
}
