/**
 * Builtins: pure functions an author may call inside a `${...}` expression.
 *
 * A `${...}` span may be a plain reference (`${name}`) or a builtin call
 * (`${ render_as_json(briefs, 4) }`). `TEMPLATE_FNS` maps a builtin name to a pure value-to-value
 * function. The unified evaluator in `expressions.ts` dispatches every call through it, so
 * builtins work in prompts, bindings and conditions; a prompt calls them against the node's
 * already-bound declared inputs.
 *
 * This is a deliberate, bounded bend of the "all computation is a node" law: builtins are
 * read-only formatting and produce no graph node or edge. See `docs/expressions.md`.
 *
 * A builtin is a host callback: it declares its parameters, so a call's arguments are checked
 * (`bindArguments`) before it runs, and it receives yian values. For now this holds the registry
 * only; the four shipped builtins (`render_as_json`, `join`, `upper`, `lower`) and the result
 * type they share land here later.
 *
 * Knows about: `typesys` (`Value`, `fromHost`) and `diagnostics` (`reprText`). Never imports:
 * `expressions`, `template`, or anything above them.
 */

import { reprText } from "../diagnostics.ts";
import type { Value } from "../typesys/index.ts";
import { fromHost } from "../typesys/values.ts";

/** One parameter of a builtin; one with an own `default` is optional. */
export interface TemplateParam {
  readonly name: string;
  readonly default?: Value;
}

/** A builtin: its parameters, and the function a call runs. */
export interface TemplateFn {
  /** The parameters in order; a param with an own `default` is optional. */
  readonly params: readonly TemplateParam[];
  /** Run the builtin with one yian value per param, defaults filled in. */
  call(...args: Value[]): unknown;
}

/** Each builtin by the name a `${name(...)}` call uses. */
export const TEMPLATE_FNS = new Map<string, TemplateFn>();

/**
 * Register `fn` as the builtin `name`, replacing any builtin of that name (last writer wins).
 *
 * Each param's `default` is read as a host value, so `{name: "indent", default: 2}` gives the int
 * `2n`.
 *
 * @throws {TypeCheckError} If a default is no yian value (`fromHost`'s message).
 */
export function registerTemplateFn(name: string, fn: TemplateFn): void {
  const params = fn.params.map((p) => {
    if (!Object.hasOwn(p, "default")) return { name: p.name };
    const where = `expression function ${reprText(name)}, default of ${reprText(p.name)}`;
    return { name: p.name, default: fromHost(p.default, where) };
  });
  TEMPLATE_FNS.set(name, { params, call: (...args) => fn.call(...args) });
}

/**
 * Bind a call's arguments to `fn`'s parameters, as Python's `Signature.bind` does.
 *
 * Each positional fills the next param; then each remaining param takes its keyword or its
 * default; then any keyword left names no param.
 *
 * @param positional The positional arguments, in call order.
 * @param keywords The keyword arguments by name, in call order.
 * @returns The arguments in parameter order with defaults filled in, or why they don't bind:
 *   `too many positional arguments`, `multiple values for argument 'x'`,
 *   `missing a required argument: 'x'`, or `got an unexpected keyword argument 'x'`.
 */
export function bindArguments<T>(
  fn: TemplateFn,
  positional: readonly T[],
  keywords: ReadonlyMap<string, T>,
): (T | Value)[] | string {
  const { params } = fn;
  const bound: (T | Value)[] = [];
  for (const [i, value] of positional.entries()) {
    if (i === params.length) return "too many positional arguments";
    const name = params[i]!.name;
    if (keywords.has(name)) return `multiple values for argument ${reprText(name)}`;
    bound.push(value);
  }
  for (const p of params.slice(positional.length)) {
    if (keywords.has(p.name)) bound.push(keywords.get(p.name) as T);
    else if (Object.hasOwn(p, "default")) bound.push(p.default as Value);
    else return `missing a required argument: ${reprText(p.name)}`;
  }
  for (const name of keywords.keys()) {
    if (!params.some((p) => p.name === name)) return `got an unexpected keyword argument ${reprText(name)}`;
  }
  return bound;
}
