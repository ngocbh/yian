/**
 * Builtins: pure functions an author may call inside a `${...}` expression.
 *
 * A `${...}` span may be a plain reference (`${name}`) or a builtin call
 * (`${ render_as_json(briefs, 4) }`). `TEMPLATE_FNS` maps a builtin name to a pure value-to-value
 * function. The unified evaluator in `expressions.ts` dispatches every call through it, so every
 * builtin works in bindings and conditions; a prompt calls only the formatting ones (all but
 * `len`, see `isNonPromptBuiltin`), against the node's already-bound declared inputs.
 *
 * This is a deliberate, bounded bend of the "all computation is a node" law: builtins are pure
 * and read-only (formatting, plus `len`, which measures), and produce no graph node or edge. See
 * the Expressions page (`docs/src/content/docs/expressions.md`).
 *
 * A builtin is a host callback: it declares its parameters, so a call's arguments are checked
 * (`bindArguments`) before it runs, and it receives yian values. Five ship here
 * (`render_as_json`, `join`, `upper`, `lower`, `len`); `builtinResultType` gives the type a
 * shipped builtin always returns, so the loader can reject a field read on its result.
 *
 * Knows about: `typesys` (`Value`, `FileRef`, `InboxRef`, `Type`, `ValueKind`, `fromHost`,
 * `valueRepr`, `writeAsText`, `writeAsJson`), `diagnostics` (`reprText`) and `grammar`
 * (`uncallableReason`). Never imports: `expressions`, `template`, or anything above them.
 */

import { reprText } from "../diagnostics.ts";
import { FileRef, InboxRef, Type, type Value, ValueKind } from "../typesys/index.ts";
import { fromHost, valueRepr, writeAsJson, writeAsText } from "../typesys/values.ts";
import { uncallableReason } from "./grammar.ts";

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
 * `name` must be one a call can reach: a name (`my-fn` would read as `my - fn`), not a literal
 * word (`true`, `none`, `inf`) and not `isinstance`.
 *
 * Each param's `default` is read as a host value, so `{name: "indent", default: 2}` gives the int
 * `2n`. `fn` must be pure and deterministic: replay and the request rebuild bind a node again from
 * the run log, calling it again with the same arguments.
 *
 * @throws {RangeError} If no expression can call `name`.
 * @throws {TypeCheckError} If a default is no yian value (`fromHost`'s message).
 */
export function registerTemplateFn(name: string, fn: TemplateFn): void {
  const uncallable = uncallableReason(name);
  if (uncallable !== null) throw new RangeError(`expression function ${reprText(name)} can never be called: ${uncallable}`);
  const params = fn.params.map((p) => {
    if (!Object.hasOwn(p, "default")) return { name: p.name };
    const where = `expression function ${reprText(name)}, default of ${reprText(p.name)}`;
    return { name: p.name, default: fromHost(p.default, where) };
  });
  // The stored entry carries the normalized params; `call` stays a method call on the host's
  // object, so `this` inside it is still that object.
  TEMPLATE_FNS.set(name, { params, call: (...args) => fn.call(...args) });
}

/**
 * Bind a call's arguments to `fn`'s parameters.
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

/**
 * Return a value's kind as messages name it: `null`, `bool`, `int`, `float`, `str`, `list`,
 * `record`, `file` or `inbox`.
 */
export function kindName(value: Value): string {
  if (value === null) return "null";
  switch (typeof value) {
    case "boolean":
      return "bool";
    case "bigint":
      return "int";
    case "number":
      return "float";
    case "string":
      return "str";
  }
  if (Array.isArray(value)) return "list";
  if (InboxRef.isInboxRef(value)) return "inbox";
  return FileRef.isFileRef(value) ? "file" : "record";
}

/**
 * Throw when a builtin argument is null, so it never becomes text like "null". `fallback` is the
 * example default the message suggests.
 */
function rejectNull(fn: string, value: Value, fallback = '""'): void {
  if (value === null) throw new TypeError(`${fn}() got null; give a default, e.g. \`${fn}(x :- ${fallback})\``);
}

/** Return `s` if it is text; throw on null or any other kind. */
function requireText(fn: string, s: Value): string {
  rejectNull(fn, s);
  if (typeof s !== "string") throw new TypeError(`${fn}() takes str, got ${kindName(s)}`);
  return s;
}

/**
 * Pretty-print `value` as JSON (`writeAsJson`): the headline builtin. A file renders as its
 * mapping `{"uri", "mime", "name"}`. A null `value` is an error; a null nested inside it renders
 * as `null`.
 */
function renderAsJson(value: Value, indent: Value): string {
  rejectNull("render_as_json", value);
  if (typeof indent !== "bigint" || indent < 0n) {
    throw new TypeError(`render_as_json() takes an int indent of 0 or more, got ${valueRepr(indent)}`);
  }
  // A huge indent becomes a huge number (or Infinity), and the writer's `repeat` then fails.
  return writeAsJson(value, Number(indent));
}

/**
 * Join a list's elements, each in the text form, with `sep`. A null list or element is an
 * error, and so is a non-list (a string is not split into characters).
 */
function join(value: Value, sep: Value): string {
  rejectNull("join", value, "[]");
  if (!Array.isArray(value)) throw new TypeError(`join() takes a list, got ${kindName(value)}`);
  if (value.includes(null)) throw new TypeError("join() got a null element; drop or default it before joining");
  if (typeof sep !== "string") throw new TypeError(`join() takes str for sep, got ${kindName(sep)}`);
  return value.map(writeAsText).join(sep);
}

registerTemplateFn("render_as_json", {
  params: [{ name: "value" }, { name: "indent", default: 2 }],
  call: renderAsJson,
});
registerTemplateFn("join", { params: [{ name: "value" }, { name: "sep", default: "\n" }], call: join });
registerTemplateFn("upper", { params: [{ name: "s" }], call: (s) => requireText("upper", s).toUpperCase() });
registerTemplateFn("lower", { params: [{ name: "s" }], call: (s) => requireText("lower", s).toLowerCase() });

/** Return the length of text (code points) or of a list (items). */
function len(value: Value): bigint {
  rejectNull("len", value, "[]");
  if (typeof value === "string") return BigInt([...value].length);
  if (Array.isArray(value)) return BigInt(value.length);
  throw new TypeError(`len() takes str or list, got ${kindName(value)}`);
}
registerTemplateFn("len", { params: [{ name: "obj" }], call: len });

// The shipped `len` entry: a host that registers its own `len` replaces it, and a prompt may call
// that one.
const SHIPPED_LEN = TEMPLATE_FNS.get("len")!;

/**
 * Return whether `name` names a shipped builtin a prompt can't call (`len`): it computes a value,
 * which belongs in an input binding. Bindings and conditions can call it.
 */
export function isNonPromptBuiltin(name: string): boolean {
  return TEMPLATE_FNS.get(name) === SHIPPED_LEN;
}

// Each shipped builtin's result type, keyed by its stored entry, so a host that registers another
// function under a shipped name gets no result type. Typedef names are PascalCase, so no typedef
// can take one of these names.
const SHIPPED = new Map<TemplateFn | undefined, Type>(
  Object.entries({
    render_as_json: Type.scalar(ValueKind.STRING),
    join: Type.scalar(ValueKind.STRING),
    upper: Type.scalar(ValueKind.STRING),
    lower: Type.scalar(ValueKind.STRING),
    len: Type.scalar(ValueKind.INTEGER),
  }).map(([name, type]) => [TEMPLATE_FNS.get(name), type]),
);

/**
 * Return the type builtin `name` always returns: `int` for the shipped `len`, `str` for the other
 * builtins shipped here, `null` for an unknown name or a host function registered under the name.
 */
export function builtinResultType(name: string): Type | null {
  return SHIPPED.get(TEMPLATE_FNS.get(name)) ?? null;
}
