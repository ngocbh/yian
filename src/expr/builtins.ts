/**
 * Builtins: pure functions an author may call inside a `${...}` expression.
 *
 * A `${...}` span may be a plain reference (`${name}`) or a builtin call
 * (`${ render_as_json(briefs, 4) }`). `TEMPLATE_FNS` maps a builtin name to a pure value-to-value
 * function. The unified evaluator in `expressions.ts` dispatches every call through it, so every
 * builtin works in bindings and conditions; a prompt calls only the ones that give text (see
 * `isNonPromptBuiltin`), against the node's already-bound declared inputs.
 *
 * This is a deliberate, bounded bend of the "all computation is a node" law: builtins are pure
 * and read-only (formatting, converting and measuring values), and produce no graph node or edge.
 * See the Expressions page (`docs/src/content/docs/expressions.md`).
 *
 * A builtin is a host callback: it declares its parameters, so a call's arguments are checked
 * (`bindArguments`) before it runs, and it receives yian values. The ones shipped here format
 * (`render_as_json`, `join`, `upper`, `lower`, `str`), work on text (`startswith`, `endswith`,
 * `split`, `strip`, `lstrip`, `rstrip`, `replace`), convert (`int`, `float`), and compute over
 * numbers and lists (`len`, `round`, `abs`, `min`, `max`, `sum`, `any`, `all`). Where Python has
 * the same operation they give Python's result, character for character and bit for bit.
 * `builtinResultType` gives the type a shipped builtin returns, from its arguments' types where it
 * depends on them, so the loader can type its result.
 *
 * Knows about: `typesys` (`Value`, `FileRef`, `InboxRef`, `MemoryRef`, `Type`, `ValueKind`,
 * `fromHost`, `isListKind`, `isRecord`, `valueRepr`, `writeAsText`, `writeAsJson`), `diagnostics`
 * (`reprText`, `compareCodePoints`) and `grammar` (`uncallableReason`). Never imports:
 * `expressions`, `template`, or anything above them.
 */

import { compareCodePoints, reprText } from "../diagnostics.ts";
import { FileRef, InboxRef, MemoryRef, Type, type Value, ValueKind } from "../typesys/index.ts";
import {
  canonicalTypeLabel,
  fromHost,
  isListKind,
  isRecord,
  valueRepr,
  writeAsJson,
  writeAsText,
} from "../typesys/values.ts";
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
 * `record`, `file`, `inbox` or `memory`.
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
  if (MemoryRef.isMemoryRef(value)) return "memory";
  return FileRef.isFileRef(value) ? "file" : "record";
}

/**
 * Throw when a builtin argument is null, so it never becomes text like "null". `fallback` is the
 * example default the message suggests.
 */
function rejectNull(fn: string, value: Value, fallback = '""'): void {
  if (value === null) throw new TypeError(`${fn}() got null; give a default, e.g. \`${fn}(x :- ${fallback})\``);
}

/** Return `value` if it is text; throw on null or any other kind, naming parameter `param` after the first. */
function requireText(fn: string, value: Value, param: string | null = null): string {
  if (param === null) rejectNull(fn, value);
  if (typeof value !== "string") {
    throw new TypeError(`${fn}() takes str${param ? ` for ${param}` : ""}, got ${kindName(value)}`);
  }
  return value;
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

// Python's `str.isspace()` set, which `strip`, `lstrip`, `rstrip` and `split` remove. JS's `trim()`
// differs: it removes U+FEFF and keeps U+001C to U+001F and U+0085.
const WHITESPACE: ReadonlySet<number> = new Set([
  0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003,
  0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
]);

/** Return whether the character `ch` (one code point) is whitespace, as Python's `str.isspace()` says. */
function isSpace(ch: string): boolean {
  return WHITESPACE.has(ch.codePointAt(0)!);
}

/** Return `value` if it is text, or null for an omitted (null) optional parameter `param`. */
function optionalText(fn: string, value: Value, param: string): string | null {
  return value === null ? null : requireText(fn, value, param);
}

registerTemplateFn("startswith", {
  params: [{ name: "s" }, { name: "prefix" }],
  call: (s, prefix) => requireText("startswith", s).startsWith(requireText("startswith", prefix, "prefix")),
});
registerTemplateFn("endswith", {
  params: [{ name: "s" }, { name: "suffix" }],
  call: (s, suffix) => requireText("endswith", s).endsWith(requireText("endswith", suffix, "suffix")),
});

/**
 * Split text into parts. With no `sep`, split on runs of whitespace and drop empty parts; with a
 * `sep`, split at each occurrence, keeping empty parts. Both work on whole characters (code
 * points), as Python's `str.split` does.
 */
function split(s: Value, sep: Value): string[] {
  const text = requireText("split", s);
  const separator = optionalText("split", sep, "sep");
  // Well-formed text and separator: a code unit match never starts or ends inside a surrogate pair.
  if (separator !== null) {
    if (separator === "") throw new TypeError("split() got an empty sep; omit it to split on whitespace");
    return text.split(separator);
  }
  const parts: string[] = [];
  let part = "";
  for (const ch of text) {
    if (!isSpace(ch)) part += ch;
    else if (part !== "") {
      parts.push(part);
      part = "";
    }
  }
  if (part !== "") parts.push(part);
  return parts;
}
registerTemplateFn("split", { params: [{ name: "s" }, { name: "sep", default: null }], call: split });

/**
 * Return `s` without the characters at its `ends` that `chars` holds, or without whitespace when
 * `chars` is null; characters are code points, as in Python's `str.strip`.
 */
function stripEnds(fn: string, s: Value, chars: Value, ends: "both" | "left" | "right"): string {
  const text = [...requireText(fn, s)];
  const given = optionalText(fn, chars, "chars");
  const set = given === null ? null : new Set(given);
  const strips = set === null ? isSpace : (ch: string) => set.has(ch);
  let start = 0;
  let stop = text.length;
  if (ends !== "right") while (start < stop && strips(text[start]!)) start += 1;
  if (ends !== "left") while (stop > start && strips(text[stop - 1]!)) stop -= 1;
  return text.slice(start, stop).join("");
}
for (const [name, ends] of [["strip", "both"], ["lstrip", "left"], ["rstrip", "right"]] as const) {
  registerTemplateFn(name, {
    params: [{ name: "s" }, { name: "chars", default: null }],
    call: (s, chars) => stripEnds(name, s, chars, ends),
  });
}

/**
 * Replace every occurrence of `old` in `s` with `new`. An empty `old` puts `new` before each
 * character (code point) and at the end, as Python's `str.replace` does.
 */
function replace(s: Value, old: Value, replacement: Value): string {
  const text = requireText("replace", s);
  const target = requireText("replace", old, "old");
  const by = requireText("replace", replacement, "new");
  // `split`/`join`, not `replaceAll`, which reads `$&` and the like in the replacement.
  if (target === "") return ["", ...text, ""].join(by);
  return text.split(target).join(by);
}
registerTemplateFn("replace", { params: [{ name: "s" }, { name: "old" }, { name: "new" }], call: replace });

// --- Conversions --- //

// The whitespace `int()` and `float()` skip around a number: Python's set without U+001C to
// U+001F, which Python's number parser doesn't skip.
const SPACE = "[\\t\\n\\v\\f\\r \\u0085\\u00a0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]*";
// ASCII digits, with single underscores between them.
const DIGITS = "[0-9](?:_?[0-9])*";
// Python's `int()` text in base 10.
const INT_TEXT = new RegExp(`^${SPACE}([+-]?)(${DIGITS})${SPACE}$`);
// Python's `float()` text: digits with an optional fraction and exponent, or `inf`, `infinity`
// or `nan` in any case.
const DECIMAL = `(?:${DIGITS}(?:\\.(?:${DIGITS})?)?|\\.${DIGITS})(?:e[+-]?${DIGITS})?`;
const FLOAT_TEXT = new RegExp(`^${SPACE}([+-]?)(?:(inf(?:inity)?)|(nan)|(${DECIMAL}))${SPACE}$`, "i");

/**
 * Return `value` as text, as `${value}` writes it into text: text as is, anything else as
 * compact JSON.
 */
function str(value: Value): string {
  rejectNull("str", value);
  // `${x}` refuses a memory too: only an AGENT renders one, as its recall.
  if (MemoryRef.isMemoryRef(value)) throw new TypeError("str() can't write a memory as text");
  return writeAsText(value);
}

/**
 * Return `value` as an int: text in Python's `int()` syntax (ASCII digits only), a float
 * truncated toward zero, a bool as 0 or 1.
 */
function int(value: Value): bigint {
  rejectNull("int", value, "0");
  if (typeof value === "bigint") return value;
  if (typeof value === "boolean") return value ? 1n : 0n;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new TypeError(`int() can't convert ${valueRepr(value)} to an int`);
    return BigInt(Math.trunc(value));
  }
  if (typeof value !== "string") throw new TypeError(`int() takes str, int, float or bool, got ${kindName(value)}`);
  const match = INT_TEXT.exec(value);
  if (match === null) throw new TypeError(`int() can't convert ${valueRepr(value)} to an int`);
  const digits = BigInt(match[2]!.replaceAll("_", ""));
  return match[1] === "-" ? -digits : digits;
}

/**
 * Return `value` as a float: text in Python's `float()` syntax (ASCII digits only, `inf` but no
 * `nan`), an int rounded to the nearest float, a bool as 0.0 or 1.0.
 */
function float(value: Value): number {
  rejectNull("float", value, "0.0");
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return value ? 1 : 0;
  if (typeof value === "bigint") {
    // `Number` rounds a bigint to the nearest float, ties to even, as Python does.
    const rounded = Number(value);
    if (!Number.isFinite(rounded)) throw new TypeError("float() got an int too large for a float");
    return rounded;
  }
  if (typeof value !== "string") throw new TypeError(`float() takes str, int, float or bool, got ${kindName(value)}`);
  const match = FLOAT_TEXT.exec(value);
  if (match === null) throw new TypeError(`float() can't convert ${valueRepr(value)} to a float`);
  if (match[3] !== undefined) throw new TypeError(`float() can't convert ${valueRepr(value)}: yian has no NaN`);
  const sign = match[1] === "-" ? -1 : 1;
  if (match[2] !== undefined) return sign * Infinity;
  // The syntax is checked, so `Number` reads only digits, a point and an exponent, and rounds to
  // the nearest float.
  return sign * Number(match[4]!.replaceAll("_", ""));
}

registerTemplateFn("str", { params: [{ name: "x" }], call: str });
registerTemplateFn("int", { params: [{ name: "x" }], call: int });
registerTemplateFn("float", { params: [{ name: "x" }], call: float });

// --- Exact float arithmetic --- //

/** Return `[m, e]` with `|x| = m · 2^e` exactly, for a finite `x`. */
export function floatParts(x: number): [m: bigint, e: number] {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  const bits = view.getBigUint64(0);
  const field = Number((bits >> 52n) & 0x7ffn);
  const fraction = bits & ((1n << 52n) - 1n);
  return field === 0 ? [fraction, -1074] : [fraction | (1n << 52n), field - 1075];
}

/**
 * Return the float nearest `num / den` for a non-negative `num` and a positive `den`: ties to
 * even, subnormals included, `Infinity` past the float range.
 */
export function roundQuotient(num: bigint, den: bigint): number {
  if (num === 0n) return 0;
  const bitLength = (value: bigint) => value.toString(2).length;
  // Scale so the integer quotient q has at least 55 bits: num / den = (q + a remainder) · 2^-s.
  const s = 55 - (bitLength(num) - bitLength(den));
  const [scaledNum, scaledDen] = s >= 0 ? [num << BigInt(s), den] : [num, den << BigInt(-s)];
  const q = scaledNum / scaledDen;
  const sticky = scaledNum % scaledDen !== 0n;
  const k = bitLength(q);
  // q's leading bit is worth 2^(k - 1 - s).
  if (k - 1 - s > 1023) return Infinity;
  // Keep 53 bits, or fewer for a subnormal, whose lowest bit is worth 2^-1074.
  const drop = Math.max(k - 53, s - 1074);
  const rest = q & ((1n << BigInt(drop)) - 1n);
  const half = 1n << BigInt(drop - 1);
  let mantissa = q >> BigInt(drop);
  if (rest > half || (rest === half && (sticky || (mantissa & 1n) === 1n))) mantissa += 1n;
  // Exact: mantissa <= 2^53 and the scale is a power of two in [2^-1074, 2^971], so only a
  // mantissa rounded up to 2^53 at the top overflows, to Infinity as IEEE rounding does.
  return Number(mantissa) * 2 ** (drop - s);
}


/** Return `num / den` rounded to an integer, ties to even, for a non-negative `num` and a positive `den`. */
function divideHalfEven(num: bigint, den: bigint): bigint {
  const q = num / den;
  const twice = (num % den) * 2n;
  return twice > den || (twice === den && q % 2n === 1n) ? q + 1n : q;
}

// Past these, `round(x, n)` of a float gives `x`, or a zero with `x`'s sign, as in Python.
const ROUND_DIGITS_MAX = 323n;
const ROUND_DIGITS_MIN = -308n;

/**
 * Round `x` to `n` decimal places, ties to even, exactly on its true value, then to the nearest
 * float, as Python's `round(x, n)` does: `round(2.675, 2)` is `2.67`, since the float 2.675 is a
 * little below it.
 */
function roundFloat(x: number, n: bigint): number {
  if (!Number.isFinite(x) || x === 0 || n > ROUND_DIGITS_MAX) return x;
  if (n < ROUND_DIGITS_MIN) return x < 0 ? -0 : 0;
  const [m, e] = floatParts(x);
  const scale = 10n ** (n < 0n ? -n : n);
  let num = n < 0n ? m : m * scale;
  let den = n < 0n ? scale : 1n;
  if (e >= 0) num <<= BigInt(e);
  else den <<= BigInt(-e);
  const q = divideHalfEven(num, den);
  const rounded = n < 0n ? roundQuotient(q * scale, 1n) : roundQuotient(q, scale);
  if (rounded === Infinity) throw new TypeError(`round() of ${valueRepr(x)} is too large for a float`);
  return x < 0 ? -rounded : rounded;
}

/** Round the int `x` to `n` decimal places, ties to even: itself for `n >= 0`. */
function roundInt(x: bigint, n: bigint): bigint {
  if (n >= 0n) return x;
  const size = x < 0n ? -x : x;
  // `size < 10^digits`, so past `digits` places it rounds to 0 without building the power.
  if (-n > BigInt(size.toString().length)) return 0n;
  const scale = 10n ** -n;
  const q = divideHalfEven(size, scale) * scale;
  return x < 0n ? -q : q;
}

/** Return `x` rounded to a whole number, ties to even. */
function roundToEven(x: number): number {
  const floor = Math.floor(x);
  // Exact: below 2^52 the fraction is representable, and above it every float is whole.
  const fraction = x - floor;
  if (fraction !== 0.5) return fraction < 0.5 ? floor : floor + 1;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Round `x` as Python's `round` does, ties to even: with `n` null, to an int; with `n`, to `n`
 * decimal places (negative `n` rounds to tens, hundreds, ...), giving a float for a float `x`
 * and an int for an int.
 */
function round(x: Value, n: Value): bigint | number {
  rejectNull("round", x, "0");
  const value = typeof x === "boolean" ? BigInt(x) : x;
  if (typeof value !== "bigint" && typeof value !== "number") {
    throw new TypeError(`round() takes a number, got ${kindName(x)}`);
  }
  if (n === null) {
    if (typeof value === "bigint") return value;
    if (!Number.isFinite(value)) throw new TypeError(`round() can't round ${valueRepr(value)} to an int`);
    return BigInt(roundToEven(value));
  }
  const places = typeof n === "boolean" ? BigInt(n) : n;
  if (typeof places !== "bigint") throw new TypeError(`round() takes int for n, got ${kindName(n)}`);
  return typeof value === "bigint" ? roundInt(value, places) : roundFloat(value, places);
}
registerTemplateFn("round", { params: [{ name: "x" }, { name: "n", default: null }], call: round });

// --- Numbers and lists --- //

/** Return whether a value is truthy: all but `false`, `null`, a zero, `""`, `[]` and a record with no field. */
export function truthy(value: Value): boolean {
  if (value === null || value === false || value === 0n || value === 0 || value === "") return false;
  if (Array.isArray(value)) return value.length > 0;
  if (isRecord(value)) return Object.keys(value).length > 0;
  return true;
}

/** Return `xs` if it is a list; throw on null or any other kind. */
function requireList(fn: string, xs: Value): Value[] {
  rejectNull(fn, xs, "[]");
  if (!Array.isArray(xs)) throw new TypeError(`${fn}() takes a list, got ${kindName(xs)}`);
  return xs;
}

/** Return a number as arithmetic reads it: an int or a bool (0 or 1) as a `bigint`, a float as is; else `null`. */
function numberOf(value: Value): bigint | number | null {
  if (typeof value === "boolean") return value ? 1n : 0n;
  return typeof value === "bigint" || typeof value === "number" ? value : null;
}

registerTemplateFn("any", { params: [{ name: "xs" }], call: (xs) => requireList("any", xs).some(truthy) });
registerTemplateFn("all", { params: [{ name: "xs" }], call: (xs) => requireList("all", xs).every(truthy) });

/**
 * Return the least (`min`) or greatest (`max`) item of a list of numbers, or of texts compared by
 * code point; the first of equal items, as in Python.
 */
function extreme(fn: "min" | "max", xs: Value): Value {
  const items = requireList(fn, xs);
  if (items.length === 0) throw new TypeError(`${fn}() got an empty list`);
  const kindOf = (item: Value): "number" | "text" => {
    if (item === null) throw new TypeError(`${fn}() got a null item`);
    if (typeof item === "string") return "text";
    if (numberOf(item) !== null) return "number";
    throw new TypeError(`${fn}() takes a list of numbers or texts, got a ${kindName(item)} item`);
  };
  let best: Value = items[0]!;
  const kind = kindOf(best);
  for (const item of items.slice(1)) {
    if (kindOf(item) !== kind) throw new TypeError(`${fn}() can't compare ${kindName(best)} and ${kindName(item)}`);
    let order: number;
    if (kind === "text") {
      order = compareCodePoints(item as string, best as string);
    } else {
      // JS compares a bigint with a number by exact value.
      const [a, b] = [numberOf(item)!, numberOf(best)!];
      order = a < b ? -1 : a > b ? 1 : 0;
    }
    // Only a strictly better item replaces `best`, so the first of equal items wins.
    if (fn === "min" ? order < 0 : order > 0) best = item;
  }
  return best;
}
registerTemplateFn("min", { params: [{ name: "xs" }], call: (xs) => extreme("min", xs) });
registerTemplateFn("max", { params: [{ name: "xs" }], call: (xs) => extreme("max", xs) });

const INT64_MIN = -(2n ** 63n);
const INT64_MAX = 2n ** 63n - 1n;

/** Whether `n` fits a 64-bit signed integer, where CPython's `sum` keeps its running int. */
function fits64(n: bigint): boolean {
  return INT64_MIN <= n && n <= INT64_MAX;
}

/** `a + b` as Python adds two numbers: two ints exactly, else as floats, an int rounded to the nearest one. */
function addNumbers(a: bigint | number, b: bigint | number): bigint | number {
  if (typeof a === "bigint" && typeof b === "bigint") return a + b;
  const [x, y] = [Number(a), Number(b)];
  if (!Number.isFinite(x) && typeof a === "bigint") throw new TypeError("sum() got an int too large for a float");
  if (!Number.isFinite(y) && typeof b === "bigint") throw new TypeError("sum() got an int too large for a float");
  return x + y;
}

/**
 * Add a list of numbers as CPython 3.12+'s `sum` does (`builtin_sum_impl` in `bltinmodule.c`),
 * so the same list gives the same float: ints stay exact; floats are added with Neumaier's
 * compensation once the running sum is a float; an empty list gives `0`.
 */
function sum(xs: Value): bigint | number {
  const items = requireList("sum", xs).map((item) => {
    const n = numberOf(item);
    if (n === null) throw new TypeError(`sum() takes a list of numbers, got a ${kindName(item)} item`);
    return n;
  });
  let i = 0;
  // CPython first keeps the sum in a 64-bit integer while every item is an int that fits.
  let whole = 0n;
  let result: bigint | number | null = null;
  for (; i < items.length && result === null; i++) {
    const item = items[i]!;
    if (typeof item === "bigint" && fits64(item) && fits64(whole + item)) whole += item;
    else result = addNumbers(whole, item);
  }
  if (result === null) return whole;
  if (typeof result === "number") {
    // Then, once the sum is a float, floats with compensation, and 64-bit ints added plainly.
    let total = result;
    let compensation = 0;
    let rest: bigint | null = null;
    for (; i < items.length && rest === null; i++) {
      const item = items[i]!;
      if (typeof item === "number") {
        const t = total + item;
        compensation += Math.abs(total) >= Math.abs(item) ? total - t + item : item - t + total;
        total = t;
      } else if (fits64(item)) {
        total += Number(item);
      } else {
        rest = item;
      }
    }
    // Not when the compensation is infinite or NaN, which would turn an infinite sum into NaN.
    if (compensation !== 0 && Number.isFinite(compensation)) total += compensation;
    result = rest === null ? total : addNumbers(total, rest);
  }
  // Past a large int, everything is added plainly.
  for (; i < items.length; i++) result = addNumbers(result, items[i]!);
  if (Number.isNaN(result)) throw new TypeError("sum() has no numeric result: it adds inf and -inf");
  return result;
}
registerTemplateFn("sum", { params: [{ name: "xs" }], call: sum });

/** Return the absolute value of a number; a bool counts as 0 or 1. */
function abs(x: Value): bigint | number {
  rejectNull("abs", x, "0");
  const n = numberOf(x);
  if (n === null) throw new TypeError(`abs() takes a number, got ${kindName(x)}`);
  if (typeof n === "number") return Math.abs(n);
  return n < 0n ? -n : n;
}
registerTemplateFn("abs", { params: [{ name: "x" }], call: abs });

/** Return the length of text (code points) or of a list (items). */
function len(value: Value): bigint {
  rejectNull("len", value, "[]");
  if (typeof value === "string") return BigInt([...value].length);
  if (Array.isArray(value)) return BigInt(value.length);
  throw new TypeError(`len() takes str or list, got ${kindName(value)}`);
}
registerTemplateFn("len", { params: [{ name: "obj" }], call: len });

const STR = Type.scalar(ValueKind.STRING);
const BOOL = Type.scalar(ValueKind.BOOLEAN);
const INT = Type.scalar(ValueKind.INTEGER);
const FLOAT = Type.scalar(ValueKind.NUMBER);

/** An argument at load: its inferred type, `null` when unknown, or `undefined` when the call leaves it out. */
export type ArgType = Type | null | undefined;

// The kind names (as `kindName` gives them) a non-null value of each scalar kind may have. A
// `float` slot keeps an int as it is, so a float-typed value may be an int.
const KIND_NAMES: ReadonlyMap<ValueKind, readonly string[]> = new Map([
  [ValueKind.STRING, ["str"]],
  [ValueKind.DATE, ["str"]],
  [ValueKind.DATETIME, ["str"]],
  [ValueKind.INTEGER, ["int"]],
  [ValueKind.NUMBER, ["float", "int"]],
  [ValueKind.BOOLEAN, ["bool"]],
  [ValueKind.FILE, ["file"]],
  [ValueKind.INBOX, ["inbox"]],
  [ValueKind.MEMORY, ["memory"]],
]);

/**
 * Return the kinds (as `kindName` names them) a non-null value of `typ` may have: none for
 * `None`; `null` when `typ` doesn't tell (`Any`, an untyped `object`).
 */
export function valueKinds(typ: Type): ReadonlySet<string> | null {
  if (typ.members !== undefined) {
    const kinds = new Set<string>();
    for (const member of typ.members) {
      const each = valueKinds(member);
      if (each === null) return null;
      for (const kind of each) kinds.add(kind);
    }
    return kinds;
  }
  if (typ.kind === ValueKind.NONE) return new Set();
  if (isListKind(typ.kind)) return new Set(["list"]);
  if (typ.kind === ValueKind.OBJECT) return typ.fields !== undefined ? new Set(["record"]) : null;
  if (typ.kind === ValueKind.NUMBER && typ.bounds !== undefined) {
    // Bounds that hold no whole number, as a literal `1.5`'s do, leave no int.
    const [lo, hi] = typ.bounds;
    if (lo !== null && hi !== null && Math.ceil(Number(lo)) > Number(hi)) return new Set(["float"]);
  }
  const kinds = KIND_NAMES.get(typ.kind);
  return kinds === undefined ? null : new Set(kinds);
}

/** Whether every non-null value of `typ` is an int or a bool; `false` when `typ` is unknown. */
function intLike(typ: ArgType): boolean {
  const kinds = typ ? valueKinds(typ) : null;
  return kinds !== null && kinds.size > 0 && [...kinds].every((kind) => kind === "int" || kind === "bool");
}

/**
 * Whether `typ` is a float type: its non-null values are floats, or ints a float slot keeps;
 * `false` when `typ` is unknown.
 */
function floatLike(typ: ArgType): boolean {
  const kinds = typ ? valueKinds(typ) : null;
  return kinds !== null && kinds.has("float") && [...kinds].every((kind) => kind === "float" || kind === "int");
}

/**
 * `round(x)` gives an int; `round(x, n)` an int for an int `x` and a float for a float `x`. A
 * null `n` is left out, so an `n` that may be null leaves a float `x`'s result unknown.
 */
function roundType([x, n]: readonly ArgType[]): Type | null {
  if (n === undefined || (n !== null && valueKinds(n)?.size === 0) || intLike(x)) return INT;
  const nNeverNull = n !== null && !n.nullable && n.members === undefined && valueKinds(n) !== null;
  return nNeverNull && floatLike(x) ? FLOAT : null;
}

/** The item type of a list type, non-null (a null item fails the call), or `null` when unknown. */
function itemOf(typ: ArgType): Type | null {
  if (!typ || !isListKind(typ.kind) || typ.element === undefined) return null;
  return typ.element.replace({ nullable: false });
}

/** `sum` of ints is an int, of floats a float (an empty list gives the int `0`, which fits a float). */
function sumType([xs]: readonly ArgType[]): Type | null {
  const item = itemOf(xs);
  return intLike(item) ? INT : floatLike(item) ? FLOAT : null;
}

/** `abs` gives the number type it takes; a bool's is an int. */
function absType([x]: readonly ArgType[]): Type | null {
  return intLike(x) ? INT : floatLike(x) ? FLOAT : null;
}

/** How a builtin's result type follows from its arguments' types, in parameter order. */
type ResultType = (args: readonly ArgType[]) => Type | null;

// Each shipped builtin's result type, or how its arguments' types give it, keyed by its stored
// entry, so a host that registers another function under a shipped name gets no result type,
// and a prompt may call it. Typedef names are PascalCase, so no typedef can take one of these
// names.
const SHIPPED = new Map<TemplateFn | undefined, { name: string; type: Type | ResultType }>(
  Object.entries({
    render_as_json: STR,
    join: STR,
    upper: STR,
    lower: STR,
    len: INT,
    startswith: BOOL,
    endswith: BOOL,
    split: new Type({ kind: ValueKind.LIST_STRING, element: STR }),
    strip: STR,
    lstrip: STR,
    rstrip: STR,
    replace: STR,
    str: STR,
    int: INT,
    float: FLOAT,
    round: roundType,
    any: BOOL,
    all: BOOL,
    min: ([xs]: readonly ArgType[]) => itemOf(xs),
    max: ([xs]: readonly ArgType[]) => itemOf(xs),
    sum: sumType,
    abs: absType,
  }).map(([name, type]) => [TEMPLATE_FNS.get(name), { name, type }]),
);

// The shipped builtins a prompt may call: those that give text. The others compute a value,
// which belongs in an input binding.
const PROMPT_BUILTINS: ReadonlySet<string> = new Set([
  "render_as_json",
  "join",
  "upper",
  "lower",
  "strip",
  "lstrip",
  "rstrip",
  "replace",
  "str",
]);

/**
 * Return whether `name` names a shipped builtin a prompt can't call (`len`, `split`, ...): it
 * computes a value, which belongs in an input binding. Bindings and conditions can call it.
 */
export function isNonPromptBuiltin(name: string): boolean {
  const shipped = SHIPPED.get(TEMPLATE_FNS.get(name));
  return shipped !== undefined && !PROMPT_BUILTINS.has(shipped.name);
}

/** What a shipped builtin's parameter takes, for the load check: kinds as `kindName` names them. */
interface Takes {
  /** The kinds it takes, `null` for any non-null value. */
  readonly kinds: ReadonlySet<string> | null;
  /** For a list, the kinds its items take. */
  readonly items?: ReadonlySet<string>;
  /** Whether a null leaves the parameter out. */
  readonly omittable?: boolean;
  /** The kinds as a message names them. */
  readonly label: string;
}

const NUMBER_KINDS = ["int", "float", "bool"];
const TEXT: Takes = { kinds: new Set(["str"]), label: "str" };
const LIST: Takes = { kinds: new Set(["list"]), label: "a list" };
const NUMBER: Takes = { kinds: new Set(NUMBER_KINDS), label: "a number" };
const ANYTHING: Takes = { kinds: null, label: "a value" };
const OPTIONAL_TEXT: Takes = { ...TEXT, omittable: true };

// What each shipped builtin's parameters take, in order.
const TAKES: Readonly<Record<string, readonly Takes[]>> = {
  render_as_json: [ANYTHING, { kinds: new Set(["int"]), label: "int" }],
  join: [LIST, TEXT],
  upper: [TEXT],
  lower: [TEXT],
  len: [{ kinds: new Set(["str", "list"]), label: "str or list" }],
  startswith: [TEXT, TEXT],
  endswith: [TEXT, TEXT],
  split: [TEXT, OPTIONAL_TEXT],
  strip: [TEXT, OPTIONAL_TEXT],
  lstrip: [TEXT, OPTIONAL_TEXT],
  rstrip: [TEXT, OPTIONAL_TEXT],
  replace: [TEXT, TEXT, TEXT],
  str: [ANYTHING],
  int: [{ kinds: new Set(["str", ...NUMBER_KINDS]), label: "str, int, float or bool" }],
  float: [{ kinds: new Set(["str", ...NUMBER_KINDS]), label: "str, int, float or bool" }],
  round: [NUMBER, { kinds: new Set(["int", "bool"]), omittable: true, label: "int" }],
  any: [LIST],
  all: [LIST],
  min: [{ ...LIST, items: new Set(["str", ...NUMBER_KINDS]), label: "a list of numbers or texts" }],
  max: [{ ...LIST, items: new Set(["str", ...NUMBER_KINDS]), label: "a list of numbers or texts" }],
  sum: [{ ...LIST, items: new Set(NUMBER_KINDS), label: "a list of numbers" }],
  abs: [NUMBER],
};

/**
 * Return why an argument of type `typ` for parameter `index` of the shipped builtin `name` can
 * never be taken, else `null`: its value is only ever null where the parameter needs one, or
 * never of a kind the parameter takes (a list's items included, for `min`, `max` and `sum`). A
 * type that may hold a value the parameter takes is left to the call.
 */
export function builtinArgumentError(name: string, index: number, typ: Type): string | null {
  const shipped = SHIPPED.get(TEMPLATE_FNS.get(name));
  const takes = shipped === undefined ? undefined : TAKES[shipped.name]?.[index];
  const kinds = valueKinds(typ);
  if (takes === undefined || kinds === null) return null;
  const param = index === 0 ? "" : ` for ${TEMPLATE_FNS.get(name)!.params[index]!.name}`;
  const refused = `${name}() takes ${takes.label}${param}, got `;
  if (kinds.size === 0) return takes.omittable ? null : `${refused}null`;
  if (takes.kinds !== null && ![...kinds].some((kind) => takes.kinds!.has(kind))) {
    return refused + canonicalTypeLabel(typ);
  }
  const items = takes.items !== undefined && kinds.size === 1 ? itemOf(typ) : null;
  const itemKinds = items === null ? null : valueKinds(items);
  if (itemKinds !== null && itemKinds.size > 0 && ![...itemKinds].some((kind) => takes.items!.has(kind))) {
    return refused + canonicalTypeLabel(typ);
  }
  return null;
}

/**
 * Return whether `fn` is a builtin shipped here, whose result is a yian value as it is: a float
 * stays a float even when whole, where a host function's `2.0` reads as the int `2`.
 */
export function isShipped(fn: TemplateFn): boolean {
  return SHIPPED.has(fn);
}

/**
 * Return the type a call to builtin `name` gives (`int` for the shipped `len`, `str` for `upper`,
 * ...), `null` for an unknown name, a host function registered under the name, or a result the
 * argument types don't decide.
 *
 * @param args Gives each parameter's argument type, in parameter order (`ArgType`), or `null`
 *   when the arguments don't bind; called only for a result that depends on them. Left out, only
 *   a result that doesn't depend on the arguments is given.
 */
export function builtinResultType(name: string, args: (() => readonly ArgType[] | null) | null = null): Type | null {
  const type = SHIPPED.get(TEMPLATE_FNS.get(name))?.type;
  if (type === undefined) return null;
  if (type instanceof Type) return type;
  const bound = args?.() ?? null;
  return bound === null ? null : type(bound);
}
