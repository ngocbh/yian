/**
 * Operator chains are flat nodes that evaluate and trace like the old binary trees, at any
 * length: a port of `tests/engine/test_expr_chains.py`.
 *
 * Not ported: the repeat tests and the repeats mixed into the traced concat chains, since
 * repetition is a type error in TS. `test_the_grammar_has_no_lalr_conflict` has no counterpart,
 * since the TS parser has no grammar table. The generated chains come from a small seeded PRNG
 * here instead of Python's `random`, and values compare kind-strictly (`1n` is not `1`).
 */

import { expect, test } from "bun:test";

import { ExpressionError, evalExpr, MISSING_ROOT, ResolveMode, type RootResolver } from "../../src/expr/expressions.ts";
import { type ExprNode, parseExpr, Token, Tree } from "../../src/expr/grammar.ts";
import {
  type FeedbackValue,
  type RootRead,
  type RoutedFeedback,
  routeFeedback,
  routeValue,
  SourceRef,
  type TracedValue,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import { validateRoute } from "../../src/runtime/contracts.ts";
import { routeFromPayload, routePayload } from "../../src/runtime/log.ts";
import type { Value, ValueRecord } from "../../src/typesys/index.ts";
import { logJson, parseJson } from "../../src/typesys/json.ts";
import { fieldOf } from "../../src/typesys/values.ts";

/** A position-blind form of `node`, as Lark's `Tree`/`Token` equality ignores positions. */
type Plain = { type: string; value: string } | { data: string; children: Plain[] };

function plain(node: ExprNode): Plain {
  if (node instanceof Token) return { type: node.type, value: node.value };
  return { data: node.data, children: node.children.map(plain) };
}

const ref = (name: string) => new Tree("refcall", [new Token("NAME", name)]);
const num = (text: string) => new Token("NUMBER", text);
const add = (symbol: string) => new Token("ADD_OP", symbol);
const mul = (symbol: string) => new Token("MUL_OP", symbol);
const POW = new Token("POW_OP", "**");

const cases: [string, ExprNode][] = [
  ["a + b - c", new Tree("sum", [ref("a"), add("+"), ref("b"), add("-"), ref("c")])],
  [
    "a * b / c % d",
    new Tree("product", [ref("a"), mul("*"), ref("b"), mul("/"), ref("c"), mul("%"), ref("d")]),
  ],
  ["a ** b ** c", new Tree("power", [ref("a"), POW, ref("b"), POW, ref("c")])],
  ["a + b * c", new Tree("sum", [ref("a"), add("+"), new Tree("product", [ref("b"), mul("*"), ref("c")])])],
  ["-a ** 2", new Tree("neg", [new Tree("power", [ref("a"), POW, num("2")])])],
  [
    "2 ** -a ** b",
    new Tree("power", [num("2"), POW, new Tree("neg", [new Tree("power", [ref("a"), POW, ref("b")])])]),
  ],
  ["a - -b", new Tree("sum", [ref("a"), add("-"), new Tree("neg", [ref("b")])])],
  ["a--b", new Tree("sum", [ref("a"), add("-"), new Tree("neg", [ref("b")])])],
  ["1 -1", new Tree("sum", [num("1"), add("-"), num("1")])],
  ["a*-b", new Tree("product", [ref("a"), mul("*"), new Tree("neg", [ref("b")])])],
  ["(a + b) + c", new Tree("sum", [new Tree("sum", [ref("a"), add("+"), ref("b")]), add("+"), ref("c")])],
  ["a + (b + c)", new Tree("sum", [ref("a"), add("+"), new Tree("sum", [ref("b"), add("+"), ref("c")])])],
  ["a/b", ref("a/b")],
];

for (const [text, expected] of cases) {
  test(`chain shape ${JSON.stringify(text)}`, () => {
    expect(plain(parseExpr(text))).toEqual(plain(expected));
  });
}

// --------------------------------------------------------------------------------------- //
// Flat chains against their fully parenthesized (old binary) forms
// --------------------------------------------------------------------------------------- //

/** Parse `text` then evaluate it, defaulting to an empty resolver (every reference misses). */
function evaluate(
  text: string,
  resolve: RootResolver = MISSING_ROOT,
  mode: ResolveMode = ResolveMode.BINDING_NONE,
): Value {
  return evalExpr(parseExpr(text), resolve, { mode });
}

/** Return a `resolve` backed by a record, reading a field per dotted step; a miss is `null`. */
function fromRecord(mapping: ValueRecord): RootResolver {
  return (names) => {
    const [present, value] = fieldOf(mapping, names[0]!);
    return { used: 1, present, value, why: null };
  };
}

/** Return the message of the `ExpressionError` `evaluate(text, ...)` throws, failing if it returns. */
function messageOf(text: string, mode?: ResolveMode): string {
  try {
    evaluate(text, undefined, mode);
  } catch (error) {
    expect(error).toBeInstanceOf(ExpressionError);
    return (error as Error).message;
  }
  throw new Error(`${text} evaluated without an error`);
}

// A generated expression is a small tree: a leaf with its text and size, a negation, or a chain
// at one level. `size` bounds a value's magnitude or length, so no generated power grows
// without limit.
type Level = "sum" | "product" | "power";
type Node =
  | { kind: "leaf"; text: string; size: number }
  | { kind: "neg"; child: Node }
  | { kind: "chain"; level: Level; operands: Node[]; operators: string[]; grouped: boolean };

const LEVELS: Record<Level, string[]> = { sum: ["+", "-"], product: ["*", "/", "%"], power: ["**"] };
const PRECEDENCE = { sum: 1, product: 2, neg: 3, power: 4, leaf: 5 };
const SIZE_LIMIT = 10_000;

const POOL: ValueRecord = { x: 3n, s: "ab", xs: [1n, 2n] };
const LEAVES: [string, number][] = [
  ["0", 0], ["1", 1], ["2", 2], ["3", 3], ["0.5", 1], ["1.5", 2], ["2.0", 2],
  ["'a'", 1], ['"b"', 1], ["[]", 0], ["[1]", 1], ["['a', 2]", 2],
  ["x", 3], ["s", 2], ["xs", 2], ["m", 0],
];

const leaf = (text: string, size = 1): Node => ({ kind: "leaf", text, size });
const neg = (child: Node): Node => ({ kind: "neg", child });
const chain = (level: Level, operands: Node[], operators: string[], grouped = false): Node => ({
  kind: "chain",
  level,
  operands,
  operators,
  grouped,
});

function precedence(node: Node): number {
  if (node.kind === "chain") return node.grouped ? PRECEDENCE.leaf : PRECEDENCE[node.level];
  return PRECEDENCE[node.kind];
}

/** The expression as an author writes it: one chain per precedence level. */
function flat(node: Node): string {
  if (node.kind === "leaf") return node.text;
  if (node.kind === "neg") {
    const text = flat(node.child);
    return precedence(node.child) < PRECEDENCE.neg ? `-(${text})` : `-${text}`;
  }
  const parts = node.operands.map((operand, index) => {
    const text = flat(operand);
    let needs: boolean;
    if (node.level === "power") {
      const lastNeg = index === node.operands.length - 1 && operand.kind === "neg";
      needs = precedence(operand) < PRECEDENCE.leaf && !lastNeg;
    } else {
      needs = precedence(operand) <= PRECEDENCE[node.level];
    }
    return needs ? `(${text})` : text;
  });
  const text = parts[0] + node.operators.map((op, i) => ` ${op} ${parts[i + 1]}`).join("");
  return node.grouped ? `(${text})` : text;
}

/** The same expression with every operator step in its own parentheses. */
function binary(node: Node): string {
  if (node.kind === "leaf") return node.text;
  if (node.kind === "neg") return `(-${binary(node.child)})`;
  const { level, operands, operators } = node;
  if (level === "power") {
    let text = binary(operands.at(-1)!);
    for (const operand of operands.slice(0, -1).reverse()) text = `(${binary(operand)} ** ${text})`;
    return text;
  }
  let text = binary(operands[0]!);
  operators.forEach((op, i) => {
    text = `(${text} ${op} ${binary(operands[i + 1]!)})`;
  });
  return text;
}

/** An upper bound on the magnitude or length of the node's value. */
function size(node: Node): number {
  if (node.kind === "leaf") return node.size;
  if (node.kind === "neg") return size(node.child);
  const sizes = node.operands.map(size);
  if (node.level === "sum") return sizes.reduce((a, b) => a + b, 0);
  if (node.level === "product") return sizes.reduce((bound, s) => bound * Math.max(s, 2), 1);
  let bound = Math.max(sizes.at(-1)!, 1);
  for (const s of sizes.slice(0, -1).reverse()) {
    if (bound > 32) return Infinity;
    bound = Math.max(s, 2) ** bound;
  }
  return bound;
}

/** A small seeded PRNG with the parts of Python's `random.Random` the generator uses. */
class Rng {
  #state: number;

  constructor(seed: number) {
    this.#state = seed >>> 0;
  }

  /** A float in [0, 1) (mulberry32). */
  random(): number {
    this.#state = (this.#state + 0x6d2b79f5) >>> 0;
    let t = this.#state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  }

  /** An int in [low, high], both included. */
  randint(low: number, high: number): number {
    return low + Math.floor(this.random() * (high - low + 1));
  }

  choice<T>(items: readonly T[]): T {
    return items[Math.floor(this.random() * items.length)]!;
  }
}

function randomLeaf(rng: Rng, leaves: [string, number][] = LEAVES): Node {
  // Mostly numbers, so most chains evaluate; text, lists and the missing `m` make the errors.
  const [text, s] = rng.choice(rng.random() < 0.93 ? [...LEAVES.slice(0, 7), ["x", 3] as [string, number]] : leaves);
  return leaf(text, s);
}

function randomNode(rng: Rng, depth = 0): Node {
  if (depth >= 2 || rng.random() < 0.3) {
    const node = randomLeaf(rng);
    return rng.random() < 0.1 ? neg(node) : node;
  }
  const level = rng.choice(Object.keys(LEVELS) as Level[]);
  let count: number;
  let operands: Node[];
  if (level === "power") {
    count = rng.randint(2, 3);
    operands = Array.from({ length: count }, () => randomLeaf(rng, [...LEAVES.slice(0, 7), ...LEAVES.slice(12)]));
    if (rng.random() < 0.3) operands[count - 1] = neg(operands[count - 1]!);
  } else {
    count = rng.randint(2, 8);
    operands = Array.from({ length: count }, () => randomNode(rng, depth + 1));
  }
  const operators = Array.from({ length: count - 1 }, () => rng.choice(LEVELS[level]));
  const node = chain(level, operands, operators, rng.random() < 0.15);
  return rng.random() < 0.05 ? neg(node) : node;
}

function randomExpressions(count: number): Node[] {
  const rng = new Rng(610);
  const nodes: Node[] = [];
  while (nodes.length < count) {
    const node = randomNode(rng);
    if (node.kind === "chain" && size(node) <= SIZE_LIMIT) nodes.push(node);
  }
  return nodes;
}

const ZERO = chain("sum", [leaf("3"), leaf("3")], ["-"]); // `3 - 3`
const DIVISION_BY_ZERO = [
  chain("product", [leaf("1"), ZERO], ["/"]),
  chain("product", [leaf("x"), leaf("2"), ZERO], ["*", "%"]),
  chain("sum", [leaf("1"), leaf("2"), chain("product", [leaf("x"), ZERO], ["/"])], ["+", "-"]),
];

type Outcome = ["value", Value] | ["error", string];

function outcome(text: string, mode: ResolveMode): Outcome {
  try {
    return ["value", evaluate(text, fromRecord(POOL), mode)];
  } catch (error) {
    if (!(error instanceof ExpressionError)) throw error;
    return ["error", error.message];
  }
}

/** Kind-strict deep equality: `1n` is not `1`, and `-0` is not `0`. */
function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => sameValue(v, b[i]));
  return Object.is(a, b);
}

test("chains match the old binary steps", () => {
  const kinds = new Set<string>();
  const mismatches: string[] = [];
  for (const node of [...randomExpressions(400), ...DIVISION_BY_ZERO]) {
    const [flatText, binaryText] = [flat(node), binary(node)];
    for (const mode of [ResolveMode.BINDING_NONE, ResolveMode.STRICT_RAISE]) {
      const [got, old] = [outcome(flatText, mode), outcome(binaryText, mode)];
      if (got[0] !== old[0] || !sameValue(got[1], old[1])) mismatches.push(`${flatText} | ${binaryText} | ${mode}`);
      kinds.add(got[0]);
    }
  }
  expect(mismatches).toEqual([]);
  expect([...kinds].sort()).toEqual(["error", "value"]);
  expect(outcome(flat(DIVISION_BY_ZERO[0]!), ResolveMode.STRICT_RAISE)).toEqual([
    "error",
    "division by zero: 1 / 0",
  ]);
});

// --------------------------------------------------------------------------------------- //
// Long chains, step errors and error precedence
// --------------------------------------------------------------------------------------- //

test.each([
  ["1", "+", 3000n],
  ["1", "*", 1n],
  ["2", "*", 2n ** 3000n],
  ["-1", "*", 1n],
  ["1", "-", 2n - 3000n],
  ["1", "**", 1n],
  ["'a'", "+", "a".repeat(3000)],
  ["[1]", "+", Array(3000).fill(1n)],
  ["x", "+", 6000n],
] as [string, string, Value][])("a long chain of %s %s evaluates", (term, op, expected) => {
  const text = Array(3000).fill(term).join(` ${op} `);
  expect(evaluate(text, fromRecord({ x: 2n }))).toEqual(expected);
});

test("a long power chain folds from the right", () => {
  // 2 ** (3 ** (1 ** ... ** 1)) is 8; a left fold or a dropped operand would differ.
  const text = "2 ** 3 ** " + Array(2998).fill("1").join(" ** ");
  expect(evaluate(text)).toBe(8n);
});

test.each([
  ["1 + 2 / 0", "division by zero: 2 / 0"],
  ["6 / 3 / 0", "division by zero: 2 / 0"],
  ["1 + 'a' + 2", 'type error evaluating 1 + "a": unsupported operand kinds for `+`: `int` and `str`'],
  ["'a' + 'b' + 1", 'type error evaluating "ab" + 1: unsupported operand kinds for `+`: `str` and `int`'],
])("a step error names the running value: %s", (text, message) => {
  expect(messageOf(text)).toBe(message);
});

test("a float power past the range is Infinity, not an error", () => {
  expect(evaluate("2 ** 3 ** 1000.0")).toBe(Infinity);
});

test.each([
  ["1 / 0 + c", "division by zero: 1 / 0"],
  ["c + 1 / 0", "unresolved reference ${c} in expression"],
  ["c ** (1 / 0)", "unresolved reference ${c} in expression"],
])("error precedence keeps left-to-right order: %s", (text, message) => {
  expect(messageOf(text, ResolveMode.STRICT_RAISE)).toBe(message);
});

// --------------------------------------------------------------------------------------- //
// Traced chains against their old binary forms
// --------------------------------------------------------------------------------------- //

/**
 * A trimmed copy of `expr_lineage.test.ts`'s `Sources` (no `reads`; `lookup` matches on sequence
 * only), kept here because importing a test file would make Bun run its tests again.
 */
class Sources {
  readonly values: ValueRecord;
  readonly identities: Record<string, SourceRef>;

  constructor(values: ValueRecord = {}) {
    this.values = values;
    this.identities = Object.fromEntries(Object.keys(values).map((name, i) => [name, new SourceRef("run", i)]));
  }

  resolve = (names: readonly string[]): RootRead => {
    const head = names[0]!;
    if (!Object.hasOwn(this.values, head)) return { used: 1, present: false, value: null, why: null };
    return { used: 1, present: true, value: this.values[head]!, why: null, source: this.identities[head]! };
  };

  lookup = (source: SourceRef): unknown => {
    const name = Object.keys(this.identities).find((key) => this.identities[key]!.executionSeq === source.executionSeq);
    if (name === undefined || source.path.length > 0) throw new Error(`no source ${source.executionSeq}`);
    return this.values[name];
  };

  evaluate(expression: string): TracedValue {
    return evalExpr(parseExpr(expression), this.resolve, { trace: true });
  }

  feedback(traced: TracedValue, feedback: FeedbackValue): RoutedFeedback {
    return routeFeedback(traced.route, feedback, this.lookup);
  }
}

/** Each contribution as `[source, feedback]`. */
function messages(routed: RoutedFeedback): [SourceRef, FeedbackValue][] {
  return routed.contributions.map((c) => [c.source, c.feedback]);
}

/** The contributions in a canonical order, each as text. */
function contributions(routed: RoutedFeedback): string[] {
  return routed.contributions
    .map((c) => JSON.stringify([c.source.runId, c.source.executionSeq, c.source.path, c.feedback]))
    .sort();
}

// The traced pool: list content, scalars and text. Repetition is a type error in TS, so the
// concat chains have list leaves only.
const TRACED: ValueRecord = { x: 3n, s: "ab", xs: [1n, 2n], ys: ["c"], t: true, q: 3n, n: -2n };
const LIST_LEAVES = ["[x]", "xs", "ys", "[]", "[s, x]", "[1]"];

function randomConcat(rng: Rng): Node {
  const count = rng.randint(2, 8);
  const operands = Array.from({ length: count }, () => leaf(rng.choice(LIST_LEAVES)));
  return chain("sum", operands, Array(count - 1).fill("+"));
}

function tracedExpressions(): Node[] {
  const rng = new Rng(610);
  const nodes = Array.from({ length: 150 }, () => randomConcat(rng));
  return [...nodes, ...randomExpressions(400).filter((node) => !flat(node).includes("m")).slice(0, 100)];
}

function jsonLike(value: Value): boolean {
  if (Array.isArray(value)) return value.every(jsonLike);
  return ["bigint", "number", "string", "boolean"].includes(typeof value);
}

test("traced chains match the old binary steps", () => {
  const sources = new Sources(TRACED);
  const shapes = new Set<string>();
  for (const node of tracedExpressions()) {
    const [flatText, binaryText] = [flat(node), binary(node)];
    let got: TracedValue;
    try {
      got = sources.evaluate(flatText);
    } catch (error) {
      if (!(error instanceof ExpressionError)) throw error;
      let oldMessage: string | null = null;
      try {
        sources.evaluate(binaryText);
      } catch (oldError) {
        expect(oldError).toBeInstanceOf(ExpressionError);
        oldMessage = (oldError as Error).message;
      }
      expect(oldMessage).toBe(error.message);
      continue;
    }
    const old = sources.evaluate(binaryText);
    expect(sameValue(got.value, old.value)).toBe(true);
    if (!jsonLike(got.value)) continue;
    shapes.add(got.route.kind);
    expect(sameValue(routeValue(got.route, sources.lookup), got.value)).toBe(true);
    const feedbacks: FeedbackValue[] = ["Whole."];
    if (Array.isArray(got.value)) feedbacks.push(got.value.map((_, i) => `m${i}`));
    for (const feedback of feedbacks) {
      expect(contributions(sources.feedback(got, feedback))).toEqual(contributions(sources.feedback(old, feedback)));
    }
    if (Array.isArray(got.value) && got.value.length === 0) {
      expect(sources.feedback(got, []).contributions).toEqual([]);
      expect(sources.feedback(old, []).contributions).toEqual([]);
    }
  }
  expect(shapes.has("concat") && shapes.has("scalar")).toBe(true);
});

/** Decode `route`'s payload after a trip through its run-log JSON. */
function roundTrip(route: ValueRoute): ValueRoute {
  return routeFromPayload(parseJson(logJson(routePayload(route)), "codec"));
}

/** The height of a route tree: 0 for a leaf. */
function depth(route: ValueRoute): number {
  return 1 + Math.max(-1, ...route.children.map(depth));
}

test("a list chain traces as balanced two-child concats", () => {
  const sources = new Sources({ left: ["a"], right: ["b", "c"] });
  const result = sources.evaluate("[] + right + []");
  const route = result.route;
  expect(route.kind).toBe("concat");
  expect(route.expression).toBe("([] + right + [])");
  expect(route.sizes).toEqual([0, 2]);
  const [first, rest] = route.children as [ValueRoute, ValueRoute];
  expect(rest.kind).toBe("concat");
  expect(rest.sizes).toEqual([2, 0]);
  expect(rest.expression).toBe("(right + [])");
  expect([first, ...rest.children]).toEqual([
    sources.evaluate("[]").route,
    sources.evaluate("right").route,
    sources.evaluate("[]").route,
  ]);
  const right = sources.identities.right!;
  expect(messages(sources.feedback(result, [null, "Bad."]))).toEqual([[right, [null, "Bad."]]]);
  expect(messages(sources.feedback(result, "Too many."))).toEqual([[right, "Too many."]]);
});

test("a long traced list chain stays shallow", () => {
  const sources = new Sources({ x: "v" });
  const result = sources.evaluate(Array(3000).fill("[x]").join(" + "));
  const route = result.route;
  expect(result.value).toEqual(Array(3000).fill("v"));
  expect(depth(route)).toBeLessThanOrEqual(13);
  expect(routeValue(route, sources.lookup)).toEqual(result.value);
  validateRoute(route);
  expect(roundTrip(route)).toEqual(route);
  const feedback = Array.from({ length: 3000 }, (_, i) => `m${i}`);
  expect(messages(sources.feedback(result, feedback))).toEqual(feedback.map((m) => [sources.identities.x!, m]));
});

test("a scalar chain traces as one scalar", () => {
  let sources = new Sources();
  const route = sources.evaluate("1 + 2 - 3").route;
  expect(route.kind).toBe("scalar");
  expect(route.children).toHaveLength(3);
  expect(route.expression).toBe("(1 + 2 - 3)");
  expect(route.valueJson).toBe("0");
  sources = new Sources({ left: 3n, right: 4n });
  expect(sources.evaluate("left + right").route.expression).toBe("(left + right)");
});

test("an old nested concat route still validates and routes feedback", () => {
  const sources = new Sources({ a: ["a"], b: ["b", "c"], c: ["d"] });
  const source = (name: string) =>
    new ValueRoute("source", { source: sources.identities[name]!, expression: name });
  const inner = new ValueRoute("concat", {
    children: [source("a"), source("b")],
    expression: "(a + b)",
    sizes: [1, 2],
  });
  const old = new ValueRoute("concat", { children: [inner, source("c")], expression: "((a + b) + c)", sizes: [3, 1] });
  validateRoute(old);
  expect(roundTrip(old)).toEqual(old);
  expect(routeValue(old, sources.lookup)).toEqual(["a", "b", "c", "d"]);
  const got = sources.evaluate("a + b + c");
  expect(got.route).not.toEqual(old);
  for (const feedback of ["Whole.", ["1", null, "3", "4"]] as FeedbackValue[]) {
    expect(contributions(routeFeedback(old, feedback, sources.lookup))).toEqual(
      contributions(sources.feedback(got, feedback)),
    );
  }
});
