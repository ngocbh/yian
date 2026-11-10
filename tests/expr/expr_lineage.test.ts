/**
 * A traced evaluation records the values and selected operations it actually evaluated: routes,
 * snapshots, projections and opaque results. A `FileRef` stands in for a temporary that isn't
 * JSON, since a file is the one yian value JSON can't hold.
 */

import { afterEach, expect, test } from "bun:test";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { evalExpr, ExpressionError, MISSING_ROOT, ResolveMode, type RootResolver } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { ResolvedValue, type RootRead, SourceRef, type TracedValue, ValueRoute } from "../../src/expr/lineage.ts";
import { evalBinding, evalTemplate, scanTemplate } from "../../src/expr/template.ts";
import { Grow } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { FileRef, type Value, type ValueRecord, VariablePool } from "../../src/typesys/index.ts";
import { logJson } from "../../src/typesys/json.ts";

const FILE = new FileRef({ uri: "file:///tmp/x" });

/** The sources `route` reads, in order. */
function sourcesOf(route: ValueRoute): SourceRef[] {
  return route.kind === "source" ? [route.source!] : route.children.flatMap(sourcesOf);
}

/** Whether `route` is `part` or holds it among its descendants. */
function holds(route: ValueRoute, part: ValueRoute): boolean {
  return route === part || route.children.some((child) => holds(child, part));
}

/** A source route over `run`/`seq`. */
function sourceRoute(seq: number, expression = ""): ValueRoute {
  return new ValueRoute("source", { source: new SourceRef("run", seq), expression });
}

/** `value` nested in `depth` single-item lists. */
function nested(value: unknown, depth: number): unknown {
  for (let i = 0; i < depth; i++) value = [value];
  return value;
}

/** Return what `run` throws. */
function thrown(run: () => unknown): unknown {
  try {
    run();
  } catch (error) {
    return error;
  }
  throw new Error("expected a throw");
}

test("literal snapshots use the codec form", () => {
  expect(ValueRoute.literal(1n).valueJson).toBe("1");
  expect(ValueRoute.literal(1.0).valueJson).toBe("1.0");
  const infinite = ValueRoute.literal(Infinity);
  expect(infinite.valueJson).toBe('{"$float":"inf"}');
  expect(infinite.expression).toBe('{"$float":"inf"}');
  expect(ValueRoute.literal(-0).valueJson).toBe("0.0");

  const escaped = ValueRoute.literal({ $record: 1n });
  expect(escaped.valueJson).toBe('{"$record":{"$record":1}}');

  const deep = nested("leaf", 10_000);
  const route = ValueRoute.literal(deep);
  expect(route.valueJson).toBe(logJson(deep));
  expect(route.snapshot).toBeNull();
});

test("a file is an unavailable leaf with a partial snapshot", () => {
  const file = ValueRoute.literal(FILE);
  expect(file.kind).toBe("literal");
  expect(file.valueJson).toBeNull();
  expect(file.resultType).toBe("file");
  expect(file.expression).toBe("<file>");
  expect(file.snapshot).toBeNull();

  const list = ValueRoute.literal(["kept", FILE]);
  expect(list.valueJson).toBeNull();
  expect(list.resultType).toBe("list");
  expect(list.expression).toBe("<list>");
  expect(list.snapshot!.kind).toBe("list");
  expect(list.snapshot!.children).toEqual([
    new ValueRoute("literal", { valueJson: '"kept"', expression: '"kept"' }),
    new ValueRoute("literal", { resultType: "file", expression: "<file>" }),
  ]);

  const record = ValueRoute.literal({ kept: 1n, file: FILE });
  expect(record.resultType).toBe("record");
  expect(record.snapshot!.kind).toBe("record");
  expect(record.snapshot!.names).toEqual(["kept", "file"]);
  expect(record.snapshot!.children.map((c) => c.valueJson)).toEqual(["1", null]);
  expect(record.snapshot!.children[1]!.resultType).toBe("file");

  let current = ValueRoute.literal(nested(FILE, 40));
  let levels = 0;
  while (current.snapshot !== null) {
    current = current.snapshot.children[0]!;
    levels++;
  }
  expect(levels).toBe(32);
  expect(current.resultType).toBe("list");
  expect(current.valueJson).toBeNull();
});

test("the depth bound applies inside a partial snapshot", () => {
  const bounded = ValueRoute.literal([FILE, nested("leaf", 32)]).snapshot!.children[1]!;
  expect(bounded.valueJson).toBeNull();
  expect(bounded.resultType).toBe("list");
  let current = bounded;
  let levels = 0;
  while (current.snapshot !== null) {
    current = current.snapshot.children[0]!;
    levels++;
  }
  expect(levels).toBe(31);
  expect(current.valueJson).toBeNull();

  const kept = ValueRoute.literal([FILE, nested("leaf", 31)]).snapshot!.children[1]!;
  expect(kept.valueJson).toBe(logJson(nested("leaf", 31)));
  expect(kept.snapshot).toBeNull();
});

test("routes are immutable and serializable", () => {
  const route = ValueRoute.record([
    ["tags", ValueRoute.literal(["original", "other"])],
    ["file", ValueRoute.literal(FILE)],
  ]);
  const encoding = JSON.stringify(route);
  expect(() => {
    // @ts-expect-error: a route's fields are read-only.
    route.kind = "literal";
  }).toThrow(TypeError);
  expect(() => {
    // @ts-expect-error: a route's children are read-only.
    route.children.push(ValueRoute.literal(1n));
  }).toThrow(TypeError);
  expect(JSON.stringify(route)).toBe(encoding);
  expect(JSON.parse(encoding).names).toEqual(["tags", "file"]);
  const spans: [number, number][] = [[0, 1]];
  const interpolation = new ValueRoute("interpolation", { valueJson: '"a"', spans, children: [route] });
  spans[0]![1] = 9;
  expect(interpolation.spans).toEqual([[0, 1]]);
  expect(Object.isFrozen(interpolation.spans[0])).toBe(true);
});

test("record and project", () => {
  const x = sourceRoute(1, "x");
  const y = sourceRoute(2, "y");
  const record = ValueRoute.record([
    ["a", x],
    ["b", y],
  ]);
  expect(record.kind).toBe("record");
  expect(record.expression).toBe("{a: x, b: y}");
  expect(record.names).toEqual(["a", "b"]);
  expect(record.children).toEqual([x, y]);

  const tag = x.project(["tags", 1]);
  expect(tag.kind).toBe("source");
  expect(tag.source).toEqual(new SourceRef("run", 1, ["tags", 1]));
  expect(tag.expression).toBe("x.tags[1]");

  const field = record.project(["a"]);
  expect(field.kind).toBe("project");
  expect(field.children[0]).toBe(record);
  expect(field.path).toEqual(["a"]);
  expect(field.expression).toBe("{a: x, b: y}.a");
  expect(x.project([])).toBe(x);
});

// --------------------------------------------------------------------------------------- //
// Traced evaluation
// --------------------------------------------------------------------------------------- //

/** Single-read references backed by distinct accepted execution identities. */
class Sources {
  readonly values: ValueRecord;
  readonly identities: Record<string, SourceRef>;
  readonly reads: string[] = [];

  constructor(values: ValueRecord = {}) {
    this.values = values;
    this.identities = Object.fromEntries(Object.keys(values).map((name, i) => [name, new SourceRef("run", i)]));
  }

  resolve = (names: readonly string[]): RootRead => {
    this.reads.push(names.join("."));
    const head = names[0]!;
    if (!Object.hasOwn(this.values, head)) return { used: 1, present: false, value: null, why: null };
    return { used: 1, present: true, value: this.values[head]!, why: null, source: this.identities[head]! };
  };

  evaluate(expression: string): TracedValue {
    return evalExpr(parseExpr(expression), this.resolve, { trace: true });
  }
}

/** A traced resolver that resolves nothing. */
const nothing = MISSING_ROOT;

// Each registration's previous entry (or its absence), put back after the test.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in builtin `name` running `call` for this test only. */
function standIn(name: string, call: (...args: Value[]) => unknown, params: TemplateParam[]): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call });
}

/** Register a no-argument builtin `name` returning `result`; return its call log. */
function countedStandIn(name: string, result: unknown): string[] {
  const calls: string[] = [];
  standIn(
    name,
    () => {
      calls.push(name);
      return result;
    },
    [],
  );
  return calls;
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

test("an infinite temporary keeps a valid final interpolation", () => {
  const calls = countedStandIn("score", Infinity);
  const traced = evalBinding("score: ${score()}", nothing, { trace: true });
  expect(traced.value).toBe("score: Infinity");
  expect(traced.route.valueJson).toBe('"score: Infinity"');
  const opaque = traced.route.children[0]!;
  expect(opaque.kind).toBe("opaque");
  expect(opaque.valueJson).toBe('{"$float":"inf"}');
  expect(opaque.resultType).toBeNull();
  expect(calls).toEqual(["score"]);
});

test("reserved MAP over uses the same capture path", async () => {
  const sources = new Sources({ items: [] });
  const node = new MapNode("map", { flowId: "child", child: {} });
  const evaluated = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { over: "${items}" },
    captureBindings: true,
    sourceResolver: sources.resolve,
  });
  expect(evaluated.bindings!.boundInputs).toEqual({ over: [] });
  expect(evaluated.bindings!.evaluatedKeys).toEqual(new Set(["over"]));
  expect(sources.reads).toEqual(["items"]);
  expect(sourcesOf(evaluated.bindings!.route)).toEqual([sources.identities.items!]);
});

test("equal MAP items get distinct recorded indices", async () => {
  const sources = new Sources({ left: ["same", null], right: ["same", null] });
  const node = new MapNode("map", { flowId: "child", child: {} });
  node.params = [new Parameter("x")];
  const evaluated = await evalNode(node, null, new VariablePool(), {
    wiringOverride: { over: "${left + right + right}", x: "${item}" },
    captureBindings: true,
    sourceResolver: sources.resolve,
  });
  const seed = (evaluated.outcome as Grow).seed as ValueRecord[];
  expect(seed).toEqual([{ x: "same" }, { x: null }, { x: "same" }, { x: null }, { x: "same" }, { x: null }]);
  expect(evaluated.bindings!.items.map(([index]) => index)).toEqual([0, 1, 2, 3, 4, 5]);
  expect(sources.reads).toEqual(["left", "right", "right"]);
  for (const [index, item] of evaluated.bindings!.items) {
    expect(item.boundInputs).toEqual(seed[index]!);
    expect(item.evaluatedKeys).toEqual(new Set(["x"]));
  }
});

test("a direct projection reads its source's path", () => {
  const original = { tags: ["one", "two"], summary: "text" };
  const root = new SourceRef("run", 7);
  const reads: string[] = [];
  const resolve: RootResolver = (names) => {
    const path = names.join(".");
    reads.push(path);
    if (path === "producer.output.tags") {
      return { used: 3, present: true, value: original.tags, why: null, source: new SourceRef("run", 7, ["tags"]) };
    }
    return { used: 1, present: true, value: "two", why: null, source: new SourceRef("run", 7, ["tags", 1]) };
  };

  const result = evalExpr(parseExpr("producer.output.tags"), resolve, { trace: true });
  expect(result.value).toBe(original.tags);
  expect(reads).toEqual(["producer.output.tags"]);
  expect(sourcesOf(result.route)).toEqual([new SourceRef(root.runId, root.executionSeq, ["tags"])]);
  const projected = evalExpr(parseExpr("tag"), resolve, { trace: true });
  expect(sourcesOf(projected.route)).toEqual([new SourceRef(root.runId, root.executionSeq, ["tags", 1])]);
});

test("accepted null has a source but absence does not", () => {
  const sources = new Sources({ accepted: null });
  const accepted = sources.identities.accepted!;
  const present = evalBinding("${accepted}", sources.resolve, { trace: true });
  const absent = evalTemplate(scanTemplate("${absent}"), sources.resolve, { trace: true });
  expect(present.value).toBeNull();
  expect(absent.value).toBeNull();
  expect(sourcesOf(present.route)).toEqual([accepted]);
  expect(sourcesOf(absent.route)).toEqual([]);
  // A strict read of an absent root raises; an accepted null is a value (a strict template refuses it).
  const strict = (name: string) =>
    evalExpr(parseExpr(name), sources.resolve, { mode: ResolveMode.STRICT_RAISE, trace: true });
  expect(strict("accepted").value).toBeNull();
  const error = thrown(() => strict("absent"));
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toMatch(/unresolved/);
});

test.each([0n, false, "", ["item"]] as Value[])("coalesce and default drop losing arm routes (%p)", (winning) => {
  const sources = new Sources({ empty: null, winner: winning, losing: "unused" });
  for (const [expression, reads] of [
    ["empty | winner | losing", ["empty", "winner"]],
    ["empty :- winner", ["empty", "winner"]],
    ["winner :- losing", ["winner"]],
    ['winner :? "Required"', ["winner"]],
  ] as [string, string[]][]) {
    sources.reads.length = 0;
    const result = sources.evaluate(expression);
    expect(result.value).toEqual(winning);
    expect(sources.reads).toEqual(reads);
    expect(sourcesOf(result.route)).toEqual([sources.identities.winner!]);
  }
  for (const expression of ['empty :- "literal"', "empty | absent | null"]) {
    expect(sourcesOf(sources.evaluate(expression).route)).toEqual([]);
  }
  expect(sourcesOf(sources.evaluate("absent :- empty").route)).toEqual([sources.identities.empty!]);
});

test("list constructor preserves nested routes", () => {
  const sources = new Sources({ producer: { tags: ["one", "two"], summary: "long" } });
  const producer = sources.identities.producer!;
  const result = sources.evaluate("[producer.tags, [producer.summary], producer.tags]");
  expect(result.value).toEqual([["one", "two"], ["long"], ["one", "two"]]);
  expect(result.route.kind).toBe("list");
  const at = (...path: (string | number)[]) => new SourceRef(producer.runId, producer.executionSeq, path);
  expect(sourcesOf(result.route)).toEqual([at("tags"), at("summary"), at("tags")]);
});

test("concat records its operand lengths", () => {
  const sources = new Sources({ left: ["a"], right: ["b", "c"] });
  const { left, right } = sources.identities as { left: SourceRef; right: SourceRef };
  const result = sources.evaluate("left + right");
  expect(result.value).toEqual(["a", "b", "c"]);
  expect(result.route.kind).toBe("concat");
  expect(result.route.sizes).toEqual([1, 2]);
  expect(sourcesOf(result.route)).toEqual([left, right]);
  const repeated = sources.evaluate("left + left");
  expect(repeated.route.sizes).toEqual([1, 1]);
  expect(sourcesOf(repeated.route)).toEqual([left, left]);
});

test.each([
  ["x + y", 7n, ["x", "y"]],
  ["x - y", -1n, ["x", "y"]],
  ["x * y", 12n, ["x", "y"]],
  ["x / y", 0.75, ["x", "y"]],
  ["x % y", 3n, ["x", "y"]],
  ["x ** y", 81n, ["x", "y"]],
  ["-x", -3n, ["x"]],
  ["x < y", true, ["x", "y"]],
  ["x == y", false, ["x", "y"]],
  ["x in items", true, ["x", "items"]],
  ["x not in items", false, ["x", "items"]],
  ["not x", false, ["x"]],
  ["x or absent", true, ["x"]],
  ["x and y", true, ["x", "y"]],
  ["zero and absent", false, ["zero"]],
] as [string, Value, string[]][])(
  "scalar and interpolation text stays unchanged: %s",
  (expression, expected, names) => {
    const sources = new Sources({ x: 3n, y: 4n, items: [3n, 4n], zero: 0n });
    const result = sources.evaluate(expression);
    expect(result.value).toEqual(expected);
    expect(sources.reads).toEqual(names);
    expect(result.route.expression).toBeTruthy();
    expect(sourcesOf(result.route)).toEqual(names.map((name) => sources.identities[name]!));
    const rendered = evalBinding("x=${x}; items=${items}", sources.resolve, { trace: true });
    expect(rendered.value).toBe("x=3; items=[3,4]");
    expect(rendered.route.valueJson).toBe('"x=3; items=[3,4]"');
    expect(sourcesOf(rendered.route)).toEqual([sources.identities.x!, sources.identities.items!]);
    expect(rendered.route.spans).toEqual([
      [2, 3],
      [11, 16],
    ]);
  },
);

test("spans count code points", () => {
  const sources = new Sources({ x: 3n, y: 4n });
  const rendered = evalBinding("😀${x}é${y}", sources.resolve, { trace: true });
  expect(rendered.value).toBe("😀3é4");
  expect(rendered.route.kind).toBe("interpolation");
  expect(rendered.route.expression).toBe("😀${x}é${y}");
  expect(rendered.route.spans).toEqual([
    [1, 2],
    [3, 4],
  ]);
});

test.each([
  ["reorder", (values: Value[]) => [values[1], values[0]], "", ["b", "a"], ["Wrong.", null]],
  [
    "record",
    (values: Value[]) => ({ second: values[1], first: values[0] }),
    "",
    { second: "b", first: "a" },
    { second: "Wrong." },
  ],
  ["count", (values: Value[]) => values.length, "", 2n, "Too many."],
  ["upper", (values: Value[]) => [values[1], values[0]], "", ["b", "a"], [null, "Wrong."]],
  ["projected", (values: Value[]) => ({ result: [values[1], values[0]] }), ".result", ["b", "a"], ["Wrong.", null]],
] as [string, (values: Value[]) => unknown, string, Value, unknown][])(
  "opaque functions record their result and execute once: %s",
  (name, fn, suffix, expected) => {
    const calls: Value[] = [];
    standIn(
      name,
      (values) => {
        calls.push(values);
        return fn(values as Value[]);
      },
      [{ name: "values" }],
    );
    const sources = new Sources({ items: ["a", "b"] });
    const result = sources.evaluate(`${name}(items)${suffix}`);
    expect(result.value).toEqual(expected);
    expect(result.route.expression).toBe(`${name}(items)${suffix}`);
    expect(sourcesOf(result.route)).toEqual([sources.identities.items!]);
    expect(sources.reads).toEqual(["items"]);
    expect(calls).toEqual([["a", "b"]]);
  },
);

test("composed item route projects fields and preserves null presence", () => {
  const sources = new Sources({ left: [{ answer: "old" }], right: [{ answer: null }] });
  const over = sources.evaluate("left + right");
  const item = new ResolvedValue((over.value as Value[])[1]!, true, over.route.project([1]));
  sources.reads.length = 0;
  const result = evalBinding("${item.answer}", sources.resolve, { item, trace: true });
  expect(result.value).toBeNull();
  expect(sources.reads).toEqual([]);
  // The read projects `.answer` of the item's own route, item 1 of the concat.
  expect(result.route.kind).toBe("project");
  expect(result.route.path).toEqual(["answer"]);
  expect(holds(result.route, item.source as ValueRoute)).toBe(true);
  const missing = evalBinding("${item?.absent}", sources.resolve, { item, trace: true });
  expect(sourcesOf(missing.route)).toEqual([]);
  const fallback = evalBinding('${item.answer :- "default"}', sources.resolve, { item, trace: true });
  expect(fallback.value).toBe("default");
  expect(sourcesOf(fallback.route)).toEqual([]);
  const localReads: string[] = [];
  const projection = over.route.project([1, "answer"]);
  const local: RootResolver = (names) => {
    localReads.push(names.join("."));
    return { used: 1, present: true, value: null, why: null, source: projection };
  };
  const localResult = evalBinding("${local}", local, { trace: true });
  expect(localReads).toEqual(["local"]);
  expect(holds(localResult.route, projection)).toBe(true);
});

test("a file temporary preserves interpolation and projection", () => {
  const calls = countedStandIn("temporary", FILE);
  const untraced = evalBinding("value=${temporary()}", MISSING_ROOT);
  expect(untraced).toBe('value={"uri":"file:///tmp/x","mime":null,"name":null}');
  expect(calls).toEqual(["temporary"]);
  const traced = evalBinding("value=${temporary()}", nothing, { trace: true });
  expect(traced.value).toBe(untraced);
  const opaque = traced.route.children[0]!;
  expect(opaque.kind).toBe("opaque");
  expect(opaque.valueJson).toBeNull();
  expect(opaque.resultType).toBe("file");
  expect(traced.route.valueJson).toBe(JSON.stringify(untraced));
  expect(calls).toEqual(["temporary", "temporary"]);

  standIn("record", () => ({ answer: "ok", temporary: FILE }), []);
  const projected = evalBinding("${record().answer}", nothing, { trace: true });
  expect(projected.value).toBe("ok");
  // A field of an opaque result is part of the opaque operation.
  expect(projected.route.kind).toBe("opaque");
  expect(projected.route.valueJson).toBe('"ok"');
});

test("routes keep opaque snapshots", () => {
  const produced = { tags: ["original", "other"] };
  standIn("snapshot", () => produced, []);
  const result = evalBinding("${snapshot()}", nothing, { trace: true });
  const encoding = JSON.stringify(result.route);
  produced.tags.push("later");
  expect(result.route.valueJson).toBe('{"tags":["original","other"]}');
  expect(JSON.stringify(result.route)).toBe(encoding);
  expect(Object.isFrozen(result.route)).toBe(true);
});

test("an opaque call keeps its concat and interpolation context", () => {
  const calls: string[] = [];
  standIn(
    "reorder",
    (values) => {
      calls.push("call");
      return [...(values as Value[])].reverse();
    },
    [{ name: "values" }],
  );
  const sources = new Sources({ items: ["one", "two"], count: 2n });
  const concatenated = sources.evaluate("reorder(items) + items");
  expect(concatenated.route.kind).toBe("concat");
  expect(concatenated.route.sizes).toEqual([2, 2]);
  expect(concatenated.route.children.map((child) => child.kind)).toEqual(["opaque", "source"]);
  expect(calls).toEqual(["call"]);
  const shortCircuit = sources.evaluate("count or reorder(items)");
  expect(sourcesOf(shortCircuit.route)).toEqual([sources.identities.count!]);
  expect(calls).toEqual(["call"]);
  const interpolated = evalBinding("items=${reorder(items)}", sources.resolve, { trace: true });
  expect(interpolated.route.spans).toEqual([[6, 19]]);
  expect(interpolated.route.valueJson).toBe(JSON.stringify('items=["two","one"]'));
  expect(calls).toEqual(["call", "call"]);
});

test("a deep temporary keeps a complete snapshot", () => {
  const calls: string[] = [];
  standIn(
    "nested",
    () => {
      calls.push("nested");
      return nested("leaf", 10_000);
    },
    [],
  );
  standIn(
    "ignore",
    () => {
      calls.push("ignore");
      return "ok";
    },
    [{ name: "value" }],
  );
  expect(evalBinding("${ignore(nested())}", MISSING_ROOT)).toBe("ok");
  const traced = evalBinding("${ignore(nested())}", nothing, { trace: true });
  expect(traced.value).toBe("ok");
  expect(traced.route.children[0]!.valueJson).not.toBeNull();
  expect(calls).toEqual(["nested", "ignore", "nested", "ignore"]);
});

test("unavailable snapshots keep container structure", () => {
  const calls = countedStandIn("temporary_list", [FILE]);
  const expected = 'items=[{"uri":"file:///tmp/x","mime":null,"name":null}]';
  const result = evalBinding("items=${temporary_list() + []}", nothing, { trace: true });
  expect(result.value).toBe(expected);
  const concat = result.route.children[0]!;
  expect(concat.kind).toBe("concat");
  expect(concat.valueJson).toBeNull();
  const opaque = concat.children[0]!;
  expect(opaque.resultType).toBe("list");
  expect(opaque.snapshot!.kind).toBe("list");
  expect(opaque.snapshot!.children[0]!.resultType).toBe("file");
  expect(calls).toEqual(["temporary_list"]);
});

test("unavailable snapshot projects available fields and composed items", () => {
  const itemValue = { answer: "ok", temporary: FILE };
  const plain = evalBinding("${item.answer}", nothing, { item: itemValue, trace: true });
  expect(plain.value).toBe("ok");
  expect(sourcesOf(plain.route)).toEqual([]);

  const calls = countedStandIn("temporary_list", [itemValue]);
  const over = evalBinding("${temporary_list() + []}", nothing, { trace: true });
  const item = new ResolvedValue((over.value as Value[])[0]!, true, over.route.project([0]));
  const projected = evalBinding("${item.answer}", nothing, { item, trace: true });
  expect(projected.value).toBe("ok");
  expect(projected.route.kind).toBe("project");
  expect(projected.route.path).toEqual(["answer"]);
  expect(holds(projected.route, item.source as ValueRoute)).toBe(true);
  expect(calls).toEqual(["temporary_list"]);
});

test("a deep temporary composes into a concat", () => {
  const calls = countedStandIn("deep", nested(1n, 10_000));
  expect(Array.isArray(evalBinding("${deep() + []}", MISSING_ROOT))).toBe(true);
  const traced = evalBinding("${deep() + []}", nothing, { trace: true });
  expect(Array.isArray(traced.value)).toBe(true);
  expect(traced.route.kind).toBe("concat");
  expect(JSON.stringify(traced.route)).toBeTruthy();
  expect(calls).toEqual(["deep", "deep"]);
});

test("a plain item has no route to follow", () => {
  const item = { answer: "ok" };
  expect(evalBinding("${item.answer}", MISSING_ROOT, { item })).toBe("ok");
  const traced = evalBinding("${item.answer}", nothing, { item, trace: true });
  expect(traced.value).toBe("ok");
  expect(sourcesOf(traced.route)).toEqual([]);
});

test("a sourced item read projects its source", () => {
  const item = new ResolvedValue({ answer: "ok" }, true, new SourceRef("run", 7, ["items", 0]));
  const traced = evalBinding("${item.answer}", nothing, { item, trace: true });
  expect(traced.value).toBe("ok");
  expect(traced.route.source).toEqual(new SourceRef("run", 7, ["items", 0, "answer"]));
});

test.each([
  ["list", 33],
  ["record", 40],
] as ["list" | "record", number][])(
  "JSON-compatible nested results keep a complete snapshot: %s",
  (container, depth) => {
    let value: unknown = "leaf";
    for (let i = 0; i < depth; i++) value = container === "list" ? [value] : { child: value };
    const calls = countedStandIn("nested", value);
    const traced = evalBinding("${nested()}", nothing, { trace: true });
    expect(traced.value).toEqual(value as Value);
    expect(traced.route.kind).toBe("opaque");
    expect(traced.route.valueJson).toBe(logJson(value as Value));
    expect(traced.route.resultType).toBeNull();
    expect(calls).toEqual(["nested"]);
  },
);

test("a slice route projects each kept item of its base", () => {
  const sources = new Sources({ producer: { tags: ["a", "b", "c"] } });
  const producer = sources.identities.producer!;
  const result = sources.evaluate("producer.tags[1:]");
  expect(result.value).toEqual(["b", "c"]);
  expect(result.route.kind).toBe("list");
  expect(result.route.children.map((child) => [child.kind, child.source?.path])).toEqual([
    ["source", ["tags", 1]],
    ["source", ["tags", 2]],
  ]);
  expect(sourcesOf(result.route).every((source) => source.executionSeq === producer.executionSeq)).toBe(true);
});

test("an index route projects its item of the base, and a text slice is a scalar over its base", () => {
  const sources = new Sources({ producer: { tags: ["a", "b", "c"], name: "Ada" } });
  const item = sources.evaluate("producer.tags[-1]");
  expect(item.value).toBe("c");
  expect([item.route.kind, item.route.source?.path]).toEqual(["source", ["tags", 2]]);
  const text = sources.evaluate("producer.name[1:]");
  expect(text.value).toBe("da");
  expect(text.route.kind).toBe("scalar");
  expect(text.route.expression).toBe("producer.name[...]");
  expect(text.route.children.map((child) => child.source?.path)).toEqual([["name"]]);
});

test("a conditional's route selects the branch it picked, not its test", () => {
  const sources = new Sources({ producer: { tags: ["a"], name: "Ada", flag: true } });
  const result = sources.evaluate("producer.name if producer.flag else producer.tags");
  expect(result.value).toBe("Ada");
  expect(result.route.kind).toBe("selected");
  expect(result.route.children.map((child) => [child.kind, child.source?.path])).toEqual([["source", ["name"]]]);
});

test("len records an opaque route", () => {
  const sources = new Sources({ producer: { tags: ["a"] } });
  expect(sources.evaluate("len(producer.tags)").route.kind).toBe("opaque");
});

test("a comprehension route carries each item's own route", () => {
  const sources = new Sources({ producer: { history: [[{ text: "a" }], [{ text: "b" }, { text: "c" }]] } });
  const producer = sources.identities.producer!;
  const result = sources.evaluate("[m.text for t in producer.history for m in t]");
  expect(result.value).toEqual(["a", "b", "c"]);
  expect(result.route.kind).toBe("list");
  expect(result.route.children.map((child) => [child.kind, child.source?.path])).toEqual([
    ["source", ["history", 0, 0, "text"]],
    ["source", ["history", 1, 0, "text"]],
    ["source", ["history", 1, 1, "text"]],
  ]);
  // The same items as a hand-built route over the base's projections.
  const base = sources.evaluate("producer.history");
  const items = [
    [0, 0],
    [1, 0],
    [1, 1],
  ].map(([i, j]) => base.route.project([i!, j!, "text"]));
  expect(sourcesOf(new ValueRoute("list", { children: items }))).toEqual(sourcesOf(result.route));
  expect(sourcesOf(result.route).every((source) => source.executionSeq === producer.executionSeq)).toBe(true);
});

test("a comprehension's conditions are not part of its route", () => {
  const sources = new Sources({ producer: { tags: ["a", "b", "c"] } });
  const result = sources.evaluate('[x for x in producer.tags if x != "b"]');
  expect(result.value).toEqual(["a", "c"]);
  expect(result.route.kind).toBe("list");
  expect(result.route.children.map((child) => [child.kind, child.source?.path])).toEqual([
    ["source", ["tags", 0]],
    ["source", ["tags", 2]],
  ]);
});

test("a traced ?. read keeps ?. in its route expression", () => {
  const resolve: RootResolver = (names) =>
    names[0] === "a"
      ? { used: 2, present: true, value: { k: "v" }, why: null, source: new SourceRef("r", 1) }
      : { used: 1, present: false, value: null, why: null };
  expect(evalExpr(parseExpr("a.output?.k"), resolve, { trace: true }).route.expression).toBe("a.output?.k");
  expect(evalExpr(parseExpr("upper(a.output?.k)"), resolve, { trace: true }).route.expression).toBe(
    "upper(a.output?.k)",
  );
});
