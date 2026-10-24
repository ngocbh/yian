/**
 * Feedback follows the values and selected operations actually evaluated: a port of
 * `tests/engine/test_expr_lineage.py`. A `FileRef` stands in for Python's non-JSON temporaries
 * (`object()`), since a file is the one yian value JSON can't hold.
 *
 * TS-only: the codec form of literal snapshots, a file as an unavailable leaf, own-key record
 * fields, the depth bound inside a partial snapshot, the focused `validateFeedback` and
 * `hasFeedback` checks, a scalar route lifting a projected selection, a traced resolver that
 * doesn't return `ResolvedValue`, interpolation spans counting code points (also in
 * `validateRoute`), frozen routed feedback, `sourceKey`, `sameSource` (separately built sources with
 * the same key), and `sameFeedback` (true for separately built equal values, false for each kind of
 * difference), a slice's `list` route of projected base items, `len`'s `opaque` route, and a
 * comprehension's `list` route of each item's own route. Python's
 * `test_routes_are_immutable_serializable_and_keep_opaque_snapshots` is split in two: `routes are
 * immutable and serializable` (hand-built) and `routes keep opaque snapshots` (through a traced
 * binding).
 *
 * Changed: an infinite temporary has a codec snapshot (`{"$float":"inf"}`), where Python's JSON
 * has none; `a deep temporary keeps a complete snapshot` (10,000 levels) replaces
 * `test_unrepresentable_temporary_does_not_prevent_the_outer_call`, since a snapshot here has no
 * recursion limit; builtin results are copied by host intake, so Python's `is` checks are
 * `toEqual`; `opaque stops preserve concat and interpolation context` evaluates
 * `reorder(items) + items` for Python's `reorder(items) * count`, so it has one stop instead of
 * two; `unavailable snapshot projects available fields and composed items` composes with `+ []`
 * and projects item 0, for Python's `* 2` and item 1; `accepted projected value stops at a
 * bounded ancestor` and `projected partial container uses recorded operation positions` build
 * their opaque route by hand (`ValueRoute.literal(x).with({kind: "opaque"})`, the route the
 * evaluator records for a builtin call), so they don't check Python's single builtin call.
 * `test_equal_map_items_get_distinct_recorded_indices` maps over `${left + right + right}` for
 * Python's `${left + right * 2}`, since list repetition is a type error in TS: the same six
 * items, with items 2-5 tracing to `right`, but `right` is read twice.
 *
 * Ported: `test_reserved_map_over_uses_the_same_capture_path`.
 *
 * Not ported:
 * - `test_wait_capture_retains_the_delivered_iso_value_after_normalization`: TS yian has no WAIT
 *   kind; the `sleep` built-in reads `until` as an ordinary input.
 * - Temporaries `fromHost` refuses, so a builtin can't return them: the NaN row of
 *   `test_nonfinite_temporary_keeps_a_valid_final_interpolation`, the lone-surrogate
 *   `test_noncanonical_unicode_temporary_preserves_comparison_and_partial_snapshot`,
 *   `test_cyclic_temporary_keeps_outer_shape_for_composition`, the `__repr__`/`__str__` render
 *   counts of the temporary tests, `test_capture_does_not_call_dictionary_subclass_hooks`,
 *   `test_repetition_capture_does_not_call_integer_conversion` and the dict-subclass hooks of
 *   the two item-tracing tests.
 * - Repetition is a type error in TS, so `test_repetition_preserves_copy_messages_and_shape_context`,
 *   `test_empty_repetition_does_not_infer_its_original_content_from_the_result` and the other
 *   `repeat` / `* 2` rows are gone.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { isDeepStrictEqual } from "node:util";

import { registerTemplateFn, TEMPLATE_FNS, type TemplateFn, type TemplateParam } from "../../src/expr/builtins.ts";
import { evalExpr, ExpressionError, ResolveMode } from "../../src/expr/expressions.ts";
import { parseExpr } from "../../src/expr/grammar.ts";
import { hasFeedback, validateFeedback } from "../../src/expr/index.ts";
import {
  type FeedbackValue,
  ResolvedValue,
  type RouteKind,
  type RoutedFeedback,
  RouteContext,
  routeFeedback,
  routeValue,
  SourceRef,
  sameFeedback,
  sameSource,
  sourceKey,
  type TracedValue,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import { evalBinding, evalTemplate, scanTemplate } from "../../src/expr/template.ts";
import { Grow } from "../../src/nodes/base.ts";
import { Parameter } from "../../src/nodes/contracts.ts";
import { MapNode } from "../../src/nodes/map/node.ts";
import { validateRoute } from "../../src/runtime/contracts.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { FileRef, isRecord, type Value, type ValueRecord, VariablePool } from "../../src/typesys/index.ts";
import { logJson } from "../../src/typesys/json.ts";

const FILE = new FileRef({ uri: "file:///tmp/x" });

/** Each contribution as `[source, feedback]`. */
function messages(routed: RoutedFeedback): [SourceRef, FeedbackValue][] {
  return routed.contributions.map((c) => [c.source, c.feedback]);
}

/** A lookup giving each source's value, matching a `SourceRef` by its run and sequence. */
function lookupFrom(entries: [SourceRef, unknown][]): (source: SourceRef) => unknown {
  return (source) => {
    const found = entries.find(([ref]) => ref.runId === source.runId && ref.executionSeq === source.executionSeq);
    if (found === undefined) throw new Error(`no source ${source.runId}/${source.executionSeq}`);
    return found[1];
  };
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

test("null feedback does not read source snapshots", () => {
  const reads: SourceRef[] = [];
  const route = new ValueRoute("source", { source: new SourceRef("run", 1) });
  const routed = routeFeedback(route, null, (source) => {
    reads.push(source);
    throw new Error("None feedback needs no value snapshot");
  });
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped).toEqual([]);
  expect(reads).toEqual([]);
});

test("accepted shape keeps full validation and exact source failures", () => {
  const source = new SourceRef("run", 1);
  const route = new ValueRoute("list", {
    children: [new ValueRoute("source", { source }), ValueRoute.literal(FILE)],
  });
  const valid = [1n, { uri: "file:///tmp/x" }];
  const reads: SourceRef[] = [];
  const lookup = (identity: SourceRef) => {
    reads.push(identity);
    return 1n;
  };
  for (const invalid of [[null], [null, { unknown: null }], [null, []]]) {
    expect(() => routeFeedback(route, invalid, lookup, { recordedValue: valid })).toThrow(RangeError);
  }
  expect(reads).toEqual([]);
  const routed = routeFeedback(route, ["Source.", { uri: "Literal." }], lookup, { recordedValue: valid });
  expect(messages(routed)).toEqual([[source, "Source."]]);

  const failure = new Error("accepted source lookup broke");
  const broken = () => {
    throw failure;
  };
  expect(thrown(() => routeFeedback(route, ["Source.", null], broken, { recordedValue: valid }))).toBe(failure);
  const absent = new ValueRoute("source", { source: new SourceRef("run", 1, ["absent"]) });
  const missing = thrown(() => routeFeedback(absent, "Review.", () => ({}), { recordedValue: "present" }));
  expect(missing).toBeInstanceOf(RangeError);
  expect((missing as Error).message).toBe("record has no field 'absent'");
  const malformed = new ValueRoute("malformed" as RouteKind);
  const unknown = thrown(() => routeFeedback(malformed, "Review.", lookup, { recordedValue: "present" }));
  expect(unknown).toBeInstanceOf(RangeError);
  expect((unknown as Error).message).toBe("unknown route kind 'malformed'");
});

test.each([Error, TypeError, RangeError])("unavailable concat child does not hide real source faults (%p)", (Fault) => {
  const unavailable = ValueRoute.literal(FILE).project(["files"]);
  const route = new ValueRoute("concat", { children: [unavailable, sourceRoute(1)], sizes: [1, 1] });
  const failure = new Fault("source lookup failed");
  const broken = () => {
    throw failure;
  };
  expect(thrown(() => routeFeedback(route, "Review.", broken, { recordedValue: [1n, 2n] }))).toBe(failure);
  const error = thrown(() => routeFeedback(route, "Review.", () => 1n, { recordedValue: [1n, 2n] }));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("concat operand is not a list");
});

test("projection over unavailable composition uses recorded bounds", () => {
  const opaque = ValueRoute.literal(FILE).with({ kind: "opaque", expression: "opaque()" });
  const selected = opaque.project(["files"]);
  const route = new ValueRoute("concat", { children: [selected, ValueRoute.literal([])], sizes: [1, 0] });
  const value = { uri: "file:///tmp/x" };
  const feedback = { uri: "Review." };
  const routed = routeFeedback(route.project([0]), feedback, () => null, { recordedValue: value });
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped.map((stop) => stop.feedback)).toEqual([feedback]);
  expect(() => routeFeedback(route.project([1]), feedback, () => null, { recordedValue: value })).toThrow(
    "concat projection index is outside its recorded sizes",
  );
});

test("literal snapshots use the codec form", () => {
  expect(ValueRoute.literal(1n).valueJson).toBe("1");
  expect(ValueRoute.literal(1.0).valueJson).toBe("1.0");
  const infinite = ValueRoute.literal(Infinity);
  expect(infinite.valueJson).toBe('{"$float":"inf"}');
  expect(infinite.expression).toBe('{"$float":"inf"}');
  expect(routeValue(infinite, () => null)).toBe(Infinity);
  expect(ValueRoute.literal(-0).valueJson).toBe("0.0");

  const escaped = ValueRoute.literal({ $record: 1n });
  expect(escaped.valueJson).toBe('{"$record":{"$record":1}}');
  expect(routeValue(escaped, () => null)).toEqual({ $record: 1n });
  const context = new RouteContext({ expression: "x", resultJson: escaped.valueJson, sources: [] });
  expect(context.result).toEqual({ $record: 1n });
  expect(new RouteContext({ expression: "x", resultJson: null, sources: [] }).result).toBeNull();

  const deep = nested("leaf", 10_000);
  const route = ValueRoute.literal(deep);
  expect(route.valueJson).toBe(logJson(deep));
  expect(route.snapshot).toBeNull();
  expect(logJson(routeValue(route, () => null))).toBe(logJson(deep));
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

describe("validateFeedback and hasFeedback", () => {
  test.each([
    ['feedback["tags"][1] requires a recorded object', { tags: [null, { a: "x" }] }, { tags: ["a", "s"] }],
    [`feedback["tags"][1] selects unknown field 'b'`, { tags: [null, { b: "x" }] }, { tags: ["a", { c: 1n }] }],
    ['feedback["tags"][1] requires a recorded list', { tags: [null, ["x"]] }, { tags: ["a", "b"] }],
    ['feedback["tags"][1] must have exactly 2 items; got 1', { tags: [null, [null]] }, { tags: ["a", ["b", "c"]] }],
  ])("%s", (message, feedback, value) => {
    const error = thrown(() => validateFeedback(feedback, value));
    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toBe(message);
  });

  test("feedback of another type is a TypeError", () => {
    expect(() => validateFeedback(1n, "s")).toThrow(
      new TypeError("feedback must be a string, object, list, or None"),
    );
    expect(() => validateFeedback(new Date(0), {})).toThrow(TypeError);
    expect(() => validateFeedback({ tags: [null, 5] }, { tags: ["a", "b"] })).toThrow(
      new TypeError('feedback["tags"][1] must be a string, object, list, or None'),
    );
  });

  test("hasFeedback finds any string", () => {
    expect(hasFeedback("")).toBe(true);
    expect(hasFeedback({ a: [null] })).toBe(false);
    expect(hasFeedback({ a: [null, { b: "x" }] })).toBe(true);
    expect(hasFeedback(null)).toBe(false);
    expect(() => hasFeedback(5 as unknown as FeedbackValue)).toThrow(
      new TypeError("feedback must be a string, object, list, or None"),
    );
  });
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

test("record fields are own keys", () => {
  const a = sourceRoute(1, "a");
  const b = sourceRoute(2, "b");
  const route = ValueRoute.record([
    ["constructor", a],
    ["__proto__", b],
  ]);
  const lookup = lookupFrom([
    [a.source!, "va"],
    [b.source!, "vb"],
  ]);
  const value = routeValue(route, lookup) as Record<string, unknown>;
  expect(Object.hasOwn(value, "__proto__")).toBe(true);
  expect(Object.getPrototypeOf(value)).toBe(Object.prototype);
  const feedback = JSON.parse('{"constructor": "A", "__proto__": "B"}') as FeedbackValue;
  expect(messages(routeFeedback(route, feedback, lookup))).toEqual([
    [a.source!, "A"],
    [b.source!, "B"],
  ]);
  const error = thrown(() => routeFeedback(route, { toString: "x" }, lookup));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe("feedback selects unknown field 'toString'");
});

test("scalar and interpolation lift a projected selection", () => {
  const xs = sourceRoute(1, "xs");
  const scalar = ValueRoute.literal(["a", "b"]).with({ kind: "scalar", children: [xs], expression: "f(xs)" });
  const routed = routeFeedback(scalar.project([1]), "m", lookupFrom([[xs.source!, ["a", "b"]]]));
  expect(messages(routed)).toEqual([[new SourceRef("run", 1), [null, "m"]]]);
  expect(routed.stopped).toEqual([]);
});

describe("accepted result shape validates unavailable temporary routes", () => {
  const forms = [
    "literal",
    "nested_literal",
    "opaque",
    "selected",
    "list",
    "record",
    "concat",
    "project_record",
    "project_list",
  ] as const;
  test.each(forms)("%s", (form) => {
    const raw = form === "project_record" ? { file: FILE, other: 1n } : form === "project_list" ? [FILE, 1n] : FILE;
    const calls = countedStandIn("accepted_temporary", raw);
    const expression = form === "selected" ? "${accepted_temporary() | null}" : "${accepted_temporary()}";
    const traced = evalBinding(expression, nothing, { trace: true });
    const opaque = form === "selected" ? traced.route.children[0]! : traced.route;
    let route = traced.route;
    let value: unknown = { uri: "file:///tmp/example.txt" };
    let feedback: FeedbackValue = { uri: "Check the URI." };
    let expectedStop: FeedbackValue = feedback;
    if (form === "literal") {
      route = ValueRoute.literal(FILE);
    } else if (form === "nested_literal") {
      route = ValueRoute.literal({ files: [FILE] });
      [value, feedback] = [{ files: [value] }, { files: [feedback] }];
    } else if (form === "list") {
      route = new ValueRoute("list", { children: [route] });
      [value, feedback] = [[value], [feedback]];
    } else if (form === "record") {
      route = ValueRoute.record([["file", route]]);
      [value, feedback] = [{ file: value }, { file: feedback }];
    } else if (form === "concat") {
      route = new ValueRoute("concat", {
        children: [new ValueRoute("list", { children: [route] }), ValueRoute.literal([])],
        sizes: [1, 0],
      });
      [value, feedback] = [[value], [feedback]];
    } else if (form === "project_record") {
      route = route.project(["file"]);
      expectedStop = { file: feedback };
    } else if (form === "project_list") {
      route = route.project([0]);
      expectedStop = [feedback, null];
    }
    const routed = routeFeedback(route, feedback, () => null, { recordedValue: value });
    expect(routed.contributions).toEqual([]);
    expect(routed.stopped.map((stop) => stop.feedback)).toEqual(
      form === "literal" || form === "nested_literal" ? [] : [expectedStop],
    );
    if (routed.stopped.length > 0) {
      const context = routed.stopped[0]!.context.at(-1)!;
      expect(context.result).toBeNull();
      expect(context.resultType).toBe(opaque.resultType);
      expect(context.snapshot).toEqual(opaque.snapshot);
    }
    expect(calls).toHaveLength(1);
    expect(() => routeFeedback(route, feedback, () => null)).toThrow(/requires a recorded object$/);
  });
});

describe("accepted projected value stops at a bounded ancestor", () => {
  const rows: ["record" | "list", boolean, "opaque" | "literal"][] = [];
  for (const container of ["record", "list"] as const) {
    for (const wrapped of [false, true]) {
      for (const boundary of ["opaque", "literal"] as const) rows.push([container, wrapped, boundary]);
    }
  }
  test.each(rows)("%s, wrapped %p, %s boundary", (container, wrapped, boundary) => {
    let value: unknown = FILE;
    const path: (string | number)[] = Array.from({ length: 40 }, () => (container === "record" ? "nested" : 0));
    for (let i = 0; i < path.length; i++) value = container === "record" ? { nested: value } : [value, "sibling"];
    const opaque = ValueRoute.literal(value).with({ kind: "opaque", expression: "bounded_temporary()" });
    const boundaryRoute = boundary === "opaque" ? opaque : ValueRoute.literal(value);
    const parent = wrapped ? ValueRoute.record([["outer", boundaryRoute]]) : boundaryRoute;
    const selectedPath = wrapped ? ["outer", ...path] : path;
    const feedback = { uri: "Review." };
    const routed = routeFeedback(parent.project(selectedPath), feedback, () => null, {
      recordedValue: { uri: "file:///tmp/x" },
    });
    expect(routed.contributions).toEqual([]);
    if (boundary === "opaque") {
      expect(routed.stopped.length).toBe(1);
      const stop = routed.stopped[0]!;
      expect(stop.feedback).toEqual(feedback);
      expect(
        stop.context.some((d) => d.role === "projection" && isDeepStrictEqual(d.operand, selectedPath)),
      ).toBe(true);
      expect(stop.context.at(-1)!.snapshot).toEqual(opaque.snapshot);
    } else {
      expect(routed.stopped).toEqual([]);
    }
  });
});

test("projected partial container uses recorded operation positions", () => {
  const opaque = ValueRoute.literal([FILE, FILE]).with({ kind: "opaque", expression: "operation_temporary()" });
  const route = new ValueRoute("concat", {
    children: [opaque, ValueRoute.literal([])],
    sizes: [2, 0],
    expression: "(operation_temporary() + [])",
  });
  const feedback = { uri: "Check." };
  const routed = routeFeedback(route.project([1]), feedback, () => null, {
    recordedValue: { uri: "file:///tmp/x" },
  });
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped.map((stop) => stop.feedback)).toEqual([[null, feedback]]);
  const context = routed.stopped[0]!.context;
  expect(context.some((d) => d.role === "projection" && isDeepStrictEqual(d.operand, [1]))).toBe(true);
  expect(context.some((d) => d.positions.some((p) => isDeepStrictEqual(p, [1])))).toBe(true);
  expect(() =>
    routeFeedback(route.project([2]), feedback, () => null, { recordedValue: { uri: "file:///tmp/x" } }),
  ).toThrow("list index 2 is out of range");
});

describe("bounded projected lists compose before opaque stopping", () => {
  const rows: ["direct" | "concat", boolean][] = [
    ["direct", false],
    ["direct", true],
    ["concat", false],
    ["concat", true],
  ];
  test.each(rows)("%s, structured %p", (operation, structured) => {
    let raw: unknown = { files: [FILE] };
    const path = [...Array.from({ length: 40 }, () => "nested"), "files"];
    for (let i = 0; i < 40; i++) raw = { nested: raw };
    const calls = countedStandIn("bounded_files", raw);
    const opaque = evalBinding("${bounded_files()}", nothing, { trace: true }).route;
    const selected = opaque.project(path);
    const route =
      operation === "concat"
        ? new ValueRoute("concat", { children: [selected, ValueRoute.literal([])], sizes: [1, 0] })
        : selected;
    const value = [{ uri: "file:///tmp/x" }];
    const feedback: FeedbackValue = structured ? [{ uri: "Review." }] : "Review.";
    const routed = routeFeedback(route, feedback, () => null, { recordedValue: value });
    expect(routed.contributions).toEqual([]);
    expect(routed.stopped.map((stop) => stop.feedback)).toEqual([feedback]);
    expect(
      routed.stopped[0]!.context.some((d) => d.role === "projection" && isDeepStrictEqual(d.operand, path)),
    ).toBe(true);
    expect(routed.stopped[0]!.context.at(-1)!.snapshot).toEqual(opaque.snapshot);
    expect(calls).toHaveLength(1);
    expect(() => routeFeedback(route, [null, null], () => null, { recordedValue: value })).toThrow(RangeError);
    expect(() => routeFeedback(route, [{ unknown: null }], () => null, { recordedValue: value })).toThrow(
      RangeError,
    );
  });
});

test("routed feedback is frozen", () => {
  const routed = routeFeedback(sourceRoute(1, "x"), "Note.", () => "v");
  expect(Object.isFrozen(routed)).toBe(true);
  expect(Object.isFrozen(routed.contributions)).toBe(true);
  expect(Object.isFrozen(routed.contributions[0])).toBe(true);
  expect(Object.isFrozen(routed.contributions[0]!.context)).toBe(true);
  expect(routed.contributions[0]!.context[0]!.role).toBe("source");
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

  resolve = (path: string): ResolvedValue => {
    this.reads.push(path);
    const [head, ...parts] = path.split(".") as [string, ...string[]];
    if (!Object.hasOwn(this.values, head)) return new ResolvedValue(null, false);
    let value = this.values[head]!;
    for (const part of parts) {
      if (!isRecord(value) || !Object.hasOwn(value, part)) return new ResolvedValue(null, false);
      value = value[part]!;
    }
    const root = this.identities[head]!;
    return new ResolvedValue(value, true, new SourceRef(root.runId, root.executionSeq, parts));
  };

  lookup = (source: SourceRef): unknown => {
    expect(source.path).toEqual([]);
    const name = Object.keys(this.identities).find((key) => {
      const identity = this.identities[key]!;
      return identity.runId === source.runId && identity.executionSeq === source.executionSeq;
    });
    if (name === undefined) throw new Error(`no source ${source.runId}/${source.executionSeq}`);
    return this.values[name];
  };

  evaluate(expression: string): TracedValue {
    return evalExpr(parseExpr(expression), this.resolve, { trace: true });
  }

  feedback(traced: TracedValue, feedback: FeedbackValue): RoutedFeedback {
    return routeFeedback(traced.route, feedback, this.lookup);
  }
}

/** A traced resolver that resolves nothing. */
const nothing = (): ResolvedValue => new ResolvedValue(null, false);

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
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
  const stopped = routeFeedback(traced.route, "Check the score.", () => null).stopped;
  expect(stopped[0]!.context.at(-1)!.result).toBe(Infinity);
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
  expect(messages(routeFeedback(evaluated.bindings!.route, { over: "Add an item." }, sources.lookup))).toEqual([
    [sources.identities.items!, "Add an item."],
  ]);
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
    const expectedSource = sources.identities[index < 2 ? "left" : "right"]!;
    const expectedFeedback = index % 2 ? [null, "Fix."] : ["Fix.", null];
    const routed = routeFeedback(item.route, { x: "Fix." }, sources.lookup);
    expect(messages(routed)).toEqual([[expectedSource, expectedFeedback]]);
    expect(routed.contributions[0]!.context.map((context) => context.operand)).toContainEqual([index]);
  }
});

test("direct projection lifts feedback into source shape", () => {
  const original = { tags: ["one", "two"], summary: "text" };
  const root = new SourceRef("run", 7);
  const reads: string[] = [];
  const resolve = (path: string) => {
    reads.push(path);
    if (path === "producer.output.tags") {
      return new ResolvedValue(original.tags, true, new SourceRef("run", 7, ["tags"]));
    }
    return new ResolvedValue("two", true, new SourceRef("run", 7, ["tags", 1]));
  };

  const result = evalExpr(parseExpr("producer.output.tags"), resolve, { trace: true });
  let routed = routeFeedback(result.route, [null, "Bad."], () => original);
  expect(result.value).toBe(original.tags);
  expect(reads).toEqual(["producer.output.tags"]);
  expect(messages(routed)).toEqual([[root, { tags: [null, "Bad."] }]]);
  const projected = evalExpr(parseExpr("tag"), resolve, { trace: true });
  routed = routeFeedback(projected.route, "Bad.", () => original);
  expect(messages(routed)).toEqual([[root, { tags: [null, "Bad."] }]]);
});

test("accepted null has a source but absence does not", () => {
  const sources = new Sources({ accepted: null });
  const accepted = sources.identities.accepted!;
  const present = evalBinding("${accepted}", sources.resolve, { trace: true });
  const absent = evalTemplate(scanTemplate("${absent}"), sources.resolve, { trace: true });
  expect(present.value).toBeNull();
  expect(absent.value).toBeNull();
  expect(messages(sources.feedback(present, "Supply a value."))).toEqual([[accepted, "Supply a value."]]);
  expect(messages(sources.feedback(absent, "Supply a value."))).toEqual([]);
  expect(messages(sources.feedback(present, null))).toEqual([]);
  expect(messages(sources.feedback(present, ""))).toEqual([[accepted, ""]]);
  for (const name of ["accepted", "absent"]) {
    const error = thrown(() =>
      evalExpr(parseExpr(name), sources.resolve, { mode: ResolveMode.STRICT_RAISE, trace: true }),
    );
    expect(error).toBeInstanceOf(ExpressionError);
    expect((error as Error).message).toMatch(/unresolved/);
  }
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
    expect(messages(sources.feedback(result, "Keep this."))).toEqual([[sources.identities.winner!, "Keep this."]]);
  }
  for (const expression of ['empty :- "literal"', "empty | absent | null"]) {
    expect(messages(sources.feedback(sources.evaluate(expression), "Feedback."))).toEqual([]);
  }
  const nullDefault = sources.evaluate("absent :- empty");
  expect(messages(sources.feedback(nullDefault, "Still empty."))).toEqual([
    [sources.identities.empty!, "Still empty."],
  ]);
});

test("list constructor preserves nested routes", () => {
  const sources = new Sources({ producer: { tags: ["one", "two"], summary: "long" } });
  const producer = sources.identities.producer!;
  const result = sources.evaluate("[producer.tags, [producer.summary], producer.tags]");
  expect(result.value).toEqual([["one", "two"], ["long"], ["one", "two"]]);
  const feedback: FeedbackValue = [[null, "Bad."], ["Too long."], "List note."];
  expect(messages(sources.feedback(result, feedback))).toEqual([
    [producer, { tags: [null, "Bad."] }],
    [producer, { summary: "Too long." }],
    [producer, { tags: "List note." }],
  ]);
  expect(messages(sources.feedback(result, "Whole list."))).toHaveLength(3);
  expect(messages(sources.feedback(result, [null, [null], [null, null]]))).toEqual([]);
  const short = thrown(() => sources.feedback(result, [null]));
  expect(short).toBeInstanceOf(RangeError);
  expect((short as Error).message).toMatch(/exactly 3/);
  const inner = thrown(() => sources.feedback(result, [["Bad."], null, null]));
  expect(inner).toBeInstanceOf(RangeError);
  expect((inner as Error).message).toMatch(/exactly 2/);
  const record = ValueRoute.record([
    ["items", result.route],
    ["constant", ValueRoute.literal(1n)],
  ]);
  expect(messages(routeFeedback(record, { items: feedback }, sources.lookup))).toEqual(
    messages(sources.feedback(result, feedback)),
  );
  const unknown = thrown(() => routeFeedback(record, { unknown: "Bad." }, sources.lookup));
  expect(unknown).toBeInstanceOf(RangeError);
  expect((unknown as Error).message).toMatch(/unknown field/);
});

test("concat splits feedback at recorded operand lengths", () => {
  const sources = new Sources({ left: ["a"], right: ["b", "c"] });
  const { left, right } = sources.identities as { left: SourceRef; right: SourceRef };
  const result = sources.evaluate("left + right");
  expect(result.value).toEqual(["a", "b", "c"]);
  expect(messages(sources.feedback(result, [null, "Bad.", null]))).toEqual([[right, ["Bad.", null]]]);
  expect(messages(sources.feedback(result, "Too many."))).toEqual([
    [left, "Too many."],
    [right, "Too many."],
  ]);
  const repeated = sources.evaluate("left + left");
  expect(messages(sources.feedback(repeated, ["First.", "Second."]))).toEqual([
    [left, ["First."]],
    [left, ["Second."]],
  ]);
  expect(messages(sources.feedback(repeated, "Whole."))).toEqual([
    [left, "Whole."],
    [left, "Whole."],
  ]);
  expect(messages(sources.feedback(sources.evaluate("[] + right + []"), [null, "Bad."]))).toEqual([
    [right, [null, "Bad."]],
  ]);
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
    const routed = sources.feedback(result, "Original text.");
    expect(messages(routed)).toEqual(names.map((name) => [sources.identities[name]!, "Original text."]));
    for (const contribution of routed.contributions) {
      expect(contribution.context[0]!.result).toEqual(expected);
      expect(contribution.context[0]!.expression).toBeTruthy();
      expect(contribution.context[0]!.sources).toEqual(names.map((name) => sources.identities[name]!));
    }
    const rendered = evalBinding("x=${x}; items=${items}", sources.resolve, { trace: true });
    expect(rendered.value).toBe("x=3; items=[3,4]");
    const renderedRouted = sources.feedback(rendered, "Be concise.");
    expect(messages(renderedRouted)).toEqual([
      [sources.identities.x!, "Be concise."],
      [sources.identities.items!, "Be concise."],
    ]);
    expect(renderedRouted.contributions.map((c) => c.context[0]!.span)).toEqual([
      [2, 3],
      [11, 16],
    ]);
    for (const c of renderedRouted.contributions) expect(c.context[0]!.result).toBe(rendered.value);
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
  const routed = sources.feedback(rendered, "Note.");
  expect(routed.contributions.map((c) => c.context[0]!.span)).toEqual([
    [1, 2],
    [3, 4],
  ]);
  // The result has 4 code points but 5 UTF-16 units, so an end of 5 is past it.
  validateRoute(rendered.route);
  expect(() => validateRoute(rendered.route.with({ spans: [[1, 2], [3, 5]] }))).toThrow(
    new RangeError("interpolation spans must be ordered within the result"),
  );
});

test("sourceKey is equal exactly when run id, sequence and path are equal", () => {
  const key = sourceKey(new SourceRef("run", 1, ["a", 0]));
  expect(sourceKey(new SourceRef("run", 1, ["a", 0]))).toBe(key);
  expect(sourceKey(new SourceRef("run", 1, ["a", "0"]))).not.toBe(key);
  expect(sourceKey(new SourceRef("other", 1, ["a", 0]))).not.toBe(key);
  expect(sourceKey(new SourceRef("run", 2, ["a", 0]))).not.toBe(key);
  expect(sourceKey(new SourceRef("run", 1, ["a"]))).not.toBe(key);
});

test("sameSource compares two separately built sources by key", () => {
  const source = new SourceRef("run", 1, ["a", 0]);
  expect(sameSource(source, new SourceRef("run", 1, ["a", 0]))).toBe(true);
  expect(sameSource(source, new SourceRef("run", 1, ["a", "0"]))).toBe(false);
  expect(sameSource(source, new SourceRef("run", 2, ["a", 0]))).toBe(false);
});

test("sameFeedback compares strings and null exactly, arrays in order and records by key set", () => {
  const build = (): FeedbackValue => ({ tags: ["Wrong.", null, { why: "" }], summary: "Short.", none: null });
  expect(sameFeedback(build(), build())).toBe(true);
  expect(sameFeedback({ summary: "Short.", none: null, tags: ["Wrong.", null, { why: "" }] }, build())).toBe(true);
  expect(sameFeedback(null, null)).toBe(true);
  expect(sameFeedback("", "")).toBe(true);
  expect(sameFeedback([], [])).toBe(true);
  expect(sameFeedback({}, {})).toBe(true);
  const differing: [FeedbackValue, FeedbackValue][] = [
    ["a", "b"],
    ["", null],
    [null, []],
    [null, {}],
    [[], {}],
    [["a"], { 0: "a" }],
    [["a", null], ["a"]],
    [["a", null], [null, "a"]],
    [{ a: "x" }, { a: "x", b: "x" }],
    [{ a: "x" }, { b: "x" }],
    [{ a: "x" }, { a: "y" }],
    [{ a: null }, {}],
    [{ a: ["x"] }, { a: ["x", null] }],
    [build(), { tags: ["Wrong.", null, { why: "Other." }], summary: "Short.", none: null }],
  ];
  for (const [a, b] of differing) {
    expect(sameFeedback(a, b)).toBe(false);
    expect(sameFeedback(b, a)).toBe(false);
  }
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
] as [string, (values: Value[]) => unknown, string, Value, FeedbackValue][])(
  "opaque functions stop feedback and execute once: %s",
  (name, fn, suffix, expected, feedback) => {
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
    for (const supplied of [feedback, "Original feedback."]) {
      const routed = sources.feedback(result, supplied);
      expect(routed.contributions).toEqual([]);
      expect(routed.stopped).toHaveLength(1);
      expect(routed.stopped[0]!.feedback).toEqual(supplied);
      const detail = routed.stopped[0]!.context.at(-1)!;
      expect(detail.result).toEqual(expected);
      expect(detail.expression).toBe(`${name}(items)${suffix}`);
      expect(detail.sources).toEqual([sources.identities.items!]);
    }
    expect(sources.reads).toEqual(["items"]);
    expect(calls).toEqual([["a", "b"]]);
    expect(sources.feedback(result, null).stopped).toEqual([]);
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
  expect(messages(sources.feedback(result, "Fill this."))).toEqual([
    [sources.identities.right!, [{ answer: "Fill this." }]],
  ]);
  const missing = evalBinding("${item.absent}", sources.resolve, { item, trace: true });
  expect(messages(sources.feedback(missing, "Fill this."))).toEqual([]);
  const fallback = evalBinding('${item.answer :- "default"}', sources.resolve, { item, trace: true });
  expect(fallback.value).toBe("default");
  expect(messages(sources.feedback(fallback, "Feedback."))).toEqual([]);
  const localReads: string[] = [];
  const local = (path: string) => {
    localReads.push(path);
    return new ResolvedValue(null, true, over.route.project([1, "answer"]));
  };
  const localResult = evalBinding("${local}", local, { trace: true });
  expect(localReads).toEqual(["local"]);
  expect(messages(sources.feedback(localResult, "Fill this."))).toEqual(
    messages(sources.feedback(result, "Fill this.")),
  );
});

test("a file temporary preserves interpolation and projection", () => {
  const calls = countedStandIn("temporary", FILE);
  const untraced = evalBinding("value=${temporary()}", () => null);
  expect(untraced).toBe('value={"uri":"file:///tmp/x","mime":null,"name":null}');
  expect(calls).toEqual(["temporary"]);
  const traced = evalBinding("value=${temporary()}", nothing, { trace: true });
  expect(traced.value).toBe(untraced);
  const routed = routeFeedback(traced.route, "Shorten.", () => null);
  expect(routed.contributions).toEqual([]);
  const detail = routed.stopped[0]!.context.at(-1)!;
  expect(detail.resultJson).toBeNull();
  expect(detail.resultType).toBe("file");
  expect(routed.stopped[0]!.context[0]!.result).toBe(untraced);
  expect(calls).toEqual(["temporary", "temporary"]);

  standIn("record", () => ({ answer: "ok", temporary: FILE }), []);
  const projected = evalBinding("${record().answer}", nothing, { trace: true });
  expect(projected.value).toBe("ok");
  const stopped = routeFeedback(projected.route, "Explain.", () => null).stopped;
  expect(stopped[0]!.context.at(-1)!.result).toBe("ok");
});

test("routes keep opaque snapshots", () => {
  const produced = { tags: ["original", "other"] };
  standIn("snapshot", () => produced, []);
  const result = evalBinding("${snapshot()}", nothing, { trace: true });
  const encoding = JSON.stringify(result.route);
  produced.tags.push("later");
  const stopped = routeFeedback(result.route, { tags: ["Bad.", null] }, () => null).stopped;
  expect(stopped[0]!.context.at(-1)!.result).toEqual({ tags: ["original", "other"] });
  expect(JSON.stringify(result.route)).toBe(encoding);
  expect(Object.isFrozen(result.route)).toBe(true);
  expect(routeFeedback(result.route, {}, () => null).stopped).toEqual([]);
  expect(routeFeedback(result.route, { tags: [null, null] }, () => null).stopped).toEqual([]);
  const short = thrown(() => routeFeedback(result.route, { tags: ["Too short."] }, () => null));
  expect(short).toBeInstanceOf(RangeError);
  expect((short as Error).message).toMatch(/exactly 2/);
});

test("opaque stops preserve concat and interpolation context", () => {
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
  const routed = sources.feedback(concatenated, ["First.", null, "Second.", null]);
  expect(routed.stopped.map((stop) => stop.feedback)).toEqual([["First.", null]]);
  expect(routed.stopped.map((stop) => stop.context[0]!.positions)).toEqual([[[0], [1]]]);
  expect(messages(routed)).toEqual([[sources.identities.items!, ["Second.", null]]]);
  expect(calls).toEqual(["call"]);
  const shortCircuit = sources.evaluate("count or reorder(items)");
  expect(sources.feedback(shortCircuit, "Note.").stopped).toEqual([]);
  expect(calls).toEqual(["call"]);
  const interpolated = evalBinding("items=${reorder(items)}", sources.resolve, { trace: true });
  const stopped = sources.feedback(interpolated, "Note.").stopped[0]!;
  expect(stopped.feedback).toBe("Note.");
  expect(stopped.context[0]!.span).toEqual([6, 19]);
  expect(stopped.context[0]!.result).toBe('items=["two","one"]');
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
  expect(evalBinding("${ignore(nested())}", () => null)).toBe("ok");
  const traced = evalBinding("${ignore(nested())}", nothing, { trace: true });
  expect(traced.value).toBe("ok");
  expect(traced.route.children[0]!.valueJson).not.toBeNull();
  const stopped = routeFeedback(traced.route, "Feedback.", () => null).stopped;
  expect(stopped[0]!.context.at(-1)!.result).toBe("ok");
  expect(calls).toEqual(["nested", "ignore", "nested", "ignore"]);
});

test("unavailable snapshots keep container structure", () => {
  const calls = countedStandIn("temporary_list", [FILE]);
  const expected = 'items=[{"uri":"file:///tmp/x","mime":null,"name":null}]';
  const result = evalBinding("items=${temporary_list() + []}", nothing, { trace: true });
  expect(result.value).toBe(expected);
  const routed = routeFeedback(result.route, "Shorten.", () => null);
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped.map((stop) => stop.feedback)).toEqual(["Shorten."]);
  expect(routed.stopped[0]!.context[0]!.result).toBe(expected);
  expect(calls).toEqual(["temporary_list"]);
});

test("unavailable snapshot projects available fields and composed items", () => {
  const itemValue = { answer: "ok", temporary: FILE };
  const plain = evalBinding("${item.answer}", nothing, { item: itemValue, trace: true });
  expect(plain.value).toBe("ok");
  expect(routeFeedback(plain.route, "Shorten.", () => null).contributions).toEqual([]);

  const calls = countedStandIn("temporary_list", [itemValue]);
  const over = evalBinding("${temporary_list() + []}", nothing, { trace: true });
  const item = new ResolvedValue((over.value as Value[])[0]!, true, over.route.project([0]));
  const projected = evalBinding("${item.answer}", nothing, { item, trace: true });
  expect(projected.value).toBe("ok");
  const stopped = routeFeedback(projected.route, "Shorten.", () => null).stopped;
  expect(stopped.map((stop) => stop.feedback)).toEqual([[{ answer: "Shorten." }]]);
  expect(calls).toEqual(["temporary_list"]);
});

test("structural routes retain selection and position context", () => {
  const sources = new Sources({ left: ["a"], right: ["b", "c"], record: { tags: ["x", "y"] } });
  const combined = sources.evaluate("left + right");
  const contributions = sources.feedback(combined, "Too long.").contributions;
  const details = contributions.map((part) => part.context.find((c) => c.role === "concat")!);
  expect(details.map((c) => c.operand)).toEqual([0, 1]);
  expect(details.map((c) => c.span)).toEqual([
    [0, 1],
    [1, 3],
  ]);
  expect(details.map((c) => c.positions)).toEqual([[[0]], [[1], [2]]]);
  for (const c of details) {
    expect(c.expression).toBe("(left + right)");
    expect(c.result).toEqual(combined.value);
  }
  expect(details[0]!.resultJson).toBe(details[1]!.resultJson);
  expect(contributions.map((part) => part.context.at(-1)!.sources)).toEqual([
    [sources.identities.left!],
    [sources.identities.right!],
  ]);
  for (const part of contributions) expect(part.context.at(-1)!.resultJson).toBeNull();

  const repeated = sources.feedback(sources.evaluate("left + left"), "Same text.").contributions;
  expect(repeated.map((part) => part.context.find((c) => c.role === "concat")!.span)).toEqual([
    [0, 1],
    [1, 2],
  ]);
  const nestedList = sources.evaluate("[[left], right]");
  const nestedContribution = sources.feedback(nestedList, [["Bad."], null]).contributions[0]!;
  const items = nestedContribution.context.filter((c) => c.role === "item");
  expect(items.map((c) => c.operand)).toEqual([0, 0]);
  expect(items.map((c) => c.result)).toEqual([nestedList.value, [["a"]]]);

  const record = ValueRoute.record([
    ["other", sources.evaluate("left").route],
    ["chosen", sources.evaluate("right").route],
  ]);
  const projected = routeFeedback(record.project(["chosen", 1]), "Bad.", sources.lookup).contributions[0]!;
  expect(projected.feedback).toEqual([null, "Bad."]);
  expect(projected.context[0]!.role).toBe("projection");
  expect(projected.context[0]!.operand).toEqual(["chosen", 1]);
  expect(projected.context[0]!.result).toBe("c");
  const field = projected.context.find((c) => c.role === "field")!;
  expect(field.operand).toBe("chosen");
  expect(field.result).toEqual({ other: ["a"], chosen: ["b", "c"] });
  expect(field.expression).toBeTruthy();

  const direct = sources.feedback(sources.evaluate("record.tags"), [null, "Bad."]).contributions[0]!;
  expect(direct.context.at(-1)!.role).toBe("projection");
  expect(direct.context.at(-1)!.sources[0]!.path).toEqual(["tags"]);
});

test.each(["Note.", null] as FeedbackValue[])(
  "a deep temporary composes with text or absent feedback (%p)",
  (feedback) => {
    const calls = countedStandIn("deep", nested(1n, 10_000));
    expect(Array.isArray(evalBinding("${deep() + []}", () => null))).toBe(true);
    const traced = evalBinding("${deep() + []}", nothing, { trace: true });
    expect(Array.isArray(traced.value)).toBe(true);
    const routed = routeFeedback(traced.route, feedback, () => null);
    expect(routed.contributions).toEqual([]);
    expect(routed.stopped.map((stop) => stop.feedback)).toEqual(feedback === null ? [] : [feedback]);
    expect(JSON.stringify(traced.route)).toBeTruthy();
    expect(calls).toEqual(["deep", "deep"]);
  },
);

test("a plain item has no route to follow", () => {
  const item = { answer: "ok" };
  expect(evalBinding("${item.answer}", () => null, { item })).toBe("ok");
  const traced = evalBinding("${item.answer}", nothing, { item, trace: true });
  expect(traced.value).toBe("ok");
  const routed = routeFeedback(traced.route, "Note.", () => null);
  expect(routed.contributions).toEqual([]);
  expect(routed.stopped).toEqual([]);
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
  "JSON-compatible nested results keep complete feedback shape: %s",
  (container, depth) => {
    let value: unknown = "leaf";
    let feedback: FeedbackValue = "Explain.";
    for (let i = 0; i < depth; i++) {
      value = container === "list" ? [value] : { child: value };
      feedback = container === "list" ? [feedback] : { child: feedback };
    }
    const calls = countedStandIn("nested", value);
    const traced = evalBinding("${nested()}", nothing, { trace: true });
    expect(traced.value).toEqual(value as Value);
    validateFeedback(feedback, traced.value);
    const stopped = routeFeedback(traced.route, feedback, () => null).stopped;
    expect(stopped).toHaveLength(1);
    expect(stopped[0]!.feedback).toEqual(feedback);
    expect(stopped[0]!.context.at(-1)!.result).toEqual(value);
    expect(stopped[0]!.context.at(-1)!.resultType).toBeNull();
    expect(calls).toEqual(["nested"]);
  },
);

test("a traced resolver must return ResolvedValue", () => {
  const resolve = (() => 1n) as unknown as (path: string) => ResolvedValue;
  const error = thrown(() => evalExpr(parseExpr("a"), resolve, { trace: true }));
  expect(error).toBeInstanceOf(TypeError);
  expect((error as Error).message).toBe("a traced resolver must return ResolvedValue");
});

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
  expect(messages(sources.feedback(result, [null, "Bad."]))).toEqual([[producer, { tags: [null, null, "Bad."] }]]);
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
  // A turn with no feedback lifts to a whole null, as a hand-built route of the same items does.
  const routed = messages(sources.feedback(result, [null, null, "Bad."]));
  expect(routed).toEqual([[producer, { history: [null, [null, { text: "Bad." }]] }]]);
  const base = sources.evaluate("producer.history");
  const items = [
    [0, 0],
    [1, 0],
    [1, 1],
  ].map(([i, j]) => base.route.project([i!, j!, "text"]));
  const handBuilt = new ValueRoute("list", { children: items });
  expect(messages(routeFeedback(handBuilt, [null, null, "Bad."], sources.lookup))).toEqual(routed);
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
