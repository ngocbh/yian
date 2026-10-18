/**
 * Feedback follows the values and selected operations actually evaluated: a port of
 * `tests/engine/test_expr_lineage.py`. A `FileRef` stands in for Python's non-JSON temporaries
 * (`object()`), since a file is the one yian value JSON can't hold.
 *
 * TS-only: the codec form of literal snapshots, a file as an unavailable leaf, own-key record
 * fields, the depth bound inside a partial snapshot, the focused `validateFeedback` and
 * `hasFeedback` checks, and a scalar route lifting a projected selection.
 *
 * Not ported:
 * - The MAP and WAIT capture tests (`test_reserved_map_over_uses_the_same_capture_path`,
 *   `test_equal_map_items_get_distinct_recorded_indices`,
 *   `test_wait_capture_retains_the_delivered_iso_value_after_normalization`) wait for the
 *   runtime.
 * - Temporaries `fromHost` refuses, so a builtin can't return them: the NaN row of
 *   `test_nonfinite_temporary_keeps_a_valid_final_interpolation`, the lone-surrogate
 *   `test_noncanonical_unicode_temporary_preserves_comparison_and_partial_snapshot`,
 *   `test_cyclic_temporary_keeps_outer_shape_for_composition`, the `__repr__`/`__str__` render
 *   counts of the temporary tests, `test_capture_does_not_call_dictionary_subclass_hooks`,
 *   `test_repetition_capture_does_not_call_integer_conversion` and the dict-subclass hooks of
 *   the two item-tracing tests.
 * - Repetition is a type error in TS, so `test_repetition_preserves_copy_messages_and_shape_context`,
 *   `test_empty_repetition_does_not_infer_its_original_content_from_the_result` and every
 *   `repeat` / `* 2` row are gone.
 */

import { describe, expect, test } from "bun:test";

import {
  type FeedbackValue,
  hasFeedback,
  type RouteKind,
  type RoutedFeedback,
  RouteContext,
  routeFeedback,
  routeValue,
  SourceRef,
  validateFeedback,
  ValueRoute,
} from "../../src/expr/lineage.ts";
import { FileRef } from "../../src/typesys/index.ts";
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

test("None feedback does not read source snapshots", () => {
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
  expect(() => routeFeedback(absent, "Review.", () => ({}), { recordedValue: "present" })).toThrow(RangeError);
  const malformed = new ValueRoute("malformed" as RouteKind);
  expect(() => routeFeedback(malformed, "Review.", lookup, { recordedValue: "present" })).toThrow(
    /unknown route kind/,
  );
  expect(() => routeFeedback(malformed, "Review.", lookup, { recordedValue: "present" })).toThrow(RangeError);
});

test.each([Error, TypeError, RangeError])("unavailable concat child does not hide real source faults (%p)", (Fault) => {
  const unavailable = ValueRoute.literal(FILE).project(["files"]);
  const route = new ValueRoute("concat", { children: [unavailable, sourceRoute(1)], sizes: [1, 1] });
  const failure = new Fault("source lookup failed");
  const broken = () => {
    throw failure;
  };
  expect(thrown(() => routeFeedback(route, "Review.", broken, { recordedValue: [1n, 2n] }))).toBe(failure);
  expect(() => routeFeedback(route, "Review.", () => 1n, { recordedValue: [1n, 2n] })).toThrow(TypeError);
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
    RangeError,
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
    ["feedback['tags'][1] requires a recorded object", { tags: [null, { a: "x" }] }, { tags: ["a", "s"] }],
    ["feedback['tags'][1] selects unknown field 'b'", { tags: [null, { b: "x" }] }, { tags: ["a", { c: 1n }] }],
    ["feedback['tags'][1] requires a recorded list", { tags: [null, ["x"]] }, { tags: ["a", "b"] }],
    ["feedback['tags'][1] must have exactly 2 items; got 1", { tags: [null, [null]] }, { tags: ["a", ["b", "c"]] }],
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
      new TypeError("feedback['tags'][1] must be a string, object, list, or None"),
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
  expect(() => routeFeedback(route, { toString: "x" }, lookup)).toThrow("selects unknown field 'toString'");
  expect(() => routeFeedback(route, { toString: "x" }, lookup)).toThrow(RangeError);
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
    const opaque = ValueRoute.literal(raw).with({ kind: "opaque", expression: "accepted_temporary()" });
    let route = form === "selected" ? new ValueRoute("selected", { children: [opaque] }) : opaque;
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
    expect(() => routeFeedback(route, feedback, () => null)).toThrow(RangeError);
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
        stop.context.some((d) => d.role === "projection" && Bun.deepEquals(d.operand, selectedPath)),
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
  expect(context.some((d) => d.role === "projection" && Bun.deepEquals(d.operand, [1]))).toBe(true);
  expect(context.some((d) => d.positions.some((p) => Bun.deepEquals(p, [1])))).toBe(true);
  expect(() =>
    routeFeedback(route.project([2]), feedback, () => null, { recordedValue: { uri: "file:///tmp/x" } }),
  ).toThrow(RangeError);
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
    const opaque = ValueRoute.literal(raw).with({ kind: "opaque", expression: "bounded_files()" });
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
      routed.stopped[0]!.context.some((d) => d.role === "projection" && Bun.deepEquals(d.operand, path)),
    ).toBe(true);
    expect(routed.stopped[0]!.context.at(-1)!.snapshot).toEqual(opaque.snapshot);
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
