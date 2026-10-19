/**
 * Route validation and backward context reads: the `validated` column of
 * `test_route_errors_on_multi_error_trees` and `test_downstream_context_reads_are_unaliased` of
 * `tests/engine/test_route_chain_costs.py`. Python's `ValueError` is a `RangeError` here.
 *
 * TS-only: one row per message `validateRoute` and `validateSource` can reach, with its class; a
 * `bigint` sequence is refused; `deep: false` skips a bad child and a bad snapshot's own fields
 * but still refuses a source inside the snapshot; `{"$float":"inf"}`, a 5000-digit int and a
 * literal nested 10,000 levels deep are valid `value_json` (the reader has no digit limit and
 * isn't recursive). A malformed `value_json` throws the reader's `JSONDecodeError`.
 *
 * Changed: reads return the frozen copy the constructor made, not a fresh copy per read, so the
 * test checks that a shared subtree is copied once, that the caller's objects aren't aliased, and
 * that the stored value is frozen.
 *
 * Not ported: Python's "route <field> must be a tuple" and span pair-length checks, since
 * `ValueRoute` and `SourceRef` copy every array field into an array and rebuild spans as pairs;
 * the long-int rows of `tests/engine/test_long_int_old_logs.py`, since TS ints have no digit limit.
 *
 * Later (T4-2): the `encoded` and `decoded` columns of `test_route_errors_on_multi_error_trees`
 * wait for the route payloads of `runtime/log.ts`.
 *
 * Later (T4-x / T4-8): the other 7 run flows or encode route payloads:
 * `test_backward_results_match_the_recorded_results`,
 * `test_route_encode_and_decode_check_each_node_once`,
 * `test_run_and_resume_route_checks_grow_linearly`, `test_backward_lookups_grow_near_linearly`,
 * `test_backward_context_copies_grow_near_linearly`, `test_backward_freeze_work_grows_near_linearly`
 * and `test_backward_peak_memory_grows_near_linearly`.
 */

import { expect, test } from "bun:test";

import { type RouteKind, SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { BackwardContext } from "../../src/nodes/base.ts";
import { validateRoute, validateSource } from "../../src/runtime/contracts.ts";
import { JSONDecodeError } from "../../src/typesys/json.ts";

/** One valid source leaf shared by the route error trees. */
const SOURCE = new ValueRoute("source", { source: new SourceRef("run", 1) });

/** A source route whose expression is not a string. */
const BAD = new ValueRoute("source", { source: new SourceRef("run", 1), expression: 5 as unknown as string });

/** Return the error class and message `call` throws. */
function raised(call: () => unknown): [unknown, string] {
  try {
    call();
  } catch (error) {
    return [(error as Error).constructor, (error as Error).message];
  }
  throw new Error("nothing was thrown");
}

/** Build a route of a kind `ValueRoute`'s type doesn't allow, or with fields of the wrong type. */
function forged(kind: string, fields: Record<string, unknown> = {}): ValueRoute {
  return new ValueRoute(kind as RouteKind, fields as never);
}

test.each([
  [
    "duplicate names and a bad child",
    new ValueRoute("record", { children: [BAD, SOURCE], names: ["a", "a"] }),
    TypeError,
    "route expression must be a string",
  ],
  [
    "a head error and a bad child",
    new ValueRoute("source", { source: new SourceRef("run", 1), children: [BAD] }),
    RangeError,
    "route source cannot carry children",
  ],
  [
    "bad spans and a bad child snapshot",
    new ValueRoute("interpolation", {
      children: [new ValueRoute("scalar", { children: [SOURCE], resultType: "Thing", snapshot: SOURCE })],
      valueJson: '"ab"',
      spans: [[2, 1]],
    }),
    RangeError,
    "partial snapshot cannot contain source routes",
  ],
  [
    "a snapshot with an invalid source",
    new ValueRoute("scalar", {
      children: [SOURCE],
      resultType: "Thing",
      snapshot: new ValueRoute("list", { children: [new ValueRoute("source", { source: new SourceRef("", 1) })] }),
    }),
    RangeError,
    "source run_id must be non-empty",
  ],
] as [string, ValueRoute, unknown, string][])("route errors on multi-error trees: %s", (_name, tree, cls, message) => {
  expect(raised(() => validateRoute(tree))).toEqual([cls, message]);
});

const SPANS = "interpolation spans must be ordered within the result";

test.each([
  ["not a ValueRoute", { kind: "literal", valueJson: "1" }, TypeError, "route must be an exact ValueRoute"],
  ["an unknown kind", forged("bogus"), RangeError, "unknown route kind"],
  ["a repeat kind", forged("repeat", { children: [SOURCE, SOURCE], sizes: [1] }), RangeError, "unknown route kind"],
  ["a non-string expression", forged("list", { expression: 5 }), TypeError, "route expression must be a string"],
  [
    "a value_json on a source",
    new ValueRoute("source", { source: new SourceRef("run", 1), valueJson: "1" }),
    RangeError,
    "route source cannot carry value_json",
  ],
  [
    "children on a literal",
    new ValueRoute("literal", { valueJson: "1", children: [SOURCE] }),
    RangeError,
    "route literal cannot carry children",
  ],
  [
    "a source on a list",
    new ValueRoute("list", { source: new SourceRef("run", 1) }),
    RangeError,
    "route list cannot carry source",
  ],
  [
    "a negative path part",
    new ValueRoute("project", { children: [SOURCE], path: [-1] }),
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  [
    "a fractional path part",
    new ValueRoute("project", { children: [SOURCE], path: [1.5] }),
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
  [
    "a non-string name",
    forged("record", { children: [SOURCE], names: [1] }),
    RangeError,
    "route names must be strings",
  ],
  [
    "duplicate record names",
    new ValueRoute("record", { children: [SOURCE, SOURCE], names: ["a", "a"] }),
    RangeError,
    "record route names must be unique",
  ],
  [
    "a negative size",
    new ValueRoute("concat", { children: [SOURCE, SOURCE], sizes: [-1, 0] }),
    RangeError,
    "route sizes must be non-negative integers",
  ],
  [
    "a selected without a child",
    new ValueRoute("selected"),
    RangeError,
    "selected and project routes require one child",
  ],
  [
    "a project with two children",
    new ValueRoute("project", { children: [SOURCE, SOURCE], path: ["a"] }),
    RangeError,
    "selected and project routes require one child",
  ],
  [
    "a project without a path",
    new ValueRoute("project", { children: [SOURCE] }),
    RangeError,
    "project route requires a path",
  ],
  [
    "a record with fewer names than children",
    new ValueRoute("record", { children: [SOURCE] }),
    RangeError,
    "record route names must match its children",
  ],
  [
    "an opaque with fewer names than children",
    new ValueRoute("opaque", { children: [SOURCE], valueJson: "1" }),
    RangeError,
    "opaque route names must match its children",
  ],
  [
    "a concat with one child",
    new ValueRoute("concat", { children: [SOURCE], sizes: [1] }),
    RangeError,
    "concat route requires two children and sizes",
  ],
  [
    "a scalar without operands",
    new ValueRoute("scalar", { valueJson: "1" }),
    RangeError,
    "scalar route requires operands",
  ],
  [
    "a non-string value_json",
    forged("literal", { valueJson: 5 }),
    TypeError,
    "route value_json must be a string or null",
  ],
  [
    "an empty result_type",
    new ValueRoute("literal", { resultType: "" }),
    RangeError,
    "route result_type must be a non-empty string or null",
  ],
  [
    "both result forms",
    new ValueRoute("literal", { valueJson: "1", resultType: "Thing" }),
    RangeError,
    "route requires either a JSON result or an unavailable type",
  ],
  [
    "neither result form",
    new ValueRoute("literal"),
    RangeError,
    "route requires either a JSON result or an unavailable type",
  ],
  [
    "a snapshot with a value",
    new ValueRoute("scalar", { children: [SOURCE], valueJson: "1", snapshot: ValueRoute.literal(1n) }),
    RangeError,
    "partial snapshot requires an unavailable result",
  ],
  [
    "a source inside a snapshot",
    new ValueRoute("literal", { resultType: "Thing", snapshot: SOURCE }),
    RangeError,
    "partial snapshot cannot contain source routes",
  ],
  [
    "a non-string interpolation result",
    new ValueRoute("interpolation", { children: [SOURCE], valueJson: "1", spans: [[0, 1]] }),
    RangeError,
    "interpolation requires a string result and one span per child",
  ],
  [
    "a span item that isn't an integer",
    new ValueRoute("interpolation", { children: [SOURCE], valueJson: '"ab"', spans: [[0, 1.5]] }),
    TypeError,
    "interpolation spans must be integer pairs",
  ],
  [
    "a span past the result",
    new ValueRoute("interpolation", { children: [SOURCE], valueJson: '"ab"', spans: [[0, 3]] }),
    RangeError,
    SPANS,
  ],
  [
    "misordered spans",
    new ValueRoute("interpolation", {
      children: [SOURCE, SOURCE],
      valueJson: '"ab"',
      spans: [
        [1, 2],
        [0, 1],
      ],
    }),
    RangeError,
    SPANS,
  ],
] as [string, unknown, unknown, string][])("validateRoute refuses %s", (_name, route, cls, message) => {
  expect(raised(() => validateRoute(route))).toEqual([cls, message]);
});

test("validateRoute throws the reader's error for a duplicate value_json key", () => {
  let error: unknown;
  try {
    validateRoute(new ValueRoute("literal", { valueJson: '{"a": 1, "a": 2}' }));
  } catch (caught) {
    error = caught;
  }
  expect(error).toBeInstanceOf(JSONDecodeError);
  expect((error as JSONDecodeError).reason).toBe('duplicate JSON object key: "a"');
});

test.each([
  ["not a SourceRef", { runId: "run", executionSeq: 1, path: [] }, TypeError, "source must be an exact SourceRef"],
  ["an empty run id", new SourceRef("", 1), RangeError, "source run_id must be non-empty"],
  [
    "a negative sequence",
    new SourceRef("run", -1),
    RangeError,
    "source execution_seq must be a non-negative integer",
  ],
  [
    "a bigint sequence",
    new SourceRef("run", 1n as unknown as number),
    RangeError,
    "source execution_seq must be a non-negative integer",
  ],
  [
    "a negative path part",
    new SourceRef("run", 1, ["a", -1]),
    RangeError,
    "route path components must be strings or non-negative integers",
  ],
] as [string, unknown, unknown, string][])("validateSource refuses %s", (_name, source, cls, message) => {
  expect(raised(() => validateSource(source))).toEqual([cls, message]);
});

test("a shallow check skips children and the snapshot's own fields but not a source in the snapshot", () => {
  const record = new ValueRoute("record", { children: [BAD], names: ["a"] });
  validateRoute(record, { deep: false });
  expect(raised(() => validateRoute(record))).toEqual([TypeError, "route expression must be a string"]);

  const badSnapshot = new ValueRoute("scalar", {
    children: [SOURCE],
    resultType: "Thing",
    snapshot: forged("list", { expression: 5 }),
  });
  validateRoute(badSnapshot, { deep: false });
  expect(raised(() => validateRoute(badSnapshot))).toEqual([TypeError, "route expression must be a string"]);

  const sourceSnapshot = new ValueRoute("scalar", {
    children: [SOURCE],
    resultType: "Thing",
    snapshot: new ValueRoute("list", { children: [SOURCE] }),
  });
  expect(raised(() => validateRoute(sourceSnapshot, { deep: false }))).toEqual([
    RangeError,
    "partial snapshot cannot contain source routes",
  ]);
});

test.each([
  ["an infinite float", '{"$float":"inf"}'],
  ["a 5000-digit int", "9".repeat(5000)],
  ["a list nested 10,000 levels", "[".repeat(10_000) + "]".repeat(10_000)],
])("validateRoute accepts a literal holding %s", (_name, valueJson) => {
  validateRoute(new ValueRoute("literal", { valueJson }));
});

test("downstream context reads are unaliased", () => {
  const shared = { note: ["kept"] };
  const items = [{ first: shared, again: shared }, { second: shared }];
  const context = new BackwardContext({}, { downstreamContext: items });

  const read = context.downstreamContext;
  expect(read).toEqual(items);
  expect(read[0]!.first).toBe(read[0]!.again!);
  expect(read[0]!.first).toBe(read[1]!.second!);
  expect(read[0]!.first).not.toBe(shared);
  shared.note.push("changed");
  expect(read[1]!.second).toEqual({ note: ["kept"] });
  expect(Object.isFrozen(read)).toBe(true);
  expect(Object.isFrozen(read[0])).toBe(true);
  expect(() => {
    (read[0]!.first as { note: string[] }).note.push("changed");
  }).toThrow(TypeError);
  expect(context.downstreamContext).toBe(read);
});
