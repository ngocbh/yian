/**
 * Route validation, route payloads and backward context reads: each message `validateRoute` and
 * `validateSource` can reach, round trips through the run-log JSON, unaliased frozen reads, and
 * backward lookups that grow near linearly with the chain.
 */

import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, spyOn, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RouteKind, SourceRef, ValueRoute } from "../../src/expr/lineage.ts";
import { BackwardContext } from "../../src/nodes/base.ts";
import { runFlow } from "../../src/run.ts";
import { validateRoute, validateSource } from "../../src/runtime/contracts.ts";
import { routeFromPayload, routePayload } from "../../src/runtime/log.ts";
import { RetainedTrace } from "../../src/runtime/trace.ts";
import { JSONDecodeError, logJson, parseJson } from "../../src/typesys/json.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { FakeMemory } from "../helpers/memory_fakes.ts";

/** One valid source leaf shared by the route error trees. */
const SOURCE = new ValueRoute("source", { source: new SourceRef("run", 1) });

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

/** A source route whose expression is not a string. */
const BAD = forged("source", { source: new SourceRef("run", 1), expression: 5 });

/** A route payload as a mutable object: nested payloads and arrays are `any`. */
type Payload = Record<string, any>;

/** Return `route`'s payload as a mutable object. */
function payloadOf(route: ValueRoute): Payload {
  return routePayload(route) as Payload;
}

/** A record with duplicate names whose first child has a non-string expression. */
function duplicateNamesAndBadChild(): [ValueRoute, Payload] {
  const tree = new ValueRoute("record", { children: [BAD, SOURCE], names: ["a", "a"] });
  const payload = payloadOf(new ValueRoute("record", { children: [SOURCE, SOURCE], names: ["a", "b"] }));
  payload.names = ["a", "a"];
  payload.children[0].expression = 5n;
  return [tree, payload];
}

/** A source route carrying a child whose expression is not a string. */
function headErrorAndBadChild(): [ValueRoute, Payload] {
  const tree = new ValueRoute("source", { source: new SourceRef("run", 1), children: [BAD] });
  const payload = payloadOf(SOURCE);
  payload.children = [{ ...payloadOf(SOURCE), expression: 5n }];
  return [tree, payload];
}

/** An interpolation with misordered spans over a scalar whose snapshot holds a source. */
function badSpansAndBadChildSnapshot(): [ValueRoute, Payload] {
  const child = new ValueRoute("scalar", { children: [SOURCE], resultType: "Thing", snapshot: SOURCE });
  const tree = new ValueRoute("interpolation", { children: [child], valueJson: '"ab"', spans: [[2, 1]] });
  const validChild = new ValueRoute("scalar", { children: [SOURCE], valueJson: "1" });
  const payload = payloadOf(
    new ValueRoute("interpolation", { children: [validChild], valueJson: '"ab"', spans: [[0, 1]] }),
  );
  payload.spans = [[2n, 1n]];
  Object.assign(payload.children[0], { value_json: null, result_type: "Thing", snapshot: payloadOf(SOURCE) });
  return [tree, payload];
}

/** A scalar whose list snapshot contains a source route with an empty run id. */
function snapshotWithInvalidSource(): [ValueRoute, Payload] {
  const snapshot = new ValueRoute("list", { children: [new ValueRoute("source", { source: new SourceRef("", 1) })] });
  const tree = new ValueRoute("scalar", { children: [SOURCE], resultType: "Thing", snapshot });
  const validSnapshot = new ValueRoute("list", { children: [ValueRoute.literal(1n)] });
  const payload = payloadOf(
    new ValueRoute("scalar", { children: [SOURCE], resultType: "Thing", snapshot: validSnapshot }),
  );
  payload.snapshot.children[0] = payloadOf(SOURCE);
  payload.snapshot.children[0].source.run_id = "";
  return [tree, payload];
}

test.each([
  [
    "duplicate names and a bad child",
    duplicateNamesAndBadChild,
    [TypeError, "route expression must be a string"],
    [TypeError, "route expression must be a string"],
    [TypeError, "route expression must be a string"],
  ],
  [
    "a head error and a bad child",
    headErrorAndBadChild,
    [RangeError, "route source cannot carry children"],
    [RangeError, "route source cannot carry children"],
    // The decoder checks each child before its parent.
    [TypeError, "route expression must be a string"],
  ],
  [
    "bad spans and a bad child snapshot",
    badSpansAndBadChildSnapshot,
    [RangeError, "partial snapshot may contain only literal, list and record routes"],
    [RangeError, "partial snapshot may contain only literal, list and record routes"],
    [RangeError, "partial snapshot may contain only literal, list and record routes"],
  ],
  [
    "a snapshot with an invalid source",
    snapshotWithInvalidSource,
    [RangeError, "source run_id must be non-empty"],
    [RangeError, "source run_id must be non-empty"],
    [RangeError, "source run_id must be non-empty"],
  ],
] as [string, () => [ValueRoute, Payload], [unknown, string], [unknown, string], [unknown, string]][])(
  "route errors on multi-error trees: %s",
  (_name, build, validated, encoded, decoded) => {
    const [tree, payload] = build();
    expect(raised(() => validateRoute(tree))).toEqual(validated);
    expect(raised(() => routePayload(tree))).toEqual(encoded);
    expect(raised(() => routeFromPayload(payload))).toEqual(decoded);
  },
);

/** A balanced `concat` route over `leaves` one-item source lists, as a traced `+` chain records it. */
function concatTree(leaves: number): ValueRoute {
  const build = (low: number, high: number): ValueRoute => {
    if (high - low === 1) return new ValueRoute("source", { source: new SourceRef("run", low) });
    const middle = low + Math.floor((high - low) / 2);
    return new ValueRoute("concat", {
      children: [build(low, middle), build(middle, high)],
      sizes: [middle - low, high - middle],
    });
  };
  return build(0, leaves);
}

test.each([
  ["concat-64", concatTree(64)],
  ["concat-256", concatTree(256)],
  ["scalar-256", new ValueRoute("scalar", { children: Array(256).fill(SOURCE), valueJson: "256" })],
] as [string, ValueRoute][])("a route round trips through its payload: %s", (_name, tree) => {
  const payload = parseJson(logJson(routePayload(tree)), "codec");
  expect(routeFromPayload(payload)).toEqual(tree);
});

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
  ["an empty array as a list's source", forged("list", { source: [] }), RangeError, "route list cannot carry source"],
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
    "partial snapshot may contain only literal, list and record routes",
  ],
  [
    "a non-string interpolation result",
    new ValueRoute("interpolation", { children: [SOURCE], valueJson: "1", spans: [[0, 1]] }),
    RangeError,
    "interpolation requires a string result or none, and one span per child",
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
    "interpolation spans must be ordered within the result",
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
    "interpolation spans must be ordered within the result",
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
    "partial snapshot may contain only literal, list and record routes",
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

// ---------- backward over long chains ----------

const dir = mkdtempSync(join(tmpdir(), "yian-route-chain-costs-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/**
 * Load a flow calling an identity child on an `n`-operand chain, with its inputs:
 * `list` concatenates `input.values` (a balanced `concat` route), `scalar` adds
 * `input.x` (one `scalar` route with `n` operands).
 */
async function chainFlow(kind: "list" | "scalar", n: number) {
  if (kind === "list") {
    const chain = Array(n).fill("input.values").join(" + ");
    const text =
      "id: lc\nname: lc\nlong_term_memory: test\ninput:\n  values: !list(str)\n" +
      "defs:\n  identity:\n    input:\n      values: !list(str)\n    nodes: {}\n" +
      "    output: ${input.values}\n" +
      "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n" +
      `      values: \${${chain}}\n` +
      "output: ${outer.output}\n";
    return { flow: await loadFlow(text), inputs: { values: ["v"] } };
  }
  const chain = Array(n).fill("input.x").join(" + ");
  const text =
    "id: bw\nname: bw\nlong_term_memory: test\ninput:\n  x: !int\n" +
    "defs:\n  identity:\n    input:\n      v: !int\n    nodes: {}\n    output: ${input.v}\n" +
    "nodes:\n  outer:\n    kind: call\n    call: identity\n    input:\n" +
    `      v: \${${chain}}\n` +
    "output: ${outer.output}\n";
  return { flow: await loadFlow(text), inputs: { x: 1 } };
}

/** Run a learnable `kind` chain of `n` operands, then return how many source lookups its backward made. */
async function backwardLookups(kind: "list" | "scalar", n: number): Promise<number> {
  const { flow, inputs } = await chainFlow(kind, n);
  const run = await runFlow(flow, inputs, {
    log: join(dir, `${kind}-${n}.jsonl`),
    learnable: true,
    memory: new Map([["test", new FakeMemory()]]),
    env: hostEnv(dir),
  });
  const lookups = spyOn(RetainedTrace.prototype, "lookupSource");
  try {
    const outcome = await run.backward("good");
    expect(outcome.status).toBe("succeeded");
    return lookups.mock.calls.length;
  } finally {
    lookups.mockRestore();
  }
}

test.each(["list", "scalar"] as const)("backward lookups grow near linearly: %s", async (kind) => {
  const small = await backwardLookups(kind, 32);
  const large = await backwardLookups(kind, 128);
  expect(small).toBeGreaterThan(0);
  expect(large / small).toBeLessThanOrEqual(7);
});
