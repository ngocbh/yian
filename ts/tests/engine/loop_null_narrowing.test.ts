/**
 * A LOOP `while:` null check and what enters the body: the loop-entry helpers of
 * `compose/nodes/loop.ts`, a port of rows of `tests/engine/test_loop_null_narrowing.py`.
 *
 * Ported: `test_while_proves_the_null_checked_paths` (9), `test_until_and_times_prove_nothing` (3)
 * and `test_body_inputs_keep_their_declared_types_after_load`; through `assembleText`, the load
 * halves of `test_a_tag_in_the_inline_body_admits_the_while_null_checked_seed`,
 * `test_inline_body_returns_null_into_a_literal_seed_under_a_while_null_check` and
 * `test_inline_body_returning_a_while_null_checked_field_unchanged_runs`, each checking the
 * inline body's generated parameters (a record seed's type is canonical, its typedef name gone).
 *
 * The seed check (the cross-flow binding check on a LOOP seed), through `assembleText`: the load
 * halves of `test_optional_seed_fills_a_strict_body_input_under_a_while_null_check`,
 * `test_every_seed_shape_is_narrowed_by_a_while_null_check` (3),
 * `test_while_null_check_admits_a_null_seed_the_body_output_can_carry` and
 * `test_nested_null_check_narrows_the_record_above_it`; both loads of
 * `test_while_null_check_lets_a_seed_record_omit_the_checked_field`;
 * `test_while_null_check_rejects_a_record_seed_the_zero_iteration_output_cannot_carry` and
 * `test_while_null_check_rejects_a_null_seed_the_zero_iteration_output_cannot_carry` (each also
 * pinned as its harvested case at the end, `assembledWith`),
 * `test_seed_of_the_wrong_type_names_the_declared_input_under_a_while_null_check`, both loads of
 * `test_partial_loop_seed_check_alone_decides_a_carried_over_field` and both loads of
 * `test_while_null_check_keeps_the_case_narrowed_seed`. The loop type contract (a
 * returned field keeps its carried type and fits the body input it is bound to next), through
 * `assembleText`: `test_until_null_check_narrows_neither_the_seed_nor_a_returned_field`,
 * `test_null_check_under_or_narrows_nothing`,
 * `test_returned_field_still_keeps_its_carried_type_under_a_while_null_check`,
 * `test_nested_null_check_keeps_the_carried_type_of_the_returned_record`,
 * `test_returned_field_of_the_wrong_type_names_the_declared_input_under_a_while_check` (also
 * pinned as its harvested case at the end, with
 * `load-types/engine_loop_null_narrowing__until_null_check_does_not_narrow_a_named_body`), the
 * load half of `test_returned_null_fits_a_strict_body_input_under_a_while_null_check`, and the
 * inline rows `test_inline_body_reads_a_while_null_checked_seed_as_its_seed_types_it`,
 * `test_inline_returned_field_of_the_wrong_type_names_the_seed_type` and
 * `test_inline_body_under_until_reads_the_seed_as_optional`. Each error row pins the whole
 * message, where Python matches a fragment of some.
 *
 * Changed: `whileNonNullPaths` returns the proven paths as distinct field-name lists in path order,
 * where Python returns a `frozenset` of tuples. The declared-types row parses the loop node
 * (`nodesOf`) and builds it over the body built by hand (`handChild`), where Python loads the
 * flow whose def is the body. Every other row's defs (`fetch_page`, `step`, `seed`, `body`) are
 * built by hand too, the scope resolving each by name. The record seed's message writes the value
 * and the missing fields in the text form (`{"n":0}`, `["next"]`, rule `messages`).
 *
 * TS-only: `nullableAt` on a record field, a path through a union, a missing field or a scalar,
 * and the null type; `loopEntryInputs` relaxing only the parameters under a `while:` path, keeping
 * their origins, and returning an unchanged parameter as is; `carriedFieldTypes` on an exact
 * record, a partly typed record (its opaque fields skipped) and anything else; `whileNonNullPaths`
 * giving a path checked twice once and in order, and nothing for a condition that doesn't parse.
 *
 * The runs, each loading the flow with the defs Python writes in it (`withDefs`) and running it
 * through `runFlow`: both runs of
 * `test_optional_seed_fills_a_strict_body_input_under_a_while_null_check`,
 * `test_every_seed_shape_is_narrowed_by_a_while_null_check` (3, every run),
 * `test_while_null_check_keeps_the_case_narrowed_seed` (both runs),
 * `test_returned_null_fits_a_strict_body_input_under_a_while_null_check`,
 * `test_partial_loop_seed_check_alone_decides_a_carried_over_field` (both runs),
 * `test_nested_null_check_narrows_the_record_above_it` (both runs),
 * `test_while_null_check_admits_a_null_seed_the_body_output_can_carry`,
 * `test_a_tag_in_the_inline_body_admits_the_while_null_checked_seed` (both runs),
 * `test_inline_body_returns_null_into_a_literal_seed_under_a_while_null_check` and
 * `test_inline_body_returning_a_while_null_checked_field_unchanged_runs` (flat, nested; every run).
 * The CODE bodies are TS, ints arriving as `number` (decision 45), a Python `assert` throwing an
 * `Error`; the runs' ints are bigints (decision 4a).
 *
 * Not ported: the run of `test_while_null_check_lets_a_seed_record_omit_the_checked_field`, covered
 * by `run-loop/engine_loop_null_narrowing__an_absent_checked_field_reads_as_null` (its events pin
 * that the body never starts).
 *
 * Later (T4-8): needs `resumeFlow`: `test_optional_seed_pagination_resumes_after_its_first_expansion`.
 */

import { afterEach, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { BuildContext } from "../../src/compose/nodes/base.ts";
import { carriedFieldTypes, loopEntryInputs, nullableAt, whileNonNullPaths } from "../../src/compose/nodes/loop.ts";
import { forget } from "../../src/defaults.ts";
import { NodeStarted } from "../../src/events.ts";
import { PartialRecordContract, ValueContract } from "../../src/nodes/contracts.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { type RunResult, runFlow } from "../../src/run.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { canonicalTypeLabel, Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembledWith,
  assembleText,
  type ChildFlows,
  handChild,
  loadConformanceFlow,
  loadError,
  nodesOf,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

afterEach(() => forget());

const STR = Type.scalar(ValueKind.STRING);
const INT = Type.scalar(ValueKind.INTEGER);

// ---------- the paths a while: proves not null ----------

test.each<[string, string[][]]>([
  ["cursor != null", [["cursor"]]],
  ["n < 3 and cursor != null", [["cursor"]]],
  ["not (cursor == null)", [["cursor"]]],
  ["page.next != null", [["page", "next"]]],
  ["${cursor != null}", [["cursor"]]],
  ["cursor != null or n > 3", []],
  ["isinstance(item, Item)", []],
  ["cursor != null and other != null", [["cursor"], ["other"]]],
  ["(a != null and b != null) or (a != null and n > 1)", [["a"]]],
])("while proves the null-checked paths: %s", (predicate, paths) => {
  expect(whileNonNullPaths("while", predicate)).toEqual(paths);
});

test.each<["until" | "times", string | null]>([
  ["until", "cursor == null"],
  ["until", "cursor != null"],
  ["times", null],
])("until and times prove nothing: %s %s", (kind, predicate) => {
  expect(whileNonNullPaths(kind, predicate)).toEqual([]);
});

test("a path checked twice is given once, the paths in order", () => {
  expect(whileNonNullPaths("while", "b != null and a.x != null and b != null and a != null")).toEqual([
    ["a"],
    ["a", "x"],
    ["b"],
  ]);
});

test("a while that doesn't parse proves nothing", () => {
  expect(whileNonNullPaths("while", "cursor != null and")).toEqual([]);
});

// ---------- the body's declared inputs ----------

// Pagination over the `fetch_page` def, which takes `cursor: str`.
const PAGE = `id: pages
name: pages
nodes:
  pages:
    kind: loop
    while: cursor != null
    input:
      cursor: \${input.start}
    call: fetch_page
`;

/** The parameters a def declares in `input` (the lines under its `input:` key). */
function params(input: string) {
  return parametersFromYaml(`input:\n${input}`, new Map());
}

test("body inputs keep their declared types after load", () => {
  // Only the checks read a null-checked input as nullable; the node keeps `str`, which the body's
  // start checks each value against.
  const body = handChild({
    inputs: params("  cursor: !str\n"),
    producers: { get: new ValueContract(STR.replace({ nullable: true })) },
    output: { cursor: "${get.output}" },
  });
  const desc = nodesOf(PAGE).get("pages")!;

  const built = desc.build(new BuildContext({ registry: new Map(), resolver: () => body })).node;
  expect(built).toBeInstanceOf(LoopNode);
  const node = built as LoopNode;

  const [cursor] = node.childInputs;
  expect(node.childInputs).toHaveLength(1);
  expect(canonicalTypeLabel(cursor!.contract!.type)).toBe("str");
  expect(cursor!.required).toBe(true);
  // The entry checks read it as nullable.
  expect(canonicalTypeLabel(loopEntryInputs(node)[0]!.contract!.type)).toBe("Optional(str)");
});

// ---------- what the seed check reads under a while null check ----------

/** The `fetch_page` def: `cursor: str` in, `{cursor: <its get node's output>}` out, `get` typed `get`. */
function pagesBody(get = "Optional(str)"): ComposedFlow {
  return handChild({
    inputs: params("  cursor: !str\n"),
    producers: { get: new ValueContract(typeFor(get, new Map())) },
    output: { cursor: "${get.output}" },
  });
}

/**
 * Pagination over the `fetch_page` def (`pagesBody`), `cursor` seeded with `seed` (each of
 * `carried`'s lines seeding another field), the loop driven by `driver`, the flow taking `inputs`.
 */
function page(
  options: { seed?: string; driver?: string; inputs?: string; carried?: string } = {},
): string {
  const {
    seed = "${input.start}",
    driver = "while: cursor != null",
    inputs = "start: !Optional(str)",
    carried = "",
  } = options;
  return `id: pages
name: pages
input:
  ${inputs}
nodes:
  pages:
    kind: loop
    ${driver}
    input:
      cursor: ${seed}${carried}
    call: fetch_page
output: \${pages.output}
`;
}

/** Assemble `text` with `children` built by hand (default: `fetch_page` as `pagesBody()`). */
function load(text: string, children: ChildFlows = { fetch_page: pagesBody() }): ComposedFlow {
  return assembleText(text, "flow.yaml", { children }).flow;
}

// ---------- the runs ----------

/**
 * `text` with `defs` (each entry indented under `defs:`) before its `nodes:`: the runs load the flow
 * with the defs Python writes in it, where the rows above build each def by hand.
 */
const withDefs = (text: string, defs: string) => text.replace(/^nodes:$/m, `defs:\n${defs}nodes:`);

/**
 * The runnable `fetch_page` def, `cursor: str` in, `output` out. Its CODE node fails on a null
 * cursor, so a run that succeeds shows the null never entered the body.
 */
const fetchPageDef = (output: string) => `  fetch_page:
    input:
      cursor: !str
    nodes:
      get:
        kind: code
        input:
          cursor: \${input.cursor}
        output: Optional(str)
        code: |
          if (inputs["cursor"] === null) throw new Error("body saw a null cursor")
          return ({p1: "p2", p2: null})[inputs["cursor"]] ?? null
    output: ${output}
`;

/** The named pagination body: it returns `{cursor: <get's output>}`. */
const FETCH_PAGE = fetchPageDef("\n      cursor: ${get.output}");

/** Count the body's CODE node runs in a finished run. */
function bodyRuns(result: RunResult): number {
  return result.events.filter((event) => event instanceof NodeStarted && event.nodeId.endsWith("/get")).length;
}

test("an optional seed fills a strict body input under a while null check", () => {
  expect(load(page()).compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);
});

test("optional seed fills a strict body input under a while null check", async () => {
  const flow = await loadConformanceFlow(withDefs(page(), FETCH_PAGE));

  for (const [start, runs] of [
    [null, 0],
    ["p1", 2],
  ] as const) {
    const result = await runFlow(flow, { start });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ cursor: null });
    expect(bodyRuns(result)).toBe(runs);
  }
});

test.each<[string, string]>([
  ["!Optional(str) ${input.start}", "start: !Optional(str)"],
  ["${input.a :- input.b}", "a: !Optional(str)\n  b: !Optional(str)"],
  ["null", "start: !Optional(str)"],
])("every seed shape is narrowed by a while null check: %s", (seed, inputs) => {
  expect(load(page({ seed, inputs })).compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);
});

test.each<[string, string, [Record<string, string | null>, number][]]>([
  [
    "!Optional(str) ${input.start}",
    "start: !Optional(str)",
    [
      [{ start: null }, 0],
      [{ start: "p1" }, 2],
    ],
  ],
  [
    "${input.a :- input.b}",
    "a: !Optional(str)\n  b: !Optional(str)",
    [
      [{ a: null, b: null }, 0],
      [{ a: null, b: "p1" }, 2],
      [{ a: "p2", b: null }, 1],
    ],
  ],
  ["null", "start: !Optional(str)", [[{ start: null }, 0]]],
])("every seed shape is narrowed by a while null check (runs): %s", async (seed, inputs, runs) => {
  const flow = await loadConformanceFlow(withDefs(page({ seed, inputs }), FETCH_PAGE));

  for (const [flowInputs, count] of runs) {
    const result = await runFlow(flow, flowInputs);
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ cursor: null });
    expect(bodyRuns(result)).toBe(count);
  }
});

test("a while null check admits a null seed the body output can carry", () => {
  expect(load(page({ seed: "null" })).compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);
});

test("while null check admits a null seed the body output can carry", async () => {
  const result = await runFlow(await loadConformanceFlow(withDefs(page({ seed: "null" }), FETCH_PAGE)), {
    start: null,
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ cursor: null });
  expect(bodyRuns(result)).toBe(0);
});

test("a while null check rejects a null seed the zero-iteration output cannot carry", () => {
  // With zero iterations the loop outputs the null seed as the body's `str` output.
  const error = loadError(() => load(page({ seed: "null" }), { fetch_page: pagesBody("str") }));

  expect(error.message).toBe(
    "loop node 'pages': binding 'cursor' — child expects 'str', but the value null does not fit",
  );
});

test("a seed of the wrong type names the declared input under a while null check", () => {
  const error = loadError(() => load(page({ seed: "${input.n}", inputs: "n: !Optional(int)" })));

  expect(error.message).toBe(
    "loop node 'pages': binding 'cursor' — child expects 'str', source is 'Optional(int)'",
  );
});

const SEED_MESSAGE = "loop node 'pages': binding 'cursor' — child expects 'str', source is 'Optional(str)'";
const RETURNED_MESSAGE =
  "loop node 'pages': the body 'fetch_page' returns 'cursor' as Optional(str), but takes 'cursor' as str; " +
  'make the input Optional(str), or tag the seed to leave the check to runtime: `cursor: !str "p1"`';

test("a returned null fits a strict body input under a while null check (loads)", () => {
  expect(load(page({ seed: "p1" })).compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);
});

test("returned null fits a strict body input under a while null check", async () => {
  const result = await runFlow(await loadConformanceFlow(withDefs(page({ seed: "p1" }), FETCH_PAGE)), {
    start: null,
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ cursor: null });
  expect(bodyRuns(result)).toBe(2);
});

test("an until null check narrows neither the seed nor a returned field", () => {
  const driver = "until: cursor == null";

  expect(loadError(() => load(page({ driver }))).message).toBe(SEED_MESSAGE);
  expect(loadError(() => load(page({ seed: "p1", driver }))).message).toBe(RETURNED_MESSAGE);
});

test("a null check under or narrows nothing", () => {
  const driver = "while: cursor != null or n > 3";
  const carried = "\n      n: 0";

  expect(loadError(() => load(page({ driver, carried }))).message).toBe(SEED_MESSAGE);
  expect(loadError(() => load(page({ seed: "p1", driver, carried }))).message).toBe(RETURNED_MESSAGE);
});

test("a returned field still keeps its carried type under a while null check", () => {
  // With zero iterations the loop outputs the seed, so a body returning `str` can't carry an
  // `Optional(str)` seed even though the `while:` keeps a null out of the body.
  const text = `id: so
name: so
input:
  start: !Optional(str)
nodes:
  pages:
    kind: loop
    while: cursor != null and n < 2
    input:
      cursor: \${input.start}
      n: 0
    call: step
output: \${pages.output}
`;
  const step = handChild({
    inputs: params("  cursor: !str\n  n: !int\n"),
    producers: { get: new ValueContract(recordType({ cursor: STR, n: INT })) },
    output: "${get.output}",
  });

  const error = loadError(() => load(text, { step }));

  expect(error.message).toBe(
    "loop node 'pages': carried field 'cursor' — body output is 'str', carried source is 'Optional(str)' " +
      "(a returned field must keep its carried type)",
  );
});

test("a returned field of the wrong type names the declared input under a while check", () => {
  const error = loadError(() => load(page({ seed: "p1" }), { fetch_page: pagesBody("Optional(int)") }));

  expect(error.message).toBe(
    "loop node 'pages': the body 'fetch_page' returns 'cursor' as Optional(int), but takes 'cursor' as str; " +
      "change the body's input or output type so they agree",
  );
});

/**
 * A partial loop driven by `driver`: the `step` def takes `cursor` but returns only `n`, so `cursor`
 * carries over.
 */
function partial(driver: string): string {
  return `id: p
name: p
input:
  start: !Optional(str)
nodes:
  loop:
    kind: loop
    ${driver}
    input:
      cursor: \${input.start}
      n: 0
    call: step
output: \${loop.output}
`;
}

test("a partial loop's seed check alone decides a carried-over field", () => {
  const step = handChild({
    inputs: params("  cursor: !str\n  n: !int\n"),
    producers: { get: new ValueContract(recordType({ n: INT })) },
    output: "${get.output}",
  });

  expect(load(partial("while: cursor != null and n < 2"), { step }).compiled.nodes.get("loop")).toBeInstanceOf(
    LoopNode,
  );

  // No `while:` null check on `cursor`, so its `Optional(str)` seed can't fill the `str` input.
  const error = loadError(() => load(partial("while: n < 2"), { step }));

  expect(error.message).toBe("loop node 'loop': binding 'cursor' — child expects 'str', source is 'Optional(str)'");
});

test("partial loop seed check alone decides a carried over field", async () => {
  const step = `  step:
    input:
      cursor: !str
      n: !int
    nodes:
      get:
        kind: code
        input:
          cursor: \${input.cursor}
          n: \${input.n}
        output:
          n: int
        code: |
          if (inputs["cursor"] === null) throw new Error("assert")
          return {n: inputs["n"] + 1}
    output: \${get.output}
`;
  const flow = await loadConformanceFlow(withDefs(partial("while: cursor != null and n < 2"), step));

  for (const [start, n] of [
    [null, 0n],
    ["c", 2n],
  ] as const) {
    const result = await runFlow(flow, { start });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ cursor: start, n });
  }
});

/** The `Page` (`next` optional) and `Strict` (`next` required) records the paging rows read. */
const PAGES = readTypedefs({ Page: { next: "Optional(str)", n: "int" }, Strict: { next: "str", n: "int" } });

/** The `step` def: a `Strict` page in, its `get` node's `{page: <returned>}` out. */
function strictStep(returned: string): ComposedFlow {
  return handChild({
    inputs: parametersFromYaml("input:\n  page: !Strict\n", PAGES),
    producers: { get: new ValueContract(recordType({ page: typeFor(returned, PAGES) })) },
    output: "${get.output}",
  });
}

// `page.next != null` proves both `page` and `page.next` not null, so the `seed` def's
// `Optional(Page)` output fills the `step` def's `Strict` input.
const NESTED = `id: nl
name: nl
typedefs:
  Page:
    next: Optional(str)
    n: int
  Strict:
    next: str
    n: int
input:
  start: !Optional(str)
nodes:
  first:
    kind: call
    call: seed
    input:
      s: \${input.start}
  pages:
    kind: loop
    while: page.next != null
    input:
      page: \${first.output}
    call: step
output: \${pages.output}
`;

/** The `seed` def: `s: Optional(str)` in, its `mk` node's `Optional(Page)` out. */
function seedDef(): ComposedFlow {
  return handChild({
    inputs: params("  s: !Optional(str)\n"),
    producers: { mk: new ValueContract(typeFor("Optional(Page)", PAGES)) },
    output: "${mk.output}",
  });
}

test("a nested null check narrows the record above it", () => {
  const flow = load(NESTED, { step: strictStep("Optional(Page)"), seed: seedDef() });

  expect(flow.compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);
});

test("nested null check narrows the record above it", async () => {
  const defs = `  step:
    input:
      page: !Strict
    nodes:
      get:
        kind: code
        input:
          page: \${input.page}
        output:
          page: Optional(Page)
        code: |
          if (inputs["page"]["next"] === null) throw new Error("assert")
          const n = inputs["page"]["n"] + 1
          return {page: {next: n >= 2 ? null : "c", n}}
    output: \${get.output}
  seed:
    input:
      s: !Optional(str)
    nodes:
      mk:
        kind: code
        input:
          s: \${input.s}
        output: Optional(Page)
        code: |
          return inputs["s"] === null ? null : {next: inputs["s"], n: 0}
    output: \${mk.output}
`;
  const flow = await loadConformanceFlow(withDefs(NESTED, defs));

  for (const [start, output] of [
    [null, { page: null }],
    ["c", { page: { next: null, n: 2n } }],
  ] as const) {
    const result = await runFlow(flow, { start });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual(output);
  }
});

test("a nested null check keeps the carried type of the returned record", () => {
  const error = loadError(() => load(NESTED, { step: strictStep("Page"), seed: seedDef() }));

  expect(error.message).toBe(
    "loop node 'pages': carried field 'page' — body output is '{\"n\":int,\"next\":Optional(str)}', " +
      "carried source is 'Optional({\"n\":int,\"next\":Optional(str)})' (a returned field must keep its carried type)",
  );
});

/** A loop over the `step` def whose literal seed `{n: 0}` has no `next`, driven by `driver`. */
function missingNext(driver: string): string {
  return `id: mn
name: mn
typedefs:
  Page:
    next: Optional(str)
    n: int
  Strict:
    next: str
    n: int
nodes:
  pages:
    kind: loop
    ${driver}
    input:
      page: {n: 0}
    call: step
output: \${pages.output}
`;
}

test("a while null check lets a seed record omit the checked field", () => {
  // An absent `next` reads as null, so the `while:` never lets the record into the body.
  const step = strictStep("Optional(Page)");
  expect(load(missingNext("while: page.next != null"), { step }).compiled.nodes.get("pages")).toBeInstanceOf(LoopNode);

  // `until:` is checked after the body, so it narrows nothing.
  const error = loadError(() => load(missingNext("until: page.next == null"), { step }));

  expect(error.message).toBe(
    "loop node 'pages': binding 'page' — child expects '{\"n\":int,\"next\":str}', " +
      'but the value {"n":0} does not fit (record missing required fields: ["next"])',
  );
});

test("a while null check rejects a record seed the zero-iteration output cannot carry", () => {
  // With zero iterations the loop outputs `{n: 0}` as the body's `Optional(Strict)` output, whose
  // `next` is required, so the `while:` can't admit the seed.
  const error = loadError(() =>
    load(missingNext("while: page.next != null"), { step: strictStep("Optional(Strict)") }),
  );

  expect(error.message).toBe(
    "loop node 'pages': binding 'page' — child expects '{\"n\":int,\"next\":str}', " +
      'but the value {"n":0} does not fit (record missing required fields: ["next"])',
  );
});

// A LOOP routed by a `case` on a `Union(Buy, Hold)` producer: the seed `order` is narrowed to `Buy`
// only under `when: buy`, and `cursor` only by the `while:`.
const ROUTED_TYPEDEFS = readTypedefs({
  Buy: { kind: "Literal(buy)", qty: "int" },
  Hold: { kind: "Literal(hold)", note: "str" },
});

/** The routed flow, the CASE sending `when: buy` to `then` and anything else to `other`. */
function routed(then: string, other: string): string {
  return `id: p
name: p
typedefs:
  Buy: {kind: 'Literal(buy)', qty: int}
  Hold: {kind: 'Literal(hold)', note: str}
input:
  s: !str
  start: !Optional(str)
nodes:
  d:
    kind: code
    input:
      s: \${input.s}
    output: Union(Buy, Hold)
    code: 'return inputs["s"] === "b" ? {kind: "buy", qty: 2n} : {kind: "hold", note: "n"}'
  route:
    kind: case
    on: \${d.output.kind}
    cases:
      - when: buy
        then: ${then}
    else: ${other}
  loop:
    kind: loop
    while: cursor != null
    input:
      order: \${d.output}
      cursor: \${input.start}
      k: 0
    call: body
  e:
    kind: code
    code: return 1n
output: \${loop.output.k}
`;
}

test("a while null check keeps the case-narrowed seed", () => {
  const body = handChild({
    inputs: parametersFromYaml("input:\n  order: !Buy\n  cursor: !str\n  k: !int\n", ROUTED_TYPEDEFS),
    producers: { step: new ValueContract(recordType({ cursor: STR.replace({ nullable: true }), k: INT })) },
    output: "${step.output}",
  });

  load(routed("loop", "e"), { body });

  // Under `else:` the seed is narrowed to `Hold`, not `Buy`.
  const error = loadError(() => load(routed("e", "loop"), { body }));

  expect(error.message).toBe(
    "loop node 'loop': binding 'order' — child expects '{\"kind\":Literal(buy),\"qty\":int}', " +
      "source is '{\"kind\":Literal(hold),\"note\":str}'",
  );
});

test("while null check keeps the case narrowed seed", async () => {
  const body = `  body:
    input:
      order: !Buy
      cursor: !str
      k: !int
    nodes:
      step:
        kind: code
        input:
          order: \${input.order}
          cursor: \${input.cursor}
          k: \${input.k}
        output:
          cursor: Optional(str)
          k: int
        code: |
          if (inputs["cursor"] === null) throw new Error("assert")
          return {cursor: null, k: inputs["k"] + inputs["order"]["qty"]}
    output: \${step.output}
`;
  const flow = await loadConformanceFlow(withDefs(routed("loop", "e"), body));

  for (const [start, k] of [
    ["c1", 2n],
    [null, 0n],
  ] as const) {
    const result = await runFlow(flow, { s: "b", start });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toBe(k);
  }
});

// ---------- inline bodies ----------

/**
 * Pagination over an inline body whose CALL node passes `cursor` (as `argument`) to the `fetch_page`
 * def's strict `str` input, `cursor` seeded with `seed`, the loop driven by `driver`.
 */
function inlinePages(options: { seed?: string; driver?: string; argument?: string } = {}): string {
  const { seed = "${input.start}", driver = "while: cursor != null", argument = "${input.cursor}" } = options;
  return `id: pages
name: pages
input:
  start: !Optional(str)
nodes:
  pages:
    kind: loop
    ${driver}
    input:
      cursor: ${seed}
    nodes:
      fetch:
        kind: call
        call: fetch_page
        input:
          cursor: ${argument}
    output:
      cursor: \${fetch.output}
output: \${pages.output}
`;
}

/** The `fetch_page` def: `cursor: str` in, its `get` node's `returned` out. */
function fetchPage(returned = "Optional(str)"): ComposedFlow {
  return handChild({
    inputs: params("  cursor: !str\n"),
    producers: { get: new ValueContract(typeFor(returned, new Map())) },
    output: "${get.output}",
  });
}

/** Assemble the inline pagination flow `text` over `fetchPage(returned)`. */
function loadInline(text: string, returned?: string): ComposedFlow {
  return load(text, { fetch_page: fetchPage(returned) });
}

/** The parameters of the inline body of LOOP `pages` in `flow`. */
function inlineInputs(flow: ComposedFlow): [string, string, boolean][] {
  const node = flow.compiled.nodes.get("pages") as LoopNode;
  return node.childInputs.map((param) => [param.name, canonicalTypeLabel(param.contract!.type), param.required]);
}

test("a tag in the inline body admits the while-null-checked seed", () => {
  const flow = loadInline(inlinePages({ argument: "!str ${input.cursor}" }));

  expect(inlineInputs(flow)).toEqual([["cursor", "Optional(str)", false]]);
});

/** The inline pagination body's runnable `fetch_page` def: it returns its `get` node's output. */
const INLINE_FETCH_PAGE = fetchPageDef("${get.output}");

test("a tag in the inline body admits the while null checked seed", async () => {
  const flow = await loadConformanceFlow(
    withDefs(inlinePages({ argument: "!str ${input.cursor}" }), INLINE_FETCH_PAGE),
  );

  for (const [start, runs] of [
    [null, 0],
    ["p1", 2],
  ] as const) {
    const result = await runFlow(flow, { start });
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual({ cursor: null });
    expect(bodyRuns(result)).toBe(runs);
  }
});

test("an inline body reads a while-null-checked seed as its seed types it", () => {
  // The body's generated input is `Optional(str)` from the seed, and the `while:` does not change
  // it, so a strict CALL argument in the body still rejects it.
  expect(loadError(() => loadInline(inlinePages())).message).toBe(
    "call node 'fetch': binding 'cursor' — child expects 'str', source is 'Optional(str)'",
  );
});

test("an inline body may return null into a literal seed under a while null check", () => {
  const flow = loadInline(inlinePages({ seed: "p1" }));

  expect(inlineInputs(flow)).toEqual([["cursor", "str", true]]);
});

test("inline body returns null into a literal seed under a while null check", async () => {
  const flow = await loadConformanceFlow(withDefs(inlinePages({ seed: "p1" }), INLINE_FETCH_PAGE));

  const result = await runFlow(flow, { start: null });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual({ cursor: null });
  expect(bodyRuns(result)).toBe(2);
});

test("an inline returned field of the wrong type names the seed type", () => {
  expect(loadError(() => loadInline(inlinePages({ seed: "p1" }), "Optional(int)")).message).toBe(
    "loop node 'pages': the body returns 'cursor' as Optional(int), but the carried field is typed str " +
      "from its seed; declare the seed's type: `cursor: !Optional(int) \"p1\"`",
  );
});

test("an inline body under until reads the seed as optional", () => {
  const driver = "until: cursor == null";

  expect(loadError(() => loadInline(inlinePages({ driver }))).message).toBe(
    "call node 'fetch': binding 'cursor' — child expects 'str', source is 'Optional(str)'",
  );
  expect(loadError(() => loadInline(inlinePages({ seed: "p1", driver }))).message).toBe(
    "loop node 'pages': the body returns 'cursor' as Optional(str), but the carried field is typed str " +
      "from its seed; declare the seed's type: `cursor: !Optional(str) \"p1\"`",
  );
});

// Inline bodies that return a null-checked field unchanged, flat and nested in a record.
const PASS_THROUGH = `id: p
name: p
input:
  tok: !Optional(str)
nodes:
  pages:
    kind: loop
    while: token != null and n < 2
    max: 5
    input:
      token: \${input.tok}
      n: 0
    nodes:
      step:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: return inputs["n"] + 1
    output:
      token: \${input.token}
      n: \${step.output}
output: \${pages.output}
`;
const NESTED_PASS_THROUGH = `id: p
name: p
typedefs:
  Page:
    next: Optional(str)
    n: int
input:
  start: !Optional(Page)
nodes:
  pages:
    kind: loop
    while: page.next != null and k < 2
    max: 5
    input:
      page: \${input.start}
      k: 0
    nodes:
      step:
        kind: code
        input:
          k: \${input.k}
        output: int
        code: return inputs["k"] + 1
    output:
      page: \${input.page}
      k: \${step.output}
output: \${pages.output}
`;

test.each<[string, string, [string, string, boolean][]]>([
  [
    "flat",
    PASS_THROUGH,
    [
      ["token", "Optional(str)", false],
      ["n", "int", true],
    ],
  ],
  [
    "nested",
    NESTED_PASS_THROUGH,
    [
      ["page", 'Optional({"n":int,"next":Optional(str)})', false],
      ["k", "int", true],
    ],
  ],
])("an inline body returning a while-null-checked field unchanged loads: %s", (_name, source, inputs) => {
  expect(inlineInputs(assembleText(source).flow)).toEqual(inputs);
});

test.each<[string, string, [Record<string, unknown>, unknown][]]>([
  [
    "flat",
    PASS_THROUGH,
    [
      [{ tok: null }, { token: null, n: 0n }],
      [{ tok: "t" }, { token: "t", n: 2n }],
    ],
  ],
  [
    "nested",
    NESTED_PASS_THROUGH,
    [
      [{ start: null }, { page: null, k: 0n }],
      [{ start: { next: "x", n: 0n } }, { page: { next: "x", n: 0n }, k: 2n }],
    ],
  ],
])("inline body returning a while null checked field unchanged runs: %s", async (_name, source, runs) => {
  const flow = await loadConformanceFlow(source);

  for (const [flowInputs, output] of runs) {
    const result = await runFlow(flow, flowInputs);
    expect(result.status, result.failure?.message).toBe("succeeded");
    expect(result.output).toEqual(output);
  }
});

// ---------- nullableAt ----------

/** A record of `fields`, each required. */
function recordType(fields: Record<string, Type>): Type {
  return new Type({
    kind: ValueKind.OBJECT,
    fields: new Map(Object.entries(fields)),
    required: new Set(Object.keys(fields)),
  });
}

test("nullableAt makes the value at a path and each record above it nullable", () => {
  const page = recordType({ next: STR, n: INT });

  const relaxed = nullableAt(page, ["next"]);

  expect(relaxed).toStrictEqual(
    new Type({
      kind: ValueKind.OBJECT,
      fields: new Map([
        ["next", STR.replace({ nullable: true })],
        ["n", INT],
      ]),
      required: new Set(["n"]),
      nullable: true,
    }),
  );
  expect(canonicalTypeLabel(relaxed)).toBe('Optional({"n":int,"next":Optional(str)})');
  // The type itself is not changed.
  expect(page.nullable).toBe(false);
  expect(page.required).toEqual(new Set(["next", "n"]));
});

test("nullableAt follows a path through nested records", () => {
  const outer = recordType({ page: recordType({ next: STR }) });

  const relaxed = nullableAt(outer, ["page", "next"]);

  expect(relaxed.nullable).toBe(true);
  expect(relaxed.required).toEqual(new Set());
  const page = relaxed.fields!.get("page")!;
  expect(page.nullable).toBe(true);
  expect(page.required).toEqual(new Set());
  expect(page.fields!.get("next")).toStrictEqual(STR.replace({ nullable: true }));
});

test("nullableAt stops at a union and still makes it nullable", () => {
  const registry = readTypedefs({ Done: { answer: "str" }, More: { hint: "str" }, Step: "Union(Done, More)" });
  const step = typeFor("Step", registry);
  const holder = recordType({ r: step });

  const relaxed = nullableAt(holder, ["r", "answer"]);

  // The union's members are left as they are.
  expect(relaxed.fields!.get("r")).toStrictEqual(step.replace({ nullable: true }));
  expect(relaxed.required).toEqual(new Set());
  expect(relaxed.nullable).toBe(true);
});

test("nullableAt stops at a missing field and still makes the record nullable", () => {
  const page = recordType({ next: STR });

  const relaxed = nullableAt(page, ["other", "x"]);

  expect(relaxed).toStrictEqual(page.replace({ nullable: true }));
});

test("nullableAt stops at a scalar and still makes it nullable", () => {
  expect(nullableAt(STR, ["x"])).toStrictEqual(STR.replace({ nullable: true }));
  expect(nullableAt(STR, [])).toStrictEqual(STR.replace({ nullable: true }));
});

test("nullableAt keeps the null type as it is", () => {
  const none = Type.scalar(ValueKind.NONE);

  expect(nullableAt(none, [])).toBe(none);
  expect(nullableAt(recordType({ x: none }), ["x"]).fields!.get("x")).toBe(none);
});

// ---------- loopEntryInputs ----------

/** A LOOP node over a body taking `cursor: str`, `page: {next: str}` and `n: int`, driven by `kind`. */
function pagesLoop(kind: "while" | "until" | "times", predicate: string | null): LoopNode {
  return new LoopNode("pages", {
    flowId: "fetch_page",
    childInputs: parametersFromYaml(
      "input:\n  cursor: !str\n  page: !Page\n  n: !int\n",
      readTypedefs({ Page: { next: "str" } }),
    ),
    predicateKind: kind,
    predicate,
  });
}

test("loopEntryInputs relaxes only the parameters under a while path", () => {
  const node = pagesLoop("while", "cursor != null and page.next != null and n < 3");
  const [cursor, page, n] = node.childInputs;

  const inputs = loopEntryInputs(node);

  expect(inputs.map((parameter) => parameter.name)).toEqual(["cursor", "page", "n"]);
  expect(canonicalTypeLabel(inputs[0]!.contract!.type)).toBe("Optional(str)");
  expect(canonicalTypeLabel(inputs[1]!.contract!.type)).toBe('Optional({"next":Optional(str)})');
  // An input under no path is the same parameter.
  expect(inputs[2]).toBe(n!);
  // A relaxed copy keeps its origins and its other fields.
  for (const [relaxed, declared] of [
    [inputs[0]!, cursor!],
    [inputs[1]!, page!],
  ] as const) {
    expect(relaxed).not.toBe(declared);
    expect(relaxed.origin).not.toBeNull();
    expect(relaxed.origin).toBe(declared.origin);
    expect(relaxed.contract!.origins).toBe(declared.contract!.origins);
    expect(relaxed.required).toBe(declared.required);
  }
  // The node's inputs stay as declared.
  expect(canonicalTypeLabel(node.childInputs[0]!.contract!.type)).toBe("str");
});

test.each<["until" | "times", string | null]>([
  ["until", "cursor != null"],
  ["times", null],
])("loopEntryInputs changes nothing under %s", (kind, predicate) => {
  const node = pagesLoop(kind, predicate);

  const inputs = loopEntryInputs(node);

  expect(inputs).toHaveLength(3);
  inputs.forEach((parameter, i) => expect(parameter).toBe(node.childInputs[i]!));
});

// ---------- carriedFieldTypes ----------

test("carriedFieldTypes gives an exact record's field types", () => {
  const contract = new ValueContract(recordType({ n: INT, cursor: STR }));

  const types = carriedFieldTypes(contract);

  expect(types).toStrictEqual(
    new Map([
      ["n", INT],
      ["cursor", STR],
    ]),
  );
  expect([...types.keys()]).toEqual(["n", "cursor"]);
});

test("carriedFieldTypes gives a partly typed record's proven field types", () => {
  const contract = new PartialRecordContract([
    ["n", new ValueContract(INT)],
    ["keep", null],
    ["cursor", new ValueContract(STR)],
  ]);

  const types = carriedFieldTypes(contract);

  expect([...types.keys()]).toEqual(["n", "cursor"]);
  expect(types.get("n")).toBe(INT);
  expect(types.get("cursor")).toBe(STR);
});

test("carriedFieldTypes proves no field of anything else", () => {
  expect(carriedFieldTypes(null)).toEqual(new Map());
  expect(carriedFieldTypes(new ValueContract(STR))).toEqual(new Map());
});

// ---------- harvested conformance cases ----------

test("assembled: load-types/engine_loop_null_narrowing__returned_field_of_the_wrong_type_names_the_input", () => {
  runCase({
    id: "load-types/engine_loop_null_narrowing__returned_field_of_the_wrong_type_names_the_input",
    call: assembledWith({ fetch_page: pagesBody("Optional(int)") }),
  });
});

test("assembled: load-types/engine_loop_null_narrowing__until_null_check_does_not_narrow_a_named_body", () => {
  runCase({
    id: "load-types/engine_loop_null_narrowing__until_null_check_does_not_narrow_a_named_body",
    call: assembledWith({ fetch_page: pagesBody() }),
  });
});

test("assembled: load-types/engine_loop_null_narrowing__while_null_check_rejects_a_null_seed", () => {
  runCase({
    id: "load-types/engine_loop_null_narrowing__while_null_check_rejects_a_null_seed",
    call: assembledWith({ fetch_page: pagesBody("str") }),
  });
});

test("assembled: load-types/engine_loop_null_narrowing__while_null_check_rejects_a_record_seed", () => {
  runCase({
    id: "load-types/engine_loop_null_narrowing__while_null_check_rejects_a_record_seed",
    call: assembledWith({ step: strictStep("Optional(Strict)") }),
  });
});
