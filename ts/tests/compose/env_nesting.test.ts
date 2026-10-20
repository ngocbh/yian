/**
 * An `env:` value nested past the limit, holding itself through a recursive alias, or holding
 * a lone surrogate is a located load error: a port of `tests/compose/test_env_nesting.py`, through
 * `loadFlow` at each site (flow, node, multi-node and single-node `defs:` entries, compact flow),
 * and TS-only through `checkEnvValues` over a parsed document's flow `env:` and the written-value
 * check `checkWrittenValue`.
 *
 * Each `env:` value is checked with the counting the run-time check applies to a value,
 * whether YAML text or aliases built it, so a 1000-link alias chain fails (under a raised
 * node limit: at the default, the chain's anchors exceed it first). A value at the limit
 * passes. A value holding itself reads `<prefix>env '<key>' at <path>: cannot descend through
 * a recursive alias occurrence`, where `<path>` names the first entry that refers back to a
 * container enclosing it. The depth check runs first, so a cyclic value nested past the limit
 * gets the depth message.
 *
 * Changed: text or a record key holding a lone surrogate is refused too, after the depth check
 * and before the cycle check (`K: &a ["\uD800", *a]` reports the surrogate at `[0]`, where
 * Python loads the text and reports the cycle at `[1]`).
 *
 * Ported through `loadFlow`: `test_env_value_over_a_1000_link_alias_chain_is_a_located_load_error`,
 * `test_flow_env_value_past_the_limit_as_text_is_a_located_load_error`,
 * `test_env_value_follows_the_configured_limit`,
 * `test_flow_env_nesting_is_checked_before_max_tool_iterations`,
 * `test_env_value_holding_itself_is_a_located_load_error`,
 * `test_env_mapping_holding_itself_is_a_located_load_error`, `test_env_scalar_value_loads`, and the
 * load halves of `test_env_value_at_the_limit_loads_and_runs` and
 * `test_env_value_repeating_an_alias_loads_and_runs`. The CODE reference is TS's
 * (`./fixtures/compose_codefns.ts:echo`). The other rows are ported through `checkEnvValues`.
 *
 * Changed: Python's check that no `RecursionError` is in the error's chain is left out.
 *
 * Not ported: the `!!pairs` row of `test_flow_env_cycle_names_the_entry_that_refers_back`
 * (TS refuses `!!pairs`).
 *
 * Later (T4-8): the run halves of `test_env_value_at_the_limit_loads_and_runs` and
 * `test_env_value_repeating_an_alias_loads_and_runs`.
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { LoadDiagnostics, LoadError } from "../../src/compose/helpers/errors.ts";
import { checkEnvValues, checkWrittenValue } from "../../src/compose/helpers/value_nesting.ts";
import { SourceDocument } from "../../src/compose/source/index.ts";
import { BUILTIN, install } from "../../src/defaults.ts";
import { SourceRange } from "../../src/diagnostics.ts";
import { TypeCheckError, type Value } from "../../src/typesys/values.ts";
import { loadConformanceFlow, loadErrorAsync } from "./declarations.ts";

const MAX_VALUE_NESTING = BUILTIN.maxValueNesting;
const LIMIT = `value nests deeper than ${MAX_VALUE_NESTING} levels`;
const CYCLE = "cannot descend through a recursive alias occurrence";
const SURROGATE = "text holding a lone surrogate is not a yian value";

// A listed chain's anchors cost quadratically many nodes: a 1000-link chain holds over
// 500,000, past the default node limit. These tests exercise the nesting checks behind it.
beforeEach(() => install({ ...BUILTIN, maxYamlNodes: 10_000_000 }));
afterEach(() => install(BUILTIN));

// Value forms: each returns `[header, value]`, where `header` holds the top-level
// `x-anchors:` line the value's aliases name (empty for written-out text).

/** Return a list written out as text, nesting `levels` levels. */
function list(levels: number): [string, string] {
  return ["", "[".repeat(levels) + "1" + "]".repeat(levels)];
}

/** Return an `x-anchors:` line whose anchor `a<i>` names a value nesting i+1 levels. */
function anchors(count: number, record: boolean): string {
  const items = [record ? "&a0 {a: 1}" : "&a0 [1]"];
  for (let i = 1; i < count; i++) items.push(record ? `&a${i} {a: *a${i - 1}}` : `&a${i} [*a${i - 1}]`);
  return `x-anchors: [${items.join(", ")}]\n`;
}

/** Return a one-item list over an alias chain, nesting exactly `levels` levels. */
function listAlias(levels: number): [string, string] {
  return [anchors(levels - 1, false), `[*a${levels - 2}]`];
}

/** Return a one-key mapping over an alias chain, nesting exactly `levels` levels. */
function recordAlias(levels: number): [string, string] {
  return [anchors(levels - 1, true), `{a: *a${levels - 2}}`];
}

/** Return a one-item list over an alias chain ending in a self-alias, `levels` deep. */
function cyclicListAlias(levels: number): [string, string] {
  const items = ["&a0 [*a0]"];
  for (let i = 1; i < levels - 1; i++) items.push(`&a${i} [*a${i - 1}]`);
  return [`x-anchors: [${items.join(", ")}]\n`, `[*a${levels - 2}]`];
}

/** Return a flow whose flow `env:` sets `K` to `value`. */
function flowEnv([header, value]: [string, string]): string {
  return `${header}id: f\nname: f\nenv:\n  K: ${value}\n`;
}

/** Return a flow whose flow `env:` is the one-line mapping `env`. */
function flowEnvInline(env: string): string {
  return `id: f\nname: f\nenv: ${env}\n`;
}

/** Check the flow `env:` of `text` as the loader does. */
function checkFlowEnv(text: string): void {
  const document = SourceDocument.parse(text);
  const root = document.root;
  checkEnvValues(root.mapping("env"), "flow ", root.child("env"), LoadDiagnostics.fromDocument(document));
}

/** Return the 1-based number of the only line of `text` starting with `prefix`. */
function lineOf(text: string, prefix: string): number {
  const numbers = text
    .split("\n")
    .flatMap((line, index) => (line.startsWith(prefix) ? [index + 1] : []));
  expect(numbers).toHaveLength(1);
  return numbers[0]!;
}

/** Return the `LoadError` `call()` throws. */
function loadError(call: () => unknown): LoadError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(LoadError);
    return error as LoadError;
  }
  throw new Error("expected a LoadError");
}

/** Return the error checking `text`'s flow `env:` throws and the line its one frame starts on. */
function envError(text: string): { message: string; line: number } {
  const { diagnostic } = loadError(() => checkFlowEnv(text));
  expect(diagnostic.code).toBe("compose.load");
  expect(diagnostic.frames).toHaveLength(1);
  return { message: diagnostic.message, line: diagnostic.frames[0]!.range.startLine };
}

/** Assert checking `text` fails with the limit error at the `  K:` line. */
function expectLimitError(text: string, limit = LIMIT): void {
  const { message, line } = envError(text);
  expect(message.startsWith("flow env 'K' at ")).toBe(true);
  expect(message).toContain(limit);
  expect(line).toBe(lineOf(text, "  K:"));
}

/** Assert checking `text` fails with exactly `<start>: <reason>` at the line `linePrefix`. */
function expectError(text: string, start: string, reason: string, linePrefix = "  K:"): void {
  const { message, line } = envError(text);
  expect(message).toBe(`${start}: ${reason}`);
  expect(line).toBe(lineOf(text, linePrefix));
}

describe("nesting", () => {
  test.each([
    ["list", listAlias],
    ["record", recordAlias],
  ])("an env value over a 1000-link %s alias chain is a located load error", (_, form) => {
    expectLimitError(flowEnv(form(1000)));
  });

  test("an env value past the limit as text is a located load error", () => {
    expectLimitError(flowEnv(list(MAX_VALUE_NESTING + 1)));
  });

  test.each([
    ["text", list],
    ["alias", listAlias],
  ])("an env value at the limit as %s passes", (_, form) => {
    checkFlowEnv(flowEnv(form(MAX_VALUE_NESTING)));
  });

  test("an env value follows the configured limit", () => {
    install({ ...BUILTIN, maxYamlNodes: 10_000_000, maxValueNesting: 5 });
    checkFlowEnv(flowEnv(list(5)));
    expectLimitError(flowEnv(list(6)), "value nests deeper than 5 levels");
  });

  test("a null cursor gives an unlocated error", () => {
    let deep: Value = 1n;
    for (let i = 0; i < MAX_VALUE_NESTING + 1; i++) deep = [deep];
    const error = loadError(() => checkEnvValues({ K: deep }, "flow ", null, new LoadDiagnostics()));
    expect(error.diagnostic.message.startsWith("flow env 'K' at ")).toBe(true);
    expect(error.diagnostic.frames).toHaveLength(0);
  });

  test("checkWrittenValue counts the enclosing levels", () => {
    let value: Value = 1n;
    for (let i = 0; i < MAX_VALUE_NESTING; i++) value = [value];
    const range = new SourceRange(1, 1, 1, 2);
    checkWrittenValue(value, "out", range, new LoadDiagnostics());
    const error = loadError(() =>
      checkWrittenValue(value, "out", range, new LoadDiagnostics(), { enclosingLevels: 1 }),
    );
    expect(error.diagnostic.message).toContain(LIMIT);
  });
});

describe("recursive aliases", () => {
  test("an env value holding itself is a located load error", () => {
    expectError(flowEnv(["", "&a [*a]"]), "flow env 'K' at [0]", CYCLE);
  });

  test.each([
    ["record", "&a {b: *a}", "b"],
    ["inner-list", "[1, &a [2, *a]]", "[1][1]"],
    ["outer-alias", "&a [&b [*a]]", "[0][0]"],
    ["merge-key", "&a {b: 1, <<: {c: *a}}", "c"],
  ])("a flow env cycle names the entry that refers back (%s)", (_, value, path) => {
    expectError(flowEnv(["", value]), `flow env 'K' at ${path}`, CYCLE);
  });

  test.each([
    ["direct", "&e {K: *e}", "K"],
    ["list", "&e {K: [*e]}", "[0].K"],
  ])("an env mapping holding itself is a located load error (%s)", (_, env, path) => {
    expectError(flowEnvInline(env), `flow env 'K' at ${path}`, CYCLE, "env:");
  });

  test("env nesting is checked before the cycle", () => {
    expectLimitError(flowEnv(cyclicListAlias(MAX_VALUE_NESTING + 1)));
    const { message } = envError(flowEnv(cyclicListAlias(MAX_VALUE_NESTING)));
    expect(message.startsWith("flow env 'K' at [0]")).toBe(true);
    expect(message.endsWith(CYCLE)).toBe(true);
  });

  test("an env value repeating an alias passes", () => {
    checkFlowEnv("x-anchors: [&s [1, 2]]\n" + flowEnv(["", "[*s, *s]"]));
  });

  test("an env scalar value passes", () => {
    checkFlowEnv(flowEnv(["", "1"]));
  });
});

describe("lone surrogates", () => {
  // Only gives `range` a source to frame.
  const SOURCE = "a\nb: 123456789\n";
  const range = new SourceRange(2, 3, 2, 9);
  const where = "node 'use' input 'topic'";

  /** Return the error `checkWrittenValue(value, where, range, …)` throws. */
  function writtenError(value: Value): LoadError {
    const diagnostics = new LoadDiagnostics("f.yaml", SOURCE);
    return loadError(() => checkWrittenValue(value, where, range, diagnostics));
  }

  test("text holding one is refused at its path, at the range", () => {
    const error = writtenError(["ok", { k: "\uD800" }]);
    expect(error.diagnostic.message).toBe(`${where} at [1].k: ${SURROGATE}`);
    expect(error.diagnostic.frames.map((frame) => frame.range)).toEqual([range]);
    expect(error.cause).toBeInstanceOf(TypeCheckError);
  });

  test("a key holding one is refused at its record's path", () => {
    expect(writtenError({ "\uDC00": 1n }).diagnostic.message).toBe(`${where}: ${SURROGATE}`);
  });

  test("a surrogate pair passes", () => {
    checkWrittenValue(["\uD83D\uDE00", { "\uD83D\uDE00": "x" }], where, range, new LoadDiagnostics());
  });

  test("a recursive value holding no bad text passes", () => {
    const items: Value[] = ["ok"];
    items.push(items);
    checkWrittenValue(items, where, range, new LoadDiagnostics());
  });

  test("an env value holding one is refused at the value", () => {
    expectError('id: f\nenv:\n  K: "\\uD800"\n', "flow env 'K'", SURROGATE);
  });

  test("an env key holding one is refused at the key", () => {
    const text = 'id: f\nenv:\n  "\\uD800": 1\n';
    const document = SourceDocument.parse(text);
    const { diagnostic } = loadError(() => checkFlowEnv(text));
    expect(diagnostic.message).toBe(`flow env '\\ud800': ${SURROGATE}`);
    expect(diagnostic.frames.map((frame) => frame.range)).toEqual([
      document.root.child("env").keyRangeOf("\uD800")!,
    ]);
  });

  test("a value past the limit that also holds one gets the depth message", () => {
    const deep = "[".repeat(MAX_VALUE_NESTING + 1) + '"\\uD800"' + "]".repeat(MAX_VALUE_NESTING + 1);
    expectLimitError(flowEnv(["", deep]));
  });

  test("the surrogate check runs before the cycle check", () => {
    expectError(flowEnv(["", '&a ["\\uD800", *a]']), "flow env 'K' at [0]", SURROGATE);
  });
});

// ---------- each site, through `loadFlow` ----------

const ECHO = "    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n";

/** A flow builder putting `value` at `env: {K: value}` on one site, `header` first. */
type SiteFlow = (header: string, value: string) => string;

/** Return a flow whose flow `env:` sets `K` to `value`, with a node. */
const loadedFlowEnv: SiteFlow = (header, value) =>
  `${header}id: f\nname: f\nenv:\n  K: ${value}\nnodes:\n  use:\n${ECHO}` +
  "    input:\n      topic: x\noutput: ${use.output}\n";

/** Return a flow whose node `use` has `env:` setting `K` to `value`. */
const nodeEnv: SiteFlow = (header, value) =>
  `${header}id: f\nname: f\nnodes:\n  use:\n${ECHO}    env:\n      K: ${value}\n` +
  "    input:\n      topic: x\noutput: ${use.output}\n";

/** Return a flow that calls multi-node def `d`, whose `env:` sets `K` to `value`. */
const multiDefEnv: SiteFlow = (header, value) =>
  `${header}id: f\nname: f\ndefs:\n  d:\n    env:\n      K: ${value}\n` +
  "    nodes:\n      keep:\n        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n          topic: x\n    output: ${keep.output}\n" +
  "nodes:\n  use:\n    kind: call\n    call: d\noutput: ${use.output}\n";

/** Return a flow that calls single-node def `d`, whose `env:` sets `K` to `value`. */
const singleDefEnv: SiteFlow = (header, value) =>
  `${header}id: f\nname: f\ndefs:\n  d:\n${ECHO}    env:\n      K: ${value}\n` +
  "    input:\n      topic: !str\n" +
  "nodes:\n  use:\n    kind: call\n    call: d\n    input:\n      topic: x\n" +
  "output: ${use.output}\n";

/** Return a compact (single-node) flow whose `env:` sets `K` to `value`. */
const compactEnv: SiteFlow = (header, value) =>
  `${header}id: f\nname: f\nenv:\n  K: ${value}\n` +
  "kind: code\ncode: ./fixtures/compose_codefns.ts:echo\ninput:\n  topic: !str x\n";

/** Site name to (flow builder, message prefix, located line). */
const SITES: [string, SiteFlow, string, string][] = [
  ["flow", loadedFlowEnv, "flow env 'K' at ", "  K:"],
  ["node", nodeEnv, "node 'use' env 'K' at ", "      K:"],
  ["multi-node-def", multiDefEnv, "defs entry 'd': env 'K' at ", "      K:"],
  ["single-node-def", singleDefEnv, "defs entry 'd': node 'd' env 'K' at ", "      K:"],
  ["compact", compactEnv, "node 'f' env 'K' at ", "  K:"],
];

/** Assert loading `text` fails with the limit error for `where` at the line `linePrefix`. */
async function expectLoadLimitError(text: string, where: string, linePrefix: string): Promise<void> {
  const { diagnostic } = await loadErrorAsync(loadConformanceFlow(text));
  expect(diagnostic.message.startsWith(where)).toBe(true);
  expect(diagnostic.message).toContain(LIMIT);
  expect(diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, linePrefix));
}

/** Assert loading `text` fails with exactly `<start>: <cycle>` at the line `linePrefix`. */
async function expectLoadCycleError(text: string, start: string, linePrefix: string): Promise<void> {
  const { diagnostic } = await loadErrorAsync(loadConformanceFlow(text));
  expect(diagnostic.message).toBe(`${start}: ${CYCLE}`);
  expect(diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, linePrefix));
}

describe("loaded sites", () => {
  describe.each(SITES)("%s", (_site, flow, where, line) => {
    test.each([
      ["list", listAlias],
      ["record", recordAlias],
    ])("an env value over a 1000-link %s alias chain is a located load error", async (_, form) => {
      await expectLoadLimitError(flow(...form(1000)), where, line);
    });

    test.each([
      ["text", list],
      ["alias", listAlias],
    ])("an env value at the limit as %s loads", async (_, form) => {
      await loadConformanceFlow(flow(...form(MAX_VALUE_NESTING)));
    });

    test("an env value holding itself is a located load error", async () => {
      await expectLoadCycleError(flow("", "&a [*a]"), `${where}[0]`, line);
    });
  });

  test("a flow env value past the limit as text is a located load error", async () => {
    await expectLoadLimitError(loadedFlowEnv(...list(MAX_VALUE_NESTING + 1)), "flow env 'K' at ", "  K:");
  });

  test("an env value follows the configured limit", async () => {
    install({ ...BUILTIN, maxYamlNodes: 10_000_000, maxValueNesting: 5 });

    await loadConformanceFlow(loadedFlowEnv(...list(5)));
    const error = await loadErrorAsync(loadConformanceFlow(loadedFlowEnv(...list(6))));
    expect(error.diagnostic.message).toContain("value nests deeper than 5 levels");
  });

  test("flow env nesting is checked before max_tool_iterations", async () => {
    const [header, value] = listAlias(1000);
    const text =
      `${header}id: f\nname: f\nenv:\n  max_tool_iterations: ${value}\nnodes:\n` +
      `  use:\n${ECHO}    input:\n      topic: x\noutput: \${use.output}\n`;

    await expectLoadLimitError(text, "flow env 'max_tool_iterations' at ", "  max_tool_iterations:");
  });

  /** Return a flow whose node `use` has the one-line `env:` mapping `env`. */
  const nodeEnvInline = (env: string) =>
    `id: f\nname: f\nnodes:\n  use:\n${ECHO}    env: ${env}\n` + "    input:\n      topic: x\noutput: ${use.output}\n";

  /** Return a flow whose flow `env:` is the one-line mapping `env`, with a node. */
  const flowEnvInlineLoaded = (env: string) =>
    `id: f\nname: f\nenv: ${env}\nnodes:\n  use:\n${ECHO}` + "    input:\n      topic: x\noutput: ${use.output}\n";

  test.each([
    ["flow", flowEnvInlineLoaded, "flow env 'K'", "env:"],
    ["node", nodeEnvInline, "node 'use' env 'K'", "    env:"],
  ] as const)("an env mapping holding itself is a located load error (%s)", async (_, flow, where, line) => {
    await expectLoadCycleError(flow("&e {K: *e}"), `${where} at K`, line);
    await expectLoadCycleError(flow("&e {K: [*e]}"), `${where} at [0].K`, line);
  });

  test("an env value repeating an alias loads", async () => {
    await loadConformanceFlow("x-anchors: [&s [1, 2]]\n" + loadedFlowEnv("", "[*s, *s]"));
  });

  test("an env scalar value loads", async () => {
    await loadConformanceFlow(loadedFlowEnv("", "1"));
  });
});
