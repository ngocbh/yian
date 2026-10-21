/**
 * A record type in `output:` counts its levels against the type nesting limit at load: a port of
 * `tests/compose/test_record_type_nesting.py`.
 *
 * A record written as a mapping, or as a list of single-key maps, in a node's `output:` is one
 * level, and so is every record inside it. A field's type text adds the levels of its resolved
 * type; the total must stay within `MAX_TYPE_NESTING` (100 by default), however YAML text or
 * aliases built the record, so a 1000-link alias chain is a located load error (under a raised
 * `YIAN_MAX_YAML_NODES`: at the default, the chain's anchors exceed the node limit first).
 *
 * Ported: `test_a_record_output_over_a_1000_link_alias_chain_is_a_located_load_error` (all seven
 * sites), `test_a_list_of_fields_output_over_a_1000_link_alias_chain_is_a_located_load_error`,
 * `test_one_record_past_the_limit_is_a_load_error`, `test_a_field_type_counts_its_lists`,
 * `test_records_and_a_deep_list_field_are_counted_together`,
 * `test_a_record_at_the_limit_loads_and_runs`, `test_types_at_the_limit_load`,
 * `test_a_record_at_the_highest_configurable_limit_loads_and_runs` (125, the TS maximum) and
 * `test_a_recursive_record_alias_keeps_its_error` (both rows).
 *
 * Changed: Python also asserts no `RecursionError` in the error's chain; here a stack overflow
 * would be a `RangeError`, not the `LoadError` the rows expect, so `loadErrorAsync` fails on it.
 * The CODE references are TS's (`./fixtures/compose_codefns.ts:echo`).
 *
 * Covered by `types/compose_record_type_nesting__a_typedef_name_counts_as_deep_as_its_definition`:
 * `test_a_typedef_name_counts_as_deep_as_its_definition`; by
 * `types/compose_record_type_nesting__record_at_a_configured_limit_loads` and
 * `types/compose_record_type_nesting__record_past_a_configured_limit`:
 * `test_record_nesting_follows_the_configured_limit`.
 *
 * Already ported: `test_a_case_output_over_a_1000_link_alias_chain_is_rejected_as_a_case_output`
 * in `validate.test.ts`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunResult, runFlow } from "../../src/run.ts";
import { loadConformanceFlow, loadErrorAsync } from "./declarations.ts";

const MAX_TYPE_NESTING = BUILTIN.maxTypeNesting;

const LIMIT = `nests deeper than ${MAX_TYPE_NESTING} levels, the limit for a type`;

// A listed chain's anchors cost quadratically many nodes: a 1000-link chain holds over 500,000,
// past the default YIAN_MAX_YAML_NODES. These rows exercise the nesting checks behind the node
// limit, so they raise it; yaml_node_limit.test.ts covers the default.
beforeEach(() => install({ ...BUILTIN, maxYamlNodes: 10_000_000 }));
afterEach(() => forget());

const ECHO = "    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n";

/** A typedef 99 lists deep. */
const T99 = "list(".repeat(99) + "int" + ")".repeat(99);

// ---------- record forms: each returns `[header, value]`, where `header` holds the top-level
// `x-anchors:` line the value's aliases name (empty for written-out text) ----------

/**
 * A record nesting exactly `levels` records, the innermost field typed `leaf`.
 *
 * `levels = 1` is `{k: <leaf>}`. Past 10 levels the record is built over an alias chain.
 */
function records(levels: number, leaf: string): [string, string] {
  if (levels <= 10) return ["", "{k: ".repeat(levels) + `"${leaf}"` + "}".repeat(levels)];
  const items = [`&m0 {k: "${leaf}"}`];
  for (let i = 1; i < levels - 1; i++) items.push(`&m${i} {k: *m${i - 1}}`);
  return [`x-anchors: [${items.join(", ")}]\n`, `{k: *m${levels - 2}}`];
}

/** `records(levels, leaf)` spelled as lists of fields, each level `[{k: ...}]`. */
function fieldListRecords(levels: number, leaf: string): [string, string] {
  if (levels <= 10) return ["", "[{k: ".repeat(levels) + `"${leaf}"` + "}]".repeat(levels)];
  const items = [`&m0 [{k: "${leaf}"}]`];
  for (let i = 1; i < levels - 1; i++) items.push(`&m${i} [{k: *m${i - 1}}]`);
  return [`x-anchors: [${items.join(", ")}]\n`, `[{k: *m${levels - 2}}]`];
}

// ---------- flows: each puts `value` at one node's `output:` ----------

/** A flow whose CODE node `use` declares `output: value`. */
const code = (header: string, value: string, typedefs = "") =>
  `${header}id: f\nname: f\n${typedefs}nodes:\n  use:\n${ECHO}` +
  `    input:\n      topic: x\n    output: ${value}\noutput: \${use.output}\n`;

/** A flow whose AGENT node `use` declares `output: value`. */
const agent = (header: string, value: string) =>
  `${header}id: f\nname: f\nnodes:\n  use:\n    kind: agent\n    prompt: hi\n` +
  `    output: ${value}\noutput: \${use.output}\n`;

/** A flow whose HUMAN_INPUT node `use` declares `output: value`. */
const humanInput = (header: string, value: string) =>
  `${header}id: f\nname: f\nnodes:\n  use:\n    kind: human_input\n    prompt: hi\n` +
  `    output: ${value}\noutput: \${use.output}\n`;

/** A flow whose TOOL node `use` declares `output: value`. */
const toolSite = (header: string, value: string) =>
  `${header}id: f\nname: f\nnodes:\n  use:\n    kind: tool\n    tool_id: http_get\n` +
  `    input:\n      url: x\n    output: ${value}\noutput: \${use.output}\n`;

/** A compact (single-node) flow declaring `output: value`. */
const compact = (header: string, value: string) =>
  `${header}id: f\nname: f\nkind: code\ncode: ./fixtures/compose_codefns.ts:echo\n` +
  `input:\n  topic: !str\noutput: ${value}\n`;

const CALL = "nodes:\n  use:\n    kind: call\n    call: d\n    input:\n      topic: x\noutput: ${use.output}\n";

/** A flow calling single-node def `d`, which declares `output: value`. */
const singleDef = (header: string, value: string) =>
  `${header}id: f\nname: f\ndefs:\n  d:\n${ECHO}    input:\n      topic: !str\n` +
  `    output: ${value}\n${CALL}`;

/** A flow calling multi-node def `d`, whose node `u` declares `output: value`. */
const multiDef = (header: string, value: string) =>
  `${header}id: f\nname: f\ndefs:\n  d:\n    input:\n      topic: !str\n` +
  "    nodes:\n      u:\n        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  `        input:\n          topic: x\n        output: ${value}\n` +
  `    output: \${u.output}\n${CALL}`;

const SITES: Record<string, (header: string, value: string) => string> = {
  code,
  agent,
  human_input: humanInput,
  tool: toolSite,
  compact,
  "single-node-def": singleDef,
  "multi-node-def": multiDef,
};

/** Assert loading `text` fails with a located limit error, and return its message. */
async function limitError(text: string, limit = LIMIT): Promise<string> {
  const error = await loadErrorAsync(loadConformanceFlow(text));
  const message = error.diagnostic.message;
  expect(message).toContain(limit);
  expect(error.diagnostic.frames.at(-1)!.range).toBeDefined();
  return message;
}

// ---------- a 1000-link alias chain ----------

test.each(Object.keys(SITES))(
  "a record output over a 1000-link alias chain is a located load error (%s)",
  async (site) => {
    await limitError(SITES[site]!(...records(1000, "str")));
  },
);

test("a list of fields output over a 1000-link alias chain is a located load error", async () => {
  await limitError(code(...fieldListRecords(1000, "str")));
});

// ---------- records and field types counted together ----------

test("one record past the limit is a load error", async () => {
  const message = await limitError(code(...records(MAX_TYPE_NESTING + 1, "str")));

  expect(message).toContain(`type at ${Array(MAX_TYPE_NESTING).fill("k").join(".")} nests`);
  expect(message).not.toContain("its type");
});

test("a field type counts its lists", async () => {
  const message = await limitError(code(...records(99, "list(list(int))")));

  expect(message.endsWith("(each record and list is a level); its type 'list(list(int))' adds 2")).toBe(true);
});

test("records and a deep list field are counted together", async () => {
  const leaf = "list(".repeat(50) + "int" + ")".repeat(50);

  await limitError(code(...records(51, leaf)));
});

// ---------- guards ----------

test("a record at the limit loads and runs", async () => {
  const result = await runFlow(await loadConformanceFlow(code(...records(MAX_TYPE_NESTING, "str"))), {});

  expect(result).toBeInstanceOf(RunResult);
});

test("types at the limit load", async () => {
  await loadConformanceFlow(code(...records(98, "list(list(int))")));
  await loadConformanceFlow(code(...records(50, "list(".repeat(50) + "int" + ")".repeat(50))));
  await loadConformanceFlow(code(...records(1, "T"), `typedefs:\n  T: "${T99}"\n`));
});

test("a record at the highest configurable limit loads and runs", async () => {
  install({ ...BUILTIN, maxYamlNodes: 10_000_000, maxTypeNesting: 125 });

  const result = await runFlow(await loadConformanceFlow(code(...records(125, "str"))), {});

  expect(result).toBeInstanceOf(RunResult);
});

test.each(["&R {a: *R}", "&R [{a: *R}]"])("a recursive record alias keeps its error (%s)", async (output) => {
  const error = await loadErrorAsync(loadConformanceFlow(code("", output)));
  expect(error.message).toContain("cannot descend through a recursive alias occurrence");
});
