/**
 * A written-out value nested past 100 levels is a located load error.
 *
 * Each value written out in a flow file at a value position is checked at load with the counting
 * the run-time check applies to it, whether YAML text or aliases built it. A value at the limit
 * loads; one past it fails at load, located, and a 1000-link alias chain fails the same way (under
 * a raised `YIAN_MAX_YAML_NODES`: at the default, the chain's anchors exceed the node limit first).
 */

import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import {
  assembled,
  assembleText,
  type CaseRow,
  loadConformanceFlow,
  loadError,
  runCase,
} from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-written-value-nesting-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

const MAX_VALUE_NESTING = BUILTIN.maxValueNesting;

const LIMIT = `value nests deeper than ${MAX_VALUE_NESTING} levels, the limit for a yian value`;

// A listed chain's anchors cost quadratically many nodes: a 1000-link chain holds over 500,000,
// past the default YIAN_MAX_YAML_NODES. These rows exercise the nesting checks behind the node
// limit, so they raise it; yaml_node_limit.test.ts covers the default.
beforeEach(() => install({ ...BUILTIN, maxYamlNodes: 10_000_000 }));
afterEach(() => forget());

// ---------- value forms: each returns `[header, value]`, where `header` holds the top-level
// `x-anchors:` line the value's aliases name (empty for written-out text) ----------

type Form = (levels: number) => [string, string];

/** A list written out as text, nesting `levels` levels. */
const list: Form = (levels) => ["", "[".repeat(levels) + "1" + "]".repeat(levels)];

/** A mapping written out as text, nesting `levels` levels. */
const record: Form = (levels) => ["", "{a: ".repeat(levels) + "1" + "}".repeat(levels)];

/** Return an `x-anchors:` line whose anchor `a<i>` names a value nesting i+1 levels. */
function anchors(count: number, options: { record: boolean }): string {
  const items = [options.record ? "&a0 {a: 1}" : "&a0 [1]"];
  for (let i = 1; i < count; i++) items.push(options.record ? `&a${i} {a: *a${i - 1}}` : `&a${i} [*a${i - 1}]`);
  return `x-anchors: [${items.join(", ")}]\n`;
}

/**
 * A one-item list over an alias chain, nesting exactly `levels` levels.
 *
 * YAML can't tag an alias, so the value wraps the last alias in one written level.
 */
const listAlias: Form = (levels) => [anchors(levels - 1, { record: false }), `[*a${levels - 2}]`];

/** A one-key mapping over an alias chain, nesting exactly `levels` levels. */
const recordAlias: Form = (levels) => [anchors(levels - 1, { record: true }), `{a: *a${levels - 2}}`];

/** A one-item list of a mapping written out as text, nesting `levels` levels. */
const recordList: Form = (levels) => ["", "[" + record(levels - 1)[1] + "]"];

/** A one-item list of a mapping over an alias chain, nesting `levels` levels. */
const recordListAlias: Form = (levels) => [anchors(levels - 2, { record: true }), `[{a: *a${levels - 3}}]`];

// ---------- flows ----------

type Flow = (header: string, value: string) => string;

const ECHO = "    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n";

/** A flow whose CODE node `use` binds `topic` to `value` and echoes it. */
const inputFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  use:\n${ECHO}    input:\n      topic: ${value}\noutput: \${use.output}\n`;

/** `inputFlow` with `topic` tagged `!object`. */
const taggedInputFlow: Flow = (header, value) => inputFlow(header, `!object ${value}`);

/** A flow whose LOOP `l` seeds its carried `s` with `value`. */
const loopFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}defs:\n  body:\n    input:\n      s: !object\n` +
  "    nodes:\n      keep:\n        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n          topic: ${input.s}\n" +
  "    output:\n      s: ${keep.output}\n" +
  "nodes:\n  l:\n    kind: loop\n    call: body\n    times: 2\n" +
  `    input:\n      s: ${value}\noutput: \${l.output.s}\n`;

/** A flow whose MAP `m` echoes each item of `over: value`. */
const mapFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}defs:\n  wrap:\n    input:\n      v: !object\n` +
  "    nodes:\n      keep:\n        kind: code\n" +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n          topic: ${input.v}\n" +
  "    output: ${keep.output}\n" +
  `nodes:\n  m:\n    kind: map\n    call: wrap\n    over: ${value}\n` +
  "    input:\n      v: ${item}\noutput: ${m.output}\n";

/** A flow whose sleep `w` sleeps `until: value`. */
const sleepFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  w:\n    kind: tool\n    tool_id: sleep\n    input:\n      until: ${value}\n` +
  `  use:\n${ECHO}    input:\n      topic: x\n    runs_after: [w]\n` +
  "output: ${use.output}\n";

/** A flow whose value entry `v` holds `!object value`, echoed by `use`. */
const valueEntryFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  v: !object ${value}\n  use:\n${ECHO}` +
  "    input:\n      topic: ${v.output}\noutput: ${use.output}\n";

/** A flow whose input `topic` defaults to `!object value`, echoed by `use`. */
const flowDefaultFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}input:\n  topic: !object ${value}\nnodes:\n  use:\n` +
  `${ECHO}    input:\n      topic: \${input.topic}\noutput: \${use.output}\n`;

/** A flow that calls def `child`, whose input `topic` defaults to `value`. */
const defDefaultFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}defs:\n  child:\n    input:\n` +
  `      topic: !object ${value}\n    nodes:\n      keep:\n        kind: code\n` +
  "        code: ./fixtures/compose_codefns.ts:echo\n" +
  "        input:\n          topic: ${input.topic}\n    output: ${keep.output}\n" +
  "nodes:\n  c:\n    kind: call\n    call: child\noutput: ${c.output}\n";

/** A compact (single-node) flow whose input `topic` defaults to `value`. */
const compactDefaultFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}input:\n  topic: !object ${value}\n` +
  "kind: code\ncode: ./fixtures/compose_codefns.ts:echo\n";

/** A flow whose AGENT `writer` declares variable `t` defaulting to `value`. */
const agentVariableFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}llm_config: {provider: ollama, model: test}\n` +
  "nodes:\n  writer:\n    kind: agent\n    variables:\n" +
  `      t: !object ${value}\n    prompt: go\n    output: !str\n` +
  "output: ${writer.output}\n";

// ---------- helpers ----------

/** Return the 1-based number of the only line of `text` starting with `prefix`. */
function lineOf(text: string, prefix: string): number {
  const numbers = text
    .split("\n")
    .flatMap((line, index) => (line.startsWith(prefix) ? [index + 1] : []));
  expect(numbers).toHaveLength(1);
  return numbers[0]!;
}

/** Assert assembling `text` fails with the limit error for `where` at the line `linePrefix` starts. */
function expectLimitError(text: string, where: string, linePrefix: string): void {
  const error = loadError(() => assembleText(text));
  expect(error.message).toContain(where);
  expect(error.message).toContain(LIMIT);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, linePrefix));
}

// ---------- node bindings ----------

interface BindingSite {
  flow: Flow;
  text: Form;
  alias: Form;
  /** The message prefix. */
  where: string;
  /** The start of the located line. */
  line: string;
  /** Whether the flow at the limit runs. */
  runs: boolean;
}

const BINDING_SITES: Readonly<Record<string, BindingSite>> = {
  input: {
    flow: inputFlow,
    text: list,
    alias: listAlias,
    where: "node 'use' input 'topic' at ",
    line: "      topic:",
    runs: true,
  },
  "tagged-input": {
    flow: taggedInputFlow,
    text: record,
    alias: recordAlias,
    where: "node 'use' input 'topic' at ",
    line: "      topic:",
    runs: true,
  },
  "loop-seed": {
    flow: loopFlow,
    text: record,
    alias: recordAlias,
    where: "node 'l' input 's' at ",
    line: "      s: {",
    runs: false,
  },
  // 100 levels in `over:` means items nesting 99.
  "map-over": {
    flow: mapFlow,
    text: recordList,
    alias: recordListAlias,
    where: "node 'm' field 'over' at ",
    line: "    over:",
    runs: true,
  },
  // A list `until` isn't a date: at the limit it is a type error at load, and past the limit the
  // nesting error comes first.
  "sleep-until": {
    flow: sleepFlow,
    text: list,
    alias: listAlias,
    where: "node 'w' input 'until' at ",
    line: "      until:",
    runs: false,
  },
  "value-entry": {
    flow: valueEntryFlow,
    text: record,
    alias: recordAlias,
    where: "value entry 'v' at ",
    line: "  v:",
    runs: true,
  },
  // A default is checked as the binding it fills.
  "flow-default": {
    flow: flowDefaultFlow,
    text: record,
    alias: recordAlias,
    where: "input 'topic' default at ",
    line: "  topic:",
    runs: true,
  },
  "def-default": {
    flow: defDefaultFlow,
    text: record,
    alias: recordAlias,
    where: "defs entry 'child': input 'topic' default at ",
    line: "      topic: !object",
    runs: true,
  },
  "compact-default": {
    flow: compactDefaultFlow,
    text: record,
    alias: recordAlias,
    where: "input 'topic' default at ",
    line: "  topic:",
    runs: true,
  },
  "agent-variable": {
    flow: agentVariableFlow,
    text: record,
    alias: recordAlias,
    where: "variable 't' default at ",
    line: "      t:",
    runs: false,
  },
};

const SITE_NAMES = Object.keys(BINDING_SITES);
const FORMS = [
  ["text", false],
  ["alias", true],
] as const;

/** Each of `sites` in each form: `[site, form id, aliased]`. */
const formRows = (sites: readonly string[]) =>
  sites.flatMap((site) => FORMS.map(([id, aliased]) => [site, id, aliased] as const));

describe("node bindings", () => {
  test.each(formRows(SITE_NAMES))(
    "at the limit passes the nesting check: %s (%s)",
    (site, _id, aliased) => {
      const { flow, text, alias } = BINDING_SITES[site]!;
      const [header, value] = (aliased ? alias : text)(MAX_VALUE_NESTING);

      if (site === "sleep-until") {
        // Past the nesting check, a list `until:` meets the type check.
        const error = loadError(() => assembleText(flow(header, value)));
        expect(error.message).toContain("node 'w' (tool_id: sleep) input 'until': must be a date or datetime, got [[[");
        return;
      }
      assembleText(flow(header, value));
    },
  );

  test.each(formRows(SITE_NAMES.filter((site) => BINDING_SITES[site]!.runs)))(
    "at the limit passes the nesting check and runs: %s (%s)",
    async (site, _id, aliased) => {
      const { flow, text, alias } = BINDING_SITES[site]!;
      const [header, value] = (aliased ? alias : text)(MAX_VALUE_NESTING);

      const result = await runFlow(await loadConformanceFlow(flow(header, value)), {}, { env });

      expect(result.failure).toBeNull();
      expect(result.status).toBe("succeeded");
    },
  );

  test.each(formRows(SITE_NAMES))(
    "past the limit is a located load error: %s (%s)",
    (site, _id, aliased) => {
      const { flow, text, alias, where, line } = BINDING_SITES[site]!;
      const [header, value] = (aliased ? alias : text)(MAX_VALUE_NESTING + 1);

      expectLimitError(flow(header, value), where, line);
    },
  );

  test.each(SITE_NAMES)("over a 1000-link alias chain is a located load error: %s", (site) => {
    const { flow, alias, where, line } = BINDING_SITES[site]!;

    expectLimitError(flow(...alias(1000)), where, line);
  });

  test("the issue's 1000-link chain, written straight into the input, fails at load", () => {
    const items = Array.from({ length: 1000 }, (_, i) => (i ? `&a${i} [*a${i - 1}]` : "&a0 [1]")).join(", ");
    const text =
      "id: f\nname: f\nnodes:\n  use:\n    kind: code\n" +
      "    code: ./fixtures/compose_codefns.ts:echo\n" +
      `    input:\n      topic: [${items}]\n    output: !str\noutput: \${use.output}\n`;

    expectLimitError(text, "node 'use' input 'topic' at ", "      topic:");
  });
});

/** A flow whose untagged list entry `v` sits beside a node `use`. */
const untaggedValueEntryFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  v: ${value}\n  use:\n${ECHO}` +
  "    input:\n      topic: x\noutput: ${use.output}\n";

/** The past-the-limit forms of a value that never loads: `[id, form, levels]`. */
const PAST_ONLY: readonly (readonly [string, Form, number])[] = [
  ["text-101", list, MAX_VALUE_NESTING + 1],
  ["alias-101", listAlias, MAX_VALUE_NESTING + 1],
  ["alias-1000", listAlias, 1000],
];

// An untagged list entry is always a load error that suggests a tag, so there is no loading case
// at the limit; past it, the limit is reported instead.
test.each(PAST_ONLY)("an untagged value entry past the limit reports the limit: %s", (_id, form, levels) => {
  expectLimitError(untaggedValueEntryFlow(...form(levels)), "value entry 'v' at ", "  v:");
});

// ---------- CASE value branches ----------

/** A flow whose CASE `r` has one `then:` branch and an `else:` branch. */
function caseFlow(header: string, then: string, else_: string): string {
  return (
    `id: f\nname: f\n${header}input:\n  b: !bool\nnodes:\n  r:\n    kind: case\n` +
    `    cases:\n      - when: \${input.b}\n        then: ${then}\n    else: ${else_}\n` +
    `  use:\n${ECHO}    input:\n      topic: x\noutput: \${use.output}\n`
  );
}

// A value branch is always a load error, so there is no loading case at the limit.
test.each((["then", "else"] as const).flatMap((branch) => PAST_ONLY.map((row) => [branch, ...row] as const)))(
  "a case value branch past the limit reports the limit: %s %s",
  (branch, _id, form, levels) => {
    const [header, value] = form(levels);
    if (branch === "then") {
      expectLimitError(caseFlow(header, value, "use"), "node 'r' then[0] value at ", "        then:");
    } else {
      expectLimitError(caseFlow(header, "use", value), "node 'r' else value at ", "    else:");
    }
  },
);

// ---------- flow outputs ----------

/** A flow whose record output has field `r: value` beside `u`. */
const recordOutputFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  use:\n${ECHO}    input:\n      topic: x\n` +
  `output:\n  r: ${value}\n  u: \${use.output}\n`;

/** A flow whose whole output section is `!object value`. */
const taggedOutputFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  use:\n${ECHO}    input:\n      topic: x\n` +
  `output: !object ${value}\n`;

/** A flow whose single output is the untagged `value`. */
const singleOutputFlow: Flow = (header, value) =>
  `id: f\nname: f\n${header}nodes:\n  use:\n${ECHO}    input:\n      topic: x\n` +
  `output: ${value}\n`;

interface OutputSite {
  flow: Flow;
  text: Form;
  alias: Form;
  where: string;
  line: string;
  /** The deepest value that loads. */
  limit: number;
}

// An untagged mapping is a record whose fields sit one level in; a tagged section or a single
// value is one value.
const OUTPUT_SITES: Readonly<Record<string, OutputSite>> = {
  record: {
    flow: recordOutputFlow,
    text: record,
    alias: recordAlias,
    where: "flow output 'r' at ",
    line: "  r:",
    limit: MAX_VALUE_NESTING - 1,
  },
  "tagged-section": {
    flow: taggedOutputFlow,
    text: record,
    alias: recordAlias,
    where: "flow output at ",
    line: "output:",
    limit: MAX_VALUE_NESTING,
  },
  single: {
    flow: singleOutputFlow,
    text: list,
    alias: listAlias,
    where: "flow output at ",
    line: "output:",
    limit: MAX_VALUE_NESTING,
  },
};

const OUTPUT_ROWS = formRows(Object.keys(OUTPUT_SITES));

describe("flow outputs", () => {
  test.each(OUTPUT_ROWS)("at the limit loads: %s (%s)", (site, _id, aliased) => {
    const { flow, text, alias, limit } = OUTPUT_SITES[site]!;

    assembleText(flow(...(aliased ? alias : text)(limit)));
  });

  test.each(OUTPUT_ROWS)("at the limit loads and runs: %s (%s)", async (site, _id, aliased) => {
    const { flow, text, alias, limit } = OUTPUT_SITES[site]!;

    const result = await runFlow(await loadConformanceFlow(flow(...(aliased ? alias : text)(limit))), {}, { env });

    expect(result.failure).toBeNull();
    expect(result.status).toBe("succeeded");
  });

  test.each(OUTPUT_ROWS)("past the limit is a located load error: %s (%s)", (site, _id, aliased) => {
    const { flow, text, alias, where, line, limit } = OUTPUT_SITES[site]!;

    expectLimitError(flow(...(aliased ? alias : text)(limit + 1)), where, line);
  });

  test.each(Object.keys(OUTPUT_SITES))("over a 1000-link alias chain is a located load error: %s", (site) => {
    const { flow, alias, where, line } = OUTPUT_SITES[site]!;

    expectLimitError(flow(...alias(1000)), where, line);
  });
});

// ---------- inline bodies ----------

test("a binding in an inline MAP body over a 1000-link alias chain is located", () => {
  // The inline body's own load reports the inner binding, not a resolve failure.
  const [header, value] = listAlias(1000);
  const text =
    `id: f\nname: f\n${header}nodes:\n  m:\n    kind: map\n    over: [1]\n` +
    "    nodes:\n      keep:\n        kind: code\n" +
    "        code: ./fixtures/compose_codefns.ts:echo\n" +
    `        input:\n          topic: ${value}\n    output: \${keep.output}\n` +
    "output: ${m.output}\n";

  expectLimitError(text, "node 'keep' input 'topic' at ", "          topic:");
});

// ---------- harvested cases ----------

test.each<CaseRow>([
  { id: "yaml/compose_written_value_nesting__binding_past_the_limit", call: assembled },
  { id: "yaml/compose_written_value_nesting__flow_output_past_the_limit", call: assembled },
])("assemble: $id", runCase);
