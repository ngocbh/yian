/**
 * A builtin called with arguments it doesn't take fails at load in every binding and prompt,
 * whether the flow is loaded from YAML or the binding is set in code on the loaded flow. The
 * question sites are a TOOL `ask`'s `prompt` and `questions` inputs; the `until` site is a TOOL
 * `sleep`'s input.
 */

import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import type { Flow } from "../../src/compile/model.ts";
import { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { ExpressionError } from "../../src/expr/expressions.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import type { CallNode } from "../../src/nodes/call/node.ts";
import { Binding, ValueTemplate } from "../../src/nodes/contracts.ts";
import type { MapNode } from "../../src/nodes/map/node.ts";
import { advance, runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { assembled, assembleText, type CaseRow, loadError, runCase } from "../helpers/declarations.ts";
import { expectFailsBeforeTheRun, rejected, stoppedRunLog } from "../helpers/run_state_fixtures.ts";

const UPPER = "expression function 'upper' takes (s): too many positional arguments";

/**
 * Load `text`, expecting a `LoadError`; return its message. With `located`, the error's last range
 * must sit on one line and cover `located`.
 */
function loadErrorOf(text: string, options: { located?: string } = {}): string {
  const error = loadError(() => assembleText(text));
  const range = error.diagnostic.frames.at(-1)!.range;
  if (options.located !== undefined) {
    expect(range.startLine).toBe(range.endLine);
    const line = text.split("\n")[range.startLine - 1]!;
    expect(line.slice(range.startCol - 1, range.endCol)).toContain(options.located);
  }
  return error.message;
}

/** Replace each `<key>` of `template` with its value. */
function fill(template: string, values: Readonly<Record<string, string>>): string {
  for (const [key, value] of Object.entries(values)) template = template.replaceAll(`<${key}>`, value);
  return template;
}

const CODE = `
id: f
name: f
input: {v: !str }
nodes:
  x: {kind: code, code: "return inputs['n']", input: {n: '<n>'}, output: !str}
output: <out>
`;

const TYPED = `
id: f
name: f
input: {v: !str }
nodes:
  x:
    kind: code
    code: "return inputs['n']"
    input:
      n: !str <n>
    output: !str
output: \${x.output}
`;

const TOOL = `
id: f
name: f
input: {v: !str }
nodes:
  t: {kind: tool, tool_id: echo, input: {n: '<n>'}, output: !str}
output: \${t.output}
`;

const BODY_FLOW = `
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {w: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '<inner>'}, output: !str}
    output: \${s.output}
nodes:
  c: {kind: call, call: body, input: {w: '<w>'}}
output: \${c.output}
`;

const CALL_KWARG = `
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {w: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '\${input.w}'}, output: !str}
    output: \${s.output}
nodes:
  x:
    kind: code
    code: "return inputs['n']"
    output: !str
    input:
      n: call(body, w=<w>)
output: \${x.output}
`;

const MAP = `
id: f
name: f
input: {v: !str }
defs:
  body:
    input: {item: !str }
    nodes:
      s: {kind: code, code: "return inputs['n']", input: {n: '\${input.item}'}, output: !str}
    output: \${s.output}
nodes:
  m: {kind: map, over: '<over>', call: body, input: {item: '<item>'}}
output: \${m.output}
`;

const LOOP = `
id: f
name: f
input: {v: !str }
defs:
  step:
    input: {n: !str }
    nodes:
      s: {kind: code, code: "return {'n': inputs['n']}", input: {n: '\${input.n}'}, output: {n: !str}}
    output: \${s.output}
nodes:
  r: {kind: loop, call: step, input: {n: '<n>'}, until: "true", max: 2}
output: \${r.output}
`;

const SLEEP = `
id: f
name: f
input: {d: !str }
nodes:
  w:
    kind: tool
    tool_id: sleep
    input:
      until: '<until>'
output: ok
`;

const CASE_ON = `
id: f
name: f
input: {v: !str }
nodes:
  g:
    kind: case
    on: <on>
    cases:
      - when: "'A'"
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: !str}
  cold: {kind: code, code: "return 'cold'", output: !str}
output: \${hot.output | cold.output}
`;

const VALUE_ENTRY = `
id: f
name: f
input: {v: !str }
nodes:
  e: !str "<value>"
output: \${e.output}
`;

const AGENT = `
id: f
name: f
input: {v: !str }
nodes:
  a: {kind: agent, input: {t: '\${input.v}'}, output: !str, prompt: '<prompt>', system_prompt: '<system>'}
output: \${a.output}
`;

const HUMAN = `
id: f
name: f
input: {v: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: '<prompt>'
      questions:
        - question: '<question>'
          header: H
          options:
            - label: '<label>'
              description: '<description>'
output: \${ask.output}
`;

const ADAPTIVE = `
id: f
name: f
input: {v: !str }
nodes:
  ask:
    kind: tool
    tool_id: ask
    input:
      t: \${input.v}
    adaptive_questions:
      prompt: '<prompt>'
output: \${ask.output}
`;

const GOOD_CODE = { n: "${input.v}", out: "${x.output}" };
const GOOD_HUMAN = { prompt: "${input.v}", question: "${input.v}", label: "${input.v}", description: "${input.v}" };

/** Each site's flow and the prefix its error line carries. */
const SITES: Record<string, [text: string, where: string]> = {
  "node-input": [fill(CODE, { ...GOOD_CODE, n: "${upper(input.v, 1)}" }), "node 'x' input 'n' from: "],
  "node-input-embedded": [fill(CODE, { ...GOOD_CODE, n: "hi ${upper(input.v, 1)}" }), "node 'x' input 'n' from: "],
  "node-input-default": [
    fill(CODE, { ...GOOD_CODE, n: "${input.v :- upper(input.v, 1)}" }),
    "node 'x' input 'n' from: ",
  ],
  "nested-value": [
    fill(CODE, { ...GOOD_CODE, n: "x" }).replace("{n: 'x'}", "{n: {a: ['${upper(input.v, 1)}']}}"),
    "node 'x' input 'n' from: ",
  ],
  "ref-free": [fill(CODE, { ...GOOD_CODE, n: '${upper("a", 1)}' }), "node 'x' input 'n' from: "],
  "typed-binding": [fill(TYPED, { n: "${upper(input.v, 1)}" }), "node 'x' input 'n' from: "],
  "tool-input": [fill(TOOL, { n: "${upper(input.v, 1)}" }), "node 't' input 'n' from: "],
  "flow-output": [fill(CODE, { ...GOOD_CODE, out: "${upper(x.output, 1)}" }), "flow output from: "],
  "flow-output-record": [fill(CODE, { ...GOOD_CODE, out: "{a: '${upper(x.output, 1)}'}" }), "flow output from: "],
  "defs-body-input": [fill(BODY_FLOW, { inner: "${upper(input.w, 1)}", w: "${input.v}" }), "node 's' input 'n' from: "],
  "call-input": [fill(BODY_FLOW, { inner: "${input.w}", w: "${upper(input.v, 1)}" }), "node 'c' input 'w' from: "],
  "call-kwarg": [fill(CALL_KWARG, { w: "${upper(input.v, 1)}" }), "node 'x' input 'n' call(body) argument 'w' from: "],
  "map-over": [fill(MAP, { over: "${upper(input.v, 1)}", item: "${item}" }), "node 'm' map over: "],
  "map-item-input": [fill(MAP, { over: "${[input.v]}", item: "${upper(item, 1)}" }), "node 'm' map input 'item' from: "],
  "loop-seed": [fill(LOOP, { n: "${upper(input.v, 1)}" }), "node 'r' input 'n' from: "],
  "sleep-until": [fill(SLEEP, { until: "${upper(input.d, 1)}" }), "node 'w' input 'until' from: "],
  "case-on": [fill(CASE_ON, { on: "${upper(input.v, 1)}" }), "node 'g' on: from: "],
  "value-entry": [fill(VALUE_ENTRY, { value: "${upper(input.v, 1)}" }), "value entry 'e' from: "],
  "agent-prompt": [fill(AGENT, { prompt: "Say ${upper(t, 1)}", system: "S" }), "node 'a' prompt: "],
  "agent-system-prompt": [fill(AGENT, { prompt: "Say", system: "S ${upper(t, 1)}" }), "node 'a' system_prompt: "],
  "adaptive-brief": [fill(ADAPTIVE, { prompt: "Ask about ${upper(t, 1)}" }), "node 'ask' adaptive_questions.prompt: "],
  ...Object.fromEntries(
    (
      [
        ["prompt", "prompt"],
        ["question", "questions"],
        ["label", "questions"],
        ["description", "questions"],
      ] as const
    ).map(([site, input]) => [
      `ask-${site}`,
      [fill(HUMAN, { ...GOOD_HUMAN, [site]: "${upper(input.v, 1)}" }), `node 'ask' input '${input}' from: `],
    ]),
  ),
};

test.each(Object.keys(SITES).sort())("a mis-called builtin fails to load at its site: %s", (site) => {
  const [text, where] = SITES[site]!;
  const message = loadErrorOf(text, { located: "upper(" });
  const lines = message.split("\n").filter((line) => line.includes(UPPER));
  expect(lines.map((line) => line.trim())).toEqual([`${where}${UPPER}`]);
});

test("an unknown function in a binding is an unknown function, with the closest builtin", () => {
  const message = loadErrorOf(fill(CODE, { ...GOOD_CODE, n: "${lenght(input.v)}" }), { located: "lenght(" });
  expect(message).toBe("x: unknown expression function 'lenght'; did you mean 'len'?");
});

test("a def called in a binding keeps the flow call message", () => {
  const defs = "defs:\n  wrap:\n    input: {s: !str }\n    nodes:\n      w: !str ${input.s}\n    output: ${w.output}\n";
  const message = loadErrorOf(defs + fill(CODE, { ...GOOD_CODE, n: "${wrap(input.v)}" }));
  expect(message).toBe(
    "x: a flow call 'wrap' inside ${...} is not supported — write it as the whole value " +
      "with the call(...) directive: call(wrap, ...)",
  );
});

test("a binding reports both an unknown ref and a bad call", () => {
  const message = loadErrorOf(fill(CODE, { ...GOOD_CODE, n: "${upper(input.nosuch, 1)}" }));
  const lines = message.split("\n").map((line) => line.trim());
  expect(lines).toContain("node 'x' input 'n' from: reference ${input.nosuch} points to unknown flow input 'nosuch'");
  expect(lines).toContain(`node 'x' input 'n' from: ${UPPER}`);
});

test("well-called builtins in every site load", () => {
  const load = (text: string) => assembleText(text);
  load(fill(CODE, { n: "${upper(input.v)}", out: "${render_as_json(x.output, indent=0)}" }));
  load(fill(MAP, { over: "${[upper(input.v)]}", item: "${lower(item)}" }));
  load(fill(AGENT, { prompt: "Say ${upper(t)}", system: 'S ${join([t], sep="-")}' }));
  load(fill(HUMAN, { ...GOOD_HUMAN, question: "${upper(input.v)}" }));
  load(fill(SLEEP, { until: "${lower(input.d)}" }));
});

// ---------- bindings set in code, checked before the run ----------

const UPPER_RUN = "expression function 'upper' takes \\(s\\): too many positional arguments$";

/** Set `flow`'s binding of `nodeId`'s `key` to `source`, replacing the node's wiring map. */
function rewire(flow: Flow, nodeId: string, key: string, source: unknown): void {
  flow.wiring.set(nodeId, new Map([...flow.wiring.get(nodeId)!, [key, new Binding(source)]]));
}

/** The static child flow of the loaded node `nodeId` (a CALL or MAP). */
const childOf = (flow: ComposedFlow, nodeId: string) =>
  (flow.compiled.nodes.get(nodeId) as CallNode | MapNode).child as Flow;

test("a code-built binding mis-calling a builtin fails before the run", async () => {
  const flow = await loadFlow(fill(CODE, GOOD_CODE));
  rewire(flow.compiled, "x", "n", "${upper(input.v, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'x' input 'n': ${UPPER_RUN}`));
});

test("a code-built flow output mis-calling a builtin fails before the run", async () => {
  const flow = await loadFlow(fill(CODE, GOOD_CODE));
  rewire(flow.compiled, "__end__", "result", "${upper(x.output, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^flow output: ${UPPER_RUN}`));
});

test("a code-built record flow output names the field", async () => {
  const flow = await loadFlow(fill(CODE, { ...GOOD_CODE, out: "{a: '${x.output}'}" }));
  rewire(flow.compiled, "__end__", "a", "${upper(x.output, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^flow output 'a': ${UPPER_RUN}`));
});

test("a code-built child flow output names its owner", async () => {
  const flow = await loadFlow(fill(BODY_FLOW, { inner: "${input.w}", w: "${input.v}" }));
  rewire(childOf(flow, "c"), "__end__", "result", "${upper(s.output, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'c' flow output: ${UPPER_RUN}`));
});

test("a code-built child binding names its static path", async () => {
  const flow = await loadFlow(fill(BODY_FLOW, { inner: "${input.w}", w: "${input.v}" }));
  rewire(childOf(flow, "c"), "s", "n", "${upper(input.w, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'c/s' input 'n': ${UPPER_RUN}`));
});

test("a code-built inline call argument names the argument", async () => {
  const flow = await loadFlow(fill(CALL_KWARG, { w: "${input.v}" }));
  rewire(flow.compiled, "__call_0", "w", "${upper(input.v, 1)}");
  await expectFailsBeforeTheRun(
    flow,
    { v: "a" },
    new RegExp(`^node 'x' input 'n' call\\(body\\) argument 'w': ${UPPER_RUN}`),
  );
});

test("code-built reserved and synthetic keys use their site labels", async () => {
  const caseFlow = await loadFlow(fill(CASE_ON, { on: "${input.v}" }));
  rewire(caseFlow.compiled, "g", "__on", "${upper(input.v, 1)}");
  await expectFailsBeforeTheRun(caseFlow, { v: "a" }, new RegExp(`^node 'g' on: ${UPPER_RUN}`));

  const mapped = await loadFlow(fill(MAP, { over: "${[input.v]}", item: "${item}" }));
  rewire(mapped.compiled, "m", "item", "${upper(item, 1)}");
  await expectFailsBeforeTheRun(mapped, { v: "a" }, new RegExp(`^node 'm' map input 'item': ${UPPER_RUN}`));
});

test("a code-built reserved key constructor is checked without a type table", async () => {
  // MAP `over:` is bound with no type table, so a name in the node's table still fails.
  const flow = await loadFlow(fill(MAP, { over: "${[input.v]}", item: "${item}" }));
  const node = flow.compiled.nodes.get("m")!;
  node.types = new Map([...node.types, ["Buy", Type.scalar(ValueKind.STRING)]]);
  rewire(flow.compiled, "m", "over", "${[Buy(input.v)]}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, /^node 'm' map over: unknown expression function 'Buy'$/);
});

test("a code-built map child binding names its static path", async () => {
  const flow = await loadFlow(fill(MAP, { over: "${[input.v]}", item: "${item}" }));
  rewire(childOf(flow, "m"), "s", "n", "${upper(input.item, 1)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'm/s' input 'n': ${UPPER_RUN}`));
});

test("a code-built agent prompt mis-calling a builtin fails before the run", async () => {
  const flow = await loadFlow(fill(AGENT, { prompt: "Say ${t}", system: "S" }));
  (flow.compiled.nodes.get("a") as AgentNode).prompt = "Say ${upper(t, 1)}";
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'a' prompt: ${UPPER_RUN}`));
});

test("a code-built question text mis-calling a builtin fails before the run", async () => {
  const flow = await loadFlow(fill(HUMAN, GOOD_HUMAN));
  const question = { question: "${upper(input.v, 1)}", header: "H", options: [{ label: "a", description: "b" }] };
  rewire(flow.compiled, "ask", "questions", new ValueTemplate([question]));
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'ask' input 'questions': ${UPPER_RUN}`));
});

test("a code-built binding calling an unknown function fails before the run", async () => {
  const flow = await loadFlow(fill(CODE, GOOD_CODE));
  rewire(flow.compiled, "x", "n", "${nosuchfn(input.v)}");
  await expectFailsBeforeTheRun(flow, { v: "a" }, /^node 'x' input 'n': unknown expression function 'nosuchfn'$/);
});

test("a code-built nested binding mis-calling a builtin fails before the run", async () => {
  const flow = await loadFlow(fill(CODE, GOOD_CODE));
  rewire(flow.compiled, "x", "n", new ValueTemplate({ a: ["${upper(input.v, 1)}"] }));
  await expectFailsBeforeTheRun(flow, { v: "a" }, new RegExp(`^node 'x' input 'n': ${UPPER_RUN}`));
});

test("a well-called code-built binding still runs", async () => {
  const flow = await loadFlow(fill(CODE, GOOD_CODE));
  rewire(flow.compiled, "x", "n", "${upper(input.v)}");
  expect((await runFlow(flow, { v: "a" })).output).toBe("A");
});

const TWO_NODES = `
id: f
name: f
input:
  v: !int
nodes:
  a:
    kind: code
    code: "return inputs['n'] * 2"
    input:
      n: \${input.v}
    output: !int
  b:
    kind: code
    code: "return inputs['n'] * 2"
    input:
      n: \${a.output}
    output: !int
output: \${b.output}
`;

test("a code-built binding mis-calling a builtin fails before resume appends", async () => {
  // Built in code, so resume runs this flow, rewired while the run is stopped.
  const flow = new ComposedFlow((await loadFlow(TWO_NODES)).compiled);
  const dir = mkdtempSync(join(tmpdir(), "span-calls-"));
  try {
    const path = join(dir, "run.jsonl");
    const lines = await stoppedRunLog(flow, { v: 1 }, path, "a", () =>
      rewire(flow.compiled, "b", "n", "${render_as_json(a.output, 1, 2)}"),
    );

    const error = await rejected(advance({ log: path, flow }));
    expect(error).toBeInstanceOf(ExpressionError);
    expect((error as Error).message).toMatch(
      /^node 'b' input 'n': expression function 'render_as_json' takes \(value, indent=2\): too many positional arguments$/,
    );
    expect(readFileSync(path, "utf8")).toBe(lines);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_binding_reports_both_an_unknown_ref_and_a_bad_call",
    "mis_called_builtin_in_a_flow_output",
    "mis_called_builtin_in_a_human_question",
    "mis_called_builtin_in_a_value_entry",
    "mis_called_builtin_in_an_agent_prompt",
    "mis_called_builtin_in_case_on",
    "mis_called_builtin_in_wait_until",
    "mis_called_builtin_in_a_call_input",
    "mis_called_builtin_in_an_inline_call_argument",
    "mis_called_builtin_in_a_map_input",
    "mis_called_builtin_in_map_over",
    "well_called_builtins_in_a_map_load",
    "mis_called_builtin_in_a_def_body",
  ].map((name) => ({ id: `load-map-call/engine_span_calls__${name}`, call: assembled })),
)("assembled: $id", runCase);
