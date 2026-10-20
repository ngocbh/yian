/**
 * A builtin called with arguments it doesn't take fails at load in every binding and prompt: a
 * port of `tests/engine/test_span_calls.py`, loading each flow with `assembleText` (`assemble` as
 * the loader calls it) where Python calls `load_flow`.
 *
 * Ported: `test_mis_called_builtin_fails_to_load_at_its_site` (every site but `defs-body-input`),
 * `test_unknown_function_in_a_binding_keeps_the_flow_call_message`,
 * `test_a_binding_reports_both_an_unknown_ref_and_a_bad_call` and
 * `test_well_called_builtins_in_every_site_load`.
 *
 * Changed: each `defs:` entry is a hand-built child (`handChild`) with the def's signature and
 * output contract.
 *
 * Pinned at the end: the harvested `load-map-call/engine_span_calls__*` cases the reference
 * validator decides and `well_called_builtins_in_a_map_load`, which loads; the five whose flow
 * calls a def build it by hand (`assembledWith`).
 *
 * Later (T3-22): the `defs-body-input` site of `test_mis_called_builtin_fails_to_load_at_its_site`
 * and its case `load-map-call/engine_span_calls__mis_called_builtin_in_a_def_body` (a def body's
 * own assembly).
 *
 * Later (T4): the `test_python_built_*` rows and `test_a_well_called_python_built_binding_still_runs`,
 * which change a loaded flow's wiring and run it.
 */

import { expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { readTypedefs, typeFor } from "../../src/typesys/types.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  type ChildFlows,
  handChild,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";

const UPPER = "expression function 'upper' takes (s): too many positional arguments";

const STR = new ValueContract(Type.scalar(ValueKind.STRING));

/** A def over the one `str` input `name` whose node `s` returns a `str`, built by hand. */
const strBody = (name: string): ComposedFlow =>
  handChild({
    inputs: parametersFromYaml(`input:\n  ${name}: !str\n`, new Map()),
    producers: { s: STR },
    output: "${s.output}",
  });

/** The `body` def of the CALL flows. */
const BODY: ChildFlows = { body: strBody("w") };
/** The `body` def of the MAP flow. */
const MAP_BODY: ChildFlows = { body: strBody("item") };
/** The `step` def of the LOOP flow, whose node `s` returns `{n: str}`. */
const STEP: ChildFlows = {
  step: handChild({
    inputs: parametersFromYaml("input:\n  n: !str\n", new Map()),
    producers: { s: new ValueContract(typeFor("Rec", readTypedefs({ Rec: { n: "str" } }))) },
    output: "${s.output}",
  }),
};

/**
 * Load `text` with `children` as its defs, expecting a `LoadError`; return its message. With
 * `located`, the error's last range must sit on one line and cover `located`.
 */
function loadErrorOf(text: string, options: { located?: string; children?: ChildFlows } = {}): string {
  const error = loadError(() => assembleText(text, "flow.yaml", { children: options.children }));
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
  x: {kind: code, code: "return inputs['n']", input: {n: '<n>'}, output: str}
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
    output: str
output: \${x.output}
`;

const TOOL = `
id: f
name: f
input: {v: !str }
nodes:
  t: {kind: tool, tool_id: echo, input: {n: '<n>'}, output: str}
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
      s: {kind: code, code: "return inputs['n']", input: {n: '<inner>'}, output: str}
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
      s: {kind: code, code: "return inputs['n']", input: {n: '\${input.w}'}, output: str}
    output: \${s.output}
nodes:
  x:
    kind: code
    code: "return inputs['n']"
    output: str
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
      s: {kind: code, code: "return inputs['n']", input: {n: '\${input.item}'}, output: str}
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
      s: {kind: code, code: "return {'n': inputs['n']}", input: {n: '\${input.n}'}, output: {n: str}}
    output: \${s.output}
nodes:
  r: {kind: loop, call: step, input: {n: '<n>'}, until: "true", max: 2}
output: \${r.output}
`;

const WAIT = `
id: f
name: f
input: {d: !str }
nodes:
  w: {kind: wait, until: '<until>'}
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
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
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
  a: {kind: agent, input: {t: '\${input.v}'}, output: str, prompt: '<prompt>', system_prompt: '<system>'}
output: \${a.output}
`;

const HUMAN = `
id: f
name: f
input: {v: !str }
nodes:
  ask:
    kind: human_input
    input: {t: '\${input.v}'}
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
    kind: human_input
    input: {t: '\${input.v}'}
    adaptive_questions:
      prompt: '<prompt>'
output: \${ask.output}
`;

const GOOD_CODE = { n: "${input.v}", out: "${x.output}" };
const GOOD_HUMAN = { prompt: "${t}", question: "${t}", label: "${t}", description: "${t}" };

/** Each site's flow, the prefix its error line carries, and the defs it calls. */
const SITES: Record<string, [text: string, where: string, children?: ChildFlows]> = {
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
  "call-input": [
    fill(BODY_FLOW, { inner: "${input.w}", w: "${upper(input.v, 1)}" }),
    "node 'c' input 'w' from: ",
    BODY,
  ],
  "call-kwarg": [
    fill(CALL_KWARG, { w: "${upper(input.v, 1)}" }),
    "node 'x' input 'n' call(body) argument 'w' from: ",
    BODY,
  ],
  "map-over": [fill(MAP, { over: "${upper(input.v, 1)}", item: "${item}" }), "node 'm' map over: ", MAP_BODY],
  "map-item-input": [
    fill(MAP, { over: "${[input.v]}", item: "${upper(item, 1)}" }),
    "node 'm' map input 'item' from: ",
    MAP_BODY,
  ],
  "loop-seed": [fill(LOOP, { n: "${upper(input.v, 1)}" }), "node 'r' input 'n' from: ", STEP],
  "wait-until": [fill(WAIT, { until: "${upper(input.d, 1)}" }), "node 'w' wait until: "],
  "case-on": [fill(CASE_ON, { on: "${upper(input.v, 1)}" }), "node 'g' on: from: "],
  "value-entry": [fill(VALUE_ENTRY, { value: "${upper(input.v, 1)}" }), "value entry 'e' from: "],
  "agent-prompt": [fill(AGENT, { prompt: "Say ${upper(t, 1)}", system: "S" }), "node 'a' prompt: "],
  "agent-system-prompt": [fill(AGENT, { prompt: "Say", system: "S ${upper(t, 1)}" }), "node 'a' system_prompt: "],
  "adaptive-brief": [fill(ADAPTIVE, { prompt: "Ask about ${upper(t, 1)}" }), "node 'ask' adaptive_questions.prompt: "],
  ...Object.fromEntries(
    (
      [
        ["prompt", "prompt"],
        ["question", "questions[0].question"],
        ["label", "questions[0].options[0].label"],
        ["description", "questions[0].options[0].description"],
      ] as const
    ).map(([site, label]) => [
      `human-${site}`,
      [fill(HUMAN, { ...GOOD_HUMAN, [site]: "${upper(t, 1)}" }), `node 'ask' ${label}: `],
    ]),
  ),
};

test.each(Object.keys(SITES).sort())("a mis-called builtin fails to load at its site: %s", (site) => {
  const [text, where, children] = SITES[site]!;
  const message = loadErrorOf(text, { located: "upper(", children });
  const lines = message.split("\n").filter((line) => line.includes(UPPER));
  expect(lines.map((line) => line.trim())).toEqual([`${where}${UPPER}`]);
});

test("an unknown function in a binding keeps the flow call message", () => {
  const message = loadErrorOf(fill(CODE, { ...GOOD_CODE, n: "${nosuchfn(input.v)}" }));
  expect(message).toContain("a flow call 'nosuchfn' inside ${...} is not supported");
});

test("a binding reports both an unknown ref and a bad call", () => {
  const message = loadErrorOf(fill(CODE, { ...GOOD_CODE, n: "${upper(input.nosuch, 1)}" }));
  const lines = message.split("\n").map((line) => line.trim());
  expect(lines).toContain("node 'x' input 'n' from: reference ${input.nosuch} points to unknown flow input 'nosuch'");
  expect(lines).toContain(`node 'x' input 'n' from: ${UPPER}`);
});

test("well-called builtins in every site load", () => {
  const load = (text: string, children: ChildFlows = {}) => assembleText(text, "flow.yaml", { children });
  load(fill(CODE, { n: "${upper(input.v)}", out: "${render_as_json(x.output, indent=0)}" }));
  load(fill(MAP, { over: "${[upper(input.v)]}", item: "${lower(item)}" }), MAP_BODY);
  load(fill(AGENT, { prompt: "Say ${upper(t)}", system: 'S ${join([t], sep="-")}' }));
  load(fill(HUMAN, { ...GOOD_HUMAN, question: "${upper(t)}" }));
  load(fill(WAIT, { until: "${lower(input.d)}" }));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>([
  ...[
    "a_binding_reports_both_an_unknown_ref_and_a_bad_call",
    "mis_called_builtin_in_a_flow_output",
    "mis_called_builtin_in_a_human_question",
    "mis_called_builtin_in_a_value_entry",
    "mis_called_builtin_in_an_agent_prompt",
    "mis_called_builtin_in_case_on",
    "mis_called_builtin_in_wait_until",
  ].map((name) => ({ id: `load-map-call/engine_span_calls__${name}`, call: assembled })),
  ...(
    [
      ["mis_called_builtin_in_a_call_input", BODY],
      ["mis_called_builtin_in_an_inline_call_argument", BODY],
      ["mis_called_builtin_in_a_map_input", MAP_BODY],
      ["mis_called_builtin_in_map_over", MAP_BODY],
      ["well_called_builtins_in_a_map_load", MAP_BODY],
    ] as const
  ).map(([name, children]) => ({
    id: `load-map-call/engine_span_calls__${name}`,
    call: assembledWith(children),
  })),
])("assembled: $id", runCase);
