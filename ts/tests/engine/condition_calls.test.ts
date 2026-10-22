/**
 * A condition may call only a builtin, with arguments it takes, or a typedef constructor: a port of
 * `tests/engine/test_condition_calls.py`, and the builtin registry's argument binder
 * (`bindArguments`) and `registerTemplateFn`.
 *
 * Ported: the `parseCondition` tests; `test_unknown_function_in_a_condition_fails_to_load` and
 * `test_a_constructor_naming_no_typedef_in_a_condition_fails_to_load`, through `assembleText`
 * where Python loads the flow (the loop sites' `step` def a hand-built child, `handChild`);
 * `test_whole_span_unknown_call_in_an_assert_keeps_the_inline_call_message`; and
 * `test_builtin_with_the_wrong_arguments_in_a_condition_fails_to_load`. Ported through `loadFlow`
 * and `runFlow`: `test_a_typedef_constructor_in_a_condition_loads_and_runs` (the CASE's type
 * table, then the runs), `test_a_condition_calling_a_builtin_correctly_loads_and_runs`, and the
 * two `test_python_built_*` rows, which set a loaded node's `preAsserts` or `cases` in code, as
 * Python sets them on the compiled node, and expect the run to throw before its log is written.
 * `test_a_builtin_without_a_readable_signature_is_not_arity_checked` has no counterpart: every
 * registered function declares its `params`.
 *
 * Changed: the CODE references and code are TS's (`./fixtures/compose_codefns.ts:double`).
 *
 * The harvested cases are pinned at the end: `load-case/engine_condition_calls__wrong_builtin_arguments_in_a_case_when`
 * (`assemble`, whose CASE build raises it), `__wrong_builtin_arguments_in_a_loop_while`
 * (`assembledWith`, the `step` def built by hand; the LOOP build raises it),
 * `__whole_span_call_in_a_flow_assert` (`assemble`, whose inline-call desugar raises it),
 * `__whole_span_call_in_a_node_assert` and `__wrong_builtin_arguments_in_a_node_assert` (the
 * node-assert check raises them), `__wrong_builtin_arguments_in_a_flow_assert` (the flow-assert
 * lowering raises it), and
 * `__unknown_function_in_a_node_assert`, `__unknown_function_in_a_flow_assert` and
 * `__unknown_function_in_a_loop_while` (the type-table pass raises them).
 */

import { afterEach, describe, expect, test } from "bun:test";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import {
  bindArguments,
  registerTemplateFn,
  TEMPLATE_FNS,
  type TemplateFn,
  type TemplateParam,
} from "../../src/expr/builtins.ts";
import { ExpressionError, parseCondition } from "../../src/expr/expressions.ts";
import { Case, type CaseNode } from "../../src/nodes/case/node.ts";
import { Assertion, ValueContract } from "../../src/nodes/contracts.ts";
import { runFlow } from "../../src/run.ts";
import { typeFor } from "../../src/typesys/index.ts";
import { readTypedefs } from "../../src/typesys/types.ts";
import { typeEquals } from "../../src/typesys/values.ts";
import {
  assembled,
  assembledWith,
  assembleText,
  type CaseRow,
  handChild,
  loadConformanceFlow,
  loadError,
  parametersFromYaml,
  runCase,
} from "../compose/declarations.ts";
import { expectFailsBeforeTheRun } from "./run_state_fixtures.ts";

// Each registration's previous entry (or its absence), put back after the test, as Python's
// `monkeypatch.setitem` does.
const saved: [string, TemplateFn | undefined][] = [];

/** Register a stand-in `name` taking `params` for this test only. */
function standIn(name: string, params: TemplateParam[]): void {
  saved.push([name, TEMPLATE_FNS.get(name)]);
  registerTemplateFn(name, { params, call: () => null });
}

afterEach(() => {
  for (const [name, fn] of saved.reverse()) {
    if (fn === undefined) TEMPLATE_FNS.delete(name);
    else TEMPLATE_FNS.set(name, fn);
  }
  saved.length = 0;
});

/** Expect `parseCondition(condition, types)` to fail with exactly `message`. */
function expectMessage(condition: string, message: string, types: ReadonlyMap<string, unknown> | null = null): void {
  let error: unknown;
  try {
    parseCondition(condition, types);
  } catch (e) {
    error = e;
  }
  expect(error).toBeInstanceOf(ExpressionError);
  expect((error as Error).message).toBe(message);
}

test("unknown function does not parse as a condition", () => {
  for (const condition of [
    "nosuchfn(n) > 0",
    "${nosuchfn(n) > 0}",
    "true or nosuchfn(n)",
    "upper(nosuchfn(n)) == 'A'",
  ]) {
    expectMessage(condition, "unknown expression function 'nosuchfn'", new Map());
  }
});

test("a constructor parses when its type is known or left to compose", () => {
  const buy = typeFor("Buy", readTypedefs({ Buy: { qty: "int" } }));
  parseCondition("Buy(d).qty > 0", new Map([["Buy", buy]]));
  // Without a type table, compose resolves the name against the flow's typedefs.
  parseCondition("Buy(d).qty > 0");
  expectMessage("Buy(d).qty > 0", "unknown expression function 'Buy'", new Map());
});

test("a builtin type name called in a condition gets the conversion hint", () => {
  expectMessage(
    "int(x) > 0",
    "unknown expression function 'int'; types in expressions are typedef names; `int(...)` is not a conversion",
    new Map(),
  );
});

test("builtin with arguments it does not take does not parse", () => {
  const cases: [string, string][] = [
    ["upper(a, b) == 'A'", "expression function 'upper' takes (s): too many positional arguments"],
    ["upper() == 'A'", "expression function 'upper' takes (s): missing a required argument: 's'"],
    [
      "render_as_json(a, width=4) != ''",
      "expression function 'render_as_json' takes (value, indent=2): got an unexpected keyword argument 'width'",
    ],
  ];
  for (const [condition, message] of cases) expectMessage(condition, message);
});

test("a default shows in its text form", () => {
  expectMessage(
    "join(a, b, c) == ''",
    'expression function \'join\' takes (value, sep="\\n"): too many positional arguments',
  );
});

test("builtin calls that bind parse", () => {
  for (const condition of [
    "upper(s) == 'A'",
    "render_as_json(a, indent=4) != ''",
    "join(xs) == 'a'",
    "join(xs, sep=',') == 'a'",
    "isinstance(x, int)",
    "lower(upper(s)) == 'a'",
  ]) {
    expect(() => parseCondition(condition)).not.toThrow();
  }
});

test("a registered builtin is accepted", () => {
  standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
  parseCondition("shout(s, times=2) == 'AA'");
  expectMessage(
    "shout(s, 2, 3) == 'AA'",
    "expression function 'shout' takes (s, times=1): too many positional arguments",
  );
});

test("a repeated keyword does not parse", () => {
  expectMessage("upper(s=a, s=b) == 'A'", "upper(...) repeats the keyword 's'");
});

describe("bindArguments", () => {
  const fn: TemplateFn = {
    params: [{ name: "value" }, { name: "indent", default: 2n }],
    call: () => null,
  };
  const keywords = (entries: Record<string, string>) => new Map(Object.entries(entries));

  test("Python's Signature.bind messages", () => {
    const cases: [string[], Record<string, string>, string][] = [
      [["a", "b", "c"], {}, "too many positional arguments"],
      [["a"], { value: "v" }, "multiple values for argument 'value'"],
      [["a", "b", "c"], { value: "v" }, "multiple values for argument 'value'"],
      [[], {}, "missing a required argument: 'value'"],
      [[], { indent: "i", zz: "z" }, "missing a required argument: 'value'"],
      [[], { x: "x", value: "v" }, "got an unexpected keyword argument 'x'"],
    ];
    for (const [positional, named, message] of cases) {
      expect(bindArguments(fn, positional, keywords(named))).toBe(message);
    }
  });

  test("the first unknown keyword in call order is named", () => {
    expect(bindArguments(fn, ["a"], keywords({ zz: "z", yy: "y" }))).toBe("got an unexpected keyword argument 'zz'");
  });

  test("arguments come back in parameter order with defaults filled in", () => {
    expect(bindArguments(fn, ["a"], new Map())).toEqual(["a", 2n]);
    expect(bindArguments(fn, [], keywords({ indent: "i", value: "v" }))).toEqual(["v", "i"]);
    expect(bindArguments(fn, ["a", "b"], new Map())).toEqual(["a", "b"]);
  });

  test("a name is shown as Python's repr", () => {
    expect(bindArguments(fn, [], keywords({ "it's": "x", value: "v" }))).toBe(
      'got an unexpected keyword argument "it\'s"',
    );
  });
});

describe("registerTemplateFn", () => {
  test("the last writer wins", () => {
    standIn("shout", [{ name: "s" }]);
    standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
    expect(TEMPLATE_FNS.get("shout")!.params.map((p) => p.name)).toEqual(["s", "times"]);
  });

  test("a default is read as a host value", () => {
    standIn("shout", [{ name: "s" }, { name: "times", default: 1 }]);
    expect(TEMPLATE_FNS.get("shout")!.params[1]!.default).toBe(1n);
  });

  test("a default that is no yian value is rejected", () => {
    saved.push(["shout", TEMPLATE_FNS.get("shout")]);
    const fn: TemplateFn = { params: [{ name: "times", default: Number.NaN }], call: () => null };
    expect(() => registerTemplateFn("shout", fn)).toThrow(
      "expression function 'shout', default of 'times': NaN is not a yian value",
    );
  });
});

// ---------- loading ----------

const NODE_ASSERT = `
id: f
name: f
input:
  v: !int
nodes:
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
    asserts:
      - "<cond>"
output: \${x.output}
`;

const FLOW_ASSERT = `
id: f
name: f
input:
  v: !int
asserts:
  - "<cond>"
nodes:
  x:
    kind: code
    code: ./fixtures/compose_codefns.ts:double
    input:
      n: \${input.v}
    output: int
output: \${x.output}
`;

const CASE = `
id: f
name: f
input:
  v: !int
nodes:
  gate:
    kind: case
    cases:
      - when: "<cond>"
        then: hot
    else: cold
  hot: {kind: code, code: "return 'hot'", output: str}
  cold: {kind: code, code: "return 'cold'", output: str}
output: \${hot.output | cold.output}
`;

const LOOP = `
id: f
name: f
defs:
  step:
    input: {n: !int , exited: !bool }
    nodes:
      s:
        kind: code
        code: "return {n: inputs['n'] + 1, exited: inputs['n'] >= 2}"
        input:
          n: \${input.n}
        output:
          n: int
          exited: bool
    output: \${s.output}
nodes:
  repeat:
    kind: loop
    call: step
    input:
      n: 0
      exited: false
    <keyword>: "<cond>"
    max: 5
output: \${repeat.output}
`;

/** Assemble `text` (its `step` def built by hand) and return the located `LoadError` it throws. */
function loadErrorOf(text: string) {
  const error = loadError(() => assembleText(text, "flow.yaml", { children: { step: stepBody() } }));
  expect(error.diagnostic.frames.at(-1)!.range).not.toBeNull();
  return error;
}

test.each<[string, string]>([
  ["node-pre", NODE_ASSERT.replace("<cond>", "nosuchfn(n) > 0")],
  ["node-post", NODE_ASSERT.replace("<cond>", "nosuchfn(output) > 0")],
  ["flow-boundary", FLOW_ASSERT.replace("<cond>", "nosuchfn(input.v) > 0")],
  ["flow-post", FLOW_ASSERT.replace("<cond>", "nosuchfn(x.output) > 0")],
  ["case", CASE.replace("<cond>", "nosuchfn(input.v) > 0")],
  ["case-short-circuit", CASE.replace("<cond>", "true or nosuchfn(input.v)")],
  ["loop-while", LOOP.replace("<keyword>", "while").replace("<cond>", "not nosuchfn(exited)")],
  ["loop-until", LOOP.replace("<keyword>", "until").replace("<cond>", "nosuchfn(exited)")],
])("unknown function in a condition fails to load: %s", (_site, text) => {
  expect(loadErrorOf(text).message).toContain("unknown expression function 'nosuchfn'");
});

test.each<[string, string]>([
  ["node", NODE_ASSERT.replace("<cond>", "upper(n, n) == 'A'")],
  ["flow", FLOW_ASSERT.replace("<cond>", "upper(input.v, 1) == 'A'")],
  ["case", CASE.replace("<cond>", "upper('a', 'b') == 'A'")],
  ["loop", LOOP.replace("<keyword>", "while").replace("<cond>", "upper(exited, 1) == 'A'")],
])("builtin with the wrong arguments in a condition fails to load: %s", (_site, text) => {
  expect(loadErrorOf(text).message).toContain("expression function 'upper' takes (s): too many positional arguments");
});

test("a whole-span unknown call in an assert keeps the inline call message", () => {
  expect(loadErrorOf(NODE_ASSERT.replace("<cond>", "${nosuchfn(n) > 0}")).message).toContain(
    "an inline call is not allowed in a node-local assert",
  );
  expect(loadErrorOf(FLOW_ASSERT.replace("<cond>", "${nosuchfn(input.v) > 0}")).message).toContain(
    "a flow call 'nosuchfn' inside ${...} is not supported",
  );
});

/** Run the loaded `flow` on `inputs`, returning its output; the run must succeed. */
async function outputOf(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>): Promise<unknown> {
  const result = await runFlow(flow, inputs);
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

test("a condition calling a builtin correctly loads and runs", async () => {
  const flow = await loadFlow(CASE.replace("<cond>", "upper(render_as_json(input.v, indent=0)) == '2'"));

  expect(await outputOf(flow, { v: 2 })).toBe("hot");
});

test("a code-built assert calling an unknown function fails before the run", async () => {
  const flow = await loadConformanceFlow(NODE_ASSERT.replace("<cond>", "n > 0"));
  flow.compiled.nodes.get("x")!.preAsserts = [new Assertion("nosuchfn(n) > 0", "nosuchfn(n) > 0")];

  await expectFailsBeforeTheRun(
    flow,
    { v: 1 },
    /^node 'x' pre-assert 'nosuchfn\(n\) > 0' is not a valid boolean expression: unknown expression function 'nosuchfn'$/,
  );
});

test("a code-built case when with the wrong arguments fails before the run", async () => {
  const flow = await loadFlow(CASE.replace("<cond>", "input.v > 0"));
  const gate = flow.compiled.nodes.get("gate") as CaseNode;
  gate.cases = [new Case("hot", { when: "upper(__r0, 1) == 'A'", origin: gate.cases[0]!.origin })];

  await expectFailsBeforeTheRun(
    flow,
    { v: 1 },
    /^node 'gate' case 'hot' `when:` .* is not a valid boolean expression: expression function 'upper' takes \(s\)/,
  );
});

const TYPEDEF_CASE = CASE.replace("name: f\n", "name: f\ntypedefs:\n  Small: int(0..9)\n");

test("a typedef constructor in a condition loads and runs", async () => {
  const flow = await loadFlow(TYPEDEF_CASE.replace("<cond>", "Small(input.v) > 4"));

  const types = flow.compiled.nodes.get("gate")!.types;
  expect([...types.keys()]).toEqual(["Small"]);
  expect(typeEquals(types.get("Small")!, typeFor("int(0..9)", new Map()))).toBe(true);
  expect(await outputOf(flow, { v: 7 })).toBe("hot");
  expect(await outputOf(flow, { v: 2 })).toBe("cold");
});

test("a constructor naming no typedef in a condition fails to load", () => {
  const text = TYPEDEF_CASE.replace("<cond>", "Nope(input.v) > 4");
  expect(loadErrorOf(text).message).toContain("unknown expression function 'Nope'");
});

// ---------- harvested conformance cases ----------

test("assembled: load-case/engine_condition_calls__wrong_builtin_arguments_in_a_case_when", () => {
  runCase({ id: "load-case/engine_condition_calls__wrong_builtin_arguments_in_a_case_when", call: assembled });
});

/**
 * The `step` def of the LOOP case: it reads `{n, exited}`, and its producer `s` returns the record
 * `{n: int, exited: bool}`, which the def outputs.
 */
function stepBody(): ComposedFlow {
  const record = new ValueContract(typeFor("Rec", readTypedefs({ Rec: { n: "int", exited: "bool" } })));
  return handChild({
    inputs: parametersFromYaml("input:\n  n: !int\n  exited: !bool\n", new Map()),
    producers: { s: record },
    output: "${s.output}",
  });
}

test("assembled: load-case/engine_condition_calls__wrong_builtin_arguments_in_a_loop_while", () => {
  runCase({
    id: "load-case/engine_condition_calls__wrong_builtin_arguments_in_a_loop_while",
    call: assembledWith({ step: stepBody() }),
  });
});

test("assembled: load-case/engine_condition_calls__whole_span_call_in_a_flow_assert", () => {
  runCase({ id: "load-case/engine_condition_calls__whole_span_call_in_a_flow_assert", call: assembled });
});

test.each<CaseRow>([
  { id: "load-case/engine_condition_calls__whole_span_call_in_a_node_assert", call: assembled },
  { id: "load-case/engine_condition_calls__wrong_builtin_arguments_in_a_node_assert", call: assembled },
  { id: "load-case/engine_condition_calls__wrong_builtin_arguments_in_a_flow_assert", call: assembled },
  { id: "load-case/engine_condition_calls__unknown_function_in_a_node_assert", call: assembled },
  { id: "load-case/engine_condition_calls__unknown_function_in_a_flow_assert", call: assembled },
  {
    id: "load-case/engine_condition_calls__unknown_function_in_a_loop_while",
    call: assembledWith({ step: stepBody() }),
  },
])("assembled: $id", runCase);
