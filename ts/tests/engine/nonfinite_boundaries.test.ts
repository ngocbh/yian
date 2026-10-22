/**
 * Non-finite numbers at the binding boundary, a port of the two `apply_defaults` tests of
 * `tests/engine/test_nonfinite_boundaries.py`.
 *
 * Changed (rule: infinity and minus infinity are float values, NaN is not a yian value):
 * `test_apply_defaults_rejects_a_nonfinite_default` now shows an `Infinity` default accepted
 * and keeps the rejection as a NaN row; a NaN default fails with host intake's message
 * (`default for 'p': NaN is not a yian value`), also before its contract is checked.
 *
 * The run rows, through `loadFlow` and `runFlow` in the test's environment (JS CODE bodies; an
 * AGENT's model a `ScriptedModel` given as `models`, where Python patches `run_agent`), each with
 * the TS expectation of that rule (header "Changed" for every row whose Python run fails on an
 * infinity):
 *
 * - Ported with TS's NaN message (`NaN is not a yian value`, Python's `non-finite number nan is
 *   not a yian value`): `test_a_nonfinite_object_output_fails_the_node_at_the_field`,
 *   `test_a_nonfinite_tool_output_fails_the_node` and `test_a_nonfinite_float_output_fails_the_node`;
 *   the `nan` row of `test_a_nonfinite_run_input_is_rejected_before_the_run_starts` (a
 *   `TypeCheckError`, no log created).
 * - Changed, an infinity is a value, so the run succeeds with it:
 *   `test_a_float_output_too_large_for_float_fails_the_node` and
 *   `test_a_record_field_too_large_for_float_names_the_field` (JS's `10**400` is Infinity),
 *   `test_a_nonfinite_undeclared_output_fails_the_node_at_the_item`,
 *   `test_a_nonfinite_value_inside_a_record_field_names_the_path` (the reader runs),
 *   `test_a_binding_that_evaluates_to_infinity_fails_its_node` (both rows),
 *   `test_a_call_binding_that_evaluates_to_infinity_fails_the_call` (the `float` row),
 *   `test_a_nonfinite_start_default_fails_start`, `test_a_nonfinite_map_over_item_fails_the_map`,
 *   `test_a_nonfinite_map_item_binding_fails_the_map` (the other item, `2.0 * 10`, comes back from
 *   the CODE body as the int `20n`: a CODE body gets a whole float as a plain number, decision 45),
 *   `test_a_nonfinite_temporary_inside_a_template_is_allowed` (the text form is `v=Infinity`),
 *   `test_a_nonfinite_agent_variable_default_fails_the_agent` (both rows; the model sees
 *   `go Infinity`), and the `coerced-string` (text `inf` reads as Infinity) and `nested-object`
 *   rows of `test_a_nonfinite_run_input_is_rejected_before_the_run_starts`.
 * - Changed otherwise: the `int` row of `test_a_call_binding_that_evaluates_to_infinity_fails_the_call`
 *   fails the child START's input contract (`contract.input` at `k/__start__`, Infinity isn't an
 *   int) where Python fails the CALL's binding; the `list` row of
 *   `test_a_nonfinite_run_input_is_rejected_before_the_run_starts` starts the run and START fails
 *   `contract.input` (a list isn't an `object`); `test_a_learnable_literal_nan_binding_fails_its_node`
 *   and `test_a_literal_binding_names_the_nonfinite_path` fail at load (YAML's `.nan` is refused);
 *   `test_a_run_input_too_large_for_float_fails_at_start` and
 *   `test_an_undeclared_output_widening_past_float_fails_the_node` pass the int `10n ** 400n`,
 *   which stays an int (a whole number in a `float` slot stays an int, and no int widens to a
 *   float), so both succeed.
 *
 * Covered by `run-core/engine_nonfinite_boundaries__a_flow_output_binding_overflows_to_infinity`:
 * `test_a_nonfinite_flow_output_binding_fails_the_end_binding` (TS succeeds with Infinity).
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { LoadError } from "../../src/compose/helpers/errors.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { type TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { applyDefaults, BindingError } from "../../src/nodes/binding.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { type RunResult, runFlow, type RunFlowOptions } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { Type, TypeCheckError, ValueKind } from "../../src/typesys/index.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected, serving, startedIds, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-nonfinite-boundaries-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Return the `BindingError` `call` throws. */
function bindingError(call: () => unknown): BindingError {
  try {
    call();
  } catch (error) {
    expect(error).toBeInstanceOf(BindingError);
    return error as BindingError;
  }
  throw new Error("call did not throw");
}

test("applyDefaults accepts an infinite default and rejects a NaN default", () => {
  expect(applyDefaults([new Parameter("p", { hasDefault: true, default: Infinity })], {})).toEqual({ p: Infinity });

  const error = bindingError(() => applyDefaults([new Parameter("p", { hasDefault: true, default: NaN })], {}));
  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message).toBe("default for 'p': NaN is not a yian value");
});

test("applyDefaults rejects a NaN default before its contract", () => {
  const parameter = new Parameter("p", {
    contract: new ValueContract(Type.scalar(ValueKind.INTEGER)),
    hasDefault: true,
    default: NaN,
  });

  const error = bindingError(() => applyDefaults([parameter], {}));
  expect(error.message).toBe("default for 'p': NaN is not a yian value");
});

// --- runs ------------------------------------------------------------------------------- //

/** Load `source` and run it in the test's environment. */
async function run(source: string, inputs: Record<string, unknown> = {}, options: RunFlowOptions = {}) {
  return runFlow(await loadFlow(source), inputs, { env: hostEnv(dir), ...options });
}

/** Assert the run succeeded with `output`. */
function expectSucceeds(result: RunResult, output: unknown): void {
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toEqual(output);
}

/** A flow whose `n` CODE node declares `output` and returns `value`. */
const codeFlow = (output: string, value: string) => `
id: overflow
name: overflow
nodes:
  n:
    kind: code
    output: ${output}
    code: |
      return ${value}
output: \${n.output}
`;

test("a float output too large for a finite float is Infinity", async () => {
  expectSucceeds(await run(codeFlow("float", "10**400")), Infinity);
});

test("a record field too large for a finite float is Infinity", async () => {
  expectSucceeds(await run(codeFlow("{stats: float}", '{"stats": 10**400}')), { stats: Infinity });
});

test("an undeclared output with a huge int keeps the int", async () => {
  const result = await run(`
id: overflow
name: overflow
nodes:
  n:
    kind: code
    code: |
      return [0.5, 10n ** 400n]
output: \${n.output}
`);

  expectSucceeds(result, [0.5, 10n ** 400n]);
});

/** A flow whose run input `x: declared` feeds one CODE node. */
const inputFlow = (declared: string) => `
id: nonfinite
name: nonfinite
input:
  x: !${declared}
nodes:
  c:
    kind: code
    input:
      y: \${input.x}
    code: |
      return inputs["y"]
output: \${c.output}
`;

test("a huge int run input for a float stays an int", async () => {
  expectSucceeds(await run(inputFlow("float"), { x: 10n ** 400n }), 10n ** 400n);
});

/** A flow whose `c` CODE node returns `value` and whose `d` node reads it. */
const readerFlow = (outputLine: string, value: string) => `
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
${outputLine}
    code: |
      return ${value}
  d:
    kind: code
    input:
      seen: \${c.output}
    code: |
      return inputs["seen"]
output: \${d.output}
`;

/** Assert `c` failed its output contract with `message` and `d` never ran. */
function expectOutputFailsBeforeReader(result: RunResult, message: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.nodeId).toBe("c");
  expect(result.failure!.message).toBe(message);
  expect(startedIds(result)).not.toContain("d");
}

test("a NaN object output fails the node at the field", async () => {
  const result = await run(readerFlow("    output: object", '{"a": NaN}'));

  expectOutputFailsBeforeReader(result, "output at a: NaN is not a yian value");
  expect(result.failure!.trace[0]!.key.endsWith("/s:output/s:contract/s:root")).toBe(true);
});

test("an infinite undeclared output reaches the reader", async () => {
  const result = await run(readerFlow("", '[{"a": -Infinity}]'));

  expectSucceeds(result, [{ a: -Infinity }]);
  expect(startedIds(result)).toContain("d");
});

test("an infinity inside a record field reaches the reader", async () => {
  const result = await run(readerFlow("    output: {stats: object}", '{"stats": {"a": Infinity}}'));

  expectSucceeds(result, { stats: { a: Infinity } });
  expect(startedIds(result)).toContain("d");
});

test("a NaN tool output fails the node", async () => {
  const measure = tool({
    name: "measure",
    description: "Measure one thing.",
    parameters: { type: "object", properties: { qty: { type: "integer" } } },
    execute: () => ({ a: NaN }),
  });

  const result = await run(
    `
id: nonfinite
name: nonfinite
nodes:
  t:
    kind: tool
    tool_id: measure
    input:
      qty: 5
output: \${t.output}
`,
    {},
    { tools: { measure } },
  );

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.nodeId).toBe("t");
  expect(result.failure!.message).toBe("output at a: NaN is not a yian value");
});

test("a NaN float output fails the node", async () => {
  const result = await run(codeFlow("float", "NaN"));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.output");
  expect(result.failure!.nodeId).toBe("n");
  expect(result.failure!.message).toBe("output: NaN is not a yian value");
});

const SCALED_BINDING_FLOW = `
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  c:
    kind: code
    input:
      y: \${input.x * 10}
    code: |
      return inputs["y"]
output: \${c.output}
`;

for (const learnable of [false, true]) {
  test(`a binding that evaluates to infinity binds it (learnable: ${learnable})`, async () => {
    expectSucceeds(await run(SCALED_BINDING_FLOW, { x: 1e308 }, { learnable }), Infinity);
  });
}

/** The `LoadError` loading `source` rejects with. */
async function loadError(source: string): Promise<LoadError> {
  const error = await rejected(loadFlow(source));
  expect(error).toBeInstanceOf(LoadError);
  return error as LoadError;
}

test("a literal NaN binding fails at load", async () => {
  const error = await loadError(`
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
    input:
      y: .nan
    code: |
      return inputs["y"]
output: \${c.output}
`);

  expect(error.message).toBe("invalid YAML: NaN is not a yian value; quote it to keep it as text");
});

/** A flow whose `k` CALL node binds `input.x * 10` to a def input `y: declared`. */
const callFlow = (declared: string) => `
id: nonfinite
name: nonfinite
input:
  x: !float
defs:
  body:
    input:
      y: !${declared}
    nodes:
      inner:
        kind: code
        input:
          y: \${input.y}
        code: |
          return inputs["y"]
    output: \${inner.output}
nodes:
  k:
    kind: call
    call: body
    input:
      y: \${input.x * 10}
output: \${k.output}
`;

test("a call binding that evaluates to infinity reaches a float input", async () => {
  expectSucceeds(await run(callFlow("float"), { x: 1e308 }), Infinity);
});

test("a call binding that evaluates to infinity fails an int input at the child START", async () => {
  const result = await run(callFlow("int"), { x: 1e308 });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.nodeId).toBe("k/__start__");
  expect(result.failure!.message).toBe("input 'y': Infinity does not match declared type int");
  expect(startedIds(result)).toEqual(["__start__", "k", "k/__start__"]);
});

test("a nested literal NaN binding fails at load", async () => {
  const error = await loadError(`
id: nonfinite
name: nonfinite
nodes:
  c:
    kind: code
    input:
      y: {a: [1, .nan]}
    code: |
      return inputs["y"]
output: \${c.output}
`);

  expect(error.message).toBe("invalid YAML: NaN is not a yian value; quote it to keep it as text");
});

test("an infinite START default is used", async () => {
  expectSucceeds(await run(inputFlow("float .inf")), Infinity);
});

/** A flow whose `m` MAP node iterates `over` and binds `y` to `itemBinding`. */
const mapFlow = (over: string, itemBinding: string) => `
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  m:
    kind: map
    over: "${over}"
    input:
      y: !float ${itemBinding}
    nodes:
      work:
        kind: code
        input:
          y: \${input.y}
        code: |
          return inputs["y"]
    output: \${work.output}
output: \${m.output}
`;

test("an infinite MAP over item is mapped", async () => {
  expectSucceeds(await run(mapFlow("${[input.x, input.x * 10]}", "${item}"), { x: 1e308 }), [1e308, Infinity]);
});

test("an infinite MAP item binding is mapped", async () => {
  // `2.0 * 10` reaches the CODE body as the plain number 20, which comes back as an int.
  expectSucceeds(await run(mapFlow("${[input.x, 2.0]}", "${item * 10}"), { x: 1e308 }), [Infinity, 20n]);
});

test("an infinite temporary inside a template renders as Infinity", async () => {
  const result = await run(
    `
id: nonfinite
name: nonfinite
input:
  x: !float
nodes:
  c:
    kind: code
    input:
      y: v=\${input.x * 10}
    code: |
      return inputs["y"]
output: \${c.output}
`,
    { x: 1e308 },
  );

  expectSucceeds(result, "v=Infinity");
});

for (const learnable of [false, true]) {
  test(`an infinite AGENT variable default reaches the prompt (learnable: ${learnable})`, async () => {
    const model = new ScriptedModel([textReply("x")], { provider: "ollama", model: "test" });

    const result = await run(
      `
id: nonfinite
name: nonfinite
llm_config: {provider: ollama, model: test}
nodes:
  writer:
    kind: agent
    variables:
      t: !float .inf
    prompt: 'go \${t}'
    output: str
output: \${writer.output}
`,
      {},
      { learnable, models: serving(model) },
    );

    expectSucceeds(result, "x");
    expect(model.requests).toHaveLength(1);
    expect((model.requests[0]!.messages.at(-1)!.parts[0] as TextPart).text).toBe("go Infinity");
  });
}

test("a NaN run input is rejected before the run starts", async () => {
  const log = join(dir, "nan.jsonl");

  const error = await rejected(run(inputFlow("float"), { x: NaN }, { log }));

  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toBe("input 'x': NaN is not a yian value");
  expect(existsSync(log)).toBe(false);
});

for (const [name, declared, value, output] of [
  ["coerced-string", "float", "inf", Infinity],
  ["nested-object", "object", { a: [1, Infinity] }, { a: [1n, Infinity] }],
] as const) {
  test(`an infinite run input is taken: ${name}`, async () => {
    expectSucceeds(await run(inputFlow(declared), { x: value }), output);
  });
}

test("an infinite list run input reaches START, which refuses a list for an object", async () => {
  const result = await run(inputFlow("object"), { x: [1.0, -Infinity] });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.nodeId).toBe("__start__");
  expect(result.failure!.message).toBe("input 'x': [1,-Infinity] does not match declared type object");
});
