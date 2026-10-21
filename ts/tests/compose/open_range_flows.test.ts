/**
 * One-sided numeric ranges in whole flows, load-time checks and runtime checks: a port of
 * `tests/compose/test_open_range_flows.py`.
 *
 * `int(0..)` means 0 or more and `int(..100)` at most 100. Flow inputs, node input tags, bindings,
 * typedefs, cross-flow calls, case `when:` values, and AGENT outputs treat an open end as unbounded
 * on that side, with the same rules as two-sided ranges.
 *
 * Ported: every row but those below; the CODE bodies are Python's, which read the same in JS
 * (decision 11), but for the typedef flow's sum, which is the harvested case's JS body. The AGENT
 * model is a `ScriptedModel` replying the record as text, where Python stubs the model's
 * structured output.
 *
 * Changed: `test_flow_input_inside_an_open_range_runs` gives `[3n, 50n, 0n]`: the ints stay ints in
 * the `list(float)` output (decision 38), and `0.0` reaches the CODE body as the `number` 0, which
 * reads back as the int 0 (decisions 45, 9A). For the same reason the typedef flow's `price` reads
 * back as an int when it is whole. `test_agent_reply_below_an_open_range_fails_structured_output`
 * fails `agent.output_invalid` (decision 36), where Python fails `node.execute` with a pydantic
 * `ValidationError`, and its message is yian's own range message naming `qty`, where Python's is
 * pydantic's ("greater than or equal to 1").
 *
 * Covered by `run-core/compose_open_range_flows__flow_input_below_an_open_range_fails`:
 * `test_flow_input_below_an_open_range_fails_the_input_contract`; by
 * `types/compose_open_range_flows__tagged_literal_outside_an_open_range`:
 * `test_tagged_literal_outside_an_open_range_is_a_load_error`; by
 * `run-core/compose_open_range_flows__tagged_binding_below_an_open_range_fails`:
 * `test_tagged_binding_checks_an_open_range_at_runtime`; by
 * `types/compose_open_range_flows__tagged_binding_disjoint_from_an_open_range` (pinned in
 * `typed_bindings_check.test.ts`): `test_tagged_binding_disjoint_from_an_open_range_is_a_load_error`;
 * by `run-core/compose_open_range_flows__record_field_below_an_open_range_fails` (the sum in JS):
 * `test_record_field_below_an_open_range_fails_the_output_contract`; by
 * `types/compose_open_range_flows__call_with_a_wider_open_range_source`:
 * `test_call_with_a_wider_open_range_source_is_a_load_error`; by
 * `types/compose_open_range_flows__case_when_outside_an_open_range`:
 * `test_case_when_outside_an_open_range_is_a_load_error`; by
 * `types/compose_open_range_flows__no_ends_in_a_tag`: `test_no_ends_in_a_tag_is_a_load_error`; by
 * `run-agent/compose_open_range_flows__an_agent_reply_inside_open_ranges_runs`:
 * `test_agent_reply_inside_open_ranges_runs`.
 */

import { expect, test } from "bun:test";

import type { Failure } from "../../src/diagnostics.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";
import { loadConformanceFlow, loadErrorAsync } from "./declarations.ts";

/** Load `text`, expecting a load error, and return its message. */
const loadErrorMessage = async (text: string): Promise<string> =>
  (await loadErrorAsync(loadConformanceFlow(text))).message;

/** Load and run `text`, expecting a failed run, and return its failure. */
async function failure(
  text: string,
  inputs: Record<string, unknown> = {},
  options: RunFlowOptions = {},
): Promise<Failure> {
  const result = await runFlow(await loadConformanceFlow(text), inputs, options);
  expect(result.status).toBe("failed");
  expect(result.failure).not.toBeNull();
  return result.failure!;
}

/** Load and run `text`, expecting success, and return the flow output. */
async function output(text: string, inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text), inputs);
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

// ---------- flow inputs ----------

/** Return a flow with one-sided flow inputs that a code node echoes. */
const inputFlow = (mDefault = "50") => `
id: open_inputs
name: open_inputs
input:
  n: !int(0..)
  m: !int(..100) ${mDefault}
  f: !float(0.0..) 0.0
nodes:
  echo:
    kind: code
    input:
      n: \${input.n}
      m: \${input.m}
      f: \${input.f}
    code: |
      return [inputs["n"], inputs["m"], inputs["f"]]
    output: list(float)
output: \${echo.output}
`;

test("flow input inside an open range runs", async () => {
  expect(await output(inputFlow(), { n: 3 })).toEqual([3n, 50n, 0n]);
});

test("flow input default above an open range is a load error", async () => {
  expect(await loadErrorMessage(inputFlow("101"))).toContain("'m': `101` is not a valid int(..100)");
});

// ---------- node input tags and bindings ----------

/** Return a flow whose code node input is the tagged literal `!<tag> value`. */
const taggedLiteralFlow = (value: string, tag = "int(0..)") => `
id: tagged_literal
name: tagged_literal
nodes:
  use:
    kind: code
    input:
      x: !${tag} ${value}
    code: |
      return inputs["x"]
    output: int
output: \${use.output}
`;

test("tagged literal inside an open range runs", async () => {
  expect(await output(taggedLiteralFlow("5"))).toBe(5n);
});

/**
 * Return a flow where `use` binds `x: !<tag> ${a.output}` from a producer `a`, which returns the
 * flow input `v`, typed by `producerOutput`.
 */
const bindingFlow = (producerOutput: string, tag: string) => `
id: tagged_binding
name: tagged_binding
input:
  v: !int
nodes:
  a:
    kind: code
    input:
      v: \${input.v}
    code: |
      return inputs["v"]
    output: ${producerOutput}
  use:
    kind: code
    input:
      x: !${tag} \${a.output}
    code: |
      return inputs["x"]
    output: int
output: \${use.output}
`;

test("tagged binding narrows an open range at runtime", async () => {
  const text = bindingFlow("int(0..)", "int(0..100)");

  const failed = await failure(text, { v: 500 });

  expect(failed.code).toBe("binding.type");
  expect(failed.message).toContain("500 is outside inclusive int bounds (0, 100)");
  expect(await output(text, { v: 5 })).toBe(5n);
});

// ---------- typedefs ----------

/**
 * Return a flow whose code node returns an `Order` record with `qty`; the price is 1.0 plus the
 * `counts` items and `maybe`, so tests can see those inputs.
 */
const typedefFlow = (qty: number) => `
id: open_typedefs
name: open_typedefs
typedefs:
  Count: int(0..)
  Order:
    qty: int(1..)
    price: float(..100.0)
input:
  counts: !list(Count) []
  maybe: !Optional(Count) null
nodes:
  order:
    kind: code
    input:
      counts: \${input.counts}
      maybe: \${input.maybe}
    code: |
      const extra = inputs["counts"].reduce((a, b) => a + b, 0) + (inputs["maybe"] ?? 0)
      return {"qty": ${qty}, "price": 1.0 + Number(extra)}
    output: Order
output: \${order.output}
`;

test("record with open range fields runs", async () => {
  expect(await output(typedefFlow(2))).toEqual({ qty: 2n, price: 1n });
});

test("list and optional of an open range alias accept their values", async () => {
  expect(await output(typedefFlow(2), { counts: [0, 5], maybe: 4 })).toEqual({ qty: 2n, price: 10n });
});

test("list of an open range alias rejects a negative item", async () => {
  const failed = await failure(typedefFlow(2), { counts: [0, -5] });

  expect(failed.code).toBe("contract.input");
  expect(failed.message).toContain("-5 is below the inclusive int minimum 0");
});

// ---------- cross-flow call ----------

/** Return a flow that calls a child taking `t: !<childType>` from a producer `a`. */
const callFlow = (sourceOutput: string, childType: string, binding = "${a.output}") => `
id: open_call
name: open_call
input:
  v: !int
defs:
  child:
    input:
      t: !${childType}
    nodes:
      echo:
        kind: code
        input:
          t: \${input.t}
        code: |
          return inputs["t"]
        output: int
    output: \${echo.output}
nodes:
  a:
    kind: code
    input:
      v: \${input.v}
    code: |
      return inputs["v"]
    output: ${sourceOutput}
  c:
    kind: call
    call: child
    input:
      t: ${binding}
output: \${c.output}
`;

test("call with a contained open range source runs", async () => {
  expect(await output(callFlow("int(5..)", "int(0..)"), { v: 7 })).toBe(7n);
});

test("call with an unbounded source into an open range is a load error", async () => {
  expect(await loadErrorMessage(callFlow("int", "int(0..)"))).toContain("child expects 'int(0..)', source is 'int'");
});

test("call with a tagged wider source is checked at runtime", async () => {
  const text = callFlow("int(0..)", "int(0..100)", "!int(0..100) ${a.output}");

  expect(await output(text, { v: 5 })).toBe(5n);
  const failed = await failure(text, { v: 500 });
  expect(failed.code).toBe("binding.type");
  expect(failed.message).toContain("500 is outside inclusive int bounds (0, 100)");
});

// ---------- case `when:` membership ----------

/** Return a flow whose case routes `on:` an `int(0..)` input with `when: <when>`. */
const caseFlow = (when: string) => `
id: open_case
name: open_case
input:
  n: !int(0..)
nodes:
  route:
    kind: case
    on: \${input.n}
    cases:
      - when: ${when}
        then: hit
    else: miss
  hit:
    kind: code
    code: |
      return "hit"
    output: str
  miss:
    kind: code
    code: |
      return "miss"
    output: str
output: \${route.output}
`;

test("case when inside an open range runs", async () => {
  expect(await output(caseFlow("3"), { n: 3 })).toBe("hit");
});

// ---------- `int(..)` ----------

test("no ends in a typedef is a load error", async () => {
  const text = `
id: no_ends
name: no_ends
typedefs:
  Whole: int(..)
nodes:
  use:
    kind: code
    code: |
      return 1
    output: Whole
output: \${use.output}
`;
  expect(await loadErrorMessage(text)).toContain("write int");
});

// ---------- AGENT structured output ----------

const AGENT_FLOW = `
id: open_agent
name: open_agent
typedefs:
  Order:
    qty: int(1..)
    price: float(..100.0)
nodes:
  writer:
    kind: agent
    prompt: write an order
    retries: 0
    llm_config:
      provider: openai
      model: gpt-4o
    output: Order
output: \${writer.output}
`;

/** Run options whose every AGENT model replies `reply` as its text. */
function replying(reply: Record<string, unknown>): RunFlowOptions {
  const message = new Message({ role: "assistant", parts: [new TextPart(JSON.stringify(reply))] });
  const model = new ScriptedModel([new Response({ message, finishReason: "stop" })], {
    provider: "openai",
    model: "gpt-4o",
  });
  return { models: { env: () => undefined, createModel: () => model } };
}

test("agent reply below an open range fails structured output", async () => {
  const failed = await failure(AGENT_FLOW, {}, replying({ qty: 0, price: 9.5 }));

  expect(failed.code).toBe("agent.output_invalid");
  expect(failed.message).toBe("structured output at qty: 0 is below the inclusive int minimum 1");
});
