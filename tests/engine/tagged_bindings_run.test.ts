/**
 * Tagged bindings checked when a run binds a node's inputs or the flow's outputs: a port of
 * `tests/engine/test_tagged_bindings_run.py`.
 *
 * A tag (`qty: !int ${size.output.qty}`) declares the value's type. The binder checks the resolved
 * value against it without converting or filling anything in, and a mismatch fails with code
 * `binding.type`, located on the tagged binding.
 *
 * Ported: every row but those covered below. The CODE bodies are Python's, which read the same in
 * JS (decision 11), but for `return null` and the gate's producer, which parses digits with a
 * regular expression and `Number`. The AGENT's model is a `ScriptedModel` given as `models`, and
 * the rows read the rendered prompt from its request, where Python replaces `run_agent` with an
 * echo of the prompt; "before the model call" is no model built. The ask request is camelCase,
 * and `place_order` is a `FunctionTool` taking any `qty`.
 *
 * Changed: `test_tagged_input_does_not_convert_an_int_declared_float` keeps the int, as Python
 * does; an int is a `bigint` (decision 4a), so the output is `30n`, and every int output reads as
 * a `bigint`.
 * The row can't tell an int from a whole float, since a CODE body sees both as a JS `number`
 * (decision 45) and a whole number it returns reads back as an int (decision 9A); decision 38 is
 * pinned by `values/int_fills_float` and the flow-output rows of `typed_flow_outputs_run.test.ts`.
 * The gate is a TOOL `ask` where Python's is a HUMAN_INPUT node (TS yian has no HUMAN_INPUT kind):
 * its tagged `count` is an ask input the handler declares, and its prompt reads `${src.output}`.
 *
 * Covered by `run-core/engine_tagged_bindings_run__narrower_source_fails_on_its_tagged_line`:
 * `test_narrower_source_fails_on_its_tagged_line`; by
 * `run-core/engine_tagged_bindings_run__tag_does_not_fill_in_a_missing_field`:
 * `test_tag_does_not_fill_in_a_missing_field`; by
 * `run-core/engine_tagged_bindings_run__isinstance_reads_a_record_without_the_field_as_false`:
 * `test_isinstance_still_reads_a_record_without_the_field_as_false`; by
 * `run-core/engine_tagged_bindings_run__tagged_map_item_arg_rejects_a_mistyped_item`:
 * `test_tagged_map_item_arg_rejects_a_mistyped_item`; by
 * `run-core/engine_tagged_bindings_run__tagged_mapping_output_returns_one_value`:
 * `test_tagged_mapping_output_returns_the_literal_as_one_value`. The `run-leaves` cases
 * `tagged_bindings_run__a_tagged_tool_input_fails_before_the_tool_runs` and
 * `tagged_bindings_run__a_tagged_human_input_context_fails_before_asking` stand an `!Any` flow
 * input in for the CODE source, and `run-core/engine_tagged_bindings_run__tagged_agent_input_fails_before_the_model_call`
 * can't pin that no model was built, so their rows are ported.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import type { TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import { tool } from "../../src/tools/function_tools.ts";
import type { AskRequest } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { serving, textReply } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-tagged-bindings-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Load `text` and run it in the test's environment. */
async function run(text: string, inputs: Record<string, unknown> = {}, options: RunFlowOptions = {}): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env, ...options });
}

/** Assert `result` succeeded and return its output. */
function succeeded(result: RunResult): unknown {
  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  return result.output;
}

/** Assert `result` failed with `binding.type` (at `nodeId` when given). */
function expectTypeFailure(result: RunResult, nodeId?: string): void {
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  if (nodeId !== undefined) expect(result.failure!.nodeId).toBe(nodeId);
}

/** Return a flow whose producer `p` returns `value` into `c`'s `x: <binding>`. */
const consumerFlow = (value: string, binding: string) => `
id: tagged_consumer
name: tagged_consumer
nodes:
  p:
    kind: code
    code: |
      return ${value}
  c:
    kind: code
    input:
      x: ${binding}
    code: |
      return inputs["x"]
output: \${c.output}
`;

test("tagged code input passes a matching value", async () => {
  expect(succeeded(await run(consumerFlow("5", "!int ${p.output}")))).toBe(5n);
});

test("tagged input does not convert an int declared float", async () => {
  expect(succeeded(await run(consumerFlow("30", "!float ${p.output}")))).toBe(30n);
});

const BUY_FLOW = `
id: tagged_buy
name: tagged_buy
typedefs:
  Buy:
    action: Literal(buy)
    conviction: float
nodes:
  d:
    kind: code
    output: object
    code: |
      return {"conviction": 0.5}
  use:
    kind: code
    input:
      decision: !Buy \${d.output}
    code: |
      return inputs["decision"]
output: \${use.output}
`;

test("tag passes a record with every field unchanged", async () => {
  const text = BUY_FLOW.replace('return {"conviction": 0.5}', 'return {"action": "buy", "conviction": 0.5}');

  expect(succeeded(await run(text))).toEqual({ action: "buy", conviction: 0.5 });
});

/** Return a flow whose producer returns `value` into an AGENT's tagged input. */
const agentFlow = (value: string) => `
id: tagged_agent
name: tagged_agent
nodes:
  src:
    kind: code
    code: |
      return ${value}
  ask:
    kind: agent
    input:
      count: !int \${src.output}
    prompt: "Count \${count}"
    output: str
output: \${ask.output}
`;

// The run resolves the AGENT's provider and model before the node binds its inputs.
const agentEnv = hostEnv(dir, { YIAN_DEFAULT_PROVIDER: "openai", YIAN_DEFAULT_MODEL: "gpt-5.5" });

test("tagged agent input renders a matching value", async () => {
  const model = new ScriptedModel([textReply("ok")]);
  const result = await runFlow(await loadFlow(agentFlow("3"), { env: agentEnv }), {}, {
    env: agentEnv,
    models: serving(model, agentEnv),
  });

  expect(succeeded(result)).toBe("ok");
  const prompt = model.requests[0]!.messages.at(-1)!;
  expect(prompt.role).toBe("user");
  expect((prompt.parts[0] as TextPart).text).toBe("Count 3");
});

test("tagged agent input fails before the model call", async () => {
  const models = serving(new ScriptedModel([]), agentEnv);
  const result = await runFlow(await loadFlow(agentFlow("'three'"), { env: agentEnv }), {}, { env: agentEnv, models });

  expectTypeFailure(result, "ask");
  expect(models.built).toBe(0);
});

/** Return a flow whose TOOL node binds `qty` as `binding`. */
const toolFlow = (binding: string) => `
id: tagged_tool
name: tagged_tool
nodes:
  src:
    kind: code
    code: |
      return "5"
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: ${binding}
output: \${place.output}
`;

/** A `place_order` tool that hands each `qty` to `execute`. */
const placeOrder = (execute: (qty: unknown) => unknown) =>
  tool({
    name: "place_order",
    description: "Place an order.",
    parameters: { type: "object", properties: { qty: {} }, required: ["qty"] },
    execute: (args: { qty?: unknown }) => execute(args.qty),
  });

test("tagged tool input passes a literal", async () => {
  const result = await run(toolFlow("!int 5"), {}, { tools: { place_order: placeOrder((qty) => `placed ${qty}`) } });

  expect(succeeded(result)).toBe("placed 5");
});

test("tagged tool input fails before the tool runs", async () => {
  const calls: unknown[] = [];
  const result = await run(toolFlow("!int ${src.output}"), {}, {
    tools: {
      place_order: placeOrder((qty) => {
        calls.push(qty);
        return "placed";
      }),
    },
  });

  expectTypeFailure(result, "place");
  expect(calls).toEqual([]);
});

const CHILD = `
defs:
  child:
    input:
      x: !int
    nodes:
      twice:
        kind: code
        input:
          x: \${input.x}
        output: int
        code: |
          return inputs["x"] * 2
    output: \${twice.output}
`;

/** Return a flow calling the strict `child` with a tagged arg over an Optional source. */
const callFlow = (returned: string) => `
id: tagged_call
name: tagged_call
${CHILD}nodes:
  maybe:
    kind: code
    output: Optional(int)
    code: |
      return ${returned}
  run:
    kind: call
    call: child
    input:
      x: !int \${maybe.output}
output: \${run.output}
`;

test("tagged call arg narrows a nullable source", async () => {
  expect(succeeded(await run(callFlow("3")))).toBe(6n);
});

test("tagged call arg rejects a null at runtime", async () => {
  expectTypeFailure(await run(callFlow("null")), "run");
});

/** Return a flow mapping an inline body over `items`, with `t: !int ${item}`. */
const mapFlow = (items: string) => `
id: tagged_map
name: tagged_map
nodes:
  src:
    kind: code
    code: |
      return ${items}
  m:
    kind: map
    over: \${src.output}
    input:
      t: !int \${item}
    nodes:
      twice:
        kind: code
        input:
          t: \${input.t}
        code: |
          return inputs["t"] * 2
    output: \${twice.output}
output: \${m.output}
`;

test("tagged map item arg runs", async () => {
  expect(succeeded(await run(mapFlow("[1, 2]")))).toEqual([2n, 4n]);
});

/** Return a flow whose inline LOOP seeds `n` as `seed`. */
const loopFlow = (seed: string) => `
id: tagged_loop
name: tagged_loop
nodes:
  src:
    kind: code
    code: |
      return "0"
  loop:
    kind: loop
    times: 2
    input:
      n: ${seed}
    nodes:
      bump:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] + 1
    output:
      n: \${bump.output}
output: \${loop.output}
`;

test("tagged literal loop seed runs", async () => {
  expect(succeeded(await run(loopFlow("!int 0")))).toEqual({ n: 2n });
});

test("tagged loop seed rejects a mistyped value", async () => {
  expectTypeFailure(await run(loopFlow("!int ${src.output}")), "loop");
});

const HUMAN_FLOW = `
id: tagged_gate
name: tagged_gate
input:
  n: !str
nodes:
  src:
    kind: code
    input:
      n: \${input.n}
    code: |
      return /^[0-9]+$/.test(inputs["n"]) ? Number(inputs["n"]) : inputs["n"]
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: "Proceed with \${src.output} items?"
      count: !int \${src.output}
output: \${ask.output}
`;

/** The run's ask handler, recording each request into `asked`; it declares the tagged `count`. */
const recordingAsk = (asked: AskRequest[]) => ({
  handler: (request: AskRequest) => {
    asked.push(request);
    return "yes";
  },
  inputs: { count: { type: "integer" } },
});

test("tagged human input context passes a matching value", async () => {
  const asked: AskRequest[] = [];
  const result = await run(HUMAN_FLOW, { n: "3" }, { tools: { ask: recordingAsk(asked) } });

  succeeded(result);
  expect(asked[0]!.question).toBe("Proceed with 3 items?");
});

test("tagged human input context fails before asking", async () => {
  const asked: AskRequest[] = [];
  const result = await run(HUMAN_FLOW, { n: "three" }, { tools: { ask: recordingAsk(asked) } });

  expectTypeFailure(result, "ask");
  expect(asked).toEqual([]);
});

/** Return a flow whose inline CALL body takes `n: !int` from an untyped producer. */
const inlineFlow = (returned: string) => `
id: tagged_inline
name: tagged_inline
nodes:
  a:
    kind: code
    code: |
      return ${returned}
  wrap:
    kind: call
    input:
      n: !int \${a.output}
    nodes:
      twice:
        kind: code
        input:
          n: \${input.n}
        output: int
        code: |
          return inputs["n"] * 2
    output: \${twice.output}
output: \${wrap.output}
`;

test("tagged inline body arg runs", async () => {
  expect(succeeded(await run(inlineFlow("4")))).toBe(8n);
});

test("tagged inline body arg rejects a mistyped value", async () => {
  expectTypeFailure(await run(inlineFlow('"4"')), "wrap");
});

/** Return a flow whose untyped `size` node feeds `output:` written as `output`. */
const outputFlow = (output: string, returned = '{"qty": 1}') => `
id: tagged_output
name: tagged_output
typedefs:
  Size:
    qty: int
  T:
    a: int
nodes:
  size:
    kind: code
    code: |
      return ${returned}
output: ${output}
`;

test("tagged output field passes a matching value", async () => {
  expect(succeeded(await run(outputFlow("\n  qty: !int ${size.output.qty}")))).toEqual({ qty: 1n });
});

test("tagged output field rejects a mistyped value", async () => {
  expectTypeFailure(await run(outputFlow("\n  qty: !int ${size.output.qty}", '{"qty": "1"}')));
});

test("tagged single output returns the bare value", async () => {
  expect(succeeded(await run(outputFlow("!Size ${size.output}")))).toEqual({ qty: 1n });
});

test("tagged single output rejects a mistyped value", async () => {
  const text = outputFlow("!Size ${size.output}", '{"qty": "x"}');
  const loaded = await loadFlow(text, { env });

  const result = await runFlow(loaded, {}, { env });

  expectTypeFailure(result);
  const frame = loaded.diagnose(result.failure!).frames.at(-1)!;
  const line = text.split("\n").indexOf("output: !Size ${size.output}") + 1;
  expect(frame.range.coveredLines(text)).toEqual([line]);
});
