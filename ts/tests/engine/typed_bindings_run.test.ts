/**
 * Typed node input bindings checked when a run binds a node's inputs: a port of
 * `tests/engine/test_typed_bindings_run.py`.
 *
 * A binding written `name: !Type value` declares the value's type. The binder checks the resolved
 * value against it without converting, and a mismatch fails the node with code `binding.type`,
 * located on the typed binding.
 *
 * Ported: every row. The CODE bodies are Python's, which read the same in JS (decision 11), but for
 * the gate's producer, which parses digits with a regular expression and `Number`. The `ask_user`
 * request is camelCase.
 *
 * Changed: `test_matching_typed_input_passes_the_value_through` reads the int as the `bigint` `5n`
 * (decision 4a). `test_typed_input_does_not_convert_an_int_declared_float`: the int stays an int
 * (decision 38, as in Python), but a CODE body receives an int as a JS `number` (decision 45), so
 * the body's `typeof` reads `number` where Python's type name reads `int`.
 * The row can't tell an int from a whole float, since a CODE body sees both as a JS `number`
 * (decision 45) and a whole number it returns reads back as an int (decision 9A); decision 38 is
 * pinned by `values/int_fills_float` and the flow-output rows of `typed_flow_outputs_run.test.ts`.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type RunFlowOptions, type RunResult, runFlow } from "../../src/run.ts";
import type { AskUserRequest } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-typed-bindings-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

/** Load `text` and run it in the test's environment. */
async function run(text: string, inputs: Record<string, unknown> = {}, options: RunFlowOptions = {}): Promise<RunResult> {
  return runFlow(await loadFlow(text, { env }), inputs, { env, ...options });
}

/** Return a flow whose untyped producer returns `{"qty": qty}` to an `int` input. */
const sizeFlow = (qty: string) => `
id: typed_size
name: typed_size
nodes:
  size:
    kind: code
    code: |
      return {"qty": ${qty}}
  use:
    kind: code
    input:
      qty: !int \${size.output.qty}
    code: |
      return inputs["qty"]
output: \${use.output}
`;

test("mismatched typed input fails the consumer on its binding", async () => {
  const text = sizeFlow("'5'");
  const loaded = await loadFlow(text, { env });

  const result = await runFlow(loaded, {}, { env });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("use");
  expect(result.failure!.message).toContain("binding 'qty'");
  const diagnostic = loaded.diagnose(result.failure!);
  expect(diagnostic.frames.length).toBeGreaterThan(0);
  const frame = diagnostic.frames.at(-1)!;
  const bindingLine = text.split("\n").findIndex((line) => line.includes("qty: !int ")) + 1;
  expect(frame.text).toBe(text);
  expect(frame.range.coveredLines(text)).toEqual([bindingLine]);
});

test("matching typed input passes the value through", async () => {
  const result = await run(sizeFlow("5"));

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(5n);
});

/** Return a flow whose producer returns `value` into an input declared `declared`. */
const typedConsumerFlow = (value: string, declared: string) => `
id: typed_consumer
name: typed_consumer
nodes:
  p:
    kind: code
    code: |
      return ${value}
  c:
    kind: code
    input:
      x: !${declared} \${p.output}
    code: |
      return typeof inputs["x"]
output: \${c.output}
`;

test("typed input does not convert an int declared float", async () => {
  const result = await run(typedConsumerFlow("30", "float"));

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("number");
});

test("typed input does not coerce a numeric string", async () => {
  const result = await run(typedConsumerFlow("'30'", "int"));

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("c");
});

const HUMAN_FLOW = `
id: typed_gate
name: typed_gate
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
    kind: human_input
    input:
      count: !int \${src.output}
    prompt: "Proceed with \${count} items?"
output: \${ask.output}
`;

/** An `ask_user` handler that records each request in `asked` and answers `yes`. */
const recording = (asked: AskUserRequest[]) => (request: AskUserRequest) => {
  asked.push(request);
  return "yes";
};

test("typed human input context binding fails before asking", async () => {
  const asked: AskUserRequest[] = [];
  const result = await run(HUMAN_FLOW, { n: "three" }, { tools: { ask_user: recording(asked) } });

  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("binding.type");
  expect(result.failure!.nodeId).toBe("ask");
  expect(asked).toEqual([]);
});

test("typed human input context binding passes a matching value", async () => {
  const asked: AskUserRequest[] = [];
  const result = await run(HUMAN_FLOW, { n: "3" }, { tools: { ask_user: recording(asked) } });

  expect(result.failure).toBeNull();
  expect(result.status).toBe("succeeded");
  expect(asked.length).toBe(1);
  expect(asked[0]!.question).toBe("Proceed with 3 items?");
  expect(result.output).toBe("yes");
});

const ADAPTIVE_FLOW = `
id: typed_adaptive
name: typed_adaptive
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
    kind: human_input
    input:
      count: !int \${src.output}
    adaptive_questions:
      prompt: "Design one question about \${count} items."
output: \${ask.output}
`;

test("adaptive questions composer keeps the gate input types", async () => {
  const wiring = (await loadFlow(ADAPTIVE_FLOW, { env })).compiled.wiring;

  const gateContract = wiring.get("ask")!.get("count")!.contract;
  const composerContract = wiring.get("ask__compose")!.get("count")!.contract;

  expect(gateContract).not.toBeNull();
  expect(composerContract).toEqual(gateContract);
});
