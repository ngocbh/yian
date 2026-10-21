/**
 * The run boundary accepts only declared input names and a record of inputs: a port of
 * `tests/engine/test_run_inputs.py`.
 *
 * Ported: `test_undeclared_input_is_rejected_before_the_log`,
 * `test_flow_without_inputs_names_none_declared`, `test_non_mapping_inputs_raise_type_error` and
 * `test_declared_inputs_in_any_mapping_run`.
 *
 * Changed: an undeclared name is a `RangeError` (Python's `ValueError`) that quotes names as JSON;
 * "any mapping" is any record (a frozen one, one with no prototype), where Python passes a
 * `MappingProxyType`.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow } from "../../src/run.ts";

const FLOW = `
id: hello
name: hello
input:
  topic: !str "world"
nodes:
  greet:
    kind: code
    input:
      t: \${input.topic}
    output: str
    code: |
      return "hello " + inputs["t"]
output: \${greet.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-inputs-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

test("undeclared input is rejected before the log", async () => {
  const flow = await loadFlow(FLOW);
  const path = join(dir, "run.jsonl");
  await expect(runFlow(flow, { topik: "Ada" }, { log: path })).rejects.toThrow(
    new RangeError(`unknown input "topik"; declared inputs: topic`),
  );
  expect(existsSync(path)).toBe(false);
});

test("flow without inputs names none declared", async () => {
  const flow = await loadFlow(FLOW.replace('input:\n  topic: !str "world"\n', "").replace("${input.topic}", "x"));
  await expect(runFlow(flow, { b: 1, a: 2 })).rejects.toThrow(
    new RangeError(`unknown inputs "a", "b"; declared inputs: (none)`),
  );
});

for (const bad of [null, [["topic", "x"]], "topic"]) {
  test(`non mapping inputs raise type error (${JSON.stringify(bad)})`, async () => {
    const flow = await loadFlow(FLOW);
    const error = await runFlow(flow, bad as never).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toStartWith("inputs must be a mapping");
  });
}

test("declared inputs in any mapping run", async () => {
  const flow = await loadFlow(FLOW);
  expect((await runFlow(flow, Object.freeze({ topic: "Ada" }))).output).toBe("hello Ada");
  expect((await runFlow(flow, Object.assign(Object.create(null), { topic: "Ada" }))).output).toBe("hello Ada");
  expect((await runFlow(flow, {})).output).toBe("hello world");
});
