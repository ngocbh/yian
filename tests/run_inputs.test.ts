/**
 * The run boundary accepts only declared input names, refusing others with a `RangeError` that
 * quotes names as JSON, and any record of inputs (frozen, or with no prototype).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { runFlow, type RunFlowOptions } from "../../src/run.ts";

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

/** `runFlow` with only the test's credential store in its environment, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  const env = new Map([["YIAN_AUTH_FILE", join(dir, "auth.json")]]);
  return runFlow(flow, inputs, { env: (name) => env.get(name), ...options });
}

test("undeclared input is rejected before the log", async () => {
  const flow = await loadFlow(FLOW);
  const path = join(dir, "run.jsonl");
  await expect(run(flow, { topik: "Ada" }, { log: path })).rejects.toThrow(
    new RangeError(`unknown input "topik"; declared inputs: topic`),
  );
  expect(existsSync(path)).toBe(false);
});

test("flow without inputs names none declared", async () => {
  const flow = await loadFlow(FLOW.replace('input:\n  topic: !str "world"\n', "").replace("${input.topic}", "x"));
  await expect(run(flow, { b: 1, a: 2 })).rejects.toThrow(
    new RangeError(`unknown inputs "a", "b"; declared inputs: (none)`),
  );
});

for (const bad of [null, [["topic", "x"]], "topic"]) {
  test(`non mapping inputs raise type error (${JSON.stringify(bad)})`, async () => {
    const flow = await loadFlow(FLOW);
    const error = await run(flow, bad as never).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toStartWith("inputs must be a mapping");
  });
}

test("declared inputs in any mapping run", async () => {
  const flow = await loadFlow(FLOW);
  expect((await run(flow, Object.freeze({ topic: "Ada" }))).output).toBe("hello Ada");
  expect((await run(flow, Object.assign(Object.create(null), { topic: "Ada" }))).output).toBe("hello Ada");
  expect((await run(flow, {})).output).toBe("hello world");
});
