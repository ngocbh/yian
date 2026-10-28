/**
 * The top-level START runs as an ordinary scheduled node: it runs through `evalNode`, emits
 * `NodeStarted` and `NodeSucceeded`, and carries the flow's boundary asserts in its ordinary
 * `postAsserts`, each `input.X` lowered to `output.X` so it reads START's bound output record.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { type CompiledFlow, END_ID, FlowOutput, START_ID } from "../../src/compile/model.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { NodeStarted, NodeSucceeded, RunFailed, RunSucceeded } from "../../src/events.ts";
import { Assertion, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { runFlow } from "../../src/run.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { engine, FuncNode, graph } from "../helpers/run_state_fixtures.ts";

const INT = new ValueContract(Type.scalar(ValueKind.INTEGER));

/**
 * Return `START -> <id> -> END` for an integer input `name`, where node `id` echoes it and its
 * value is the output; START asserts `assert` (over `input`) when given.
 */
function echoFlow(
  id: string,
  name: string,
  options: { default?: bigint; assert?: [expression: string, display: string] } = {},
): CompiledFlow {
  const parameter =
    options.default === undefined
      ? new Parameter(name, { contract: INT, required: true })
      : new Parameter(name, { contract: INT, hasDefault: true, default: options.default });
  const start = new StartNode(START_ID, { params: [parameter] });
  if (options.assert !== undefined) start.postAsserts = [new Assertion(...options.assert)];
  const echo = new FuncNode(id, (inputs) => inputs["x"], { reads: { x: `\${input.${name}}` }, outputContract: INT });
  return graph([start, echo], [
    [START_ID, id],
    [id, END_ID],
  ], { outputs: [new FlowOutput("output", `\${${id}.output}`)] });
}

/** The age flow: START requires `age` and asserts `input.age >= 18`. */
function ageFlow(): CompiledFlow {
  return echoFlow("n", "age", { assert: ["output.age >= 18", "input.age >= 18"] });
}

test("boundary assert violating value fails", async () => {
  const last = (await engine(ageFlow(), { inputs: { age: 10n } }).run()).at(-1);
  expect(last).toBeInstanceOf(RunFailed);
  expect((last as RunFailed).failure.message).toContain("assert failed");
});

test("boundary assert satisfying value succeeds", async () => {
  expect((await engine(ageFlow(), { inputs: { age: 20n } }).run()).at(-1)).toEqual(new RunSucceeded({ output: 20n }));
});

test("start emits started and succeeded", async () => {
  // START is an ordinary scheduled node, so it emits both lifecycle events.
  const events = await engine(ageFlow(), { inputs: { age: 20n } }).run();
  expect(events.some((event) => event instanceof NodeStarted && event.nodeId === START_ID)).toBe(true);
  expect(events.some((event) => event instanceof NodeSucceeded && event.nodeId === START_ID)).toBe(true);
});

test("typed int default is used when omitted", async () => {
  // The caller omits `n`; parameter binding supplies the integer default before START runs.
  const events = await engine(echoFlow("echo", "n", { default: 5n })).run();
  const started = events.find((event) => event instanceof NodeSucceeded && event.nodeId === START_ID);
  expect((started as NodeSucceeded).output).toEqual({ n: 5n });
  expect(events.at(-1)).toEqual(new RunSucceeded({ output: 5n }));
});

test("boundary assert reads typed default", async () => {
  // `n` is omitted, so the boundary assert reads START's bound integer output.
  const flow = echoFlow("echo", "n", { default: 5n, assert: ["output.n >= 5", "input.n >= 5"] });
  expect((await engine(flow).run()).at(-1)).toEqual(new RunSucceeded({ output: 5n }));
});

// --- a def's boundary assert fires from the cloned child START ------------------------------ //

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));

const CHILD_ASSERT_FLOW = `
id: d
name: d
input:
  v: !int
defs:
  checked:
    input:
      n: !int
    nodes:
      x:
        kind: code
        code: ./compose_codefns.ts:double
        input:
          n: \${input.n}
        output: int
    output: \${x.output}
    asserts:
      - "input.n >= 0"
nodes:
  call_it:
    kind: call
    call: checked
    input:
      n: \${input.v}
output: \${call_it.output}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-start-engine-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

/** Run the child-assert flow on `v` with `numWorkers`, with only the test's credential store in its environment. */
async function runChildAssert(v: number, numWorkers: number) {
  const flow = await loadFlow(CHILD_ASSERT_FLOW, { baseDir: FIXTURES });
  return runFlow(flow, { v }, { numWorkers, env: hostEnv(dir) });
}

for (const numWorkers of [0, 4]) {
  test(`child boundary assert violation fails run (numWorkers ${numWorkers})`, async () => {
    const result = await runChildAssert(-1, numWorkers);
    expect(result.status).toBe("failed");
    expect(result.failure!.message).toContain("assert failed");
  });

  test(`child boundary assert satisfied succeeds (numWorkers ${numWorkers})`, async () => {
    const result = await runChildAssert(5, numWorkers);
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(10n);
  });
}

// ---------- through runFlow ----------

const AGE_FLOW = `
id: age
name: age
input:
  age: !int
nodes:
  n:
    kind: code
    input:
      x: \${input.age}
    output: int
    code: ./compose_codefns.ts:echo_x
output: \${n.output}
asserts:
  - "input.age >= 18"
`;

/** Load `source` with the conformance CODE fixtures and run it on `inputs`. */
async function run(source: string, inputs: Record<string, unknown>) {
  return runFlow(await loadFlow(source, { baseDir: FIXTURES }), inputs, { env: hostEnv(dir) });
}

test("boundary assert violating value fails", async () => {
  const result = await run(AGE_FLOW, { age: 10 });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("assert failed");
});

test("boundary assert satisfying value succeeds", async () => {
  const result = await run(AGE_FLOW, { age: 20 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe(20n);
});

test("typed int default is used when omitted, through runFlow", async () => {
  const source = `
id: dflt
name: dflt
input:
  n: !int 5
nodes:
  echo:
    kind: code
    input:
      x: \${input.n}
    output: int
    code: ./compose_codefns.ts:echo_x
output: \${echo.output}
`;
  const result = await run(source, {});
  expect([result.status, result.output]).toEqual(["succeeded", 5n]);
});

test("loaded flow reusable across runs, shared IR untouched", async () => {
  const loaded = await loadFlow(
    `
id: reuse
name: reuse
input:
  topic: !str
  window: !int 30
nodes:
  emit:
    kind: code
    input:
      topic: \${input.topic}
      window: \${input.window}
    output:
      report: str
      n: int
    code: ./compose_codefns.ts:make_report
output:
  report: \${emit.output.report}
  n: \${emit.output.n}
`,
    { baseDir: FIXTURES },
  );
  const fullWiring = new Map(loaded.compiled.wiring.get(START_ID)!);
  expect([...fullWiring.keys()].sort()).toEqual(["topic", "window"]);

  // Both keys present: the run must not narrow the shared START wiring.
  const first = await runFlow(loaded, { topic: "A", window: 10 }, { env: hostEnv(dir) });
  expect(first.status).toBe("succeeded");
  expect(first.input).toEqual({ topic: "A", window: 10n });
  expect(loaded.compiled.wiring.get(START_ID)).toEqual(fullWiring);

  // `window` omitted: START fills the default, and the input keeps only the caller's key.
  const second = await runFlow(loaded, { topic: "B" }, { env: hostEnv(dir) });
  expect(second.status).toBe("succeeded");
  expect(second.input).toEqual({ topic: "B" });
  expect(second.output).toEqual({ report: "report for B", n: 1n });
  expect(loaded.compiled.wiring.get(START_ID)).toEqual(fullWiring);
});
