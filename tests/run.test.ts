/**
 * End-to-end fresh runs through `runFlow`, and `inspectRun`, `staticNodes` and `runLogPath`.
 *
 * The flows use CODE, through `./compose_codefns.ts:<fn>` references (resolved from the
 * conformance fixtures as `baseDir`) or inline TS bodies; an AGENT's model is a `ScriptedModel`.
 * A flow failure, a false assert included, comes back as a `RunResult` whose `status` is
 * `"failed"`. Every run sees only the test's credential store in its environment.
 */

import { afterEach, beforeEach, expect, spyOn, test } from "bun:test";
import fs from "node:fs";
import {
  chmodSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  statSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { flowFingerprint } from "../src/compile/manifest.ts";
import { END_ID, START_ID } from "../src/compile/model.ts";
import type { ComposedFlow } from "../src/compose/build/pipeline.ts";
import { loadFlow } from "../src/compose/loader.ts";
import { current, DefaultsError, DefaultsMismatchError, type Env } from "../src/defaults.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "../src/events.ts";
import { ExpressionError } from "../src/expr/index.ts";
import { ScriptedModel } from "../src/llm/testing.ts";
import type { ModelAccess } from "../src/llm_clients/factory.ts";
import type { CaseNode } from "../src/nodes/case/node.ts";
import { Binding } from "../src/nodes/contracts.ts";
import {
  agentCall,
  agentCalls,
  foldRunLog,
  inspectRun,
  iterRunRecords,
  loadRecordedFlow,
  pendingCalls,
  type AdvanceOptions,
  RunResult,
  type RunInspection,
  advance,
  runFlow,
  type RunFlowOptions,
  RunLogLineTooLongError,
  RunLogSchemaError,
  runLogPath,
  staticNodes,
  staticPathOf,
} from "../src/run.ts";
import { virtualScheduler } from "../src/platform.ts";
import {
  JsonlRunLog,
  NodeStartedRecord,
  NodeSucceededRecord,
  readRunLog,
  type RunStartedRecord,
  RunStoppedRecord,
  SCHEMA_VERSION,
} from "../src/runtime/log.ts";
import { commitPrepared, prepareNodeStarted } from "../src/runtime/state.ts";
import { Skill } from "../src/skills/catalog.ts";
import { tool } from "../src/tools/function_tools.ts";
import { TypeCheckError } from "../src/typesys/values.ts";

import { folderSyncs } from "./helpers/folder_syncs.ts";
import { hostEnv } from "./helpers/mcp_helpers.ts";
import {
  crashAfter,
  NO_ENV,
  prepareOutput,
  progressOf,
  rejected,
  requiredEndFlow,
  serving,
  startedRun,
  textReply,
  thrownBy,
} from "./helpers/run_state_fixtures.ts";

const FIXTURES = fileURLToPath(new URL("./conformance/fixtures", import.meta.url));

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** `runFlow` in `hostEnv(dir)` unless `options` sets `env`, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(dir), ...options });
}

const PLAN_FLOW = `
id: run-plan
name: plan_flow
input:
  topic: !str
nodes:
  plan:
    kind: code
    code: ./compose_codefns.ts:make_plan
    input:
      topic: \${input.topic}
    output:
      rating: str
      score: float
output:
  rating: \${plan.output.rating}
  score: \${plan.output.score}
`;

// A seed-02-shape coalesce flow: `score` produces a number from `seed`, the `gate` case routes
// to one branch, and the flow returns whichever ran.
const CASE_FLOW = `
id: run-case
name: case_flow
input:
  topic: !str
  seed: !float
nodes:
  score:
    kind: code
    code: ./compose_codefns.ts:score
    input:
      seed: \${input.seed}
    output: float
  gate:
    kind: case
    cases:
      - when: "score.output >= 0.5"
        then: positive
    else: cautious
  positive:
    kind: code
    code: ./compose_codefns.ts:positive
    input:
      topic: \${input.topic}
    output: str
  cautious:
    kind: code
    code: ./compose_codefns.ts:cautious
    input:
      topic: \${input.topic}
    output: str
output: \${positive.output | cautious.output}
`;

// A boundary assert (over `input`) and a post-terminal assert (over a node output), driven by the
// inputs so one fixture covers all-true, boundary-false and post-false.
const ASSERT_FLOW = `
id: run-asserts
name: assert_flow
input:
  topic: !str
  seed: !float
nodes:
  plan:
    kind: code
    code: ./compose_codefns.ts:make_plan
    input:
      topic: \${input.topic}
    output:
      rating: str
      score: float
output:
  rating: \${plan.output.rating}
  score: \${plan.output.score}
asserts:
  - "input.seed >= 0.0"
  - "plan.output.score >= 0.5"
`;

const BOOL_FLOW = `
id: bool_flag
name: bool_flag
input:
  flag: !bool
nodes:
  echo:
    kind: code
    input:
      flag: \${input.flag}
    code: |
      return inputs["flag"]
output: \${echo.output}
`;

const RUN_ID_FLOW = `
id: run-runid
name: runid_flow
input:
  topic: !str
nodes:
  echo:
    kind: code
    code: ./compose_codefns.ts:echo_rid
    input:
      rid: \${system.run_id}
    output: str
output: \${echo.output}
`;

/** A flow echoing its one input `x` of type `type`. */
function echoFlow(type: string): string {
  return `
id: echo
name: echo
input:
  x: !${type}
nodes:
  echo:
    kind: code
    input:
      x: \${input.x}
    code: |
      return inputs.x
output: \${echo.output}
`;
}

function hasNodeOutput(result: RunResult): boolean {
  return result.events.some((event) => event instanceof NodeSucceeded);
}

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-run-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

// ---- a CODE-only flow runs end-to-end ----

test("the readers built on iterRunRecords pass maxLineBytes on, and a longer line throws", async () => {
  const flow = await load(PLAN_FLOW);
  const log = join(dir, "run.jsonl");
  await run(flow, { topic: "ACME" }, { log });
  const limit = 100;
  const message = `run log ${log} line 1 (seq 0): the line is ${readFileSync(log).indexOf(0x0a)} bytes, over maxLineBytes ${limit}`;
  const env = hostEnv(dir);
  const readers: [string, () => unknown][] = [
    ["inspectRun", () => inspectRun(log, { maxLineBytes: limit })],
    ["inspectRun with flow", () => inspectRun(log, { flow, env, maxLineBytes: limit })],
    ["pendingCalls", () => pendingCalls(log, { maxLineBytes: limit })],
    ["agentCalls", () => agentCalls(log, { maxLineBytes: limit })],
    ["loadRecordedFlow", () => loadRecordedFlow(log, { env, maxLineBytes: limit })],
    ["foldRunLog", () => foldRunLog(log, { env, maxLineBytes: limit })],
    ["agentCall", () => agentCall(log, { nodeId: "plan", n: 1 }, { env, maxLineBytes: limit })],
  ];
  for (const [name, read] of readers) {
    let caught: unknown;
    try {
      await read();
    } catch (error) {
      caught = error;
    }
    expect([name, caught]).toEqual([name, expect.any(RunLogLineTooLongError)]);
    expect((caught as Error).message).toBe(message);
  }
  expect(inspectRun(log, { maxLineBytes: 1 << 20 }).status).toBe("succeeded");
});

test("code flow runs to declared output", async () => {
  const result = await run(await load(PLAN_FLOW), { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rating: "plan for ACME", score: 0.9 });
});

test("leaf binds from flow wiring", async () => {
  // The flow owns each source; the node carries only the param name.
  const loaded = await load(PLAN_FLOW);
  const wiring = loaded.compiled.wiring.get("plan")!;
  const parameters = loaded.compiled.nodes.get("plan")!.params!;
  expect(Object.fromEntries([...wiring].map(([name, binding]) => [name, binding.source]))).toEqual({
    topic: "${input.topic}",
  });
  expect(wiring.get("topic")!.origin).not.toBeNull();
  expect(parameters.map((parameter) => parameter.name)).toEqual(["topic"]);
  expect(parameters[0]!.origin).not.toBeNull();
  const result = await run(loaded, { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rating: "plan for ACME", score: 0.9 });
});

// ---- a coalesce output resolves the taken branch ----

test("coalesce output resolves taken branch positive", async () => {
  const result = await run(await load(CASE_FLOW), { topic: "BETA", seed: 0.7 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("pro case for BETA");
});

test("case gate binds from flow wiring", async () => {
  // The desugared CaseNode binds its __rN inputs from the flow wiring.
  const loaded = await load(CASE_FLOW);
  const gate = loaded.compiled.nodes.get("gate") as CaseNode;
  const binding = loaded.compiled.wiring.get("gate")!.get("__r0")!;
  expect(binding).toBeInstanceOf(Binding);
  expect(binding.source).toBe("${score.output}");
  expect(binding.origin).toBe(gate.cases[0]!.origin);
  expect(gate.params!.map((parameter) => parameter.name)).toEqual(["__r0"]);
  const result = await run(loaded, { topic: "BETA", seed: 0.7 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("pro case for BETA");
});

test("coalesce output resolves taken branch cautious", async () => {
  const result = await run(await load(CASE_FLOW), { topic: "BETA", seed: 0.2 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("cautious note for BETA");
});

test("coalesce skips untaken branch", async () => {
  const result = await run(await load(CASE_FLOW), { topic: "BETA", seed: 0.7 });
  const succeeded = new Set(result.events.flatMap((event) => (event instanceof NodeSucceeded ? [event.nodeId] : [])));
  expect(succeeded.has("positive")).toBe(true);
  expect(succeeded.has("cautious")).toBe(false);
});

// ---- input types are enforced at the flow boundary ----

test("input type mismatch fails at boundary", async () => {
  const result = await run(await load(CASE_FLOW), { topic: "BETA", seed: "high" });
  expect(result.status).toBe("failed");
  const message = result.failure!.message;
  expect(message).toContain("input 'seed'");
  expect(message).toContain("does not match declared type float");
  expect(message).toContain("high");
  expect(hasNodeOutput(result)).toBe(false);
});

test("coercible string input passes boundary", async () => {
  const result = await run(await load(CASE_FLOW), { topic: "BETA", seed: "0.7" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("pro case for BETA");
});

for (const [raw, expected] of [
  ["true", true],
  ["no", false],
] as const) {
  test(`bool spelling ${raw} coerces at boundary`, async () => {
    const result = await run(await load(BOOL_FLOW), { flag: raw });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(expected);
  });
}

test("unrecognized bool spelling fails at boundary", async () => {
  // A typo such as "ture" must not silently become false.
  const result = await run(await load(BOOL_FLOW), { flag: "ture" });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toContain("input 'flag'");
  expect(result.failure!.message).toContain("does not match declared type bool");
  expect(hasNodeOutput(result)).toBe(false);
});

test("bool text is not converted for a child input", async () => {
  // A CALL edge is not a flow boundary: untyped upstream text reaching a called def's bool input
  // is validated as is, so even a recognized spelling fails.
  const flow = await load(`
id: gatekeeper
name: gatekeeper
defs:
  gate:
    input:
      ok: !bool
    nodes:
      e:
        kind: code
        input:
          ok: \${input.ok}
        code: |
          return inputs["ok"]
    output: \${e.output}
input:
  answer: !str
nodes:
  src:
    kind: code
    input:
      a: \${input.answer}
    code: |
      return inputs["a"]
  g:
    kind: call
    call: gate
    input:
      ok: \${src.output}
output: \${g.output}
`);
  for (const answer of ["yes", "approve"]) {
    const result = await run(flow, { answer });
    expect(result.status).toBe("failed");
    expect(result.failure).not.toBeNull();
    expect(result.failure!.code).toBe("contract.input");
    expect(result.failure!.message).toContain(`input 'ok': ${JSON.stringify(answer)} does not match declared type bool`);
  }
});

// ---- asserts ----

test("boundary assert false fails before any node runs", async () => {
  const result = await run(await load(ASSERT_FLOW), { topic: "ACME", seed: -1.0 });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("assert failed");
  expect(hasNodeOutput(result)).toBe(false);
});

test("post terminal assert false flips succeeded to failed", async () => {
  const flow = await load(ASSERT_FLOW.replace("plan.output.score >= 0.5", "plan.output.score >= 2.0"));
  const result = await run(flow, { topic: "ACME", seed: 1.0 });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("assert failed");
  // The run reached the node: the assert fired after it, not at the boundary.
  expect(hasNodeOutput(result)).toBe(true);
});

test("all asserts true succeeds", async () => {
  const result = await run(await load(ASSERT_FLOW), { topic: "ACME", seed: 1.0 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rating: "plan for ACME", score: 0.9 });
});

// ---- system.run_id ----

test("run id host override resolves", async () => {
  const result = await run(await load(RUN_ID_FLOW), { topic: "ACME" }, { runId: "rid-123" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("rid-123");
  expect(result.runId).toBe("rid-123");
});

test("run id minted when omitted", async () => {
  const result = await run(await load(RUN_ID_FLOW), { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(typeof result.output).toBe("string");
  expect(result.output).not.toBe("");
  expect(result.runId).toBe(result.output as string);
});

// ---- the boundary ----

test("an aborted signal rejects with its reason before the log exists", async () => {
  const controller = new AbortController();
  const reason = new Error("host stop");
  controller.abort(reason);
  const path = join(dir, "run.jsonl");
  const flow = await load(PLAN_FLOW);
  await expect(run(flow, { topic: "ACME" }, { log: path, signal: controller.signal })).rejects.toBe(reason);
  expect(existsSync(path)).toBe(false);
});

test.each([
  [{ scheduler: { now: () => 0 } }, "scheduler must be a Scheduler or null"],
  [{ random: 0.5 }, "random must be a function or null"],
  [{ models: { env: NO_ENV } }, "models must provide env(name) and createModel(config, options), or be null"],
])("a bad test option is refused before the log exists (%#)", async (option, message) => {
  const path = join(dir, "run.jsonl");
  const flow = await load(PLAN_FLOW);
  const options = { log: path, ...option } as unknown as RunFlowOptions;
  await expect(run(flow, { topic: "ACME" }, options)).rejects.toThrow(new TypeError(message));
  expect(existsSync(path)).toBe(false);
});

test.each([
  [{ onEvent: 5 }, "onEvent must be a function or null"],
  [{ onStream: 5 }, "onStream must be a function or null"],
])("a callback that isn't a function is refused before the log exists (%#)", async (option, message) => {
  const path = join(dir, "run.jsonl");
  const flow = await load(PLAN_FLOW);
  const options = { log: path, ...option } as unknown as RunFlowOptions;
  await expect(run(flow, { topic: "ACME" }, options)).rejects.toThrow(new TypeError(message));
  expect(existsSync(path)).toBe(false);
});

test("an existing log path is refused and left unchanged", async () => {
  const path = join(dir, "run.jsonl");
  writeFileSync(path, "keep\n");
  const flow = await load(PLAN_FLOW);
  await expect(run(flow, { topic: "ACME" }, { log: path })).rejects.toMatchObject({ code: "EEXIST" });
  expect(readFileSync(path, "utf8")).toBe("keep\n");
});

test("result.log is the path as given", async () => {
  const given = relative(process.cwd(), join(dir, "run.jsonl"));
  const flow = await load(PLAN_FLOW);
  const result = await run(flow, { topic: "ACME" }, { log: given });
  expect(result.log).toBe(given);
  expect(existsSync(join(dir, "run.jsonl"))).toBe(true);
  expect((await run(flow, { topic: "ACME" })).log).toBeNull();
});

test("crashedAttempts is an empty map for a fresh run", async () => {
  const result = await run(await load(PLAN_FLOW), { topic: "ACME" });
  expect(result.crashedAttempts).toBeInstanceOf(Map);
  expect(result.crashedAttempts.size).toBe(0);
});

// ---- host intake of inputs ----

test("an integral number for an int input is the int 3n", async () => {
  const result = await run(await load(echoFlow("int")), { x: 3 });
  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ x: 3n });
  expect(result.output).toBe(3n);
});

test("NaN is refused where it sits", async () => {
  const path = join(dir, "run.jsonl");
  const flow = await load(echoFlow("Any"));
  const error = await run(flow, { x: { a: [NaN] } }, { log: path }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toBe("input 'x' at a[0]: NaN is not a yian value");
  expect(existsSync(path)).toBe(false);
});

test("an accessor input or a Proxy inputs record is refused", async () => {
  const flow = await load(echoFlow("str"));
  const inputs = {};
  Object.defineProperty(inputs, "x", { get: () => "hidden", enumerable: true });
  const error = await run(flow, inputs).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toBe("input 'x': accessor property is not a yian value");
  await expect(run(flow, new Proxy({ x: "a" }, {}))).rejects.toThrow(
    new TypeError("inputs must be a mapping of input names to values, got Proxy"),
  );
});

test("Infinity is kept for a float input", async () => {
  const result = await run(await load(echoFlow("float")), { x: Infinity });
  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ x: Infinity });
  expect(result.output).toBe(Infinity);
});

test("a too-deep input is refused", async () => {
  let value: unknown = [];
  for (let level = 1; level <= current().maxValueNesting; level += 1) value = [value];
  const error = await run(await load(echoFlow("Any")), { x: value }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toStartWith("input 'x' at ");
  expect((error as Error).message).toContain(`value nests deeper than ${current().maxValueNesting} levels`);
});

test("the run's env sets the YIAN_* values it reads and records", async () => {
  const path = join(dir, "run.jsonl");
  const env = hostEnv(dir, { YIAN_MAX_VALUE_NESTING: "3" });
  const flow = await load(echoFlow("Any"));
  const result = await run(flow, { x: [[1]] }, { log: path, env });
  expect(result.status).toBe("succeeded");
  expect((readRunLog(path)[0]!.record as RunStartedRecord).defaults).toEqual({ MAX_VALUE_NESTING: 3n });
  const error = await run(flow, { x: [[[[1]]]] }, { env }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toContain("value nests deeper than 3 levels");
});

test("a bad YIAN_* value in the run's env rejects before the log exists", async () => {
  const path = join(dir, "run.jsonl");
  const env = hostEnv(dir, { YIAN_MAX_VALUE_NESTING: "deep" });
  const flow = await load(PLAN_FLOW);
  const error = await run(flow, { topic: "ACME" }, { log: path, env }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(DefaultsError);
  expect((error as Error).message).toStartWith('YIAN_MAX_VALUE_NESTING="deep" is not an integer');
  expect(existsSync(path)).toBe(false);
});

// ---- the run-start checks ----

/** A flow whose one TOOL node `place` calls `toolId` with `inputBlock`'s inputs. */
function toolFlow(inputBlock: string, toolId = "place_order"): string {
  return `
id: placing
name: placing
nodes:
  place:
    kind: tool
    tool_id: ${toolId}
    input:
${inputBlock}
output: \${place.output}
`;
}

/** A `place_order` tool taking an int `qty` and a str `sym`, both required. */
const placeOrder = tool({
  name: "place_order",
  description: "Place one order.",
  parameters: {
    type: "object",
    properties: { qty: { type: "integer" }, sym: { type: "string" } },
    required: ["qty", "sym"],
  },
  execute: () => "placed",
});

/** Return the error `runFlow` rejects with, after checking that it left no log behind. */
async function runStartError(flow: ComposedFlow, options: RunFlowOptions = {}, inputs = {}) {
  const path = join(dir, "run.jsonl");
  const error = await run(flow, inputs, { ...options, log: path }).catch((caught: unknown) => caught);
  expect(existsSync(path)).toBe(false);
  return error as Error;
}

test("an unknown tool id with a close match fails at run start", async () => {
  const flow = await load(toolFlow("      qty: 5\n      sym: X", "place_ordr"));
  const error = await runStartError(flow, { tools: { place_order: placeOrder } });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(
    "TOOL node 'place' names tool 'place_ordr', which isn't in this run's tools; " +
      `known tools: ["place_order"]; did you mean 'place_order'?`,
  );
});

test("an AGENT granting an unknown tool group fails at run start", async () => {
  const flow = await load(`
id: groups
name: groups
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [toolbox, workspace, company]
output: \${agent.output}
`);
  const error = await runStartError(flow, { tools: { toolbox: [], workspace: [] } });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(
    `AGENT node 'agent' grants unknown tool or group 'company'; tool groups in this run: [toolbox, workspace]`,
  );
});

test("a TOOL node naming ask_user, ask's old name, without a handler fails at run start", async () => {
  const flow = await load(`
id: asking
name: asking
nodes:
  ask:
    kind: tool
    tool_id: ask_user
    input:
      prompt: Proceed?
output: \${ask.output}
`);
  const error = await runStartError(flow, { tools: {} });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toStartWith("TOOL node 'ask' names 'ask': no ask handler for this run");
});

test("an input the tool does not take fails at run start", async () => {
  const flow = await load(toolFlow("      qtty: 5\n      sym: X"));
  const error = await runStartError(flow, { tools: { place_order: placeOrder } });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(
    `TOOL node 'place' input 'qtty': tool 'place_order' takes no such input; it takes ["qty","sym"]`,
  );
});

test("a missing required input fails at run start", async () => {
  const one = await runStartError(await load(toolFlow("      sym: X")), { tools: { place_order: placeOrder } });
  expect(one).toBeInstanceOf(RangeError);
  expect(one.message).toBe("TOOL node 'place' is missing input 'qty' that tool 'place_order' requires");
  const both = await runStartError(await load(toolFlow("      note: x")), {
    tools: {
      place_order: tool({ ...placeOrderOptions(), parameters: placeOrderParameters(["qty", "sym", "note"]) }),
    },
  });
  expect(both.message).toBe(`TOOL node 'place' is missing inputs ["qty","sym"] that tool 'place_order' requires`);
});

/** The options `placeOrder` was built from, without `parameters`. */
function placeOrderOptions() {
  return { name: "place_order", description: "Place one order.", execute: () => "placed" };
}

/** A `place_order` schema with str `note`, int `qty` and str `sym`, requiring `required`. */
function placeOrderParameters(required: string[]) {
  return {
    type: "object",
    properties: { note: { type: "string" }, qty: { type: "integer" }, sym: { type: "string" } },
    required,
  };
}

test("a disjoint input type fails at run start", async () => {
  const typed = await load(`
id: placing
name: placing
input:
  s: !str
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: \${input.s}
      sym: X
output: \${place.output}
`);
  const error = await runStartError(typed, { tools: { place_order: placeOrder } }, { s: "10" });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is str");

  const literal = await runStartError(await load(toolFlow('      qty: "10"\n      sym: X')), {
    tools: { place_order: placeOrder },
  });
  expect(literal.message).toBe(
    `TOOL node 'place' input 'qty': tool 'place_order' takes int, but the binding is "10" (str)`,
  );
  const long = await runStartError(await load(toolFlow(`      qty: "${"x".repeat(60)}"\n      sym: X`)), {
    tools: { place_order: placeOrder },
  });
  expect(long.message).toEndWith(`but the binding is "${"x".repeat(37)}..." (str)`);
});

test("a return type disjoint from the declared output fails at run start", async () => {
  const flow = await load(`
id: count
name: count
nodes:
  place:
    kind: tool
    tool_id: place_order
    input:
      qty: 5
    output: int
output: \${place.output}
`);
  const placeText = tool({
    name: "place_order",
    description: "Place one order.",
    parameters: { type: "object", properties: { qty: { type: "integer" } }, required: ["qty"] },
    output: { type: "string" },
    execute: () => "placed",
  });
  const error = await runStartError(flow, { tools: { place_order: placeText } });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("TOOL node 'place' output: declared int, but tool 'place_order' returns str");
});

const SKILLS_FLOW = `
id: review
name: review
llm_config: {provider: openai, model: test-model}
nodes:
  review:
    kind: agent
    prompt: go
    skills: [code-review, release-notes]
output: \${review.output}
`;

/** A skill named `name`. */
function skill(name: string): Skill {
  return new Skill({ name, description: "d", instructions: "" });
}

test("a flow granting skills needs a skills map", async () => {
  const error = await runStartError(await load(SKILLS_FLOW));
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(
    "AGENT node 'review' grants skills, but no skills= mapping was passed; pass the run's skills " +
      "(or an empty Map for none)",
  );
});

test("a granted skill missing from the map fails at run start", async () => {
  const skills = new Map([["code-review", skill("code-review")]]);
  const error = await runStartError(await load(SKILLS_FLOW), { skills });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe(`AGENT node 'review' grants skills not supplied to this run: ["release-notes"]`);
});

test("the skills map must be keyed by skill name", async () => {
  const flow = await load(SKILLS_FLOW);
  await expect(run(flow, {}, { skills: new Map([["alias", skill("code-review")]]) })).rejects.toThrow(
    new TypeError("skills['alias'] must be a Skill named 'alias'"),
  );
});

const SPAN_FLOW = `
id: f
name: f
input: {v: !str }
nodes:
  x: {kind: code, code: "return inputs['n']", input: {n: '\${input.v}'}, output: str}
output: \${x.output}
`;

const UPPER = "expression function 'upper' takes (s): too many positional arguments";

/** Replace `nodeId`'s binding of `key` in `flow`'s top-level wiring with `source`. */
function rewire(flow: ComposedFlow, nodeId: string, key: string, source: string): void {
  const wiring = flow.compiled.wiring;
  wiring.set(nodeId, new Map([...wiring.get(nodeId)!, [key, new Binding(source)]]));
}

test("a bad template span set in code fails at run start, named by its subject", async () => {
  const input = await load(SPAN_FLOW);
  rewire(input, "x", "n", "${upper(input.v, 1)}");
  const error = await runStartError(input, {}, { v: "a" });
  expect(error).toBeInstanceOf(ExpressionError);
  expect(error.message).toBe(`node 'x' input 'n': ${UPPER}`);

  const output = await load(SPAN_FLOW);
  rewire(output, "__end__", "result", "${upper(x.output, 1)}");
  expect((await runStartError(output, {}, { v: "a" })).message).toBe(`flow output: ${UPPER}`);
});

test("a boundary callback whose promise rejects stops the run", async () => {
  const unhandled: unknown[] = [];
  const collect = (reason: unknown): void => {
    unhandled.push(reason);
  };
  process.on("unhandledRejection", collect);
  try {
    const failure = new Error("host listener failed");
    const path = join(dir, "run.jsonl");
    const onEvent = (event: RunEvent): Promise<void> | undefined =>
      event instanceof RunStarted ? Promise.reject(failure) : undefined;
    const flow = await load(PLAN_FLOW);
    await expect(run(flow, { topic: "ACME" }, { log: path, onEvent })).rejects.toBe(failure);
    const log = JsonlRunLog.resume(path);
    try {
      const stopped = log.records().flatMap(({ record }) => (record instanceof RunStoppedRecord ? [record] : []));
      expect(stopped.map((record) => [record.reason, record.message])).toEqual([["callback", "host listener failed"]]);
      await new Promise((resolve) => setTimeout(resolve, 0));
      expect(unhandled).toEqual([]);
    } finally {
      log[Symbol.dispose]();
    }
  } finally {
    process.off("unhandledRejection", collect);
  }
});

test("a boundary callback that throws rejects the run with no stop record", async () => {
  const failure = new Error("host listener failed");
  const path = join(dir, "run.jsonl");
  const onEvent = (event: RunEvent): void => {
    if (event instanceof RunStarted) throw failure;
  };
  await expect(run(await load(PLAN_FLOW), { topic: "ACME" }, { log: path, onEvent })).rejects.toBe(failure);
  const log = JsonlRunLog.resume(path);
  try {
    // The run start (RunStarted and its four commits) is written before its observer is published.
    expect(log.records().map((envelope) => envelope.kind)).toEqual(["run_started", ...Array(4).fill("pool_commit")]);
  } finally {
    log[Symbol.dispose]();
  }
});

// ---- The run-start LLM chain ----

const SENTINEL = "sk-sentinel-5b2e8d41";

/**
 * A model access over `env` that records each config it builds from in `built` and answers each
 * call with `ok`.
 */
function scripted(built: Record<string, unknown>[], env: Env = hostEnv(dir)): ModelAccess {
  return {
    env,
    createModel: (config) => {
      built.push({ ...config });
      const identity = { provider: config["provider"] as string, model: config["model"] as string };
      return new ScriptedModel([textReply("ok")], identity);
    },
  };
}

/** A model access whose every build fails the test. */
const forbidden: ModelAccess = {
  env: NO_ENV,
  createModel: () => {
    throw new Error("no model may be built when the run cannot start");
  },
};

test("an AGENT with no provider fails before the log exists", async () => {
  const flow = await load("id: f\nname: f\nnodes:\n  write: {kind: agent, prompt: hi}\noutput: ${write.output}\n");
  const error = await runStartError(flow, { models: forbidden });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("Agent 'write'");
  for (const fix of ["llm_config.provider/model", "--provider/--model", "yian auth <provider>"]) {
    expect(error.message).toContain(fix);
  }
});

const KEYED_FLOW = "id: keyed\nname: keyed\nnodes:\n  agent: {kind: agent, prompt: hi}\noutput: ${agent.output}\n";

test("llm_config.api_key without provider is refused without echoing the key", async () => {
  const error = await runStartError(await load(KEYED_FLOW), { models: forbidden, llmConfig: { api_key: SENTINEL } });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toContain("provider");
  expect(error.message).not.toContain(SENTINEL);
});

for (const llmConfig of ["x", 5, [1]]) {
  test(`an llmConfig that isn't a record is refused (${JSON.stringify(llmConfig)})`, async () => {
    const error = await runStartError(await load(PLAN_FLOW), { llmConfig: llmConfig as never }, { topic: "ACME" });
    expect(error).toBeInstanceOf(TypeError);
    expect(error.message).toBe("llmConfig must be a record or null");
  });
}

test("a host llm_config with a free-form field nested too deeply is refused at the field before the log exists", async () => {
  const limit = current().maxValueNesting;
  for (const levels of [limit, 20_000]) {
    let deep: unknown = 1;
    for (let level = 0; level < levels; level++) deep = { a: deep };
    const llmConfig = { provider: "vllm", model: "m1", vllm: { chat_template_kwargs: { a: deep } } };
    const error = await runStartError(await load(KEYED_FLOW), { models: forbidden, llmConfig });
    expect(error).toBeInstanceOf(RangeError);
    expect(error.message).toBe(
      "1 validation error for LLMConfig\nvllm.chat_template_kwargs\n" +
        `  Value nests deeper than ${limit} levels, the limit for a yian value (each record and list is a level)`,
    );
  }
});

test("a key that marks no agent is dropped", async () => {
  const other = await load(
    "id: other\nname: other\nnodes:\n  agent:\n    kind: agent\n    prompt: hi\n" +
      "    llm_config: {provider: anthropic, model: claude-sonnet-4-5}\noutput: ${agent.output}\n",
  );
  const plain = await load(echoFlow("str"));
  const llmConfig = { provider: "vllm", model: "served-model", api_key: SENTINEL };
  for (const [name, flow, inputs, keys] of [
    ["other", other, {}, [undefined]],
    ["plain", plain, { x: "ok" }, []],
  ] as const) {
    const built: Record<string, unknown>[] = [];
    const path = join(dir, `${name}.jsonl`);
    const result = await run(flow, inputs, { log: path, models: scripted(built), llmConfig });
    expect(result.status).toBe("succeeded");
    expect(built.map((config) => config["api_key"])).toEqual([...keys]);
    const text = readFileSync(path, "utf8");
    expect(text).not.toContain("_host_key");
    expect(text).not.toContain(SENTINEL);
  }
});

test("an integral number in a host llm_config is recorded as an int", async () => {
  const path = join(dir, "run.jsonl");
  const built: Record<string, unknown>[] = [];
  const llmConfig = { provider: "anthropic", model: "m", anthropic: { max_tokens: 1000 } };
  const result = await run(await load(KEYED_FLOW), {}, { log: path, models: scripted(built), llmConfig });
  expect(result.status).toBe("succeeded");
  const started = readRunLog(path)[0]!.record as RunStartedRecord;
  const recorded = started.effectiveLlmByStaticPath["agent"] as Record<string, unknown>;
  expect(recorded["anthropic"]).toEqual({ max_tokens: 1000n });
  expect(built[0]!["anthropic"]).toEqual({ max_tokens: 1000n });
});

// `t * 10` overflows to Infinity, which a float keeps, so the provider block's check is what
// refuses it.

const TAGGED = `
id: f
name: f
typedefs: {D: "dict(str, str)"}
input:
  d: !D
  t: !float 1.0
llm_config: {provider: openai}
nodes:
  chat:
    kind: agent
    prompt: Go.
    llm_config: {model: m, openai: {temperature: !float '\${input.t * 10}'}}
output: \${chat.output}
`;

test("a non-finite value names the block field", async () => {
  const built: Record<string, unknown>[] = [];
  const events: RunEvent[] = [];
  const error = await runStartError(
    await load(TAGGED),
    { models: scripted(built), onEvent: (event) => events.push(event) },
    { d: {}, t: 1e308 },
  );
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toBe("node 'chat' llm_config.openai.temperature: Input should be a finite number");
  expect(events).toEqual([]);
  expect(built).toEqual([]);
});

const PICKING = `
id: f
name: f
nodes:
  pick: {kind: code, output: float, code: "return 1e308"}
  chat:
    kind: agent
    prompt: Go.
    llm_config: {provider: openai, model: m, openai: {temperature: !float '\${pick.output * 10}'}}
output: \${chat.output}
`;

test("a non-finite call-time value names the block field", async () => {
  const built: Record<string, unknown>[] = [];
  const result = await run(await load(PICKING), {}, { models: scripted(built) });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toBe("node 'chat' llm_config.openai.temperature: Input should be a finite number");
  expect(built).toEqual([]);
});

// ---- inspectRun ----

const WORK_FLOW = `
id: inspect-work
name: inspect_work
nodes:
  work:
    kind: code
    code: return "done"
    output: str
output: \${work.output}
`;

const STOP_FLOW = `
id: inspect-stop
name: inspect_stop
nodes:
  work:
    kind: tool
    tool_id: work
output: \${work.output}
`;

const POOLED_FAILURE_FLOW = `
id: inspect-pooled-failure
name: inspect_pooled_failure
nodes:
  fail:
    kind: code
    code: ./compose_codefns.ts:boom
    output: str
  waiter:
    kind: tool
    tool_id: waiter
output:
  fail: \${fail.output}
  waiter: \${waiter.output}
`;

const END_FIRST_FLOW = `
id: inspect-end-first
name: inspect_end_first
input:
  x: !str
nodes:
  side:
    kind: code
    code: return "side"
    output: str
output: \${input.x}
`;

/**
 * `advance` in `hostEnv(dir)` unless `options` sets `env`; `flow` is passed only when it was
 * built in code, since resume compiles a loaded flow's recorded files.
 */
function resume(flow: ComposedFlow, options: AdvanceOptions) {
  return advance({ env: hostEnv(dir), ...(flow.source === null ? { flow } : {}), ...options });
}

/**
 * `inspectRun` of `path` without and with `flow` (read in `hostEnv(dir)`), which must agree;
 * returns the progress parts.
 */
function inspectBoth(path: string, flow: ComposedFlow) {
  return progressOf(inspectWithAndWithout(path, flow));
}

/** `inspectRun` of `path` without and with `flow` (read in `hostEnv(dir)`), which must agree. */
function inspectWithAndWithout(path: string, flow: ComposedFlow): RunInspection {
  const without = inspectRun(path);
  expect(inspectRun(path, { flow, env: hostEnv(dir) })).toEqual(without);
  return without;
}

/**
 * A log of `WORK_FLOW` whose process ended while `work` ran: records 0-8 are kept, the run start
 * (RunStarted and four commits), START's start, commit and success, and work's start.
 */
async function crashedWork(path: string): Promise<ComposedFlow> {
  const flow = await load(WORK_FLOW);
  await crashAfter(9, () => run(flow, {}, { log: path, numWorkers: 0 }));
  return flow;
}

test("inspectRun reads a succeeded and a failed run, with and without the flow", async () => {
  const succeeded = join(dir, "succeeded.jsonl");
  const plan = await load(PLAN_FLOW);
  await run(plan, { topic: "ACME" }, { log: succeeded });
  expect(inspectBoth(succeeded, plan)).toEqual({ status: "succeeded", inFlight: [], crashedAttempts: new Map() });

  const failed = join(dir, "failed.jsonl");
  const boom = await load(
    "id: f\nname: f\nnodes:\n  fail: {kind: code, code: ./compose_codefns.ts:boom, output: str}\n" +
      "output: ${fail.output}\n",
  );
  await run(boom, {}, { log: failed });
  expect(inspectBoth(failed, boom)).toEqual({ status: "failed", inFlight: [], crashedAttempts: new Map() });
});

test("inspectRun counts the node in flight at a crash once", async () => {
  const path = join(dir, "crash.jsonl");
  const flow = await crashedWork(path);
  const before = readFileSync(path);

  expect(inspectBoth(path, flow)).toEqual({
    status: "crashed",
    inFlight: ["work"],
    crashedAttempts: new Map([["work", 1]]),
  });
  expect(readFileSync(path)).toEqual(before);
});

test("inspectRun: a RunStopped covers its in-flight nodes", async () => {
  const path = join(dir, "stopped.jsonl");
  const flow = await load(STOP_FLOW);
  const host = new AbortController();
  const stop = new Error("host stop");
  let stopping = true;
  const work = tool({
    name: "work",
    description: "Stop the run the first time.",
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) => {
      if (!stopping) return "done";
      stopping = false;
      host.abort(stop);
      throw ctx.signal.reason;
    },
  });
  expect(await rejected(run(flow, {}, { log: path, numWorkers: 0, signal: host.signal, tools: { work } }))).toBe(stop);

  expect(inspectBoth(path, flow)).toEqual({ status: "stopped", inFlight: ["work"], crashedAttempts: new Map() });
  const resumed = await resume(flow, { log: path, tools: { work } });
  expect(resumed.status).toBe("succeeded");
  expect(inspectBoth(path, flow)).toEqual({ status: "succeeded", inFlight: [], crashedAttempts: new Map() });
});

test("inspectRun: resumed then crashed again counts 2, and the count stays after the node finishes", async () => {
  const path = join(dir, "twice.jsonl");
  const flow = await crashedWork(path);
  // RunResumed and work's second NodeStarted are kept; its outcome is refused again.
  await crashAfter(11, () => resume(flow, { log: path }));

  expect(inspectBoth(path, flow)).toEqual({
    status: "crashed",
    inFlight: ["work"],
    crashedAttempts: new Map([["work", 2]]),
  });
  const resumed = await resume(flow, { log: path });
  expect(resumed.crashedAttempts).toEqual(new Map([["work", 2]]));
  expect(inspectBoth(path, flow)).toEqual({
    status: "succeeded",
    inFlight: [],
    crashedAttempts: new Map([["work", 2]]),
  });
});

test("inspectRun: a locked log is running, with the recorded counts only", async () => {
  const path = join(dir, "locked.jsonl");
  const flow = await crashedWork(path);
  const held = JsonlRunLog.resume(path);
  try {
    expect(inspectBoth(path, flow)).toEqual({ status: "running", inFlight: ["work"], crashedAttempts: new Map() });
  } finally {
    held.close();
  }
  expect(inspectRun(path).status).toBe("crashed");
});

test("inspectRun: END succeeded with another node ready but unstarted is succeeded without the flow only", async () => {
  const path = join(dir, "end-first.jsonl");
  const flow = await load(END_FIRST_FLOW);
  const log = JsonlRunLog.create(path, "end-first");
  try {
    const state = startedRun(flow, log, { inputs: { x: "Ada" } });
    commitPrepared(log, state, prepareNodeStarted(state, START_ID));
    commitPrepared(log, state, prepareOutput(log, state, START_ID, { x: "Ada" }));
    expect(state.ready).toEqual([END_ID, "side"]);
    commitPrepared(log, state, prepareNodeStarted(state, END_ID));
    commitPrepared(log, state, prepareOutput(log, state, END_ID, "Ada"));
    expect(state.result).toBeNull();
  } finally {
    log.close();
  }

  expect(progressOf(inspectRun(path))).toEqual({ status: "succeeded", inFlight: [], crashedAttempts: new Map() });
  expect(progressOf(inspectRun(path, { flow, env: hostEnv(dir) }))).toEqual({
    status: "crashed",
    inFlight: [],
    crashedAttempts: new Map(),
  });
  const resumed = await resume(flow, { log: path });
  expect(resumed.output).toBe("Ada");
  expect(resumed.events.some((event) => event instanceof NodeSucceeded && event.nodeId === "side")).toBe(true);
});

test("inspectRun: a finished pooled failure has nothing in flight and adds nothing", async () => {
  const path = join(dir, "pooled-failure.jsonl");
  const flow = await load(POOLED_FAILURE_FLOW);
  // The sibling holds its worker until the failure aborts the run.
  const waiter = tool({
    name: "waiter",
    description: "Wait for the run's signal.",
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) =>
      new Promise<never>((_resolve, reject) => {
        if (ctx.signal.aborted) reject(ctx.signal.reason);
        else ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
      }),
  });
  const result = await run(flow, {}, { log: path, numWorkers: 2, tools: { waiter } });
  expect(result.status).toBe("failed");
  // The abandoned sibling started and never settled.
  expect(readRunLog(path).some((envelope) => envelope.payload["node_id"] === "waiter")).toBe(true);
  expect(readRunLog(path).at(-1)!.kind).toBe("node_failed");

  expect(inspectBoth(path, flow)).toEqual({ status: "failed", inFlight: [], crashedAttempts: new Map() });
});

test("inspectRun refuses a missing log and an empty log no writer holds", async () => {
  const missing = thrownBy(() => inspectRun(join(dir, "missing.jsonl")));
  expect((missing as NodeJS.ErrnoException).code).toBe("ENOENT");

  const empty = join(dir, "empty.jsonl");
  writeFileSync(empty, "");
  expect(() => inspectRun(empty)).toThrow(new RangeError(`cannot inspect an empty run log: ${empty}`));
});

const UNREACHABLE_FLOW = `
id: inspect-unreachable
name: inspect_unreachable
input:
  score: !float
nodes:
  gate:
    kind: case
    cases:
      - when: "input.score >= 1.0"
        then: selected
  selected:
    kind: code
    code: return "selected"
    output: str
output: \${selected.output}
`;

test("inspectRun reads a route that proved END unreachable as failed, with and without the flow", async () => {
  const path = join(dir, "unreachable.jsonl");
  const flow = requiredEndFlow(await load(UNREACHABLE_FLOW));
  const result = await run(flow, { score: 0.0 }, { log: path });
  expect(result.failure?.code).toBe("graph.terminal_unreachable");
  const last = readRunLog(path).at(-1)!;
  expect(last.kind).toBe("node_routed");
  expect(last.payload["terminal_failure"]).not.toBeNull();

  expect(inspectBoth(path, flow)).toEqual({ status: "failed", inFlight: [], crashedAttempts: new Map() });
});

test("inspectRun replays with the run's env", async () => {
  const path = join(dir, "env.jsonl");
  const flow = await load(WORK_FLOW);
  const env = hostEnv(dir, { YIAN_MAX_TOTAL_NODES: "5000" });
  await crashAfter(9, () => run(flow, {}, { log: path, numWorkers: 0, env }));

  expect(progressOf(inspectRun(path, { flow, env }))).toEqual({
    status: "crashed",
    inFlight: ["work"],
    crashedAttempts: new Map([["work", 1]]),
  });
  expect(thrownBy(() => inspectRun(path, { flow, env: hostEnv(dir) }))).toBeInstanceOf(DefaultsMismatchError);
  // Reading without the flow doesn't check the recorded values.
  expect(inspectRun(path, { env: hostEnv(dir) }).status).toBe("crashed");
});

// A MAP over a def whose TOOL node asks adaptive questions: the composer agent's model is picked
// at call time, so each item's agent records its config as a step, under its runtime id.
const MAPPED_COMPOSER_FLOW = `id: f
name: f
defs:
  item:
    input: {m: !str }
    nodes:
      ask:
        kind: tool
        tool_id: ask
        adaptive_questions:
          prompt: "Q"
          llm_config: {provider: openai, model: '\${input.m}'}
    output: \${ask.output}
nodes:
  pick: {kind: code, output: str, code: "return 'gpt-picked'"}
  each:
    kind: map
    call: item
    over: \${[1]}
    input: {m: '\${pick.output}'}
output: \${each.output}
`;

test("a resume's LLM-field error labels an in-flight mapped agent by its static path", async () => {
  const env = hostEnv(dir);
  const flow = await loadFlow(MAPPED_COMPOSER_FLOW, { env });
  const options = (log: string) => ({
    log,
    env,
    models: serving(new ScriptedModel([textReply("nope")]), env),
    tools: { ask: () => "a" },
  });
  const uncut = join(dir, "uncut.jsonl");
  await runFlow(flow, {}, options(uncut));
  const records = readFileSync(uncut, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  const at = records.findIndex(({ kind, payload }) => kind === "node_step" && payload.node_id === "each#0/ask__compose");
  // Cut right after the agent's recorded config, then make that config refuse a field.
  const log = join(dir, "run.jsonl");
  await crashAfter(at + 1, () => runFlow(flow, {}, options(log)));
  const cut = readFileSync(log, "utf8").trim().split("\n").map((line) => JSON.parse(line));
  expect(cut.at(-1).payload.step.kind).toBe("llm_config");
  cut.at(-1).payload.step.effective_llm_config.temperature = 0.5;
  writeFileSync(log, `${cut.map((record) => JSON.stringify(record)).join("\n")}\n`);

  const error = await rejected(advance(options(log)));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toStartWith("node 'ask' adaptive_questions.llm_config in 'each': ");
});

test("inspectRun reads the run id, schema, fingerprint and last seq, with and without the flow", async () => {
  const path = join(dir, "inspected.jsonl");
  const plan = await load(PLAN_FLOW);
  const result = await run(plan, { topic: "ACME" }, { log: path });
  const records = readRunLog(path);
  expect(inspectWithAndWithout(path, plan)).toEqual({
    status: "succeeded",
    inFlight: [],
    crashedAttempts: new Map(),
    runId: result.runId,
    schema: SCHEMA_VERSION,
    flowFingerprint: flowFingerprint(plan.compiled),
    lastSeq: records.length - 1,
    startedAt: records[0]!.at,
    endedAt: records.at(-1)!.at,
    failure: null,
    waits: [],
    ready: false,
    wakeAt: null,
  });

  const crashed = join(dir, "crash.jsonl");
  const work = await crashedWork(crashed);
  const inspection = inspectWithAndWithout(crashed, work);
  expect(inspection.lastSeq).toBe(8);
  expect(inspection.failure).toBeNull();
});

test("inspectRun reads the failure of a failed run, with and without the flow", async () => {
  const failed = join(dir, "failed.jsonl");
  const boom = await load(
    "id: f\nname: f\nnodes:\n  fail: {kind: code, code: ./compose_codefns.ts:boom, output: str}\n" +
      "output: ${fail.output}\n",
  );
  const result = await run(boom, {}, { log: failed });
  expect(result.failure!.nodeId).toBe("fail");
  expect(inspectWithAndWithout(failed, boom).failure).toEqual(result.failure);

  const unreachable = join(dir, "unreachable.jsonl");
  const flow = requiredEndFlow(await load(UNREACHABLE_FLOW));
  const routed = await run(flow, { score: 0.0 }, { log: unreachable });
  expect(routed.failure!.code).toBe("graph.terminal_unreachable");
  expect(inspectWithAndWithout(unreachable, flow).failure).toEqual(routed.failure);
});

test("inspectRun reads a running log with no record yet as running with nothing known", () => {
  const path = join(dir, "fresh.jsonl");
  const log = JsonlRunLog.create(path, "fresh");
  try {
    expect(inspectRun(path)).toEqual({
      status: "running",
      inFlight: [],
      crashedAttempts: new Map(),
      runId: null,
      schema: null,
      flowFingerprint: null,
      lastSeq: null,
      startedAt: null,
      endedAt: null,
      failure: null,
      waits: [],
      ready: false,
      wakeAt: null,
    });
  } finally {
    log[Symbol.dispose]();
  }
});

test("inspectRun refuses a log that doesn't start with RunStarted, and a schema it can't read", () => {
  const path = join(dir, "headless.jsonl");
  const line = (schema: number) =>
    `{"at":"2026-10-20T14:02:11.123Z","kind":"node_started","payload":{"node_id":"work"},"run_id":"run","schema":${schema},"seq":0}\n`;
  writeFileSync(path, line(SCHEMA_VERSION));
  expect(() => inspectRun(path)).toThrow(new RangeError("the first run log record must be RunStarted"));

  for (const schema of [9, 10]) {
    writeFileSync(path, line(schema));
    const error = thrownBy(() => inspectRun(path));
    expect(error).toBeInstanceOf(RunLogSchemaError);
    expect((error as RunLogSchemaError).schema).toBe(schema);
  }
});

const TIMED_FLOW = `
id: inspect-timed
name: inspect_timed
nodes:
  settle:
    kind: tool
    tool_id: sleep
    input:
      until: "2026-10-20T14:05:00+00:00"
output: \${settle.output}
`;

/** Epoch milliseconds of 2026-10-20 at `hours`:`minutes` UTC. */
function octoberTwentieth(hours: number, minutes: number): number {
  return Date.UTC(2026, 9, 20, hours, minutes);
}

test("inspectRun reads when a run started and ended, and a node's outcome less its start is its duration", async () => {
  const path = join(dir, "timed.jsonl");
  const flow = await load(TIMED_FLOW);
  const waited = await run(flow, {}, { log: path, numWorkers: 0, scheduler: virtualScheduler(octoberTwentieth(14, 0)) });
  expect(waited.status).toBe("waiting");
  // The sleep is due when the run is advanced at 14:05.
  await resume(flow, { log: path, numWorkers: 0, scheduler: virtualScheduler(octoberTwentieth(14, 5)) });

  const inspection = inspectWithAndWithout(path, flow);
  expect(inspection.status).toBe("succeeded");
  expect(inspection.startedAt).toBe("2026-10-20T14:00:00.000Z");
  expect(inspection.endedAt).toBe("2026-10-20T14:05:00.000Z");
  const records = readRunLog(path);
  const timeOf = (cls: typeof NodeStartedRecord | typeof NodeSucceededRecord) =>
    new Date(records.find((envelope) => envelope.record instanceof cls && envelope.record.nodeId === "settle")!.at!).getTime();
  expect(timeOf(NodeSucceededRecord) - timeOf(NodeStartedRecord)).toBe(5 * 60_000);
});

test("a resume keeps the recorded times and stamps its own records from its clock; an unfinished run has no end", async () => {
  const path = join(dir, "resumed.jsonl");
  const flow = await load(WORK_FLOW);
  await crashAfter(9, () => run(flow, {}, { log: path, numWorkers: 0, scheduler: virtualScheduler(octoberTwentieth(14, 0)) }));
  const crashed = inspectRun(path);
  expect([crashed.status, crashed.startedAt, crashed.endedAt]).toEqual(["crashed", "2026-10-20T14:00:00.000Z", null]);
  const before = readRunLog(path);

  const result = await resume(flow, { log: path, scheduler: virtualScheduler(octoberTwentieth(16, 30)) });
  expect(result.output).toBe("done");
  const records = readRunLog(path);
  expect(records.slice(0, before.length)).toEqual(before);
  expect(new Set(records.slice(before.length).map((envelope) => envelope.at))).toEqual(new Set(["2026-10-20T16:30:00.000Z"]));
  const resumed = inspectWithAndWithout(path, flow);
  expect([resumed.status, resumed.startedAt, resumed.endedAt]).toEqual([
    "succeeded",
    "2026-10-20T14:00:00.000Z",
    "2026-10-20T16:30:00.000Z",
  ]);
});

// ---- staticNodes ----

const NESTED_FLOW = `
id: nested
name: nested
llm_config: {provider: openai, model: test-model}
defs:
  child:
    input: {topic: !str}
    nodes:
      write:
        kind: agent
        input:
          topic: \${input.topic}
        output: str
        prompt: Write about \${topic}.
    output: \${write.output}
nodes:
  call:
    kind: call
    call: child
    input: {topic: cats}
  each:
    kind: map
    over: \${input.topics}
    call: child
    input:
      topic: \${item}
input:
  topics: !list(str)
output:
  one: \${call.output}
  many: \${each.output}
`;

test("staticNodes lists every node path of a flow and its child flows, with its node", async () => {
  const flow = await load(NESTED_FLOW);
  expect(staticNodes(flow).map(([nodePath, node]) => [nodePath, node.kind])).toEqual([
    ["call", "call"],
    ["call/write", "agent"],
    ["call/__start__", "start"],
    ["call/__end__", "end"],
    ["each", "map"],
    ["each/write", "agent"],
    ["each/__start__", "start"],
    ["each/__end__", "end"],
    ["__start__", "start"],
    ["__end__", "end"],
  ]);
  expect(staticNodes(flow.compiled)).toEqual(staticNodes(flow));
  expect(staticNodes(flow)[0]![1]).toBe(flow.compiled.nodes.get("call")!);
});

test("staticNodes lists the paths a run's effective model map uses", async () => {
  const flow = await load(NESTED_FLOW);
  const path = join(dir, "nested.jsonl");
  const result = await run(flow, { topics: ["dogs"] }, { log: path, models: scripted([]) });
  expect(result.status).toBe("succeeded");
  const started = readRunLog(path)[0]!.record as RunStartedRecord;
  const effective = Object.keys(started.effectiveLlmByStaticPath);
  expect(effective.length).toBeGreaterThan(0);
  const kinds = new Map(staticNodes(flow).map(([nodePath, node]) => [nodePath, node.kind]));
  expect(effective.map((nodePath) => kinds.get(nodePath))).toEqual(effective.map(() => "agent"));
});

const LOOP_FLOW = `
id: looping
name: looping
defs:
  step:
    input:
      n: !int
      exited: !bool
    nodes:
      bump:
        kind: code
        code: ./compose_codefns.ts:loop_bump
        input:
          n: \${input.n}
          exited: \${input.exited}
        output:
          n: int
          exited: bool
    output: \${bump.output}
nodes:
  node:
    kind: loop
    call: step
    input:
      n: 0
      exited: false
    while: n < 3
    max: 5
output: \${node.output}
`;

test("staticPathOf drops each segment's MAP element and LOOP iteration suffix", () => {
  expect(staticPathOf("each#1/echo")).toBe("each/echo");
  expect(staticPathOf("node#0/bump")).toBe("node/bump");
  expect(staticPathOf("node~2")).toBe("node");
  expect(staticPathOf("outer#3/inner#0/leaf")).toBe("outer/inner/leaf");
  expect(staticPathOf("call/write")).toBe("call/write");
});

test("staticPathOf maps every node id of a MAP run and a LOOP run onto a staticNodes path", async () => {
  const mapped = await load(NESTED_FLOW);
  const mapLog = join(dir, "map.jsonl");
  expect((await run(mapped, { topics: ["dogs", "cats"] }, { log: mapLog, models: scripted([]) })).status).toBe("succeeded");
  const looped = await load(LOOP_FLOW);
  const loopLog = join(dir, "loop.jsonl");
  expect((await run(looped, {}, { log: loopLog })).status).toBe("succeeded");

  for (const [flow, log, grown] of [
    [mapped, mapLog, "each#1/write"],
    [looped, loopLog, "node~2"],
  ] as const) {
    const ids = new Set([...iterRunRecords(log, { payloads: false })].flatMap((entry) => entry.nodeId ?? []));
    expect(ids.has(grown)).toBe(true);
    const paths = new Set(staticNodes(flow).map(([nodePath]) => nodePath));
    expect([...ids].filter((nodeId) => !paths.has(staticPathOf(nodeId)))).toEqual([]);
  }
});

test("inspectRun reads a locked log as running when a writer's rollback shrinks it under the read", async () => {
  const path = join(dir, "rollback.jsonl");
  await crashedWork(path);
  const text = readFileSync(path, "utf8");
  const kept = Buffer.byteLength(text);
  // A tenth record, a second start of `work`, which the writer is about to roll back.
  writeFileSync(path, `${text}${text.split("\n").at(-2)!.replace('"seq":8', '"seq":9')}\n`);
  const held = JsonlRunLog.resume(path);
  const realRead = fs.readSync;
  let cut = false;
  // The first read sees the appended line; the writer then rolls it back.
  const readSpy = spyOn(fs, "readSync").mockImplementation(((...args: any[]) => {
    const count = Reflect.apply(realRead, fs, args) as number;
    if (!cut && count > kept) {
      cut = true;
      fs.truncateSync(path, kept);
    }
    return count;
  }) as typeof fs.readSync);
  try {
    const inspection = inspectRun(path);
    expect(cut).toBe(true);
    expect(progressOf(inspection)).toEqual({ status: "running", inFlight: ["work"], crashedAttempts: new Map() });
    expect(inspection.lastSeq).toBe(8);
  } finally {
    readSpy.mockRestore();
    held.close();
  }
});

// ---- runLogPath ----

test("runLogPath is log.jsonl in the run's own 0700 folder, and creates no file", () => {
  const runs = join(dir, "runs");
  const path = runLogPath(runs, "r1");
  expect(path).toBe(join(runs, "r1", "log.jsonl"));
  expect(statSync(join(runs, "r1")).mode & 0o777).toBe(0o700);
  expect(existsSync(path)).toBe(false);
});

test("runLogPath syncs the runs folder once, and the folders above the ones it made", () => {
  const runs = join(dir, "state", "runs");
  const real = fs.realpathSync(dir);
  expect(folderSyncs(() => runLogPath(runs, "r1"))).toEqual([join(real, "state", "runs"), join(real, "state"), real]);
  expect(folderSyncs(() => runLogPath(runs, "r2"))).toEqual([join(real, "state", "runs")]);
});

test("runLogPath syncs the runs folder once when its path climbs out of folders it doesn't make", () => {
  mkdirSync(join(dir, "p"));
  let path: string | undefined;
  const synced = folderSyncs(() => {
    path = runLogPath(`${dir}/p/q/../..`, "r1");
  });
  expect(synced).toEqual([fs.realpathSync(dir)]);
  expect(path).toBe(join(dir, "r1", "log.jsonl"));
  expect(existsSync(join(dir, "p", "q"))).toBe(false);
});

test("runLogPath throws a folder sync's I/O error and removes the run folder", () => {
  expect(() =>
    folderSyncs(() => {
      runLogPath(dir, "r1");
    }, "EIO"),
  ).toThrow(/EIO/);
  expect(existsSync(join(dir, "r1"))).toBe(false);
});

test("runLogPath refuses an existing run folder, and a symlink to one, leaving its target as it was", () => {
  mkdirSync(join(dir, "r2"));
  expect(() => runLogPath(dir, "r2")).toThrow(/EEXIST/);
  mkdirSync(join(dir, "shared"));
  chmodSync(join(dir, "shared"), 0o755);
  symlinkSync(join(dir, "shared"), join(dir, "r4"));
  expect(() => runLogPath(dir, "r4")).toThrow(/EEXIST/);
  expect(statSync(join(dir, "shared")).mode & 0o777).toBe(0o755);
});

test("runLogPath refuses a file holding the run folder's path", () => {
  writeFileSync(join(dir, "r3"), "");
  expect(() => runLogPath(dir, "r3")).toThrow(/EEXIST/);
});

test("runLogPath refuses a run id that isn't one folder name", () => {
  for (const runId of ["", ".", "..", "a/b", "../r", "a\\b", "a\0b"]) {
    expect(() => runLogPath(dir, runId)).toThrow(RangeError);
  }
  expect(existsSync(join(dir, "a"))).toBe(false);
});

test("runLogPath takes a run id holding two dots as one folder name", () => {
  expect(runLogPath(dir, "a..b")).toBe(join(dir, "a..b", "log.jsonl"));
});
