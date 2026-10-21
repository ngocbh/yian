/**
 * End-to-end fresh runs through `runFlow`: a port of `tests/engine/test_run.py`.
 *
 * The flows use CODE, through `./compose_codefns.ts:<fn>` references (resolved from the
 * conformance fixtures as `baseDir`) or inline TS bodies; an AGENT's model is a `ScriptedModel`.
 * A flow failure, a false assert included, comes back as a `RunResult` whose `status` is
 * `"failed"`. Every run sees only the test's credential store in its environment.
 *
 * Ported: `test_code_flow_runs_to_declared_output`, `test_leaf_binds_from_flow_wiring`,
 * `test_coalesce_output_resolves_taken_branch_positive`, `test_case_gate_binds_from_flow_wiring`,
 * `test_coalesce_output_resolves_taken_branch_cautious`, `test_coalesce_skips_untaken_branch`,
 * `test_input_type_mismatch_fails_at_boundary`, `test_coercible_string_input_passes_boundary`,
 * `test_bool_spelling_coerces_at_boundary` (both spellings),
 * `test_unrecognized_bool_spelling_fails_at_boundary`,
 * `test_bool_text_is_not_converted_for_a_child_input` (a CALL, both answers; the `yes` run is also
 * the harvested `run-core/engine_run__bool_text_is_not_converted_for_a_child_input`; Changed: the
 * message quotes the text JSON style, `"yes"`, where Python shows `'yes'` (decision 7)),
 * `test_boundary_assert_false_fails_before_any_node_runs`,
 * `test_post_terminal_assert_false_flips_succeeded_to_failed`, `test_all_asserts_true_succeeds`,
 * `test_run_id_host_override_resolves` and `test_run_id_minted_when_omitted`.
 *
 * TS-only: an already-aborted signal rejects with its reason before the log exists, as does a bad
 * `scheduler`, `random` or `models`; an existing log path is refused (`EEXIST`) and left
 * unchanged; `result.log` is the path as given (`null` without one); `interruptedAttempts` is an empty map; host intake of inputs (`{n: 3}` for an
 * `int` is `3n`, NaN refused where it sits, an accessor or a Proxy inputs record refused,
 * `Infinity` kept for a `float`, a too-deep value refused); the run's `env` scopes the `YIAN_*`
 * values it reads and records, and a bad one rejects before the log exists.
 *
 * TS-only, one row for each run-start check's message (Python pins them in
 * `test_typed_tool_node.py`, `test_run_tools_flow.py`, `test_run_skills.py` and
 * `test_span_calls.py`): an unknown tool id with a close match, an unknown tool group, `ask_user`
 * without a handler, an input the tool doesn't take, a missing required input, a disjoint input
 * type (a literal shown by its value, a long one cut), a disjoint output, skills without a map, a
 * missing skill, a skill keyed by another name, and a bad template span named by its subject; a
 * boundary callback whose promise rejects stops the run, and one that throws rejects the run with
 * no stop record; an `llmConfig` that isn't a record is refused.
 *
 * The run-start LLM chain, where an AGENT's model is a `ScriptedModel` from `models`. TS rows
 * this file owns, not ports of their Python files: an AGENT with no provider fails before the log
 * exists (the same checks as `test_no_model_anywhere_fails_before_any_node`, so
 * `no_builtin_defaults.test.ts` points here); `llm_config.api_key` without `provider` is refused
 * without echoing the key (the check of `test_run_api_key.py`'s
 * `test_key_without_a_provider_is_rejected`, which stays with that file's mirror); of
 * `test_a_key_that_marks_no_agent_does_not_block_replay`, the fresh run (no agent gets the key and
 * the log holds no marker); an integral number in a host `llm_config` is recorded as an int.
 *
 * Ported: `test_a_non_finite_value_names_the_block_field` of `test_llm_config_values_run_start.py`
 * and of `test_llm_config_values_call_time.py`. Changed: TS keeps Infinity, so the provider block
 * refuses it ("Input should be a finite number", a `contract.input` failure when the agent runs)
 * where Python refuses `non-finite number inf`.
 *
 * TS-only, `inspectRun` (each row without the flow, and with it where the two agree): a succeeded
 * and a failed run; a crash mid-node counts the in-flight node once (a crash is a log that refuses
 * appends past a record); a `RunStopped` covers its in-flight nodes; resumed then crashed again
 * counts 2, and the count stays after the node finishes; a locked log (a `JsonlRunLog` held open)
 * is `running` with the recorded counts only; END succeeded with another node ready but unstarted
 * reads `succeeded` without the flow and `interrupted` with it; a finished pooled failure has
 * nothing in flight and adds nothing; a missing log throws `ENOENT`, and an empty log no writer
 * holds is refused.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import { flowFingerprint } from "../../src/compile/manifest.ts";
import { copyGraph, END_ID, START_ID } from "../../src/compile/model.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { current, DefaultsError, type Env } from "../../src/defaults.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import type { CaseNode } from "../../src/nodes/case/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import {
  inspectRun,
  type ResumeFlowOptions,
  RunResult,
  type RunInspection,
  resumeFlow,
  runFlow,
  type RunFlowOptions,
} from "../../src/run.ts";
import {
  JsonlRunLog,
  readRunLog,
  type RunRecord,
  type RunStartedRecord,
  RunStoppedRecord,
} from "../../src/runtime/log.ts";
import {
  commitPrepared,
  prepareNodeStarted,
  prepareNodeSucceeded,
  prepareRunStarted,
  RunState,
} from "../../src/runtime/state.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { seedSystemClock } from "../../src/typesys/seeding.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";

import { textReply } from "./run_state_fixtures.ts";

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** An environment with only `vars` set, and the credential store in the test's directory. */
function hostEnv(vars: Record<string, string> = {}): Env {
  const all = new Map(Object.entries({ YIAN_AUTH_FILE: join(dir, "auth.json"), ...vars }));
  return (name) => all.get(name);
}

/** `runFlow` in `hostEnv()` unless `options` sets `env`, so no host variable or login decides a row. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(), ...options });
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

// ---- TS-only: the boundary ----

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
  [{ models: { env: () => undefined } }, "models must provide env(name) and createModel(config, options), or be null"],
])("a bad test option is refused before the log exists (%#)", async (option, message) => {
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

test("interruptedAttempts is an empty map for a fresh run", async () => {
  const result = await run(await load(PLAN_FLOW), { topic: "ACME" });
  expect(result.interruptedAttempts).toBeInstanceOf(Map);
  expect(result.interruptedAttempts.size).toBe(0);
  expect(result.learning).toBeNull();
  expect(result.backwardOutcome).toBeNull();
});

// ---- TS-only: host intake of inputs ----

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
  const env = hostEnv({ YIAN_MAX_VALUE_NESTING: "3" });
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
  const env = hostEnv({ YIAN_MAX_VALUE_NESTING: "deep" });
  const flow = await load(PLAN_FLOW);
  const error = await run(flow, { topic: "ACME" }, { log: path, env }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(DefaultsError);
  expect((error as Error).message).toStartWith('YIAN_MAX_VALUE_NESTING="deep" is not an integer');
  expect(existsSync(path)).toBe(false);
});

// ---- TS-only: the run-start checks ----

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
    `AGENT node 'agent' grants unknown tool or group 'company'; tool groups in this run: ["toolbox","workspace"]`,
  );
});

test("a TOOL node naming ask_user without a handler fails at run start", async () => {
  const flow = await load(`
id: asking
name: asking
nodes:
  ask:
    kind: tool
    tool_id: ask_user
    input:
      question: Proceed?
      options: "yes, no"
output: \${ask.output}
`);
  const error = await runStartError(flow, { tools: {} });
  expect(error).toBeInstanceOf(RangeError);
  expect(error.message).toStartWith("TOOL node 'ask' names 'ask_user': no ask_user handler for this run");
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

  const waiting = await load(`
id: f
name: f
input: {d: !str }
nodes:
  w: {kind: wait, until: '\${input.d}'}
output: ok
`);
  rewire(waiting, "w", "until", "${upper(input.d, 1)}");
  expect((await runStartError(waiting, {}, { d: "2026-01-01" })).message).toBe(`node 'w' wait until: ${UPPER}`);
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
    using log = JsonlRunLog.resume(path);
    const stopped = log.records().flatMap(({ record }) => (record instanceof RunStoppedRecord ? [record] : []));
    expect(stopped.map((record) => [record.reason, record.message])).toEqual([["callback", "host listener failed"]]);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(unhandled).toEqual([]);
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
  using log = JsonlRunLog.resume(path);
  expect(log.records().map((envelope) => envelope.kind)).toEqual(["run_started"]);
});

// ---- The run-start LLM chain ----

const SENTINEL = "sk-sentinel-5b2e8d41";

/**
 * A model access over `env` that records each config it builds from in `built` and answers each
 * call with `ok`.
 */
function scripted(built: Record<string, unknown>[], env: Env = hostEnv()): ModelAccess {
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
  env: () => undefined,
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

// Ported from `test_llm_config_values_run_start.py` and `test_llm_config_values_call_time.py`.
// Changed: `t * 10` overflows to Infinity, which TS keeps as a float (Python refuses it as
// `non-finite number inf`), so the provider block's check is what refuses it.

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
    kind: wait
    until: "2099-01-01T00:00:00+00:00"
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

/** `resumeFlow` in `hostEnv()` unless `options` sets `env`. */
function resume(flow: ComposedFlow, options: ResumeFlowOptions) {
  return resumeFlow(flow, { env: hostEnv(), ...options });
}

/** `inspectRun` of `path` without and with `flow`, which must agree. */
function inspectBoth(path: string, flow: ComposedFlow): RunInspection {
  const without = inspectRun(path);
  expect(inspectRun(path, { flow })).toEqual(without);
  return without;
}

/** Run `fn` with every durable log refusing appends once it holds `k` records; it must reject with that. */
async function crashAfter(k: number, fn: () => Promise<unknown>): Promise<void> {
  const append = JsonlRunLog.prototype.append;
  const crash = new Error(`crash after record ${k - 1}`);
  JsonlRunLog.prototype.append = function (this: JsonlRunLog, record: RunRecord) {
    if (this.nextSequence >= k) throw crash;
    return append.call(this, record);
  };
  try {
    let caught: unknown;
    try {
      await fn();
    } catch (error) {
      caught = error;
    }
    expect(caught).toBe(crash);
  } finally {
    JsonlRunLog.prototype.append = append;
  }
}

/** A log of `WORK_FLOW` whose process ended while `work` ran: records 0-3 are kept. */
async function crashedWork(path: string): Promise<ComposedFlow> {
  const flow = await load(WORK_FLOW);
  await crashAfter(4, () => run(flow, {}, { log: path, numWorkers: 0 }));
  return flow;
}

test("inspectRun reads a succeeded and a failed run, with and without the flow", async () => {
  const succeeded = join(dir, "succeeded.jsonl");
  const plan = await load(PLAN_FLOW);
  await run(plan, { topic: "ACME" }, { log: succeeded });
  expect(inspectBoth(succeeded, plan)).toEqual({ status: "succeeded", inFlight: [], interruptedAttempts: new Map() });

  const failed = join(dir, "failed.jsonl");
  const boom = await load(
    "id: f\nname: f\nnodes:\n  fail: {kind: code, code: ./compose_codefns.ts:boom, output: str}\n" +
      "output: ${fail.output}\n",
  );
  await run(boom, {}, { log: failed });
  expect(inspectBoth(failed, boom)).toEqual({ status: "failed", inFlight: [], interruptedAttempts: new Map() });
});

test("inspectRun counts the node in flight at a crash once", async () => {
  const path = join(dir, "crash.jsonl");
  const flow = await crashedWork(path);
  const before = readFileSync(path);

  expect(inspectBoth(path, flow)).toEqual({
    status: "interrupted",
    inFlight: ["work"],
    interruptedAttempts: new Map([["work", 1]]),
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
  await expect(run(flow, {}, { log: path, numWorkers: 0, signal: host.signal, tools: { work } })).rejects.toBe(stop);

  expect(inspectBoth(path, flow)).toEqual({ status: "stopped", inFlight: ["work"], interruptedAttempts: new Map() });
  const resumed = await resume(flow, { log: path, tools: { work } });
  expect(resumed.status).toBe("succeeded");
  expect(inspectBoth(path, flow)).toEqual({ status: "succeeded", inFlight: [], interruptedAttempts: new Map() });
});

test("inspectRun: resumed then crashed again counts 2, and the count stays after the node finishes", async () => {
  const path = join(dir, "twice.jsonl");
  const flow = await crashedWork(path);
  // RunResumed and work's second NodeStarted are kept; its outcome is refused again.
  await crashAfter(6, () => resume(flow, { log: path }));

  expect(inspectBoth(path, flow)).toEqual({
    status: "interrupted",
    inFlight: ["work"],
    interruptedAttempts: new Map([["work", 2]]),
  });
  const resumed = await resume(flow, { log: path });
  expect(resumed.interruptedAttempts).toEqual(new Map([["work", 2]]));
  expect(inspectBoth(path, flow)).toEqual({
    status: "succeeded",
    inFlight: [],
    interruptedAttempts: new Map([["work", 2]]),
  });
});

test("inspectRun: a locked log is running, with the recorded counts only", async () => {
  const path = join(dir, "locked.jsonl");
  const flow = await crashedWork(path);
  const held = JsonlRunLog.resume(path);
  try {
    expect(inspectBoth(path, flow)).toEqual({ status: "running", inFlight: ["work"], interruptedAttempts: new Map() });
  } finally {
    held.close();
  }
  expect(inspectRun(path).status).toBe("interrupted");
});

test("inspectRun: END succeeded with another node ready but unstarted is succeeded without the flow only", async () => {
  const path = join(dir, "end-first.jsonl");
  const flow = await load(END_FIRST_FLOW);
  const privateFlow = copyGraph(flow.compiled);
  const pool = new VariablePool(privateFlow.startId);
  pool.set(privateFlow.startId, { x: "Ada" });
  seedSystemClock(pool);
  pool.addSystem("run_id", "end-first");
  const state = new RunState(privateFlow);
  const log = JsonlRunLog.create(path, "end-first");
  try {
    commitPrepared(
      log,
      state,
      prepareRunStarted(state, {
        runId: "end-first",
        flowFingerprint: flowFingerprint(flow.compiled),
        initialPool: pool,
        startWiringOverride: { x: "${input.x}" },
        effectiveLlmByStaticPath: {},
        numWorkers: 0,
      }),
    );
    commitPrepared(log, state, prepareNodeStarted(state, START_ID));
    commitPrepared(log, state, prepareNodeSucceeded(state, START_ID, { x: "Ada" }));
    expect(state.ready).toEqual([END_ID, "side"]);
    commitPrepared(log, state, prepareNodeStarted(state, END_ID));
    commitPrepared(log, state, prepareNodeSucceeded(state, END_ID, "Ada"));
    expect(state.result).toBeNull();
  } finally {
    log.close();
  }

  expect(inspectRun(path)).toEqual({ status: "succeeded", inFlight: [], interruptedAttempts: new Map() });
  expect(inspectRun(path, { flow })).toEqual({ status: "interrupted", inFlight: [], interruptedAttempts: new Map() });
  const resumed = await resume(flow, { log: path });
  expect(resumed.output).toBe("Ada");
  expect(resumed.events.some((event) => event instanceof NodeSucceeded && event.nodeId === "side")).toBe(true);
});

test("inspectRun: a finished pooled failure has nothing in flight and adds nothing", async () => {
  const path = join(dir, "pooled-failure.jsonl");
  const flow = await load(POOLED_FAILURE_FLOW);
  const result = await run(flow, {}, { log: path, numWorkers: 2 });
  expect(result.status).toBe("failed");
  // The abandoned sibling started and never settled.
  expect(readRunLog(path).some((envelope) => envelope.payload["node_id"] === "waiter")).toBe(true);
  expect(readRunLog(path).at(-1)!.kind).toBe("node_failed");

  expect(inspectBoth(path, flow)).toEqual({ status: "failed", inFlight: [], interruptedAttempts: new Map() });
});

test("inspectRun refuses a missing log and an empty log no writer holds", async () => {
  let caught: unknown;
  try {
    inspectRun(join(dir, "missing.jsonl"));
  } catch (error) {
    caught = error;
  }
  expect((caught as NodeJS.ErrnoException).code).toBe("ENOENT");

  const empty = join(dir, "empty.jsonl");
  writeFileSync(empty, "");
  expect(() => inspectRun(empty)).toThrow(new RangeError(`cannot inspect an empty run log: ${empty}`));
});
