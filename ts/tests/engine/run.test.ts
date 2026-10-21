/**
 * End-to-end fresh runs through `runFlow`: a port of `tests/engine/test_run.py`.
 *
 * Every flow uses CODE, through `./compose_codefns.ts:<fn>` references (resolved from the
 * conformance fixtures as `baseDir`) or inline TS bodies, so no model is needed. A flow failure,
 * a false assert included, comes back as a `RunResult` whose `status` is `"failed"`.
 *
 * Ported: `test_code_flow_runs_to_declared_output`, `test_leaf_binds_from_flow_wiring`,
 * `test_coalesce_output_resolves_taken_branch_positive`, `test_case_gate_binds_from_flow_wiring`,
 * `test_coalesce_output_resolves_taken_branch_cautious`, `test_coalesce_skips_untaken_branch`,
 * `test_input_type_mismatch_fails_at_boundary`, `test_coercible_string_input_passes_boundary`,
 * `test_bool_spelling_coerces_at_boundary` (both spellings),
 * `test_unrecognized_bool_spelling_fails_at_boundary`,
 * `test_boundary_assert_false_fails_before_any_node_runs`,
 * `test_post_terminal_assert_false_flips_succeeded_to_failed`, `test_all_asserts_true_succeeds`,
 * `test_run_id_host_override_resolves` and `test_run_id_minted_when_omitted`.
 *
 * TS-only: an already-aborted signal rejects with its reason before the log exists; an existing
 * log path is refused (`EEXIST`) and left unchanged; `result.log` is the path as given (`null`
 * without one); `interruptedAttempts` is an empty map; host intake of inputs (`{n: 3}` for an
 * `int` is `3n`, NaN refused where it sits, an accessor or a Proxy inputs record refused,
 * `Infinity` kept for a `float`, a too-deep value refused); the `memory` and run-owned MCP server
 * stubs.
 *
 * TS-only, one row for each run-start check's message (Python pins them in
 * `test_typed_tool_node.py`, `test_run_tools_flow.py`, `test_run_skills.py` and
 * `test_span_calls.py`): an unknown tool id with a close match, an unknown tool group, `ask_user`
 * without a handler, an input the tool doesn't take, a missing required input, a disjoint input
 * type (a literal shown by its value, a long one cut), a disjoint output, skills without a map, a
 * missing skill, a skill keyed by another name, and a bad template span named by its subject; a
 * boundary callback whose promise rejects stops the run.
 *
 * Later (T4-10): `test_bool_text_is_not_converted_for_a_child_input` (a CALL).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { current } from "../../src/defaults.ts";
import { NodeSucceeded, type RunEvent, RunStarted } from "../../src/events.ts";
import { ExpressionError } from "../../src/expr/index.ts";
import type { CaseNode } from "../../src/nodes/case/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { RunResult, runFlow } from "../../src/run.ts";
import { JsonlRunLog, RunStoppedRecord } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { mcpServer } from "../../src/tools/mcp.ts";
import { TypeCheckError } from "../../src/typesys/values.ts";

const FIXTURES = fileURLToPath(new URL("../../../conformance/fixtures", import.meta.url));

/** Load inline Compose YAML whose CODE references resolve from the conformance fixtures. */
function load(text: string): Promise<ComposedFlow> {
  return loadFlow(text, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
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
  const result = await runFlow(await load(PLAN_FLOW), { topic: "ACME" });
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
  const result = await runFlow(loaded, { topic: "ACME" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rating: "plan for ACME", score: 0.9 });
});

// ---- a coalesce output resolves the taken branch ----

test("coalesce output resolves taken branch positive", async () => {
  const result = await runFlow(await load(CASE_FLOW), { topic: "BETA", seed: 0.7 });
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
  const result = await runFlow(loaded, { topic: "BETA", seed: 0.7 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("pro case for BETA");
});

test("coalesce output resolves taken branch cautious", async () => {
  const result = await runFlow(await load(CASE_FLOW), { topic: "BETA", seed: 0.2 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("cautious note for BETA");
});

test("coalesce skips untaken branch", async () => {
  const result = await runFlow(await load(CASE_FLOW), { topic: "BETA", seed: 0.7 });
  const succeeded = new Set(result.events.flatMap((event) => (event instanceof NodeSucceeded ? [event.nodeId] : [])));
  expect(succeeded.has("positive")).toBe(true);
  expect(succeeded.has("cautious")).toBe(false);
});

// ---- input types are enforced at the flow boundary ----

test("input type mismatch fails at boundary", async () => {
  const result = await runFlow(await load(CASE_FLOW), { topic: "BETA", seed: "high" });
  expect(result.status).toBe("failed");
  const message = result.failure!.message;
  expect(message).toContain("input 'seed'");
  expect(message).toContain("does not match declared type float");
  expect(message).toContain("high");
  expect(hasNodeOutput(result)).toBe(false);
});

test("coercible string input passes boundary", async () => {
  const result = await runFlow(await load(CASE_FLOW), { topic: "BETA", seed: "0.7" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("pro case for BETA");
});

for (const [raw, expected] of [
  ["true", true],
  ["no", false],
] as const) {
  test(`bool spelling ${raw} coerces at boundary`, async () => {
    const result = await runFlow(await load(BOOL_FLOW), { flag: raw });
    expect(result.status).toBe("succeeded");
    expect(result.output).toBe(expected);
  });
}

test("unrecognized bool spelling fails at boundary", async () => {
  // A typo such as "ture" must not silently become false.
  const result = await runFlow(await load(BOOL_FLOW), { flag: "ture" });
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("contract.input");
  expect(result.failure!.message).toContain("input 'flag'");
  expect(result.failure!.message).toContain("does not match declared type bool");
  expect(hasNodeOutput(result)).toBe(false);
});

// ---- asserts ----

test("boundary assert false fails before any node runs", async () => {
  const result = await runFlow(await load(ASSERT_FLOW), { topic: "ACME", seed: -1.0 });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("assert failed");
  expect(hasNodeOutput(result)).toBe(false);
});

test("post terminal assert false flips succeeded to failed", async () => {
  const flow = await load(ASSERT_FLOW.replace("plan.output.score >= 0.5", "plan.output.score >= 2.0"));
  const result = await runFlow(flow, { topic: "ACME", seed: 1.0 });
  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("assert failed");
  // The run reached the node: the assert fired after it, not at the boundary.
  expect(hasNodeOutput(result)).toBe(true);
});

test("all asserts true succeeds", async () => {
  const result = await runFlow(await load(ASSERT_FLOW), { topic: "ACME", seed: 1.0 });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ rating: "plan for ACME", score: 0.9 });
});

// ---- system.run_id ----

test("run id host override resolves", async () => {
  const result = await runFlow(await load(RUN_ID_FLOW), { topic: "ACME" }, { runId: "rid-123" });
  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("rid-123");
  expect(result.runId).toBe("rid-123");
});

test("run id minted when omitted", async () => {
  const result = await runFlow(await load(RUN_ID_FLOW), { topic: "ACME" });
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
  await expect(runFlow(flow, { topic: "ACME" }, { log: path, signal: controller.signal })).rejects.toBe(reason);
  expect(existsSync(path)).toBe(false);
});

test("an existing log path is refused and left unchanged", async () => {
  const path = join(dir, "run.jsonl");
  writeFileSync(path, "keep\n");
  const flow = await load(PLAN_FLOW);
  await expect(runFlow(flow, { topic: "ACME" }, { log: path })).rejects.toMatchObject({ code: "EEXIST" });
  expect(readFileSync(path, "utf8")).toBe("keep\n");
});

test("result.log is the path as given", async () => {
  const given = relative(process.cwd(), join(dir, "run.jsonl"));
  const flow = await load(PLAN_FLOW);
  const result = await runFlow(flow, { topic: "ACME" }, { log: given });
  expect(result.log).toBe(given);
  expect(existsSync(join(dir, "run.jsonl"))).toBe(true);
  expect((await runFlow(flow, { topic: "ACME" })).log).toBeNull();
});

test("interruptedAttempts is an empty map for a fresh run", async () => {
  const result = await runFlow(await load(PLAN_FLOW), { topic: "ACME" });
  expect(result.interruptedAttempts).toBeInstanceOf(Map);
  expect(result.interruptedAttempts.size).toBe(0);
  expect(result.learning).toBeNull();
  expect(result.backwardOutcome).toBeNull();
});

// ---- TS-only: host intake of inputs ----

test("an integral number for an int input is the int 3n", async () => {
  const result = await runFlow(await load(echoFlow("int")), { x: 3 });
  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ x: 3n });
  expect(result.output).toBe(3n);
});

test("NaN is refused where it sits", async () => {
  const path = join(dir, "run.jsonl");
  const flow = await load(echoFlow("Any"));
  const error = await runFlow(flow, { x: { a: [NaN] } }, { log: path }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toBe("input 'x' at a[0]: NaN is not a yian value");
  expect(existsSync(path)).toBe(false);
});

test("an accessor input or a Proxy inputs record is refused", async () => {
  const flow = await load(echoFlow("str"));
  const inputs = {};
  Object.defineProperty(inputs, "x", { get: () => "hidden", enumerable: true });
  const error = await runFlow(flow, inputs).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toBe("input 'x': accessor property is not a yian value");
  await expect(runFlow(flow, new Proxy({ x: "a" }, {}))).rejects.toThrow(
    new TypeError("inputs must be a mapping of input names to values, got Proxy"),
  );
});

test("Infinity is kept for a float input", async () => {
  const result = await runFlow(await load(echoFlow("float")), { x: Infinity });
  expect(result.status).toBe("succeeded");
  expect(result.input).toEqual({ x: Infinity });
  expect(result.output).toBe(Infinity);
});

test("a too-deep input is refused", async () => {
  let value: unknown = [];
  for (let level = 1; level <= current().maxValueNesting; level += 1) value = [value];
  const error = await runFlow(await load(echoFlow("Any")), { x: value }).catch((caught: unknown) => caught);
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as Error).message).toStartWith("input 'x' at ");
  expect((error as Error).message).toContain(`value nests deeper than ${current().maxValueNesting} levels`);
});

// ---- TS-only: not ported yet ----

test("a memory option is not ported yet", async () => {
  const flow = await load(PLAN_FLOW);
  await expect(runFlow(flow, { topic: "ACME" }, { memory: new Map() })).rejects.toThrow("memory is not ported yet");
});

test("a run-owned MCP server is not ported yet", async () => {
  const flow = await load(PLAN_FLOW);
  const server = mcpServer({ command: "never-started" });
  await expect(runFlow(flow, { topic: "ACME" }, { tools: { mcp: server } })).rejects.toThrow(
    "run-owned MCP servers are not ported yet",
  );
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
async function runStartError(flow: ComposedFlow, options: Parameters<typeof runFlow>[2] = {}, inputs = {}) {
  const path = join(dir, "run.jsonl");
  const error = await runFlow(flow, inputs, { ...options, log: path }).catch((caught: unknown) => caught);
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
  await expect(runFlow(flow, {}, { skills: new Map([["alias", skill("code-review")]]) })).rejects.toThrow(
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
    await expect(runFlow(flow, { topic: "ACME" }, { log: path, onEvent })).rejects.toBe(failure);
    using log = JsonlRunLog.resume(path);
    const stopped = log.records().flatMap(({ record }) => (record instanceof RunStoppedRecord ? [record] : []));
    expect(stopped.map((record) => [record.reason, record.message])).toEqual([["callback", "host listener failed"]]);
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(unhandled).toEqual([]);
  } finally {
    process.off("unhandledRejection", collect);
  }
});
