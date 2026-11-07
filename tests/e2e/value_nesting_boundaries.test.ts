/**
 * Value nesting at the binding boundary, in AGENT structured replies and tool arguments, in runs,
 * loads and replay.
 *
 * Both structured strategies check a reply's nesting before its declared types, and a nesting error
 * has no path into the output, so an exhausted failure keeps the node origin. A growth seed's two
 * containers don't count toward the value's nesting (MAP: list, then record; LOOP: pair, then
 * record). A CODE result over the limit meets the node's post-asserts before the commit refuses it;
 * one too deep to copy back from the worker (over 1,000 levels) fails `contract.output` there,
 * before its post-asserts.
 */

import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import { BUILTIN, forget, install, processEnv } from "../../src/defaults.ts";
import { OriginRef } from "../../src/diagnostics.ts";
import { ModelCallFinished, ToolCallStarted } from "../../src/events.ts";
import { Finish, ToolCallDelta } from "../../src/llm/events.ts";
import { Message, Response, TextPart, ToolCallPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Output } from "../../src/nodes/base.ts";
import { applyDefaults, BindingError, bindParams } from "../../src/nodes/binding.ts";
import { ContractOrigins, Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { advance, type RunResult, runFlow } from "../../src/run.ts";
import { EvaluationFailure } from "../../src/runtime/eval_node.ts";
import { decodeRecord, encodeRecord } from "../../src/runtime/log.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { Type, TypeCheckError, type Value, ValueKind, VariablePool } from "../../src/typesys/index.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { drive, NO_ENV, rejected, reply, serving } from "../helpers/run_state_fixtures.ts";

// Start each test from the built-in values, whatever `YIAN_*` variables the shell sets.
beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

/** Return `"leaf"` wrapped in `levels` records under `a`. */
function nestRecord(levels: number): Value {
  let value: Value = "leaf";
  for (let i = 0; i < levels; i++) value = { a: value };
  return value;
}

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

test("applyDefaults rejects a default over the limit", () => {
  const error = bindingError(() =>
    applyDefaults([new Parameter("p", { hasDefault: true, default: nestRecord(101) })], {}),
  );
  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});

test("bindParams rejects a filled default over the limit", () => {
  const parameter = new Parameter("p", { hasDefault: true, default: nestRecord(101) });

  const error = bindingError(() => bindParams([parameter], new Map(), new VariablePool()));

  expect(error.failureCode).toBe("binding.evaluate");
  expect(error.message.startsWith("default for 'p' at a.a")).toBe(true);
  expect(error.message).toContain("deeper than 100 levels");
});

// A structured AGENT reply over the limit self-corrects under `retries:`, and fails once they are
// spent.

const NATIVE = { provider: "openai", model: "gpt-5.5" };
const PROMPTED = { provider: "vllm", model: "no-structured-sentinel" };
const FREEFORM = Type.scalar(ValueKind.OBJECT);
/** `{t: object, n: int}`. */
const T_AND_N = new Type({
  kind: ValueKind.OBJECT,
  fields: new Map([
    ["t", Type.scalar(ValueKind.OBJECT)],
    ["n", Type.scalar(ValueKind.INTEGER)],
  ]),
  required: new Set(["t", "n"]),
});

/** Return JSON text nesting `"leaf"` `levels` records deep under `a`, built without recursion. */
function deepText(levels: number): string {
  return '{"a": '.repeat(levels) + '"leaf"' + "}".repeat(levels);
}

/** A model over `config` replying each of `texts` in turn. */
function texts(config: Record<string, string>, ...replies: string[]): ScriptedModel {
  return new ScriptedModel(
    replies.map(
      (text) =>
        new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" }),
    ),
    config,
  );
}

/** Evaluate an AGENT `agent` with output `output` and `retries` over `model`; return its outcome. */
function runAgent(model: ScriptedModel, output: Type, retries: number, origins?: ContractOrigins) {
  const llmConfig = { provider: model.provider, model: model.model };
  const node = new AgentNode("agent", { prompt: "q", llmConfig, retries });
  node.origin = new OriginRef("node");
  node.outputContract = new ValueContract(output, origins);
  return drive(node, { models: { env: NO_ENV, createModel: () => model } });
}

/** The correction the model's second request ends with. */
function correction(model: ScriptedModel): string {
  return model.requests[1]!.messages.at(-1)!.text();
}

/** Expect `message` to name `where`, a path into the value, and the limit. */
function expectReplyTooDeep(message: string, where: string): void {
  expect(message).toContain(`${where} at `);
  expect(message).toContain("deeper than 100 levels");
  expect(message).not.toContain("recursion");
}

const STRATEGIES = [
  ["native", NATIVE],
  ["fallback", PROMPTED],
] as const;
const ROWS = STRATEGIES.flatMap(([name, config]) => [101, 1500].map((levels) => [name, config, levels] as const));

test.each(ROWS)("a structured agent over the limit self-corrects under retries (%s, %d levels)", async (_, config, levels) => {
  const model = texts(config, `{"value": ${deepText(levels)}}`, '{"value": {"k": 1}}');

  expect(await runAgent(model, FREEFORM, 1)).toEqual(new Output({ k: 1n }));
  expect(model.requests).toHaveLength(2);
  expectReplyTooDeep(correction(model), "structured output");
  expect(correction(model)).not.toContain("provider error (");
});

test.each([125, 1500])("a deep native reply that misses the schema retries (%d levels)", async (levels) => {
  const model = texts(NATIVE, `{"t": ${deepText(levels)}, "n": "notint"}`, '{"t": {}, "n": 1}');

  expect(await runAgent(model, T_AND_N, 1)).toEqual(new Output({ t: {}, n: 1n }));
  // The nesting is checked before the declared types, so the correction names it, not `n`.
  expectReplyTooDeep(correction(model), "structured output");
  expect(correction(model)).toContain("structured output at t.a.a");
  expect(correction(model)).not.toContain("provider error (");
});

test.each([125, 1500])("an exhausted deep native schema miss keeps the node origin (%d levels)", async (levels) => {
  const origins = new ContractOrigins({ default: new OriginRef("output"), entries: [[["n"], new OriginRef("output/n")]] });
  const model = texts(NATIVE, `{"t": ${deepText(levels)}, "n": "notint"}`);

  const outcome = await runAgent(model, T_AND_N, 0, origins);

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect(failure.code).toBe("agent.output_invalid");
  expect(failure.trace).toEqual([new OriginRef("node")]);
});

test.each(ROWS)("a structured agent over the limit without retries fails (%s, %d levels)", async (_, config, levels) => {
  const model = texts(config, `{"value": ${deepText(levels)}}`);

  const outcome = await runAgent(model, FREEFORM, 0, new ContractOrigins({ default: new OriginRef("output") }));

  expect(outcome).toBeInstanceOf(EvaluationFailure);
  const { failure } = outcome as EvaluationFailure;
  expect([failure.code, failure.nodeId]).toEqual(["agent.output_invalid", "agent"]);
  expectReplyTooDeep(failure.message, "structured output");
  expect(failure.trace).toEqual([new OriginRef("node")]);
});

// ---------- the tool loop's argument checks ----------

/** A `lookup` tool that records the arguments of each call in `calls`. */
function lookupTool(calls: unknown[]): FunctionTool {
  return new FunctionTool({
    name: "lookup",
    description: "Look one thing up.",
    parametersJsonSchema: { type: "object", properties: { x: {}, q: {} } },
    execute: (args) => {
      calls.push(args);
      return "ok";
    },
  });
}

/** A run with an AGENT `agent` granted a `lookup` tool that records its arguments. */
function toolRun(model: ScriptedModel, options: { emitEvent?: (event: unknown) => void } = {}) {
  const calls: unknown[] = [];
  const lookup = lookupTool(calls);
  const node = new AgentNode("agent", { prompt: "q", llmConfig: NATIVE, tools: ["lookup"] });
  const outcome = drive(node, {
    models: { env: NO_ENV, createModel: () => model },
    tools: new ToolSet({ lookup }),
    emitEvent: options.emitEvent ?? null,
  });
  return { outcome, calls };
}

/** An unparsed `lookup` call whose arguments `{"x": ...}` nest `levels` levels. */
function deepCall(levels: number, id = "c2"): ToolCallPart {
  return new ToolCallPart({ id, name: "lookup", arguments: null, rawArguments: `{"x": ${deepText(levels - 1)}}` });
}

/** A reply holding `parts`. */
function partsReply(...parts: (TextPart | ToolCallPart)[]): Response {
  return new Response({ message: new Message({ role: "assistant", parts }), finishReason: "tool_calls" });
}

/** The failure `outcome` carries; fails the test when it isn't one. */
function failureOf(outcome: unknown) {
  expect(outcome).toBeInstanceOf(EvaluationFailure);
  return (outcome as EvaluationFailure).failure;
}

test.each([101, 1500])("tool arguments over the limit fail the agent before any call (%d levels)", async (levels) => {
  const model = new ScriptedModel([partsReply(deepCall(levels))], NATIVE);
  const { outcome, calls } = toolRun(model);

  const failure = failureOf(await outcome);
  expect([failure.code, failure.nodeId]).toEqual(["node.execute", "agent"]);
  expectReplyTooDeep(failure.message, "tool call 'lookup' arguments");
  expect(calls).toEqual([]);
  expect(model.requests).toHaveLength(1);
});

test("tool arguments over the limit in a later call stop every call", async () => {
  const first = new ToolCallPart({ id: "c1", name: "lookup", arguments: { x: 1.5 } });
  const { outcome, calls } = toolRun(new ScriptedModel([partsReply(first, deepCall(101))], NATIVE));

  const failure = failureOf(await outcome);
  expect(failure.code).toBe("node.execute");
  expect(failure.message).toContain("deeper than 100 levels");
  expect(calls).toEqual([]);
});

test("tool arguments at the limit run", async () => {
  const atLimit = nestRecord(99);
  const call = new ToolCallPart({ id: "1", name: "lookup", arguments: { q: atLimit } as JsonObject });
  const { outcome, calls } = toolRun(new ScriptedModel([partsReply(call), partsReply(new TextPart("finished"))], NATIVE));

  expect(await outcome).toEqual(new Output("finished"));
  expect(calls).toEqual([{ q: atLimit }]);
});

const STRUCTURED_FLOW = `id: nesting
name: nesting
llm_config: {provider: openai, model: gpt-5.5}
nodes:
  w:
    kind: agent
    prompt: go
    output: {t: object}
output: \${w.output}
`;

/** A native model replying the structured `{t: ...}` texts `values`. */
function structured(...values: string[]): ScriptedModel {
  return new ScriptedModel(values.map((value) => reply(`{"t": ${value}}`)), NATIVE);
}

/** `result` failed node `w` with code `code`, naming `where` and the limit. */
function expectFailedTooDeep(result: RunResult, code: string, where: string): void {
  expect(result.status).toBe("failed");
  expect([result.failure!.code, result.failure!.nodeId]).toEqual([code, "w"]);
  expect(result.failure!.message).toContain(where);
  expect(result.failure!.message).toContain("deeper than 100 levels");
}

test.each([126, 1500])("a model value over the limit fails the agent (%d levels)", async (levels) => {
  const flow = await loadFlow(STRUCTURED_FLOW);
  // The first reply and both `retries:` attempts.
  const model = structured(deepText(levels), deepText(levels), deepText(levels));
  const result = await runFlow(flow, {}, { models: serving(model), env: NO_ENV });

  expectFailedTooDeep(result, "agent.output_invalid", "structured output at t.a");
});

test.each([124, 1500])("tool arguments over the limit fail the agent (%d levels)", async (levels) => {
  const flow = await loadFlow(`id: nesting
name: nesting
llm_config: {provider: openai, model: gpt-5.5}
nodes:
  w:
    kind: agent
    prompt: go
    tools: [lookup]
    output: str
output: \${w.output}
`);
  const rawArguments = `{"q": ${deepText(levels)}}`;
  const call = new ToolCallPart({ id: "1", name: "lookup", arguments: null, rawArguments });
  const model = new ScriptedModel([partsReply(call)], NATIVE);
  const calls: unknown[] = [];
  const result = await runFlow(flow, {}, {
    tools: { lookup: lookupTool(calls) },
    models: serving(model),
    env: NO_ENV,
  });

  expectFailedTooDeep(result, "node.execute", "tool call 'lookup' arguments at q.a");
  expect(calls).toEqual([]);
});

test("a too-deep later call of a streamed reply stops every call of the reply", async () => {
  const deep = `{"x": ${deepText(100)}}`;
  const events = [
    new ToolCallDelta({ index: 0, id: "c1", name: "lookup", arguments: '{"x": 1.5}' }),
    new ToolCallDelta({ index: 1, id: "c2", name: "lookup", arguments: deep.slice(0, 50) }),
    new ToolCallDelta({ index: 1, id: null, name: null, arguments: deep.slice(50) }),
    new Finish({ reason: "tool_calls" }),
  ];
  const { outcome, calls } = toolRun(new ScriptedModel([events], NATIVE));

  const failure = failureOf(await outcome);
  expect(failure.message).toContain("tool call 'lookup' arguments at x.a");
  expect(failure.message).toContain("deeper than 100 levels");
  expect(calls).toEqual([]);
});

test("live events show the model call but no tool call", async () => {
  const observed: unknown[] = [];
  const { outcome } = toolRun(new ScriptedModel([partsReply(deepCall(101))], NATIVE), {
    emitEvent: (event) => void observed.push(event),
  });

  failureOf(await outcome);
  const finished = observed.filter((event) => event instanceof ModelCallFinished);
  expect(finished).toHaveLength(1);
  expect(finished.every((event) => event.error === null)).toBe(true);
  expect(observed.some((event) => event instanceof ToolCallStarted)).toBe(false);
});

test("an over-deep call on a final reply fails it too", async () => {
  const model = new ScriptedModel([partsReply(new TextPart("the answer"), deepCall(101))], NATIVE);
  const { outcome, calls } = toolRun(model);

  expectReplyTooDeep(failureOf(await outcome).message, "tool call 'lookup' arguments");
  expect(calls).toEqual([]);
});

/** Return a flow-style YAML literal nesting `{a: ...}` `levels` levels deep. */
const flowLiteral = (levels: number) => "{a: ".repeat(levels) + "1" + "}".repeat(levels);

/** Return the 1-based number of the first line of `text` containing `needle`. */
const lineOf = (text: string, needle: string) => text.split("\n").findIndex((line) => line.includes(needle)) + 1;

// The path of the 101st level of `flowLiteral(101)`, and the nesting refusal.
const DEEP_PATH = Array(100).fill("a").join(".");
const TOO_DEEP = "value nests deeper than 100 levels, the limit for a yian value (each record and list is a level)";

// A written-out default past the limit is a load error; `applyDefaults` and `bindParams` above keep
// the run-time check covered for programmatic defaults.
test("an agent variable default over the limit fails at load", async () => {
  const text = `
id: nesting
name: nesting
llm_config: {provider: ollama, model: test}
nodes:
  writer:
    kind: agent
    variables:
      t: !object ${flowLiteral(101)}
    prompt: go
    output: str
output: \${writer.output}
`;

  const error = await loadErrorAsync(loadFlow(text));

  expect(error.diagnostic.message).toBe(`variable 't' default at ${DEEP_PATH}: ${TOO_DEEP}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "t: !object"));
});

test("a START default over the limit fails at load", async () => {
  const text = `
id: nesting
name: nesting
input:
  x: !object ${flowLiteral(101)}
nodes:
  c:
    kind: code
    input:
      y: \${input.x}
    code: |
      return 1
output: \${c.output}
`;

  const error = await loadErrorAsync(loadFlow(text));

  expect(error.diagnostic.message).toBe(`input 'x' default at ${DEEP_PATH}: ${TOO_DEEP}`);
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(lineOf(text, "x: !object"));
});

// ---------- runs: outputs, inputs and bindings ----------

const dir = mkdtempSync(join(tmpdir(), "yian-value-nesting-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** Return a flow whose `deep` CODE node runs the inline `body`. */
function codeFlow(body: string, options: { outputLine?: string; flowOutput?: string } = {}): string {
  const { outputLine = "", flowOutput = "${deep.output}" } = options;
  const indented = body
    .trim()
    .split("\n")
    .map((line) => `      ${line}`)
    .join("\n");
  return `
id: nesting
name: nesting
nodes:
  deep:
    kind: code
${outputLine}
    code: |
${indented}
output: ${flowOutput}
`;
}

/** A CODE body returning `{tree: v}`, `v` nested `levels` records deep. */
const treeBody = (levels: number) =>
  `let v = "leaf"\nfor (let i = 0; i < ${levels}; i++) v = { a: v }\nreturn { tree: v }`;

/** A CODE body returning a list chain `levels` levels deep. */
const listBody = (levels: number) => `let v = "leaf"\nfor (let i = 0; i < ${levels}; i++) v = [v]\nreturn v`;

/** Return `leaf` wrapped in `levels` lists. */
function nestList(levels: number, leaf: Value = "leaf"): Value {
  let value = leaf;
  for (let i = 0; i < levels; i++) value = [value];
  return value;
}

const TREE_OUTPUT = "    output: {tree: object}";

/** Load `text` with the conformance fixtures and run it on `inputs` in `hostEnv`, with `options`. */
async function run(text: string, inputs: Record<string, unknown> = {}, options: Parameters<typeof runFlow>[2] = {}) {
  return runFlow(await loadConformanceFlow(text), inputs, { env: hostEnv(dir), ...options });
}

/** Expect the run failed `nodeId`'s output contract on value depth. */
function expectTooDeep(result: RunResult, nodeId: string): void {
  expect(result.status).toBe("failed");
  const failure = result.failure!;
  expect([failure.code, failure.nodeId]).toEqual(["contract.output", nodeId]);
  expect(failure.message).toContain("deeper than 100 levels");
  expect(failure.message).not.toContain("Circular");
}

/** Expect `nodeId` failed `binding.evaluate` on value depth with a `prefix` message. */
function expectBindingTooDeep(result: RunResult, nodeId: string, prefix: string): void {
  expect(result.status).toBe("failed");
  const failure = result.failure!;
  expect([failure.code, failure.nodeId]).toEqual(["binding.evaluate", nodeId]);
  expect(failure.message).toStartWith(prefix);
  expect(failure.message).toContain("deeper than 100 levels");
}

test("a record output at the limit succeeds", async () => {
  const result = await run(codeFlow(treeBody(99), { outputLine: TREE_OUTPUT }));
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ tree: nestRecord(99) });
});

test("a record output over the limit fails at the field", async () => {
  const result = await run(codeFlow(treeBody(100), { outputLine: TREE_OUTPUT }));
  expectTooDeep(result, "deep");
  expect(result.failure!.message).toStartWith("output at tree.a");
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:deep/s:output/s:contract/s:path/s:tree");
});

test.each([200, 1500])("a far too deep inline output fails without throwing (%d levels)", async (levels) => {
  expectTooDeep(await run(codeFlow(treeBody(levels), { outputLine: TREE_OUTPUT })), "deep");
});

test("a far too deep reference output fails on depth", async () => {
  const text = `
id: nesting
name: nesting
input:
  n: !int
nodes:
  deep:
    kind: code
    input:
      n: \${input.n}
    code: ./fixtures/compose_codefns.ts:deep_tree
output: \${deep.output}
`;
  const result = await run(text, { n: 1500 });
  expectTooDeep(result, "deep");
  expect(result.failure!.message).not.toContain("canonical data nesting");
});

test("an untyped output over the limit fails", async () => {
  expectTooDeep(await run(codeFlow(listBody(101))), "deep");
});

/** `codeFlow` of `body` with the post-assert `output == null`, which a list result fails. */
const assertedFlow = (body: string) =>
  codeFlow(body).replace("    code: |", '    asserts:\n      - "output == null"\n    code: |');

test("a CODE result over the limit meets its post-asserts before the commit refuses it", async () => {
  const result = await run(assertedFlow(listBody(101)));
  expect(result.status).toBe("failed");
  expect([result.failure!.code, result.failure!.nodeId]).toEqual(["assertion.failed", "deep"]);
});

test("a CODE result too deep to copy back fails its output before its post-asserts", async () => {
  const deep = await run(assertedFlow(listBody(10_000)));
  expectTooDeep(deep, "deep");
  // The worker's refusal reads as the commit's: same message and trace, the trace located at the
  // output field the too-deep value sits in.
  const field = { outputLine: "    output: {s: Any}" };
  const wrapped = (levels: number) => listBody(levels).replace("return v", "return { s: v }");
  const deepField = await run(codeFlow(wrapped(10_000), field));
  const committed = await run(codeFlow(wrapped(150), field));
  expectTooDeep(deepField, "deep");
  expectTooDeep(committed, "deep");
  expect([deepField.failure!.message, deepField.failure!.trace]).toEqual([
    committed.failure!.message,
    committed.failure!.trace,
  ]);
});

test("a map result list at the limit succeeds", async () => {
  const text = `
id: nesting
name: nesting
nodes:
  m:
    kind: map
    over: "\${[1, 2]}"
    input:
      y: !int \${item}
    nodes:
      work:
        kind: code
        input:
          y: \${input.y}
        code: |
          let v = inputs["y"]
          for (let i = 0; i < 99; i++) v = [v]
          return v
    output: \${work.output}
output: \${m.output}
`;
  const result = await run(text);
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([nestList(99, 1n), nestList(99, 2n)]);
});

const ECHO_X = `
id: nesting
name: nesting
input:
  x: !object
nodes:
  echo:
    kind: code
    input:
      x: \${input.x}
    code: |
      return inputs["x"]
output: \${echo.output}
`;

test("a run input record may carry a value at the limit", async () => {
  const value = nestRecord(100);
  const result = await run(ECHO_X, { x: value });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual(value);
});

test.each([101, 1500])("a run input over the limit is rejected before the run starts (%d levels)", async (levels) => {
  // START's committed record may sit one level over the limit, so the input check is what keeps
  // a run input record from going further.
  const text = "id: nesting\nname: nesting\ninput:\n  x: !object\nnodes: {}\noutput: ${input.x}\n";
  const log = join(dir, `input-${levels}.jsonl`);
  const error = await rejected(run(text, { x: nestRecord(levels) }, { log }));
  expect(error).toBeInstanceOf(TypeCheckError);
  expect((error as TypeCheckError).message).toStartWith("input 'x' at a.a");
  expect((error as TypeCheckError).message).toContain("deeper than 100 levels");
  expect((error as TypeCheckError).path).toHaveLength(100);
  expect(existsSync(log)).toBe(false);
});

const WRAPPED_BINDING_FLOW = `
id: nesting
name: nesting
input:
  x: !object
nodes:
  c:
    kind: code
    input:
      y: \${[input.x]}
    code: |
      return 1
output: \${c.output}
`;

test("a binding around a value at the limit fails its node", async () => {
  const result = await run(WRAPPED_BINDING_FLOW, { x: nestRecord(100) });
  expectBindingTooDeep(result, "c", "input 'y' at [0].a");
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:c/s:input/s:y/s:binding");
});

test("a value template around a value at the limit fails its node", async () => {
  // A record written around a nested `${...}` adds a level, like `${[x]}`.
  const text = WRAPPED_BINDING_FLOW.replace("y: ${[input.x]}", 'y: {w: "${input.x}"}');
  expect((await run(text, { x: nestRecord(99) })).status).toBe("succeeded");
  expectBindingTooDeep(await run(text, { x: nestRecord(100) }), "c", "input 'y' at w.a");
});

/** A flow whose `k` CALL binds the def input `y: declared` to `binding`. */
const callFlow = (binding: string, declared: string) => `
id: nesting
name: nesting
input:
  x: !object
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
          return 1
    output: \${inner.output}
nodes:
  k:
    kind: call
    call: body
    input:
      y: ${binding}
output: \${k.output}
`;

test("a call bound to a value at the limit succeeds", async () => {
  const result = await run(callFlow("${input.x}", "object"), { x: nestRecord(100) });
  expect([result.status, result.output]).toEqual(["succeeded", 1n]);
});

test("a MAP whose input binds a value at the limit succeeds", async () => {
  // The growth seed's list and record don't count toward the value's nesting.
  const text = `
id: nesting
name: nesting
input:
  x: !object
nodes:
  m:
    kind: map
    over: \${[1]}
    input:
      y: !object \${input.x}
    nodes:
      work:
        kind: code
        code: |
          return 1
    output: \${work.output}
output: \${m.output}
`;
  const result = await run(text, { x: nestRecord(100) });
  expect([result.status, result.output]).toEqual(["succeeded", [1n]]);
});

test("a LOOP carrying a value at the limit fails its body's end, not on growth", async () => {
  // The seed's pair and record don't count; the body's output record around the value does.
  const text = `
id: nesting
name: nesting
input:
  x: !object
defs:
  keep:
    input:
      y: !object
      done: !bool
    nodes: {}
    output:
      y: \${input.y}
      done: true
nodes:
  loop:
    kind: loop
    call: keep
    input:
      y: \${input.x}
      done: false
    while: not done
    max: 3
output: \${loop.output.done}
`;
  const result = await run(text, { x: nestRecord(100) });
  expect(result.status).toBe("failed");
  const failure = result.failure!;
  expect([failure.code, failure.nodeId]).toEqual(["contract.output", "loop#0/__end__"]);
  expect(failure.message).toStartWith(`output at y${".a".repeat(99)}: value nests deeper than 100 levels`);
});

test("a call bound over the limit fails the call, not its START", async () => {
  const result = await run(callFlow("${[input.x]}", "list(object)"), { x: nestRecord(100) });
  expectBindingTooDeep(result, "k", "input 'y' at [0].a");
});

test("a MAP over a list around a value at the limit fails the MAP", async () => {
  const text = `
id: nesting
name: nesting
input:
  x: !object
nodes:
  m:
    kind: map
    over: \${[input.x]}
    input:
      y: !object \${item}
    nodes:
      work:
        kind: code
        code: |
          return 1
    output: \${work.output}
output: \${m.output}
`;
  const result = await run(text, { x: nestRecord(100) });
  expectBindingTooDeep(result, "m", "MAP node 'm': `over`");
  expect(result.failure!.trace[0]!.key).toBe("/s:root/s:node/s:m/s:reserved/s:over/s:binding");
});

// ---------- runs: replay ----------

test("replay accepts a value recorded before the limit", async () => {
  const text = codeFlow(listBody(120));
  const path = join(dir, "before-the-limit.jsonl");
  install({ ...BUILTIN, maxValueNesting: 150 });
  const recorded = await runFlow(await loadConformanceFlow(text), {}, { log: path, env: processEnv });
  install({ ...BUILTIN, modelRetries: 0 });
  expect(recorded.status).toBe("succeeded");
  // A log from before the limit existed recorded no value for it.
  const lines = readFileSync(path, "utf8").split(/(?<=\n)/);
  const envelope = JSON.parse(lines[0]!);
  expect(envelope.payload.defaults).toEqual({ MAX_VALUE_NESTING: 150 });
  envelope.payload.defaults = {};
  lines[0] = encodeRecord(decodeRecord(JSON.stringify(envelope))) + "\n";
  writeFileSync(path, lines.join(""));

  const replayed = await advance({ log: path, env: hostEnv(dir) });

  expect(replayed.status).toBe("succeeded");
  expect(replayed.failure).toBeNull();
  expect(replayed.output).toEqual(recorded.output);
  expect(replayed.output).toEqual(nestList(120));
});
