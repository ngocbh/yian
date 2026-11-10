/**
 * An `openai` block with `store: false` makes the API key route's requests stateless, through a
 * real run and its resume; without it the requests and their recorded digests stay as they were.
 */

import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterAll, expect, test } from "bun:test";

import { loadFlow } from "../../src/compose/loader.ts";
import type { Env } from "../../src/defaults.ts";
import { DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { advance, agentCall, agentCalls, runFlow } from "../../src/run.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { PLAIN_FLOW, recording, type Replier, sse } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-openai-stateless-flow-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

/** The agent flow with the `double` tool, on `gpt-5.5` with `block` as its `openai` block. */
function toolFlow(block: string | null): string {
  const config = block === null ? "{provider: openai, model: gpt-5.5}" : `{provider: openai, model: gpt-5.5, openai: ${block}}`;
  return PLAIN_FLOW.replace("{provider: openai, model: test-model}", config).replace(
    "    kind: agent\n",
    "    kind: agent\n    tools: [double]\n",
  );
}

let files = 0;

/** A fresh folder, with an environment holding an OpenAI API key. */
function fresh(): { env: Env; root: string } {
  const root = join(dir, String(files++));
  mkdirSync(root, { mode: 0o700 });
  return { env: hostEnv(root, { OPENAI_API_KEY: "sk-test-stateless" }), root };
}

const double = new FunctionTool({
  name: "double",
  description: "Double an integer.",
  parametersJsonSchema: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
  execute: (args) => (args["value"] as number) * 2,
});

const REASONING = { id: "rs_test", type: "reasoning", summary: [], encrypted_content: "synthetic-continuation" };
const TOOL_CALL = {
  id: "fc_test",
  type: "function_call",
  name: "double",
  arguments: '{"value": 3}',
  call_id: "call_test",
  status: "completed",
};
const ANSWER = {
  id: "msg_test",
  type: "message",
  role: "assistant",
  status: "completed",
  content: [{ type: "output_text", text: "6", annotations: [] }],
};

/** A Responses server: a reasoning item, then a call of `double` on the first request or the text `6` after. */
const server: Replier = (_sent, number) => {
  const last = number === 1 ? TOOL_CALL : ANSWER;
  const response = { id: `resp_${number}`, status: "completed", model: "gpt-5.5", output: [REASONING, last] };
  return sse([
    ["response.created", { response: { ...response, status: "in_progress", output: [] } }],
    ["response.output_item.done", { output_index: 0, item: REASONING }],
    ["response.output_item.done", { output_index: 1, item: last }],
    ["response.completed", { response }],
  ]);
};

/** The JSON lines of the run log at `path`. */
const lines = (path: string) => readFileSync(path, "utf8").split("\n").filter((line) => line !== "");

/** The steps of the agent's `model_call` steps in the log at `path`. */
function modelCalls(path: string): Record<string, unknown>[] {
  return lines(path)
    .map((line) => JSON.parse(line))
    .filter((record) => record.kind === "node_step" && record.payload.step.kind === "model_call")
    .map((record) => record.payload.step);
}

/** Run the tool flow with `block` to a log in a fresh folder; return the bodies sent and the log. */
async function run(block: string | null) {
  const { env, root } = fresh();
  const models = recording(env, server);
  const log = join(root, "run.jsonl");
  const flow = await loadFlow(toolFlow(block), { env });
  const result = await runFlow(flow, { question: "Double 3." }, { tools: { double }, models, env, log });
  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("6");
  return { env, root, log, bodies: models.requests.map((sent) => sent.body as Record<string, unknown>) };
}

/** Check that every model call of the log at `path` rebuilds to the request its digest recorded. */
async function expectRebuilt(path: string, env: Env): Promise<void> {
  const calls = agentCalls(path);
  expect(calls.length).toBe(2);
  for (const call of calls) {
    const view = await agentCall(path, call, { env });
    expect(view.request).not.toBeNull();
    expect(view.digestMismatch).toBe(false);
  }
}

const USER = { type: "message", role: "user", content: "Double 3." };
const OUTPUT = { type: "function_call_output", call_id: "call_test", output: "6" };
const TOOLS = [
  {
    type: "function",
    name: "double",
    description: "Double an integer.",
    parameters: { type: "object", properties: { value: { type: "integer" } }, required: ["value"] },
  },
];

test("store false sends stateless requests, replaying reasoning with its encrypted content", async () => {
  const { log, bodies } = await run("{store: false}");
  const stateless = { store: false, include: ["reasoning.encrypted_content"], instructions: DEFAULT_SYSTEM };
  expect(bodies).toEqual([
    { model: "gpt-5.5", input: [USER], stream: true, tools: TOOLS, ...stateless },
    { model: "gpt-5.5", input: [USER, REASONING, TOOL_CALL, OUTPUT], stream: true, tools: TOOLS, ...stateless },
  ]);
  for (const body of bodies) {
    expect(Object.keys(body)).toEqual(["model", "input", "stream", "tools", "store", "include", "instructions"]);
  }
  // The log records the reasoning with its encrypted content, which the next turn replays.
  for (const step of modelCalls(log)) {
    const parts = (step["output"] as { parts: { provider_payload: { item?: unknown } | null }[] }).parts;
    expect(parts[0]!.provider_payload!.item).toEqual(REASONING);
  }
});

for (const block of ["{store: false}", null]) {
  test(`a run cut after its tool result resumes with the same second request (${block ?? "no block"})`, async () => {
    const whole = await run(block);
    await expectRebuilt(whole.log, whole.env);

    const records = lines(whole.log);
    const cut = records.findIndex((line) => {
      const record = JSON.parse(line);
      return record.kind === "node_step" && record.payload.step.kind === "tool_result";
    });
    expect(cut).toBeGreaterThan(0);
    const { env, root } = fresh();
    const log = join(root, "run.jsonl");
    writeFileSync(log, records.slice(0, cut + 1).join("\n") + "\n");
    const models = recording(env, (sent) => server(sent, 2));
    const resumed = await advance({ log, tools: { double }, models, env });
    expect(resumed.status, resumed.failure?.message).toBe("succeeded");
    expect(resumed.output).toBe("6");
    // Equal as JSON: a replayed item's members come back from the log in its sorted key order.
    expect(models.requests.map((sent) => sent.body)).toEqual([whole.bodies[1]!]);
    expect(modelCalls(log).map((step) => step["request_sha256"])).toEqual(
      modelCalls(whole.log).map((step) => step["request_sha256"]),
    );
    await expectRebuilt(log, env);
  });
}

test("without store the requests and their recorded digests are today's", async () => {
  const { env, log, bodies } = await run(null);
  const system = JSON.stringify({ type: "message", role: "system", content: DEFAULT_SYSTEM });
  const tools = JSON.stringify(TOOLS);
  expect(bodies.map((body) => JSON.stringify(body))).toEqual([
    `{"model":"gpt-5.5","input":[${system},${JSON.stringify(USER)}],"stream":true,"tools":${tools}}`,
    `{"model":"gpt-5.5","input":[${system},${JSON.stringify(USER)},${JSON.stringify(REASONING)},` +
      `${JSON.stringify(TOOL_CALL)},${JSON.stringify(OUTPUT)}],"stream":true,"tools":${tools}}`,
  ]);
  expect(modelCalls(log).map((step) => step["request_sha256"])).toEqual([
    "c43df876a62efe4393899a0574e0c56d6fd6c4924dc557d19825236e4a1f1114",
    "38fdf78e6eaff477a5f1cfb3638792b75aed4db158c12af6c00d6794afc04aea",
  ]);
  await expectRebuilt(log, env);
});
