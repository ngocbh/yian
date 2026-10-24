/**
 * Every shipped example loads and compiles (no live LLM): mirrors
 * `tests/examples/test_examples.py`.
 *
 * Ported: `test_example_loads_and_compiles` (one test per `examples/*.yaml`, loaded with the
 * examples directory as the search path), `test_examples_present` and
 * `test_debate_agents_are_tool_free`. TS-only: the shape of `human-approval.yaml`'s `ask` gate,
 * and `chat.yaml` on its `Inbox(str)` input, run end to end on a scripted model.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { AgentNode } from "../../src/nodes/agent/node.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { advance, close, runFlow, send } from "../../src/run.ts";
import { ASK } from "../../src/tools/context.ts";
import { isInboxType, ValueKind } from "../../src/typesys/values.ts";
import { NO_ENV, serving, textReply } from "../engine/run_state_fixtures.ts";

const EXAMPLES_DIR = fileURLToPath(new URL("../../../examples", import.meta.url));
const EXAMPLES = readdirSync(EXAMPLES_DIR)
  .filter((name) => name.endsWith(".yaml"))
  .sort();

const load = (name: string) =>
  loadFlow(readFileSync(join(EXAMPLES_DIR, name), "utf8"), { searchPaths: [EXAMPLES_DIR] });

test.each(EXAMPLES)("example %s loads and compiles", async (name) => {
  const loaded = await load(name);
  expect(loaded.compiled).toBeDefined();
  expect(loaded.input).toBeDefined();
});

test("examples present", () => {
  for (const name of ["hello.yaml", "summarize.yaml", "classify.yaml"]) {
    expect(EXAMPLES).toContain(name);
  }
});

test("debate agents are tool-free", async () => {
  const loaded = await load("debate.yaml");
  for (const nodeId of ["frame", "for_case", "against_case", "verdict"]) {
    const node = loaded.compiled.nodes.get(nodeId) as AgentNode;
    expect(node.tools).toEqual([]);
  }
});

test("human-approval's approve is a TOOL ask answered with the Approval enum", async () => {
  const loaded = await load("human-approval.yaml");
  const node = loaded.compiled.nodes.get("approve");
  expect(node).toBeInstanceOf(ToolNode);
  expect((node as ToolNode).builtin).toBe(ASK);
  expect((node as ToolNode).answerSchema).toEqual([
    { name: "answer", type: "str", required: true, enum: ["approve", "revise"] },
  ]);
});

test("chat's one input is the `messages` inbox of text", async () => {
  const loaded = await load("chat.yaml");
  const input = loaded.input!;
  expect(input.map((param) => param.name)).toEqual(["messages"]);
  const type = input[0]!.contract!.type;
  expect(isInboxType(type)).toBe(true);
  expect(type!.element?.kind).toBe(ValueKind.STRING);
});

const tmp = mkdtempSync(join(tmpdir(), "yian-examples-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

test("chat waits for each message, answers it, and ends when its inbox closes", async () => {
  const loaded = await load("chat.yaml");
  const config = { provider: "anthropic", model: "claude-test" };
  const models = serving(new ScriptedModel([textReply("hello")], config));
  const log = join(tmp, "chat.jsonl");
  const options = { env: NO_ENV, numWorkers: 0, models };
  let result = await runFlow(loaded, {}, { ...options, log, llmConfig: config });
  expect(result.status).toBe("waiting");
  await send(log, "messages", "hi", { env: NO_ENV });
  result = await advance({ ...options, log });
  expect(result.status).toBe("waiting");
  await close(log, "messages", { env: NO_ENV });
  result = await advance({ ...options, log });
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual([
    { role: "user", text: "hi" },
    { role: "assistant", text: "hello" },
  ]);
}, 30_000);
