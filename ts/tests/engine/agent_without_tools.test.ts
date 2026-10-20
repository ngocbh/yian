/**
 * An AGENT without tools or skills makes a single model call; one with tools runs the loop:
 * mirrors `tests/engine/test_agent_without_tools.py`.
 *
 * Ported: `test_skills_and_steerable_need_no_mode`.
 *
 * Changed: `test_text_agent_without_tools_makes_one_unbound_call` and
 * `test_structured_agent_without_tools_makes_one_structured_call` run a hand-built
 * `START -> chat -> END` flow through `FlowEngine` (Python: `load_flow` and `run_flow`, T3-22/T4-8),
 * with a `ScriptedModel` from `FlowEngineOptions.models`; the request sending no tools stands for
 * Python's unbound and structured calls.
 *
 * Later (T5-11b): `test_python_built_agent_with_tools_runs_the_tool_loop`,
 * `test_steer_joins_the_single_request_after_the_prompt` and
 * `test_interrupt_while_streaming_keeps_the_partial_text` (the last two with T3-22).
 */

import { afterEach, beforeEach, expect, test } from "bun:test";

import { CompiledFlow, Edge, END_ID, START_ID } from "../../src/compile/model.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { RunSucceeded } from "../../src/events.ts";
import { Message, Response, TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import type { Node } from "../../src/nodes/base.ts";
import { ValueContract } from "../../src/nodes/contracts.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { Type, ValueKind } from "../../src/typesys/values.ts";
import { engine } from "./run_state_fixtures.ts";

beforeEach(() => install({ ...BUILTIN, modelRetries: 0 }));
afterEach(forget);

const CONFIG = { provider: "anthropic", model: "claude-test" };

function reply(text: string): Response {
  return new Response({ message: new Message({ role: "assistant", parts: [new TextPart(text)] }), finishReason: "stop" });
}

/** Run `START -> chat -> END`, `chat` an AGENT with no tools and output `output`; return the run's output. */
async function run(output: Type, model: ScriptedModel): Promise<unknown> {
  const chat = new AgentNode("chat", { prompt: "Help the user.", llmConfig: CONFIG });
  chat.outputContract = new ValueContract(output);
  const flow = CompiledFlow.fromParts(
    new Map<string, Node>([
      [START_ID, new StartNode(START_ID)],
      ["chat", chat],
      [END_ID, EndNode.value(END_ID, { outputName: "output" })],
    ]),
    [
      new Edge("start-chat", START_ID, "chat", { optional: true, ordering: true }),
      new Edge("chat-end", "chat", END_ID, { inputGroup: "output" }),
    ],
    {
      wiring: new Map([
        [START_ID, new Map()],
        ["chat", new Map()],
        [END_ID, new Map([["output", "${chat.output}"]])],
      ]),
    },
  );
  const events = await engine(flow, { models: { env: () => undefined, createModel: () => model } }).run();
  const last = events.at(-1);
  expect(last).toBeInstanceOf(RunSucceeded);
  return (last as RunSucceeded).output;
}

test("text agent without tools makes one unbound call", async () => {
  const model = new ScriptedModel([reply("hello")], CONFIG);
  expect(await run(Type.scalar(ValueKind.STRING), model)).toBe("hello");
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("structured agent without tools makes one structured call", async () => {
  const output = new Type({
    kind: ValueKind.OBJECT,
    fields: new Map([
      ["name", Type.scalar(ValueKind.STRING)],
      ["score", Type.scalar(ValueKind.INTEGER)],
    ]),
    required: new Set(["name", "score"]),
  });
  const model = new ScriptedModel([reply('{"name": "Ada", "score": 3}')], CONFIG);
  expect(await run(output, model)).toEqual({ name: "Ada", score: 3n });
  expect(model.requests).toHaveLength(1);
  expect(model.requests[0]!.tools).toEqual([]);
});

test("skills and steerable need no mode", () => {
  expect(new AgentNode("n", { prompt: "p", skills: ["review"] }).skills).toEqual(["review"]);
  expect(new AgentNode("n", { prompt: "p", steerable: true }).steerable).toBe(true);
});
