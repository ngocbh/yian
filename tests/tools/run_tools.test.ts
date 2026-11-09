/**
 * Per-run tools through the evaluator: `evalNode` and `FlowEngine` give TOOL and AGENT nodes the
 * run's `ToolSet` and `ask` handler.
 */

import { expect, test } from "bun:test";

import { END_ID, START_ID } from "../../src/compile/model.ts";
import { Message, Response, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node, type NodeCaps, NodeKind, type NodeResult, Output } from "../../src/nodes/base.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { evalNode } from "../../src/runtime/eval_node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { type AskHandler, type AskRequest, ToolSet } from "../../src/tools/registry.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { engine, graph, serving, textReply } from "../helpers/run_state_fixtures.ts";

/** A tool that returns a fixed reply. */
const PING = new FunctionTool({
  name: "ping",
  description: "Return a fixed reply.",
  parametersJsonSchema: { type: "object", properties: {} },
  execute: () => "run-pong",
});

/** An `ask_user` handler that records each request in `seen` and returns `answer`. */
function recordingHandler(seen: AskRequest[], answer: unknown = null): AskHandler {
  return (request) => {
    seen.push(request);
    return answer;
  };
}

/** Record the capabilities each evaluation supplies. */
class CapsProbe extends Node {
  readonly kind = NodeKind.CODE;
  seen: NodeCaps = {};

  override forward(_inputs: Record<string, unknown>, caps: NodeCaps): NodeResult {
    this.seen = caps;
    return new Output(null);
  }
}

/** The same probe with the tools trait. */
class ToolsProbe extends CapsProbe {
  override readonly needsTools = true;
}

test("only nodes that declare needsTools receive the toolset", async () => {
  const tools = new ToolSet({ ping: PING });
  const plain = new CapsProbe("plain");
  const optedIn = new ToolsProbe("opted_in");

  await evalNode(plain, null, new VariablePool(), { tools });
  await evalNode(optedIn, null, new VariablePool(), { tools });

  expect([Object.keys(plain.seen).sort(), Object.keys(optedIn.seen).sort()]).toEqual([["signal"], ["signal", "tools"]]);
  expect(optedIn.seen.tools).toBe(tools);
});

test("a node declaring needsTools without a run tool set receives a fresh ToolSet", async () => {
  const node = new ToolsProbe("opted_in");

  await evalNode(node, null, new VariablePool());

  expect(node.seen.tools).toBeInstanceOf(ToolSet);
});

test("eval node passes the run toolset to a tool node", async () => {
  const node = new ToolNode("ping", { toolId: "ping" });

  const evaluated = await evalNode(node, null, new VariablePool(), { tools: new ToolSet({ ping: PING }) });

  expect(evaluated.outcome).toEqual(new Output("run-pong"));
});

test("flow engine passes its toolset to every tool node", async () => {
  const flow = graph([new ToolNode("ping", { toolId: "ping" })], [
    [START_ID, "ping"],
    ["ping", END_ID],
  ]);
  const built = engine(flow, { tools: new ToolSet({ ping: PING }) });

  await built.run();

  expect(built.state.pool.get("ping")).toBe("run-pong");
});

test("human input asks the run ask_user handler", async () => {
  const seen: AskRequest[] = [];
  const tools = new ToolSet({ ask: recordingHandler(seen, "yes") });
  const node = new ToolNode("approval", { toolId: "ask", title: "Gate" });

  expect(await node.run({ prompt: "Ship v2?" }, { tools })).toEqual(new Output("yes"));
  expect(seen).toEqual([
    {
      question: "Ship v2?",
      options: null,
      questions: null,
      answerSchema: null,
      title: "Gate",
      nodeId: "approval",
      // Outside a run each call gets a fresh id.
      callId: expect.any(String),
      deadline: null,
      extra: {},
    },
  ]);
});

test("eval node gives human input the run toolset", async () => {
  const seen: AskRequest[] = [];
  const tools = new ToolSet({ ask: recordingHandler(seen, "yes") });

  const node = new ToolNode("ask", { toolId: "ask" });

  const evaluated = await evalNode(node, null, new VariablePool(), { tools });

  expect(evaluated.outcome).toEqual(new Output("yes"));
  expect(seen.map((request) => request.nodeId)).toEqual(["ask"]);
});

// ---------- the AGENT tool loop ----------

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** A tool `name` taking `text`, returning `transform(text)`. */
function textTool(name: string, transform: (text: string) => string): FunctionTool {
  return new FunctionTool({
    name,
    description: name,
    parametersJsonSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: (args) => transform(args.text as string),
  });
}

const upper = (name = "shout") => textTool(name, (text) => text.toUpperCase());
/** A tool that echoes its text under `name`. */
const named = (name: string) => textTool(name, (text) => `${name}:${text}`);

/** A model reply calling one tool as `call-1`. */
function toolCallReply(name: string, args: JsonObject): Response {
  const call = new ToolCallPart({ id: "call-1", name, arguments: args });
  return new Response({ message: new Message({ role: "assistant", parts: [call] }), finishReason: "tool_calls" });
}

/** One tool-call reply per call, then a final answer. */
function replies(...calls: [string, JsonObject][]): Response[] {
  return [...calls.map(([name, args]) => toolCallReply(name, args)), textReply("done")];
}

/** Evaluate an AGENT `agent` granted `grants` over `model` with the run's `tools`; return its outcome. */
async function runAgentNode(model: ScriptedModel, grants: string[], tools: ToolSet) {
  const node = new AgentNode("agent", { prompt: "go", llmConfig: CONFIG, tools: grants });
  return (await evalNode(node, null, new VariablePool(), { tools, models: serving(model) })).outcome;
}

/** The names of the tools bound in `model`'s first request. */
function boundNames(model: ScriptedModel): string[] {
  return model.requests[0]!.tools.map((tool) => tool.name);
}

/** The tool results' contents in `model`'s last request. */
function resultContents(model: ScriptedModel): string[] {
  return model.requests.at(-1)!.messages.flatMap((message) =>
    message.parts.filter((part) => part instanceof ToolResultPart).map((part) => part.content),
  );
}

test("agent binds and calls granted per-run tools", async () => {
  const model = new ScriptedModel(replies(["shout", { text: "hi" }]), CONFIG);
  const seen: AskRequest[] = [];
  const tools = new ToolSet({ shout: upper(), ask_user: recordingHandler(seen) });

  const outcome = await runAgentNode(model, ["shout", "ask_user"], tools);

  expect(outcome).toEqual(new Output("done"));
  expect(boundNames(model)).toEqual(["shout", "ask_user"]);
  expect(Object.keys(model.requests[0]!.tools[1]!.parameters.properties as object).sort()).toEqual(["options", "question", "timeout"]);
  expect(model.requests[1]!.messages.at(-1)!.parts).toEqual([
    new ToolResultPart({ callId: "call-1", name: "shout", content: "HI" }),
  ]);
  expect(seen).toEqual([]);
});

test("agent cannot call a per-run tool the flow did not grant: the model reads it's unavailable", async () => {
  const model = new ScriptedModel(replies(["shout", { text: "hi" }]), CONFIG);
  const tools = new ToolSet({ shout: upper(), ask_user: () => "yes" });

  const outcome = await runAgentNode(model, ["ask_user"], tools);

  expect(outcome).toEqual(new Output("done"));
  expect(resultContents(model)).toEqual(["Unavailable tool: shout"]);
});

test("agent granted three groups sees the union of their tools", async () => {
  const model = new ScriptedModel(replies(["toolbox_a", { text: "1" }], ["workspace_c", { text: "3" }]), CONFIG);
  const tools = new ToolSet({
    toolbox: [named("a")],
    workspace: [named("b"), named("c")],
    company: [named("d")],
    hidden: [named("e")],
  });

  const outcome = await runAgentNode(model, ["toolbox", "workspace", "company"], tools);

  expect(outcome).toEqual(new Output("done"));
  expect(boundNames(model)).toEqual(["toolbox_a", "workspace_b", "workspace_c", "company_d"]);
  expect(resultContents(model)).toEqual(["a:1", "c:3"]);
});

test.each(["e", "toolbox", "a"])("agent grant check sees expanded group names (%s)", async (name) => {
  const model = new ScriptedModel(replies([name, { text: "x" }]), CONFIG);
  const tools = new ToolSet({ toolbox: [named("a")], hidden: [named("e")] });

  const outcome = await runAgentNode(model, ["toolbox"], tools);

  expect(outcome).toEqual(new Output("done"));
  expect(resultContents(model)).toEqual([`Unavailable tool: ${name}`]);
});

test("agent binds a tool shared by overlapping groups once", async () => {
  const model = new ScriptedModel(replies(["toolbox_web", { text: "q" }]), CONFIG);
  const web = named("web");
  const tools = new ToolSet({ toolbox: [web, named("a")], company: [web, named("b")] });

  const outcome = await runAgentNode(model, ["toolbox", "company", "toolbox"], tools);

  expect(outcome).toEqual(new Output("done"));
  expect(boundNames(model)).toEqual(["toolbox_web", "toolbox_a", "company_b"]);
});

test("agent with groups and ask_user asks the top-level handler", async () => {
  const model = new ScriptedModel(replies(["ask_user", { question: "Which?" }], ["toolbox_a", { text: "go" }]), CONFIG);
  const seen: AskRequest[] = [];
  const tools = new ToolSet({ toolbox: [named("a")], ask_user: recordingHandler(seen, "B") });
  const node = new AgentNode("agent", { prompt: "go", llmConfig: CONFIG, tools: ["toolbox", "ask_user"] });
  const flow = graph([node], [
    [START_ID, "agent"],
    ["agent", END_ID],
  ]);
  const built = engine(flow, { tools, models: serving(model) });

  await built.run();

  expect(built.state.pool.get("agent")).toBe("done");
  expect(boundNames(model)).toEqual(["toolbox_a", "ask_user"]);
  expect(seen.map((request) => request.question)).toEqual(["Which?"]);
  expect(resultContents(model)).toEqual(["B", "a:go"]);
});
