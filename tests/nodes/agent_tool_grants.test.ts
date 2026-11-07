/**
 * An AGENT's tool grants as its model sees them: a group's tools go out as `<group>_<tool>`, a call
 * is routed back through its grant to the group's tool, a call to a name the agent wasn't granted
 * fails the node naming the visible names, and a tool in two granted groups is bound once, under
 * the first grant's name. Each row runs `runAgent` directly on a scripted model.
 */

import { expect, test } from "bun:test";

import { ScriptedModel } from "../../src/llm/testing.ts";
import { MemoryContext } from "../../src/memory/contracts.ts";
import { AgentLoopError, type AgentRunContext, DEFAULT_SYSTEM } from "../../src/nodes/agent/context.ts";
import { runAgent } from "../../src/nodes/agent/loop.ts";
import { userMessage } from "../../src/nodes/agent/structured.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { parseToolGrants } from "../../src/tools/grants.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { callReply, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const CONFIG = { provider: "openai", model: "gpt-5.5" };

/** A tool `name` taking `topic`; each call runs `onCall`, and it answers `<name>:<topic>`. */
function topicTool(name: string, onCall: () => void = () => {}): FunctionTool {
  return new FunctionTool({
    name,
    description: `The ${name} tool.`,
    parametersJsonSchema: { type: "object", properties: { topic: { type: "string" } }, required: ["topic"] },
    execute: (args) => {
      onCall();
      return `${name}:${String(args.topic)}`;
    },
  });
}

/** A direct loop context for agent `a` prompted "go", with `tools` granted from `toolset`. */
function context(model: ScriptedModel, tools: AgentRunContext["tools"], toolset: ToolSet): AgentRunContext {
  return {
    nodeId: "a",
    messages: [userMessage("go")],
    promptText: "go",
    tools,
    llmConfig: CONFIG,
    outputType: null,
    retries: 0,
    maxToolIterations: -1,
    memoryContext: new MemoryContext(),
    toolset,
    baseSystemPrompt: DEFAULT_SYSTEM,
    skills: [],
    models: serving(model),
    sent: () => [],
  };
}

test("a group grant sends <group>_<tool> names and routes the call to the group's tool", async () => {
  const calls: string[] = [];
  const docs = topicTool("search", () => calls.push("docs"));
  const web = topicTool("search", () => calls.push("web"));
  const model = new ScriptedModel([callReply(["docs_search", { topic: "a" }, "c1"]), textReply("ok")], CONFIG);
  const toolset = new ToolSet({ docs: [docs], web: [web] });

  await runAgent(context(model, parseToolGrants([{ docs: ["search"] }, "web"], "n"), toolset));

  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["docs_search", "web_search"]);
  expect(model.requests[0]!.tools[0]!.description).toBe("The search tool.");
  expect(calls).toEqual(["docs"]);
});

test("a call to an ungranted visible name fails the node, listing the visible names", async () => {
  const model = new ScriptedModel([callReply(["search", { topic: "a" }, "c1"]), textReply("ok")], CONFIG);
  const toolset = new ToolSet({ docs: [topicTool("search")], web: [topicTool("search")] });

  const error = await rejected(runAgent(context(model, parseToolGrants([{ docs: ["search"] }], "n"), toolset)));

  expect(error).toBeInstanceOf(AgentLoopError);
  expect((error as Error).message).toBe(
    `agent node 'a' requested tool 'search', but it is not granted; granted tools: ["docs_search"]`,
  );
});

test("a tool in two granted groups is bound once, under the first grant's name", async () => {
  const calls: string[] = [];
  const web = topicTool("web", () => calls.push("web"));
  const model = new ScriptedModel([callReply(["toolbox_web", { topic: "a" }, "c1"]), textReply("ok")], CONFIG);
  const toolset = new ToolSet({ toolbox: [web, topicTool("a")], company: [web, topicTool("b")] });

  await runAgent(context(model, parseToolGrants(["toolbox", "company"], "n"), toolset));

  expect(model.requests[0]!.tools.map((tool) => tool.name)).toEqual(["toolbox_web", "toolbox_a", "company_b"]);
  expect(calls).toEqual(["web"]);
});
