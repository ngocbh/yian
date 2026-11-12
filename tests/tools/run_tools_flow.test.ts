/**
 * `runFlow` and `advance` give each run its own tools and ask handler.
 *
 * An AGENT's model is the test's `ChatModel` given as `models` (a fresh one per build); it calls
 * the first tool the request offers, then answers with the tool's result and the offered names as
 * JSON text. A host stop is the handler or tool aborting the run's signal and throwing its reason;
 * the run rejects with that reason. Two concurrent runs meet at a promise barrier. Every run sees
 * only the test's credential store in its environment.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/results.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { advance, pendingCalls, type RunFlowOptions, runFlow } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { type AskRequest, registerTool, TOOL_REGISTRY, type ToolEntry } from "../../src/tools/registry.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { barrier, NO_ENV, rejected, textReply } from "../helpers/run_state_fixtures.ts";
import { YianRangeError, YianTypeError } from "../../src/typesys/errors.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-run-tools-flow-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// Keep process-global registrations isolated between tests.
const registered = new Set(TOOL_REGISTRY.keys());
afterEach(() => {
  for (const id of TOOL_REGISTRY.keys()) if (!registered.has(id)) TOOL_REGISTRY.delete(id);
});

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** `runFlow` in the test's environment. */
const run = async (source: string, inputs: Record<string, unknown>, options: RunFlowOptions = {}) =>
  runFlow(await loadFlow(source), inputs, { env: hostEnv(dir), ...options });

const GREETING_FLOW = `
id: greeting
name: greeting
nodes:
  who:
    kind: tool
    tool_id: whoami
  ask:
    kind: tool
    tool_id: ask
    input:
      prompt: "Hello \${who.output}?"
output: \${ask.output}
`;

const GLOBAL_TOOL_FLOW = `
id: global_tool
name: global_tool
nodes:
  call:
    kind: tool
    tool_id: run_flow_global_probe
output: \${call.output}
`;

const LOOP_ASK_FLOW = `
id: loop_ask
name: loop_ask
defs:
  turn:
    input:
      count: !int
      last: !str
    nodes:
      ask:
        kind: tool
        tool_id: ask
        input:
          prompt: "next turn"
        output: !str
    output:
      count: \${input.count + 1}
      last: \${ask.output}
nodes:
  loop:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 2
    max: 5
output: \${loop.output}
`;

const MAP_ASK_FLOW = `
id: map_ask
name: map_ask
defs:
  one:
    input:
      item: !str
    nodes:
      ask:
        kind: tool
        tool_id: ask
        input:
          prompt: "item?"
        output: !str
    output: \${ask.output}
input:
  items: !list(str)
nodes:
  m:
    kind: map
    over: \${input.items}
    call: one
    input:
      item: \${item}
output: \${m.output}
`;

/** A tool `name` with no arguments returning `value()`. */
const fixed = (name: string, value: () => unknown) =>
  tool({ name, description: `Return ${name}.`, parameters: { type: "object", properties: {} }, execute: () => value() });

/** Echo the rendered question back as the answer. */
const answerWithQuestion = (request: AskRequest) => `answer to ${request.question}`;

test("runFlow uses per-run tools and ask_user handler", async () => {
  const result = await run(GREETING_FLOW, {}, {
    tools: { whoami: fixed("whoami", () => "alice"), ask: answerWithQuestion },
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("answer to Hello alice?");
  expect(TOOL_REGISTRY.has("whoami") || TOOL_REGISTRY.has("ask")).toBe(false);
});

test("a tools mapping hides the global registry", async () => {
  registerTool("run_flow_global_probe", fixed("run_flow_global_probe", () => "global"));
  const flow = await loadFlow(GLOBAL_TOOL_FLOW);

  expect((await runFlow(flow, {}, { env: hostEnv(dir) })).output).toBe("global");
  const error = await rejected(runFlow(flow, {}, { env: hostEnv(dir), tools: {} }));
  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toContain(
    "names tool 'run_flow_global_probe', which isn't in this run's tools; known tools: []",
  );
});

test("concurrent runs never see each other's tools or handler", async () => {
  const bothAsking = barrier(2);
  const runAs = (name: string) =>
    run(GREETING_FLOW, {}, {
      tools: {
        whoami: fixed("whoami", () => name),
        ask: async (request: AskRequest) => {
          await bothAsking();
          return `${name} answers ${request.question}`;
        },
      },
    });

  const [ann, bob] = await Promise.all([runAs("ann"), runAs("bob")]);

  expect(ann.output).toBe("ann answers Hello ann?");
  expect(bob.output).toBe("bob answers Hello bob?");
});

test("advance asks through the handler passed to resume", async () => {
  const path = freshLog();
  const flow = await loadFlow(GREETING_FLOW);
  const host = new AbortController();
  const stop = new Error("host stop");
  const interrupt = (_request: AskRequest) => {
    host.abort(stop);
    throw stop;
  };

  const error = await rejected(
    runFlow(flow, {}, {
      log: path,
      env: hostEnv(dir),
      signal: host.signal,
      tools: { whoami: fixed("whoami", () => "carol"), ask: interrupt },
    }),
  );
  expect(error).toBe(stop);

  const resumed = await advance({
    log: path,
    env: hostEnv(dir),
    tools: { whoami: fixed("whoami", () => "unused"), ask: answerWithQuestion },
  });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("answer to Hello carol?");
});

test("each map item asks with its own node id", async () => {
  const nodeIds: (string | null)[] = [];

  const result = await run(MAP_ASK_FLOW, { items: ["a", "b"] }, {
    tools: {
      ask: (request: AskRequest) => {
        nodeIds.push(request.nodeId);
        return "ok";
      },
    },
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(nodeIds.sort()).toEqual(["m#0/ask", "m#1/ask"]);
});

const whoami = fixed("whoami", () => "x");
for (const [name, tools, message] of [
  ["a list", [whoami], "tools must be a record of tool id to tool, or null"],
  [
    "a non-function ask_user",
    { ask_user: "not callable" },
    'tools["ask_user"] must be a handler function or {handler, inputs, required?}; yian supplies the base schema',
  ],
  ["an empty key", { "": whoami }, "tool ids in tools must be non-empty strings"],
  [
    "a bare function",
    { whoami: () => "x" },
    'tools["whoami"] must be a FunctionTool, an MCP server, or an array of FunctionTools, got function',
  ],
] as const) {
  test(`bad tools fail before the log is created: ${name}`, async () => {
    const path = freshLog();

    const error = await rejected(run(GREETING_FLOW, {}, { log: path, tools: tools as never }));

    expect(error).toEqual(new YianTypeError(message));
    expect(existsSync(path)).toBe(false);
  });
}

test("an ask without a run handler fails at run start", async () => {
  const path = freshLog();

  const error = await rejected(run(LOOP_ASK_FLOW, {}, { log: path }));

  expect(error).toEqual(
    new YianRangeError(
      "TOOL node 'loop/ask' names 'ask': no ask handler for this run; pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
    ),
  );
  expect(existsSync(path)).toBe(false);
});

// --- AGENT tool groups ------------------------------------------------------------------ //

const GROUPS_FLOW = `
id: groups
name: groups
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [toolbox, workspace, company]
output: \${agent.output}
`;

/** A model calling the first offered tool once, then answering with its result and the offered names. */
function firstToolChat(): ChatModel {
  return {
    provider: "openai",
    model: "test-model",
    stream(request: Request) {
      const names = request.tools.map((spec) => spec.name);
      const last = request.messages.at(-1)!.parts.at(-1);
      const reply =
        last instanceof ToolResultPart
          ? textReply(`${last.content} via ${JSON.stringify(names)}`)
          : new Response({
              message: new Message({
                role: "assistant",
                parts: [new ToolCallPart({ id: "c1", name: names[0]!, arguments: { text: "x" } })],
              }),
              finishReason: "tool_calls",
            });
      return new EventStream(
        (async function* (): AsyncGenerator<Event> {
          yield* new ScriptedModel([reply], { provider: "openai", model: "test-model" }).stream(request);
        })(),
      );
    },
  };
}

/** A fresh `firstToolChat` per AGENT call. */
const chats: ModelAccess = { env: NO_ENV, createModel: () => firstToolChat() };

/** A tool `name` taking `text`; by default it returns `name:text`. */
const textTool = (name: string, action?: (text: string) => unknown): FunctionTool =>
  tool({
    name,
    description: name,
    parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: (args) => (action === undefined ? `${name}:${args.text}` : action(args.text as string)),
  });

/** The three groups the flow grants, with `toolbox` first. */
const groups = (toolbox: FunctionTool[]): Record<string, ToolEntry> => ({
  toolbox,
  workspace: [textTool("ws")],
  company: [textTool("co")],
});

test("agent sees the union of three host-filled groups", async () => {
  const result = await run(GROUPS_FLOW, {}, { tools: groups([textTool("box")]), models: chats });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe('box:x via ["toolbox_box","workspace_ws","company_co"]');
});

test("concurrent runs with different toolboxes never cross", async () => {
  const bothCalling = barrier(2);
  const runAs = (name: string) =>
    run(GROUPS_FLOW, {}, {
      models: chats,
      tools: groups([
        textTool("lookup", async (text) => {
          await bothCalling();
          return `${name} looked up ${text}`;
        }),
      ]),
    });

  const [ann, bob] = await Promise.all([runAs("ann"), runAs("bob")]);

  expect(ann.output).toBe('ann looked up x via ["toolbox_lookup","workspace_ws","company_co"]');
  expect(bob.output).toBe('bob looked up x via ["toolbox_lookup","workspace_ws","company_co"]');
});

for (const [name, tools, message] of [
  [
    "missing group",
    { toolbox: [], workspace: [] },
    "AGENT node 'agent' grants unknown tool or group 'company'; tool groups in this run: [toolbox, workspace]",
  ],
  ["group ask_user", groups([textTool("ask_user")]), "group 'toolbox' has a tool named 'ask_user'"],
] as const) {
  test(`bad groups fail at run start: ${name}`, async () => {
    const path = freshLog();

    const error = await rejected(run(GROUPS_FLOW, {}, { log: path, tools, models: chats }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain(message);
    expect(existsSync(path)).toBe(false);
  });
}

test("an ambiguous bare grant fails at run start with the group form as its fix", async () => {
  const path = freshLog();
  const flow = GROUPS_FLOW.replace("tools: [toolbox, workspace, company]", "tools: [search]");
  const tools = { github: [textTool("search")], docs: [textTool("search")] };

  const error = await rejected(run(flow, {}, { log: path, tools, models: chats }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(
    "AGENT node 'agent' grants 'search': 'search' is a tool of groups 'github' and 'docs'; grant one as `{github: [search]}`",
  );
  expect(existsSync(path)).toBe(false);
});

test("a group whose tools would show under a name a model can't call fails at run start", async () => {
  const path = freshLog();
  const flow = GROUPS_FLOW.replace("tools: [toolbox, workspace, company]", "tools: [files.v2]");

  const error = await rejected(run(flow, {}, { log: path, tools: { "files.v2": [textTool("read")] }, models: chats }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toStartWith(
    "AGENT node 'agent' grants group `files.v2`'s `read`, which would show as `files.v2_read`; ",
  );
  expect((error as Error).message).toEndWith("; rename the group");
  expect(existsSync(path)).toBe(false);
});

test("a group grant naming a tool the group lacks fails at run start", async () => {
  const path = freshLog();
  const flow = GROUPS_FLOW.replace("tools: [toolbox, workspace, company]", "tools: [{github: [upsert]}]");

  const error = await rejected(run(flow, {}, { log: path, tools: { github: [textTool("search")] }, models: chats }));

  expect(error).toBeInstanceOf(RangeError);
  expect((error as Error).message).toBe(
    "AGENT node 'agent' grants {github: [upsert]}: `github` has no tool `upsert`; it has search",
  );
  expect(existsSync(path)).toBe(false);
});

test("a TOOL node's {group: tool} reaches the group's tool, and its pending step names github_search", async () => {
  const log = freshLog();
  const flow = `
id: group_tool
name: group_tool
nodes:
  found:
    kind: tool
    tool_id: {github: search}
    input: {text: owls}
output: \${found.output}
`;
  const held = tool({
    name: "search",
    description: "search",
    parameters: { type: "object", properties: { text: { type: "string" } }, required: ["text"] },
    execute: () => pending(),
  });
  const tools = { github: [held], docs: [textTool("search")] };

  const result = await run(flow, {}, { log, tools });

  expect(result.status, result.failure?.message).toBe("waiting");
  expect(pendingCalls(log).map(({ node, tool: name }) => [node, name])).toEqual([["found", "github_search"]]);
});

test.each([
  ["github_search", "`{github: search}`"],
  ["github_serch", "`{github: search}`"],
  ["lookp", "'lookup'"],
])("a TOOL node's unknown tool_id %s suggests the form to write: %s", async (toolId, suggestion) => {
  const flow = `id: t\nname: t\nnodes:\n  found:\n    kind: tool\n    tool_id: ${toolId}\n    input: {text: owls}\noutput: \${found.output}\n`;
  const tools = { lookup: textTool("lookup"), github: [textTool("search")] };

  const error = await rejected(run(flow, {}, { log: freshLog(), tools }));

  expect((error as Error).message).toBe(
    `TOOL node 'found' names tool '${toolId}', which isn't in this run's tools; ` +
      `known tools: ["lookup","{github: search}"]; did you mean ${suggestion}?`,
  );
});

test("a TOOL node naming a group offers one of its tools as {group: tool}", async () => {
  const flow = `id: t\nname: t\nnodes:\n  found:\n    kind: tool\n    tool_id: github\n    input: {text: owls}\noutput: \${found.output}\n`;

  const error = await rejected(run(flow, {}, { log: freshLog(), tools: { github: [textTool("search")] } }));

  expect((error as Error).message).toBe(
    "TOOL node 'found' names 'github': 'github' is a tool group, not a tool; name one of its tools, as `{github: search}`",
  );
});

test("an AGENT granting a group tool's visible name is offered the group form", async () => {
  const flow = GROUPS_FLOW.replace("tools: [toolbox, workspace, company]", "tools: [github_search]");

  const error = await rejected(run(flow, {}, { log: freshLog(), tools: { github: [textTool("search")] }, models: chats }));

  expect((error as Error).message).toBe(
    "AGENT node 'agent' grants unknown tool or group 'github_search'; tool groups in this run: [github]; " +
      "did you mean `{github: [search]}`?",
  );
});

test("a TOOL node naming a group the run lacks fails at run start", async () => {
  const flow = `
id: group_tool
name: group_tool
nodes:
  found:
    kind: tool
    tool_id: {github: search}
    input: {text: owls}
output: \${found.output}
`;

  const error = await rejected(run(flow, {}, { log: freshLog(), tools: { docs: [textTool("search")] } }));

  expect((error as Error).message).toBe(
    "TOOL node 'found' names {github: search}: unknown tool group 'github'; tool groups in this run: [docs]",
  );
});

test("a resume whose group holds its tools in another order is refused", async () => {
  const path = freshLog();
  const flow = GROUPS_FLOW.replace("tools: [toolbox, workspace, company]", "tools: [toolbox]");
  const host = new AbortController();
  const stop = new Error("host stop");
  const interrupt = () => {
    host.abort(stop);
    throw stop;
  };
  const first = textTool("first", interrupt);
  const second = textTool("second");
  const started = run(flow, {}, { log: path, signal: host.signal, models: chats, tools: { toolbox: [first, second] } });
  expect(await rejected(started)).toBe(stop);

  const refused = await rejected(
    advance({ log: path, env: hostEnv(dir), models: chats, tools: { toolbox: [second, first] } }),
  );

  expect((refused as Error).message).toBe("resume refused: tool_groups.toolbox differs from the recorded one");
});

test("resume uses the group contents passed to resume", async () => {
  const path = freshLog();
  const flow = await loadFlow(GROUPS_FLOW);
  const host = new AbortController();
  const stop = new Error("host stop");
  const interrupt = () => {
    host.abort(stop);
    throw stop;
  };

  const error = await rejected(
    runFlow(flow, {}, { log: path, env: hostEnv(dir), signal: host.signal, models: chats, tools: groups([textTool("old", interrupt)]) }),
  );
  expect(error).toBe(stop);

  const refused = await rejected(
    advance({ log: path, env: hostEnv(dir), models: chats, tools: { workspace: [], company: [] } }),
  );
  expect(refused).toBeInstanceOf(RangeError);
  expect((refused as Error).message).toContain("unknown tool or group 'toolbox'");
  // The recorded tool specs are the run's, so a group holding another tool is refused too.
  const other = await rejected(
    advance({ log: path, env: hostEnv(dir), models: chats, tools: groups([textTool("new")]) }),
  );
  expect(other).toBeInstanceOf(RangeError);
  expect((other as Error).message).toBe("resume refused: tools.toolbox_new isn't recorded; the run started without it");
  // The same tools in other groups are refused: a tool's record name names its group.
  const moved = await rejected(
    advance({
      log: path,
      env: hostEnv(dir),
      models: chats,
      tools: { toolbox: [textTool("old"), textTool("ws")], workspace: [], company: [textTool("co")] },
    }),
  );
  expect(moved).toBeInstanceOf(RangeError);
  expect((moved as Error).message).toBe("resume refused: tools.toolbox_ws isn't recorded; the run started without it");
  const resumed = await advance({ log: path, env: hostEnv(dir), models: chats, tools: groups([textTool("old")]) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // The stop cut the call off before its result, so the resume closes it, outcome unknown, rather
  // than running it again.
  expect(resumed.output).toBe(`${TOOL_OUTCOME_UNKNOWN} via ["toolbox_old","workspace_ws","company_co"]`);
});

const NESTED_GROUPS_FLOW = `
id: nested_groups
name: nested_groups
llm_config: {provider: openai, model: test-model}
defs:
  inner:
    nodes:
      agent:
        kind: agent
        prompt: go
        tools: [toolbox]
    output: \${agent.output}
nodes:
  outer:
    kind: call
    call: inner
output: \${outer.output}
`;

const LOOP_GROUPS_FLOW = `
id: loop_groups
name: loop_groups
llm_config: {provider: openai, model: test-model}
defs:
  turn:
    input:
      count: !int
      last: !str
    nodes:
      agent:
        kind: agent
        prompt: go
        tools: [toolbox]
    output:
      count: \${input.count + 1}
      last: \${agent.output}
nodes:
  again:
    kind: loop
    call: turn
    input:
      count: 0
      last: ""
    while: count < 1
    max: 3
output: \${again.output}
`;

for (const [name, source, nodePath] of [
  ["call", NESTED_GROUPS_FLOW, "outer/agent"],
  ["loop", LOOP_GROUPS_FLOW, "again/agent"],
] as const) {
  test(`a missing group in a child flow fails at run start: ${name}`, async () => {
    const path = freshLog();

    const error = await rejected(run(source, {}, { log: path, tools: {}, models: chats }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain(`AGENT node '${nodePath}' grants unknown tool or group 'toolbox'`);
    expect(existsSync(path)).toBe(false);
  });
}

const PLAIN_GRANT_FLOW = `
id: plain_grant
name: plain_grant
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [lookup, ask_user]
output: \${agent.output}
`;

for (const [name, tools, message] of [
  ["unknown plain tool", { ask_user: answerWithQuestion }, "AGENT node 'agent' grants unknown tool or group 'lookup'"],
  [
    "ask_user without a handler",
    { lookup: textTool("lookup") },
    "AGENT node 'agent' grants 'ask_user': no ask handler for this run",
  ],
] as const) {
  test(`an unresolvable plain grant fails at run start: ${name}`, async () => {
    const path = freshLog();

    const error = await rejected(run(PLAIN_GRANT_FLOW, {}, { log: path, tools, models: chats }));

    expect(error).toBeInstanceOf(RangeError);
    expect((error as Error).message).toContain(message);
    expect(existsSync(path)).toBe(false);
  });
}

/** A flow whose one AGENT grants `tool`. */
const grantFlow = (tool: string) => `
id: grant_builtin
name: grant_builtin
llm_config: {provider: openai, model: test-model}
nodes:
  agent:
    kind: agent
    prompt: go
    tools: [${tool}]
output: \${agent.output}
`;

for (const [tool, message] of [
  ["take", "AGENT node 'agent' grants 'take': an engine built-in a TOOL node reaches with tool_id: take"],
  ["sleep", "AGENT node 'agent' grants 'sleep': an engine built-in a TOOL node reaches with tool_id: sleep"],
  ["ask", "AGENT node 'agent' grants 'ask': grant ask_user; ask is a TOOL-node built-in in this version"],
] as const) {
  test(`an AGENT can't grant the built-in ${tool}`, async () => {
    const path = freshLog();

    const error = await rejected(run(grantFlow(tool), {}, { log: path, tools: { ask: answerWithQuestion }, models: chats }));

    expect(error).toEqual(new YianRangeError(message));
    expect(existsSync(path)).toBe(false);
  });
}
