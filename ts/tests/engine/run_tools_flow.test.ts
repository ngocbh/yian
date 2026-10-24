/**
 * `runFlow` and `advance` give each run its own tools and `ask_user` handler: a port of
 * `tests/engine/test_run_tools_flow.py`.
 *
 * An AGENT's model is the test's `ChatModel` given as `models` (a fresh one per build, as
 * Python's fixture patches `model_from_config`); it calls the first tool the request offers, then
 * answers with the tool's result and the offered names, the names as JSON text where Python
 * prints a list repr. A host stop is the handler or tool aborting the run's signal and throwing
 * its reason, where Python raises `_HostStop`; the run rejects with that reason. Two concurrent
 * runs meet at a promise barrier, Python's `threading.Barrier`. Every run sees only the test's
 * credential store in its environment.
 *
 * Ported: `test_run_flow_uses_per_run_tools_and_ask_user_handler`,
 * `test_a_tools_mapping_hides_the_global_registry`,
 * `test_concurrent_runs_never_see_each_others_tools_or_handler`,
 * `test_resume_flow_asks_through_the_handler_passed_to_resume`, the `map` row of
 * `test_each_map_item_and_call_asks_with_its_own_node_id`,
 * `test_human_input_without_a_run_handler_fails_the_node`,
 * `test_agent_sees_the_union_of_three_host_filled_groups`,
 * `test_concurrent_runs_with_different_toolboxes_never_cross`, `test_bad_groups_fail_at_run_start`
 * (all three rows), `test_resume_uses_the_group_contents_passed_to_resume`, the `call` and `loop`
 * rows of `test_a_missing_group_in_a_child_flow_fails_at_run_start` and
 * `test_an_unresolvable_plain_grant_fails_at_run_start` (both rows). A run-start refusal is a
 * `RangeError` where Python raises `ValueError`, and the missing-group message lists the run's
 * groups as JSON text (`["toolbox","workspace"]`, decision 7).
 *
 * Changed: `test_resume_uses_the_group_contents_passed_to_resume`: the run records the specs of
 * the tools its AGENTs grant, so a resume whose group holds another tool is refused, and the
 * resume uses a tool of the same spec that acts differently (Python resumes with another tool).
 * `test_bad_tools_fail_before_the_log_is_created`: TS tools are `FunctionTool`s, tool
 * groups (arrays of them) and the `ask_user` handler (a function), so a list, a non-function
 * `ask_user` entry and an empty key are each refused with a `TypeError` before the log exists, as
 * in Python, and a bare function under any other key is refused too.
 *
 * Covered by `run-leaves/run_tools_flow__each_loop_occurrence_asks_with_its_own_node_id`:
 * `test_each_loop_occurrence_asks_with_its_own_node_id`; by
 * `run-leaves/run_tools_flow__each_call_asks_with_its_own_node_id`: the `call` row of
 * `test_each_map_item_and_call_asks_with_its_own_node_id`; by
 * `run-leaves/engine_run_tools_flow__a_missing_group_in_a_mapped_child_fails_at_run_start`: the
 * `map` row of `test_a_missing_group_in_a_child_flow_fails_at_run_start`.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { type ChatModel, type Event, EventStream } from "../../src/llm/events.ts";
import { Message, type Request, Response, ToolCallPart, ToolResultPart } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/memory.ts";
import type { ModelAccess } from "../../src/llm_clients/factory.ts";
import { advance, type RunFlowOptions, runFlow } from "../../src/run.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import { type AskUserRequest, registerTool, TOOL_REGISTRY, type ToolEntry } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { barrier, NO_ENV, rejected, textReply } from "./run_state_fixtures.ts";

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
    kind: human_input
    input: {name: "\${who.output}"}
    prompt: "Hello \${name}?"
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
        kind: human_input
        prompt: "next turn"
        output: str
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
        kind: human_input
        prompt: "item?"
        output: str
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
const answerWithQuestion = (request: AskUserRequest) => `answer to ${request.question}`;

test("runFlow uses per-run tools and ask_user handler", async () => {
  const result = await run(GREETING_FLOW, {}, {
    tools: { whoami: fixed("whoami", () => "alice"), ask_user: answerWithQuestion },
  });

  expect(result.status, result.failure?.message).toBe("succeeded");
  expect(result.output).toBe("answer to Hello alice?");
  expect(TOOL_REGISTRY.has("whoami") || TOOL_REGISTRY.has("ask_user")).toBe(false);
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
        ask_user: async (request: AskUserRequest) => {
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
  const interrupt = (_request: AskUserRequest) => {
    host.abort(stop);
    throw stop;
  };

  const error = await rejected(
    runFlow(flow, {}, {
      log: path,
      env: hostEnv(dir),
      signal: host.signal,
      tools: { whoami: fixed("whoami", () => "carol"), ask_user: interrupt },
    }),
  );
  expect(error).toBe(stop);

  const resumed = await advance({
    log: path,
    env: hostEnv(dir),
    tools: { whoami: fixed("whoami", () => "unused"), ask_user: answerWithQuestion },
  });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  expect(resumed.output).toBe("answer to Hello carol?");
});

test("each map item asks with its own node id", async () => {
  const nodeIds: (string | null)[] = [];

  const result = await run(MAP_ASK_FLOW, { items: ["a", "b"] }, {
    tools: {
      ask_user: (request: AskUserRequest) => {
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

    expect(error).toEqual(new TypeError(message));
    expect(existsSync(path)).toBe(false);
  });
}

test("human input without a run handler fails the node", async () => {
  const result = await run(LOOP_ASK_FLOW, {});

  expect(result.status).toBe("failed");
  expect(result.failure!.message).toContain("no ask handler for this run");
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
  expect(result.output).toBe('box:x via ["box","ws","co"]');
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

  expect(ann.output).toBe('ann looked up x via ["lookup","ws","co"]');
  expect(bob.output).toBe('bob looked up x via ["lookup","ws","co"]');
});

for (const [name, tools, message] of [
  [
    "missing group",
    { toolbox: [], workspace: [] },
    "AGENT node 'agent' grants unknown tool or group 'company'; tool groups in this run: [\"toolbox\",\"workspace\"]",
  ],
  ["name clash", groups([textTool("ws")]), "tool 'ws' is in both groups 'toolbox' and 'workspace'"],
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
  expect((other as Error).message).toBe("resume refused: tools.new isn't recorded; the run started without it");
  // The same tools in other groups are refused: the recorded membership is what learning expands.
  const moved = await rejected(
    advance({
      log: path,
      env: hostEnv(dir),
      models: chats,
      tools: { toolbox: [textTool("old"), textTool("ws")], workspace: [], company: [textTool("co")] },
    }),
  );
  expect(moved).toBeInstanceOf(RangeError);
  expect((moved as Error).message).toBe("resume refused: tool_groups.toolbox differs from the recorded one");
  const resumed = await advance({ log: path, env: hostEnv(dir), models: chats, tools: groups([textTool("old")]) });

  expect(resumed.status, resumed.failure?.message).toBe("succeeded");
  // The stop cut the call off before its result, so the resume closes it, outcome unknown, rather
  // than running it again.
  expect(resumed.output).toBe(`${TOOL_OUTCOME_UNKNOWN} via ["old","ws","co"]`);
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

    expect(error).toEqual(new RangeError(message));
    expect(existsSync(path)).toBe(false);
  });
}
