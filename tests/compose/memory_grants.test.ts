/**
 * Memory grants: `tools: [lessons]`, `tools: [{lessons: [read]}]` and `tool_id: {lessons: read}`
 * name a memory input of the node's own flow, are checked against its type's tools at load, and
 * are routed through the reserved wiring `__memory_<input>`.
 */
import { afterEach, beforeEach, expect, test } from "bun:test";

import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow, loadFlowSource } from "../../src/compose/loader.ts";
import { flowFingerprint } from "../../src/compile/manifest.ts";
import { BUILTIN, forget, install } from "../../src/defaults.ts";
import { MemoryBase, type MemoryContract, type MemorySession } from "../../src/memory/base.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Binding } from "../../src/nodes/contracts.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { runFlow } from "../../src/run.ts";
import { VariablePool } from "../../src/typesys/pool.ts";
import { MemoryRef, TypedValue, ValueKind } from "../../src/typesys/values.ts";
import { loadErrorAsync } from "../helpers/declarations.ts";

beforeEach(() => install(BUILTIN));
afterEach(forget);

/** A host memory class with the contract `type` and read tools `tools`. */
function hostType(type: string, tools: readonly string[]): typeof MemoryBase {
  const contract: MemoryContract = {
    type,
    tools: tools.map((name) => ({ name, write: false, description: name, parameters: { type: "object" } })),
  };
  return class extends MemoryBase {
    static override readonly contract = contract;
    override open(): Promise<MemorySession> {
      throw new Error("not opened at load");
    }
    override async reset(): Promise<void> {}
  };
}

const TRADE_JOURNAL = hostType("TradeJournal", ["record_trade"]);

/** A flow with inputs `inputs` (YAML) whose nodes are `nodes` (YAML under `nodes:`). */
const flow = (inputs: string, nodes: string, extra = "") =>
  `id: f\nname: f\n${extra}input:\n${inputs.replace(/^/gm, "  ")}\nnodes:\n${nodes.replace(/^/gm, "  ")}\noutput: null\n`;
const agent = (tools: string, extra = "") =>
  `a:\n  kind: agent\n  output: str\n  prompt: hi\n  tools: ${tools}\n${extra}`;
const nodeOf = <T>(loaded: { compiled: { nodes: ReadonlyMap<string, unknown> } }, id: string) =>
  loaded.compiled.nodes.get(id) as T;

test("a whole memory grant splits off the host grants and is wired to the input", async () => {
  const loaded = await loadFlow(flow("lessons: !FileMemory", agent("[lessons, search, {github: [create_issue]}]")));
  const node = nodeOf<AgentNode>(loaded, "a");
  expect(node.memoryGrants()).toEqual([{ input: "lessons", tools: null }]);
  expect(node.grantedTools()).toEqual([
    { kind: "name", name: "search" },
    { kind: "group", group: "github", tools: ["create_issue"] },
  ]);
  expect(node.reservedWiringKeys()).toEqual(new Set(["__memory_lessons"]));
  const wired = loaded.compiled.wiring.get("a")!.get("__memory_lessons");
  expect(wired instanceof Binding ? wired.source : wired).toBe("${input.lessons}");
  expect(node.dataInputNames()).toEqual([]);
});

test("a group grant lists some of a memory's tools", async () => {
  const loaded = await loadFlow(flow("lessons: !FileMemory", agent("[{lessons: [read, search]}]")));
  expect(nodeOf<AgentNode>(loaded, "a").memoryGrants()).toEqual([{ input: "lessons", tools: ["read", "search"] }]);
});

test("an unknown memory tool is refused with the type's tools", async () => {
  const error = await loadErrorAsync(loadFlow(flow("lessons: !FileMemory", agent("[{lessons: [upsert]}]"))));
  expect(error.diagnostic.message).toContain(
    "`lessons` has no tool `upsert`; it has list, read, search, append, create, edit, delete, rewrite",
  );
});

test("a bare tool name never reaches a memory: it stays a host grant", async () => {
  const node = nodeOf<AgentNode>(await loadFlow(flow("lessons: !FileMemory", agent("[read]"))), "a");
  expect(node.memoryGrants()).toEqual([]);
  expect(node.grantedTools()).toEqual([{ kind: "name", name: "read" }]);
});

test("a TOOL node names a memory's tool by its group form", async () => {
  const nodes = "found:\n  kind: tool\n  tool_id: {lessons: read}\n  input:\n    path: notes.md\n";
  const loaded = await loadFlow(flow("lessons: !FileMemory", nodes));
  const node = nodeOf<ToolNode>(loaded, "found");
  expect(node.memoryGrants()).toEqual([{ input: "lessons", tools: ["read"] }]);
  expect(node.hostToolCall()).toBeNull();
  expect(node.reservedWiringKeys()).toEqual(new Set(["__memory_lessons"]));
  expect(node.dataInputNames()).toEqual(["path"]);
});

test("a TOOL node's memory tool is checked against the type's tools", async () => {
  const nodes = "found:\n  kind: tool\n  tool_id: {lessons: upsert}\n";
  const error = await loadErrorAsync(loadFlow(flow("lessons: !FileMemory", nodes)));
  expect(error.diagnostic.message).toContain("`lessons` has no tool `upsert`; it has list, read");
});

test("a TOOL node can't name an Optional memory", async () => {
  const nodes = "found:\n  kind: tool\n  tool_id: {notes: read}\n";
  const error = await loadErrorAsync(loadFlow(flow("notes: !Optional(FileMemory)", nodes)));
  expect(error.diagnostic.message).toContain(
    "`tool_id: {notes: read}` names `notes`, an Optional memory; a TOOL node needs one that is always bound",
  );
});

test.each([
  ["an agent input", agent("[lessons]", "  input:\n    __memory_lessons: ${input.lessons}\n")],
  [
    "a TOOL node input",
    "found:\n  kind: tool\n  tool_id: {lessons: read}\n  input:\n    __memory_x: 1\n",
  ],
])("%s named like a reserved memory key is refused", async (_, nodes) => {
  const error = await loadErrorAsync(loadFlow(flow("lessons: !FileMemory", nodes)));
  expect(error.diagnostic.message).toMatch(/input '__memory_\w+' is reserved for memory grants; rename the input/);
});

test("an agent variable named like a reserved memory key is refused", async () => {
  const text = "id: f\nname: f\nkind: agent\nvariables:\n  __memory_x: !int 1\nprompt: hi\noutput: str\n";
  const error = await loadErrorAsync(loadFlow(text));
  expect(error.diagnostic.message).toContain("variable '__memory_x' is reserved for memory grants; rename the variable");
});

test("a def grants its own renamed memory input", async () => {
  const text = `id: f
name: f
input:
  lessons: !FileMemory
defs:
  child:
    input:
      notes: !FileMemory
    nodes:
      a:
        kind: agent
        output: str
        prompt: hi
        tools: [notes]
    output: \${a.output}
nodes:
  c:
    kind: call
    call: child
    input:
      notes: \${input.lessons}
output: \${c.output}
`;
  const loaded = await loadFlow(text);
  const call = loaded.compiled.nodes.get("c") as unknown as { child: { nodes: ReadonlyMap<string, unknown>; wiring: ReadonlyMap<string, ReadonlyMap<string, unknown>> } };
  const inner = call.child.nodes.get("a") as AgentNode;
  expect(inner.memoryGrants()).toEqual([{ input: "notes", tools: null }]);
  const wired = call.child.wiring.get("a")!.get("__memory_notes");
  expect(wired instanceof Binding ? wired.source : wired).toBe("${input.notes}");
});

test("an inline loop body grants its own input, and the parent's name isn't a memory there", async () => {
  const body = (tools: string) => `id: f
name: f
input:
  lessons: !FileMemory
nodes:
  l:
    kind: loop
    input:
      m: \${input.lessons}
      done: false
    nodes:
      a:
        kind: agent
        output: bool
        prompt: hi
        tools: ${tools}
    output:
      done: \${a.output}
    while: not done
output: null
`;
  const own = await loadFlow(body("[{m: [read]}]"));
  const agents = [...own.compiled.nodes.values()].flatMap((node) => {
    const child = (node as { child?: { nodes: ReadonlyMap<string, unknown> } | null }).child;
    return child === undefined || child === null ? [] : [...child.nodes.values()];
  });
  const inner = agents.find((node) => node instanceof AgentNode) as AgentNode;
  expect(inner.memoryGrants()).toEqual([{ input: "m", tools: ["read"] }]);
  const parents = await loadFlow(body("[lessons]"));
  const other = [...parents.compiled.nodes.values()]
    .flatMap((node) => [...((node as { child?: { nodes: ReadonlyMap<string, unknown> } | null }).child?.nodes.values() ?? [])])
    .find((node) => node instanceof AgentNode) as AgentNode;
  expect(other.memoryGrants()).toEqual([]);
  expect(other.grantedTools()).toEqual([{ kind: "name", name: "lessons" }]);
});

test("an inline body agent granting its parent's memory input fails at run start, saying what grants name", async () => {
  const text = `id: f
name: f
input:
  lessons: !Optional(FileMemory)
nodes:
  l:
    kind: loop
    input:
      done: false
    nodes:
      a:
        kind: agent
        output: bool
        prompt: hi
        tools: [lessons]
        llm_config: {provider: openai, model: gpt-test}
    output:
      done: \${a.output}
    while: not done
output: null
`;
  const loaded = await loadFlow(text);
  const log = join(mkdtempSync(join(tmpdir(), "yian-grants-")), "log.jsonl");
  await expect(runFlow(loaded, {}, { env: () => undefined, log })).rejects.toThrow(
    "grants unknown tool or group 'lessons'",
  );
  await expect(runFlow(loaded, {}, { env: () => undefined, log: `${log}.2` })).rejects.toThrow(
    "; memory grants name the node's own flow inputs",
  );
});

test.each([
  [hostType("Notes", ["skill"]), "load", "the tool name `load_skill` of `{load: [skill]}` is yian's; rename the input"],
  [hostType("Notes", ["user"]), "ask", "the tool name `ask_user` of `{ask: [user]}` is yian's; rename the input"],
])("a memory tool named like a built-in is refused", async (cls, input, message) => {
  const error = await loadErrorAsync(
    loadFlow(flow(`${input}: !Notes`, agent(`[{${input}: [${cls.contract.tools[0]!.name}]}]`)), { memoryTypes: [cls] }),
  );
  expect(error.diagnostic.message).toContain(message);
});

test.each(["[a_b, a]", "[a, a_b]", "[{a_b: [read]}, {a: [b_read]}]"])(
  "two memory tools that would show one name are refused (%s)",
  async (tools) => {
    const TA = hostType("TA", ["b_read"]);
    const error = await loadErrorAsync(loadFlow(flow("a_b: !FileMemory\na: !TA", agent(tools)), { memoryTypes: [TA] }));
    expect(error.diagnostic.message).toMatch(/tools 'a_b_read' \((a_b's read|a's b_read)\) and 'a_b_read' \((a_b's read|a's b_read)\) have one name; grant one/);
  },
);

test("a memory tool whose name is past 64 characters is refused", async () => {
  const long = "l".repeat(60);
  for (const tools of [`[${long}]`, `[{${long}: [rewrite]}]`]) {
    const error = await loadErrorAsync(loadFlow(flow(`${long}: !FileMemory`, agent(tools))));
    expect(error.diagnostic.message).toContain(
      `, which would show as \`${long}_`,
    );
    expect(error.diagnostic.message).toContain("shorten the group or the tool");
  }
});

test("a host memory tool a model can't call by name is refused", async () => {
  const BadName = hostType("BadName", ["bad.name"]);
  const error = await loadErrorAsync(loadFlow(flow("j: !BadName", agent("[j]")), { memoryTypes: [BadName] }));
  expect(error.diagnostic.message).toContain("grants group `j`'s `bad.name`, which would show as `j_bad.name`");
  expect(error.diagnostic.message).toContain("rename the tool");
});

test("a TOOL node's memory tool named like a built-in says to rename the input", async () => {
  const Notes = hostType("Notes", ["skill"]);
  const nodes = "found:\n  kind: tool\n  tool_id: {load: skill}\n";
  const error = await loadErrorAsync(loadFlow(flow("load: !Notes", nodes), { memoryTypes: [Notes] }));
  expect(error.diagnostic.message).toBe(
    "node 'found' (kind=tool): the tool name `load_skill` of `{load: [skill]}` is yian's; rename the input",
  );
});

test("a TOOL node's host group tool named like a built-in still says to rename the group or the tool", async () => {
  const error = await loadErrorAsync(loadFlow(flow("q: !str", "found:\n  kind: tool\n  tool_id: {load: skill}\n")));
  expect(error.diagnostic.message).toContain("would show as `load_skill`, a name yian reserves; rename the group or the tool");
});

test("a host type's grants are checked against its class's tools", async () => {
  const text = flow("j: !TradeJournal", agent("[{j: [record_trade]}]"));
  const loaded = await loadFlow(text, { memoryTypes: [TRADE_JOURNAL] });
  expect(nodeOf<AgentNode>(loaded, "a").memoryGrants()).toEqual([{ input: "j", tools: ["record_trade"] }]);
  const error = await loadErrorAsync(loadFlow(flow("j: !TradeJournal", agent("[{j: [erase]}]")), { memoryTypes: [TRADE_JOURNAL] }));
  expect(error.diagnostic.message).toContain("`j` has no tool `erase`; it has record_trade");
});

test("a recorded flow compiles without memoryTypes, skipping the opaque type's tool check", async () => {
  const loaded = await loadFlow(flow("j: !TradeJournal", agent("[{j: [record_trade]}]")), { memoryTypes: [TRADE_JOURNAL] });
  const again = await loadFlowSource(loaded.source!);
  expect(nodeOf<AgentNode>(again, "a").memoryGrants()).toEqual([{ input: "j", tools: ["record_trade"] }]);
  expect(flowFingerprint(again.compiled)).toBe(flowFingerprint(loaded.compiled));
  await expect(loadFlow(flow("j: !TradeJournal", agent("[{j: [record_trade]}]")))).rejects.toThrow("TradeJournal");
});

test("a memory grant changes the fingerprint; the same grant written twice doesn't", async () => {
  const one = await loadFlow(flow("lessons: !FileMemory", agent("[lessons]")));
  const same = await loadFlow(flow("lessons: !FileMemory", agent("[lessons]")));
  const some = await loadFlow(flow("lessons: !FileMemory", agent("[{lessons: [read]}]")));
  const none = await loadFlow(flow("lessons: !FileMemory", agent("[]")));
  expect(flowFingerprint(one.compiled)).toBe(flowFingerprint(same.compiled));
  expect(flowFingerprint(one.compiled)).not.toBe(flowFingerprint(some.compiled));
  expect(flowFingerprint(one.compiled)).not.toBe(flowFingerprint(none.compiled));
});

/** A recorded memory value for identity `identity`. */
const memory = (identity: string) =>
  new MemoryRef({
    identity,
    type: "FileMemory",
    settings: {},
    version: "v1",
    recall: { text: "notes", cut: null, hint: null },
    tools: [{ name: "read", write: false, description: "Read a file.", parameters: { type: "object" } }],
  });

test("bindReserved binds each granted memory, and null for an Optional one left out", async () => {
  const loaded = await loadFlow(flow("lessons: !FileMemory\nnotes: !Optional(FileMemory)", agent("[lessons, {notes: [read]}]")));
  const node = nodeOf<AgentNode>(loaded, "a");
  const wiring = loaded.compiled.wiring.get("a")!;
  const pool = new VariablePool();
  const lessons = memory("t/a");
  pool.store.set("__start__", new TypedValue(ValueKind.OBJECT, { lessons }));
  const bound = node.bindReserved(wiring, pool);
  expect(bound["__memory_lessons"]).toBe(lessons);
  expect(bound["__memory_notes"]).toBeNull();
});

test("bindReserved refuses a value that isn't a memory", async () => {
  const loaded = await loadFlow(flow("lessons: !FileMemory", agent("[lessons]")));
  const pool = new VariablePool();
  pool.store.set("__start__", new TypedValue(ValueKind.OBJECT, { lessons: "notes.md" }));
  expect(() => nodeOf<AgentNode>(loaded, "a").bindReserved(loaded.compiled.wiring.get("a")!, pool)).toThrow(
    "`lessons` in `tools:` must be a memory",
  );
});
