/**
 * A TOOL node calls a memory's tool by its group form, `tool_id: {lessons: read}`: the call goes
 * to the run's session on the memory bound to the input, with the node's call id as a change's id,
 * and a refusal or a reused id fails the node unretried while a busy memory is retried.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { ChangeIdReusedError, MemoryBusyError } from "../../src/memory/base.ts";
import { FileMemory } from "../../src/memory/file.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { runFlow } from "../../src/run.ts";
import { ToolNode } from "../../src/nodes/tool/node.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { ToolSet } from "../../src/tools/registry.ts";
import { FakeMemory, fileInfo, fileRef, RecordingSession } from "../helpers/memory_fakes.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";

const tmp = mkdtempSync(join(tmpdir(), "yian-tool-memory-"));
afterAll(() => rmSync(tmp, { recursive: true, force: true }));

/** A flow whose TOOL node `found` calls `tool` of `lessons` with `input` (YAML) and `retries`. */
function flow(tool: string, input: string, retries = 0): string {
  return `id: f
name: f
input:
  lessons: !FileMemory
nodes:
  found:
    kind: tool
    tool_id: {lessons: ${tool}}
    retries: ${retries}
    input: ${input}
output: \${found.output}
`;
}

/** Run `text` with `lessons` bound to `memory` as run `r`. */
async function run(text: string, memory: unknown) {
  const log = join(mkdtempSync(join(tmp, "run-")), "log.jsonl");
  const result = await runFlow(await loadFlow(text), { lessons: memory }, {
    env: NO_ENV,
    log,
    runId: "r",
    scheduler: virtualScheduler(),
    random: () => 0,
  });
  return { result, log };
}

test("tool_id: {lessons: read} returns the read result as ${found.output}", async () => {
  const root = mkdtempSync(join(tmp, "m-"));
  writeFileSync(join(root, "a.md"), "first line\nsecond line");
  const { result } = await run(flow("read", "{path: a.md}"), new FileMemory({ root, id: "t/lessons" }));
  expect(result.status).toBe("succeeded");
  expect((result.output as Record<string, unknown>)["text"]).toContain("second line");
  expect((result.output as Record<string, unknown>)["path"]).toBe("a.md");
});

test("an append lands once, with the change id <run>/<node> and the TOOL node as author", async () => {
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onApply = () => ({ ok: true, version: "v2", result: { path: "a.md", hash: "h", version: "v2" } });
    return session;
  });
  const { result } = await run(flow("append", "{path: a.md, text: x}"), memory);
  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ path: "a.md", hash: "h", version: "v2" });
  expect(memory.sessions[0]!.applied).toEqual([
    {
      id: "r/found",
      change: { op: "append", path: "a.md", text: "x" },
      author: { kind: "tool", name: "found", run: "r", node: "found" },
    },
  ]);
});

test("a conflict fails the node with node.execute, and retries: 2 doesn't retry it", async () => {
  const root = mkdtempSync(join(tmp, "m-"));
  writeFileSync(join(root, "a.md"), "first");
  const { result, log } = await run(
    flow("edit", "{path: a.md, old: missing, new: x}", 2),
    new FileMemory({ root, id: "t/lessons" }),
  );
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.message).toContain("the old text isn't in `a.md`");
  expect(readFileSync(log, "utf8")).not.toContain('"tool_attempt"');
  expect(readFileSync(join(root, "a.md"), "utf8")).toBe("first");
});

test("a reused change id fails the node with node.execute, unretried", async () => {
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onApply = () => {
      throw new ChangeIdReusedError("r/found");
    };
    return session;
  });
  const { result } = await run(flow("append", "{path: a.md, text: x}", 2), memory);
  expect(result.status).toBe("failed");
  expect(result.failure!.code).toBe("node.execute");
  expect(result.failure!.message).toContain("change id `r/found` was used for another change");
  expect(memory.sessions[0]!.applied).toHaveLength(1);
});

test("a busy memory is retried with the same call id", async () => {
  let calls = 0;
  const memory = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onApply = () => {
      calls += 1;
      if (calls === 1) throw new MemoryBusyError("held");
      return { ok: true, version: "v2", result: { path: "a.md", version: "v2" } };
    };
    return session;
  });
  const { result } = await run(flow("append", "{path: a.md, text: x}", 2), memory);
  expect(result.status).toBe("succeeded");
  expect(memory.sessions[0]!.applied.map((each) => each.id)).toEqual(["r/found", "r/found"]);

  const always = new FakeMemory(() => {
    const session = new RecordingSession(fileInfo("t/lessons"));
    session.onApply = () => {
      throw new MemoryBusyError("held");
    };
    return session;
  });
  const failed = await run(flow("append", "{path: a.md, text: x}", 1), always);
  expect(failed.result.failure!.code).toBe("node.execute");
  expect(failed.result.failure!.message).toContain("`lessons` is busy; try again");
  expect(always.sessions[0]!.applied).toHaveLength(2);
});

test("a memory tool never reaches a run tool group of the same name", async () => {
  const node = new ToolNode("found", { toolId: { group: "lessons", name: "read" }, memoryTool: true });
  const hostRead = new FunctionTool({
    name: "read",
    description: "Read.",
    parametersJsonSchema: { type: "object", properties: { path: { type: "string" } } },
    execute: () => "host",
  });
  const session = new RecordingSession(fileInfo("t/lessons"));
  session.onCall = () => ({ text: "memory" });
  const tools = new ToolSet({ lessons: [hostRead] }).forRun("r");
  const caps = { tools, memories: { session: () => session } };
  const result = await node.run({ path: "a.md", __memory_lessons: fileRef("t/lessons") }, caps);
  expect(result).toMatchObject({ value: { text: "memory" } });
  expect(session.calls).toEqual([{ tool: "read", args: { path: "a.md" } }]);
});
