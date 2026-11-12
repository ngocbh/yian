/**
 * Learning and the old memories are gone: a run result has no backward pass, the removed options are
 * refused by name, the backward and memory commit records are unknown kinds, `run_started` holds no
 * memory state, the package root exports no old memory name, and no node has a backward hook.
 */

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { AgentNode } from "../../src/nodes/agent/node.ts";
import { Node } from "../../src/nodes/base.ts";
import { CaseNode } from "../../src/nodes/case/node.ts";
import { EndNode } from "../../src/nodes/end/node.ts";
import { LoopNode } from "../../src/nodes/loop/node.ts";
import { StartNode } from "../../src/nodes/start/node.ts";
import { advance, agentCall, runFlow } from "../../src/run.ts";
import { decodeRecord } from "../../src/runtime/log.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const FIXTURES = fileURLToPath(new URL("../conformance/fixtures", import.meta.url));

const PLAN_FLOW = `
id: run-plan
name: plan_flow
input:
  topic: !str
nodes:
  plan:
    kind: code
    code: ./compose_codefns.ts:make_plan
    input:
      topic: \${input.topic}
    output:
      rating: str
      score: float
output:
  rating: \${plan.output.rating}
`;

let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-learning-removed-"));
});

afterEach(() => {
  rmSync(dir, { recursive: true, force: true });
});

function load(): Promise<ComposedFlow> {
  return loadFlow(PLAN_FLOW, { sourceLabel: "flow.yaml", baseDir: FIXTURES });
}

/** Run the one-CODE-node flow to a durable log in the test's folder. */
async function runSimple() {
  const log = join(dir, "run.jsonl");
  const result = await runFlow(await load(), { topic: "owls" }, { log, env: hostEnv(dir) });
  return { result, log };
}

test("a run result has no backward pass", async () => {
  const { result } = await runSimple();
  expect(result.status).toBe("succeeded");
  expect("backward" in result).toBe(false);
  expect("learning" in result).toBe(false);
  expect("backwardOutcome" in result).toBe(false);
});

test("runFlow refuses the removed learnable option by name", async () => {
  const flow = await load();
  await expect(runFlow(flow, { topic: "owls" }, { learnable: true, env: hostEnv(dir) } as never)).rejects.toThrow(
    "learnable was removed: yian no longer learns from runs; drop the option",
  );
});

test("runFlow refuses the removed memory option by name", async () => {
  const flow = await load();
  await expect(runFlow(flow, { topic: "owls" }, { memory: new Map(), env: hostEnv(dir) } as never)).rejects.toThrow(
    "the memory option was removed: pass each memory as a flow input",
  );
});

test("agentCall refuses the removed memory option by name; advance refuses the old map of memories", async () => {
  const { log } = await runSimple();
  const message = "the memory option was removed: pass each memory as a flow input";
  await expect(agentCall(log, { nodeId: "plan", n: 1 }, { memory: new Map() } as never)).rejects.toThrow(message);
  // advance's memory binds a run's memory inputs again, by name; the old map isn't that.
  await expect(advance({ log, memory: new Map(), env: hostEnv(dir) } as never)).rejects.toThrow(
    "memory must be a record of memory input names to MemoryBase instances",
  );
});

test("run_started records no old memory state and no learnable flag", async () => {
  const { log } = await runSimple();
  const start = JSON.parse(readFileSync(log, "utf8").split("\n")[0]!);
  expect(start.kind).toBe("run_started");
  for (const field of ["memory_snapshots", "memories", "memory_stacks", "learnable"]) {
    expect(Object.keys(start.payload)).not.toContain(field);
  }
});

test.each([
  "backward_started",
  "backward_ready",
  "backward_finished",
  "memory_commit_requested",
  "memory_commit_succeeded",
  "memory_commit_failed",
])("a %s record is an unknown kind", (kind) => {
  const line = JSON.stringify({ at: "2026-11-06T00:00:00.000Z", kind, payload: {}, run_id: "r", schema: 32, seq: 1 });
  expect(() => decodeRecord(line)).toThrow(`run-record kind '${kind}' is unsupported in schema 32`);
});

test("a schema-26 log is refused by name", () => {
  const line26 = JSON.stringify({ at: "2026-11-06T00:00:00.000Z", kind: "node_started", payload: {}, run_id: "r", schema: 26, seq: 1 });
  expect(() => decodeRecord(line26)).toThrow("run log schema 26 predates schema 32 and is not supported");
});

test.each([Node, AgentNode, EndNode, StartNode, LoopNode, CaseNode].map((cls) => [cls.name, cls] as const))(
  "%s has no backward hook",
  (_name, cls) => {
    for (const hook of ["backward", "backwardInputNames", "captureBackwardContext", "renderCapture", "boundLlmConfig"]) {
      expect(hook in cls.prototype).toBe(false);
    }
  },
);

test("the package root has none of the old memory names", async () => {
  const root = await import("../../src/index.ts");
  for (const name of [
    "buildMemories",
    "ProfileMemory",
    "SQLiteStorage",
    "MarkdownStore",
    "FileMarkdownStore",
    "MarkdownVersionMismatch",
  ]) {
    expect(name in root).toBe(false);
  }
});

test("a node's declared inputs, without reserved wiring, are its dataInputNames", () => {
  expect(typeof Node.prototype.dataInputNames).toBe("function");
  expect(EndNode.record("__end__", { outputNames: ["a", "b"] }).dataInputNames()).toEqual(["a", "b"]);
});
