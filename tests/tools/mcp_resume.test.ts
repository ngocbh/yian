/**
 * `advance` opens the MCP servers passed to it unopened only for an unfinished run, over the fake
 * server in `tests/fixtures/mcp/`. Every run sees only the test's credential store in its
 * environment.
 */

import { afterAll, afterEach, beforeEach, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { type AdvanceOptions, type RunFlowOptions, advance, runFlow } from "../../src/run.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { McpConnectError, type McpServer, mcpServer } from "../../src/tools/mcp.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { expectClosed, hostEnv, killMarkers, stdioOptions, strayProcesses } from "../helpers/mcp_helpers.ts";
import { Crash, rejected } from "../helpers/run_state_fixtures.ts";

const SLOW = 30_000;
const markers: string[] = [];
let dir: string;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "yian-mcp-resume-"));
});

afterEach(() => {
  killMarkers(markers.splice(0));
  rmSync(dir, { recursive: true, force: true });
});

afterAll(() => {
  expect(strayProcesses()).toEqual([]);
});

const WHOAMI_FLOW = `
id: mcp_resume
name: mcp_resume
nodes:
  call:
    kind: tool
    tool_id: whoami
output: \${call.output}
`;

/** `runFlow` in `hostEnv(dir)`. */
function run(flow: ComposedFlow, inputs: Readonly<Record<string, unknown>>, options: RunFlowOptions = {}) {
  return runFlow(flow, inputs, { env: hostEnv(dir), ...options });
}

/** `advance` in `hostEnv(dir)`. */
function resume(flow: ComposedFlow, options: AdvanceOptions) {
  return advance({ env: hostEnv(dir), ...options });
}

/** An unopened fake server over stdio, its pid appended to a fresh marker file. */
function fake(): [McpServer, string] {
  const file = join(dir, `marker-${markers.length + 1}`);
  markers.push(file);
  return [mcpServer(stdioOptions("--marker", file)), file];
}

/** Crash a run of `WHOAMI_FLOW` inside its TOOL node and return the call's context. */
async function crashedRun(log: string): Promise<ToolContext> {
  const seen: ToolContext[] = [];
  const crash = new Crash("simulated crash");
  let frozen = false;
  const whoami = tool({
    name: "whoami",
    description: "Crash the process on the call.",
    parameters: { type: "object", properties: {} },
    execute: (_args, ctx) => {
      seen.push(ctx);
      frozen = true;
      return "never recorded";
    },
  });
  const freeze = () => {
    if (frozen) throw crash;
  };
  const flow = await loadFlow(WHOAMI_FLOW);
  const crashed = withAppendHook(freeze, () => run(flow, {}, { log, tools: { whoami }, runId: "run-5c2e" }));
  expect(await rejected(crashed)).toBe(crash);
  expect(seen).toHaveLength(1);
  return seen[0]!;
}

test(
  "a resumed tool node calls an unopened server which the resume closes",
  async () => {
    const log = join(dir, "run.jsonl");
    const first = await crashedRun(log);
    const [server, marker] = fake();

    const resumed = await resume(await loadFlow(WHOAMI_FLOW), { log, tools: { fake: server } });

    expect(resumed.failure).toBeNull();
    expect(resumed.status).toBe("succeeded");
    expect(first.callId).toBe("run-5c2e/call");
    expect(resumed.output).toBe(`run-5c2e/call ${first.attempt + 1}`);
    await expectClosed(server, marker);
  },
  SLOW,
);

test(
  "resuming a finished run starts no server",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    const whoami = tool({
      name: "whoami",
      description: "Answer without a server.",
      parameters: { type: "object", properties: {} },
      execute: () => "local",
    });
    const finished = await run(flow, {}, { log, tools: { whoami } });
    expect(finished.status).toBe("succeeded");
    const [server, marker] = fake();

    const resumed = await resume(flow, { log, tools: { fake: server } });

    expect(resumed.status).toBe("succeeded");
    expect(resumed.output).toBe("local");
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
  },
  SLOW,
);

test(
  "a bad entry fails before the log is touched",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const before = [readFileSync(log), statSync(log, { bigint: true }).mtimeNs];
    const [server, marker] = fake();

    const error = await rejected(
      resume(flow, { log, tools: { fake: server, g: [(text: string) => text] as never } }),
    );

    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toStartWith('tools["g"][0] must be a FunctionTool');
    expect([readFileSync(log), statSync(log, { bigint: true }).mtimeNs]).toEqual(before);
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
  },
  SLOW,
);

test(
  "a server that fails to open on resume closes the log",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const bad = mcpServer({ command: "/nonexistent/x" });

    expect(await rejected(resume(flow, { log, tools: { fake: bad } }))).toBeInstanceOf(McpConnectError);

    expect(bad.isOpen).toBe(false);
    const [server] = fake();
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
    await expectClosed(server);
  },
  SLOW,
);

test(
  "a check that needs no tools fails before the server starts",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const [server, marker] = fake();

    const error = await rejected(resume(flow, { log, tools: { fake: server }, skills: new Map([["x", {}]]) as never }));

    expect(error).toBeInstanceOf(TypeError);
    expect((error as Error).message).toContain("skills");
    expect(existsSync(marker)).toBe(false);
    await expectClosed(server);
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
  },
  SLOW,
);

test(
  "a stop before the servers open on resume starts none",
  async () => {
    const log = join(dir, "run.jsonl");
    const flow = await loadFlow(WHOAMI_FLOW);
    await crashedRun(log);
    const [a, markerA] = fake();
    const [b, markerB] = fake();
    const stop = new AbortController();
    stop.abort(new Error("stopped"));

    expect(await rejected(resume(flow, { log, tools: { a, b }, signal: stop.signal }))).toBe(stop.signal.reason);

    expect(existsSync(markerA) || existsSync(markerB)).toBe(false);
    await expectClosed(a);
    await expectClosed(b);
    const [server] = fake();
    const resumed = await resume(flow, { log, tools: { fake: server } });
    expect(resumed.status).toBe("succeeded");
    await expectClosed(server);
  },
  SLOW,
);
