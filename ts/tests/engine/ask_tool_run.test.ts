/**
 * The run-start tool checks for the TOOL built-ins, through `runFlow`: `take` and `sleep` need no
 * tool in the run's tools, an `ask` node's inputs are checked against yian's base surface plus
 * the host's declared extras, and `ask` needs the run's handler. TS-only: Python yian has no
 * `take`, `sleep` or `ask` built-ins.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { type RunFlowOptions, runFlow } from "../../src/run.ts";
import type { AskRequest } from "../../src/tools/registry.ts";
import { hostEnv } from "./mcp_helpers.ts";
import { rejected } from "./run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-ask-tool-run-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);

/** `runFlow` in the test's environment. */
const run = async (source: string, options: RunFlowOptions = {}) =>
  runFlow(await loadFlow(source), {}, { env: hostEnv(dir), ...options });

/** A flow whose one TOOL node `gate` names `tool` with the `input` lines given. */
const gateFlow = (tool: string, input: string) => `
id: gate
name: gate
nodes:
  gate:
    kind: tool
    tool_id: ${tool}
    input:
${input}
output: \${gate.output}
`;

const START = Date.UTC(2026, 9, 23);

test("an ask answered inline succeeds; the request has the call id and a deadline from the clock", async () => {
  const seen: AskRequest[] = [];
  const result = await run(gateFlow("ask", '      prompt: "Ship?"\n      timeout: "1h"'), {
    log: freshLog(),
    scheduler: virtualScheduler(START),
    tools: { ask: (request: AskRequest) => (seen.push(request), "yes") },
  });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("yes");
  expect(seen).toHaveLength(1);
  expect(seen[0]!.callId).toBe(`${result.runId}/gate`);
  expect(seen[0]!.question).toBe("Ship?");
  expect(seen[0]!.deadline).toBe("2026-10-23T01:00:00.000Z");
});

test("an ask input the host didn't declare is refused before the log exists", async () => {
  const path = freshLog();
  const flow = gateFlow("ask", '      prompt: "?"\n      to: lead');

  const error = await rejected(run(flow, { log: path, tools: { ask: () => "yes" } }));

  expect(error).toEqual(
    new RangeError(
      `TOOL node 'gate' input 'to': tool 'ask' takes no such input; it takes ["__questions","prompt","questions","timeout"]`,
    ),
  );
  expect(existsSync(path)).toBe(false);
});

test("a declared extra runs and reaches the handler in extra", async () => {
  const seen: AskRequest[] = [];
  const handler = (request: AskRequest) => (seen.push(request), "yes");
  const result = await run(gateFlow("ask", '      prompt: "?"\n      to: lead'), {
    log: freshLog(),
    tools: { ask: { handler, inputs: { to: { type: "string" } } } },
  });

  expect(result.status).toBe("succeeded");
  expect(seen[0]!.extra).toEqual({ to: "lead" });
});

test("an ask with no handler is refused before the log exists", async () => {
  const path = freshLog();

  const error = await rejected(run(gateFlow("ask", '      prompt: "?"'), { log: path, tools: {} }));

  expect(error).toEqual(
    new RangeError(
      "TOOL node 'gate' names 'ask': no ask handler for this run; pass tools: {...registeredTools(), ask: handler} to runFlow or advance",
    ),
  );
  expect(existsSync(path)).toBe(false);
});

for (const [tool, input] of [
  ["sleep", '      for: "1s"'],
  ["take", '      take: "all"'],
] as const) {
  test(`a YAML TOOL ${tool} passes the run-start checks with no tools`, async () => {
    const path = freshLog();

    // The run gets past its checks and writes its log; how the wait ends is the engine's.
    await run(gateFlow(tool, input), { log: path, tools: {} }).catch((error: unknown) => {
      expect(String(error)).not.toContain("isn't in this run's tools");
      expect(String(error)).not.toContain("names");
    });

    expect(existsSync(path)).toBe(true);
  });
}
