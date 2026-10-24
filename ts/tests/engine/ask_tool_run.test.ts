/**
 * The run-start tool checks for the TOOL built-ins, through `runFlow`: `take` and `sleep` need no
 * tool in the run's tools, an `ask` node's inputs are checked against yian's base surface plus
 * the host's declared extras, and `ask` needs the run's handler. TS-only: Python yian has no
 * `take`, `sleep` or `ask` built-ins.
 *
 * Rows that pinned the same behaviour on HUMAN_INPUT and WAIT (`blocking_human_input.test.ts`,
 * `blocking_wait.test.ts`) run here on the built-ins: an ask sends its prompt, its answer schema's
 * enum and its title (`node_name:`); an inline null answer is an ordinary output; literal
 * questions render their templates, labels and descriptions included; a sleep until a past date,
 * datetime or offset datetime returns null at once.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { advance, type RunFlowOptions, runFlow } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
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

// The run boundary that reports a waiting run lands after the engine that writes one; until then
// it throws this once the engine has ended waiting.
const WAITING_UNREPORTED = "Error: engine returned without deriving a terminal run result";

test("a YAML TOOL take passes the run-start checks with no tools and waits", async () => {
  const path = freshLog();
  const flow = `
id: gate
name: gate
input:
  m: !Inbox(str)
nodes:
  gate:
    kind: tool
    tool_id: take
    input:
      from: \${input.m}
output: \${gate.output}
`;

  const outcome = await run(flow, { log: path, tools: {} }).then(
    (result) => result.status,
    (error: unknown) => String(error),
  );

  expect(["waiting", WAITING_UNREPORTED]).toContain(outcome);
  expect(readFileSync(path, "utf8")).toContain('"run_waiting"');
});

test("a YAML TOOL sleep passes the run-start checks with no tools and waits", async () => {
  const path = freshLog();

  const outcome = await run(gateFlow("sleep", '      for: "1s"'), { log: path, tools: {} }).then(
    (result) => result.status,
    (error: unknown) => String(error),
  );

  // The node bound its inputs and returned Pending, and the engine wrote the wait.
  expect(["waiting", WAITING_UNREPORTED]).toContain(outcome);
  expect(readFileSync(path, "utf8")).toContain('"run_waiting"');
});

// Rows ported from the HUMAN_INPUT and WAIT tests (TS yian has neither kind).

test("an ask sends its prompt, its answer schema's enum and its title, and returns the answer", async () => {
  const seen: AskRequest[] = [];
  const flow = `
id: gate
name: gate
input:
  plan: !str
nodes:
  approve:
    kind: tool
    tool_id: ask
    node_name: Plan approval
    input:
      prompt: "Approve \${input.plan}?"
    output: Literal(approve, revise)
output: \${approve.output}
`;
  const result = await runFlow(await loadFlow(flow), { plan: "release" }, {
    env: hostEnv(dir),
    log: freshLog(),
    tools: { ask: (request: AskRequest) => (seen.push(request), "approve") },
  });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBe("approve");
  expect(seen).toEqual([
    {
      question: "Approve release?",
      options: null,
      questions: null,
      answerSchema: [{ name: "answer", type: "str", required: true, enum: ["approve", "revise"] }],
      title: "Plan approval",
      nodeId: "approve",
      callId: `${result.runId}/approve`,
      deadline: null,
      extra: {},
    },
  ]);
});

test("an inline null answer is an ordinary output", async () => {
  const result = await run(gateFlow("ask", '      prompt: "Anything to add?"'), {
    log: freshLog(),
    tools: { ask: () => null },
  });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBeNull();
});

test("literal questions render their templates, labels and descriptions included", async () => {
  const seen: AskRequest[] = [];
  const flow = `
id: choose
name: choose
input:
  project: !str
  candidate: !str
nodes:
  choose:
    kind: tool
    tool_id: ask
    input:
      prompt: "Choose for \${input.project}"
      questions:
        - question: "Which framework for \${input.project}?"
          header: Framework
          options:
            - label: \${input.candidate}
              description: "Use \${input.candidate}"
output: \${choose.output}
`;
  const result = await runFlow(await loadFlow(flow), { project: "Atlas", candidate: "Vue" }, {
    env: hostEnv(dir),
    log: freshLog(),
    tools: { ask: (request: AskRequest) => (seen.push(request), { Framework: "Vue" }) },
  });

  expect(result.status).toBe("succeeded");
  expect(result.output).toEqual({ Framework: "Vue" });
  expect(seen.map((request) => [request.question, request.questions])).toEqual([
    [
      "Choose for Atlas",
      [
        {
          question: "Which framework for Atlas?",
          header: "Framework",
          options: [{ label: "Vue", description: "Use Vue" }],
          multiSelect: false,
        },
      ],
    ],
  ]);
});

test.each([
  ["a date", "2000-01-01"],
  ["a datetime", "2000-01-01T01:02:03"],
  ["a datetime with an offset", "2000-01-01T05:30:00+05:30"],
])("a sleep until a past deadline returns null at once: %s", async (_, until) => {
  const path = freshLog();

  const result = await run(gateFlow("sleep", `      until: "${until}"`), { log: path, tools: {} });

  expect(result.status).toBe("succeeded");
  expect(result.output).toBeNull();
  expect(readFileSync(path, "utf8")).not.toContain('"node_waiting"');
});

test("an ask with no output: that nobody answers by its deadline fails tool.timeout", async () => {
  const log = freshLog();
  const flow = gateFlow("ask", '      prompt: "Ship?"\n      timeout: "1h"');
  const tools = { ask: () => pending() };
  const waiting = await run(flow, { log, scheduler: virtualScheduler(START), tools });
  expect(waiting.status).toBe("waiting");

  const result = await advance({ log, env: hostEnv(dir), scheduler: virtualScheduler(START + 2 * 3_600_000), tools });

  expect(result.status).toBe("failed");
  expect(result.failure?.code).toBe("tool.timeout");
  expect(result.failure?.message).toBe(
    "TOOL node 'gate' (ask): nobody answered by 2026-10-23T01:00:00.000Z; declare output: Optional(...) to route on a timeout",
  );
});

test("a run with no log refuses an ask's pending(): nothing could answer it", async () => {
  const result = await run(gateFlow("ask", '      prompt: "Ship?"'), { log: null, tools: { ask: () => pending() } });

  expect(result.status).toBe("failed");
  expect(result.failure?.nodeId).toBe("gate");
  expect(result.failure?.message).toContain("returned pending() outside a durable run; give the run a log");
});
