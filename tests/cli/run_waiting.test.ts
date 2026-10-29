/**
 * `yian run` on a waiting or cancelled result: the waiting document and text, `run.cancelled`, and
 * no prompt or usage error for an inbox input.
 */
import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { gatherMissingInputs, waitingDocument, waitingText } from "../../src/cli/run.ts";
import { UsageError } from "../../src/cli/utils.ts";
import { Parameter, ValueContract } from "../../src/nodes/contracts.ts";
import { inboxType, Type, ValueKind } from "../../src/typesys/values.ts";
import { printJson } from "../../src/cli/utils.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { replayRunLog, RunResult, runFlow } from "../../src/run.ts";
import { JsonlRunLog } from "../../src/runtime/log.ts";
import { commitPrepared, prepareRunCancelled } from "../../src/runtime/state.ts";
import { NO_ENV, rejected, runCli } from "../helpers/run_state_fixtures.ts";
import { loadSchema, type Schema, validate } from "../helpers/schema.ts";

const ROOT = mkdtempSync(join(tmpdir(), "run-waiting-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));

const schema = loadSchema("cli.schema.json");

/** What `write` wrote, joined. */
async function printed(write: (out: { write(text: string): Promise<void> }) => Promise<void>): Promise<string> {
  const parts: string[] = [];
  await write({ write: async (text) => void parts.push(text) });
  return parts.join("");
}

const waiting = new RunResult({
  input: {},
  status: "waiting",
  runId: "r",
  log: "/l.jsonl",
  waits: [{ node: "ask", callId: "r/ask", tool: "ask", inbox: null, deadline: "2026-10-24T00:00:00.000Z", settled: false }],
});

test("the waiting document matches run_document", async () => {
  const doc = JSON.parse(await printed((out) => printJson(waitingDocument(waiting), out)));
  expect(doc).toEqual({
    format: expect.anything(),
    status: "waiting",
    wake_at: "2026-10-24T00:00:00.000Z",
    waits: [{ node: "ask", call_id: "r/ask", tool: "ask", inbox: null, deadline: "2026-10-24T00:00:00.000Z", settled: false }],
  });
  expect(validate(schema, (schema.$defs as Record<string, Schema>).run_document!, doc)).toEqual([]);
});

test("the text form names each wait and how to continue", () => {
  expect(waitingText(waiting)).toBe("run waiting: ask (ask)\ncontinue with: yian run --resume /l.jsonl\n");
});

test("the command to continue quotes a log path the shell would split", () => {
  const log = "/my runs/l.jsonl";
  const spaced = new RunResult({ input: {}, status: "waiting", runId: "r", log, waits: waiting.waits });
  expect(waitingText(spaced)).toEndWith("continue with: yian run --resume '/my runs/l.jsonl'\n");
});

const NAP = `id: nap
name: nap
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      for: "1h"
output: \${nap.output}
`;

test("yian run --resume on a waiting run prints it and exits 0", async () => {
  const log = join(ROOT, "waiting.jsonl");
  expect((await runFlow(await loadFlow(NAP), {}, { log, env: NO_ENV })).status).toBe("waiting");
  const { code, stdout, stderr } = await runCli(["--resume", log, "--json"], () => "");
  expect(code).toBe(0);
  const doc = JSON.parse(stdout);
  expect([doc.status, doc.waits[0].node, doc.waits[0].tool]).toEqual(["waiting", "nap", "sleep"]);
  expect(validate(schema, (schema.$defs as Record<string, Schema>).run_document!, doc)).toEqual([]);
  expect(stderr).not.toContain("resume failed");
});

test("yian run --resume on a cancelled run prints run cancelled and exits 1 with run.cancelled", async () => {
  const log = join(ROOT, "cancelled.jsonl");
  const flow = await loadFlow(NAP);
  expect((await runFlow(flow, {}, { log, env: NO_ENV })).status).toBe("waiting");
  const authority = JsonlRunLog.resume(log);
  try {
    const state = replayRunLog(flow, authority);
    commitPrepared(authority, state, prepareRunCancelled(state));
  } finally {
    authority.close();
  }
  const text = await runCli(["--resume", log], () => "");
  expect(text.code).toBe(1);
  expect(text.stderr).toContain("run cancelled");
  const json = await runCli(["--resume", log, "--json"], () => "");
  expect(json.code).toBe(1);
  expect(JSON.parse(json.stdout).error.code).toBe("run.cancelled");
});

test("input gathering skips an inbox input, which takes events through send", async () => {
  const inbox = new Parameter("m", { contract: new ValueContract(inboxType(Type.scalar(ValueKind.STRING))), required: true });
  const topic = new Parameter("topic", { contract: new ValueContract(Type.scalar(ValueKind.STRING)), required: true });
  expect(await gatherMissingInputs([inbox], {}, null)).toEqual({});
  const error = await rejected(gatherMissingInputs([inbox, topic], {}, null));
  expect(error).toBeInstanceOf(UsageError);
  expect((error as Error).message).toEndWith("missing required flow input: topic");
});
