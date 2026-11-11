/**
 * `examples/approve-plan.yaml`, the escalation example, end to end. Its `ask` nodes go
 * to a host that declares the `to:` extension and answers later with `complete`; a timeout
 * closes an unanswered ask with `null` (or fails a strict one with `tool.timeout`), and the
 * flow then asks the manager.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { loadFlow } from "../../src/compose/loader.ts";
import { advance, complete, DeliveryRefused, pendingCalls, runFlow } from "../../src/run.ts";
import { pending } from "../../src/tools/context.ts";
import type { AskRequest } from "../../src/tools/registry.ts";
import { handClock } from "../helpers/inbox_fixtures.ts";
import { NO_ENV as env, rejected } from "../helpers/run_state_fixtures.ts";

const EXAMPLES_DIR = fileURLToPath(new URL("../../examples", import.meta.url));
const TEXT = readFileSync(join(EXAMPLES_DIR, "approve-plan.yaml"), "utf8");
const ROOT = mkdtempSync(join(tmpdir(), "yian-escalation-"));
afterAll(() => rmSync(ROOT, { recursive: true, force: true }));

const T0 = Date.UTC(2026, 9, 24, 9, 0, 0);
const HOUR = 3600 * 1000;
const PLAN = { plan: "Ship v2 on Friday" };
let logs = 0;

/** A fresh log path, the asks the host saw, and a host whose ask always returns `pending()`. */
function setUp() {
  logs += 1;
  const asked: AskRequest[] = [];
  const tools = {
    ask: {
      handler: (request: AskRequest) => {
        asked.push(request);
        return pending();
      },
      inputs: { to: { type: "string" } },
    },
  };
  return { log: join(ROOT, `run-${logs}.jsonl`), asked, tools };
}

test("the lead answers: the run ends with the lead's answer and never asks the manager", async () => {
  const { log, asked, tools } = setUp();
  const flow = await loadFlow(TEXT);
  const started = await runFlow(flow, PLAN, { log, env, tools, scheduler: handClock(T0) });
  expect(started.status).toBe("waiting");
  const [call, ...rest] = pendingCalls(log);
  expect(rest).toEqual([]);
  expect(call).toMatchObject({ node: "lead", tool: "ask", deadline: "2026-10-25T09:00:00.000Z" });
  expect(call!.arguments!["to"]).toBe("lead");
  expect(asked.map((request) => request.extra)).toEqual([{ to: "lead" }]);
  await complete(log, call!.callId, { output: { decision: "approve", note: null } }, { env });
  const done = await advance({ log, env, tools, scheduler: handClock(T0 + HOUR) });
  expect([done.status, done.output]).toEqual(["succeeded", { decision: "approve", note: null }]);
  expect(asked).toHaveLength(1);
});

test("the lead times out: the manager is asked, and a late answer from the lead is refused", async () => {
  const { log, asked, tools } = setUp();
  const flow = await loadFlow(TEXT);
  await runFlow(flow, PLAN, { log, env, tools, scheduler: handClock(T0) });
  const [lead] = pendingCalls(log);
  const later = T0 + 24 * HOUR + 1000;
  const escalated = await advance({ log, env, tools, scheduler: handClock(later) });
  expect(escalated.status).toBe("waiting");
  const [manager, ...rest] = pendingCalls(log);
  expect(rest).toEqual([]);
  expect(manager).toMatchObject({ node: "manager", tool: "ask", deadline: "2026-10-27T09:00:01.000Z" });
  expect(asked.map((request) => request.extra)).toEqual([{ to: "lead" }, { to: "manager" }]);
  const refused = await rejected(complete(log, lead!.callId, { output: { decision: "approve", note: null } }, { env }));
  expect(refused).toBeInstanceOf(DeliveryRefused);
  expect((refused as DeliveryRefused).reason).toBe("unknown_call");
  await complete(log, manager!.callId, { output: { decision: "reject", note: "too risky" } }, { env });
  const done = await advance({ log, env, tools, scheduler: handClock(later) });
  expect([done.status, done.output]).toEqual(["succeeded", { decision: "reject", note: "too risky" }]);
});

test("both time out: the run ends with null", async () => {
  const { log, tools } = setUp();
  await runFlow(await loadFlow(TEXT), PLAN, { log, env, tools, scheduler: handClock(T0) });
  await advance({ log, env, tools, scheduler: handClock(T0 + 24 * HOUR + 1000) });
  const done = await advance({ log, env, tools, scheduler: handClock(T0 + 72 * HOUR + 2000) });
  expect([done.status, done.output]).toEqual(["succeeded", null]);
});

test("a strict answer type: the lead's timeout fails the run with tool.timeout", async () => {
  const { log, tools } = setUp();
  const strict = TEXT.replace(
    "    output: !Optional(Approval)          # null: the lead didn't answer in time",
    "    output: !Approval",
  );
  expect(strict).not.toBe(TEXT);
  await runFlow(await loadFlow(strict), PLAN, { log, env, tools, scheduler: handClock(T0) });
  const done = await advance({ log, env, tools, scheduler: handClock(T0 + 24 * HOUR + 1000) });
  expect([done.status, done.failure?.code]).toEqual(["failed", "tool.timeout"]);
});

test("a host that answers at once: the run ends without waiting", async () => {
  const { log } = setUp();
  const tools = {
    ask: { handler: () => ({ decision: "approve", note: null }), inputs: { to: { type: "string" } } },
  };
  const done = await runFlow(await loadFlow(TEXT), PLAN, { log, env, tools, scheduler: handClock(T0) });
  expect([done.status, done.output]).toEqual(["succeeded", { decision: "approve", note: null }]);
  expect(pendingCalls(log)).toEqual([]);
});
