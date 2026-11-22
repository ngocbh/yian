/**
 * A held call the host released (`release`): its agent runs it once, as the model asked it, at the
 * first evaluation that began after the release, and the hook isn't asked about it again. When a
 * crash may have cut it after it started, a resumable tool is sent again under the same call id
 * with its next attempt, and any other is closed crashed; a release is never lost before its tool
 * starts, and an interrupt before it starts skips it.
 */

import { afterAll, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { loadFlow } from "../../src/compose/loader.ts";
import { CallHeld, NodeWaiting, NodeWoken, type RunEvent, ToolCallFinished, ToolCallStarted } from "../../src/events.ts";
import type { Response } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import type { ToolResultPart } from "../../src/llm/messages.ts";
import { TOOL_OUTCOME_UNKNOWN } from "../../src/nodes/agent/results.ts";
import { advance, complete, interrupt, pendingCalls, release, runFlow, type RunResult } from "../../src/run.ts";
import { NodeStepRecord, readRunLog, type RunLog, type RunRecord } from "../../src/runtime/log.ts";
import { pending } from "../../src/tools/context.ts";
import { FunctionTool, tool } from "../../src/tools/function_tools.ts";
import type { BeforeToolCall, GatedCall } from "../../src/tools/gate.ts";
import { mcpServer, type McpServer } from "../../src/tools/mcp.ts";
import type { JsonObject } from "../../src/typesys/json.ts";
import { withAppendHook } from "../helpers/append_hook.ts";
import { gated } from "../helpers/gate.ts";
import { expectClosed, hostEnv, stdioOptions, strayProcesses } from "../helpers/mcp_helpers.ts";
import { callReply, Crash, crashAfter, rejected, serving, textReply } from "../helpers/run_state_fixtures.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-agent-release-"));
afterAll(() => {
  rmSync(dir, { recursive: true, force: true });
  expect(strayProcesses()).toEqual([]);
});
let logs = 0;
/** A fresh log path. */
const freshLog = () => join(dir, `run-${++logs}.jsonl`);
const env = hostEnv(dir);
const CONFIG = { provider: "openai", model: "test-model" };

/** Where the module tool `price` notes each start, a line `<attempt> <call id>` each. */
const priced = join(dir, "priced");
writeFileSync(
  join(dir, "price.ts"),
  `import { appendFileSync } from "node:fs";\n` +
    `export const price = { name: "price", description: "Price.", parameters: { type: "object", properties: { q: { type: "string" } } }, ` +
    `resumable: true, execute: (args, ctx) => { appendFileSync(${JSON.stringify(priced)}, ctx.attempt + " " + ctx.callId + "\\n"); return "price:" + args.q; } };\n`,
);

/** One agent `reply` reading steer inbox `m`, granted the host tools, `price` (a module tool), `shout` (a flow) and `extra`. */
const flowOf = (extra = "") =>
  loadFlow(
    `
id: release
name: release
llm_config: {provider: openai, model: test-model}
uses:
  price: "./price.ts:price"
defs:
  shout:
    description: Shout the text back.
    input: {q: !str}
    kind: code
    code: return inputs.q.toUpperCase()
    output: !str
input:
  m: !Inbox(str)
nodes:
  reply:
    kind: agent
    prompt: go
    tools: [keep, once, alone, again, defer, checked, price, shout${extra}]
    steer_inbox: \${input.m}
    output: !Optional(str)
output: \${reply.output}
`,
    { searchPaths: [dir] },
  );

/** One start of a host tool's call: `<tool>:<q>@<attempt>`, its call id, and whether the log held an interrupt then. */
interface Start {
  readonly run: string;
  readonly callId: string;
  readonly afterInterrupt: boolean;
}

/**
 * Host tools noting each start in `starts`: `keep` (resumable) and `once` (not), safe to run
 * together, `alone` and `again` (resumable), which run alone, `defer`, whose handler returns
 * `pending()`, and `checked`, whose validator, once the log at `log` holds a release, refuses a
 * `q` ending in `refuse` and throws for one ending in `throw`. Each waits a moment, so calls that
 * start together are all started before any ends.
 */
function hostTools(log: string, starts: Start[]) {
  const make = (name: string, resumable: boolean, safe = true) =>
    tool({
      name,
      description: `Tool ${name}.`,
      parameters: { type: "object", properties: { q: { type: "string" } } },
      isConcurrencySafe: () => safe,
      resumable,
      execute: async (args, ctx) => {
        const afterInterrupt = existsSync(log) && readRunLog(log).some(({ kind }) => kind === "interrupted");
        starts.push({ run: `${name}:${String(args.q)}@${ctx.attempt}`, callId: ctx.callId, afterInterrupt });
        await new Promise((resolve) => setTimeout(resolve, 2));
        return name === "defer" ? pending() : `${name}:${String(args.q)}`;
      },
    });
  const released = () => existsSync(log) && stepsOf(log).some((step) => step["kind"] === "released");
  const checked = new FunctionTool({
    name: "checked",
    description: "Tool checked.",
    parametersJsonSchema: { type: "object", properties: { q: { type: "string" } } },
    parametersValidator: {
      "~standard": {
        version: 1,
        vendor: "test",
        validate: (value: unknown) => {
          const q = String((value as JsonObject)["q"]);
          if (released() && q.endsWith("throw")) throw new Error("validator down");
          return released() && q.endsWith("refuse") ? { issues: [{ message: "no longer allowed" }] } : { value };
        },
      },
    },
    execute: () => (starts.push({ run: "checked", callId: "", afterInterrupt: false }), "checked"),
  });
  return {
    keep: make("keep", true),
    once: make("once", false),
    alone: make("alone", false, false),
    again: make("again", true, false),
    defer: make("defer", false),
    checked,
  };
}

/** The call id of the first reply's call `index`. */
const heldId = (index: number) => `r/reply/1/${index}`;

/** The steps node `reply` recorded in the log at `log`; none before the log exists. */
const stepsOf = (log: string) =>
  !existsSync(log) ? [] : readRunLog(log).flatMap(({ record }) => (record instanceof NodeStepRecord && record.nodeId === "reply" ? [record.step] : []));

/** How many model calls node `reply` recorded in the log at `log`. */
const modelCalls = (log: string) => stepsOf(log).filter((step) => step["kind"] === "model_call").length;

/** The call ids of the tool results `model` read in its request `index`, in order. */
const readIds = (model: ScriptedModel, index = 0) =>
  model.requests[index]!.messages.flatMap((message) => (message.role === "tool" ? message.parts.map((part) => (part as ToolResultPart).callId) : []));

/** How the host releases each hold: once the run is idle, from `CallHeld` while its agent runs, or from `NodeWaiting` while the run moves. */
type When = "idle" | "flight" | "live";

/**
 * How a run of `life` went: its result, its log, the host tools' starts, the calls the hook was
 * asked about, its events, and the model its last part called.
 */
interface Life {
  readonly result: RunResult;
  readonly model: ScriptedModel;
  readonly log: string;
  readonly starts: readonly Start[];
  readonly asked: readonly GatedCall[];
  readonly events: readonly RunEvent[];
}

/** The options of `life`. */
interface LifeOptions {
  readonly when?: When;
  /** Where the life's parts are cut, in order. */
  readonly cuts?: readonly number[];
  /** Interrupt once, from the first `NodeWoken`, or with the first release in flight. */
  readonly interruptAt?: "wake" | "hold";
  readonly mcp?: boolean;
  /** The calls of a second reply. */
  readonly next?: readonly [string, string][];
  /** The status the life must end with; `succeeded` by default. */
  readonly status?: RunResult["status"];
}

/**
 * Run the flow (granting the MCP server `fake` too with `mcp`), whose agent's first reply makes
 * `calls` (`[tool, q]` each, call id `c<index>`), then `next` when given, to its end, under a hook
 * holding each call whose `q` (or `text`) starts with `h`. The host releases each hold `when`, and
 * before each advance releases any it hasn't and answers each other call pending; it interrupts
 * the agent once at `interruptAt`. The first part of the life (a start, an advance or a host's
 * call) to reach the first of `cuts` records is cut there, as a crash would, then the next to
 * reach the next, and the host resumes the run after each.
 */
async function life(
  calls: readonly [string, string][],
  { when = "idle", cuts = [], interruptAt, mcp = false, next, status = "succeeded" }: LifeOptions = {},
): Promise<Life> {
  rmSync(priced, { force: true });
  const log = freshLog();
  const flow = await flowOf(mcp ? ", fake" : "");
  const starts: Start[] = [];
  const asked: GatedCall[] = [];
  const hook: BeforeToolCall = (call) => {
    asked.push(call);
    return String(call.arguments["q"] ?? call.arguments["text"]).startsWith("h") ? pending() : undefined;
  };
  const events: RunEvent[] = [];
  const delivered: Promise<unknown>[] = [];
  const held = new Set<string>();
  let interrupted = false;
  const onEvent = (event: RunEvent) => {
    events.push(event);
    if (event instanceof CallHeld) held.add(event.callId);
    const releasing = when === "flight" && event instanceof CallHeld ? [event.callId] : when === "live" && event instanceof NodeWaiting ? event.callIds : [];
    for (const callId of releasing.filter((each) => held.has(each))) delivered.push(release(log, callId, { env }).catch(() => null));
    if (!interrupted && (interruptAt === "wake" ? event instanceof NodeWoken : interruptAt === "hold" && event instanceof CallHeld)) {
      interrupted = true;
      delivered.push(interrupt(log, "m", { env }).catch(() => null));
    }
  };
  const servers: McpServer[] = [];
  const tools = () => {
    const host = hostTools(log, starts);
    if (!mcp) return host;
    servers.push(mcpServer(stdioOptions()));
    return { ...host, fake: servers.at(-1)! };
  };
  const replyOf = (each: readonly [string, string][]) =>
    callReply(...each.map(([name, q], index): [string, JsonObject, string] => [name, name.startsWith("fake_") ? { text: q } : { q }, `c${index}`]));
  const replies = [replyOf(calls), ...(next === undefined ? [] : [replyOf(next)]), textReply("done")];
  let model = new ScriptedModel([], CONFIG);
  const options = () => {
    model = new ScriptedModel(replies.slice(modelCalls(log)), CONFIG);
    return gated({ env, log, runId: "r", tools: tools(), models: serving(model), onEvent }, hook);
  };
  let cutsDone = 0;
  /** Run `fn`, cut at the next of `cuts`; `null` when it was cut. */
  const part = async <T>(fn: () => Promise<T>): Promise<T | null> => {
    const cut = cuts[cutsDone];
    const crash = new Crash(`cut at ${String(cut)}`);
    const refuse = (target: RunLog) => {
      if (cut !== undefined && target.path !== null && target.nextSequence >= cut) throw crash;
    };
    let crashed = false;
    try {
      return await withAppendHook(refuse, fn);
    } catch (error) {
      if (error !== crash) throw error;
      crashed = true;
      return null;
    } finally {
      // A delivery from `onEvent` lands, or is cut, before the life goes on.
      await Promise.allSettled(delivered);
      if (crashed) cutsDone += 1;
    }
  };
  let result = await part(() => runFlow(flow, {}, options()));
  for (let round = 0; round < 8 && (result === null || result.status === "waiting"); round++) {
    for (const call of pendingCalls(log)) {
      await part(() => (call.held ? release(log, call.callId, { env }) : complete(log, call.callId, { output: "answered" }, { env })));
    }
    result = await part(() => advance(options()));
  }
  for (const server of servers) await expectClosed(server);
  expect(result?.status, result?.failure?.message).toBe(status);
  return { result: result!, model, log, starts, asked, events };
}

/** The seq of the first record of the log at `log` that `match` holds of; -1 when none does. */
const recordWhere = (log: string, match: (step: JsonObject | null, kind: string) => boolean) =>
  readRunLog(log).findIndex(({ kind, record }) => match(record instanceof NodeStepRecord ? record.step : null, kind));

/** The seq of the `released` step of the log at `log`, and of call `c0`'s result. */
const releaseSpan = (log: string): [number, number] => [
  recordWhere(log, (step) => step?.["kind"] === "released"),
  recordWhere(log, (step) => step?.["kind"] === "tool_result" && step["call_id"] === "c0"),
];

test("a release, idle, live or in flight, runs the held call once as the model asked it; the hook is asked about the next reply, never the call again", async () => {
  for (const when of ["idle", "live", "flight"] as const) {
    const { log, starts, asked, events } = await life([["defer", "h"]], { when, next: [["once", "n"]] });
    // The second reply is the agent's second model call.
    const next = stepsOf(log).flatMap((step, index) => (step["kind"] === "model_call" ? [index + 1] : []))[1];
    expect(starts.map((start) => [start.run, start.callId]), when).toEqual([["defer:h@1", heldId(0)], ["once:n@1", `r/reply/${next}/0`]]);
    expect(asked.map((call) => [call.arguments["q"], call.attempt]), when).toEqual([["h", 1], ["n", 1]]);
    // The released call's tool pended: it registered again as a plain host call, which the host answered.
    expect(stepsOf(log).filter((step) => step["kind"] === "pending").map((step) => [step["call_id"], step["held"] ?? false]), when).toEqual([
      [heldId(0), true],
      [heldId(0), false],
    ]);
    // The node waits on the hold (released in flight, it pends again and wakes at once), then on the call its tool pended.
    const waits = events.filter((event) => event instanceof NodeWaiting).map((event) => (event as NodeWaiting).callIds);
    expect(waits, when).toEqual([[heldId(0)], [heldId(0)]]);
    expect(stepsOf(log).find((step) => step["kind"] === "tool_result")?.["output"], when).toBe("answered");
  }
});

/**
 * Cut the life of `calls`, released `when`, at every record from the release through call
 * `c0`'s result, and check each cut life: the call has one result, the hook is asked about it once,
 * and it is never closed crashed unless it started. `check` checks the rest.
 */
async function sweep(calls: readonly [string, string][], when: When, check: (cut: Life, where: string) => void, mcp = false) {
  const [released, result] = releaseSpan((await life(calls, { when, mcp })).log);
  expect(released).toBeGreaterThan(0);
  for (let k = released; k <= result + 1; k++) {
    const cut = await life(calls, { when, cuts: [k], mcp });
    const where = `${when}, cut at ${k}`;
    const results = stepsOf(cut.log).filter((step) => step["kind"] === "tool_result" && step["call_id"] === "c0");
    expect(results.length, where).toBe(1);
    expect(cut.asked.filter((call) => call.callId === heldId(0)).length, where).toBe(1);
    check(cut, where);
  }
}

/** The attempts of the host tool starts of call `index` in `cut`, each under the call's id. */
const attemptsOf = (cut: Life, index = 0) => {
  const own = cut.starts.filter((start) => start.callId === heldId(index));
  return own.map((start) => Number(start.run.split("@")[1]));
};

/** Call `c0`'s result in `cut`. */
const outputOf = (cut: Life) => String(stepsOf(cut.log).find((step) => step["kind"] === "tool_result" && step["call_id"] === "c0")!["output"]);

test.each(["idle", "live", "flight"] as const)(
  "a crash at any record from a release (%s) through its result runs a non-resumable call once, never lost, and the later call after it",
  async (when) => {
    await sweep([["once", "h"], ["alone", "L"]], when, (cut, where) => {
      expect(attemptsOf(cut), where).toEqual([1]);
      expect([TOOL_OUTCOME_UNKNOWN, "once:h"], where).toContain(outputOf(cut));
      // D4: the later call runs after the released one, at most once.
      expect(cut.starts.map((start) => start.run.split("@")[0]), where).toEqual(["once:h", "alone:L"]);
    });
  },
  60_000,
);

test.each(["idle", "live", "flight"] as const)(
  "a crash at any record from a release (%s) through its result sends a resumable call again under its call id, with its next attempt",
  async (when) => {
    await sweep([["keep", "h"], ["alone", "L"]], when, (cut, where) => {
      const attempts = attemptsOf(cut);
      expect([[1], [1, 2]], where).toContainEqual(attempts);
      expect(outputOf(cut), where).toBe("keep:h");
      expect(cut.starts.filter((start) => start.run.startsWith("alone:L")).length, where).toBe(1);
    });
  },
  60_000,
);

test("a crash at any record from a release through its result sends a module tool's call again on its thread, under its call id", async () => {
  await sweep([["price", "h"]], "idle", (cut, where) => {
    const lines = readFileSync(priced, "utf8").trim().split("\n");
    expect([[`1 ${heldId(0)}`], [`1 ${heldId(0)}`, `2 ${heldId(0)}`]], where).toContainEqual(lines);
    expect(outputOf(cut), where).toBe("price:h");
  });
}, 60_000);

test("a crash at any record from a release through its result starts an MCP tool's call at most once, and never loses it", async () => {
  await sweep(
    [["fake_echo", "h"]],
    "idle",
    (cut, where) => {
      const started = cut.events.filter((event) => event instanceof ToolCallStarted && event.toolCallId === "c0");
      expect(started.length, where).toBe(1);
      expect([TOOL_OUTCOME_UNKNOWN, "h"], where).toContain(outputOf(cut));
    },
    true,
  );
}, 120_000);

test("a released flow call grows its scope once, only after its release, across a crash at any record through its answer", async () => {
  const uncut = await life([["shout", "h"]]);
  const [released] = releaseSpan(uncut.log);
  expect(recordWhere(uncut.log, (_, kind) => kind === "node_expanded")).toBeGreaterThan(released);
  expect(stepsOf(uncut.log).filter((step) => step["flow"] === true).map((step) => step["scope"])).toEqual(["reply/shout#0"]);
  for (const when of ["idle", "flight"] as const) {
    await sweep([["shout", "h"]], when, (cut, where) => {
      expect(readRunLog(cut.log).filter(({ kind }) => kind === "node_expanded").length, where).toBe(1);
      expect(outputOf(cut), where).toBe("H");
    });
  }
}, 60_000);

test("an interrupt after a release, before its run, skips the call unstarted; across a crash at any record, only its resend starts after it", async () => {
  for (const name of ["keep", "once"]) {
    const calls: [string, string][] = [[name, "h"], ["again", "L"]];
    const uncut = await life(calls, { interruptAt: "wake" });
    expect(uncut.starts).toEqual([]);
    expect(stepsOf(uncut.log).filter((step) => step["kind"] === "tool_result").map((step) => step["closed"])).toEqual(["skipped", "skipped"]);
    const [released] = releaseSpan(uncut.log);
    for (let k = released; k <= readRunLog(uncut.log).length; k++) {
      const cut = await life(calls, { interruptAt: "wake", cuts: [k] });
      // A cut at the skip leaves the resume unable to tell it from a cut while the call ran, as the
      // woken evaluation began before the interrupt: a resumable call is sent again (attempt 2).
      const resend = (start: Start) => start.callId === heldId(0) && !start.run.endsWith("@1");
      expect(cut.starts.filter((start) => start.afterInterrupt && !resend(start)), `${name}, cut at ${k}`).toEqual([]);
      expect(cut.starts.filter((start) => start.callId === heldId(0)).length, `${name}, cut at ${k}`).toBeLessThanOrEqual(name === "keep" ? 2 : 1);
      // The later call never started unless the cut fell before the interrupt: it reads skipped.
      const later = stepsOf(cut.log).find((step) => step["kind"] === "tool_result" && step["call_id"] === "c1")!;
      if (later["closed"] === "crashed") expect(cut.starts.map((start) => start.run), `${name}, cut at ${k}`).toContain("again:L@1");
    }
  }
}, 60_000);

test("a release and an interrupt in flight skip the call; across a crash at its skip, and the resume's, it never starts", async () => {
  for (const name of ["keep", "once"]) {
    const uncut = await life([[name, "h"]], { when: "flight", interruptAt: "hold" });
    const skipAt = recordWhere(uncut.log, (step) => step?.["closed"] === "skipped");
    expect(readRunLog(uncut.log)[skipAt - 1]!.kind, name).toBe("interrupted");
    // The resume writes run_resumed and node_started, then its own skip, cut too: the next start
    // finds every evaluation after the release began with the interrupt pending.
    for (const cuts of [[], [skipAt], [skipAt, skipAt + 2]]) {
      const cut = await life([[name, "h"]], { when: "flight", interruptAt: "hold", cuts });
      expect(cut.starts, `${name}, cut at ${cuts}`).toEqual([]);
      expect(stepsOf(cut.log).filter((step) => step["kind"] === "tool_result").map((step) => step["closed"]), `${name}, cut at ${cuts}`).toEqual(["skipped"]);
    }
  }
});

test("a release and an interrupt that land while the agent's office is down skip the call, which never starts", async () => {
  const flow = await flowOf();
  const log = freshLog();
  const starts: Start[] = [];
  const options = (replies: Response[]) =>
    gated({ env, log, runId: "r", tools: hostTools(log, starts), models: serving(new ScriptedModel(replies, CONFIG)) }, () => pending());
  const reply = callReply(["keep", { q: "h" }, "c0"]);
  // Cut right after the hold, before its agent waits: the agent is in flight in the log.
  await crashAfter(11, () => runFlow(flow, {}, options([reply])));
  expect((readRunLog(log).at(-1)!.record as NodeStepRecord).step["held"]).toBe(true);
  await release(log, heldId(0), { env });
  await interrupt(log, "m", { env });
  expect((await advance(options([]))).status).toBe("succeeded");
  expect(starts).toEqual([]);
  expect(stepsOf(log).filter((step) => step["kind"] === "tool_result").map((step) => step["closed"])).toEqual(["skipped"]);
});

test("a release that lands between its node's wake and the node reading its calls pends the call again, which then runs once", async () => {
  // [keep h1, keep h2], both held. h1's answer wakes the node; h2's release lands from NodeWoken,
  // after the wake record, so that evaluation pends it and the next one runs it.
  const log = freshLog();
  const starts: Start[] = [];
  const delivered: Promise<unknown>[] = [];
  const events: RunEvent[] = [];
  const onEvent = (event: RunEvent) => {
    events.push(event);
    if (event instanceof NodeWoken && delivered.length === 0) delivered.push(release(log, heldId(1), { env }));
  };
  const hook: BeforeToolCall = () => pending();
  const replies = [callReply(["keep", { q: "h1" }, "c0"], ["keep", { q: "h2" }, "c1"]), textReply("done")];
  const options = (left: typeof replies) =>
    gated({ env, log, runId: "r", tools: hostTools(log, starts), models: serving(new ScriptedModel(left, CONFIG)), onEvent }, hook);
  expect((await runFlow(await flowOf(), {}, options(replies))).status).toBe("waiting");
  await complete(log, heldId(0), { output: "one" }, { env });
  const done = await advance(options(replies.slice(1)));
  await Promise.all(delivered);
  expect(done.status, done.failure?.message).toBe("succeeded");
  expect(starts.map((start) => start.run)).toEqual(["keep:h2@1"]);
  const waits = events.filter((event) => event instanceof NodeWaiting).map((event) => (event as NodeWaiting).callIds);
  expect(waits).toEqual([[heldId(0), heldId(1)], [heldId(1)]]);
});

test("a released call runs before a call cut beside its hold is sent again, so a crash during that resend never loses it", async () => {
  // [keep x, once h]: one group, h held. A crash cuts x's result; h is released while the run is
  // down; the resume runs h, then sends x again, and a crash cuts that resend's result too.
  const calls: [string, string][] = [["keep", "x"], ["once", "h"]];
  const resultOf = (log: string) => recordWhere(log, (step) => step?.["kind"] === "tool_result" && step["call_id"] === "c0");
  const ran = resultOf((await life(calls)).log);
  const resent = resultOf((await life(calls, { cuts: [ran] })).log);
  const { starts, log, model } = await life(calls, { cuts: [ran, resent] });
  expect(starts.map((start) => start.run)).toEqual(["keep:x@1", "once:h@1", "keep:x@2", "keep:x@3"]);
  // The model reads the results in call order, though h ran before x was sent again.
  expect(readIds(model)).toEqual(["c0", "c1"]);
  expect(stepsOf(log).find((step) => step["kind"] === "tool_result" && step["call_id"] === "c1")!["output"]).toBe("once:h");
});

test("two released calls of one group start together", async () => {
  const { starts, events } = await life([["keep", "h1"], ["once", "h2"]]);
  expect(starts.map((start) => start.run)).toEqual(["keep:h1@1", "once:h2@1"]);
  const brackets = events.flatMap((event) =>
    event instanceof ToolCallStarted ? [`start ${event.toolCallId}`] : event instanceof ToolCallFinished ? [`end ${event.toolCallId}`] : []);
  expect(brackets).toEqual(["start c0", "start c1", "end c0", "end c1"]);
});

/**
 * Run `fn`, refusing appends once it appends call `callId`'s first `tool_result` step, as a crash
 * there would.
 */
async function cutAtResult(callId: string, fn: () => Promise<unknown>): Promise<void> {
  const crash = new Crash(`cut at ${callId}'s result`);
  let cut = false;
  const refuse = (_target: RunLog, record: RunRecord) => {
    cut ||= record instanceof NodeStepRecord && record.step["kind"] === "tool_result" && record.step["call_id"] === callId;
    if (cut) throw crash;
  };
  expect(await withAppendHook(refuse, () => fn().then(() => null, (error: unknown) => error))).toBe(crash);
}

test.each(["a crash", "a stop"])(
  "two holds of a group released at different times: %s before the first runs, then a crash at its closer, never loses the second",
  async (how) => {
    // [once h1, once h2], held together. h1 is released, and its first evaluation is cut by a crash
    // as it ran, or stopped before it started; h2 is released meanwhile. The next start closes h1
    // crashed and runs h2 fresh, and a crash lands on h1's closer.
    const log = freshLog();
    const starts: Start[] = [];
    const replies = [callReply(["once", { q: "h1" }, "c0"], ["once", { q: "h2" }, "c1"]), textReply("done")];
    const stop = new AbortController();
    const onEvent = (event: RunEvent) => void (how === "a stop" && event instanceof NodeWoken && stop.abort(new Error("office shutting down")));
    const options = (extra = {}) =>
      gated({ env, log, runId: "r", tools: hostTools(log, starts), models: serving(new ScriptedModel(replies.slice(modelCalls(log)), CONFIG)), ...extra }, () => pending());
    expect((await runFlow(await flowOf(), {}, options())).status).toBe("waiting");
    await release(log, heldId(0), { env });
    if (how === "a crash") await cutAtResult("c0", () => advance(options()));
    else expect(await rejected(advance(options({ signal: stop.signal, onEvent })))).toBe(stop.signal.reason);
    await release(log, heldId(1), { env });
    await cutAtResult("c0", () => advance(options()));
    expect((await advance(options())).status).toBe("succeeded");
    expect(starts.map((start) => start.run)).toEqual([...(how === "a crash" ? ["once:h1@1"] : []), "once:h2@1"]);
    const results = stepsOf(log).flatMap((step) => (step["kind"] === "tool_result" ? [[step["call_id"], step["closed"] ?? step["output"]]] : []));
    expect(results).toEqual([["c1", "once:h2"], ["c0", "crashed"]]);
  },
  60_000,
);

test("releasing one hold of a group runs it while the other stays held, and the later call waits for both", async () => {
  const log = freshLog();
  const starts: Start[] = [];
  const replies = [callReply(["keep", { q: "h1" }, "c0"], ["keep", { q: "h2" }, "c1"], ["alone", { q: "L" }, "c2"]), textReply("done")];
  const hook: BeforeToolCall = (call) => (String(call.arguments["q"]).startsWith("h") ? pending() : undefined);
  const options = () =>
    gated({ env, log, runId: "r", tools: hostTools(log, starts), models: serving(new ScriptedModel(replies.slice(modelCalls(log)), CONFIG)) }, hook);
  expect((await runFlow(await flowOf(), {}, options())).status).toBe("waiting");
  await release(log, heldId(0), { env });
  expect((await advance(options())).status).toBe("waiting");
  expect(starts.map((start) => start.run)).toEqual(["keep:h1@1"]);
  expect(pendingCalls(log).map((call) => [call.callId, call.held])).toEqual([[heldId(1), true]]);
  await release(log, heldId(1), { env });
  expect((await advance(options())).status).toBe("succeeded");
  expect(starts.map((start) => start.run)).toEqual(["keep:h1@1", "keep:h2@1", "alone:L@1"]);
});

test("a released call checked again runs its validator: a refusal is the call's error result, and a throw fails the node", async () => {
  const refused = await life([["checked", "h-refuse"]]);
  expect(refused.starts).toEqual([]);
  expect(stepsOf(refused.log).find((step) => step["kind"] === "tool_result")).toMatchObject({ is_error: true, started_at: null });
  const read = refused.model.requests[0]!.messages.at(-1)!.parts[0] as ToolResultPart;
  expect([read.content, read.isError]).toEqual([expect.stringContaining("no longer allowed"), true]);
  const thrown = await life([["checked", "h-throw"]], { status: "failed" });
  expect(thrown.result.failure?.message).toContain("validator down");
  expect([thrown.starts, stepsOf(thrown.log).some((step) => step["kind"] === "tool_result")]).toEqual([[], false]);
});
