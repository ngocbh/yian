// The chat assistant's run tools: flows in a temporary root started as child runs.

import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { EventQueue } from "../../src/cli/chat_events.ts";
import { AUTHORING_TOOLS } from "../../src/cli/authoring.ts";
import { chatRunTools, shownStatus } from "../../src/cli/chat_run_tools.ts";
import { ChildRuns } from "../../src/cli/chat_runs.ts";
import {
  type ChatToolHost,
  chatFileTools,
  stringArguments,
  ToolRefused,
} from "../../src/cli/chat_tools.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import type { ApproveRequest } from "../../src/cli/terminal.ts";
import type { Response } from "../../src/llm/messages.ts";
import { compareCodeUnits } from "../../src/diagnostics.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { isReportedToolError } from "../../src/tools/errors.ts";
import { type FunctionTool, tool } from "../../src/tools/function_tools.ts";
import type { ToolEntry } from "../../src/tools/registry.ts";
import { callReply, NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

let tmp: string;
let root: string;

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-run-tools-")));
  root = join(tmp, "root");
  mkdirSync(join(root, "sub"), { recursive: true });
  writeFileSync(join(tmp, "outside.yaml"), "id: out\n");
});

/** A host over `root` whose approvals `approve` answers. */
function host(fields: Partial<ChatToolHost> = {}): ChatToolHost {
  return {
    root,
    approve: async () => {
      throw new Error("not expected to ask");
    },
    env: NO_ENV,
    docs: new Map(),
    ...fields,
  };
}

/** What `promise` rejected with. */
async function rejection(promise: Promise<unknown>): Promise<Error> {
  try {
    await promise;
  } catch (error) {
    return error as Error;
  }
  throw new Error("expected a rejection");
}

describe("the run tools", () => {
  const TIMEOUT = 10_000;
  let runs: ChildRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  const GREET = `id: greet
name: greet
input:
  who: !str
nodes:
  hi:
    kind: code
    input:
      who: \${input.who}
    output: str
    code: |
      return "hi " + inputs["who"];
output: \${hi.output}
`;

  const ASK = `id: ask
name: ask
nodes:
  sure:
    kind: tool
    tool_id: ask
    input:
      prompt: "Which region?"
    output: str
output: \${sure.output}
`;

  const ECHO = `id: echo
name: echo
input:
  messages: !Inbox(str)
defs:
  turn:
    input:
      closed: !bool
      messages: !Inbox(str)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.messages}
          take: all
      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: str
        code: |
          return "echo: " + inputs["events"].join(" | ");
    output:
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      closed: false
      messages: \${input.messages}
    while: not closed
output: done
`;

  /** The run tools over a fresh `ChildRuns` on a held clock, approvals answered by `answer`. */
  function runTools(
    answer: string | null = "Approve",
    scheduler = heldScheduler().scheduler,
    queue: EventQueue | null = null,
  ) {
    runs = new ChildRuns({
      root,
      logDir: join(tmp, "runs"),
      env: NO_ENV,
      llmConfig: {},
      scheduler,
      chatRunId: "chat-1",
      skillsFor: () => null,
    });
    const requests: ApproveRequest[] = [];
    const seen: [string, number][] = [];
    const tools = chatRunTools(
      host({
        approve: async (request) => {
          requests.push(request);
          return answer;
        },
      }),
      runs,
      {
        seen: {
          markSeen: (run, seq) => {
            seen.push([run, seq]);
            queue?.markSeen(run, seq);
          },
          seen: (run) => queue?.seen(run) ?? 0,
        },
        scheduler,
      },
    );
    const signal = new AbortController();
    const invoke = async (name: string, args: Record<string, unknown>, abort: AbortSignal = signal.signal) => {
      const found = tools.find((each) => each.name === name)!;
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: name,
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: abort,
      });
      return found.execute(args, ctx);
    };
    const json = async (name: string, args: Record<string, unknown>, abort?: AbortSignal) =>
      JSON.parse(String(await invoke(name, args, abort))) as Record<string, unknown>;
    return { runs, tools, requests, seen, invoke, json };
  }

  /** Wait on `run` until its status leaves `running`. */
  async function settledJson(json: (name: string, args: Record<string, unknown>) => Promise<Record<string, unknown>>, run: string) {
    let status = await json("flow_status", { run });
    while (status["status"] === "running") status = await json("wait_flow", { run, seconds: 60 });
    return status;
  }

  test(
    "an approved start runs the flow; its status says where it stands and wait_flow sees it end",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, json, seen } = runTools();
      const started = await json("start_flow", { path: "greet.yaml", inputs: { who: "ada" } });
      expect(started).toMatchObject({ run: "r1", log: join(tmp, "runs", "chat-1-r1", "log.jsonl") });
      expect(started["seq"]).toBeUndefined();
      expect(requests).toHaveLength(1);
      expect(requests[0]!.title).toBe("Run greet.yaml");
      expect(requests[0]!.choices).toEqual(["Approve", "Approve this flow for this chat", "Reject"]);
      expect(requests[0]!.diff).toBeNull();
      expect(requests[0]!.body).toBe(
        'Starts greet.yaml in the background.\ninputs: {"who":"ada"}\nnode kinds: code\n' +
          "runs code on this machine (CODE nodes)\n" +
          "an approval for this chat doesn't cover changes to the files the code imports",
      );
      expect(await settledJson(json, "r1")).toMatchObject({ run: "r1", status: "succeeded", output: "hi ada" });
      expect(seen.length).toBeGreaterThan(0);
      expect(seen.every(([run]) => run === "r1")).toBe(true);
    },
    TIMEOUT,
  );

  test(
    "a draft's content starts once approved, its whole text on the card, nothing written",
    async () => {
      const { requests, json, invoke } = runTools();
      const args = { path: "sub/greet.yaml", content: GREET, inputs: { who: "ada" } };
      expect(await json("start_flow", args)).toMatchObject({ run: "r1" });
      expect(requests[0]!.title).toBe("Run sub/greet.yaml (a draft, not written)");
      const first = requests[0]!.body.split("\n")[0];
      expect(first).toBe("Starts the draft sub/greet.yaml in the background.");
      expect(requests[0]!.diff).toContain("+++ sub/greet.yaml (draft)");
      expect(requests[0]!.diff).toContain('+      return "hi " + inputs["who"];');
      const settled = await settledJson(json, "r1");
      expect(settled).toMatchObject({ status: "succeeded", output: "hi ada" });
      expect(existsSync(join(root, "sub", "greet.yaml"))).toBe(false);
      const outside = { path: "../greet.yaml", content: GREET, inputs: {} };
      const refused = await rejection(Promise.resolve(invoke("start_flow", outside)));
      expect(refused).toBeInstanceOf(ToolRefused);
      const bad = { ...args, content: "id: [\n" };
      const broken = await rejection(Promise.resolve(invoke("start_flow", bad)));
      expect(broken.message).toStartWith("invalid:\n");
    },
    TIMEOUT,
  );

  test(
    "the approval card cuts long inputs",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools("Reject");
      await invoke("start_flow", { path: "greet.yaml", inputs: { who: "x".repeat(5000) } });
      const inputs = requests[0]!.body.split("\n")[1]!;
      expect(inputs).toBe(`inputs: {"who":"${"x".repeat(4000 - '{"who":"'.length)}... (cut at 4000 characters)`);
    },
    TIMEOUT,
  );

  test(
    "the approval lists what the flow and the flows it uses can do",
    async () => {
      writeFileSync(join(root, "nap.yaml"), "id: nap\nname: nap\nnodes:\n  nap: {kind: tool, tool_id: sleep, input: {for: 1s}}\noutput: ${nap.output}\n");
      writeFileSync(
        join(root, "main.yaml"),
        "id: main\nname: main\nnodes:\n  work:\n    kind: call\n    call: nap\noutput: ${work.output}\nuses:\n  nap: ./nap\n",
      );
      const { requests, invoke } = runTools("Reject");
      expect(await invoke("start_flow", { path: "main.yaml", inputs: {} })).toBe("not started: rejected");
      expect(requests[0]!.body).toBe("Starts main.yaml in the background.\ninputs: {}\nnode kinds: call, tool\ntools it calls: sleep");
    },
    TIMEOUT,
  );

  test(
    "a rejected or unanswered start starts nothing",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const rejected = runTools("Reject");
      expect(await rejected.invoke("start_flow", { path: "greet.yaml", inputs: { who: "a" } })).toBe("not started: rejected");
      await runs!.dispose();
      const unanswered = runTools(null);
      expect(await unanswered.invoke("start_flow", { path: "greet.yaml", inputs: { who: "a" } })).toBe(
        "not started: not approved (no terminal, or the prompt ended)",
      );
      const error = await rejection(Promise.resolve(unanswered.invoke("flow_status", { run: "r1" })));
      expect(isReportedToolError(error)).toBe(true);
      expect(existsSync(join(tmp, "runs"))).toBe(false);
    },
    TIMEOUT,
  );

  test(
    "approving a flow for this chat skips its later approvals until the file changes",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, json } = runTools("Approve this flow for this chat");
      expect(await json("start_flow", { path: "greet.yaml", inputs: { who: "a" } })).toMatchObject({ run: "r1" });
      expect(await json("start_flow", { path: "./sub/../greet.yaml", inputs: { who: "b" } })).toMatchObject({ run: "r2" });
      expect(requests).toHaveLength(1);
      writeFileSync(join(root, "greet.yaml"), GREET.replace('"hi "', '"hello "'));
      expect(await json("start_flow", { path: "greet.yaml", inputs: { who: "c" } })).toMatchObject({ run: "r3" });
      expect(requests).toHaveLength(2);
    },
    TIMEOUT,
  );

  test(
    "an approval for this chat is asked again after a used flow or a referenced code file changes",
    async () => {
      writeFileSync(join(root, "child.yaml"), GREET.replace("id: greet", "id: child"));
      writeFileSync(join(root, "lib.ts"), 'export function run() { return "one"; }\n');
      writeFileSync(
        join(root, "main.yaml"),
        "id: main\nname: main\nnodes:\n  work:\n    kind: call\n    call: child\n    input:\n      who: x\n" +
          "  lib:\n    kind: code\n    output: str\n    code: ./lib.ts:run\noutput: ${work.output}\nuses:\n  child: ./child\n",
      );
      const { requests, json } = runTools("Approve this flow for this chat");
      const start = () => json("start_flow", { path: "main.yaml", inputs: {} });
      await start();
      expect(requests[0]!.body).toContain("runs code on this machine (CODE nodes; files: lib.ts)");
      await start();
      expect(requests).toHaveLength(1);
      // The used flow's inline code changes.
      writeFileSync(join(root, "child.yaml"), GREET.replace("id: greet", "id: child").replace('"hi "', '"yo "'));
      await start();
      expect(requests).toHaveLength(2);
      await start();
      expect(requests).toHaveLength(2);
      // The referenced file changes.
      writeFileSync(join(root, "lib.ts"), 'export function run() { return "two"; }\n');
      await start();
      expect(requests).toHaveLength(3);
    },
    TIMEOUT,
  );

  test(
    "inputs of the wrong type, or a required input missing, are refused before asking",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools();
      const typed = await rejection(Promise.resolve(invoke("start_flow", { path: "greet.yaml", inputs: { who: 3 } })));
      expect(isReportedToolError(typed)).toBe(true);
      expect(typed.message).toStartWith("bad inputs: ");
      expect(typed.message).toContain("who");
      const missing = await rejection(Promise.resolve(invoke("start_flow", { path: "greet.yaml", inputs: {} })));
      expect(missing.message).toStartWith("bad inputs: ");
      expect(missing.message).toContain("who");
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "inputs a flow doesn't declare, or not a record, are refused before asking",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { requests, invoke } = runTools();
      const error = await rejection(Promise.resolve(invoke("start_flow", { path: "greet.yaml", inputs: { whom: "a" } })));
      expect(isReportedToolError(error)).toBe(true);
      expect(error.message).toStartWith('bad inputs: unknown input "whom"');
      const notRecord = await rejection(Promise.resolve(invoke("start_flow", { path: "greet.yaml", inputs: [] })));
      expect(isReportedToolError(notRecord)).toBe(true);
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "a flow that doesn't load, or a path outside the root, is refused before asking",
    async () => {
      writeFileSync(join(root, "bad.yaml"), "id: [\n");
      const { requests, invoke } = runTools();
      const invalid = await rejection(Promise.resolve(invoke("start_flow", { path: "bad.yaml", inputs: {} })));
      expect(isReportedToolError(invalid)).toBe(true);
      expect(invalid.message).toStartWith("invalid:\n");
      const outside = await rejection(Promise.resolve(invoke("start_flow", { path: "../outside.yaml", inputs: {} })));
      expect(outside).toBeInstanceOf(ToolRefused);
      expect(requests).toHaveLength(0);
    },
    TIMEOUT,
  );

  test(
    "a question is relayed and answered; a null answer is an error the model reads",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { json, invoke } = runTools();
      await json("start_flow", { path: "ask.yaml", inputs: {} });
      const waiting = await settledJson(json, "r1");
      expect(waiting).toMatchObject({ status: "waiting", asks: [{ call: "chat-1-r1/sure", question: "Which region?" }] });
      const nulled = await rejection(Promise.resolve(invoke("answer_flow", { run: "r1", call: "chat-1-r1/sure", answer: null })));
      expect(isReportedToolError(nulled)).toBe(true);
      expect(await json("answer_flow", { run: "r1", call: "chat-1-r1/sure", answer: "eu" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: "eu" });
    },
    TIMEOUT,
  );

  test(
    "messages reach a child's inbox, its replies are shown once, and a close ends it",
    async () => {
      writeFileSync(join(root, "echo.yaml"), ECHO);
      const { json } = runTools();
      await json("start_flow", { path: "echo.yaml", inputs: {} });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "waiting" });
      expect(await json("message_flow", { run: "r1", inbox: "messages", text: "hi" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "waiting", replies: ["echo: hi"] });
      expect((await json("flow_status", { run: "r1" }))["replies"]).toBeUndefined();
      expect(await json("close_inbox", { run: "r1", inbox: "messages" })).toMatchObject({ status: "running" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "succeeded", output: "done" });
      expect((await json("message_flow", { run: "r1", inbox: "messages", text: "late" }))["note"]).toContain("ended");
    },
    TIMEOUT,
  );

  test(
    "stop_flow cancels a child",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { json } = runTools();
      await json("start_flow", { path: "ask.yaml", inputs: {} });
      await settledJson(json, "r1");
      await json("stop_flow", { run: "r1" });
      expect(await settledJson(json, "r1")).toMatchObject({ status: "cancelled" });
    },
    TIMEOUT,
  );

  test("an unknown run is an error the model reads, for every run tool", async () => {
    const { invoke } = runTools();
    for (const [name, args] of [
      ["flow_status", {}],
      ["wait_flow", { seconds: 1 }],
      ["answer_flow", { call: "c", answer: "x" }],
      ["message_flow", { inbox: "m", text: "x" }],
      ["close_inbox", { inbox: "m" }],
      ["stop_flow", {}],
    ] as const) {
      const error = await rejection(Promise.resolve(invoke(name, { run: "r9", ...args })));
      expect(isReportedToolError(error)).toBe(true);
      expect(error.message).toBe("no run 'r9' in this chat");
    }
  });

  test(
    "wait_flow returns at its timeout, at once for a status not yet shown, and when its call ends",
    async () => {
      writeFileSync(join(root, "ask.yaml"), ASK);
      const { scheduler, open } = heldScheduler();
      const { json, runs: children } = runTools("Approve", scheduler);
      await json("start_flow", { path: "ask.yaml", inputs: {} });
      // Not shown yet: returned at once, though the clock is held.
      while (children.status("r1").status === "running") await children.next("r1");
      const waiting = await json("wait_flow", { run: "r1", seconds: 60 });
      expect(waiting).toMatchObject({ status: "waiting" });
      // Shown: the next wait lasts until the call's signal aborts.
      const call = new AbortController();
      const ended = json("wait_flow", { run: "r1", seconds: 60 }, call.signal);
      await until(() => children.waitingNexts() === 1);
      call.abort();
      expect(await ended).toMatchObject({ status: "waiting" });
      // ... or until the chat cancels the waits (an interrupt)...
      const cancelled = json("wait_flow", { run: "r1", seconds: 60 });
      await until(() => children.waitingNexts() === 1);
      children.cancelWaits();
      expect(await cancelled).toMatchObject({ status: "waiting" });
      // ... or its timeout, capped.
      const timed = json("wait_flow", { run: "r1", seconds: 10_000 });
      await until(() => children.waitingNexts() === 1);
      const before = scheduler.now();
      open();
      expect(await timed).toMatchObject({ status: "waiting" });
      expect(scheduler.now() - before).toBe(120_000);
    },
    TIMEOUT,
  );

  test(
    "a status's long text fields are cut, with a note",
    async () => {
      writeFileSync(join(root, "long.yaml"), GREET.replace('return "hi " + inputs["who"];', 'return "x".repeat(5000);'));
      const { json } = runTools();
      await json("start_flow", { path: "long.yaml", inputs: { who: "a" } });
      const status = await settledJson(json, "r1");
      expect(status["output"]).toBe(`${"x".repeat(4000)}... (cut at 4000 characters)`);
    },
    TIMEOUT,
  );

  test(
    "a status the chat relayed as an event counts as seen: wait_flow waits past it and doesn't repeat its replies",
    async () => {
      writeFileSync(join(root, "echo.yaml"), ECHO);
      const queue = new EventQueue();
      const { json, runs: children } = runTools("Approve", heldScheduler().scheduler, queue);
      await json("start_flow", { path: "echo.yaml", inputs: {} });
      await settledJson(json, "r1");
      const asked = children.status("r1").seq;
      await json("message_flow", { run: "r1", inbox: "messages", text: "hi" });
      let relayed = children.status("r1");
      while (relayed.seq <= asked + 1 || relayed.status === "running") relayed = await children.next("r1", relayed.seq);
      queue.push(relayed);
      // The flusher sends it: the model has seen it.
      expect(queue.takeAll().map((event) => event.seq)).toEqual([relayed.seq]);
      children.markRead("r1", relayed.seq);
      const waited = json("wait_flow", { run: "r1", seconds: 60 });
      await until(() => children.waitingNexts() === 1);
      children.cancelWaits();
      const status = await waited;
      expect(status).toMatchObject({ status: "waiting" });
      expect(status["replies"]).toBeUndefined();
    },
    TIMEOUT,
  );

  test("a status lists at most 20 asks and waits, with a note", () => {
    const asks = Array.from({ length: 25 }, (_, n) => ({ call: `c${n}`, question: "q" }));
    const waits = Array.from({ length: 22 }, (_, n) => ({ node: `n${n}` }));
    const shown = shownStatus({ run: "r1", status: "waiting", asks, waits, log: "l", seq: 1 });
    expect((shown["asks"] as unknown[]).length).toBe(20);
    expect((shown["waits"] as unknown[]).length).toBe(20);
    expect(shown["not_shown"]).toBe("5 more asks and 2 more waits; the run's log has all");
  });

  test("only the reading tools may run beside other calls", () => {
    const { tools } = runTools();
    const safe = tools.filter((each) => each.isConcurrencySafe?.({}) === true).map((each) => each.name);
    expect(safe).toEqual(["flow_status", "wait_flow"]);
    expect(chatFileTools(host()).find((each) => each.name === "write_flow")!.isConcurrencySafe).toBeNull();
  });
});

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

describe("report_issue", () => {
  const TIMEOUT = 10_000;
  let runs: ChildRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  test(
    "starts the bundled report flow with no start card, the chat's facts first in its context",
    async () => {
      const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
      const report = { title: "check crashes", body: "It crashed." };
      const model = new ScriptedModel([textReply("Drafted."), textReply(JSON.stringify(report))], {
        provider: "xai",
        model: "grok-4.3",
      });
      runs = new ChildRuns({
        root,
        logDir: join(tmp, "runs"),
        env,
        llmConfig: { provider: "xai", model: "grok-4.3" },
        models: serving(model, env),
        scheduler: heldScheduler().scheduler,
        chatRunId: "chat-1",
        skillsFor: () => null,
      });
      const filed: Record<string, unknown>[] = [];
      const fileIssue = tool({
        name: "file_issue",
        description: "Take the report.",
        parameters: stringArguments(["title", "", true], ["body", "", true]),
        execute: (args) => {
          filed.push(args);
          return "not filed: saved only";
        },
      });
      const tools = chatRunTools(host({ env }), runs, {
        report: {
          facts: "yian version: 0.0.4\nchat run log: ~/c.jsonl",
          homes: ["/home/ada"],
          fileIssue,
        },
      });
      const reportIssue = tools.find((each) => each.name === "report_issue")!;
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: "report_issue",
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: new AbortController().signal,
      });
      const args = { problem: "yian check crashed", context: "ran: yian check a.yaml" };
      const started = JSON.parse(String(await reportIssue.execute(args, ctx)));
      expect(started).toMatchObject({ run: "r1" });
      let status = runs.status("r1");
      while (status.status === "running") status = await runs.next("r1", status.seq);
      expect(status).toMatchObject({ status: "succeeded", output: "not filed: saved only" });
      expect(filed).toEqual([report]);
      const prompt = model.requests[0]!.messages.at(-1)!.text();
      expect(prompt).toContain("The problem: yian check crashed");
      expect(prompt).toContain(
        "yian version: 0.0.4\nchat run log: ~/c.jsonl\nran: yian check a.yaml",
      );
      expect(runs.live()).toEqual([]);
      // Without its options, the chat's run tools have no report_issue.
      expect(chatRunTools(host(), runs).some((each) => each.name === "report_issue")).toBe(false);
    },
    TIMEOUT,
  );
});

describe("author_flow and review_flow", () => {
  const TIMEOUT = 20_000;
  const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
  const guide = new Skill({
    name: "composing-agents",
    description: "How to write flows.",
    instructions: "Write small typed nodes.",
  });
  let runs: ChildRuns | null = null;

  afterEach(async () => {
    await runs?.dispose();
    runs = null;
  });

  const GREET = `id: greet
name: greet
input:
  who: !str
nodes:
  hi:
    kind: code
    input:
      who: \${input.who}
    output: str
    code: |
      return "hi " + inputs["who"];
output: \${hi.output}
`;

  /**
   * The run tools with the authoring tools, over runs whose agents `replies` answer and whose
   * approvals `answer` answers; `requests` holds each approval asked.
   */
  function authoringTools(replies: Response[], answer: string | null = "Approve") {
    const model = new ScriptedModel(replies, { provider: "xai", model: "grok-4.3" });
    runs = new ChildRuns({
      root,
      logDir: join(tmp, "runs"),
      env,
      llmConfig: { provider: "xai", model: "grok-4.3" },
      models: serving(model, env),
      scheduler: heldScheduler().scheduler,
      chatRunId: "chat-1",
      skillsFor: () => null,
    });
    const requests: ApproveRequest[] = [];
    const approve = async (request: ApproveRequest) => {
      requests.push(request);
      return answer;
    };
    const queue = new EventQueue();
    const tools = chatRunTools(host({ env, approve, docs: new Map([["flows", "page"]]) }), runs, {
      seen: queue,
      authoring: { skills: new Map([[guide.name, guide]]) },
    });
    const invoke = async (name: string, args: Record<string, unknown>) => {
      const ctx = new ToolContext({
        callId: "chat-1/reply",
        attempt: 1,
        toolName: name,
        runId: "chat-1",
        node: "reply",
        context: undefined,
        signal: new AbortController().signal,
      });
      return tools.find((each) => each.name === name)!.execute(args, ctx);
    };
    return { runs, model, requests, invoke, queue };
  }

  /**
   * Drive child `id` to its end, answering each question it asks with the next of `answers`;
   * return its last status and the questions asked.
   */
  async function drive(children: ChildRuns, id: string, answers: string[]) {
    const asked: string[] = [];
    let status = children.status(id);
    for (;;) {
      if (status.status === "running") status = await children.next(id, status.seq);
      else if (status.status === "waiting" && (status.asks ?? []).length > 0) {
        const ask = status.asks![0]!;
        asked.push(ask.question);
        const reply = answers.shift();
        if (reply === undefined) throw new Error(`no answer left for ${ask.question}`);
        status = await children.answer(id, ask.call, reply);
      } else return { status, asked };
    }
  }

  test(
    "author_flow goes from the goal through a fixed check failure and a review to a write",
    async () => {
      const brief = {
        purpose: "Greet someone.",
        runs: "The user, daily.",
        inputs: "who",
        output: "a greeting",
        done: "It says hi.",
      };
      const contract = {
        path: "greet.yaml",
        inputs: "who: !str",
        output: "str",
        outline: ["hi: code, greets who"],
      };
      const finding = {
        severity: "low",
        area: "clarity",
        node: "flow",
        problem: "No description.",
        fix: "Add one.",
      };
      const draftArgs = { path: "greet.yaml", content: GREET, inputs: { who: "ada" } };
      const { runs, model, requests, invoke } = authoringTools([
        callReply(["ask_user", { question: "Who runs it, and how often?" }, "b1"]),
        textReply("Got it."),
        textReply(JSON.stringify(brief)),
        callReply(["ask_user", { question: "Here is the contract. Does it look right?" }, "c1"]),
        textReply("Confirmed."),
        textReply(JSON.stringify(contract)),
        textReply("id: greet\nnodes: [\n"),
        // A fenced reply is taken from inside its fence.
        textReply(`Here is the fixed flow:\n\n\`\`\`yaml\n${GREET}\`\`\`\n`),
        textReply("Reviewed."),
        textReply(JSON.stringify({ findings: [finding] })),
        callReply(["start_flow", draftArgs, "t1"]),
        callReply(["wait_flow", { run: "r2", seconds: 60 }, "t2"]),
        textReply("With who=ada it said hi ada."),
        callReply(["ask_user", { question: "It loads and said hi ada. Write it?" }, "p1"]),
        textReply("Writing."),
        textReply(JSON.stringify({ next: "write", changes: "" })),
      ]);
      const goal = "greet someone by name";
      const started = JSON.parse(String(await invoke("author_flow", { goal })));
      expect(started).toMatchObject({ run: "r1", status: "running" });
      const answers = ["Me, daily.", "Yes.", "Write it."];
      const { status, asked } = await drive(runs, "r1", answers);
      expect(status.failure ?? null).toBeNull();
      expect(status).toMatchObject({
        status: "succeeded",
        output: { path: "greet.yaml", outcome: "wrote greet.yaml" },
      });
      expect(asked).toEqual([
        "Who runs it, and how often?",
        "Here is the contract. Does it look right?",
        "It loads and said hi ada. Write it?",
      ]);
      expect(readFileSync(join(root, "greet.yaml"), "utf8")).toBe(GREET);
      // The try ran the draft before it was written, and the write asked last.
      expect(requests.map((each) => each.title)).toEqual([
        "Run greet.yaml (a draft, not written)",
        "Write greet.yaml",
      ]);
      expect(runs.status("r2")).toMatchObject({ status: "succeeded", output: "hi ada" });
      const prompts = model.requests.map((request) => request.messages.at(-1)!.text());
      // The fix saw the check's error, and the review the fixed flow.
      const fix = model.requests.find((r) => r.messages[0]!.text().startsWith("You fix a yian"));
      expect(fix!.messages.at(-1)!.text()).toContain("invalid:");
      expect(prompts.some((each) => each.startsWith("Review the flow greet.yaml:"))).toBe(true);
      const plan = model.requests.find((r) => r.messages[0]!.text().includes("finish a yian"));
      expect(plan!.messages.map((each) => each.text()).join("\n")).toContain("No description.");
      const drafting = model.requests.find((r) =>
        r.messages[0]!.text().includes("You write flows"),
      );
      expect(drafting!.tools.map((each) => each.name)).toContain("load_skill");
    },
    TIMEOUT,
  );

  test(
    "review_flow starts the review of a file with no start card, and the output is its findings",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const finding = {
        severity: "medium",
        area: "security",
        node: "hi",
        problem: "Code runs.",
        fix: "Check it.",
      };
      const { runs, requests, invoke, model } = authoringTools([
        textReply("Reviewed."),
        textReply(JSON.stringify({ findings: [finding] })),
      ]);
      const started = JSON.parse(String(await invoke("review_flow", { path: "greet.yaml" })));
      expect(started).toMatchObject({ run: "r1" });
      const { status } = await drive(runs, "r1", []);
      expect(status).toMatchObject({ status: "succeeded", output: [finding] });
      expect(requests).toEqual([]);
      expect(model.requests[0]!.messages.at(-1)!.text()).toContain('return "hi " + inputs["who"];');
      const outside = await rejection(
        Promise.resolve(invoke("review_flow", { path: "../x.yaml" })),
      );
      expect(outside).toBeInstanceOf(ToolRefused);
    },
    TIMEOUT,
  );

  test(
    "only the authoring runs get the tools, and their runs stay the assistant's to relay",
    async () => {
      writeFileSync(join(root, "greet.yaml"), GREET);
      const { runs, invoke, queue } = authoringTools([
        textReply("Reviewed."),
        textReply(JSON.stringify({ findings: [] })),
      ]);
      const names = chatRunTools(host(), runs).map((each) => each.name);
      expect(names).not.toContain("author_flow");
      expect(names).not.toContain("review_flow");
      // A flow the user wrote can't call the tools the authoring runs are granted.
      const greedy = await loadConfinedText(
        root,
        join(root, "greedy.yaml"),
        "id: g\nname: g\nnodes:\n  w: {kind: tool, tool_id: write_flow, input: {path: a.yaml, " +
          "content: x}}\noutput: ${w.output}\n",
        NO_ENV,
      );
      await expect(runs.start(greedy, {})).rejects.toThrow("write_flow");
      // The tools the review run is started with.
      let granted: Record<string, ToolEntry> = {};
      const start = runs.start.bind(runs);
      runs.start = (flow, inputs, options = {}) => {
        granted = { ...options.tools };
        return start(flow, inputs, options);
      };
      await invoke("review_flow", { path: "greet.yaml" });
      runs.start = start;
      expect(Object.keys(granted).sort(compareCodeUnits)).toEqual(
        [...AUTHORING_TOOLS].sort(compareCodeUnits),
      );
      expect((await drive(runs, "r1", [])).status).toMatchObject({ status: "succeeded" });
      const ctx = new ToolContext({
        callId: "chat-1-r1/review",
        attempt: 1,
        toolName: "wait_flow",
        runId: "chat-1-r1",
        node: "review",
        context: undefined,
        signal: new AbortController().signal,
      });
      const call = (name: string, args: Record<string, unknown>) =>
        (granted[name] as FunctionTool).execute(args, ctx);
      // Its wait_flow reads only the runs it started.
      const other = await rejection(Promise.resolve(call("wait_flow", { run: "r1", seconds: 0 })));
      expect(other.message).toBe("no run 'r1' started here");
      const tried = JSON.parse(
        String(await call("start_flow", { path: "greet.yaml", inputs: { who: "ada" } })),
      );
      expect(tried).toMatchObject({ run: "r2" });
      let status = JSON.parse(String(await call("wait_flow", { run: "r2", seconds: 60 })));
      while (status.status === "running") {
        status = JSON.parse(String(await call("wait_flow", { run: "r2", seconds: 60 })));
      }
      expect(status).toMatchObject({ status: "succeeded", output: "hi ada" });
      // What it saw isn't marked seen: the chat still tells the assistant how the run ended.
      expect(queue.seen("r2")).toBe(0);
    },
    TIMEOUT,
  );
});
