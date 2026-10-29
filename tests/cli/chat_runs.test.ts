/**
 * The chat's child runs: each driven in process by its own driver over a held clock, from flows
 * loaded confined to a temporary root.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { existsSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ChildRuns, type ChildRunsOptions, type ChildStatus } from "../../src/cli/chat_runs.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import { MissingKeyError } from "../../src/cli/utils.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { advance, inspectRun } from "../../src/run.ts";
import { readRunLog, type RunStartedRecord } from "../../src/runtime/log.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

const TIMEOUT = 10_000;
const CHAT = "chat-1";

let root: string;
let runs: ChildRuns | null;

beforeEach(() => {
  root = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-runs-")));
  runs = null;
});

afterEach(async () => {
  await runs?.dispose();
  rmSync(root, { recursive: true, force: true });
});

/** A `ChildRuns` over the temporary root, its logs under `<root>/runs`, on a held clock. */
function childRuns(options: Partial<ChildRunsOptions> = {}): ChildRuns {
  runs = new ChildRuns({
    root,
    logDir: join(root, "runs"),
    env: NO_ENV,
    llmConfig: {},
    scheduler: heldScheduler().scheduler,
    chatRunId: CHAT,
    skillsFor: () => null,
    ...options,
  });
  return runs;
}

/** Load `text` as the flow `<root>/<name>`, its busy nodes held by the gate file `<root>/gate`. */
function flow(text: string, name = "child.yaml"): Promise<ComposedFlow> {
  return loadConfinedText(root, join(root, name), text.replaceAll("@GATE@", join(root, "gate")), NO_ENV);
}

/** Let every busy node finish. */
function openGate(): void {
  writeFileSync(join(root, "gate"), "");
}

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

/** Child `id`'s first status numbered after `after` (default: any) that isn't `running`. */
async function settled(children: ChildRuns, id: string, after = 0): Promise<ChildStatus> {
  let status = children.status(id);
  while (status.seq <= after || status.status === "running") status = await children.next(id, status.seq);
  return status;
}

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

test(
  "a started child runs in the background and its next status is its output",
  async () => {
    const children = childRuns();
    const started = await children.start(await flow(GREET), { who: "ada" });
    expect(started).toEqual({ id: "r1", log: join(root, "runs", `${CHAT}-r1`, "log.jsonl"), seq: 1 });
    expect(children.status("r1")).toMatchObject({ run: "r1", status: "running", log: started.log });
    const status = await children.next("r1");
    expect(status).toMatchObject({ run: "r1", status: "succeeded", output: "hi ada", log: started.log });
    const inspected = inspectRun(started.log);
    expect(inspected.runId).toBe(`${CHAT}-r1`);
    expect(inspected.status).toBe("succeeded");
    const start = readRunLog(started.log)[0]!.record as RunStartedRecord;
    expect(start.parent).toMatchObject({ runId: CHAT });
  },
  TIMEOUT,
);

test(
  "each child gets the next id and a run id under the chat's",
  async () => {
    const children = childRuns();
    const greet = await flow(GREET);
    const first = await children.start(greet, { who: "a" });
    const second = await children.start(greet, { who: "b" });
    expect([first.id, second.id]).toEqual(["r1", "r2"]);
    expect((await settled(children, "r2")).output).toBe("hi b");
    expect((await settled(children, "r1")).output).toBe("hi a");
    expect(inspectRun(second.log).runId).toBe(`${CHAT}-r2`);
  },
  TIMEOUT,
);

test("the status of an id that names no child throws", () => {
  expect(() => childRuns().status("r9")).toThrow(new RangeError("no run 'r9' in this chat"));
});

/** Sleep a minute, then succeed with "rested". */
const NAP = `id: nap
name: nap
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      for: "1m"
  done:
    kind: code
    input:
      slept: \${nap.output}
    output: str
    code: |
      return "rested";
output: \${done.output}
`;

/**
 * A CODE node that runs until the test opens the gate (`openGate`), so the engine loops meanwhile;
 * at most 5 s, so a test that never opens it still ends.
 */
const BUSY_NODE = `  busy:
    kind: code
    output: str
    code: |
      const fs = await import("node:fs");
      const cap = Date.now() + 5000;
      while (!fs.existsSync("@GATE@") && Date.now() < cap) await new Promise((wake) => setTimeout(wake, 5));
      return "1m";
`;

/** Keep busy, then say "1m". */
const BUSY = `id: busy
name: busy
nodes:
${BUSY_NODE}output: \${busy.output}
`;

/** NAP for as long as a busy node says. */
const BUSY_NAP = NAP.replace("nodes:\n", `nodes:\n${BUSY_NODE}`).replace('for: "1m"', "for: ${busy.output}");

test(
  "a sleeping child wakes at its wakeAt with nothing else moving it",
  async () => {
    const { scheduler, open } = heldScheduler();
    const children = childRuns({ scheduler });
    await children.start(await flow(NAP), {});
    const waiting = await children.next("r1");
    expect(waiting).toMatchObject({ status: "waiting", wake_at: "2026-01-01T00:01:00.000Z" });
    expect(waiting.waits).toEqual([expect.objectContaining({ node: "nap", tool: "sleep" })]);
    open();
    expect(await children.next("r1")).toMatchObject({ status: "succeeded", output: "rested" });
  },
  TIMEOUT,
);

test(
  "a nudge while the engine loops moves the child once more after the pass, never alongside it",
  async () => {
    const { scheduler, open } = heldScheduler();
    let advances = 0;
    const children = childRuns({
      scheduler,
      advance: (options) => {
        advances += 1;
        return advance(options);
      },
    });
    await children.start(await flow(BUSY_NAP), {});
    children.nudge("r1");
    children.nudge("r1");
    expect(children.status("r1").status).toBe("running");
    openGate();
    expect(await settled(children, "r1")).toMatchObject({ status: "waiting" });
    // The pass the nudges asked for runs once the start's pass has ended.
    await until(() => advances === 1);
    open();
    expect(await children.next("r1")).toMatchObject({ status: "succeeded", output: "rested" });
    expect(advances).toBe(2);
  },
  TIMEOUT,
);

/** Ask whether to proceed, and say the answer. */
const ASK = `id: ask
name: ask
nodes:
  sure:
    kind: tool
    tool_id: ask
    input:
      prompt: "Proceed?"
    output: str
output: \${sure.output}
`;

/** A chat turn per message: take everything sent and reply with an echo of it, until closed. */
const ECHO = `id: echo
name: echo
input:
  messages: !Inbox(str)
defs:
  turn:
    input:
      said: !list(str)
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
      said: \${input.said + heard.output.events}
      closed: \${heard.output.closed}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      said: []
      closed: false
      messages: \${input.messages}
    while: not closed
output: \${chat.output.said}
`;

/** A CODE node that throws. */
const BOOM = `id: boom
name: boom
nodes:
  boom:
    kind: code
    output: str
    code: |
      throw new Error("kaboom");
output: \${boom.output}
`;

test(
  "a child's question waits for the answer the chat completes it with",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    const waiting = await settled(children, "r1");
    const call = `${CHAT}-r1/sure`;
    expect(waiting).toMatchObject({ status: "waiting", asks: [{ call, question: "Proceed?" }], wake_at: null });
    const answered = await children.answer("r1", call, "yes");
    expect(answered).toEqual({ run: "r1", status: "running", log: waiting.log, seq: waiting.seq + 1 });
    expect(children.status("r1")).toBe(answered);
    expect(await settled(children, "r1", waiting.seq)).toMatchObject({ status: "succeeded", output: "yes" });
  },
  TIMEOUT,
);

test(
  "an answer the run refuses is a note, not a throw",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    const waiting = await settled(children, "r1");
    const nulled = await children.answer("r1", `${CHAT}-r1/sure`, null);
    expect(nulled).toMatchObject({ status: "waiting", seq: waiting.seq });
    expect(nulled.note).toContain("an ask's answer can't be null");
    expect((await children.answer("r1", `${CHAT}-r1/other`, "yes")).note).toContain("other");
    // The stored status carries no note.
    expect(children.status("r1").note).toBeUndefined();
  },
  TIMEOUT,
);

test(
  "an inbox child's reply nodes are the replies of the pass a message starts, and a close ends it",
  async () => {
    const children = childRuns();
    await children.start(await flow(ECHO), {});
    expect(await settled(children, "r1")).toMatchObject({ status: "waiting", wake_at: null });
    expect(children.status("r1").replies).toBeUndefined();
    let seen = children.status("r1").seq;
    await children.message("r1", "messages", "hi");
    expect(await settled(children, "r1", seen)).toMatchObject({ status: "waiting", replies: ["echo: hi"] });
    // The same reply again is a new status.
    seen = children.status("r1").seq;
    children.markRead("r1", seen);
    await children.message("r1", "messages", "hi");
    expect(await settled(children, "r1", seen)).toMatchObject({ status: "waiting", replies: ["echo: hi"] });
    seen = children.status("r1").seq;
    children.markRead("r1", seen);
    await children.closeInbox("r1", "messages");
    expect(await settled(children, "r1", seen)).toMatchObject({
      status: "succeeded",
      output: ["hi", "hi"],
      replies: ["echo: "],
    });
    expect((await children.message("r1", "messages", "late")).note).toContain("ended");
    expect((await children.message("r1", "nowhere", "x")).note).toBeDefined();
  },
  TIMEOUT,
);

test(
  "a child whose node throws fails with the run's failure",
  async () => {
    const children = childRuns();
    await children.start(await flow(BOOM), {});
    const failed = await settled(children, "r1");
    expect(failed.status).toBe("failed");
    expect(failed.failure).toMatchObject({ code: expect.any(String), node: "boom" });
    expect(failed.failure!.message).toContain("kaboom");
  },
  TIMEOUT,
);

test(
  "stop cancels a resting child and a running one",
  async () => {
    const children = childRuns();
    await children.start(await flow(NAP), {});
    const waiting = await settled(children, "r1");
    expect(waiting).toMatchObject({ status: "waiting" });
    expect((await children.stop("r1")).note).toBeUndefined();
    expect(await settled(children, "r1", waiting.seq)).toMatchObject({ status: "cancelled" });
    await children.start(await flow(BUSY_NAP), {});
    await children.stop("r2");
    openGate();
    expect(await settled(children, "r2")).toMatchObject({ status: "cancelled" });
    expect(inspectRun(children.status("r2").log).status).toBe("cancelled");
    expect((await children.stop("r2")).note).toContain("ended");
  },
  TIMEOUT,
);

test(
  "aborting the signal given with a start doesn't stop the child",
  async () => {
    const children = childRuns();
    const caller = new AbortController();
    await children.start(await flow(BUSY), {}, { signal: caller.signal });
    caller.abort(new Error("the tool call ended"));
    expect(children.status("r1").status).toBe("running");
    openGate();
    expect(await settled(children, "r1")).toMatchObject({ status: "succeeded", output: "1m" });
  },
  TIMEOUT,
);

test(
  "dispose cancels running children and leaves waiting ones, saying what each waits on",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    await children.start(await flow(ECHO), {});
    await children.start(await flow(NAP), {});
    await children.start(await flow(GREET), { who: "ada" });
    for (const id of ["r1", "r2", "r3", "r4"]) await settled(children, id);
    await children.start(await flow(BUSY_NAP), {});
    const logs = ["r1", "r2", "r3", "r4", "r5"].map((id) => children.status(id).log);
    const disposing = children.dispose();
    openGate();
    expect(await disposing).toEqual([
      { run: "r1", kind: "question", log: logs[0]! },
      { run: "r2", kind: "inbox", log: logs[1]! },
      { run: "r3", kind: "timer", log: logs[2]! },
    ]);
    expect(logs.map((log) => inspectRun(log).status)).toEqual(["waiting", "waiting", "waiting", "succeeded", "cancelled"]);
    expect(children.status("r5").status).toBe("cancelled");
    await expect(children.start(await flow(GREET), { who: "x" })).rejects.toThrow("disposed");
  },
  TIMEOUT,
);

test(
  "cancelWaits resolves the next calls waiting with the current status",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    const waiting = await settled(children, "r1");
    const next = children.next("r1");
    children.cancelWaits();
    expect(await next).toBe(children.status("r1"));
    expect((await next).seq).toBe(waiting.seq);
  },
  TIMEOUT,
);

test(
  "a start the run refuses before it begins throws and leaves no run folder",
  async () => {
    const children = childRuns();
    await expect(children.start(await flow(GREET), { who: "ada", whom: "x" })).rejects.toThrow("whom");
    expect(existsSync(join(root, "runs", `${CHAT}-r1`))).toBe(false);
    expect(() => children.status("r1")).toThrow(RangeError);
    // Refused by `runFlow` itself, after the key check.
    const fake = "id: fake\nname: fake\nnodes:\n  f: {kind: tool, tool_id: fake}\noutput: ${f.output}\n";
    await expect(children.start(await flow(fake, "fake.yaml"), {})).rejects.toThrow("fake");
    expect(existsSync(join(root, "runs", `${CHAT}-r1`))).toBe(false);
    // The id goes to the next start that succeeds.
    expect((await children.start(await flow(GREET), { who: "ada" })).id).toBe("r1");
  },
  TIMEOUT,
);

const AGENT = "id: f\nname: f\nnodes:\n  a: {kind: agent, prompt: hi}\noutput: ${a.output}\n";

test(
  "a child whose provider has no key isn't started and leaves no log",
  async () => {
    const children = childRuns({ llmConfig: { provider: "openai", model: "gpt-5.5" } });
    await expect(children.start(await flow(AGENT), {})).rejects.toThrow(MissingKeyError);
    await expect(children.start(await flow(AGENT), {})).rejects.toThrow("$OPENAI_API_KEY is not set");
    expect(existsSync(join(root, "runs"))).toBe(false);
  },
  TIMEOUT,
);

test(
  "a start whose signal has already aborted throws and writes nothing",
  async () => {
    const children = childRuns();
    const caller = new AbortController();
    caller.abort(new Error("gone"));
    await expect(children.start(await flow(GREET), { who: "a" }, { signal: caller.signal })).rejects.toThrow("gone");
    expect(existsSync(join(root, "runs"))).toBe(false);
  },
  TIMEOUT,
);

/** Log a line from CODE, then say "ok". */
const CHATTY = `id: chatty
name: chatty
nodes:
  talk:
    kind: code
    output: str
    code: |
      console.log("x");
      console.error("y");
      return "ok";
output: \${talk.output}
`;

// These replace the process's console methods, so they run one at a time (a file's tests do).
describe("console", () => {
  test(
    "a child's CODE console output goes to the console file while it runs, and the console comes back after",
    async () => {
      const printed: string[] = [];
      const original = { log: console.log, error: console.error };
      const record = (...args: unknown[]) => void printed.push(args.join(" "));
      console.log = record;
      console.error = record;
      try {
        const consoleFile = join(root, "console.log");
        const children = childRuns({ consoleFile });
        await children.start(await flow(CHATTY), {});
        expect(await settled(children, "r1")).toMatchObject({ status: "succeeded", output: "ok" });
        expect(readFileSync(consoleFile, "utf8")).toBe("x\ny\n");
        expect(console.log).toBe(record);
        expect(console.error).toBe(record);
        expect(printed).toEqual([]);
      } finally {
        console.log = original.log;
        console.error = original.error;
      }
    },
    TIMEOUT,
  );

  test(
    "without a console file a child's console output stays on the console",
    async () => {
      const printed: string[] = [];
      const original = console.log;
      console.log = (...args: unknown[]) => void printed.push(args.join(" "));
      try {
        const children = childRuns();
        await children.start(await flow(CHATTY.replace('      console.error("y");\n', "")), {});
        await settled(children, "r1");
        expect(printed).toEqual(["x"]);
      } finally {
        console.log = original;
      }
    },
    TIMEOUT,
  );
});

/** BUSY_NODE, run once `node` has succeeded. */
const after = (node: string) => BUSY_NODE.replace("    output: str\n", `    input:\n      after: \${${node}.output}\n    output: str\n`);

/** Sleep a minute, then keep busy. */
const NAP_THEN_BUSY = `id: nap_busy
name: nap_busy
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      for: "1m"
${after("nap")}output: \${busy.output}
`;

/** Ask, then keep busy. */
const ASK_THEN_BUSY = ASK.replace("output: ${sure.output}\n", `${after("sure")}output: \${busy.output}\n`);

test(
  "haltAll ends a dispose held by a cancel that never settles, and reports the halted child",
  async () => {
    const children = childRuns({
      // The pass the answer starts runs until its stop; the cancel never settles.
      advance: (options) =>
        new Promise((_, reject) => {
          const signal = options.signal!;
          signal.addEventListener("abort", () => reject(signal.reason), { once: true });
        }),
      cancel: () => new Promise(() => {}),
    });
    await children.start(await flow(ASK), {});
    const asking = await settled(children, "r1");
    await children.answer("r1", asked(asking), "yes");
    expect(children.status("r1").status).toBe("running");
    let done = false;
    const disposing = children.dispose().then((left) => {
      done = true;
      return left;
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(done).toBe(false);
    children.haltAll();
    expect(await disposing).toEqual([{ run: "r1", kind: "halted", log: asking.log }]);
  },
  TIMEOUT,
);

test(
  "dispose goes by the published status: a waiting child mid wake pass is left, a just-answered one cancelled",
  async () => {
    const { scheduler, open } = heldScheduler();
    let advances = 0;
    const children = childRuns({
      scheduler,
      advance: (options) => {
        advances += 1;
        return advance(options);
      },
    });
    await children.start(await flow(NAP_THEN_BUSY), {});
    await children.start(await flow(ASK_THEN_BUSY, "ask_busy.yaml"), {});
    const asked = await settled(children, "r2");
    expect(asked.asks).toHaveLength(1);
    expect(await settled(children, "r1")).toMatchObject({ status: "waiting", wake_at: expect.any(String) });
    expect((await children.answer("r2", asked.asks![0]!.call, "yes")).status).toBe("running");
    const logs = ["r1", "r2"].map((id) => children.status(id).log);
    // r1's wake pass runs its busy node while its status still says waiting.
    open();
    await until(() => advances >= 2 && inspectRun(logs[0]!).inFlight.includes("busy"));
    expect(children.status("r1").status).toBe("waiting");
    const disposing = children.dispose();
    openGate();
    expect(await disposing).toEqual([{ run: "r1", kind: "timer", log: logs[0]! }]);
    expect(children.status("r2").status).toBe("cancelled");
    expect(inspectRun(logs[1]!).status).toBe("cancelled");
    // The halted wake pass leaves the run to resume.
    expect(inspectRun(logs[0]!).status).toBe("stopped");
  },
  TIMEOUT,
);

test(
  "replies stay on each status until marked read",
  async () => {
    const children = childRuns();
    await children.start(await flow(ECHO), {});
    let seen = (await settled(children, "r1")).seq;
    await children.message("r1", "messages", "a");
    seen = (await settled(children, "r1", seen)).seq;
    const sent = await children.message("r1", "messages", "b");
    // Unread, a's reply rides along on every status, the delivery's own included.
    expect(sent).toMatchObject({ status: "running", replies: ["echo: a"] });
    const both = await settled(children, "r1", seen);
    expect(both.replies).toEqual(["echo: a", "echo: b"]);
    children.markRead("r1", both.seq);
    expect((await children.message("r1", "messages", "c")).replies).toBeUndefined();
    expect((await settled(children, "r1", both.seq)).replies).toEqual(["echo: c"]);
  },
  TIMEOUT,
);

test(
  "a next whose signal aborts resolves with the current status and stops waiting",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    const waiting = await settled(children, "r1");
    for (let round = 0; round < 3; round++) {
      const reader = new AbortController();
      const next = children.next("r1", undefined, { signal: reader.signal });
      expect(children.waitingNexts()).toBe(1);
      reader.abort();
      expect(await next).toBe(waiting);
      expect(children.waitingNexts()).toBe(0);
    }
    const gone = new AbortController();
    gone.abort();
    expect(await children.next("r1", undefined, { signal: gone.signal })).toBe(waiting);
    expect(children.waitingNexts()).toBe(0);
  },
  TIMEOUT,
);

test(
  "onStatus sees every published status of every child, and a throw in it changes nothing",
  async () => {
    const seen: string[] = [];
    const children = childRuns({
      onStatus: (status) => {
        seen.push(`${status.run} ${status.status}`);
        throw new Error("a reader's bug");
      },
    });
    await children.start(await flow(GREET), { who: "a" });
    await children.start(await flow(ASK), {});
    expect(await settled(children, "r1")).toMatchObject({ status: "succeeded" });
    const asked = await settled(children, "r2");
    await children.answer("r2", asked.asks![0]!.call, "yes");
    expect(await settled(children, "r2", asked.seq)).toMatchObject({ status: "succeeded" });
    expect(seen.filter((line) => line.startsWith("r1"))).toEqual(["r1 succeeded"]);
    expect(seen.filter((line) => line.startsWith("r2"))).toEqual(["r2 waiting", "r2 running", "r2 succeeded"]);
  },
  TIMEOUT,
);

test(
  "a start that never begins publishes nothing",
  async () => {
    const statuses: ChildStatus[] = [];
    const children = childRuns({ onStatus: (status) => void statuses.push(status) });
    const fake = "id: fake\nname: fake\nnodes:\n  f: {kind: tool, tool_id: fake}\noutput: ${f.output}\n";
    await expect(children.start(await flow(fake, "fake.yaml"), {})).rejects.toThrow("fake");
    expect(statuses).toEqual([]);
  },
  TIMEOUT,
);

test(
  "concurrent starts take ids in call order, and one whose signal aborts while queued throws",
  async () => {
    const children = childRuns();
    const greet = await flow(GREET);
    const queued = new AbortController();
    const first = children.start(greet, { who: "a" });
    const second = children.start(greet, { who: "b" }, { signal: queued.signal });
    const third = children.start(greet, { who: "c" });
    queued.abort(new Error("gone"));
    expect((await first).id).toBe("r1");
    await expect(second).rejects.toThrow("gone");
    expect((await third).id).toBe("r2");
    expect((await settled(children, "r2")).output).toBe("hi c");
  },
  TIMEOUT,
);

test(
  "after dispose a delivery changes nothing and says the chat has ended",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    const waiting = await settled(children, "r1");
    await children.dispose();
    const answered = await children.answer("r1", asked(waiting), "yes");
    expect(answered).toEqual({ ...waiting, note: "the chat has ended" });
    expect((await children.stop("r1")).note).toBe("the chat has ended");
    expect(children.status("r1")).toBe(waiting);
    expect(inspectRun(waiting.log).status).toBe("waiting");
  },
  TIMEOUT,
);

/** The call id of `status`'s one question. */
function asked(status: ChildStatus): string {
  return status.asks![0]!.call;
}
