/**
 * The chat's child events, queued from child statuses, the count of the assistant's tool calls
 * in flight, and the chat sending the events into its flow's inbox: at once mid-turn, held while a
 * tool call is in flight, in a turn of their own while the prompt is up.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  CHAT_WAKE,
  type ChildEvent,
  childEvent,
  EventQueue,
  InFlight,
  toMessage,
  trackInFlight,
} from "../../src/cli/chat_events.ts";
import { ChildRuns, type ChildStatus } from "../../src/cli/chat_runs.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import { type ChatEvents, inboxChat } from "../../src/cli/inbox_chat.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import { virtualScheduler } from "../../src/platform.ts";
import { advance, send } from "../../src/run.ts";
import { PoolCommitRecord, readRunLog } from "../../src/runtime/log.ts";
import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";
import { ENTER, ESCAPE, FakeInput, FakeOutput, screen } from "./terminal_both_runtimes.ts";

const TIMEOUT = 10_000;

let tmp: string;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-events-")));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** A status of run `run` numbered `seq`, with `fields`. */
function status(run: string, seq: number, fields: Partial<ChildStatus> = {}): ChildStatus {
  return { run, status: "waiting", log: `/logs/${run}`, seq, ...fields };
}

const INBOX_WAIT = { node: "chat/heard", callId: "c", tool: "take", inbox: "messages", deadline: null, settled: false };

describe("childEvent", () => {
  test("a question names its call and quotes the child's text", () => {
    const asked = status("r1", 3, { asks: [{ call: "c1", question: 'Which "region"?\nEU?' }] });
    expect(childEvent(asked)).toEqual({ run: "r1", seq: 3, text: 'run r1 asks (call c1): "Which \\"region\\"?\\nEU?"' });
  });

  test("an inbox wait, with what the run said", () => {
    const waiting = status("r2", 4, { waits: [INBOX_WAIT], replies: ["hi", "there"] });
    expect(childEvent(waiting)?.text).toBe('run r2 waits for a message on inbox "messages"; said "hi", "there"');
  });

  test("an end: succeeded, failed or cancelled", () => {
    expect(childEvent(status("r1", 1, { status: "succeeded", output: { n: 1 } }))?.text).toBe(
      'run r1 succeeded with output {"n":1}',
    );
    const failure = { code: "code.error", message: "kaboom", node: "boom" };
    expect(childEvent(status("r1", 1, { status: "failed", failure }))?.text).toBe(
      'run r1 failed (code.error at boom): "kaboom"',
    );
    expect(childEvent(status("r1", 1, { status: "failed", note: "the log went" }))?.text).toBe(
      'run r1 failed: "the log went"',
    );
    expect(childEvent(status("r1", 1, { status: "cancelled" }))?.text).toBe("run r1 was cancelled");
  });

  test("a run running, or waiting for a timer with nothing said, is no event", () => {
    expect(childEvent(status("r1", 1, { status: "running", replies: ["old"] }))).toBeNull();
    const timer = { ...INBOX_WAIT, tool: "sleep", inbox: null };
    expect(childEvent(status("r1", 1, { waits: [timer] }))).toBeNull();
    expect(childEvent(status("r1", 1, { waits: [timer], replies: ["z"] }))?.text).toBe('run r1 said "z"');
  });

  test("a long line is cut, with a note", () => {
    const text = childEvent(status("r1", 1, { replies: ["x".repeat(3000)] }))!.text;
    expect(text).toBe(`run r1 said "${"x".repeat(2000 - 'run r1 said "'.length)}... (cut; flow_status shows all)`);
  });

  test("its message is a user message authored yian", () => {
    expect(toMessage({ run: "r1", seq: 1, text: "run r1 was cancelled" })).toEqual({
      role: "user",
      author: "yian",
      parts: [{ type: "text", text: "run r1 was cancelled" }],
    });
  });
});

describe("EventQueue", () => {
  test("a push queues the status's event and wakes; one that says nothing doesn't", () => {
    const wake = new EventTarget();
    let woken = 0;
    wake.addEventListener(CHAT_WAKE, () => void (woken += 1));
    const queue = new EventQueue({ wake });
    queue.push(status("r1", 1, { status: "running" }));
    expect([queue.size, woken]).toEqual([0, 0]);
    queue.push(status("r1", 2, { status: "cancelled" }));
    expect([queue.size, woken]).toEqual([1, 1]);
  });

  test("a newer status replaces its run's queued event in place", () => {
    const queue = new EventQueue();
    queue.push(status("r1", 1, { replies: ["a"] }));
    queue.push(status("r2", 2, { status: "cancelled" }));
    queue.push(status("r1", 3, { replies: ["a", "b"] }));
    expect(queue.takeAll().map((event) => [event.run, event.seq])).toEqual([
      ["r1", 3],
      ["r2", 2],
    ]);
    expect(queue.size).toBe(0);
  });

  test("takeAll marks each event read; markSeen drops a run's events up to a seq", () => {
    const read: [string, number][] = [];
    const queue = new EventQueue({ markRead: (run, seq) => void read.push([run, seq]) });
    queue.push(status("r1", 1, { status: "cancelled" }));
    queue.push(status("r2", 2, { status: "cancelled" }));
    queue.markSeen("r1", 1);
    queue.markSeen("r2", 1);
    expect(queue.takeAll().map((event) => event.run)).toEqual(["r2"]);
    expect(read).toEqual([["r2", 2]]);
  });

  test("requeued events go first, in order, and a newer status doesn't replace them", () => {
    const queue = new EventQueue();
    queue.push(status("r1", 1, { replies: ["a"] }));
    queue.push(status("r2", 2, { replies: ["b"] }));
    const taken = queue.takeAll();
    queue.push(status("r3", 3, { replies: ["c"] }));
    queue.requeueFront(taken);
    queue.push(status("r1", 4, { replies: ["d"] }));
    expect(queue.takeAll().map((event: ChildEvent) => event.seq)).toEqual([1, 2, 3, 4]);
  });

  test(
    "fed from a child's statuses, it queues the child's question",
    async () => {
      const queue = new EventQueue();
      const runs = new ChildRuns({
        root: tmp,
        logDir: join(tmp, "runs"),
        env: NO_ENV,
        llmConfig: {},
        scheduler: heldScheduler().scheduler,
        chatRunId: "chat-1",
        skillsFor: () => null,
        onStatus: (published) => queue.push(published),
      });
      try {
        const text = 'id: ask\nname: ask\nnodes:\n  sure: {kind: tool, tool_id: ask, input: {prompt: "Proceed?"}, output: str}\noutput: ${sure.output}\n';
        await runs.start(await loadConfinedText(tmp, join(tmp, "ask.yaml"), text, NO_ENV), {});
        while (queue.size === 0) await runs.next("r1");
        expect(queue.takeAll().map((event) => event.text)).toEqual(['run r1 asks (call chat-1-r1/sure): "Proceed?"']);
      } finally {
        await runs.dispose();
      }
    },
    TIMEOUT,
  );
});

describe("trackInFlight", () => {
  /** A context for calling a tool directly. */
  const context = () =>
    new ToolContext({
      callId: "c",
      attempt: 1,
      toolName: "t",
      runId: "r",
      node: "n",
      context: undefined,
      signal: new AbortController().signal,
    });

  /** A tool whose calls each wait for their own release, and throw when released with an error. */
  function gated() {
    const releases: ((error?: Error) => void)[] = [];
    const held = new FunctionTool({
      name: "held",
      description: "waits",
      parametersJsonSchema: { type: "object" },
      isConcurrencySafe: () => true,
      execute: () =>
        new Promise((resolve, reject) => {
          releases.push((error) => (error === undefined ? resolve("done") : reject(error)));
        }),
    });
    return { held, releases };
  }

  test("busy while any call runs, overlapping or throwing, and the last to end wakes", async () => {
    const wake = new EventTarget();
    let woken = 0;
    wake.addEventListener(CHAT_WAKE, () => void (woken += 1));
    const counter = new InFlight(wake);
    const { held, releases } = gated();
    const [tracked] = trackInFlight([held], counter);
    expect(tracked!.name).toBe("held");
    expect(tracked!.isConcurrencySafe?.({})).toBe(true);
    expect(counter.busy).toBe(false);
    const first = Promise.resolve(tracked!.execute({}, context()));
    const second = Promise.resolve(tracked!.execute({}, context()));
    expect(counter.busy).toBe(true);
    releases[0]!(new Error("boom"));
    expect(await first.catch((error: Error) => error.message)).toBe("boom");
    expect([counter.busy, woken]).toEqual([true, 0]);
    releases[1]!();
    expect(await second).toBe("done");
    expect([counter.busy, woken]).toEqual([false, 1]);
  });

  test("a call that throws at once still ends", async () => {
    const counter = new InFlight();
    const [tracked] = trackInFlight(
      [
        new FunctionTool({
          name: "bad",
          description: "throws",
          parametersJsonSchema: { type: "object" },
          execute: () => {
            throw new Error("now");
          },
        }),
      ],
      counter,
    );
    expect(await Promise.resolve(tracked!.execute({}, context())).catch((error: Error) => error.message)).toBe("now");
    expect(counter.busy).toBe(false);
  });
});

describe("the chat's flusher", () => {
  /** Each turn takes every message and replies through the `echo` tool, once that returns. */
  const CHAT = `id: events_chat
name: events_chat
input:
  messages: !Inbox(Message)
defs:
  turn:
    input:
      closed: !bool
      said: !str
      messages: !Inbox(Message)
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.messages}
          take: all
      reply:
        kind: tool
        tool_id: echo
        input:
          events: \${heard.output.events}
        output: str
    output:
      closed: \${heard.output.closed}
      said: \${reply.output}
nodes:
  chat:
    kind: loop
    call: turn
    input:
      closed: false
      said: ""
      messages: \${input.messages}
    while: not closed
output: done
`;

  let stdin: FakeInput | null = null;
  const signals = new EventEmitter();

  afterEach(() => {
    stdin?.end();
    stdin = null;
    signals.removeAllListeners();
  });

  /** A writer collecting what it is given. */
  function sink() {
    const chunks: string[] = [];
    const out: Writer = { write: async (text) => void chunks.push(text) };
    return { out, text: () => chunks.join("") };
  }

  /**
   * The `echo` tool: it says each message it got (`author: text`); the call numbered `held` (from
   * 1) waits until `release`. `started` settles when that call begins.
   */
  function echoTool(held: number | null = null) {
    const { promise: started, resolve: start } = Promise.withResolvers<void>();
    const { promise: gate, resolve: release } = Promise.withResolvers<void>();
    let calls = 0;
    const echo = new FunctionTool({
      name: "echo",
      description: "echoes the messages",
      parametersJsonSchema: { type: "object", properties: { events: { type: "array" } } },
      execute: async (args: { events?: { author?: string; parts: { text: string }[] }[] }) => {
        calls += 1;
        if (calls === held) {
          start();
          await gate;
        }
        const said = (args.events ?? []).map((m) => `${m.author === undefined ? "" : `${m.author}: `}${m.parts[0]!.text}`);
        return `echo: ${said.join(" | ")}`;
      },
    });
    return { echo, started, release };
  }

  /**
   * A chat on CHAT over a fake TTY, with child events, the `echo` tool (its call `held` held) and
   * `advanceStep` for the turns after the first.
   */
  function chat({ held = null as number | null, advanceStep = advance } = {}) {
    const tty = true;
    stdin = new FakeInput({ tty });
    const out = sink();
    const err = sink();
    const stderrStream = Object.assign(new FakeOutput(), { isTTY: tty });
    const io: CliIo = { stdout: out.out, stderr: err.out, stdin, stderrStream, env: NO_ENV };
    const wake = new EventTarget();
    let interrupts = 0;
    const events: ChatEvents = {
      queue: new EventQueue({ wake }),
      inFlight: new InFlight(wake),
      wake,
      onInterrupt: () => void (interrupts += 1),
    };
    const { echo, started, release } = echoTool(held);
    const path = join(tmp, "chat.yaml");
    writeFileSync(path, CHAT);
    const log = join(tmp, "run.jsonl");
    const done = inboxChat(path, { inputValues: [], engineTrace: false, log }, io, {
      scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")),
      signals,
      tools: { echo },
      events,
      advance: advanceStep,
    });
    const sends = () =>
      readRunLog(log).filter(
        (envelope) => envelope.record instanceof PoolCommitRecord && envelope.record.key === "__start__.messages",
      ).length;
    const lastRow = () => screen(stderrStream.text).rows.at(-1);
    return {
      stdin,
      done,
      events,
      log,
      started,
      release,
      sends,
      lastRow,
      interrupts: () => interrupts,
      stdout: out.text,
      stderr: err.text,
    };
  }

  /** Resolve once `check` holds, polling each turn of the event loop. */
  async function until(check: () => boolean): Promise<void> {
    while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
  }

  const CANCELLED = (run = "r1", seq = 5) => status(run, seq, { status: "cancelled" });

  test(
    "an event queued at the prompt wakes it: it is sent, authored yian, in a turn of its own, and the draft comes back",
    async () => {
      const c = chat();
      await until(() => c.lastRow() === "> You:");
      c.stdin.write("draft");
      await until(() => c.lastRow() === "> You: draft");
      c.events.queue.push(CANCELLED());
      await until(() => c.stdout().includes("echo: yian: run r1 was cancelled\n"));
      await until(() => c.lastRow() === "> You: draft");
      c.stdin.write(ENTER);
      await until(() => c.stdout().includes("echo: draft\n"));
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: yian: run r1 was cancelled\necho: draft\necho: \n");
      expect(c.events.queue.size).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "an event queued mid-reply with no tool call in flight is sent at once",
    async () => {
      const c = chat({ held: 1 });
      c.stdin.write("hi" + ENTER);
      await c.started;
      c.events.queue.push(CANCELLED());
      // Sent while the reply still runs: the live engine takes it.
      await until(() => c.sends() === 2);
      expect(c.stdout()).toBe("");
      c.release();
      await until(() => c.stdout().includes("run r1"));
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: hi\necho: yian: run r1 was cancelled\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "an event queued while a tool call is in flight waits for it, and one it showed is dropped",
    async () => {
      const c = chat({ held: 1 });
      c.stdin.write("hi" + ENTER);
      await c.started;
      const inFlight = c.events.inFlight as InFlight;
      inFlight.enter();
      c.events.queue.push(CANCELLED("r1", 5));
      c.events.queue.push(CANCELLED("r2", 6));
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(c.sends()).toBe(1);
      // The call returned r1's status: r1's event is seen; r2's goes once the call ends.
      c.events.queue.markSeen("r1", 5);
      inFlight.leave();
      await until(() => c.sends() === 2);
      c.release();
      await until(() => c.stdout().includes("run r2"));
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: hi\necho: yian: run r2 was cancelled\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "an event sent after the engine's loop ended has the run advanced again before the prompt",
    async () => {
      let advances = 0;
      let queue: EventQueue | null = null;
      const c = chat({
        advanceStep: async (options) => {
          const result = await advance(options);
          // Pushed once the loop has ended but before the turn does: the send is appended offline.
          if (++advances === 1) queue!.push(CANCELLED());
          return result;
        },
      });
      queue = c.events.queue;
      c.stdin.write("hi" + ENTER);
      await until(() => c.stdout().includes("run r1"));
      expect(advances).toBe(2);
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: hi\necho: yian: run r1 was cancelled\necho: \n");
      expect(advances).toBe(3);
    },
    TIMEOUT,
  );

  test(
    "Esc during a turn puts sent events back in the queue and typed lines in the prompt; they go with the next line",
    async () => {
      const c = chat({ held: 1 });
      c.stdin.write("hi" + ENTER);
      await c.started;
      c.events.queue.push(CANCELLED());
      await until(() => c.sends() === 2);
      await send(c.log, "messages", { role: "user", parts: [{ type: "text", text: "second" }] }, { env: NO_ENV });
      c.stdin.write(ESCAPE);
      await until(() => c.stderr().includes("(withdrew 1 message; back in the prompt)"));
      expect(c.stderr()).toContain("(withdrew 1 child event; sent again before your next message)\n");
      expect(c.interrupts()).toBe(1);
      c.release();
      await until(() => c.lastRow() === "> You: second");
      // Held for the next line: the prompt stays up.
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(c.stdout()).toBe("echo: hi\n");
      c.stdin.write(ENTER);
      await until(() => c.stdout().includes("second"));
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: hi\necho: yian: run r1 was cancelled | second\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "/exit prints the events still queued",
    async () => {
      const c = chat();
      await until(() => c.lastRow() === "> You:");
      // Esc at the prompt holds the events for the next line.
      c.stdin.write(ESCAPE);
      await new Promise((resolve) => setTimeout(resolve, 600));
      c.events.queue.push(CANCELLED());
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(c.sends()).toBe(0);
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stderr()).toContain("(not sent: run r1 was cancelled)\n");
      expect(c.stdout()).toBe("echo: \n");
    },
    TIMEOUT,
  );

  test(
    "Ctrl-C during a turn calls onInterrupt, and holds the events for the next line",
    async () => {
      const c = chat({ held: 1 });
      c.stdin.write("hi" + ENTER);
      await c.started;
      signals.emit("SIGINT");
      expect(c.interrupts()).toBe(1);
      c.events.queue.push(CANCELLED());
      c.release();
      await until(() => c.stdout().includes("echo: hi"));
      await new Promise((resolve) => setTimeout(resolve, 50));
      expect(c.sends()).toBe(1);
      c.stdin.write("next" + ENTER);
      await until(() => c.stdout().includes("next"));
      c.stdin.write("/exit" + ENTER);
      expect(await c.done).toBe(0);
      expect(c.stdout()).toBe("echo: hi\necho: yian: run r1 was cancelled | next\necho: \n");
    },
    TIMEOUT,
  );
});
