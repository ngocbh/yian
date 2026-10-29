/**
 * The chat's child events, queued from child statuses, and the count of the assistant's tool
 * calls in flight.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync, realpathSync, rmSync } from "node:fs";
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
import { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { NO_ENV } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";

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
