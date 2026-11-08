/**
 * The chat's child runs: each driven in process by its own driver over a held clock, from flows
 * loaded confined to a temporary root.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  realpathSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { childEvent } from "../../src/cli/chat_events.ts";
import { ChildRuns, type ChildRunsOptions, type ChildStatus } from "../../src/cli/chat_runs.ts";
import { loadConfinedText } from "../../src/cli/confined_load.ts";
import { defaultIdentity, defaultLocation, projectKey } from "../../src/cli/memory_bind.ts";
import { MissingKeyError } from "../../src/cli/utils.ts";
import type { ComposedFlow } from "../../src/compose/build/pipeline.ts";
import { loadFlow } from "../../src/compose/loader.ts";
import { advance, inspectRun, RunResult } from "../../src/run.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { readRunLog, type RunStartedRecord } from "../../src/runtime/log.ts";
import { Skill } from "../../src/skills/catalog.ts";
import { tool } from "../../src/tools/function_tools.ts";
import { NO_ENV, serving, textReply } from "../helpers/run_state_fixtures.ts";
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
    home: join(root, "home"),
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

test(
  "a child started with tools of its own can call them, and only that child",
  async () => {
    const children = childRuns();
    const shout = await flow(`id: shout
name: shout
nodes:
  loud:
    kind: tool
    tool_id: shout
    input:
      text: hi
    output: str
output: \${loud.output}
`);
    const tools = {
      shout: tool({
        name: "shout",
        description: "Shout the text.",
        parameters: {
          type: "object",
          properties: { text: { type: "string" } },
          required: ["text"],
        },
        execute: (args) => String(args["text"]).toUpperCase(),
      }),
    };
    const started = await children.start(shout, {}, { tools });
    expect((await settled(children, started.id)).output).toBe("HI");
    expect(children.start(shout, {})).rejects.toThrow("shout");
  },
  TIMEOUT,
);

test(
  "a child started with skills of its own gets them in place of the chat folder's",
  async () => {
    const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
    const model = new ScriptedModel([textReply("done")], { provider: "xai", model: "grok-4.3" });
    const children = childRuns({
      env,
      llmConfig: { provider: "xai", model: "grok-4.3" },
      models: serving(model, env),
      skillsFor: () => {
        throw new Error("not expected to read the folder's skills");
      },
    });
    const guided = await flow(`id: guided
name: guided
nodes:
  write:
    kind: agent
    skills: [guide]
    prompt: Write.
output: \${write.output}
`);
    const guide = new Skill({
      name: "guide",
      description: "How to write.",
      instructions: "Be brief.",
    });
    const started = await children.start(guided, {}, { skills: new Map([["guide", guide]]) });
    expect((await settled(children, started.id)).output).toBe("done");
    expect(model.requests[0]!.tools.map((each) => each.name)).toContain("load_skill");
    const texts = model.requests[0]!.messages.map((m) => m.text());
    expect(JSON.stringify(texts)).toContain("How to write.");
  },
  TIMEOUT,
);

test(
  "a child started after setLlmConfig runs on the layer it set",
  async () => {
    const env = (name: string) => (name === "XAI_API_KEY" ? "test-xai-key" : NO_ENV(name));
    const model = new ScriptedModel([textReply("done")], { provider: "xai", model: "grok-4.3" });
    const access = serving(model, env);
    const llmConfig = { provider: "xai", model: "grok-4.3" };
    const children = childRuns({ env, llmConfig, models: access });
    children.setLlmConfig({ provider: "xai", model: "grok-4" });
    const plain = await flow(
      "id: plain\nname: plain\nnodes:\n  say: {kind: agent, prompt: Say.}\noutput: ${say.output}\n",
    );
    const started = await children.start(plain, {});
    expect((await settled(children, started.id)).output).toBe("done");
    expect(access.configs.map((config) => config["model"])).toEqual(["grok-4"]);
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
  "a child loaded with no code root, as a bundled flow is, takes its answer too",
  async () => {
    const children = childRuns();
    await children.start(await loadFlow(ASK, { sourceLabel: "ask.yaml", env: NO_ENV }), {});
    const waiting = await settled(children, "r1");
    expect((await children.answer("r1", `${CHAT}-r1/sure`, "yes")).status).toBe("running");
    const done = await settled(children, "r1", waiting.seq);
    expect(done).toMatchObject({ status: "succeeded", output: "yes" });
  },
  TIMEOUT,
);

test(
  "live lists the children still running or waiting, by id, with the name each was started by",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {}, { name: "ask.yaml" });
    await children.start(await flow(GREET), { who: "ada" });
    await settled(children, "r2");
    const waiting = await settled(children, "r1");
    // The finished one is gone; the one waiting on its question stays.
    expect(children.live()).toEqual([{ id: "r1", name: "ask.yaml", since: expect.any(Number), status: waiting }]);
    await children.answer("r1", `${CHAT}-r1/sure`, "yes");
    await settled(children, "r1", waiting.seq);
    expect(children.live()).toEqual([]);
  },
  TIMEOUT,
);

test(
  "a child started with no name is listed by its flow's name",
  async () => {
    const children = childRuns();
    await children.start(await flow(ASK), {});
    expect(children.live().map((child) => child.name)).toEqual(["ask"]);
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

/** A flow that appends to its memory's index. */
const REMEMBER = `id: answer
name: answer
input:
  lessons: !FileMemory
nodes:
  add:
    kind: tool
    tool_id: {lessons: append}
    input: {path: MEMORY.md, text: noted}
output: \${add.output.path}
`;

/** Takes a line from its inbox, then appends to its memory's index. */
const REMEMBER_LATER = `id: answer
name: answer
input:
  lessons: !FileMemory
  messages: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.messages}
  add:
    kind: tool
    tool_id: {lessons: append}
    depends_on: [heard]
    input: {path: MEMORY.md, text: heard}
output: \${add.output.path}
`;

describe("memory", () => {
  test(
    "a child's memory input gets its default, as yian run binds it, and the chat notes it created it",
    async () => {
      const children = childRuns();
      const remember = await flow(REMEMBER);
      await children.start(remember, {}, { project: root });
      const first = await settled(children, "r1");
      expect(first.status).toBe("succeeded");
      // The flow's folder is the chat's root, so the identity is the one `yian run` gives there.
      const identity = defaultIdentity(root, "answer", "lessons");
      const location = defaultLocation(join(root, "home"), identity, "FileMemory");
      // The identity only: the chat's statuses reach its log and its model, which never hold a location.
      expect(first.memory_notes).toEqual([`memory lessons: ${identity} (new)`]);
      expect(childEvent(first)!.text).toContain(`; noted ${JSON.stringify(first.memory_notes![0])}`);
      expect(JSON.stringify(first)).not.toContain(join(root, "home"));
      // Read once, they aren't said again.
      children.markRead("r1", first.seq);
      await children.start(remember, {}, { project: root });
      const second = await settled(children, "r2");
      expect(second.status).toBe("succeeded");
      expect(second.memory_notes).toBeUndefined();
      expect(readFileSync(join(location, "MEMORY.md"), "utf8")).toBe("noted\nnoted");
    },
    TIMEOUT,
  );

  test(
    "a waiting child moves on with its memory bound again, and a drift its advance finds is noted",
    async () => {
      const passed: unknown[] = [];
      const children = childRuns({
        advance: async (options) => {
          passed.push(options.memory);
          const result = await advance(options);
          const drift = { input: "lessons", identity: "i", recorded: "v1", current: "v2" };
          return new RunResult({ ...result, memoryDrift: [drift] });
        },
      });
      await children.start(await flow(REMEMBER_LATER), {}, { project: root });
      const waiting = await settled(children, "r1");
      expect(waiting.status).toBe("waiting");
      children.markRead("r1", waiting.seq);
      await children.message("r1", "messages", "hi");
      const done = await settled(children, "r1", waiting.seq);
      expect(done.status).toBe("succeeded");
      expect(passed).toHaveLength(1);
      expect(Object.keys(passed[0] as object)).toEqual(["lessons"]);
      expect(done.memory_notes).toEqual([
        "memory `lessons` changed since this run started; the run keeps its recorded recall",
      ]);
      const location = defaultLocation(join(root, "home"), defaultIdentity(root, "answer", "lessons"), "FileMemory");
      expect(readFileSync(join(location, "MEMORY.md"), "utf8")).toBe("heard");
    },
    TIMEOUT,
  );

  test(
    "a refusal to bind or open a child's memory names no location",
    async () => {
      const home = join(root, "home");
      const memories = join(home, ".yian", "memory");
      const remember = await flow(REMEMBER);
      const location = defaultLocation(home, defaultIdentity(root, "answer", "lessons"), "FileMemory");
      // A file where the memory's folder goes: its open refuses it.
      mkdirSync(dirname(location), { recursive: true });
      writeFileSync(location, "");
      const opened = await childRuns()
        .start(remember, {}, { project: root })
        .then(() => null, (error: Error) => error.message);
      expect(opened).toContain("memory `lessons`");
      expect(opened).not.toContain(root);
      // A symlinked project folder: the bind refuses it.
      rmSync(join(memories, projectKey(root)), { recursive: true });
      mkdirSync(join(root, "elsewhere"));
      symlinkSync(join(root, "elsewhere"), join(memories, projectKey(root)));
      const bound = await childRuns()
        .start(remember, {}, { project: root })
        .then(() => null, (error: Error) => error.message);
      expect(bound).toContain(`~/.yian/memory/${projectKey(root)} is a symlink`);
      expect(bound).not.toContain(root);
    },
    TIMEOUT,
  );

  test(
    "a child given no project folder has no default memories",
    async () => {
      const children = childRuns();
      await expect(children.start(await flow(REMEMBER), {})).rejects.toThrow(
        "a flow with no file has no default memories; bind `lessons`",
      );
    },
    TIMEOUT,
  );

  test(
    "a child whose flow has no plain id isn't started",
    async () => {
      const children = childRuns();
      const unnamed = await flow(REMEMBER.replace("id: answer", 'id: "Answer questions"'));
      await expect(children.start(unnamed, {}, { project: root })).rejects.toThrow("give the flow a plain `id:`");
      expect(existsSync(join(root, "runs"))).toBe(false);
    },
    TIMEOUT,
  );
});
