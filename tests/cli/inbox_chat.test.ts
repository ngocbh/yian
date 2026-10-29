/**
 * `yian chat`: a terminal host that sends each typed line to a flow's `Inbox(str)` or
 * `Inbox(Message)` input and advances the run. Driven in process on a scripted stdin (a fake TTY where keys matter), the
 * run's clock virtual and its signals a test emitter, so no model, timer or real signal is needed.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  inboxChat,
  type InboxChatDeps,
  type InboxChatHost,
  type InboxChatOptions,
  runOrChat,
  type RunOrChatOptions,
} from "../../src/cli/inbox_chat.ts";
import { main } from "../../src/cli/index.ts";
import { type CliIo, HostStop, UsageError, type Writer } from "../../src/cli/utils.ts";
import { FileLockError, lockFile, virtualScheduler } from "../../src/platform.ts";
import { cancel, inspectRun, readRunIndex, send } from "../../src/run.ts";
import {
  InboxClosedRecord,
  InboxWithdrawnRecord,
  InterruptedRecord,
  NodeStepRecord,
  PoolCommitRecord,
  readRunLog,
  RunCancelledRecord,
  RunStoppedRecord,
  RunWaitingRecord,
} from "../../src/runtime/log.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { NO_ENV, SUMMARIZER_FIRST_LINE, writeSmallBudgetChat } from "../helpers/run_state_fixtures.ts";
import { heldScheduler } from "../helpers/schedulers.ts";
import { CTRL_C, CTRL_U, ENTER, ESCAPE, FakeInput, FakeOutput, screen } from "./terminal_both_runtimes.ts";

const TIMEOUT = 10_000;
const INTERRUPTING = "(interrupting the reply; Ctrl-C again within 2 s to cancel the run)";
const STOPPING = "(stopping the reply)";

let tmp: string;
let stdin: FakeInput | undefined;
let release: (() => void) | undefined;
let signals: EventEmitter;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-chat-"));
  signals = new EventEmitter();
});

afterEach(() => {
  release?.();
  release = undefined;
  stdin?.end();
  stdin = undefined;
  signals.removeAllListeners();
  rmSync(tmp, { recursive: true, force: true });
});

/** Write `text` to `tmp/<name>` and return its path. */
function flowFile(text: string, name = "chat.yaml"): string {
  const path = join(tmp, name);
  writeFileSync(path, text);
  return path;
}

/** A writer collecting what it is given. */
function sink() {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  return { out, text: () => chunks.join("") };
}

/**
 * A fresh io over a scripted stdin (a fake TTY when `tty`), its outputs collected and marked TTYs
 * when `outTty`.
 */
function chatIo({
  tty = false,
  outTty = false,
  env = {},
}: { tty?: boolean; outTty?: boolean; env?: Record<string, string> } = {}) {
  stdin = new FakeInput({ tty });
  const out = sink();
  const err = sink();
  // A TTY chat draws on a TTY stderr; a piped one on a plain one.
  const stderrStream = new FakeOutput({ tty });
  const io: CliIo = {
    stdout: { ...out.out, isTTY: outTty },
    stderr: { ...err.out, isTTY: outTty },
    stdin,
    stderrStream,
    env: (name) => (Object.hasOwn(env, name) ? env[name] : NO_ENV(name)),
  };
  return { io, stdin, stdout: out.text, stderr: err.text };
}

/**
 * `chat` on `flow` with a fresh log, the virtual clock and the test's signals; `extra` holds test
 * seams and what a host supplies, split between the two.
 */
async function runChat(
  flow: string,
  io: CliIo,
  options: Partial<InboxChatOptions> = {},
  extra: InboxChatDeps & InboxChatHost = {},
): Promise<{ code: number; log: string }> {
  const log = join(tmp, "run.jsonl");
  const { tools, events, inputs, skills, runId, flow: loaded, ...deps } = extra;
  const code = await inboxChat(
    flow,
    { inputValues: [], engineTrace: false, log, ...options },
    io,
    { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals, ...deps },
    { tools, events, inputs, skills, runId, flow: loaded },
  );
  return { code, log };
}

/** The records of the log at `log`. */
function records(log: string) {
  return readRunLog(log).map((envelope) => envelope.record);
}

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 1));
}

/** Wait past readline's escape timeout, so a lone Escape written before arrives as the Escape key. */
function pastEscapeTimeout(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 600));
}

/** A chat turn per line: take everything sent, reply with `reply` (YAML lines, 6 spaces in). */
function turnFlow(reply: string, { input = "", nodes = "" } = {}): string {
  return `id: echo_chat
name: echo_chat
input:
  messages: !Inbox(str)
${input}defs:
  turn:
    input:
      said: !list(str)
      closed: !bool
      messages: !Inbox(str)
${input.replace(/^ {2}/gm, "      ")}    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.messages}
          take: all
${nodes}${reply}    output:
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
${input.replace(/^ {2}(\w+):.*$/gm, "      $1: ${input.$1}")}    while: not closed
output: \${chat.output.said}
`;
}

const ECHO_REPLY = `      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: str
        code: |
          return "echo: " + inputs["events"].join(" | ");
`;

const ECHO = turnFlow(ECHO_REPLY);

/** ECHO over an `Inbox(Message)`, echoing each message's text. */
const MESSAGE_ECHO = turnFlow(ECHO_REPLY.replace('inputs["events"]', 'inputs["events"].map((m) => m.parts[0].text)'))
  .replaceAll("Inbox(str)", "Inbox(Message)")
  .replace("said: !list(str)", "said: !list(Message)");

/** One turn: take a line, ask about it, reply. */
const ASK_ONCE = `id: once
name: once
input:
  messages: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.messages}
  question:
    kind: code
    input:
      events: \${heard.output.events}
    output: str
    code: |
      return "Sure about " + inputs["events"].join(" ") + "?";
  sure:
    kind: tool
    tool_id: ask
    input:
      prompt: \${question.output}
    output: str
  reply:
    kind: code
    input:
      events: \${heard.output.events}
      sure: \${sure.output}
    output: str
    code: |
      return "echo: " + inputs["events"].join(" | ") + " (" + inputs["sure"] + ")";
output: \${reply.output}
`;

/** A chat turn that sleeps 60 s after taking the lines, then replies. */
const NAP_NODES = `      delay:
        kind: code
        input:
          events: \${heard.output.events}
        output: str
        code: |
          return "60s";
      nap:
        kind: tool
        tool_id: sleep
        input:
          for: \${delay.output}
`;
const NAP_REPLY = `      reply:
        kind: code
        input:
          events: \${heard.output.events}
          napped: \${nap.output}
        output: str
        code: |
          return "echo: " + inputs["events"].join(" | ");
`;
const NAP = turnFlow(NAP_REPLY, { nodes: NAP_NODES });

describe("yian chat", () => {
  test(
    "each line is sent and answered; /exit closes the inbox and the run succeeds",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.write("hi\nthere\n/exit\n");
      const { code, log } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(0);
      // The closing turn takes nothing and still replies.
      expect(stdout()).toBe("echo: hi\necho: there\necho: \n");
      expect(stderr()).toContain(`Run log: ${log}`);
      expect(stderr()).toContain("session ended");
      const sent = records(log).filter(
        (record) => record instanceof PoolCommitRecord && record.key === "__start__.messages",
      );
      expect(sent.length).toBe(2);
      expect(records(log).filter((record) => record instanceof InboxClosedRecord).length).toBe(1);
      expect(inspectRun(log).status).toBe("succeeded");
      // Off a TTY the prompt is plain: no escape codes or carriage returns on stderr.
      expect((io.stderrStream as FakeOutput).text).toBe("> You: \n".repeat(3));
    },
    TIMEOUT,
  );

  test(
    "/quit closes the inbox too",
    async () => {
      const { io, stdin, stdout } = chatIo();
      stdin.write("hi\n/quit\n");
      const { code } = await runChat(flowFile(ECHO), io);
      expect([code, stdout()]).toEqual([0, "echo: hi\necho: \n"]);
    },
    TIMEOUT,
  );

  test(
    "the end of input is /exit",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.end("hi\n");
      const { code, log } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(0);
      expect(stdout()).toBe("echo: hi\necho: \n");
      expect(stderr()).toContain("session ended");
      expect(inspectRun(log).status).toBe("succeeded");
    },
    TIMEOUT,
  );

  test(
    "Ctrl-C at the prompt cancels the run",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ tty: true });
      stdin.write(CTRL_C);
      const { code, log } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(130);
      expect(stderr()).toContain("chat cancelled");
      expect(stderr()).not.toContain("Ctrl-C again");
      expect(stdout()).toBe("");
      expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
    },
    TIMEOUT,
  );

  test(
    "SIGINT twice within 2 s while a turn runs cancels it through the live engine",
    async () => {
      const { started, tools } = holdTool();
      const { io, stdin, stderr } = chatIo();
      stdin.write("hi\n");
      const scheduler = virtualScheduler(Date.parse("2026-01-01T00:00:00Z"));
      const chatting = runChat(flowFile(HOLD), io, {}, { tools, scheduler });
      await started;
      signals.emit("SIGINT");
      await until(() => stderr().includes(INTERRUPTING));
      await scheduler.sleep(1_000);
      signals.emit("SIGINT");
      const { code, log } = await chatting;
      expect(code).toBe(130);
      expect(stderr()).toContain("chat cancelled");
      expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
    },
    TIMEOUT,
  );

  test(
    "two SIGINTs 3 s apart during one turn both interrupt; with no agent to cut, the first records an empty Stop and the second nothing",
    async () => {
      const { started, tools } = holdTool();
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.write("hi\n");
      const scheduler = virtualScheduler(Date.parse("2026-01-01T00:00:00Z"));
      const chatting = runChat(flowFile(HOLD), io, {}, { tools, scheduler });
      await started;
      const log = join(tmp, "run.jsonl");
      signals.emit("SIGINT");
      await until(() => stderr().includes(INTERRUPTING));
      await scheduler.sleep(3_000);
      signals.emit("SIGINT");
      await until(() => stderr().split(INTERRUPTING).length - 1 === 2);
      // Give the second interrupt time to settle before the turn ends.
      await new Promise((resolve) => setTimeout(resolve, 50));
      const first = release!;
      first();
      await until(() => stdout().includes("released"));
      // The closing turn calls `hold` again; release that call too.
      stdin.write("/exit\n");
      await until(() => release !== first);
      release!();
      const { code } = await chatting;
      expect(code).toBe(0);
      expect(stderr()).not.toContain("chat cancelled");
      const stops = records(log).filter((record) => record instanceof InterruptedRecord);
      expect(stops.map((record) => [...(record as InterruptedRecord).nodes])).toEqual([[]]);
    },
    TIMEOUT,
  );

  test(
    "SIGTERM stops the run and leaves its log",
    async () => {
      const { started, tools } = holdTool();
      const { io, stdin, stderr } = chatIo();
      stdin.write("hi\n");
      const chatting = runChat(flowFile(HOLD), io, {}, { tools });
      await started;
      signals.emit("SIGTERM");
      const { code, log } = await chatting;
      expect(code).toBe(130);
      expect(stderr()).toContain(`chat stopped; the run is in ${log}`);
      expect(records(log).at(-1)).toBeInstanceOf(RunStoppedRecord);
      expect(signals.listenerCount("SIGTERM") + signals.listenerCount("SIGINT")).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "a sleep's deadline moves the run with no new line",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.write("hi\n");
      const chatting = runChat(flowFile(NAP), io);
      // The reply comes when the sleep ends, with stdin still open.
      await until(() => stdout().includes("echo: hi"));
      stdin.write("/exit\n");
      const { code, log } = await chatting;
      expect(code).toBe(0);
      expect(stdout()).toBe("echo: hi\necho: \n");
      expect(stderr()).toContain("session ended");
      expect(inspectRun(log).status).toBe("succeeded");
    },
    TIMEOUT,
  );

  test(
    "a deadline taking the prompt down keeps the half-typed line for the next prompt",
    async () => {
      const { io, stdin, stdout } = chatIo({ tty: true });
      stdin.write("hi" + ENTER + "hal");
      const chatting = runChat(flowFile(NAP), io);
      await until(() => stdout().includes("echo: hi"));
      stdin.write("f" + ENTER);
      await until(() => stdout().includes("echo: half"));
      stdin.write("/exit" + ENTER);
      const { code } = await chatting;
      expect(code).toBe(0);
      expect(stdout()).toBe("echo: hi\necho: half\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "an ask prompts at the terminal while the run moves",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ tty: true });
      stdin.write("hi" + ENTER + "yes" + ENTER);
      const { code } = await runChat(flowFile(ASK_ONCE), io);
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stdout()).toBe("echo: hi (yes)\n");
      expect((io.stderrStream as FakeOutput).text).toContain("Sure about hi?");
    },
    TIMEOUT,
  );

  test(
    "Ctrl-C at an ask prompt stops the run and leaves its log",
    async () => {
      const { io, stdin, stderr } = chatIo({ tty: true });
      stdin.write("hi" + ENTER + CTRL_C);
      const { code, log } = await runChat(flowFile(ASK_ONCE), io);
      expect(code).toBe(130);
      expect(stderr()).toContain(`chat stopped; the run is in ${log}`);
      expect(records(log).at(-1)).toBeInstanceOf(RunStoppedRecord);
    },
    TIMEOUT,
  );

  test(
    "a failing reply prints the run's diagnostic and exits 1",
    async () => {
      const reply = `      reply:
        kind: code
        input:
          events: \${heard.output.events}
        output: str
        code: |
          throw new Error("no reply today");
`;
      const { io, stdin, stderr } = chatIo();
      stdin.write("hi\n");
      const { code, log } = await runChat(flowFile(turnFlow(reply)), io);
      expect(code).toBe(1);
      expect(stderr()).toContain("no reply today");
      expect(inspectRun(log).status).toBe("failed");
    },
    TIMEOUT,
  );

  test(
    "another writer holding the run's log ends the chat: chat failed, exit 1",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.write("hi\n");
      const log = join(tmp, "run.jsonl");
      const chatting = runChat(flowFile(ECHO), io);
      await until(() => stdout().includes("echo: hi"));
      // Taken once the run waits again and the chat's `advance` has released it.
      let holder: ReturnType<typeof lockFile> | undefined;
      await until(() => {
        try {
          holder = lockFile(`${realpathSync(log)}.lock`);
          return true;
        } catch (error) {
          if (error instanceof FileLockError) return false;
          throw error;
        }
      });
      try {
        stdin.write("there\n");
        expect((await chatting).code).toBe(1);
      } finally {
        holder!.release();
      }
      expect(stderr()).toContain(`chat failed: run log ${log} is locked by another writer`);
    },
    TIMEOUT,
  );

  test(
    "a line the inbox refuses is not sent, and the chat goes on",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ env: { YIAN_MAX_EVENT_BYTES: "16" } });
      stdin.write(`${"x".repeat(40)}\nhi\n/exit\n`);
      const { code } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(0);
      expect(stderr()).toContain("not sent: ");
      expect(stdout()).toBe("echo: hi\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "a run cancelled from outside ends the chat at the next line with the run's end",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo();
      stdin.write("hi\n");
      const chatting = runChat(flowFile(ECHO), io);
      await until(() => stdout().includes("echo: hi"));
      // The second prompt is up once the turn's `advance` has returned, so the cancel lands offline.
      await until(() => (io.stderrStream as FakeOutput).text === "> You: \n".repeat(2));
      await cancel(join(tmp, "run.jsonl"), { env: io.env });
      stdin.write("there\n/exit\n");
      const { code, log } = await chatting;
      expect(code).toBe(130);
      expect(stderr()).toContain("chat cancelled");
      expect(stderr()).not.toContain("not sent: ");
      expect(stderr()).not.toContain("chat failed");
      expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
    },
    TIMEOUT,
  );
});

test(
  "a reply's escape sequences are dropped on a TTY stdout, and its control characters shown",
  async () => {
    const reply = `      reply:
        kind: code
        output: str
        code: |
          return "\\u001b[31mred\\u001b[0m\\u0007 text";
`;
    const { io, stdin, stdout } = chatIo({ outTty: true, env: { NO_COLOR: "1" } });
    stdin.write("hi\n/exit\n");
    const { code } = await runChat(flowFile(turnFlow(reply)), io);
    expect([code, stdout()]).toEqual([0, "red^G text\nred^G text\n"]);
  },
  TIMEOUT,
);

test(
  "a chat on the default log path stopped before its run starts leaves no run folder",
  async () => {
    const home = join(tmp, "home");
    mkdirSync(home);
    const { io, stderr } = chatIo({ env: { HOME: home } });
    const stop = new AbortController();
    // The stop lands as the log path is printed: after the run folder is made, before the log.
    const write = io.stderr.write;
    const stderrOut: Writer = {
      write: async (text) => {
        await write(text);
        if (text.startsWith("Run log: ")) stop.abort(new HostStop("SIGTERM"));
      },
    };
    const code = await inboxChat(
      flowFile(ECHO),
      { inputValues: [], engineTrace: false },
      { ...io, stderr: stderrOut, stop: stop.signal },
      { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals },
    );
    expect(code).toBe(130);
    expect(stderr()).toContain("Run log: ");
    const runsDir = join(home, ".yian", "runs");
    expect(readdirSync(runsDir).filter((name) => !name.startsWith("index.sqlite"))).toEqual([]);
  },
  TIMEOUT,
);

test(
  "a chat on the default log path is in the run index once it ends, before any yian status",
  async () => {
    const home = join(tmp, "home");
    mkdirSync(home);
    const { io, stdin } = chatIo({ env: { HOME: home } });
    stdin.write("hi\n/exit\n");
    const code = await inboxChat(
      flowFile(ECHO),
      { inputValues: [], engineTrace: false },
      io,
      { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals },
    );
    expect(code).toBe(0);
    const runsDir = join(home, ".yian", "runs");
    const runs = readdirSync(runsDir).filter((name) => !name.startsWith("index.sqlite"));
    expect(readRunIndex(runsDir).map((row) => [row.runId, row.status])).toEqual([[runs[0]!, "succeeded"]]);
  },
  TIMEOUT,
);

const ESCAPED = "x\x1b]52;c;aGk=\x07y\x1b]8;;https://e.example\x1b\\z\x1b[2J";
const ESCAPED_REPLY = `      reply:
        kind: code
        output: str
        code: |
          return "x\\u001b]52;c;aGk=\\u0007y\\u001b]8;;https://e.example\\u001b\\\\z\\u001b[2J";
`;

describe("yian chat's model text with escape sequences", () => {
  test(
    "a reply is stripped on a TTY stdout and kept exactly when piped",
    async () => {
      for (const [outTty, reply] of [
        [true, "xyz"],
        [false, ESCAPED],
      ] as const) {
        const { io, stdin, stdout } = chatIo({ outTty, env: { NO_COLOR: "1" } });
        stdin.write("/exit\n");
        const { code } = await runChat(flowFile(turnFlow(ESCAPED_REPLY)), io, { log: join(tmp, `${outTty}.jsonl`) });
        expect([code, stdout()]).toEqual([0, `${reply}\n`]);
      }
    },
    TIMEOUT,
  );

  test(
    "chat failed: is stripped on a TTY stderr and kept exactly when piped",
    async () => {
      // A log whose folder is a file: the run can't create it, and the error names its path.
      writeFileSync(join(tmp, "t\x1b]52;c;aGk=\x07"), "");
      const log = join(tmp, "t\x1b]52;c;aGk=\x07", "run.jsonl");
      for (const [outTty, folder] of [
        [true, "/t/"],
        [false, "/t\x1b]52;c;aGk=\x07/"],
      ] as const) {
        const { io, stderr } = chatIo({ outTty, env: { NO_COLOR: "1" } });
        const { code } = await runChat(flowFile(ECHO), io, { log });
        expect(code).toBe(1);
        const failed = stderr().slice(stderr().indexOf("chat failed: "));
        expect(failed).toStartWith("chat failed: ENOTDIR");
        expect(failed).toContain(`${folder}run.jsonl`);
      }
    },
    TIMEOUT,
  );
});

describe("yian chat's failures under --engine-trace", () => {
  test(
    "a load error is located, with the stack below only under --engine-trace",
    async () => {
      const broken = flowFile("id: b\nname: b\nnodes:\n  n: {kind: code, output: str, code: return ${nope.output}}\n");
      for (const engineTrace of [false, true]) {
        const { io, stderr } = chatIo();
        const { code } = await runChat(broken, io, { engineTrace });
        expect(code).toBe(1);
        expect(stderr()).toContain("chat.yaml:");
        expect(stderr().includes("    at ")).toBe(engineTrace);
      }
    },
    TIMEOUT,
  );

  test(
    "chat failed: adds the stack box under --engine-trace",
    async () => {
      // A log whose folder is a file: the run can't create it.
      writeFileSync(join(tmp, "taken"), "");
      const log = join(tmp, "taken", "run.jsonl");
      for (const engineTrace of [false, true]) {
        const { io, stderr } = chatIo();
        const { code } = await runChat(flowFile(ECHO), io, { engineTrace, log });
        expect(code).toBe(1);
        const failed = stderr().indexOf("chat failed: ");
        expect(failed).toBeGreaterThan(-1);
        expect(stderr().indexOf("─ traceback ") > failed).toBe(engineTrace);
        expect(stderr().includes("    at ")).toBe(engineTrace);
      }
    },
    TIMEOUT,
  );
});

/** A turn whose reply is the host tool `hold`, which waits until the test releases it. */
const HOLD = turnFlow(`      reply:
        kind: tool
        tool_id: hold
        input:
          events: \${heard.output.events}
        output: str
`);

/**
 * `HOLD` whose turn ends only after its reply, so the next take doesn't wait while `hold` runs and
 * the run isn't waiting for the next message.
 */
const HOLD_IN_TURN = HOLD.replace(
  "said: ${input.said + heard.output.events}",
  "said: ${input.said + heard.output.events + [reply.output]}",
);

/** `HOLD_IN_TURN` with a 60 s sleep before the reply. */
const NAP_HOLD = turnFlow(
  `      reply:
        kind: tool
        tool_id: hold
        input:
          events: \${heard.output.events}
          napped: \${nap.output}
        output: str
`,
  { nodes: NAP_NODES },
).replace("said: ${input.said + heard.output.events}", "said: ${input.said + heard.output.events + [reply.output]}");

/**
 * The `hold` tool: it resolves `started` when called, then waits until the test's `release` (in
 * `afterEach`) or its signal aborts, rejecting with the abort's reason.
 */
function holdTool() {
  const { promise: started, resolve: start } = Promise.withResolvers<void>();
  const hold = new FunctionTool({
    name: "hold",
    description: "waits for the test",
    parametersJsonSchema: {
      type: "object",
      properties: { events: { type: "array", items: { type: "string" } }, napped: {} },
    },
    execute: (_args, ctx: ToolContext) =>
      new Promise<string>((resolve, reject) => {
        release = () => resolve("released");
        ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
        start();
      }),
  });
  return { started, tools: { hold } };
}

test(
  "Esc after /exit prints the withdrawn text, since no prompt comes back",
  async () => {
    const { started, tools } = holdTool();
    const { io, stdin, stdout, stderr } = chatIo({ tty: true });
    const { scheduler, open } = heldScheduler();
    stdin.write("hi" + ENTER);
    const chatting = runChat(flowFile(NAP_HOLD), io, {}, { tools, scheduler });
    const log = join(tmp, "run.jsonl");
    const waits = () => records(log).filter((record) => record instanceof RunWaitingRecord).length;
    await until(() => existsSync(log) && waits() === 2);
    await send(log, "messages", "second", { env: io.env });
    stdin.write("/exit" + ENTER);
    await until(() => records(log).some((record) => record instanceof InboxClosedRecord));
    open();
    await started;
    stdin.write(ESCAPE);
    await until(() => stderr().includes(STOPPING));
    const first = release!;
    first();
    await until(() => release !== first);
    release!();
    expect((await chatting).code).toBe(0);
    expect(stderr()).toContain("(withdrew 1 message: second)\n");
    expect(stderr()).not.toContain("back in the prompt");
    expect(stdout()).toBe("released\nreleased\n");
  },
  TIMEOUT,
);

describe("the chat's inbox and inputs", () => {
  let calls = 0;

  /**
   * Chat with `flow` given `options` on a stdin that has ended: its exit code, or 2 and the message
   * of the usage error it throws, as the command line reports one.
   */
  async function cli(flow: string, options: Partial<InboxChatOptions> = {}) {
    const { io, stdin, stdout, stderr } = chatIo();
    stdin.end();
    try {
      const { code } = await runChat(flow, io, { log: join(tmp, `cli-${(calls += 1)}.jsonl`), ...options });
      return { code, stdout: stdout(), stderr: stderr() };
    } catch (error) {
      if (!(error instanceof UsageError)) throw error;
      return { code: 2, stdout: stdout(), stderr: `${stderr()}${error.message}\n` };
    }
  }

  /** A flow `f` whose inputs are `inboxes` (name to element type), each read by a take. */
  const bare = (inboxes: Record<string, string>) => {
    const names = Object.keys(inboxes);
    const input = names.map((name) => `  ${name}: !Inbox(${inboxes[name]})\n`).join("");
    const takes = names
      .map((name) => `  take_${name}:\n    kind: tool\n    tool_id: take\n    input:\n      from: \${input.${name}}\n`)
      .join("");
    const node = "  n:\n    kind: code\n    output: str\n    code: return \"x\";\n";
    return `id: f\nname: f\n${input ? `input:\n${input}` : ""}nodes:\n${node}${takes}output: \${n.output}\n`;
  };

  test(
    "a flow with no Inbox(str) or Inbox(Message) input is a usage error",
    async () => {
      const none = await cli(flowFile(bare({})));
      expect(none.code).toBe(2);
      expect(none.stderr).toContain(
        "flow 'f' has no Inbox(str) or Inbox(Message) input; the chat sends each line to one",
      );
      // An Inbox(int) isn't a candidate.
      const ints = await cli(flowFile(bare({ n: "int" })));
      expect(ints.code).toBe(2);
      expect(ints.stderr).toContain("flow 'f' has no Inbox(str) or Inbox(Message) input");
    },
    TIMEOUT,
  );

  test(
    "several Inbox(str) or Inbox(Message) inputs need --inbox, which must name one",
    async () => {
      const flow = flowFile(bare({ a: "str", b: "Message", c: "int" }));
      const several = await cli(flow);
      expect(several.code).toBe(2);
      expect(several.stderr).toContain(
        "flow 'f' has several Inbox(str) or Inbox(Message) inputs (a, b); pick one with --inbox",
      );
      const wrong = await cli(flow, { inbox: "c" });
      expect(wrong.code).toBe(2);
      expect(wrong.stderr).toContain("--inbox 'c' is not one of the flow's Inbox(str) or Inbox(Message) inputs (a, b)");
      // A chosen inbox goes on to the run. The end of input closes it, and the takes of the other
      // inboxes, which nothing can send to, leave the run waiting.
      const picked = await cli(flow, { inbox: "b" });
      expect(picked.code).toBe(0);
      expect(picked.stderr).toContain("run waiting: take_a (take), take_c (take)");
    },
    TIMEOUT,
  );

  test(
    "an Inbox(Message) input takes each line as a user message",
    async () => {
            const { io, stdin, stdout } = chatIo();
      stdin.write("hi\n/exit\n");
      const { code, log } = await runChat(flowFile(MESSAGE_ECHO), io);
      expect([code, stdout()]).toEqual([0, "echo: hi\necho: \n"]);
      const sent = records(log).filter(
        (record) => record instanceof PoolCommitRecord && record.key === "__start__.messages",
      );
      expect(sent.map((record) => (record as PoolCommitRecord).value)).toEqual([
        { kind: "object", value: { role: "user", parts: [{ type: "text", text: "hi" }] } },
      ]);
    },
    TIMEOUT,
  );

  test(
    "-i supplies another input; a missing one is a usage error off a TTY; the inbox is never asked",
    async () => {
      const topic = `      reply:
        kind: code
        input:
          topic: \${input.topic}
          events: \${heard.output.events}
        output: str
        code: |
          return inputs["topic"] + ": " + inputs["events"].join(" | ");
`;
      const path = flowFile(turnFlow(topic, { input: "  topic: !str\n" }));
      const { io, stdin, stdout } = chatIo();
      stdin.write("hi\n/exit\n");
      const { code } = await runChat(path, io, { inputValues: ["topic=x"] });
      expect([code, stdout()]).toEqual([0, "x: hi\nx: \n"]);
      const missing = await cli(path);
      expect(missing.code).toBe(2);
      expect(missing.stderr).toContain("missing required flow input: topic");
      expect(missing.stderr).not.toContain("messages");
      const inbox = await cli(path, { inputValues: ["topic=x", "messages=hi"] });
      expect(inbox.code).toBe(2);
      expect(inbox.stderr).toContain('input "messages" is an inbox; it takes messages through send, not a value');
    },
    TIMEOUT,
  );
});

describe("runOrChat", () => {
  let runs = 0;

  /**
   * `yian run`'s command on `flow` with a fresh log in `tmp` (`flow` undefined under `--resume`),
   * stdin and stderr fake TTYs when `tty`, `typed` written ahead, the chat's clock virtual and its
   * signals the test's.
   */
  async function runOn(
    flow: string | undefined,
    { tty = true, typed = "", ...options }: Partial<RunOrChatOptions> & { tty?: boolean; typed?: string } = {},
  ) {
    const { io, stdin, stdout, stderr } = chatIo({ tty });
    stdin.write(typed);
    const log = join(tmp, `run-${(runs += 1)}.jsonl`);
    const code = await runOrChat(
      flow,
      {
        inputValues: [],
        quiet: true,
        memoryStorage: [],
        verbose: false,
        engineTrace: false,
        mcpConfig: [],
        json: false,
        ...(options.resume === undefined ? { log } : {}),
        ...options,
      },
      io,
      { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals },
    );
    return { code, stdout: stdout(), stderr: stderr(), log };
  }

  /** `yian run`'s own handling of a chat flow: it waits at the take, prints nothing on stdout, exits 0. */
  const waited = (result: { code: number; stdout: string; stderr: string }) => {
    expect(result.code).toBe(0);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("run waiting: ");
    expect(result.stderr).not.toContain("session ended");
  };

  /** Lines a chat answers and then ends on; `run` leaves them unread. */
  const BYE = "hi" + ENTER + "/exit" + ENTER;

  /** ECHO with a second inbox, `other`, read by its own take. */
  const TWO_INBOXES = ECHO.replace("  messages: !Inbox(str)\n", "  messages: !Inbox(str)\n  other: !Inbox(str)\n")
    .replace(
      "nodes:\n  chat:",
      "nodes:\n  other:\n    kind: tool\n    tool_id: take\n    input:\n      from: \${input.other}\n  chat:",
    );

  test(
    "on a terminal each typed line is sent and answered; /exit ends the session",
    async () => {
      const result = await runOn(flowFile(ECHO), { typed: BYE });
      expect(result.code).toBe(0);
      expect(result.stdout).toBe("echo: hi\necho: \n");
      expect(result.stderr).toContain("session ended");
    },
    TIMEOUT,
  );

  test(
    "-i supplies the chat's other inputs",
    async () => {
      const topic = `      reply:
        kind: code
        input:
          topic: \${input.topic}
          events: \${heard.output.events}
        output: str
        code: |
          return inputs["topic"] + ": " + inputs["events"].join(" | ");
`;
      const flow = flowFile(turnFlow(topic, { input: "  topic: !str\n" }), "topic.yaml");
      const result = await runOn(flow, { inputValues: ["topic=x"], typed: BYE });
      expect([result.code, result.stdout]).toEqual([0, "x: hi\nx: \n"]);
    },
    TIMEOUT,
  );

  test(
    "off a terminal, or under --json, the run waits",
    async () => {
      const flow = flowFile(ECHO);
      waited(await runOn(flow, { tty: false, typed: BYE }));
      const json = await runOn(flow, { json: true, typed: BYE });
      expect(json.code).toBe(0);
      expect(JSON.parse(json.stdout).status).toBe("waiting");
      expect(json.stderr).not.toContain("session ended");
    },
    TIMEOUT,
  );

  test(
    "--resume on a terminal goes on with the chat, and off one recovers the run; with FLOW too it stays run's usage error",
    async () => {
      const flow = flowFile(ECHO);
      const first = await runOn(flow, { tty: false });
      waited(first);
      waited(await runOn(undefined, { resume: first.log, tty: false, typed: BYE }));
      const resumed = await runOn(undefined, { resume: first.log, typed: BYE });
      expect([resumed.code, resumed.stdout]).toEqual([0, "echo: hi\necho: \n"]);
      expect(resumed.stderr).toContain("session ended");
      await expect(runOn(flow, { resume: first.log })).rejects.toThrow(
        new UsageError("Invalid value: --resume runs the flow recorded in the log; leave out <flow>"),
      );
    },
    TIMEOUT,
  );

  test(
    "several inboxes run as before unless --inbox picks one; a name that isn't one is a usage error",
    async () => {
      const flow = flowFile(TWO_INBOXES, "two.yaml");
      waited(await runOn(flow, { typed: BYE }));
      const picked = await runOn(flow, { inbox: "messages", typed: BYE });
      expect(picked.code).toBe(0);
      expect(picked.stdout).toBe("echo: hi\necho: \n");
      // The other inbox, which nothing sends to, leaves the run waiting.
      expect(picked.stderr).toContain("run waiting: other (take)");
      await expect(runOn(flow, { inbox: "nope" })).rejects.toThrow(
        new UsageError("--inbox 'nope' is not one of the flow's Inbox(str) or Inbox(Message) inputs (messages, other)"),
      );
      // Off a terminal --inbox has no use, so it is left alone.
      waited(await runOn(flow, { tty: false, inbox: "nope" }));
    },
    TIMEOUT,
  );

  test(
    "a flow with memory, or a memory flag, runs as before",
    async () => {
      const memory = ECHO.replace("name: echo_chat\n", "name: echo_chat\nlong_term_memory: episodic\n");
      waited(await runOn(flowFile(memory, "memory.yaml"), { typed: BYE }));
      const echo = flowFile(ECHO);
      await expect(runOn(echo, { memoryStorage: ["notes=" + join(tmp, "notes")] })).rejects.toThrow(
        new UsageError(
          "Invalid value for --memory-storage: key 'notes' is not a memory declared by this flow (declared: none)",
        ),
      );
      waited(await runOn(echo, { memoryRoot: join(tmp, "memory"), typed: BYE }));
    },
    TIMEOUT,
  );

  test(
    "a flow with no Inbox(str) or Inbox(Message) input goes to run",
    async () => {
      const ints = ECHO.replaceAll("!Inbox(str)", "!Inbox(int)").replace("said: !list(str)", "said: !list(int)");
      waited(await runOn(flowFile(ints, "ints.yaml"), { typed: BYE }));
      const code = '  n: {kind: code, output: str, code: return "x";}\n';
      const plain = flowFile(`id: plain\nname: plain\nnodes:\n${code}output: \${n.output}\n`, "plain.yaml");
      const result = await runOn(plain, { typed: BYE });
      expect([result.code, result.stdout]).toEqual([0, "x\n"]);
    },
    TIMEOUT,
  );

  test(
    "an Inbox(Message) input chats too",
    async () => {
      const result = await runOn(flowFile(MESSAGE_ECHO, "message.yaml"), { typed: BYE });
      expect([result.code, result.stdout]).toEqual([0, "echo: hi\necho: \n"]);
    },
    TIMEOUT,
  );

  test(
    "a flow that needs MCP tools goes to run, which reads the MCP config",
    async () => {
      const tool = "  fetch:\n    kind: tool\n    tool_id: fake\n";
      const flow = flowFile(ECHO.replace("nodes:\n  chat:", `nodes:\n${tool}  chat:`), "mcp.yaml");
      const result = await runOn(flow, { typed: BYE });
      expect(result.code).toBe(1);
      expect(result.stdout).toBe("");
      expect(result.stderr).toContain("TOOL node 'fetch' names tool 'fake', which isn't in this run's tools");
      expect(result.stderr).not.toContain("session ended");
    },
    TIMEOUT,
  );

  test(
    "a flow that doesn't load prints its diagnostic once",
    async () => {
      const result = await runOn(flowFile(ECHO.replace("kind: loop", "kind: lop"), "bad.yaml"));
      expect(result.code).toBe(1);
      expect(result.stderr.split("unknown kind 'lop'").length - 1).toBe(1);
    },
    TIMEOUT,
  );
});

/** The chat example (an `Inbox(Message)`): one model call per turn, the history carried by bindings. */
const EXAMPLE_CHAT = fileURLToPath(new URL("../../examples/chat.yaml", import.meta.url));

/** A one-word ("7") chat-completions stream. */
const SEVEN =
  [
    { choices: [{ index: 0, delta: { role: "assistant", content: "7" }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ]
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .join("") + "data: [DONE]\n\n";

test(
  "examples/chat.yaml replies to each line through the provider",
  async () => {
    const env = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
    const { io, stdin, stdout, stderr } = chatIo({ env });
    stdin.write("hi\nhow are you?\n/exit\n");
    let calls = 0;
    const original = globalThis.fetch;
    globalThis.fetch = (async () => {
      calls += 1;
      return new Response(SEVEN, { headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    try {
      const { code, log } = await runChat(EXAMPLE_CHAT, io);
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stdout()).toBe("7\n7\n");
      expect(calls).toBe(2);
      expect(inspectRun(log).status).toBe("succeeded");
    } finally {
      globalThis.fetch = original;
    }
  },
  TIMEOUT,
);

/** A chat-completions stream of `text`, built like `SEVEN`. */
const stream = (text: string) =>
  [
    { choices: [{ index: 0, delta: { role: "assistant", content: text }, finish_reason: null }] },
    { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
  ]
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .join("") + "data: [DONE]\n\n";

/**
 * Mock `fetch` with a chat-completions stream that sends "Once upon", resolves `started`, and
 * finishes once the test's `release` is called; `restore` puts `fetch` back.
 */
function gatedStream() {
  const { promise: started, resolve: start } = Promise.withResolvers<void>();
  const { promise: gate, resolve: open } = Promise.withResolvers<void>();
  release = open;
  const chunk = (delta: Record<string, unknown>, finish: string | null) =>
    `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const counter = { calls: 0 };
  const original = globalThis.fetch;
  globalThis.fetch = (async () => {
    counter.calls += 1;
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      async start(controller) {
        controller.enqueue(encoder.encode(chunk({ role: "assistant", content: "Once upon" }, null)));
        start();
        await gate;
        controller.enqueue(encoder.encode(chunk({ content: " a time" }, null) + chunk({}, "stop") + "data: [DONE]\n\n"));
        controller.close();
      },
    });
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  }) as unknown as typeof fetch;
  return { started, counter, restore: () => (globalThis.fetch = original) };
}

const XAI = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };

test(
  "Ctrl-C while the example's reply streams interrupts it; the chat goes on to the next line",
  async () => {
    const { io, stdin, stdout, stderr } = chatIo({ env: XAI });
    stdin.write("tell me a story\n");
    const { started, counter, restore } = gatedStream();
    try {
      const chatting = runChat(EXAMPLE_CHAT, io);
      const log = join(tmp, "run.jsonl");
      await started;
      signals.emit("SIGINT");
      // The live engine takes the interrupt before the stream goes on.
      await until(() => records(log).some((record) => record instanceof InterruptedRecord));
      release!();
      await until(() => stderr().includes("(interrupted)"));
      stdin.write("/exit\n");
      const { code } = await chatting;
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stderr()).toContain(`${INTERRUPTING}\n`);
      expect(stderr().indexOf(INTERRUPTING)).toBeLessThan(stderr().indexOf("(interrupted)"));
      expect(stdout()).toBe("");
      expect(counter.calls).toBe(1);
      const interrupted = records(log).find((record) => record instanceof InterruptedRecord) as InterruptedRecord;
      expect([interrupted.inbox, interrupted.nodes]).toEqual(["owner_messages", ["chat#0/answered/reply"]]);
      const cut = records(log).find(
        (record) => record instanceof NodeStepRecord && record.step["kind"] === "model_call",
      ) as NodeStepRecord;
      expect(cut.step["finish"]).toBe("interrupted");
      const output = inspectRun(log);
      expect(output.status).toBe("succeeded");
    } finally {
      restore();
    }
  },
  TIMEOUT,
);

test(
  "Esc while the example's reply streams stops it; the chat goes on",
  async () => {
    const { io, stdin, stdout, stderr } = chatIo({ tty: true, env: XAI });
    stdin.write("tell me a story" + ENTER);
    const { started, counter, restore } = gatedStream();
    try {
      const chatting = runChat(EXAMPLE_CHAT, io);
      const log = join(tmp, "run.jsonl");
      await started;
      stdin.write(ESCAPE);
      await until(() => records(log).some((record) => record instanceof InterruptedRecord));
      release!();
      await until(() => stderr().includes("(interrupted)"));
      stdin.write("/exit" + ENTER);
      const { code } = await chatting;
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stderr()).toContain(`${STOPPING}\n`);
      expect(stderr()).not.toContain("chat cancelled");
      expect(stderr()).not.toContain(INTERRUPTING);
      expect(stderr()).not.toContain("withdrew");
      expect(stdout()).toBe("");
      expect(counter.calls).toBe(1);
      const stops = records(log).filter((record) => record instanceof InterruptedRecord) as InterruptedRecord[];
      expect(stops.map((record) => [record.inbox, [...record.nodes]])).toEqual([
        ["owner_messages", ["chat#0/answered/reply"]],
      ]);
      expect(inspectRun(log).status).toBe("succeeded");
    } finally {
      restore();
    }
  },
  TIMEOUT,
);

test(
  "Esc twice while a turn's hold tool runs records one empty Stop and never cancels",
  async () => {
    const { started, tools } = holdTool();
    const { io, stdin, stdout, stderr } = chatIo({ tty: true });
    stdin.write("hi" + ENTER);
    const chatting = runChat(flowFile(HOLD_IN_TURN), io, {}, { tools });
    await started;
    const log = join(tmp, "run.jsonl");
    stdin.write(ESCAPE);
    await until(() => stderr().includes(STOPPING));
    // Within 2 s on the chat's clock, which no Esc reads: the second is already stopping.
    stdin.write(ESCAPE);
    await pastEscapeTimeout();
    await new Promise((resolve) => setTimeout(resolve, 50));
    const first = release!;
    first();
    await until(() => stdout().includes("released"));
    stdin.write("/exit" + ENTER);
    await until(() => release !== first);
    release!();
    const { code } = await chatting;
    expect(code).toBe(0);
    expect(stderr()).not.toContain("chat cancelled");
    expect(stderr()).not.toContain(INTERRUPTING);
    expect(stderr().split(STOPPING).length - 1).toBe(1);
    const stops = records(log).filter((record) => record instanceof InterruptedRecord);
    expect(stops.map((record) => [...(record as InterruptedRecord).nodes])).toEqual([[]]);
    expect(records(log).some((record) => record instanceof RunCancelledRecord)).toBe(false);
  },
  TIMEOUT,
);

test(
  "Ctrl-C keys while a turn runs keep the SIGINT rule: the first interrupts, a second within 2 s cancels",
  async () => {
    const { started, tools } = holdTool();
    const { io, stdin, stderr } = chatIo({ tty: true });
    stdin.write("hi" + ENTER);
    const chatting = runChat(flowFile(HOLD_IN_TURN), io, {}, { tools });
    await started;
    stdin.write(CTRL_C);
    await until(() => stderr().includes(INTERRUPTING));
    stdin.write(CTRL_C);
    const { code, log } = await chatting;
    expect(code).toBe(130);
    expect(stderr()).toContain("chat cancelled");
    expect(records(log).filter((record) => record instanceof InterruptedRecord).length).toBe(1);
    expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
  },
  TIMEOUT,
);

/**
 * A chat turn that takes the lines, sleeps 60 s, then replies with an agent reading the inbox as
 * its steer inbox; the turn ends after the reply.
 */
const NAP_AGENT = `id: nap_agent
name: nap_agent
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
${NAP_NODES}      reply:
        kind: agent
        input:
          events: \${heard.output.events}
          napped: \${nap.output}
        steer_inbox: \${input.messages}
        output: Optional(str)
        prompt: "Reply to: \${events}"
    output:
      said: \${input.said + heard.output.events + [reply.output :- ""]}
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

test(
  "Esc at the prompt during a mid-turn sleep records the Stop; the reply after the sleep stops",
  async () => {
    const { io, stdin, stdout, stderr } = chatIo({ tty: true, env: XAI });
    const { scheduler, open } = heldScheduler();
    let calls = 0;
    const original = globalThis.fetch;
    globalThis.fetch = (async () => {
      calls += 1;
      return new Response(SEVEN, { headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    try {
      stdin.write("hi" + ENTER);
      const chatting = runChat(flowFile(NAP_AGENT), io, {}, { scheduler });
      const log = join(tmp, "run.jsonl");
      // The turn sleeps: the run waits again, with the prompt up.
      const waits = () => records(log).filter((record) => record instanceof RunWaitingRecord).length;
      await until(() => existsSync(log) && waits() === 2);
      stdin.write(ESCAPE);
      await until(() => stderr().includes(STOPPING));
      open();
      await until(() => stderr().includes("(interrupted)"));
      stdin.write("/exit" + ENTER);
      const { code } = await chatting;
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stderr()).not.toContain("chat cancelled");
      expect(stderr().split(STOPPING).length - 1).toBe(1);
      const stops = records(log).filter((record) => record instanceof InterruptedRecord) as InterruptedRecord[];
      expect(stops.map((record) => [...record.nodes])).toEqual([[], ["chat#0/reply"]]);
      // Stopped before its first model call; only the closing turn's reply calls the model.
      expect(calls).toBe(1);
      expect(stdout()).toBe("7\n");
    } finally {
      globalThis.fetch = original;
    }
  },
  TIMEOUT,
);

test(
  "Esc at the prompt while the chat waits at the take appends nothing and keeps the draft",
  async () => {
    const { io, stdin, stdout, stderr } = chatIo({ tty: true });
    stdin.write("hal");
    const chatting = runChat(flowFile(ECHO), io);
    await new Promise((resolve) => setTimeout(resolve, 50));
    stdin.write(ESCAPE);
    await pastEscapeTimeout();
    stdin.write("f" + ENTER);
    await until(() => stdout().includes("echo: half"));
    stdin.write("/exit" + ENTER);
    const { code, log } = await chatting;
    expect(code).toBe(0);
    expect(stdout()).toBe("echo: half\necho: \n");
    expect(stderr()).not.toContain(STOPPING);
    expect(stderr()).not.toContain("withdrew");
    const kept = records(log).filter(
      (record) => record instanceof InterruptedRecord || record instanceof InboxWithdrawnRecord,
    );
    expect(kept).toEqual([]);
  },
  TIMEOUT,
);

/**
 * A chat turn whose reply is the host tool `sender`; the turn ends after it. `T` is the inbox's
 * element type.
 */
const SENDER = (element: string) => `id: sender_chat
name: sender_chat
input:
  messages: !Inbox(${element})
defs:
  turn:
    input:
      said: !list(str)
      closed: !bool
      messages: !Inbox(${element})
    nodes:
      heard:
        kind: tool
        tool_id: take
        input:
          from: \${input.messages}
          take: all
      reply:
        kind: tool
        tool_id: sender
        input:
          events: \${heard.output.events}
        output: str
    output:
      said: \${input.said + [reply.output]}
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

for (const message of [false, true]) {
  test(
    `Esc puts a message the run hasn't read back in the prompt (Inbox(${message ? "Message" : "str"}))`,
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ tty: true });
      const log = join(tmp, "run.jsonl");
      const text = (event: unknown) => (message ? (event as { parts: { text: string }[] }).parts[0]!.text : event);
      const { promise: started, resolve: start } = Promise.withResolvers<void>();
      let calls = 0;
      // Its first call sends a line no take will read before Esc, then waits for the test.
      const sender = new FunctionTool({
        name: "sender",
        description: "sends a second line once",
        parametersJsonSchema: { type: "object", properties: { events: { type: "array" } } },
        execute: async (args: { events?: unknown[] }) => {
          const got = `got: ${(args.events ?? []).map(text).join(" ")}`;
          if (calls++ > 0) return got;
          const second = message ? { role: "user", parts: [{ type: "text", text: "second" }] } : "second";
          await send(log, "messages", second, { env: io.env });
          const { promise: held, resolve } = Promise.withResolvers<void>();
          release = resolve;
          start();
          await held;
          return got;
        },
      });
      stdin.write("hi" + ENTER);
      const chatting = runChat(flowFile(SENDER(message ? "Message" : "str")), io, {}, { tools: { sender } });
      await started;
      stdin.write(ESCAPE);
      await until(() => stderr().includes(STOPPING));
      release!();
      await until(() => stdout().includes("got: hi"));
      // The withdrawn line is the next prompt's text: ENTER sends it.
      await until(() => screen((io.stderrStream as FakeOutput).text).rows.at(-1) === "> You: second");
      stdin.write(ENTER);
      await until(() => stdout().includes("got: second"));
      stdin.write("/exit" + ENTER);
      const { code } = await chatting;
      expect(code).toBe(0);
      expect(stdout()).toBe("got: hi\ngot: second\ngot: \n");
      expect(stderr()).toContain("(withdrew 1 message; back in the prompt)\n");
      // The text is back before the interrupt, so a refused interrupt can't lose it.
      expect(stderr().indexOf("(withdrew")).toBeLessThan(stderr().indexOf(STOPPING));
      const all = readRunLog(log);
      const sends = all.filter(
        (envelope) => envelope.record instanceof PoolCommitRecord && envelope.record.key === "__start__.messages",
      );
      expect(sends.length).toBe(3);
      const withdrawn = all.flatMap((envelope) =>
        envelope.record instanceof InboxWithdrawnRecord ? [[envelope.record.inbox, [...envelope.record.events]]] : [],
      );
      expect(withdrawn).toEqual([["messages", [sends[1]!.seq]]]);
    },
    TIMEOUT,
  );
}

test(
  "Esc at the prompt puts the withdrawn text before the draft",
  async () => {
    const { io, stdin, stderr } = chatIo({ tty: true, env: XAI });
    const { scheduler, open } = heldScheduler();
    const original = globalThis.fetch;
    globalThis.fetch = (async () =>
      new Response(SEVEN, { headers: { "content-type": "text/event-stream" } })) as unknown as typeof fetch;
    try {
      stdin.write("hi" + ENTER);
      const chatting = runChat(flowFile(NAP_AGENT), io, {}, { scheduler });
      const log = join(tmp, "run.jsonl");
      const waits = () => records(log).filter((record) => record instanceof RunWaitingRecord).length;
      await until(() => existsSync(log) && waits() === 2);
      // Sent while the turn sleeps: nothing reads it before Esc.
      await send(log, "messages", "second", { env: io.env });
      stdin.write("draft");
      await new Promise((resolve) => setTimeout(resolve, 50));
      stdin.write(ESCAPE);
      await until(() => stderr().includes("(withdrew 1 message; back in the prompt)"));
      await until(() => screen((io.stderrStream as FakeOutput).text).rows.at(-1) === "> You: second draft");
      stdin.write(CTRL_U);
      open();
      await until(() => stderr().includes("(interrupted)"));
      stdin.write("/exit" + ENTER);
      expect((await chatting).code).toBe(0);
    } finally {
      globalThis.fetch = original;
    }
  },
  TIMEOUT,
);

test(
  "the summary a fold writes is never printed",
  async () => {
    const path = writeSmallBudgetChat(tmp);
    const env = { XAI_API_KEY: "test-xai-key", YIAN_DEFAULT_PROVIDER: "xai", YIAN_DEFAULT_MODEL: "grok-4.3" };
    const { io, stdin, stdout, stderr } = chatIo({ env });
    // Five turns of 51 characters (a 50-character line and the reply "7") pass 200 and keep 4,
    // so the closing turn folds once.
    stdin.write(`${["a", "b", "c", "d", "e"].map((c) => c.repeat(50)).join("\n")}\n/exit\n`);
    let calls = 0;
    const original = globalThis.fetch;
    globalThis.fetch = (async (input: Request | string | URL, init?: RequestInit) => {
      calls += 1;
      const body = input instanceof Request ? await input.text() : String(init?.body);
      const text = body.includes(SUMMARIZER_FIRST_LINE) ? "SUMMARY-TEXT" : "7";
      return new Response(stream(text), { headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    try {
      const { code, log } = await runChat(path, io);
      expect([code, stderr()]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stdout()).toBe("7\n".repeat(5));
      expect(stdout()).not.toContain("SUMMARY-TEXT");
      expect(stderr()).not.toContain("SUMMARY-TEXT");
      expect(calls).toBe(6);
      expect(inspectRun(log).status).toBe("succeeded");
    } finally {
      globalThis.fetch = original;
    }
  },
  TIMEOUT,
);

describe("a chat in its own process, signalled", () => {
  const SRC = fileURLToPath(new URL("../../src/cli/", import.meta.url));

  /**
   * Start a chat on `flow` with its log at `log` in a child on a piped stdin, as the command line
   * runs it (`yian run` hands a chat over only on a terminal), collecting its streams.
   */
  function spawnChat(flow: string, log: string) {
    const script = join(tmp, "chat-entry.ts");
    writeFileSync(
      script,
      `import { inboxChat } from ${JSON.stringify(join(SRC, "inbox_chat.ts"))};\n` +
        `import { flush, processIo } from ${JSON.stringify(join(SRC, "utils.ts"))};\n` +
        "const [flow, log] = process.argv.slice(2);\n" +
        "const io = processIo();\n" +
        "const code = await inboxChat(flow!, { inputValues: [], engineTrace: false, log }, io);\n" +
        "await flush(io);\n" +
        "process.exit(code);\n",
    );
    const runtime = process.versions.bun !== undefined ? ["--no-install", "--no-env-file"] : [];
    const child = spawn(process.execPath, [...runtime, script, flow, log], {
      cwd: tmp,
      env: { HOME: tmp, PATH: process.env.PATH ?? "" },
    });
    const streams = { out: "", err: "" };
    child.stdout.setEncoding("utf8").on("data", (text: string) => (streams.out += text));
    child.stderr.setEncoding("utf8").on("data", (text: string) => (streams.err += text));
    const closed = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) => {
      child.on("close", (code, signal) => resolve({ code, signal }));
    });
    return { child, streams, closed };
  }

  /** Resolve once `check` holds, or fail after five seconds. */
  async function soon(check: () => boolean, what: string): Promise<void> {
    for (let i = 0; i < 250 && !check(); i++) await new Promise((resolve) => setTimeout(resolve, 20));
    if (!check()) throw new Error(`timed out waiting for ${what}`);
  }

  test(
    "SIGINT at the prompt cancels the run: exit 130",
    async () => {
      const log = join(tmp, "signalled.jsonl");
      const { child, streams, closed } = spawnChat(flowFile(ECHO), log);
      try {
        child.stdin.write("hi\n");
        await soon(() => streams.out.includes("echo: hi"), "the reply");
        // The prompt comes back once the turn is over: the chat is idle.
        await soon(() => streams.err.split("You:").length - 1 >= 2, "the next prompt");
        child.kill("SIGINT");
        expect(await closed).toEqual({ code: 130, signal: null });
        expect(streams.err).not.toContain("Ctrl-C again");
        expect(streams.err).toContain("chat cancelled");
        expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
      } finally {
        child.kill("SIGKILL"); // a no-op once it has exited
      }
    },
    30_000,
  );

  test(
    "SIGTERM at the prompt ends the chat and leaves the run waiting: exit 130",
    async () => {
      const log = join(tmp, "terminated.jsonl");
      const { child, streams, closed } = spawnChat(flowFile(ECHO), log);
      try {
        // The prompt is drawn on stderr once the run waits.
        await soon(() => streams.err.includes("You:"), "the prompt");
        child.kill("SIGTERM");
        expect(await closed).toEqual({ code: 130, signal: null });
        expect(streams.err).toContain(`chat stopped; the run is in ${log}`);
        expect(inspectRun(log).status).toBe("waiting");
      } finally {
        child.kill("SIGKILL");
      }
    },
    30_000,
  );
});

describe("yian chat's skills", () => {
  // Restores the cwd and `fetch` even when a test times out, before the outer `afterEach` removes `tmp`.
  let restore: (() => void) | null = null;
  afterEach(() => {
    restore?.();
    restore = null;
  });

  /** A chat whose `reply` agent grants `code-review`. */
  const SKILLED = turnFlow(`      reply:
        kind: agent
        input:
          events: \${heard.output.events}
        skills: [code-review]
        output: str
        prompt: "Reply to: \${events}"
`);

  /** Write a skill `name` under `root`, or a broken one when `description` is null. */
  function skill(root: string, name: string, description: string | null): void {
    mkdirSync(join(root, name), { recursive: true });
    const text = description === null ? "no frontmatter\n" : `---\ndescription: ${description}\n---\nBody.\n`;
    writeFileSync(join(root, name, "SKILL.md"), text);
  }

  /** Chat on `flow` from `project` with `lines` typed and a model answering "7"; also return each system prompt. */
  async function chatIn(project: string, flow: string, lines: string) {
    const { io, stdin, stdout, stderr } = chatIo({ env: { ...XAI, HOME: join(tmp, "home") } });
    stdin.write(lines);
    const systems: string[] = [];
    const original = globalThis.fetch;
    const before = process.cwd();
    globalThis.fetch = (async (input: Request | string | URL, init?: RequestInit) => {
      const body = JSON.parse(input instanceof Request ? await input.text() : String(init?.body)) as {
        messages: { role: string; content: unknown }[];
      };
      const system = body.messages.find((message) => message.role === "system");
      systems.push(typeof system?.content === "string" ? system.content : JSON.stringify(system?.content));
      return new Response(SEVEN, { headers: { "content-type": "text/event-stream" } });
    }) as unknown as typeof fetch;
    process.chdir(project);
    restore = () => {
      process.chdir(before);
      globalThis.fetch = original;
    };
    const { code, log } = await runChat(flow, io);
    return { code, log, stdout: stdout(), stderr: stderr(), systems };
  }

  test(
    "a flow that grants skills reads the user and project skill folders once, as yian run does",
    async () => {
      const project = realpathSync(tmp);
      mkdirSync(join(project, ".git"));
      skill(join(project, ".claude", "skills"), "code-review", "Project review.");
      skill(join(project, ".claude", "skills"), "broken", null);
      skill(join(tmp, "home", ".claude", "skills"), "code-review", "Personal review.");
      const { code, log, stdout, stderr, systems } = await chatIn(project, flowFile(SKILLED), "hi\nthere\n/exit\n");
      expect([code, stderr]).toEqual([0, expect.stringContaining("session ended")]);
      expect(stdout).toBe("7\n7\n7\n");
      expect(stderr.split("warning: skipped skill: ").length - 1).toBe(1);
      expect(stderr).toContain("broken");
      expect(systems.length).toBe(3);
      for (const system of systems) expect(system).toEndWith("- code-review: Personal review.");
      expect(inspectRun(log).status).toBe("succeeded");
    },
    TIMEOUT,
  );

  test(
    "a flow that grants a skill no folder has fails before it starts",
    async () => {
      mkdirSync(join(tmp, ".git"));
      const { code, log, stderr } = await chatIn(realpathSync(tmp), flowFile(SKILLED), "/exit\n");
      expect(code).toBe(1);
      expect(stderr).toContain("grants skills not supplied to this run");
      expect(existsSync(log)).toBe(false);
    },
    TIMEOUT,
  );

  test(
    "a flow without grants reads no skill folders",
    async () => {
      const project = realpathSync(tmp);
      mkdirSync(join(project, ".git"));
      skill(join(project, ".claude", "skills"), "broken", null);
      const flow = flowFile(SKILLED.replace("        skills: [code-review]\n", ""));
      const { code, stdout, stderr } = await chatIn(project, flow, "hi\n/exit\n");
      expect([code, stdout]).toEqual([0, "7\n7\n"]);
      expect(stderr).not.toContain("skipped skill");
    },
    TIMEOUT,
  );
});

describe("the inbox chat's resume (yian run --resume on a terminal)", () => {
  /** `inboxChat` resuming `log` with the virtual clock and the test's signals. */
  function resumeChat(log: string, io: CliIo, options: Partial<InboxChatOptions> = {}, host: InboxChatHost = {}) {
    return inboxChat(
      undefined,
      { inputValues: [], engineTrace: false, resume: log, ...options },
      io,
      { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals },
      host,
    );
  }

  /** A `hold` tool that answers at once, for the resumed side of a `HOLD` chat. */
  const instantHold = {
    hold: new FunctionTool({
      name: "hold",
      description: "answers at once",
      parametersJsonSchema: { type: "object", properties: { events: { type: "array", items: { type: "string" } } } },
      execute: async () => "released",
    }),
  };

  test(
    "a chat stopped mid-reply continues: the cut reply prints, then lines go on as usual",
    async () => {
      const { started, tools } = holdTool();
      const first = chatIo();
      first.stdin.write("hi\n");
      const chatting = runChat(flowFile(HOLD), first.io, {}, { tools });
      await started;
      signals.emit("SIGTERM");
      const { code, log } = await chatting;
      expect(code).toBe(130);
      expect(first.stderr()).toContain(
        `chat stopped; the run is in ${log}\ncontinue with: yian run --resume ${log}\n`,
      );

      const second = chatIo();
      second.stdin.write("again\n/exit\n");
      expect(await resumeChat(log, second.io, {}, { tools: instantHold })).toBe(0);
      // The stopped turn's reply, then one per line; the closing turn replies too.
      expect(second.stdout()).toBe("released\nreleased\nreleased\n");
      expect(second.stderr()).toContain(`Run log: ${log}`);
      expect(second.stderr()).toContain("session ended");
      expect(inspectRun(log).status).toBe("succeeded");
    },
    TIMEOUT,
  );

  test(
    "a chat stopped at the prompt continues with no reply printed twice",
    async () => {
      const first = chatIo();
      first.stdin.write("hi\n");
      const chatting = runChat(flowFile(ECHO), first.io);
      await until(() => first.stdout().includes("echo: hi"));
      await until(() => (first.io.stderrStream as FakeOutput).text.split("You:").length - 1 >= 2);
      signals.emit("SIGTERM");
      const { code, log } = await chatting;
      expect(code).toBe(130);

      const second = chatIo();
      second.stdin.write("there\n/exit\n");
      expect(await resumeChat(log, second.io)).toBe(0);
      expect(second.stdout()).toBe("echo: there\necho: \n");
      expect(inspectRun(log).status).toBe("succeeded");
    },
    TIMEOUT,
  );

  test(
    "a resumed chat whose inbox was closed shows no prompt and waits only for deadlines",
    async () => {
      const { scheduler } = heldScheduler();
      const first = chatIo();
      first.stdin.write("hi\n/exit\n");
      const chatting = runChat(flowFile(NAP), first.io, {}, { scheduler });
      const log = join(tmp, "run.jsonl");
      await until(
        () =>
          existsSync(log) &&
          records(log).some((record) => record instanceof InboxClosedRecord) &&
          records(log).at(-1) instanceof RunWaitingRecord,
      );
      signals.emit("SIGTERM");
      expect((await chatting).code).toBe(130);

      const second = chatIo();
      expect(await resumeChat(log, second.io)).toBe(0);
      expect(second.stdout()).toBe("echo: hi\necho: \n");
      expect(second.stderr()).toContain("session ended");
      expect((second.io.stderrStream as FakeOutput).text).not.toContain("You:");
    },
    TIMEOUT,
  );

  test(
    "the printed command quotes the log path and names the inbox when the flow has several",
    async () => {
      const flow = flowFile(`id: two
name: two
input:
  messages: !Inbox(str)
  other: !Inbox(str)
nodes:
  heard:
    kind: tool
    tool_id: take
    input:
      from: \${input.messages}
  side:
    kind: tool
    tool_id: take
    input:
      from: \${input.other}
  reply:
    kind: code
    input:
      events: \${heard.output.events}
    output: str
    code: |
      return "echo: " + inputs["events"].join(" | ");
output: \${reply.output}
`);
      mkdirSync(join(tmp, "my runs"));
      const log = join(tmp, "my runs", "run.jsonl");
      const first = chatIo();
      const chatting = runChat(flow, first.io, { inbox: "messages", log });
      await until(() => existsSync(log) && records(log).at(-1) instanceof RunWaitingRecord);
      signals.emit("SIGTERM");
      expect((await chatting).code).toBe(130);
      expect(first.stderr()).toContain(`continue with: yian run --resume '${log}' --inbox messages\n`);

      const second = chatIo();
      second.stdin.write("hi\n/exit\n");
      expect(await resumeChat(log, second.io, { inbox: "messages" })).toBe(0);
      expect(second.stdout()).toBe("echo: hi\n");
    },
    TIMEOUT,
  );

  test(
    "a finished run is refused and nothing is appended",
    async () => {
      const first = chatIo();
      first.stdin.write("/exit\n");
      const { code, log } = await runChat(flowFile(ECHO), first.io);
      expect(code).toBe(0);
      const before = records(log).length;

      const second = chatIo();
      expect(await resumeChat(log, second.io)).toBe(1);
      expect(second.stderr()).toContain(`chat failed: the run in ${log} has already ended (succeeded)`);
      expect(records(log).length).toBe(before);
    },
    TIMEOUT,
  );

  test(
    "the recorded keys are checked before the run moves; with them the queued line gets its reply first",
    async () => {
      const first = chatIo({ env: XAI });
      const chatting = runChat(EXAMPLE_CHAT, first.io);
      const log = join(tmp, "run.jsonl");
      await until(() => existsSync(log) && records(log).at(-1) instanceof RunWaitingRecord);
      signals.emit("SIGTERM");
      expect((await chatting).code).toBe(130);

      const { XAI_API_KEY: _, ...keyless } = XAI;
      const second = chatIo({ env: keyless });
      second.stdin.write("hi\n");
      expect(await resumeChat(log, second.io)).toBe(1);
      expect(second.stderr()).toContain("$XAI_API_KEY is not set; set it and resume again");

      let calls = 0;
      const original = globalThis.fetch;
      globalThis.fetch = (async () => {
        calls += 1;
        return new Response(SEVEN, { headers: { "content-type": "text/event-stream" } });
      }) as unknown as typeof fetch;
      try {
        const third = chatIo({ env: XAI });
        third.stdin.write("/exit\n");
        expect(await resumeChat(log, third.io)).toBe(0);
        // "hi" was sent before the refused move; it is answered before the prompt shows.
        expect(third.stdout()).toBe("7\n");
        expect(calls).toBe(1);
        expect(inspectRun(log).status).toBe("succeeded");
      } finally {
        globalThis.fetch = original;
      }
    },
    TIMEOUT,
  );

  test(
    "yian run --resume on a terminal chats only with no FLOW or fresh flags, which stay run's usage errors",
    async () => {
      const first = chatIo();
      first.stdin.write("/exit\n");
      const { log } = await runChat(flowFile(ECHO), first.io);
      /** Run `yian run <args>` through `main` on a terminal whose stdin has ended. */
      const cli = async (args: readonly string[]) => {
        const { io, stdin, stderr } = chatIo({ tty: true });
        stdin.end();
        return { code: await main(["run", ...args], io), stderr: stderr() };
      };
      const withFlow = await cli([flowFile(ECHO, "other.yaml"), "--resume", log]);
      expect([withFlow.code, withFlow.stderr]).toEqual([2, expect.stringContaining("leave out <flow>")]);
      const fresh = await cli(["--resume", log, "-i", "x=1", "--model", "m", "--log", join(tmp, "new.jsonl")]);
      expect(fresh.code).toBe(2);
      expect(fresh.stderr).toContain("--resume cannot be combined with --input, --model, --log");
    },
    TIMEOUT,
  );
});
