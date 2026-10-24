/**
 * `yian chat`: a terminal host that sends each typed line to a flow's `Inbox(str)` input and
 * advances the run. Driven in process on a scripted stdin (a fake TTY where keys matter), the
 * run's clock virtual and its signals a test emitter, so no model, timer or real signal is needed.
 * TS-only: Python's `yian chat` blocks inside a live `human_input` and has no inboxes.
 */
import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { EventEmitter } from "node:events";
import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { chat, type ChatDeps, type ChatOptions } from "../../src/cli/chat.ts";
import { main } from "../../src/cli/index.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import { FileLockError, lockFile, virtualScheduler } from "../../src/platform.ts";
import { inspectRun } from "../../src/run.ts";
import {
  InboxClosedRecord,
  PoolCommitRecord,
  readRunLog,
  RunCancelledRecord,
  RunStoppedRecord,
} from "../../src/runtime/log.ts";
import type { ToolContext } from "../../src/tools/context.ts";
import { FunctionTool } from "../../src/tools/function_tools.ts";
import { NO_ENV } from "../engine/run_state_fixtures.ts";
import { CTRL_C, ENTER, FakeInput, FakeOutput } from "./terminal_both_runtimes.ts";

const TIMEOUT = 10_000;

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

/** A fresh io over a scripted stdin (a fake TTY when `tty`), its outputs collected. */
function chatIo({ tty = false, env = {} }: { tty?: boolean; env?: Record<string, string> } = {}) {
  stdin = new FakeInput({ tty });
  const out = sink();
  const err = sink();
  // A TTY chat draws on a TTY stderr; a piped one on a plain one.
  const stderrStream = Object.assign(new FakeOutput(), { isTTY: tty });
  const io: CliIo = {
    stdout: out.out,
    stderr: err.out,
    stdin,
    stderrStream,
    env: (name) => (Object.hasOwn(env, name) ? env[name] : NO_ENV(name)),
  };
  return { io, stdin, stdout: out.text, stderr: err.text };
}

/** `chat` on `flow` with a fresh log, the virtual clock and the test's signals. */
async function runChat(
  flow: string,
  io: CliIo,
  options: Partial<ChatOptions> = {},
  deps: ChatDeps = {},
): Promise<{ code: number; log: string }> {
  const log = join(tmp, "run.jsonl");
  const code = await chat(
    flow,
    { inputValues: [], engineTrace: false, log, ...options },
    io,
    { scheduler: virtualScheduler(Date.parse("2026-01-01T00:00:00Z")), signals, ...deps },
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
    "Ctrl-C prints a hint; a second in a row cancels the run",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ tty: true });
      stdin.write(CTRL_C + CTRL_C);
      const { code, log } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(130);
      expect(stderr()).toContain("(Ctrl-C again to cancel the run; /exit to finish it)");
      expect(stderr()).toContain("chat cancelled");
      expect(stdout()).toBe("");
      expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
    },
    TIMEOUT,
  );

  test(
    "a line sent between two Ctrl-Cs resets the count",
    async () => {
      const { io, stdin, stdout, stderr } = chatIo({ tty: true });
      stdin.write(CTRL_C + "hi" + ENTER + CTRL_C + "/exit" + ENTER);
      const { code } = await runChat(flowFile(ECHO), io);
      expect(code).toBe(0);
      expect(stderr().split("Ctrl-C again").length - 1).toBe(2);
      expect(stdout()).toBe("echo: hi\necho: \n");
    },
    TIMEOUT,
  );

  test(
    "SIGINT twice while the run moves cancels it through the live engine",
    async () => {
      const { started, tools } = holdTool();
      const { io, stdin, stderr } = chatIo();
      stdin.write("hi\n");
      const chatting = runChat(flowFile(HOLD), io, {}, { tools });
      await started;
      signals.emit("SIGINT");
      await until(() => stderr().includes("Ctrl-C again"));
      signals.emit("SIGINT");
      const { code, log } = await chatting;
      expect(code).toBe(130);
      expect(stderr()).toContain("chat cancelled");
      expect(records(log).at(-1)).toBeInstanceOf(RunCancelledRecord);
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
 * The `hold` tool: it resolves `started` when called, then waits until the test's `release` (in
 * `afterEach`) or its signal aborts, rejecting with the abort's reason.
 */
function holdTool() {
  const { promise: started, resolve: start } = Promise.withResolvers<void>();
  const hold = new FunctionTool({
    name: "hold",
    description: "waits for the test",
    parametersJsonSchema: { type: "object", properties: { events: { type: "array", items: { type: "string" } } } },
    execute: (_args, ctx: ToolContext) =>
      new Promise<string>((resolve, reject) => {
        release = () => resolve("released");
        ctx.signal.addEventListener("abort", () => reject(ctx.signal.reason), { once: true });
        start();
      }),
  });
  return { started, tools: { hold } };
}

describe("yian chat's inbox and inputs", () => {
  /** Run `yian chat <args>` through `main` on a stdin that has ended. */
  async function cli(args: readonly string[]) {
    const { io, stdin, stdout, stderr } = chatIo();
    stdin.end();
    const code = await main(["chat", ...args, "--log", join(tmp, "cli.jsonl")], io);
    return { code, stdout: stdout(), stderr: stderr() };
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
    "a flow with no Inbox(str) input is a usage error",
    async () => {
      const none = await cli([flowFile(bare({}))]);
      expect(none.code).toBe(2);
      expect(none.stderr).toContain("flow 'f' has no Inbox(str) input; yian chat sends each line to one");
      // An Inbox(int) isn't a candidate.
      const ints = await cli([flowFile(bare({ n: "int" }))]);
      expect(ints.code).toBe(2);
      expect(ints.stderr).toContain("flow 'f' has no Inbox(str) input");
    },
    TIMEOUT,
  );

  test(
    "several Inbox(str) inputs need --inbox, which must name one",
    async () => {
      const flow = flowFile(bare({ a: "str", b: "str", c: "int" }));
      const several = await cli([flow]);
      expect(several.code).toBe(2);
      expect(several.stderr).toContain("flow 'f' has several Inbox(str) inputs (a, b); pick one with --inbox");
      const wrong = await cli([flow, "--inbox", "c"]);
      expect(wrong.code).toBe(2);
      expect(wrong.stderr).toContain("--inbox 'c' is not one of the flow's Inbox(str) inputs (a, b)");
      // A chosen inbox goes on to the run. The end of input closes it, and the takes of the other
      // inboxes, which nothing can send to, leave the run waiting.
      const picked = await cli([flow, "--inbox", "b"]);
      expect(picked.code).toBe(0);
      expect(picked.stderr).toContain("run waiting: take_a (take), take_c (take)");
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
      const missing = await cli([path]);
      expect(missing.code).toBe(2);
      expect(missing.stderr).toContain("missing required flow input: topic");
      expect(missing.stderr).not.toContain("messages");
      const inbox = await cli([path, "-i", "topic=x", "-i", "messages=hi"]);
      expect(inbox.code).toBe(2);
      expect(inbox.stderr).toContain('unknown input "messages"');
    },
    TIMEOUT,
  );
});

/** The chat example: one model call per turn, the history carried by bindings. */
const EXAMPLE_CHAT = fileURLToPath(new URL("../../../examples/chat.yaml", import.meta.url));

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

describe("the entry's chat, signalled", () => {
  const ENTRY = fileURLToPath(new URL("../../src/cli/index.ts", import.meta.url));

  /** Start `yian chat <flow> --log <log>` in a child on a piped stdin, collecting its streams. */
  function spawnChat(flow: string, log: string) {
    const runtime = process.versions.bun !== undefined ? ["--no-install", "--no-env-file"] : [];
    const child = spawn(process.execPath, [...runtime, ENTRY, "chat", flow, "--log", log], {
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
    "SIGINT prints the hint; a second cancels the run: exit 130",
    async () => {
      const log = join(tmp, "signalled.jsonl");
      const { child, streams, closed } = spawnChat(flowFile(ECHO), log);
      try {
        child.stdin.write("hi\n");
        await soon(() => streams.out.includes("echo: hi"), "the reply");
        child.kill("SIGINT");
        await soon(() => streams.err.includes("Ctrl-C again"), "the hint");
        child.kill("SIGINT");
        expect(await closed).toEqual({ code: 130, signal: null });
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
