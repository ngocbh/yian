/**
 * `yian chat`'s look on a terminal (`chat_view.ts`): its pieces on their own, and the chat drawn
 * on a fake TTY 60 columns wide, read back as the screen shows it, in colour and with `NO_COLOR`
 * (the plain lines).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, realpathSync, renameSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { chat, type ChatDeps } from "../../src/cli/chat.ts";
import { childEvent } from "../../src/cli/chat_events.ts";
import {
  chatBanner,
  ChatView,
  homePath,
  loginLabel,
  richChat,
  shortArgs,
  summary,
} from "../../src/cli/chat_view.ts";
import { cells } from "../../src/cli/cells.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import type { Env } from "../../src/defaults.ts";
import {
  ModelCallFinished,
  ModelCallStarted,
  NodeSucceeded,
  StreamChunk,
  ToolCallFinished,
  ToolCallStarted,
} from "../../src/events.ts";
import type { Scheduler } from "../../src/platform.ts";
import { type ChatModel, EventStream } from "../../src/llm/events.ts";
import type { Request, Response } from "../../src/llm/messages.ts";
import { Finish, PartDone, TextDelta } from "../../src/llm/events.ts";
import { TextPart } from "../../src/llm/messages.ts";
import { ScriptedModel, type ScriptedReply } from "../../src/llm/testing.ts";
import { callReply, serving, textReply } from "../helpers/run_state_fixtures.ts";
import { ENTER, FakeInput, FakeOutput, screen } from "./terminal_both_runtimes.ts";

const TIMEOUT = 20_000;
const COLUMNS = 60;

let tmp: string;
let home: string;
let root: string;
let stdin: FakeInput | undefined;

beforeEach(() => {
  tmp = realpathSync(mkdtempSync(join(tmpdir(), "yian-chat-view-")));
  home = join(tmp, "home");
  root = join(home, "proj");
  mkdirSync(root, { recursive: true });
});

afterEach(() => {
  stdin?.end();
  stdin = undefined;
  rmSync(tmp, { recursive: true, force: true });
});

/** The chat's environment: a provider, model and key for the assistant, the home, and `extra`. */
function envWith(extra: Record<string, string> = {}): Env {
  const values: Record<string, string> = {
    HOME: home,
    XAI_API_KEY: "test-xai-key",
    YIAN_DEFAULT_PROVIDER: "xai",
    YIAN_DEFAULT_MODEL: "grok-4.3",
    ...extra,
  };
  return (name) => values[name];
}

/**
 * An io on one fake TTY 60 columns wide: stdout and stderr are written where the terminal draws,
 * each `\n` as `\r\n` as a cooked terminal shows it. With `noColor`, `NO_COLOR` is set.
 */
function terminalIo({ noColor = false } = {}) {
  stdin = new FakeInput({ tty: true });
  const output = Object.assign(new FakeOutput({ columns: COLUMNS }), { isTTY: true });
  const writer: Writer = {
    write: async (text) => void output.write(text.replaceAll("\n", "\r\n")),
    isTTY: true,
    columns: COLUMNS,
  };
  const env = envWith(noColor ? { NO_COLOR: "1" } : {});
  const io: CliIo = { stdout: writer, stderr: writer, stdin, stderrStream: output, env };
  return { io, stdin, output, rows: () => screen(output.text, COLUMNS).rows };
}

/** A model answering each request with `answer` of it and its count (from 1). */
class AnsweringModel implements ChatModel {
  readonly provider = "xai";
  readonly model = "grok-4.3";
  readonly requests: Request[] = [];
  readonly #answer: (request: Request, count: number) => ScriptedReply;

  constructor(answer: (request: Request, count: number) => ScriptedReply) {
    this.#answer = answer;
  }

  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    this.requests.push(request);
    return new ScriptedModel([this.#answer(request, this.requests.length)], this).stream(request, options);
  }
}

/** Run `yian chat` in `root`, its log in `tmp`. */
function runChat(io: CliIo, model: ChatModel, flow?: string, deps: ChatDeps = {}): Promise<number> {
  return chat(flow, { engineTrace: false, log: join(tmp, "chat.jsonl") }, io, {
    root,
    models: serving(model, io.env),
    signals: new EventEmitter(),
    ...deps,
  });
}

/** Resolve once `check` holds, polling each turn of the event loop. */
async function until(check: () => boolean): Promise<void> {
  while (!check()) await new Promise((resolve) => setTimeout(resolve, 5));
}

/** A box row 60 columns wide holding `text` of one-cell characters. */
const row = (text: string) => `│ ${text.padEnd(COLUMNS - 4)} │`;
const TOP = `╭${"─".repeat(COLUMNS - 2)}╮`;
const BOTTOM = `╰${"─".repeat(COLUMNS - 2)}╯`;
const HINT = "  /exit to leave · ctrl-c interrupts · esc withdraws";

describe("the banner's pieces", () => {
  test("loginLabel names a subscription and a key that is there", () => {
    expect(loginLabel({ method: "subscription", source: "store" })).toBe("ChatGPT login");
    expect(loginLabel({ method: "api_key", source: "environment" })).toBe("API key");
    expect(loginLabel({ method: "api_key", source: "store" })).toBe("API key");
    expect(loginLabel({ method: "api_key", source: "missing" })).toBeNull();
    expect(loginLabel(undefined)).toBeNull();
  });

  test("homePath shows the home as ~", () => {
    expect(homePath("/home/ada/radev/yian", "/home/ada")).toBe("~/radev/yian");
    expect(homePath("/home/ada", "/home/ada/")).toBe("~");
    expect(homePath("/home/adam", "/home/ada")).toBe("/home/adam");
    expect(homePath("/srv", undefined)).toBe("/srv");
    expect(homePath("/srv", "/")).toBe("/srv");
  });

  test("homePath also shows the home's real path as ~, for a home reached through a symlink", () => {
    expect(homePath("/real/ada/proj", "/home/ada", "/real/ada")).toBe("~/proj");
    expect(homePath("/home/ada/proj", "/home/ada", "/real/ada")).toBe("~/proj");
    expect(homePath("/real/adam", "/home/ada", "/real/ada")).toBe("/real/adam");
  });

  test("a long value wraps under its label, and the text is sanitized", () => {
    const output = Object.assign(new FakeOutput({ columns: COLUMNS }), { isTTY: true });
    const info = { model: "openai/gpt\x1b[31m", login: "ChatGPT login", cwd: `~/${"deep/".repeat(12)}x`, focus: null };
    output.write(chatBanner(info, output, () => undefined).replaceAll("\n", "\r\n"));
    expect(screen(output.text, COLUMNS).rows).toEqual([
      TOP,
      row("✻ yian chat"),
      row("  model   openai/gpt (ChatGPT login)"),
      // A word wider than the column is folded by characters.
      row("  cwd     ~/deep/deep/deep/deep/deep/deep/deep/deep/deep"),
      row("          /deep/deep/deep/x"),
      BOTTOM,
    ]);
  });
});

describe("yian chat on a terminal", () => {
  test(
    "draws the banner box and the framed prompt with its hint",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const chatting = runChat(io, new AnsweringModel(() => textReply("hello")));
      await until(() => rows().includes(HINT));
      const shown = rows();
      expect(shown.slice(0, 5)).toEqual([
        TOP,
        row("✻ yian chat"),
        row("  model   xai/grok-4.3 (API key)"),
        row("  cwd     ~/proj"),
        BOTTOM,
      ]);
      expect(shown.slice(-4)).toEqual([TOP, row(">"), BOTTOM, HINT]);
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      expect(rows()).toContain("> /exit");
    },
    TIMEOUT,
  );

  test(
    "names the folder from ~ when the home is a symlink to where the folder really is",
    async () => {
      const real = join(tmp, "real");
      renameSync(home, real);
      symlinkSync(real, home);
      root = join(real, "proj");
      const { io, stdin, rows } = terminalIo();
      const chatting = runChat(io, new AnsweringModel(() => textReply("hello")));
      await until(() => rows().includes(HINT));
      expect(rows()[3]).toBe(row("  cwd     ~/proj"));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "with NO_COLOR prints today's plain lines",
    async () => {
      const { io, stdin, output, rows } = terminalIo({ noColor: true });
      const chatting = runChat(io, new AnsweringModel(() => textReply("hello")));
      await until(() => rows().some((line) => line.startsWith("> You:")));
      expect(rows()[0]).toBe("yian chat · xai/grok-4.3");
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      expect(rows()).not.toContain(TOP);
      expect(output.text).not.toContain("╭");
    },
    TIMEOUT,
  );
});

/** A clock that moves only when the test advances it. */
function manualClock(): Scheduler & { advance(ms: number): Promise<void> } {
  let now = 0;
  let sleepers: { at: number; wake: () => void }[] = [];
  return {
    now: () => now,
    sleep(ms, signal) {
      return new Promise<void>((resolve, reject) => {
        if (signal?.aborted) return reject(signal.reason);
        const sleeper = { at: now + ms, wake: resolve };
        sleepers.push(sleeper);
        signal?.addEventListener(
          "abort",
          () => {
            sleepers = sleepers.filter((other) => other !== sleeper);
            reject(signal.reason);
          },
          { once: true },
        );
      });
    },
    async advance(ms) {
      const until = now + ms;
      while (true) {
        const due = sleepers.filter((sleeper) => sleeper.at <= until).sort((a, b) => a.at - b.at)[0];
        if (due === undefined) break;
        sleepers = sleepers.filter((other) => other !== due);
        now = due.at;
        due.wake();
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      now = until;
    },
  };
}

/** A view on a fake TTY 60 columns wide, its clock, and the rows the screen shows. */
function viewOn(rows?: number) {
  const output = Object.assign(new FakeOutput({ columns: COLUMNS }), { isTTY: true, rows });
  const clock = manualClock();
  const view = new ChatView(output, () => undefined, clock);
  return { view, clock, output, rows: () => screen(output.text, COLUMNS).rows };
}

const REPLY = "chat/answered/reply";
const chunk = (n: number, text: string) => new StreamChunk({ nodeId: REPLY, key: "text", chunk: text, n, index: 0 });
const started = (n: number) => new ModelCallStarted({ nodeId: REPLY, n, purpose: "chat", startedAt: 0 });
const finished = (n: number, error: string | null = null) =>
  new ModelCallFinished({ nodeId: REPLY, n, purpose: "chat", error });

describe("the chat's view", () => {
  test("a reply streams after ⏺, rendered as Markdown line by line, and isn't printed again", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onEvent(started(1));
    view.onStream(chunk(1, "hello ran and "));
    expect(rows()).toEqual(["", "⏺ hello ran and", "· Thinking… (0s · esc to interrupt)"]);
    view.onStream(chunk(1, "returned **Hi Ada**.\n- one"));
    expect(rows()).toEqual([
      "",
      "⏺ hello ran and returned Hi Ada.",
      "  - one",
      "· Thinking… (0s · esc to interrupt)",
    ]);
    view.onEvent(finished(1));
    view.onEvent(new NodeSucceeded({ nodeId: REPLY, output: "..." }));
    view.reply("hello ran and returned **Hi Ada**.\n- one");
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ hello ran and returned Hi Ada.", "  • one"]);
  });

  test("a failed attempt's text is replaced by (retrying) once the next starts, in a new block", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "first try\nmore"));
    view.onEvent(finished(1, "ProviderError"));
    // Not known to be retried yet: the text is gone and nothing stands in for it.
    expect(rows()).toEqual(["", "· Thinking… (0s · esc to interrupt)"]);
    view.onEvent(started(2));
    view.onStream(chunk(2, "second"));
    view.onEvent(finished(2));
    view.turn(false);
    expect(rows()).toEqual(["", "  (retrying)", "", "⏺ second"]);
  });

  test("a failed attempt that isn't retried shows (failed) when the turn ends", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "half a rep"));
    view.onEvent(finished(1, "ProviderError"));
    view.turn(false);
    expect(rows()).toEqual(["", "  (failed)"]);
  });

  test("a \\r\\n split across chunks ends one line, and a tab in the line coming is expanded", () => {
    const { view, rows } = viewOn();
    view.onStream(chunk(1, "one\r"));
    view.onStream(chunk(1, "\ntwo\tx"));
    expect(rows()).toEqual(["", "⏺ one", "  two    x"]);
    view.onEvent(finished(1));
    expect(rows()).toEqual(["", "⏺ one", "  two    x"]);
  });

  test("a long line still streaming prints its top for good, so the live rows fit the screen", () => {
    const { view, output, rows } = viewOn(10);
    view.turn(true);
    view.onStream(chunk(1, "word ".repeat(400)));
    // Every move up the live rows take stays within the screen: at most rows - 1 rows drawn.
    const ups = [...output.text.matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(Math.max(0, ...ups)).toBeLessThanOrEqual(8);
    view.onEvent(finished(1));
    view.turn(false);
    const text = rows().join(" ").replace(/[⏺\s]+/g, " ").trim();
    expect(text).toBe("word ".repeat(400).trim());
  });

  test("each live row is cut to the width: a fence's long label, a deep bullet, the spinner", () => {
    const { view, rows } = viewOn();
    const narrow = viewOn();
    view.turn(true);
    view.onStream(chunk(1, `\`\`\`${"x".repeat(80)}\n${" ".repeat(70)}- deep\nmore`));
    for (const row of rows()) expect(cells(row)).toBeLessThanOrEqual(COLUMNS);
    narrow.output.columns = 20;
    narrow.view.turn(true);
    expect(narrow.rows()).toEqual(["· Thinking… (0s · es"]);
  });

  test("on a resize, the live rows are erased from as high as they were drawn, and drawn at the new width", () => {
    const { view, output } = viewOn();
    view.turn(true);
    const before = output.text.length;
    output.columns = 20;
    output.emit("resize");
    // One row was drawn: the cursor never goes up past it, whatever the terminal did to it.
    expect(output.text.slice(before)).toBe("\r\x1b[J\x1b[36m·\x1b[0m \x1b[2mThinking… (0s · es\x1b[0m");
    view.turn(false);
    // It listens only while a turn draws live rows, and not at all once closed.
    expect(output.listenerCount("resize")).toBe(0);
    view.turn(true);
    view.close();
    expect(output.listenerCount("resize")).toBe(0);
  });

  test("a resize while a reply streams keeps the live rows within the smaller screen", () => {
    const { view, output, rows } = viewOn(30);
    view.turn(true);
    view.onStream(chunk(1, "1\n2\n3\n4\n5\n6\n7\n8\n9\nten"));
    output.rows = 8;
    output.emit("resize");
    // Four rows stay live, above the spinner; the rest are printed for good.
    expect(rows().slice(-5)).toEqual(["  7", "  8", "  9", "  ten", "· Thinking… (0s · esc to interrupt)"]);
    const before = output.text.length;
    view.onStream(chunk(1, "!"));
    const ups = [...output.text.slice(before).matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(ups).toEqual([4]);
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ 1", "  2", "  3", "  4", "  5", "  6", "  7", "  8", "  9", "  ten!"]);
  });

  test("a prompt taking the terminal while a reply streams hides it, and gives it back below", () => {
    const { view, output, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "a\nb"));
    view.active(true);
    expect(rows()).toEqual([]);
    output.write("? Run hello.yaml Approve\r\n");
    view.active(false);
    expect(rows()).toEqual(["", "? Run hello.yaml Approve", "⏺ a", "  b", "· Thinking… (0s · esc to interrupt)"]);
  });

  test("the cursor is hidden while a turn runs and shown after it, a prompt or close", () => {
    const { view, output } = viewOn();
    const modes = () => screen(output.text, COLUMNS).modes;
    view.turn(true);
    expect(modes()).toEqual(["?25l"]);
    view.active(true);
    view.active(false);
    // A prompt shows the cursor as it ends; the turn hides it again.
    expect(modes().at(-1)).toBe("?25l");
    view.turn(false);
    expect(modes().at(-1)).toBe("?25h");
    view.turn(true);
    view.close();
    expect(modes().at(-1)).toBe("?25h");
  });

  test("a child event between two chunks of a reply prints above it, and the reply goes on as it was", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onEvent(started(1));
    view.onStream(chunk(1, "Here:\n```py\nx = 1\ny = "));
    const status = { run: "r1", seq: 1, log: "r1.jsonl", status: "succeeded" as const, output: "ok" };
    view.childEvent(childEvent(status)!);
    view.onStream(chunk(1, "2\n```\nDone **now**.\n"));
    view.onEvent(finished(1));
    view.turn(false);
    const frame = (text: string) => `  │ ${text.padEnd(54)} │`;
    expect(rows()).toEqual([
      "",
      "⏺ Here:",
      `  ╭─ py ${"─".repeat(51)}╮`,
      frame("x = 1"),
      "",
      "⏺ r1 succeeded",
      '  ⎿  with output "ok"',
      frame("y = 2"),
      `  ╰${"─".repeat(56)}╯`,
      "  Done now.",
    ]);
  });

  test("a line printed in pieces is printed once whole", () => {
    const { view, rows } = viewOn();
    view.print("partial ");
    expect(rows()).toEqual([]);
    view.print("rest\nnext\n");
    expect(rows()).toEqual(["partial rest", "next"]);
    view.print("left");
    view.close();
    expect(rows()).toEqual(["partial rest", "next", "left"]);
  });

  test("a long line printed in part keeps its raw tail: a sequence or a \\r\\n split across chunks", () => {
    const { view, output, rows } = viewOn(10);
    view.onStream(chunk(1, `${"word ".repeat(100)}\x1b[3`));
    view.onStream(chunk(1, "1mred\x1b[0m end\r"));
    view.onStream(chunk(1, "\nnext"));
    view.onEvent(finished(1));
    expect(output.text).not.toContain("^[");
    expect(rows().slice(-2)).toEqual([expect.stringMatching(/word red end$/), "  next"]);
  });

  test("streamed text is sanitized", () => {
    const { view, output, rows } = viewOn();
    view.onStream(chunk(1, "a\x1b[2Jb\x1b]52;c;eA==\x07 \u202e"));
    view.onEvent(finished(1));
    expect(rows()).toEqual(["", "⏺ ab \\u{202e}"]);
    expect(output.text).not.toContain("\x1b[2J");
    expect(output.text).not.toContain("\x1b]52");
  });

  test("a reply that didn't stream is printed as a block", () => {
    const { view, rows } = viewOn();
    view.onEvent(started(1));
    view.reply("# Done\nall `good`");
    expect(rows()).toEqual(["", "⏺ Done", "  all good"]);
  });

  test("a tool call is a ⏺ line with short arguments and a ⎿ result of at most two rows", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    const call = { nodeId: REPLY, n: 1, callIndex: 0, toolCallId: "c1" };
    view.onEvent(new ToolCallStarted({ ...call, name: "check_flow", arguments: { path: "examples/hello.yaml" } }));
    const checked = "ok\nname: hello\ninputs:\n  name: str\n";
    view.onEvent(new ToolCallFinished({ ...call, name: "check_flow", result: checked }));
    const start = { ...call, callIndex: 1, toolCallId: "c2", name: "start_flow" };
    const args = { path: "examples/hello.yaml", inputs: { name: "Ada" } };
    view.onEvent(new ToolCallStarted({ ...start, arguments: args }));
    view.onEvent(new ToolCallFinished({ ...start, result: `invalid: \x1b[31m${"word ".repeat(40)}` }));
    view.onEvent(new ToolCallStarted({ ...call, toolCallId: "c3", name: "wait_flow", arguments: { run: "r1" } }));
    view.onEvent(new ToolCallFinished({ ...call, toolCallId: "c3", name: "wait_flow", error: "interrupted" }));
    view.turn(false);
    expect(rows()).toEqual([
      "",
      "⏺ check_flow(examples/hello.yaml)",
      "  ⎿  ok · name: hello · inputs: name: str",
      '⏺ start_flow(examples/hello.yaml, {name: "Ada"})',
      "  ⎿  invalid: word word word word word word word word word",
      "     word word word word word word word word word word word…",
      "⏺ wait_flow(r1)",
      "  ⎿  interrupted",
    ]);
  });

  test("a child event is a card coloured by its status", () => {
    const { view, output, rows } = viewOn();
    const status = (fields: object) => ({ run: "r1", seq: 1, log: "r1.jsonl", status: "waiting" as const, ...fields });
    view.childEvent(childEvent(status({ status: "succeeded", output: "Hi Ada" }))!);
    view.childEvent(childEvent(status({ asks: [{ call: "c1", question: "Which region?" }] }))!);
    const failure = { code: "code.error", message: "kaboom", node: "x" };
    view.childEvent(childEvent(status({ status: "failed", failure }))!);
    view.childEvent({ run: null, seq: 0, text: "run r9 was cancelled" });
    expect(rows()).toEqual([
      "",
      "⏺ r1 succeeded",
      '  ⎿  with output "Hi Ada"',
      "⏺ r1 asks",
      '  ⎿  (call c1): "Which region?"',
      "⏺ r1 failed",
      '  ⎿  (code.error at x): "kaboom"',
      "⏺ yian",
      "  ⎿  run r9 was cancelled",
    ]);
    expect(output.text).toContain("\x1b[32m⏺");
    expect(output.text).toContain("\x1b[31m⏺");
  });

  test("the spinner ticks on its own clock, keeps clear of lines printed, and goes when the turn ends", async () => {
    const { view, clock, rows } = viewOn();
    view.turn(true);
    expect(rows()).toEqual(["· Thinking… (0s · esc to interrupt)"]);
    await clock.advance(3_000);
    // Twelve frames of six: back at the first.
    expect(rows()).toEqual(["· Thinking… (3s · esc to interrupt)"]);
    await clock.advance(250);
    expect(rows()).toEqual(["✢ Thinking… (3s · esc to interrupt)"]);
    await view.writer.write("warning: a skill didn't load\n");
    expect(rows()).toEqual(["warning: a skill didn't load", rows()[1]!]);
    expect(rows()[1]).toContain("Thinking… (3s");
    // A prompt has the terminal: the spinner goes until it gives it back.
    view.active(true);
    expect(rows()).toEqual(["warning: a skill didn't load"]);
    view.active(false);
    expect(rows()[1]).toContain("Thinking…");
    view.turn(false);
    expect(rows()).toEqual(["warning: a skill didn't load"]);
  });

  test("a block longer than the screen prints its top for good", () => {
    const { view, rows } = viewOn(8);
    view.onStream(chunk(1, "1\n2\n3\n4\n5\n6\n"));
    view.onEvent(finished(1, "ProviderError"));
    view.onEvent(started(2));
    // Its top is gone by: the block stays as it was, and (retrying) follows it.
    expect(rows()).toEqual(["", "⏺ 1", "  2", "  3", "  4", "  5", "  6", "  (retrying)"]);
  });
});

describe("the view's helpers", () => {
  test("richChat needs stdin, stdout and stderr on a TTY and NO_COLOR unset", () => {
    const io = (stdout: boolean, noColor = false): CliIo => ({
      stdin: new FakeInput({ tty: true }),
      stderrStream: Object.assign(new FakeOutput(), { isTTY: true }),
      stdout: { write: async () => {}, isTTY: stdout },
      stderr: { write: async () => {}, isTTY: true },
      env: envWith(noColor ? { NO_COLOR: "1" } : {}),
    });
    expect(richChat(io(true))).toBe(true);
    expect(richChat(io(false))).toBe(false);
    expect(richChat(io(true, true))).toBe(false);
  });

  test("shortArgs shows a path or run bare and other values short", () => {
    expect(shortArgs({ path: "a.yaml", inputs: { n: 1, tags: ["x"] } })).toBe('a.yaml, {n: 1, tags: ["x"]}');
    expect(shortArgs({ content: "y".repeat(60) })).toBe(`"${"y".repeat(38)}…"`);
    expect(shortArgs(null)).toBe("");
  });

  test("summary joins lines and keeps to two rows", () => {
    expect(summary("ok\ninputs:\n  a: int\n  b: str\n\nnext", 60)).toEqual(["ok · inputs: a: int, b: str · next"]);
    expect(summary("x ".repeat(50), 20)).toEqual(["x x x x x x x x x x", "x x x x x x x x x x…"]);
  });
});

/** A reply streamed in `chunks`. */
function streamed(...chunks: string[]): ScriptedReply {
  const text = chunks.join("");
  const deltas = chunks.map((chunk) => new TextDelta(0, chunk));
  return [...deltas, new PartDone(0, new TextPart(text)), new Finish({ reason: "stop" })];
}

const HELLO = `id: hello
name: hello
input:
  name: !str
nodes:
  greet:
    kind: code
    input:
      name: \${input.name}
    output: str
    code: |
      return "Hi " + inputs["name"];
output: \${greet.output}
`;

describe("a session on a terminal", () => {
  /** Run the hello session: check, start once approved, wait, and a streamed reply. */
  async function hello(noColor: boolean): Promise<{ rows: string[]; text: string }> {
    writeFileSync(join(root, "hello.yaml"), HELLO);
    const { io, stdin, output, rows } = terminalIo({ noColor });
    const script: ScriptedReply[] = [
      callReply(["check_flow", { path: "hello.yaml" }, "c1"]),
      callReply(["start_flow", { path: "hello.yaml", inputs: { name: "Ada" } }, "c2"]),
      callReply(["wait_flow", { run: "r1", seconds: 30 }, "c3"]),
      streamed("hello ran and ", "returned **Hi Ada**."),
    ];
    const model = new AnsweringModel((_, count) => script[count - 1] ?? textReply("noted"));
    const chatting = runChat(io, model, undefined, { ticker: manualClock() });
    const prompt = noColor ? "> You:" : "│ >";
    await until(() => rows().some((line) => line.startsWith(prompt)));
    stdin.write(`run hello for Ada${ENTER}`);
    await until(() => output.text.includes("Run hello.yaml"));
    stdin.write(ENTER);
    await until(() => model.requests.length >= 4 && rows().some((line) => line.startsWith(prompt)));
    stdin.write(`/exit${ENTER}`);
    expect(await chatting).toBe(0);
    const shown = rows();
    return { rows: shown.slice(shown.findIndex((line) => line.includes("run hello for Ada"))), text: output.text };
  }

  /** The text of the approval card. */
  const CARD = [
    "Starts hello.yaml in the background.",
    'inputs: {"name":"Ada"}',
    "node kinds: code",
    "runs code on this machine (CODE nodes)",
    "an approval for this chat doesn't cover changes to the files the code imports",
  ];

  test(
    "draws tool lines, the approval, and the streamed reply in colour",
    async () => {
      const { rows } = await hello(false);
      expect(rows).toEqual([
        "> run hello for Ada",
        "",
        "⏺ check_flow(hello.yaml)",
        "  ⎿  ok · name: hello · inputs: name: str",
        '⏺ start_flow(hello.yaml, {name: "Ada"})',
        `╭─ Run hello.yaml ${"─".repeat(41)}╮`,
        ...CARD.slice(0, 4).map(row),
        row("an approval for this chat doesn't cover changes to the"),
        row("files the code imports"),
        BOTTOM,
        "? Run hello.yaml Approve",
        // Of several fields, each takes at most half the row: the log's path is cut.
        expect.stringMatching(/^ {2}⎿ {2}run: r1 · status: running · log: \S+…$/),
        "⏺ wait_flow(r1, 30)",
        expect.stringMatching(/^ {2}⎿ {2}run: r1 · log: \S+… · status:$/),
        "     succeeded · output: Hi Ada",
        "",
        "⏺ hello ran and returned Hi Ada.",
        "",
        "> /exit",
        "session ended",
      ]);
    },
    TIMEOUT,
  );

  test(
    "prints today's plain lines with NO_COLOR",
    async () => {
      const { rows, text } = await hello(true);
      expect(rows).toEqual([
        "> You: run hello for Ada",
        ...CARD.slice(0, 4),
        "an approval for this chat doesn't cover changes to the files",
        " the code imports",
        "? Run hello.yaml Approve",
        "hello ran and returned **Hi Ada**.",
        "> You: /exit",
        "session ended",
      ]);
      expect(text).not.toContain("⏺");
      expect(text).not.toContain("╭");
    },
    TIMEOUT,
  );
});
