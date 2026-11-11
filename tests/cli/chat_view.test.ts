/**
 * `yian chat`'s look on a terminal (`chat_view.ts`): its pieces on their own, and the chat drawn
 * on a fake TTY 60 columns wide, read back as the screen shows it, in colour and with `NO_COLOR`
 * (the plain lines).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import {
  existsSync,
  mkdirSync,
  mkdtempSync,
  readdirSync,
  readFileSync,
  realpathSync,
  renameSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
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
  runRows,
  shortArgs,
  summary,
} from "../../src/cli/chat_view.ts";
import type { ChildStatus, LiveChild } from "../../src/cli/chat_runs.ts";
import { LineEditor, Terminal } from "../../src/cli/terminal.ts";
import { cells } from "../../src/cli/cells.ts";
import { liveRows, SYNC_OFF, SYNC_ON } from "../../src/cli/live_rows.ts";
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
import {
  callReply,
  chatServing,
  namingRequest,
  textReply,
} from "../helpers/run_state_fixtures.ts";
import {
  CTRL_C,
  CTRL_L,
  DOWN,
  ENTER,
  ESCAPE,
  FakeInput,
  FakeOutput,
  PASTE_END,
  PASTE_START,
  screen,
  tick,
} from "./terminal_both_runtimes.ts";

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
 * each `\n` as `\r\n` as a cooked terminal shows it. With `noColor`, `NO_COLOR` is set; `extra`
 * adds variables.
 */
function terminalIo({ noColor = false, extra = {} as Record<string, string> } = {}) {
  stdin = new FakeInput({ tty: true });
  const output = Object.assign(new FakeOutput({ columns: COLUMNS }), { isTTY: true });
  const writer: Writer = {
    write: async (text) => void output.write(text.replaceAll("\n", "\r\n")),
    isTTY: true,
    columns: COLUMNS,
  };
  const env = envWith({ ...extra, ...(noColor ? { NO_COLOR: "1" } : {}) });
  const io: CliIo = { stdout: writer, stderr: writer, stdin, stderrStream: output, env };
  return { io, stdin, output, rows: () => screen(output.text, COLUMNS).rows };
}

/**
 * A model answering each request with `answer` of it and its count (from 1). The call naming the
 * chat is kept apart (`named`) and answered `A chat`.
 */
class AnsweringModel implements ChatModel {
  readonly provider = "xai";
  readonly model = "grok-4.3";
  readonly requests: Request[] = [];
  readonly named: Request[] = [];
  readonly #answer: (request: Request, count: number) => ScriptedReply;

  constructor(answer: (request: Request, count: number) => ScriptedReply) {
    this.#answer = answer;
  }

  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    if (namingRequest(request)) {
      this.named.push(request);
      return new ScriptedModel([textReply("A chat")], this).stream(request, options);
    }
    this.requests.push(request);
    return new ScriptedModel([this.#answer(request, this.requests.length)], this).stream(request, options);
  }
}

/**
 * Wait out the moment a card that opens over the input box drops keys, so a key meant for it
 * counts.
 */
const pastGrace = () => new Promise((resolve) => setTimeout(resolve, 450));

/** Run `yian chat` in `root`, its log in `tmp`. */
function runChat(io: CliIo, model: ChatModel, flow?: string, deps: ChatDeps = {}): Promise<number> {
  return chat(flow, { engineTrace: false, log: join(tmp, "chat.jsonl") }, io, {
    root,
    models: chatServing(model, io.env),
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
    "names the folder as the shell reached it ($PWD), through a symlink under the home",
    async () => {
      const real = join(tmp, "elsewhere");
      mkdirSync(join(real, "proj"), { recursive: true });
      symlinkSync(real, join(home, "radev"));
      root = join(real, "proj");
      const { io, stdin, rows } = terminalIo({ extra: { PWD: join(home, "radev", "proj") } });
      const chatting = runChat(io, new AnsweringModel(() => textReply("hello")));
      await until(() => rows().includes(HINT));
      expect(rows()[3]).toBe(row("  cwd     ~/radev/proj"));
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

  test("the spinner names the model the view is given", () => {
    const { view, rows } = viewOn();
    view.model("openai/gpt-5.5");
    view.turn(true);
    expect(rows()).toEqual(["· Thinking… (0s · openai/gpt-5.5 · esc to interrupt)"]);
    view.model(null);
    expect(rows()).toEqual(["· Thinking… (0s · esc to interrupt)"]);
    view.turn(false);
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
    const spin = "\x1b[36m·\x1b[0m \x1b[2mThinking… (0s · es\x1b[0m";
    expect(output.text.slice(before)).toBe(`${SYNC_ON}\r\x1b[J\r${spin}${SYNC_OFF}`);
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
    // Only the row that changed is written again: the cursor goes up from the spinner to it.
    const ups = [...output.text.slice(before).matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(ups).toEqual([1]);
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ 1", "  2", "  3", "  4", "  5", "  6", "  7", "  8", "  9", "  ten!"]);
  });

  test("a prompt opening while a reply streams draws over it, and gives it back", () => {
    const { view, output, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "a\nb"));
    view.active(true);
    // The prompt draws on the same live rows, over the reply's.
    const live = liveRows(output);
    live.draw(["? Run hello.yaml Approve (Use arrow keys)", " » Approve"], null);
    expect(rows()).toEqual(["", "? Run hello.yaml Approve (Use arrow keys)", " » Approve"]);
    live.stage(["? Run hello.yaml Approve"]);
    live.set([], null);
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

  test("one long chunk with no newline is split in time, whatever its characters", () => {
    for (const [unit, size] of [["a", 128 * 1024], ["ab\tc", 32 * 1024], ["y\u00e9\u202e", 16 * 1024]] as const) {
      const { view } = viewOn(24);
      const start = performance.now();
      view.onStream(chunk(1, unit.repeat(size / unit.length)));
      expect(performance.now() - start).toBeLessThan(1000);
    }
  });

  test("a long chunk of tabs or BIDI characters is printed in rows as it was drawn", () => {
    const tabs = viewOn(6);
    tabs.view.onStream(chunk(1, "ab\tc".repeat(40)));
    expect(tabs.rows()).toEqual([
      "",
      "⏺ ab    cab    cab    cab    cab    cab    cab    cab    cab",
      "  cab    cab    cab    cab    cab    cab    cab    cab",
      "  cab    cab    cab    cab    cab    cab    cab    cab",
      "  cab    cab    cab    cab    cab    cab    cab    cab",
      "  cab    cab    cab    cab    cab    cab    cab    c",
    ]);
    const bidi = viewOn(6);
    bidi.view.onStream(chunk(1, "y\u00e9\u202e".repeat(40)));
    expect(bidi.rows()).toEqual([
      "",
      "⏺ yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202",
      "  e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{2",
      "  02e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u",
      "  {202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé",
      "  \\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}",
      "  yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202",
      "  e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}yé\\u{202e}",
    ]);
  });

  test("a long chunk printed in part is cut where its rows were drawn, a sequence across the cut included", () => {
    // Drawn, each is 290 characters: five rows of 58, of which the top three are printed for good.
    const link = "\x1b]8;;https://e.example\x1b\\link\x1b]8;;\x1b\\";
    const cases = [
      [`${"x".repeat(170)}${link}${"y".repeat(116)}`, `${"x".repeat(170)}link${"y".repeat(116)}`],
      [`${"x".repeat(172)}\x1b7${"y".repeat(115)}`, `${"x".repeat(172)}^[7${"y".repeat(115)}`],
    ];
    for (const [raw, drawn] of cases) {
      const { view, rows } = viewOn(6);
      view.onStream(chunk(1, raw!));
      const shown = rows().slice(1);
      expect(shown.length).toBeGreaterThan(3);
      expect(shown.map((row) => row.slice(2)).join("")).toBe(drawn!);
    }
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
    const { view, clock, output, rows } = viewOn();
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
    // A prompt has the terminal: it draws over the spinner, which comes back once it is given back.
    view.active(true);
    liveRows(output).draw(["? Pick"], null);
    await clock.advance(250);
    expect(rows()).toEqual(["warning: a skill didn't load", "? Pick"]);
    liveRows(output).set([], null);
    view.active(false);
    expect(rows()[1]).toContain("Thinking…");
    view.turn(false);
    expect(rows()).toEqual(["warning: a skill didn't load"]);
  });

  test("a reply that streams only blank lines, or whose text is blank, draws no ⏺ line", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onEvent(started(1));
    view.onStream(chunk(1, "\n\n  \n"));
    const call = { nodeId: REPLY, n: 1, callIndex: 0, toolCallId: "c1" };
    view.onEvent(new ToolCallStarted({ ...call, name: "answer_flow", arguments: { run: "r1" } }));
    view.onEvent(new ToolCallFinished({ ...call, name: "answer_flow", result: "ok" }));
    view.onEvent(started(2));
    view.onEvent(finished(2));
    view.onEvent(new NodeSucceeded({ nodeId: REPLY, output: "" }));
    view.reply("\n");
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ answer_flow(r1)", "  ⎿  ok"]);
  });

  test("a reply's blank lines before its text are dropped, and so are those after it", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "\n\nStarted it (r1).\n\n"));
    view.onEvent(finished(1));
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ Started it (r1)."]);
  });

  test("a turn of a child event after a reply's turn leaves one blank line between them, not two", () => {
    const { view, rows } = viewOn();
    view.turn(true);
    view.onStream(chunk(1, "Started it (r1)."));
    view.onEvent(finished(1));
    view.turn(false);
    // The prompt went up and was taken down for the event, leaving nothing on the screen.
    view.active(true);
    view.active(false, false);
    view.turn(true);
    const status = { run: "r1", seq: 2, log: "r1.jsonl", status: "succeeded" as const, output: "ok" };
    view.childEvent(childEvent(status)!);
    view.turn(false);
    expect(rows()).toEqual(["", "⏺ Started it (r1).", "", "⏺ r1 succeeded", '  ⎿  with output "ok"']);
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

describe("the input box during a turn", () => {
  /** A box row 60 columns wide holding `text`, as the prompt draws it. */
  const boxed = (text: string) => `│ ${text.padEnd(COLUMNS - 4)} │`;
  const SPIN = "· Thinking… (0s · esc to interrupt)";

  test("stays under the spinner, below a reply streaming, with the cursor in it", () => {
    const { view, output, rows } = viewOn();
    const editor = new LineEditor({ columns: () => COLUMNS, hint: HINT.trim() });
    view.turn(true);
    view.input(editor);
    view.onStream(chunk(1, "Once upon\na"));
    expect(rows()).toEqual(["", "⏺ Once upon", "  a", SPIN, TOP, boxed(">"), BOTTOM, HINT]);
    editor.set("also blue");
    view.refresh();
    expect(rows().slice(-4)).toEqual([TOP, boxed("> also blue"), BOTTOM, HINT]);
    // The cursor is after the text, on the box's row, and shown there.
    expect(screen(output.text, COLUMNS).cursor).toEqual({ row: 5, col: 13 });
    expect(screen(output.text, COLUMNS).modes.at(-1)).toBe("?25h");
    // A line printed for good lands above the live rows, which come back whole.
    view.echo("also blue");
    expect(rows()).toEqual(["", "⏺ Once upon", "> also blue", "  a", SPIN, TOP, boxed("> also blue"), BOTTOM, HINT]);
    view.turn(false);
    // The box stays as the turn ends, where the prompt coming next draws the same box.
    const box = [TOP, boxed("> also blue"), BOTTOM, HINT];
    expect(rows()).toEqual(["", "⏺ Once upon", "> also blue", "  a", "", ...box]);
    expect(screen(output.text, COLUMNS).modes.at(-1)).toBe("?25h");
  });

  test("goes while a prompt has the terminal and comes back after it", () => {
    const { view, output, rows } = viewOn();
    const editor = new LineEditor({ columns: () => COLUMNS, hint: HINT.trim() });
    view.turn(true);
    view.input(editor);
    view.active(true);
    const live = liveRows(output);
    live.draw(["? Run color.yaml Approve (Use arrow keys)"], null);
    expect(rows()).toEqual(["? Run color.yaml Approve (Use arrow keys)"]);
    live.stage(["? Run color.yaml Approve"]);
    live.set([], null);
    view.active(false, true);
    expect(rows()).toEqual(["? Run color.yaml Approve", SPIN, TOP, boxed(">"), BOTTOM, HINT]);
    view.turn(false);
  });

  test("leaves the reply room on a short screen: the box and spinner stay, the block's top is printed", () => {
    const { view, output, rows } = viewOn(12);
    const editor = new LineEditor({ columns: () => COLUMNS, hint: HINT.trim() });
    view.turn(true);
    view.input(editor);
    view.onStream(chunk(1, "1\n2\n3\n4\n5\n6\n7\n8\nnine"));
    const ups = [...output.text.matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(Math.max(0, ...ups)).toBeLessThanOrEqual(11);
    expect(rows().slice(-7)).toEqual(["  8", "  nine", SPIN, TOP, boxed(">"), BOTTOM, HINT]);
    view.turn(false);
  });

  test("keeps within the screen with a paste taller than it, the box cut around its cursor", () => {
    const { view, output, rows } = viewOn(12);
    const editor = new LineEditor({ columns: () => COLUMNS, rows: () => 12, hint: HINT.trim() });
    // Typed, not pasted: a draft's text never collapses into a token.
    editor.set(["word ".repeat(300)]);
    view.turn(true);
    view.input(editor);
    view.onStream(chunk(1, "1\n2\n3\n4\nfive"));
    const ups = [...output.text.matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(Math.max(0, ...ups)).toBeLessThanOrEqual(11);
    expect(rows().at(-8)).toStartWith("╭─ ↑ ");
    expect(rows().slice(-2)).toEqual([BOTTOM, HINT]);
    view.turn(false);
  });

  test("keeps its live rows within the screen when the box alone is taller, dropping its top", () => {
    const { view, output } = viewOn(12);
    // No height given: the box folds onto every row it needs.
    const editor = new LineEditor({ columns: () => COLUMNS, hint: HINT.trim() });
    // Typed, not pasted: a draft's text never collapses into a token.
    editor.set(["word ".repeat(300)]);
    view.turn(true);
    view.input(editor);
    const ups = [...output.text.matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(Math.max(0, ...ups)).toBeLessThanOrEqual(10);
    view.refresh();
    const later = [...output.text.matchAll(/\x1b\[(\d+)A/g)].map((match) => Number(match[1]));
    expect(Math.max(0, ...later)).toBeLessThanOrEqual(10);
    view.turn(false);
  });

  test("while held, nothing is written; let go, all of it is, in order", () => {
    const { view, output, rows } = viewOn();
    view.turn(true);
    view.hold(true);
    const before = output.text;
    view.print("a child said hi\n");
    view.turn(false);
    expect(output.text).toBe(before);
    view.hold(false);
    expect(rows()).toEqual(["a child said hi"]);
  });
});

describe("the live rows' frames", () => {
  const SPIN = "· Thinking… (0s · esc to interrupt)";
  const boxed = (text: string) => `│ ${text.padEnd(COLUMNS - 4)} │`;
  /** What each write from here on writes, as `output` collects it. */
  const recorder = (output: FakeOutput) => {
    const writes: string[] = [];
    const write = output.write.bind(output);
    output.write = ((chunk: unknown, ...rest: unknown[]) => {
      writes.push(String(chunk));
      return (write as (...args: unknown[]) => boolean)(chunk, ...rest);
    }) as typeof output.write;
    return writes;
  };

  test("each frame is one write, wrapped in synchronized output, and each call draws one", () => {
    const { view, output } = viewOn();
    const writes = recorder(output);
    view.turn(true);
    view.onStream(chunk(1, "one\ntw"));
    view.onStream(chunk(1, "o"));
    const status = { run: "r1", seq: 1, log: "l", status: "succeeded" as const, output: "ok" };
    view.childEvent(childEvent(status)!);
    expect(writes).toHaveLength(4);
    for (const write of writes) {
      expect(write.startsWith(SYNC_ON)).toBe(true);
      expect(write.endsWith(SYNC_OFF)).toBe(true);
    }
  });

  test("a spinner tick rewrites only the spinner row, and clears nothing below", async () => {
    const { view, clock, output } = viewOn();
    const editor = new LineEditor({ columns: () => COLUMNS, hint: HINT.trim() });
    view.input(editor);
    view.turn(true);
    view.onStream(chunk(1, "Once upon\na time"));
    const writes = recorder(output);
    await clock.advance(250);
    expect(writes).toHaveLength(1);
    const tick = writes[0]!.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "");
    // Up to the spinner's row, the row, and down again to the box's cursor.
    expect(tick.trimEnd()).toBe("\r✢ Thinking… (0s · esc to interrupt)");
    expect(writes[0]).not.toContain("\x1b[J");
  });

  test("a long reply streamed in 200 pieces never clears below, and writes each line once", () => {
    const { view, output, rows } = viewOn(16);
    const editor = new LineEditor({ columns: () => COLUMNS, rows: () => 16, hint: HINT.trim() });
    view.input(editor);
    view.turn(true);
    const lines = Array.from({ length: 40 }, (_, i) => `line ${String(i + 1).padStart(3, "0")}`);
    const text = `${lines.join("\n")}\n`;
    const writes = recorder(output);
    const size = Math.ceil(text.length / 200);
    for (let at = 0; at < text.length; at += size) {
      view.onStream(chunk(1, text.slice(at, at + size)));
    }
    expect(writes.length).toBeGreaterThan(150);
    for (const write of writes) expect(write).not.toContain("\x1b[J");
    // A line is written whole once: as it ends, where it was streaming. Printed for good as the
    // reply outgrows the screen, it stays where it is.
    const all = writes.join("");
    for (const line of lines) expect(all.split(line).length - 1).toBe(1);
    view.onEvent(finished(1));
    view.turn(false);
    expect(rows().filter((row) => row.includes("line ")).map((row) => row.slice(2))).toEqual(lines);
  });

  test("the box, its hint and the run list are the same at the prompt and mid-reply", async () => {
    const { view, output, rows } = viewOn();
    const input = new FakeInput({ tty: true });
    stdin = input;
    const onActive = (active: boolean, printed: boolean) => view.active(active, printed);
    const terminal = new Terminal({ input, output, onActive });
    const row = { key: "r1", text: "r1  color.yaml  ● running · 3s" };
    const list = { rows: () => [row], hint: "↓ runs" };
    const editor = terminal.lineEditor({ hint: view.hint, list });
    // Between turns the view keeps the box, as the chat leaves it.
    view.input(editor);
    const read = terminal.readLine("You:", { frame: true, escape: true, editor });
    await tick();
    const atPrompt = rows();
    expect(atPrompt).toEqual([TOP, boxed(">"), BOTTOM, HINT, "  r1  color.yaml  ● running · 3s"]);
    const before = output.text.length;
    input.write("hi\r");
    expect(await read).toEqual({ value: "hi" });
    view.turn(true);
    expect(rows()).toEqual(["> hi", SPIN, ...atPrompt]);
    // Neither the prompt ending nor the turn starting cleared the screen below.
    expect(output.text.slice(before)).not.toContain("\x1b[J");
    view.turn(false);
    expect(rows()).toEqual(["> hi", ...atPrompt]);
  });
});

describe("Ctrl-L", () => {
  const SPIN = "· Thinking… (0s · esc to interrupt)";
  const boxed = (text: string) => `│ ${text.padEnd(COLUMNS - 4)} │`;

  test("at the prompt and mid-reply draws each live row again in place, once", async () => {
    const { view, output, rows } = viewOn();
    const input = new FakeInput({ tty: true });
    stdin = input;
    const onActive = (active: boolean, printed: boolean) => view.active(active, printed);
    const terminal = new Terminal({ input, output, onActive });
    const list = { rows: () => [{ key: "r1", text: "r1  color.yaml" }], hint: "↓ runs" };
    const editor = terminal.lineEditor({ hint: view.hint, list });
    view.print("earlier\n");
    view.input(editor);
    /**
     * Press Ctrl-L, and check it wrote one frame that goes up to the live rows' top, clears below
     * and holds each row of `live` once, the transcript above it left where it was.
     */
    const press = async (live: string[]) => {
      const shown = rows();
      expect(shown[0]).toBe("earlier");
      expect(shown.slice(-live.length)).toEqual(live);
      const before = { text: output.text.length, writes: output.writes };
      input.write(CTRL_L);
      await tick();
      const text = output.text.slice(before.text);
      expect(output.writes).toBe(before.writes + 1);
      expect(text.startsWith(SYNC_ON)).toBe(true);
      expect(text.endsWith(SYNC_OFF)).toBe(true);
      expect(text).toMatch(/^\x1b\[\?2026h\x1b\[\?25l\x1b\[\d+A\r\x1b\[J/);
      expect(text).not.toContain("\x1b[2J");
      expect(text).not.toContain("\x1b[3J");
      const plain = text.replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "");
      expect(plain).not.toContain("earlier");
      for (const row of live) expect(plain.split(row.trim())).toHaveLength(2);
      expect(rows()).toEqual(shown);
      expect(screen(output.text, COLUMNS).scrollback).toEqual([]);
    };
    const read = terminal.readLine("You:", { frame: true, escape: true, editor });
    input.write("hi");
    await tick();
    await press([TOP, boxed("> hi"), BOTTOM, HINT, "  r1  color.yaml"]);
    input.write(ENTER);
    expect(await read).toEqual({ value: "hi" });
    // Mid-reply, as the chat runs a turn: the box under the spinner takes the keys.
    const keys = new AbortController();
    terminal.watchKeys(keys.signal, () => {}, (key) => {
      editor.key(key);
      view.refresh();
      return true;
    });
    view.turn(true);
    view.onStream(chunk(1, "Once upon\na dragon"));
    input.write("more");
    await tick();
    const reply = ["⏺ Once upon", "  a dragon", SPIN];
    await press([...reply, TOP, boxed("> more"), BOTTOM, HINT, "  r1  color.yaml"]);
    // In the box, below the transcript that stayed above the reply.
    const top = rows().indexOf(reply[0]!);
    expect(screen(output.text, COLUMNS).cursor).toEqual({ row: top + 4, col: 8 });
    // The next frame goes on from there, rewriting only what changed.
    view.onStream(chunk(1, " slept"));
    const slept = ["⏺ Once upon", "  a dragon slept", SPIN];
    expect(rows().slice(0, top + 3)).toEqual(["earlier", "> hi", "", ...slept]);
    keys.abort();
    view.turn(false);
  });
});

describe("the user's lines", () => {
  test("are on a gray band across the width, the > bold, wrapped rows banded too", () => {
    const { view, output, rows } = viewOn();
    view.echo(`${"word ".repeat(15)}end`);
    const band = "\x1b[48;5;237;38;5;252m";
    const written = output.text.split("\r\n").filter((row) => row.includes(band));
    expect(written).toHaveLength(2);
    expect(written[0]).toContain(`${band}\x1b[1m>\x1b[22m word word`);
    for (const row of written) expect(cells(row.slice(row.indexOf(band)))).toBe(COLUMNS);
    expect(rows()).toEqual([`> ${"word ".repeat(11).trim()}`, `${"word ".repeat(4)}end`]);
  });

  test("are plain `> text` with NO_COLOR", () => {
    const output = Object.assign(new FakeOutput({ columns: COLUMNS }), { isTTY: true });
    const env = (name: string) => (name === "NO_COLOR" ? "1" : undefined);
    const view = new ChatView(output, env, manualClock());
    view.echo("hello there");
    expect(output.text).toContain("> hello there\x1b[K");
    expect(output.text.replace(/\x1b\[\?[0-9]+[hl]|\x1b\[K|\r|\n/g, "")).toBe("> hello there");
  });
});

describe("the run list", () => {
  const status = (fields: Partial<ChildStatus>): ChildStatus => ({ run: "r1", seq: 1, log: "l", status: "running", ...fields });
  const children: LiveChild[] = [
    { id: "r1", name: "color.yaml", since: 0, status: status({}) },
    { id: "r2", name: "ask.yaml", since: 5_000, status: status({ status: "waiting", asks: [{ call: "c", question: "?" }] }) },
    { id: "r10", name: "notes.yaml", since: 0, status: status({ status: "waiting", waits: [{ tool: "take", settled: false }] }) },
    { id: "r11", name: "nap.yaml", since: 0, status: status({ status: "waiting", waits: [{ tool: "sleep", settled: false }] }) },
  ];

  test("has a row per run, its columns lined up, coloured by what the run does", () => {
    const out = { isTTY: true };
    const rows = runRows(children, 72_000, 60, out, () => undefined);
    expect(rows.map((row) => row.replace(/\x1b\[[0-9;]*m/g, ""))).toEqual([
      "r1   color.yaml  ● running · 1m12s",
      "r2   ask.yaml    ● asks · 1m07s",
      "r10  notes.yaml  ● waits for a message · 1m12s",
      "r11  nap.yaml    ● waits · 1m12s",
    ]);
    expect(rows[0]).toContain("\x1b[36m● running");
    expect(rows[1]).toContain("\x1b[33m● asks");
    expect(rows[3]).toContain("\x1b[2m● waits");
  });

  test("is plain with NO_COLOR, and each row is cut to the width", () => {
    const rows = runRows(children.slice(0, 1), 0, 20, { isTTY: true }, (name) => (name === "NO_COLOR" ? "1" : undefined));
    expect(rows).toEqual(["r1  color.yaml  ● ru"]);
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

  test("summary shows a JSON record's fields, a null one left out", () => {
    const written = JSON.stringify({ written: true, path: "a.yaml", reason: null });
    expect(summary(written, 60)).toEqual(["written: true · path: a.yaml"]);
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
    await pastGrace();
    stdin.write(ENTER);
    // The turn ends once the chat has named itself, its spinner gone.
    const ended = () =>
      model.named.length === 1 && !rows().some((line) => line.includes("Thinking…"));
    await until(() => ended() && rows().some((line) => line.startsWith(prompt)));
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
        `  ⎿  ok: true · name: hello · inputs: [{name: "name", ty…`,
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

/**
 * A model whose first reply streams "Once upon", then waits for `open` before it ends; each later
 * reply is "Short.". The call naming the chat is kept apart (`named`) and answered `A chat`, so
 * `requests` holds only the replies, whenever the naming call comes.
 */
class GatedModel implements ChatModel {
  readonly provider = "xai";
  readonly model = "grok-4.3";
  readonly requests: Request[] = [];
  readonly named: Request[] = [];
  readonly #gate = Promise.withResolvers<void>();
  readonly open = () => this.#gate.resolve();

  stream(request: Request, options: { signal?: AbortSignal } = {}): EventStream {
    if (namingRequest(request)) {
      this.named.push(request);
      return new ScriptedModel([textReply("A chat")], this).stream(request, options);
    }
    this.requests.push(request);
    const first = this.requests.length === 1;
    const gate = this.#gate.promise;
    async function* events() {
      if (first) {
        yield new TextDelta(0, "Once upon");
        await gate;
        yield new TextDelta(0, " a time.");
      }
      const text = first ? "Once upon a time." : "Short.";
      if (!first) yield new TextDelta(0, text);
      yield new PartDone(0, new TextPart(text));
      yield new Finish({ reason: "stop" });
    }
    return new EventStream(events());
  }
}

/** Whether a request's messages hold `text` in a user message. */
const asked = (request: Request, text: string) =>
  request.messages.some((message) => message.role === "user" && message.text().includes(text));

/** Wait past readline's escape timeout, so a lone Escape written before arrives as the Escape key. */
const pastEscape = () => new Promise((resolve) => setTimeout(resolve, 600));

describe("the input box while a reply runs", () => {
  test(
    "draws a chunk of keys typed into it once, not once per key",
    async () => {
      const { io, stdin, output, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      const before = output.writes;
      // 37 full rows of 54, then the last two.
      stdin.write("x".repeat(2000));
      await until(() => rows().includes(row("  xx")));
      expect(output.writes - before).toBeLessThan(5);
      model.open();
      await until(() => rows().at(-1) === HINT && !rows().some((line) => line.includes("Thinking")));
      stdin.write(`\x15/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "stays up under the spinner, and a line entered there steers the reply at once",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      // The box is under the spinner, which names the model, ready for a line.
      const spinner = "· Thinking… (0s · xai/grok-4.3 · esc to interrupt)";
      expect(rows().slice(-5)).toEqual([spinner, TOP, row(">"), BOTTOM, HINT]);
      stdin.write("make it short");
      await until(() => rows().includes(row("> make it short")));
      stdin.write(ENTER);
      await until(() => rows().includes("> make it short"));
      // The reply hasn't ended: the line went to the inbox while it ran, and it reads it next.
      expect(model.requests.length).toBe(1);
      model.open();
      await until(() => model.requests.length === 2 && rows().includes("⏺ Short."));
      expect(asked(model.requests[1]!, "make it short")).toBe(true);
      await until(() => rows().at(-1) === HINT && !rows().some((line) => line.includes("Thinking")));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      const shown = rows();
      expect(shown.slice(shown.indexOf("> tell me a story"))).toEqual([
        "> tell me a story",
        "",
        // Printed for good above the reply's line still streaming, which goes on below it.
        "> make it short",
        "⏺ Once upon a time.",
        "",
        "⏺ Short.",
        "",
        "> /exit",
        "session ended",
      ]);
    },
    TIMEOUT,
  );
});

describe("the first line of a chat", () => {
  test(
    "repaints the box in place from Enter through the reply's start, clearing nothing below",
    async () => {
      const { io, stdin, output, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      const before = output.text.length;
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      expect(output.text.slice(before)).not.toContain("\x1b[J");
      model.open();
      const idle = () => rows().at(-1) === HINT && !rows().some((line) => line.includes("Thinking"));
      await until(idle);
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      // /exit at the prompt takes the box down: what the chat prints after it draws no box.
      expect(output.text.slice(output.text.lastIndexOf("session ended"))).not.toContain("╭");
      expect(rows().at(-1)).toBe("session ended");
    },
    TIMEOUT,
  );
});

describe("the command menu", () => {
  /** The menu's row for `name` and its description, without its styles. */
  const menuRow = (name: string, description: string) => `  ${name.padEnd(11)}${description}`;
  const MODEL = menuRow("/model", "Choose the model for the next replies");
  const QUIT = menuRow("/quit", "Leave the chat");

  test(
    "/ lists the chat's commands at the prompt, and Enter runs the highlighted one",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const offered = async () => ({ models: [{ provider: "xai", model: "grok-4" }], failed: [] });
      const chatting = runChat(io, new GatedModel(), undefined, {
        ticker: manualClock(),
        modelChoices: offered,
      });
      await until(() => rows().includes(HINT));
      stdin.write("/");
      await until(() => rows().includes(QUIT));
      // Above the box, its hint under it.
      expect(rows().slice(-9)).toEqual([
        MODEL,
        menuRow("/provider", "Choose a provider and its model, signing in"),
        menuRow("/resume", "Go on from an earlier chat in this folder"),
        menuRow("/exit", "Leave the chat"),
        QUIT,
        TOP,
        row("> /"),
        BOTTOM,
        HINT,
      ]);
      stdin.write(ENTER);
      await until(() => rows().some((line) => line.includes("Model for the next replies")));
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().includes(HINT));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "/ opens it while a reply runs: Esc closes it without withdrawing, and /exit typed whole runs",
    async () => {
      const { io, stdin, output, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write("/");
      await until(() => rows().includes(QUIT));
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => !rows().includes(QUIT));
      expect(rows()).toContain(row("> /"));
      expect(output.text).not.toContain("withdrew");
      expect(output.text).not.toContain("stopping the reply");
      // Typed whole, Enter runs /exit as typed; it takes the box down once sent.
      stdin.write(`exit${ENTER}`);
      await until(() => rows().includes("> /exit") && !rows().includes(HINT));
      model.open();
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );
});

describe("the command menu over the box", () => {
  const SPIN = "· Thinking… (0s · esc to interrupt)";
  const boxed = (text: string) => `│ ${text.padEnd(COLUMNS - 4)} │`;
  const COMMANDS = [
    { name: "/model", description: "Choose the model" },
    { name: "/exit", description: "Leave the chat" },
  ];
  const MENU = ["  /model  Choose the model", "  /exit   Leave the chat"];

  test("sits above the box at the prompt and mid-reply, the run list under it", async () => {
    const { view, output, rows } = viewOn();
    const input = new FakeInput({ tty: true });
    stdin = input;
    const onActive = (active: boolean, printed: boolean) => view.active(active, printed);
    const terminal = new Terminal({ input, output, onActive });
    const run = "  r1  color.yaml  ● running · 3s";
    const list = { rows: () => [{ key: "r1", text: run.trim() }], hint: "↓ runs" };
    const editor = terminal.lineEditor({ hint: view.hint, list, commands: COMMANDS });
    view.input(editor);
    const read = terminal.readLine("You:", { frame: true, escape: true, editor });
    input.write("/");
    await tick();
    expect(rows()).toEqual([...MENU, TOP, boxed("> /"), BOTTOM, HINT, run]);
    expect(screen(output.text, COLUMNS).cursor).toEqual({ row: 3, col: 5 });
    // The typed slash is bold in each name.
    expect(editor.lines()[1]).toContain("\x1b[1m/\x1b[22mexit");
    input.write(ESCAPE);
    await pastEscape();
    input.write("\x15hi\r");
    expect(await read).toEqual({ value: "hi" });
    // Mid-reply, as the chat runs a turn: the menu goes between the spinner and the box.
    const keys = new AbortController();
    terminal.watchKeys(keys.signal, () => {}, (key) => {
      editor.key(key);
      view.refresh();
      return true;
    });
    view.turn(true);
    input.write("/");
    await tick();
    expect(rows()).toEqual(["> hi", SPIN, ...MENU, TOP, boxed("> /"), BOTTOM, HINT, run]);
    expect(screen(output.text, COLUMNS).cursor).toEqual({ row: 5, col: 5 });
    // Ctrl-L draws the menu again with the rest, in place.
    const before = output.text.length;
    input.write(CTRL_L);
    await tick();
    const again = output.text.slice(before).replace(/\x1b\[[0-9;?]*[A-Za-z]/g, "");
    for (const each of MENU) expect(again).toContain(each.trim());
    expect(rows()).toEqual(["> hi", SPIN, ...MENU, TOP, boxed("> /"), BOTTOM, HINT, run]);
    keys.abort();
    view.turn(false);
  });
  test("capped to a short screen mid-reply, it leaves the spinner and reply drawn", async () => {
    const { view, output, rows } = viewOn(16);
    const input = new FakeInput({ tty: true });
    stdin = input;
    const onActive = (active: boolean, printed: boolean) => view.active(active, printed);
    const terminal = new Terminal({ input, output, onActive });
    const many = Array.from({ length: 12 }, (_, i) => ({ name: `/c${i}`, description: "" }));
    const list = { rows: () => [{ key: "r1", text: "r1  color.yaml" }], hint: "↓ runs" };
    const editor = terminal.lineEditor({ hint: view.hint, list, commands: many });
    view.input(editor);
    const keys = new AbortController();
    terminal.watchKeys(keys.signal, () => {}, (key) => {
      editor.key(key);
      view.refresh();
      return true;
    });
    view.turn(true);
    view.onStream(chunk(1, "Once upon"));
    input.write("/");
    await tick();
    // Sixteen rows: the menu keeps `LIST_MARGIN` rows free, for the reply and the spinner.
    const menu = [...Array.from({ length: 6 }, (_, i) => `  /c${i}`), "  +6 more"];
    const box = [TOP, boxed("> /"), BOTTOM, HINT, "  r1  color.yaml"];
    expect(rows().slice(-14)).toEqual(["⏺ Once upon", SPIN, ...menu, ...box]);
    // The reply's line is still live, not printed for good above the live rows.
    expect(output.text.split("Once upon").length - 1).toBe(1);
    keys.abort();
    view.turn(false);
  });
});

describe("the input box after a line is entered while a reply runs", () => {
  test(
    "/exit there takes the box down, and keys after it go nowhere",
    async () => {
      const { io, stdin, output, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`/exit${ENTER}`);
      await until(() => rows().includes("> /exit") && !rows().includes(HINT));
      stdin.write(`abc${ENTER}`);
      await new Promise((resolve) => setTimeout(resolve, 50));
      model.open();
      expect(await chatting).toBe(0);
      expect(output.text).not.toContain("abc");
      expect(rows().at(-1)).toBe("session ended");
    },
    TIMEOUT,
  );

  test(
    "Esc takes a line just entered back into the box, and the reply never reads it",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`make it short${ENTER}${ESCAPE}`);
      await until(() => rows().includes("(withdrew 1 message; back in the prompt)"));
      expect(rows()).toContain(row("> make it short"));
      model.open();
      await until(() => rows().at(-1) === HINT && !rows().some((line) => line.includes("Thinking")));
      expect(model.requests.some((request) => asked(request, "make it short"))).toBe(false);
      stdin.write(`\x15/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );
});

describe("a paste into the input box while a reply runs", () => {
  test(
    "is one line, sent once on Enter, even when it holds /exit",
    async () => {
      const { io, stdin, output, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      // Bracketed paste is on while the box is up under the spinner.
      expect(screen(output.text, COLUMNS).modes.filter((mode) => mode.startsWith("?2004")).at(-1)).toBe("?2004h");
      stdin.write(`${PASTE_START}one\r/exit\rthree${PASTE_END}`);
      await until(() => rows().includes(row("> one⏎/exit⏎three")));
      expect(output.text).not.toContain("> one");
      stdin.write(ENTER);
      await until(() => rows().includes("> one⏎/exit⏎three"));
      model.open();
      await until(() => model.requests.length === 2 && rows().at(-1) === HINT);
      const said = model.requests[1]!.messages.filter((message) => message.role === "user").at(-1)!.text();
      expect(said).toBe("one\n/exit\nthree");
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );
});

/** A paste of `count` lines, as a terminal sends it, and the text it reaches the chat as. */
const pastedLines = (count: number) =>
  Array.from({ length: count }, (_, i) => `line ${i + 1}\r`).join("");
const pastedText = (count: number) =>
  Array.from({ length: count }, (_, i) => `line ${i + 1}\n`).join("");

/** Whether the chat waits at its prompt again, the reply done. */
const prompting = (rows: string[]) =>
  rows.at(-1) === HINT && !rows.some((line) => line.includes("Thinking"));

/** The text of the last user message of `request`. */
const lastSaid = (request: Request) =>
  request.messages.filter((message) => message.role === "user").at(-1)!.text();

describe("a long paste in the input box", () => {
  test(
    "is a token at the prompt, echoed as the token, and the model gets the whole text",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new AnsweringModel(() => textReply("got it"));
      const chatting = runChat(io, model);
      await until(() => rows().includes(HINT));
      stdin.write(`look ${PASTE_START}${pastedLines(40)}${PASTE_END}`);
      await until(() => rows().includes(row("> look [Pasted text #1 +40 lines]")));
      stdin.write(ENTER);
      await until(() => rows().includes("⏺ got it") && rows().at(-1) === HINT);
      expect(rows()).toContain("> look [Pasted text #1 +40 lines]");
      expect(lastSaid(model.requests[0]!)).toBe(`look ${pastedText(40)}`);
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "is a token in the box while a reply runs, sent whole on Enter and echoed as the token",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`count the lines of this paste, then say its last one: `);
      stdin.write(`${PASTE_START}${pastedLines(40)}${PASTE_END}`);
      await until(() => rows().includes(row("  [Pasted text #1 +40 lines]")));
      stdin.write(ENTER);
      // The echo wraps rather than cutting the token.
      await until(() => rows().includes("> count the lines of this paste, then say its last one:"));
      expect(rows()).toContain("[Pasted text #1 +40 lines]");
      model.open();
      await until(() => model.requests.length === 2 && rows().at(-1) === HINT);
      expect(lastSaid(model.requests[1]!)).toBe(
        `count the lines of this paste, then say its last one: ${pastedText(40)}`,
      );
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "leaves a typed line as long as a long paste as text when Esc withdraws it",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`${"x".repeat(900)}${ENTER}${ESCAPE}`);
      await until(() => rows().includes("(withdrew 1 message; back in the prompt)"));
      expect(rows()).toContain(row(`> ${"x".repeat(54)}`));
      expect(rows().join("\n")).not.toContain("[Pasted text");
      model.open();
      await until(() => prompting(rows()));
      stdin.write(`\x15/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "is printed as the token when Esc withdraws it after /exit",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`${PASTE_START}${pastedLines(40)}${PASTE_END}${ENTER}/exit${ENTER}${ESCAPE}`);
      await until(() => rows().includes("(withdrew 1 message: [Pasted text #1 +40 lines])"));
      model.open();
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "comes back as the token when Esc withdraws it, and is sent whole from there",
    async () => {
      const { io, stdin, rows } = terminalIo();
      const model = new GatedModel();
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`tell me a story${ENTER}`);
      await until(() => rows().includes("⏺ Once upon"));
      stdin.write(`see ${PASTE_START}${pastedLines(40)}${PASTE_END}${ENTER}${ESCAPE}`);
      await until(() => rows().includes("(withdrew 1 message; back in the prompt)"));
      expect(rows()).toContain(row("> see [Pasted text #1 +40 lines]"));
      model.open();
      await until(() => prompting(rows()));
      expect(rows()).toContain(row("> see [Pasted text #1 +40 lines]"));
      stdin.write(ENTER);
      await until(() => model.requests.some((request) => asked(request, "line 40")));
      expect(lastSaid(model.requests.at(-1)!)).toBe(`see ${pastedText(40)}`);
      await until(() => prompting(rows()));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );
});

const COLOR_FLOW = `id: color
name: color
nodes:
  pick:
    kind: tool
    tool_id: ask
    input:
      prompt: "Which colour?"
    output: str
output: \${pick.output}
`;

describe("the run list under the input box", () => {
  test(
    "lists a run waiting on its question; Down moves into it, Esc back, and Enter opens its view",
    async () => {
      writeFileSync(join(root, "color.yaml"), COLOR_FLOW);
      const { io, stdin, output, rows } = terminalIo();
      const script: ScriptedReply[] = [
        callReply(["start_flow", { path: "color.yaml", inputs: {} }, "c1"]),
        textReply("Started color.yaml (r1)."),
        textReply("r1 asks which colour you want."),
      ];
      const model = new AnsweringModel((_, count) => script[count - 1] ?? textReply("noted"));
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`pick a colour${ENTER}`);
      await until(() => output.text.includes("Run color.yaml"));
      await pastGrace();
      stdin.write(ENTER);
      const listed = /^ {2}r1 {2}color\.yaml {2}● asks · \d+s$/;
      // The turn ends once the chat has named itself, its spinner gone.
      await until(
        () =>
          model.named.length === 1 &&
          !rows().some((line) => line.includes("Thinking…")) &&
          rows().at(-1)!.match(listed) !== null,
      );
      const before = rows();
      // At 60 columns the list's hint doesn't fit after the box's.
      expect(before.slice(-5, -1)).toEqual([TOP, row(">"), BOTTOM, HINT]);
      stdin.write(DOWN);
      await until(() => rows().at(-1)!.startsWith("» r1"));
      // Esc in the list goes back to the box and withdraws nothing.
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().at(-1)!.match(listed) !== null);
      expect(output.text).not.toContain("withdrew");
      expect(output.text).not.toContain("stopping the reply");
      stdin.write(DOWN);
      await until(() => rows().at(-1)!.startsWith("» r1"));
      stdin.write(ENTER);
      await until(() => rows()[0] === "r1  color.yaml  ● asks");
      const view = rows();
      expect(view[1]).toMatch(/^log {2}\/\S+$/);
      expect(view).toContainEqual(expect.stringMatching(/^◷ pick \[\d+s\] waiting$/));
      expect(view.slice(-3)).toEqual(["? Which colour?", "", "esc to go back to the chat"]);
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().at(-1)!.startsWith("» r1"));
      // The chat comes back as it was, the row still in focus.
      expect(rows().slice(0, -1)).toEqual(before.slice(0, -1));
      // A typed character goes back to the box.
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      expect(rows()).toContain("> /exit");
    },
    TIMEOUT,
  );
});

/**
 * A model answering by what a request holds: `answer` gets the request and gives a reply, or a
 * promise of one, so a test can hold a reply back.
 */
class StepModel implements ChatModel {
  readonly provider = "xai";
  readonly model = "grok-4.3";
  readonly requests: Request[] = [];
  readonly #answer: (request: Request) => ScriptedReply | Promise<ScriptedReply>;

  constructor(answer: (request: Request) => ScriptedReply | Promise<ScriptedReply>) {
    this.#answer = answer;
  }

  stream(request: Request): EventStream {
    this.requests.push(request);
    const reply = this.#answer(request);
    const model = this;
    async function* events() {
      yield* new ScriptedModel([await reply], model).stream(request);
    }
    return new EventStream(events());
  }
}

/** Whether a request's last message is a tool's result. */
const afterTool = (request: Request) => request.messages.at(-1)?.role === "tool";

const BLUE = `id: blue
name: blue
nodes:
  say: !str blue
output: \${say.output}
`;

const NAP_FLOW = `id: nap
name: nap
nodes:
  nap:
    kind: tool
    tool_id: sleep
    input:
      for: "2s"
  done:
    kind: code
    input:
      slept: \${nap.output}
    output: str
    code: |
      return "rested";
output: \${done.output}
`;

describe("a run's view while the chat goes on", () => {
  test(
    "opens from the list while a reply runs; Esc in the list withdraws nothing; an approval behind it is said",
    async () => {
      writeFileSync(join(root, "color.yaml"), COLOR_FLOW);
      const { io, stdin, output, rows } = terminalIo();
      const gate = Promise.withResolvers<void>();
      let started = false;
      let wrote = false;
      const model = new StepModel((request) => {
        if (afterTool(request)) return textReply("done.");
        if (!started && asked(request, "pick a colour")) {
          started = true;
          return callReply(["start_flow", { path: "color.yaml", inputs: {} }, "c1"]);
        }
        if (!wrote && asked(request, "write it")) {
          wrote = true;
          return gate.promise.then(() => callReply(["write_flow", { path: "blue.yaml", content: BLUE }, "c2"]));
        }
        return textReply("noted.");
      });
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      const idle = () => rows().at(-1)!.includes("● asks") && !rows().some((line) => line.includes("Thinking"));
      await until(() => rows().includes(HINT));
      stdin.write(`pick a colour${ENTER}`);
      await until(() => output.text.includes("Run color.yaml"));
      await pastGrace();
      stdin.write(ENTER);
      await until(idle);
      stdin.write(`write it${ENTER}`);
      await until(() => wrote && rows().some((line) => line.includes("Thinking")));
      // While the reply runs, Down moves into the list, and Esc there only comes back.
      stdin.write(DOWN);
      await until(() => rows().at(-1)!.startsWith("» r1  color.yaml"));
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().at(-1)!.startsWith("  r1  color.yaml"));
      expect(output.text).not.toContain("withdrew");
      expect(output.text).not.toContain("stopping the reply");
      expect(rows().some((line) => line.includes("Thinking"))).toBe(true);
      stdin.write(DOWN);
      await until(() => rows().at(-1)!.startsWith("» r1"));
      stdin.write(ENTER);
      await until(() => rows()[0] === "r1  color.yaml  ● asks");
      expect(rows().at(-1)).toBe("esc to go back to the chat");
      // The reply asks to write a file: its approval waits behind the view, which says so.
      gate.resolve();
      await until(() => rows().at(-1) === "the chat waits on you · esc to answer");
      expect(rows()[0]).toBe("r1  color.yaml  ● asks");
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().includes("? Write blue.yaml (Use arrow keys)"));
      // What the chat drew while the view was up is printed above the card, not after it.
      const shown = rows();
      const call = shown.findIndex((line) => line.startsWith("⏺ write_flow(blue.yaml"));
      expect(call).toBeGreaterThan(-1);
      expect(call).toBeLessThan(shown.findIndex((line) => line.startsWith("╭─ Write blue.yaml")));
      await pastGrace();
      stdin.write(ENTER);
      await until(() => rows().includes("⏺ done.") && idle());
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );

  test(
    "opened from the prompt, lets the chat take a run's event in a turn of its own before Esc",
    async () => {
      writeFileSync(join(root, "nap.yaml"), NAP_FLOW);
      const { io, stdin, output, rows } = terminalIo();
      let started = false;
      const model = new StepModel((request) => {
        if (afterTool(request)) return textReply("Started.");
        if (!started && asked(request, "start nap")) {
          started = true;
          return callReply(["start_flow", { path: "nap.yaml", inputs: {} }, "c1"]);
        }
        return asked(request, "run r1 succeeded") ? textReply("r1 is done.") : textReply("noted.");
      });
      const chatting = runChat(io, model, undefined, { ticker: manualClock() });
      await until(() => rows().includes(HINT));
      stdin.write(`start nap${ENTER}`);
      await until(() => output.text.includes("Run nap.yaml"));
      await pastGrace();
      stdin.write(ENTER);
      await until(() => rows().at(-1)!.includes("● waits") && !rows().some((line) => line.includes("Thinking")));
      stdin.write(DOWN);
      await until(() => rows().at(-1)!.startsWith("» r1"));
      stdin.write(ENTER);
      await until(() => rows()[0]?.startsWith("r1  nap.yaml") === true);
      // The run ends while its view is up, and the chat's turn for that runs meanwhile.
      await until(() => model.requests.some((request) => asked(request, "run r1 succeeded")));
      expect(rows()[0]).toStartWith("r1  nap.yaml");
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().includes("⏺ r1 is done.") && rows().at(-1) === HINT);
      expect(rows()).toContain("⏺ r1 succeeded");
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
    },
    TIMEOUT,
  );
});

/** Every file under `dir`, read as text. */
function filesUnder(dir: string): string[] {
  return readdirSync(dir, { recursive: true, withFileTypes: true })
    .filter((entry) => entry.isFile())
    .map((entry) => readFileSync(join(entry.parentPath, entry.name), "utf8"));
}

/** The models the fake deepseek listing gives. */
const DEEPSEEK_MODELS = ["deepseek-chat", "deepseek-reasoner"];

describe("/provider", () => {
  test(
    "/provider signs in to a provider without a login, its model picked once; a cancel keeps it",
    async () => {
      const key = "sk-deepseek-typed-secret";
      const sentKeys: (string | null)[] = [];
      const listModels = async (_: unknown, options: { apiKey?: string | null } = {}) => {
        sentKeys.push(options.apiKey ?? null);
        return DEEPSEEK_MODELS;
      };
      const { io, stdin, output, rows } = terminalIo();
      const screen = () => output.text;
      const model = new AnsweringModel(() => textReply("noted"));
      const access = chatServing(model, io.env);
      const log = join(tmp, "chat.jsonl");
      const chatting = chat(undefined, { engineTrace: false, log }, io, {
        root,
        models: access,
        signals: new EventEmitter(),
        ticker: manualClock(),
        auth: { listModels },
      });
      const shownSince = (from: number, text: string) => screen().slice(from).includes(text);
      await until(() => rows().includes(HINT));
      // openai, xai, deepseek: deepseek has no login yet.
      let from = screen().length;
      stdin.write(`/provider${ENTER}`);
      await until(() => shownSince(from, "Provider for the next replies"));
      // A searchable picker, on the current provider, each login dim after its name.
      await until(() => rows().includes("» xai  key from XAI_API_KEY (current)"));
      expect(rows()).toContain("  deepseek  not logged in");
      stdin.write("deeps");
      await until(() => rows().at(-1) === "» deepseek  not logged in");
      stdin.write(ENTER);
      await until(() => shownSince(from, "deepseek API key:"));
      stdin.write(key + ENTER);
      // The chat's model picker stands for yian auth's "Model:": the login is saved with its pick.
      await until(() => shownSince(from, "Model for the next replies"));
      stdin.write(DOWN + ENTER);
      const chosen = "(model: deepseek/deepseek-reasoner, from the next reply)";
      await until(() => shownSince(from, chosen));
      expect(screen()).toContain("Saved deepseek login (api-key) with model deepseek-reasoner");
      expect(screen()).not.toContain("Model:");
      // Logged in now: picking it again goes straight to its models.
      from = screen().length;
      stdin.write(`/provider${ENTER}`);
      await until(() => rows().includes("» deepseek  logged in (current)"));
      stdin.write(ENTER);
      await until(() => shownSince(from, "Model for the next replies"));
      expect(screen().slice(from)).not.toContain("API key:");
      stdin.write(ESCAPE);
      await new Promise((resolve) => setTimeout(resolve, 600));
      // A cancel during a sign-in keeps the model.
      from = screen().length;
      stdin.write(`/provider${ENTER}`);
      await until(() => shownSince(from, "Provider for the next replies"));
      stdin.write(`openai${ENTER}`);
      await until(() => shownSince(from, "OpenAI authentication method:"));
      stdin.write(ESCAPE);
      await until(() => shownSince(from, "the model is unchanged"));
      stdin.write(`hello${ENTER}`);
      await until(() => screen().includes("noted"));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      const built = access.configs.map((config) => `${config["provider"]}/${config["model"]}`);
      expect(built).toEqual(["deepseek/deepseek-reasoner"]);
      expect(sentKeys).toContain(key);
      // The key went to the saved login, and nowhere the chat writes.
      const authFile = join(home, ".config", "yian", "auth.json");
      expect(readFileSync(authFile, "utf8")).toContain(key);
      const written = [screen(), readFileSync(log, "utf8"), ...filesUnder(join(home, ".yian"))];
      for (const text of written) expect(text).not.toContain(key);
    },
    TIMEOUT,
  );

  test(
    "a browser sign-in waiting in /provider is cancelled by Esc or Ctrl-C, and the model kept",
    async () => {
      const opened: string[] = [];
      // A browser login that never completes: it ends only when its signal aborts.
      const openAIAuth = ({ signal }: { signal?: AbortSignal }) =>
        ({
          browserLogin: async (open: (url: string) => Promise<void> | void) => {
            await open("https://auth.example/login");
            return new Promise<never>((_, reject) =>
              signal!.addEventListener("abort", () => reject(signal!.reason), { once: true }),
            );
          },
        }) as unknown as ReturnType<NonNullable<NonNullable<ChatDeps["auth"]>["openAIAuth"]>>;
      const { io, stdin, output, rows } = terminalIo();
      const model = new AnsweringModel(() => textReply("noted"));
      const access = chatServing(model, io.env);
      const chatting = chat(undefined, { engineTrace: false, log: join(tmp, "chat.jsonl") }, io, {
        root,
        models: access,
        signals: new EventEmitter(),
        ticker: manualClock(),
        auth: { openAIAuth, openUrl: (url) => void opened.push(url) },
      });
      const shownSince = (from: number, text: string) => output.text.slice(from).includes(text);
      await until(() => rows().includes(HINT));
      for (const key of [ESCAPE, CTRL_C]) {
        const from = output.text.length;
        stdin.write(`/provider${ENTER}`);
        await until(() => shownSince(from, "Provider for the next replies"));
        // openai: a subscription, signed in in the browser.
        stdin.write(`openai${ENTER}`);
        await until(() => shownSince(from, "OpenAI authentication method:"));
        stdin.write(ENTER);
        await until(() => shownSince(from, "Subscription sign-in:"));
        stdin.write(ENTER);
        await until(() => shownSince(from, "(Esc or Ctrl-C cancels the sign-in)"));
        stdin.write(key);
        await until(() => shownSince(from, "(sign-in cancelled; the model is unchanged)"));
        await until(() => rows().includes(HINT));
      }
      expect(opened).toEqual(["https://auth.example/login", "https://auth.example/login"]);
      stdin.write(`hello${ENTER}`);
      await until(() => output.text.includes("noted"));
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      const built = access.configs.map((config) => `${config["provider"]}/${config["model"]}`);
      expect(built).toEqual(["xai/grok-4.3"]);
    },
    TIMEOUT,
  );

  test(
    "a key from the environment is saved only on Yes; No uses it for the chat, Esc changes nothing",
    async () => {
      const key = "sk-deepseek-env-secret";
      const listModels = async () => DEEPSEEK_MODELS;
      const { io, stdin, output, rows } = terminalIo({ extra: { DEEPSEEK_API_KEY: key } });
      const model = new AnsweringModel(() => textReply("noted"));
      const access = chatServing(model, io.env);
      const log = join(tmp, "chat.jsonl");
      const chatting = chat(undefined, { engineTrace: false, log }, io, {
        root,
        models: access,
        signals: new EventEmitter(),
        ticker: manualClock(),
        auth: { listModels },
      });
      const authFile = join(home, ".config", "yian", "auth.json");
      const shownSince = (from: number, text: string) => output.text.slice(from).includes(text);
      /** Pick deepseek in /provider and wait for the question about its key. */
      const pickDeepseek = async (): Promise<number> => {
        const from = output.text.length;
        stdin.write(`/provider${ENTER}`);
        await until(() => shownSince(from, "Provider for the next replies"));
        stdin.write("deepseek");
        await until(() => rows().at(-1)!.startsWith("» deepseek  key from DEEPSEEK_API_KEY"));
        stdin.write(ENTER);
        await until(() => shownSince(from, "Save this key as a yian login?"));
        return from;
      };
      await until(() => rows().includes(HINT));
      let from = await pickDeepseek();
      // Esc changes nothing.
      stdin.write(ESCAPE);
      await pastEscape();
      await until(() => rows().includes(HINT));
      expect(output.text.slice(from)).not.toContain("Model for the next replies");
      // No, highlighted first: the key serves the chat, and nothing is saved.
      from = await pickDeepseek();
      stdin.write(ENTER);
      await until(() => shownSince(from, "Model for the next replies"));
      stdin.write(DOWN + ENTER);
      const chosen = "(model: deepseek/deepseek-reasoner, from the next reply)";
      await until(() => shownSince(from, chosen));
      expect(existsSync(authFile)).toBe(false);
      stdin.write(`hello${ENTER}`);
      await until(() => output.text.includes("noted"));
      // Yes: saved as yian auth saves it, with no key prompt, then the model picked.
      from = await pickDeepseek();
      stdin.write(DOWN + ENTER);
      await until(() => shownSince(from, "Model for the next replies"));
      expect(output.text.slice(from)).not.toContain("API key:");
      stdin.write(ENTER);
      const saved = "Saved deepseek login (api-key) with model deepseek-chat";
      await until(() => shownSince(from, saved));
      expect(readFileSync(authFile, "utf8")).toContain(key);
      stdin.write(`/exit${ENTER}`);
      expect(await chatting).toBe(0);
      const built = access.configs.map((config) => `${config["provider"]}/${config["model"]}`);
      expect(built).toEqual(["deepseek/deepseek-reasoner"]);
      for (const text of [output.text, readFileSync(log, "utf8")]) expect(text).not.toContain(key);
    },
    TIMEOUT,
  );
});
