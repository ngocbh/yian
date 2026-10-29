/**
 * `yian chat`'s look on a terminal (`chat_view.ts`): its pieces on their own, and the chat drawn
 * on a fake TTY 60 columns wide, read back as the screen shows it, in colour and with `NO_COLOR`
 * (the plain lines).
 */

import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { EventEmitter } from "node:events";
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { chat, type ChatDeps } from "../../src/cli/chat.ts";
import { chatBanner, homePath, loginLabel } from "../../src/cli/chat_view.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
import type { Env } from "../../src/defaults.ts";
import { type ChatModel, EventStream } from "../../src/llm/events.ts";
import type { Request, Response } from "../../src/llm/messages.ts";
import { ScriptedModel } from "../../src/llm/testing.ts";
import { serving, textReply } from "../helpers/run_state_fixtures.ts";
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
  readonly #answer: (request: Request, count: number) => Response;

  constructor(answer: (request: Request, count: number) => Response) {
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
