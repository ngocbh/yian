/**
 * The look of `yian chat` on a terminal: a banner box, the assistant's replies as `⏺` blocks of
 * rendered Markdown, its tool calls and its child runs as `⏺` lines with a `⎿` result, a spinner
 * while a turn runs, and a framed prompt (drawn by the terminal). Everything the chat prints goes
 * through one writer, which keeps the spinner's line clear of other text.
 *
 * Off a terminal, or with `NO_COLOR`, the chat keeps its plain lines and none of this is used.
 *
 * Knows about: the box and colour helpers, the terminal's cell widths and the sanitizing of text
 * printed on the terminal.
 */

import type { Env } from "../defaults.ts";
import { box, type Colored, paint, type Span, wrapSpans } from "./render.ts";
import { sanitize } from "./sanitize.ts";
import type { CliIo, Writer } from "./utils.ts";

/** The hint under the chat's framed prompt. */
export const CHAT_HINT = "/exit to leave · ctrl-c interrupts · esc withdraws";

/** What the chat's banner shows. */
export interface BannerInfo {
  /** `provider/model` of the assistant's replies. */
  model: string;
  /** How the model is reached (`ChatGPT login`, `API key`), or null when not known. */
  login: string | null;
  /** The folder the chat works in, as shown (`~/...`). */
  cwd: string;
  /** The flow in focus as the assistant names it, or null for none. */
  focus: string | null;
}

/** An output the view draws on: a TTY, so colour is on, and its width. */
export type ViewOutput = Colored & { readonly columns?: number };

/**
 * Whether `yian chat` draws its look on `io`: stdin, stdout and stderr are all a TTY and
 * `NO_COLOR` is unset or empty.
 */
export function richChat(io: CliIo): boolean {
  const stderr = io.stderrStream as { isTTY?: boolean };
  return io.stdin.isTTY === true && stderr.isTTY === true && io.stdout.isTTY === true && !io.env("NO_COLOR");
}

/**
 * How a connection (an agent config's `_connection`) reaches the model: `ChatGPT login` for a
 * subscription, `API key` for a key that is there, null for anything else.
 */
export function loginLabel(connection: unknown): string | null {
  if (typeof connection !== "object" || connection === null) return null;
  const { method, source } = connection as { method?: unknown; source?: unknown };
  if (method === "subscription") return "ChatGPT login";
  return method === "api_key" && source !== "missing" ? "API key" : null;
}

/** `path` with `home` (when set and not the root) shown as `~`. */
export function homePath(path: string, home: string | undefined): string {
  const trimmed = home?.replace(/[\\/]+$/, "");
  if (!trimmed) return path;
  if (path === trimmed) return "~";
  for (const sep of ["/", "\\"]) {
    if (path.startsWith(trimmed + sep)) return `~${path.slice(trimmed.length)}`;
  }
  return path;
}

/** Width of the banner's label column: `  focus   `. */
const LABEL_CELLS = 10;

/**
 * The banner box across `out`'s width: `✻ yian chat`, then the model and how it is reached, the
 * folder and the flow in focus, each sanitized and wrapped under its label; ends in a newline.
 */
export function chatBanner(info: BannerInfo, out: ViewOutput, env: Env): string {
  const inner = Math.max(LABEL_CELLS + 1, (out.columns || 80) - 4);
  const rows: [string, string][] = [
    [`✻ yian chat`, `${paint("✻", "cyan", out, env)} ${paint("yian chat", "bold", out, env)}`],
  ];
  const field = (label: string, spans: Span[]) => {
    const head = `  ${label.padEnd(LABEL_CELLS - 2)}`;
    wrapSpans(spans, inner - LABEL_CELLS, out, env).forEach(([plain, printed], i) => {
      const lead = i === 0 ? head : " ".repeat(LABEL_CELLS);
      rows.push([lead + plain, (i === 0 ? paint(lead, "dim", out, env) : lead) + printed]);
    });
  };
  const model: Span[] = [[sanitize(info.model), null]];
  if (info.login !== null) model.push([` (${info.login})`, "dim"]);
  field("model", model);
  field("cwd", [[sanitize(info.cwd), null]]);
  if (info.focus !== null) field("focus", [[sanitize(info.focus), null]]);
  return `${box(rows, "dim", out, env, "", inner)}\n`;
}

/**
 * The chat's view on a terminal, drawn on `output`: what `inboxChat` and `yian chat` print goes
 * through `writer`.
 */
export class ChatView {
  /** The hint under the framed prompt. */
  readonly hint = CHAT_HINT;
  readonly #output: Pick<NodeJS.WritableStream, "write"> & ViewOutput;

  constructor(output: Pick<NodeJS.WritableStream, "write"> & ViewOutput) {
    this.#output = output;
  }

  /** A writer printing whole lines on the view; a `\n` is written as `\r\n`, raw mode or not. */
  readonly writer: Writer = {
    write: async (text) => this.print(text),
    isTTY: true,
  };

  /** Print `text`, whole lines. */
  print(text: string): void {
    this.#output.write(text.replaceAll("\r\n", "\n").replaceAll("\n", "\r\n"));
  }
}
