/**
 * The small, line-based Markdown `yian chat` draws an assistant's reply in: `**bold**`, `*italic*`,
 * `` `code` ``, `#` headings, `-` and `1.` lists, and fenced code in a frame, wrapped to the
 * terminal's width. Anything else prints as written. Lines are rendered one at a time
 * (`MarkdownLines`), so a reply can be drawn as it streams.
 *
 * Knows about: the box and colour helpers, the terminal's cell widths and the sanitizing of text
 * printed on the terminal.
 */

import type { Env } from "../defaults.ts";
import { cells, fit } from "./cells.ts";
import { type Colored, paint, type Span, type Style, wrapSpans } from "./render.ts";
import { sanitize } from "./sanitize.ts";

const FENCE = /^\s*(```|~~~)\s*([^`\s]*)/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^(\s*)[-*+]\s+(.*)$/;
const NUMBERED = /^(\s*)(\d{1,9}[.)])\s+(.*)$/;

/** Spaces a tab is drawn as, so each character takes the cells it is measured at. */
export const TAB = "    ";

/** `base` with `style` added, or `style` alone. */
function combine(base: Style | null, style: Style): Style {
  return base === null || base === style ? style : (`${style} ${base.split(" ")[0]}` as Style);
}

/**
 * The spans of `text`'s inline Markdown in `base`: `` `code` `` cyan, `**bold**` (or `__bold__`)
 * bold and `*italic*` italic, their markers dropped; a marker with no end stays as written.
 */
export function inlineSpans(text: string, base: Style | null = null): Span[] {
  const spans: Span[] = [];
  let plain = "";
  const flush = () => {
    if (plain) spans.push([plain, base]);
    plain = "";
  };
  let i = 0;
  while (i < text.length) {
    const rest = text.slice(i);
    if (rest.startsWith("`")) {
      const end = text.indexOf("`", i + 1);
      if (end > i + 1) {
        flush();
        spans.push([text.slice(i + 1, end), combine(base, "cyan")]);
        i = end + 1;
        continue;
      }
    }
    const strong = rest.startsWith("**") ? "**" : rest.startsWith("__") ? "__" : null;
    if (strong !== null) {
      const end = text.indexOf(strong, i + 2);
      if (end > i + 2 && !/\s/.test(text[i + 2]!)) {
        flush();
        spans.push(...inlineSpans(text.slice(i + 2, end), combine(base, "bold")));
        i = end + 2;
        continue;
      }
    }
    if (rest.startsWith("*") && !rest.startsWith("**") && /\S/.test(text[i + 1] ?? " ")) {
      // The closing `*` follows a non-space and isn't half of a `**`.
      let end = text.indexOf("*", i + 1);
      while (end !== -1 && (/\s/.test(text[end - 1]!) || text[end + 1] === "*")) end = text.indexOf("*", end + 2);
      if (end !== -1) {
        flush();
        spans.push(...inlineSpans(text.slice(i + 1, end), combine(base, "italic")));
        i = end + 1;
        continue;
      }
    }
    plain += text[i];
    i += 1;
  }
  flush();
  return spans;
}

/**
 * Renders Markdown one source line at a time, `width` cells wide, styled for `out` and `env`:
 * each call returns the rows the line draws as, keeping track of an open code fence.
 */
export class MarkdownLines {
  readonly #width: number;
  readonly #out: Colored;
  readonly #env: Env;
  /** The open fence's marker, or null outside a fence. */
  #fence: string | null = null;

  constructor(width: number, out: Colored, env: Env) {
    this.#width = Math.max(1, width);
    this.#out = out;
    this.#env = env;
  }

  /** The rows `line` (one line, no newline) draws as; it is sanitized first. */
  line(line: string): string[] {
    const text = sanitize(line).replaceAll("\t", TAB);
    const fence = FENCE.exec(text);
    if (this.#fence !== null) {
      if (fence !== null && fence[1] === this.#fence && fence[2] === "") return this.end();
      return this.#code(text);
    }
    if (fence !== null) {
      this.#fence = fence[1]!;
      const title = fence[2] ? `─ ${fit(fence[2], Math.max(1, this.#width - 6))} ` : "";
      return [this.#dim(`╭${title}${"─".repeat(Math.max(0, this.#width - 2 - cells(title)))}╮`)];
    }
    const heading = HEADING.exec(text);
    if (heading !== null) return this.#wrap(inlineSpans(heading[2]!, "bold"), "");
    const bullet = BULLET.exec(text);
    if (bullet !== null) return this.#wrap(inlineSpans(bullet[2]!), `${bullet[1]}• `);
    const numbered = NUMBERED.exec(text);
    if (numbered !== null) return this.#wrap(inlineSpans(numbered[3]!), `${numbered[1]}${numbered[2]} `);
    if (text.trim() === "") return [""];
    return this.#wrap(inlineSpans(text), "");
  }

  /** The rows that close what is still open: a fence's bottom edge, or none. */
  end(): string[] {
    if (this.#fence === null) return [];
    this.#fence = null;
    return [this.#dim(`╰${"─".repeat(Math.max(0, this.#width - 2))}╯`)];
  }

  /** `spans` wrapped after `lead`, each row after the first indented under the text. */
  #wrap(spans: Span[], lead: string): string[] {
    const indent = cells(lead);
    const rows = wrapSpans(spans, Math.max(1, this.#width - indent), this.#out, this.#env);
    return rows.map(([, printed], i) => (i === 0 ? lead : " ".repeat(indent)) + printed);
  }

  /** A line of code inside the frame, folded to fit it. */
  #code(text: string): string[] {
    const inner = Math.max(1, this.#width - 4);
    const edge = this.#dim("│");
    return wrapSpans([[text, null]], inner, this.#out, this.#env).map(
      ([plain, printed]) => `${edge} ${printed}${" ".repeat(Math.max(0, inner - cells(plain)))} ${edge}`,
    );
  }

  #dim(text: string): string {
    return paint(text, "dim", this.#out, this.#env);
  }
}

/**
 * Return `text` rendered as Markdown `width` cells wide (`MarkdownLines`), its lines joined by
 * `\n`. When `out` takes no colour (not a TTY, or `NO_COLOR`), `text` is returned unchanged.
 */
export function renderMarkdown(text: string, width: number, out: Colored, env: Env): string {
  if (!out.isTTY || env("NO_COLOR")) return text;
  const lines = new MarkdownLines(width, out, env);
  const rows = text.split(/\r\n|\n|\r/).flatMap((line) => lines.line(line));
  return [...rows, ...lines.end()].join("\n");
}
