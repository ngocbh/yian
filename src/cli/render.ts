/**
 * What the CLI commands print on stderr: styled text, boxes, a diagnostic's boxed source frames
 * (`renderDiagnostic`) or its `--json` form (`diagnosticJson`), and a boxed traceback
 * (`writeEngineTrace` under `--engine-trace`).
 *
 * Output is raw ANSI, coloured only on a TTY without `NO_COLOR`.
 *
 * Knows about: diagnostics, the terminal's cell widths and the CLI's io.
 */

import type { Env } from "../defaults.ts";
import type { Diagnostic } from "../diagnostics.ts";
import { cells, wrapLine } from "./cells.ts";
import { type CliIo, forTerminal, type Writer } from "./utils.ts";

const SGR = { bold: 1, dim: 2, italic: 3, red: 31, green: 32, yellow: 33, cyan: 36 } as const;

/** @internal A text style: a colour or weight, or two of them (`"red bold"`). */
export type Style = keyof typeof SGR | `${keyof typeof SGR} ${keyof typeof SGR}`;

/** @internal What `paint` reads of an output: whether it is a TTY. */
export type Colored = Pick<Writer, "isTTY">;

/**
 * @internal Return `text` in `style` when `out` is a TTY and `NO_COLOR` is unset or empty, else as
 * is.
 */
export function paint(text: string, style: Style, out: Colored, env: Env): string {
  if (!out.isTTY || env("NO_COLOR")) return text;
  const codes = style.split(" ").map((word) => SGR[word as keyof typeof SGR]);
  return `\x1b[${codes.join(";")}m${text}\x1b[0m`;
}

/**
 * The band a line the user sent is drawn on: a dark gray background (256-colour 237) under light
 * gray text (252). The text's colour is set with the band's, so it reads on a light theme as on a
 * dark one, where the terminal's own text colour might not.
 */
const USER_BAND = "\x1b[48;5;237;38;5;252m";

/**
 * @internal Return a line the user sent, `text` (already safe to print), as the transcript shows
 * it: `> <text>` wrapped at its spaces to `width` cells, each row on the user's gray band across
 * the width with the `>` bold; with `colour` false, the plain rows.
 */
export function userRows(text: string, width: number, colour: boolean): string[] {
  const line = `> ${text}`;
  const rows = wrapLine(line, Math.max(1, width)).map(([start, end]) => line.slice(start, end));
  if (!colour) return rows;
  return rows.map((row, i) => {
    const body = i === 0 ? `\x1b[1m>\x1b[22m${row.slice(1)}` : row;
    return `${USER_BAND}${body}${" ".repeat(Math.max(0, width - cells(row)))}\x1b[0m`;
  });
}

/** @internal Write `text` in `style` and a newline to `io.stderr`. */
export function printStyled(io: CliIo, text: string, style: Style): Promise<void> {
  return io.stderr.write(`${paint(forTerminal(text, io.stderr), style, io.stderr, io.env)}\n`);
}

/**
 * @internal Return `text`'s lines as source ranges count them, with no empty line after a final
 * break.
 */
export function splitLines(text: string): string[] {
  const lines = text.split(/\r\n|\r|\n/);
  if (lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/**
 * @internal Return `rows` in a box drawn in `border`, `title` set into its top edge, at least
 * `width` cells inside. Each row is its plain text, which sizes the box, and the text to print.
 */
export function box(
  rows: readonly (readonly [string, string])[],
  border: Style,
  out: Colored,
  env: Env,
  title = "",
  width = 0,
): string {
  const heading = title ? `─ ${title} ` : "";
  const inner = Math.max(width, cells(heading) - 1, ...rows.map(([plain]) => cells(plain)));
  const edge = (text: string) => paint(text, border, out, env);
  const lines = [edge(`╭${heading}${"─".repeat(inner + 2 - cells(heading))}╮`)];
  for (const [plain, printed] of rows) {
    lines.push(`${edge("│")} ${printed}${" ".repeat(inner - cells(plain))} ${edge("│")}`);
  }
  lines.push(edge(`╰${"─".repeat(inner + 2)}╯`));
  return lines.join("\n");
}

/** @internal One piece of a line: its text and its style, or `null` for none. */
export type Span = readonly [text: string, style: Style | null];

/**
 * @internal Return the line `spans` make wrapped to `width` cells (`wrapLine`), as `box` rows:
 * each row's plain text and its text with each span's piece painted in its style.
 */
export function wrapSpans(spans: readonly Span[], width: number, out: Colored, env: Env): [string, string][] {
  const plain = spans.map(([text]) => text).join("");
  return wrapLine(plain, width).map(([start, end]) => {
    let printed = "";
    let at = 0;
    for (const [text, style] of spans) {
      const piece = text.slice(Math.max(0, start - at), Math.max(0, end - at));
      if (piece) printed += style === null ? piece : paint(piece, style, out, env);
      at += text.length;
    }
    return [plain.slice(start, end), printed];
  });
}

// Lines of source shown above and below the marked lines, so a large flow isn't printed whole.
const ERR_CONTEXT = 5;

/**
 * Return one boxed source frame, or `null` when no line of `marks` is in `text`.
 *
 * The box shows the source from 5 lines above the first mark to 5 below the last, numbered, each
 * marked line flagged `❱`, and is titled `title:line[,line...]`.
 */
function renderFrameBox(text: string, marks: readonly number[], title: string, out: Writer, env: Env): string | null {
  const lines = splitLines(text);
  const shown = [...new Set(marks)].filter((mark) => 1 <= mark && mark <= lines.length).sort((a, b) => a - b);
  if (shown.length === 0) return null;
  const start = Math.max(1, shown[0]! - ERR_CONTEXT);
  const end = Math.min(lines.length, shown[shown.length - 1]! + ERR_CONTEXT);
  const width = String(end).length;
  const rows: [string, string][] = [];
  for (let line = start; line <= end; line++) {
    const marked = shown.includes(line);
    const gutter = `${marked ? "❱" : " "} ${String(line).padStart(width)} `;
    const code = lines[line - 1]!;
    rows.push([gutter + code, paint(gutter, marked ? "red bold" : "dim", out, env) + code]);
  }
  return box(rows, "red", out, env, `${title}:${shown.join(",")}`);
}

/**
 * Write `diag` to `out`: each frame, outermost first, then the message, then the notes.
 *
 * A frame is boxed around the lines its range covers (`SourceRange.coveredLines`), or shown as
 * its label when none is in its text; its note follows it. Several frames are headed `source
 * locations:`. A diagnostic without frames prints only its message. On a TTY the message and notes,
 * which can carry provider or tool text, go through `forTerminal`.
 */
export async function renderDiagnostic(diag: Diagnostic, out: Writer, env: Env): Promise<void> {
  const parts: string[] = [];
  if (diag.frames.length > 1) parts.push(paint("source locations:", "red bold", out, env));
  for (const frame of diag.frames) {
    const framed = renderFrameBox(frame.text, frame.range.coveredLines(frame.text), frame.label, out, env);
    parts.push(framed ?? paint(frame.label, "red bold", out, env));
    if (frame.note !== null) parts.push(paint(`  ↳ ${forTerminal(frame.note, out)}`, "yellow", out, env));
  }
  parts.push(paint(forTerminal(diag.message, out), "red", out, env));
  for (const note of diag.notes) parts.push(paint(`  ↳ ${forTerminal(note, out)}`, "yellow", out, env));
  await out.write(`${parts.join("\n")}\n`);
}

/** A diagnostic in the `yian.cli.v1` form. */
export interface DiagnosticJson {
  code: string;
  message: string;
  frames?: { label: string; range: [bigint, bigint, bigint, bigint]; note?: string }[];
  notes?: string[];
}

/**
 * Return `diag` in the `yian.cli.v1` diagnostic form.
 *
 * A frame's range is `[startLine, startCol, endLine, endCol]`, 1-based with an exclusive end
 * column, as ints (`bigint`s, so `printJson` writes them without `.0`). `frames`, `notes` and a
 * frame's `note` are left out when empty.
 */
export function diagnosticJson(diag: Diagnostic): DiagnosticJson {
  const document: DiagnosticJson = { code: diag.code, message: diag.message };
  if (diag.frames.length > 0) {
    document.frames = diag.frames.map(({ label, range, note }) => ({
      label,
      range: [BigInt(range.startLine), BigInt(range.startCol), BigInt(range.endLine), BigInt(range.endCol)],
      ...(note ? { note } : {}),
    }));
  }
  if (diag.notes.length > 0) document.notes = [...diag.notes];
  return document;
}

/**
 * @internal Return `text`, a stack or captured traceback, in a box titled `traceback`, ending in
 * a newline.
 */
export function tracebackBox(text: string, out: Writer, env: Env): string {
  const lines = splitLines(forTerminal(text, out).trimEnd());
  const rows = lines.map((line) => [line, paint(line, "dim", out, env)] as const);
  return `${box(rows, "red", out, env, "traceback")}\n`;
}

/** @internal Write `error`'s stack in a traceback box on stderr when `engineTrace` is set. */
export async function writeEngineTrace(error: Error, engineTrace: boolean, io: CliIo): Promise<void> {
  if (engineTrace) await io.stderr.write(tracebackBox(error.stack ?? String(error), io.stderr, io.env));
}

/**
 * Return `ms` as a bracketed elapsed time in whole units: `[5s]`, `[1m05s]`, `[1h02m]`.
 */
export function elapsedTag(ms: number): string {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  if (seconds < 60) return `[${seconds}s]`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `[${minutes}m${String(seconds % 60).padStart(2, "0")}s]`;
  return `[${Math.floor(minutes / 60)}h${String(minutes % 60).padStart(2, "0")}m]`;
}

/**
 * @internal The text of a run's line for node `nodeId` that ended: `✓ <node>`, `✓ <node> →
 * <handle>` for a router, then `took` (` [<elapsed>]`, or "").
 */
export function endedNodeText(nodeId: string, took: string, handle: string | null = null): string {
  return `✓ ${nodeId}${handle === null ? "" : ` → ${handle}`}${took}`;
}

/** @internal The text of a run's line for node `nodeId` that failed: `✗ <node>`, then `took`. */
export function failedNodeText(nodeId: string, took: string): string {
  return `✗ ${nodeId}${took}`;
}
