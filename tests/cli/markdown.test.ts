/**
 * The chat's Markdown (`markdown.ts`), rendered 60 columns wide on a fake TTY and read back as the
 * screen shows it; in colour, and unchanged with `NO_COLOR` or off a TTY.
 */

import { describe, expect, test } from "bun:test";

import { inlineSpans, MarkdownLines, renderMarkdown } from "../../src/cli/markdown.ts";
import type { Env } from "../../src/defaults.ts";
import { FakeOutput, screen } from "./terminal_both_runtimes.ts";

const noEnv: Env = () => undefined;
const noColor: Env = (name) => (name === "NO_COLOR" ? "1" : undefined);

/** `text` rendered 60 wide, then drawn on a fake TTY: the rows the screen shows, and what was written. */
function drawn(text: string, env: Env = noEnv): { rows: string[]; written: string } {
  const out = Object.assign(new FakeOutput({ columns: 60 }), { isTTY: true });
  const written = renderMarkdown(text, 60, out, env);
  out.write(`${written.replaceAll("\n", "\r\n")}\r\n`);
  return { rows: screen(out.text, 60).rows, written };
}

const REPLY = [
  "# Plan",
  "hello ran and returned **Hi Ada**, *quickly*, with `name: str`.",
  "",
  "- check the flow",
  "- run it with a long list item that goes on past the width of the terminal",
  "2. then report",
  "```yaml",
  "id: hello",
  "```",
].join("\n");

describe("renderMarkdown", () => {
  test("styles inline text, headings and lists, and frames code, in colour", () => {
    const { rows, written } = drawn(REPLY);
    expect(rows).toEqual([
      "Plan",
      "hello ran and returned Hi Ada, quickly, with name: str.",
      "",
      "• check the flow",
      "• run it with a long list item that goes on past the width",
      "  of the terminal",
      "2. then report",
      `╭─ yaml ${"─".repeat(51)}╮`,
      `│ id: hello${" ".repeat(47)} │`,
      `╰${"─".repeat(58)}╯`,
    ]);
    expect(written).toContain("\x1b[1mPlan\x1b[0m");
    expect(written).toContain("\x1b[1mHi Ada\x1b[0m");
    expect(written).toContain("\x1b[3mquickly\x1b[0m");
    expect(written).toContain("\x1b[36mname: str\x1b[0m");
  });

  test("with NO_COLOR, or off a TTY, returns the text unchanged", () => {
    expect(drawn(REPLY, noColor).written).toBe(REPLY);
    expect(renderMarkdown(REPLY, 60, { isTTY: false }, noEnv)).toBe(REPLY);
  });

  test("sanitizes each line first", () => {
    const { rows, written } = drawn("**bold\x1b[2J** and \x1b]8;;http://x\x07link‮");
    expect(rows).toEqual(["bold and link\\u{202e}"]);
    expect(written).not.toContain("\x1b[2J");
    expect(written).not.toContain("\x1b]");
  });

  test("a marker with no end stays as written, and a fence left open is closed", () => {
    const { rows } = drawn("a * b and **open and `tick\n```\nx");
    expect(rows).toEqual([
      "a * b and **open and `tick",
      `╭${"─".repeat(58)}╮`,
      `│ x${" ".repeat(55)} │`,
      `╰${"─".repeat(58)}╯`,
    ]);
  });

  test("__bold__ opens only after a non-word character", () => {
    expect(inlineSpans("snake__case__name and __this__")).toEqual([
      ["snake__case__name and ", null],
      ["this", "bold"],
    ]);
  });

  test("inline styles nest", () => {
    expect(inlineSpans("**a `b` *c* d**")).toEqual([
      ["a ", "bold"],
      ["b", "cyan bold"],
      [" ", "bold"],
      ["c", "italic bold"],
      [" d", "bold"],
    ]);
  });

  test("MarkdownLines renders line by line, keeping a fence open across calls", () => {
    const lines = new MarkdownLines(20, { isTTY: false }, noEnv);
    expect(lines.line("```")).toEqual([`╭${"─".repeat(18)}╮`]);
    expect(lines.line("a\tb")).toEqual([`│ a    b${" ".repeat(10)} │`]);
    expect(lines.line("```")).toEqual([`╰${"─".repeat(18)}╯`]);
    expect(lines.end()).toEqual([]);
  });
});
