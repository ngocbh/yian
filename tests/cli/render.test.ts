// The format-neutral diagnostic renderer.

import { describe, expect, test } from "bun:test";

import { box, renderDiagnostic, wrapSpans } from "../../src/cli/render.ts";
import type { Writer } from "../../src/cli/utils.ts";
import type { Env } from "../../src/defaults.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";
import { FakeOutput, screen } from "./terminal_both_runtimes.ts";

const SOURCE = `id: f
name: f
nodes:
  echo:
    kind: code
    input:
      topic: \${input.topic}
    output: str
    code: tests.seeds.fns:echo_value
output:
  topic: \${echo.output}
`;

const noEnv: Env = () => undefined;

/** Render `diag` to a non-TTY writer (or a TTY one) and return the text written. */
async function render(diag: Diagnostic, options: { isTTY?: boolean; env?: Env } = {}): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text), isTTY: options.isTTY ?? false };
  await renderDiagnostic(diag, out, options.env ?? noEnv);
  return chunks.join("");
}

/**
 * The line numbers the frame actually highlighted (`❱`), not merely displayed.
 *
 * The box renders a window of surrounding context, so asserting that some text appears proves
 * nothing about the highlight; only the marker does.
 */
function marked(out: string): number[] {
  return [...out.matchAll(/❱\s+(\d+)/g)].map((m) => Number(m[1]));
}

/** A one-frame diagnostic over `SOURCE` at `range`. */
function oneFrame(message: string, range: SourceRange): Diagnostic {
  return new Diagnostic({
    code: "compose.load",
    message,
    frames: [new DiagnosticFrame({ label: "flow.yaml", text: SOURCE, range })],
  });
}

describe("renderDiagnostic", () => {
  test("boxes one frame", async () => {
    // One resolved frame supplies its own source, label, and exact mark.
    const out = await render(oneFrame("bad thing happened", new SourceRange(5, 5, 5, 15)));
    expect(out).toContain("bad thing happened");
    expect(out).toContain("flow.yaml:5");
    expect(out).toContain("kind: code"); // line 5 is inside the boxed source window
    expect(marked(out)).toEqual([5]); // ...and it is the line actually highlighted
    // A ±5-line window: lines 1 to 10, so line 11 is left out.
    expect(out).toContain("id: f");
    expect(out).toContain("10 output:");
    expect(out).not.toContain("${echo.output}");
  });

  test("a block range stops before the next field", async () => {
    // `input:`'s value is a block mapping (line 7), so it ends at the start of the next token,
    // `output:` on line 8, column 5. The end is exclusive and line 8 holds only indentation
    // before column 5, so the span covers line 7 alone.
    const out = await render(oneFrame("bad ref", new SourceRange(7, 7, 8, 5)));
    expect(marked(out)).toEqual([7]);
  });

  test("a multi-line block keeps its content lines", async () => {
    // A span genuinely covering several lines still marks them all; it ends at 10:1, so line
    // 10 is excluded and 4-9 are not.
    const out = await render(oneFrame("bad node", new SourceRange(4, 3, 10, 1)));
    expect(marked(out)).toEqual([4, 5, 6, 7, 8, 9]);
    expect(out).toContain("flow.yaml:4,5,6,7,8,9");
  });

  test("renders ordered frames and context notes", async () => {
    // Resolved frames keep their order and both note channels stay visible.
    const childText = "nodes:\n  leaf:\n    kind: code\n";
    const diag = new Diagnostic({
      code: "node.execute",
      message: "leaf failed",
      frames: [
        new DiagnosticFrame({
          label: "root.yaml",
          text: SOURCE,
          range: new SourceRange(4, 3, 4, 8),
          note: "while invoking child",
        }),
        new DiagnosticFrame({
          label: "child.yaml",
          text: childText,
          range: new SourceRange(2, 3, 2, 8),
          note: "inside leaf",
        }),
      ],
      notes: ["check the child input"],
    });
    const out = await render(diag);
    expect(out).toContain("source locations:");
    expect(out.indexOf("source locations:")).toBeLessThan(out.indexOf("root.yaml:4"));
    expect(out.indexOf("root.yaml:4")).toBeLessThan(out.indexOf("↳ while invoking child"));
    expect(out.indexOf("↳ while invoking child")).toBeLessThan(out.indexOf("child.yaml:2"));
    expect(out.indexOf("child.yaml:2")).toBeLessThan(out.indexOf("↳ inside leaf"));
    expect(out.indexOf("↳ inside leaf")).toBeLessThan(out.indexOf("leaf failed"));
    expect(out.indexOf("leaf failed")).toBeLessThan(out.indexOf("↳ check the child input"));
    expect(marked(out)).toEqual([4, 2]);
  });

  test("an unlocated message needs no source context", async () => {
    // A source-free diagnostic keeps its message without inventing a label.
    const out = await render(new Diagnostic({ code: "node.execute", message: "unchanged failure message" }));
    expect(out.trim()).toBe("unchanged failure message");
  });

  test("colour only on a TTY without NO_COLOR", async () => {
    const diag = oneFrame("bad thing happened", new SourceRange(5, 5, 5, 15));
    expect(await render(diag)).not.toContain("\x1b[");
    expect(await render(diag, { isTTY: true })).toContain("\x1b[31m");
    const noColor: Env = (name) => (name === "NO_COLOR" ? "1" : undefined);
    expect(await render(diag, { isTTY: true, env: noColor })).not.toContain("\x1b[");
  });
});

describe("provider text with escape sequences", () => {
  const raw = "bad\x1b]8;;https://x.example\x07 request\x1b[1A";
  const diag = new Diagnostic({ code: "node.execute", message: raw, notes: [raw] });

  test("is stripped from the message and notes on a TTY", async () => {
    expect(await render(diag, { isTTY: true, env: () => "1" })).toBe("bad request\n  ↳ bad request\n");
  });

  test("is written exactly off a TTY", async () => {
    expect(await render(diag)).toBe(`${raw}\n  ↳ ${raw}\n`);
  });
});

describe("flow text with escape sequences in a frame", () => {
  const source = "id: f\nname: f\n# note \x1b]52;c;RlJBTUU=\x07 \x1b[2J\nnodes: {}\n";
  const label = "y\x1b]0;TITLE\x07.yaml";
  const framed = new Diagnostic({
    code: "compose.load",
    message: "invalid YAML",
    frames: [new DiagnosticFrame({ label, text: source, range: new SourceRange(3, 1, 3, 5) })],
  });
  // A range past the text has no line to box, so the frame prints its label.
  const unboxed = new Diagnostic({
    code: "compose.load",
    message: "invalid YAML",
    frames: [new DiagnosticFrame({ label, text: source, range: new SourceRange(9, 1, 9, 2) })],
  });
  const noColor: Env = (name) => (name === "NO_COLOR" ? "1" : undefined);

  test("shows its control characters in the source and the title on a TTY, the box in line", async () => {
    const out = await render(framed, { isTTY: true, env: noColor });
    expect(out).not.toContain("\x1b");
    expect(out).not.toContain("\x07");
    expect(out).toContain("# note ^[]52;c;RlJBTUU=^G ^[[2J");
    expect(out).toContain("y^[]0;TITLE^G.yaml:3");
    const rows = out.split("\n").filter((line) => /^[╭│╰]/.test(line));
    expect(new Set(rows.map((row) => [...row].length)).size).toBe(1);
    expect(await render(unboxed, { isTTY: true, env: noColor })).toBe("y^[]0;TITLE^G.yaml\ninvalid YAML\n");
  });

  test("is written exactly off a TTY", async () => {
    const out = await render(framed);
    expect(out).toContain("# note \x1b]52;c;RlJBTUU=\x07 \x1b[2J");
    expect(out).toContain(`${label}:3`);
    expect(await render(unboxed)).toBe(`${label}\ninvalid YAML\n`);
  });
});

describe("box and wrapSpans", () => {
  const noColor: Env = (name) => (name === "NO_COLOR" ? "1" : undefined);

  /** A TTY output 60 columns wide, as the chat draws on. */
  const tty = () => Object.assign(new FakeOutput({ columns: 60 }), { isTTY: true });

  for (const [label, env] of [
    ["in colour", noEnv],
    ["with NO_COLOR", noColor],
  ] as const) {
    test(`a box sizes a CJK row by its cells, ${label}`, () => {
      const out = tty();
      const rows = [
        ...wrapSpans([["名前 ", "bold"], ["Ada", null]], 56, out, env),
        ...wrapSpans([["ok", "dim"]], 56, out, env),
      ];
      // The replayer moves down a row at a newline without going back to column 0.
      out.write(`${box(rows, "cyan", out, env, "", 10).replaceAll("\n", "\r\n")}\r\n`);
      expect(screen(out.text, 60).rows).toEqual([
        "╭────────────╮",
        "│ 名前 Ada   │",
        "│ ok         │",
        "╰────────────╯",
      ]);
      expect(out.text.includes("\x1b[")).toBe(env === noEnv);
    });
  }

  test("wrapSpans wraps the plain text and keeps each piece's style", () => {
    const out = tty();
    const rows = wrapSpans([["漢字漢字 ", "bold"], ["and more words", null]], 10, out, noEnv);
    expect(rows.map(([plain]) => plain)).toEqual(["漢字漢字", "and more", "words"]);
    expect(rows[0]![1]).toBe("\x1b[1m漢字漢字\x1b[0m");
    expect(rows[1]![1]).toBe("and more");
  });
});
