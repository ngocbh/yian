// The format-neutral diagnostic renderer.

import { describe, expect, test } from "bun:test";

import { cells } from "../../src/cli/cells.ts";
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
    output: !str
    code: tests.seeds.fns:echo_value
output:
  topic: \${echo.output}
`;

const noEnv: Env = () => undefined;

/** Render `diag` to a non-TTY writer (or a TTY one) and return the text written. */
async function render(
  diag: Diagnostic,
  options: { isTTY?: boolean; env?: Env; columns?: number } = {},
): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = {
    write: async (text) => void chunks.push(text),
    isTTY: options.isTTY ?? false,
    ...(options.columns === undefined ? {} : { columns: options.columns }),
  };
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
    expect(out).toContain("flow.yaml:4-9");
  });

  test("a long span shows its first and last lines around a count of the rest", async () => {
    const text = Array.from({ length: 5005 }, (_, i) => `line${i + 1}: x`).join("\n");
    const diag = new Diagnostic({
      code: "compose.load",
      message: "missing name",
      frames: [new DiagnosticFrame({ label: "big.yaml", text, range: new SourceRange(1, 1, 5001, 1) })],
    });
    const out = await render(diag);
    expect(out.length).toBeLessThan(4096);
    expect(out).toContain("big.yaml:1-5000");
    expect(out).toContain("⋮ 4992 lines");
    expect(marked(out)).toEqual([1, 2, 3, 4, 4997, 4998, 4999, 5000]);
    // The five lines below the span are still shown.
    expect(out).toContain("line5005: x");
    expect(out).not.toContain("line5: x");
  });

  test("a span of ten lines is shown whole", async () => {
    const text = Array.from({ length: 12 }, (_, i) => `line${i + 1}: x`).join("\n");
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad",
      frames: [new DiagnosticFrame({ label: "f.yaml", text, range: new SourceRange(1, 1, 11, 1) })],
    });
    const out = await render(diag);
    expect(marked(out)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(out).not.toContain("⋮");
  });

  test("a long marked line is cut to the terminal around the error's column", async () => {
    const line = `v: ${"a, ".repeat(3000)}TARGET${", b".repeat(1000)}`;
    const text = `id: m\nname: m\n${line}\nother: ${"z".repeat(500)}\n`;
    const at = line.indexOf("TARGET") + 1;
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad value",
      frames: [new DiagnosticFrame({ label: "m.yaml", text, range: new SourceRange(3, at, 3, at + 6) })],
    });
    const out = await render(diag, { isTTY: true, columns: 80, env: () => "1" });
    const lines = out.trimEnd().split("\n");
    for (const row of lines) expect(cells(row)).toBeLessThanOrEqual(80);
    const row = lines.find((text) => text.includes("❱"))!;
    expect(row).toContain("TARGET");
    expect(row.match(/…/g)?.length).toBe(2);
    expect(lines.find((text) => text.includes("other:"))).toMatch(/z… │$/);
  });

  test("a cut never splits a wide character", async () => {
    const line = `v: ${"漢".repeat(150)}X${"字".repeat(150)}`;
    const at = line.indexOf("X") + 1;
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad value",
      frames: [new DiagnosticFrame({ label: "w.yaml", text: `${line}\n`, range: new SourceRange(1, at, 1, at + 1) })],
    });
    const out = await render(diag, { isTTY: true, columns: 41, env: () => "1" });
    const lines = out.trimEnd().split("\n");
    for (const row of lines) expect(cells(row)).toBeLessThanOrEqual(41);
    expect(lines.find((row) => row.includes("❱"))).toMatch(/│ ❱ 1 …[漢]+X[字]+…/);
  });

  test("off a terminal a long line is cut at 200 cells and ordinary lines stay whole", async () => {
    const ordinary = `prompt: ${"w".repeat(140)}`;
    const text = `id: m\nname: m\n${ordinary}\nv: ${"y".repeat(10_000)}\n`;
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad value",
      frames: [new DiagnosticFrame({ label: "m.yaml", text, range: new SourceRange(4, 1, 4, 3) })],
    });
    const out = await render(diag);
    const lines = out.trimEnd().split("\n");
    for (const row of lines) expect(cells(row)).toBeLessThanOrEqual(200);
    expect(out).toContain(ordinary);
    expect(lines.find((row) => row.includes("❱"))).toContain("y…");
  });

  test("a long label is cut to the box", async () => {
    const label = `${"dir/".repeat(100)}flow.yaml`;
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad",
      frames: [new DiagnosticFrame({ label, text: SOURCE, range: new SourceRange(5, 5, 5, 15) })],
    });
    const out = await render(diag, { isTTY: true, columns: 80, env: () => "1" });
    for (const row of out.trimEnd().split("\n")) expect(cells(row)).toBeLessThanOrEqual(80);
    expect(out).toContain("…");
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

  test("a long line cut around the error keeps the error's place once its controls show", async () => {
    // Each `\x07` shows as two cells, so the raw column would land well left of the error.
    const line = `${"\x07".repeat(60)}id: ERROR_HERE ${"x".repeat(100)}`;
    const at = line.indexOf("ERROR_HERE");
    const diag = new Diagnostic({
      code: "compose.load",
      message: "bad",
      frames: [new DiagnosticFrame({ label: "f.yaml", text: `${line}\n`, range: new SourceRange(1, at + 1, 1, at + 11) })],
    });
    const out = await render(diag, { isTTY: true, env: noColor });
    expect(out).toContain("ERROR_HERE");
    expect(out).not.toContain("\x07");
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

  test("a box with a maxWidth cuts its title and rows to it", () => {
    const out = tty();
    const rows = [["x".repeat(30), "x".repeat(30)] as const, ["ok", "ok"] as const];
    const drawn = box(rows, "cyan", out, noColor, "a long title for this box", 0, 16);
    expect(drawn.split("\n")).toEqual([
      "╭─ a long tit… ╮",
      "│ xxxxxxxxxxx… │",
      "│ ok           │",
      "╰──────────────╯",
    ]);
  });

  test("wrapSpans wraps the plain text and keeps each piece's style", () => {
    const out = tty();
    const rows = wrapSpans([["漢字漢字 ", "bold"], ["and more words", null]], 10, out, noEnv);
    expect(rows.map(([plain]) => plain)).toEqual(["漢字漢字", "and more", "words"]);
    expect(rows[0]![1]).toBe("\x1b[1m漢字漢字\x1b[0m");
    expect(rows[1]![1]).toBe("and more");
  });
});
