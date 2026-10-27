// The format-neutral diagnostic renderer.

import { describe, expect, test } from "bun:test";

import { renderDiagnostic } from "../../src/cli/run.ts";
import type { Writer } from "../../src/cli/utils.ts";
import type { Env } from "../../src/defaults.ts";
import { Diagnostic, DiagnosticFrame, SourceRange } from "../../src/diagnostics.ts";

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
