// Ported from tests/engine/test_cli_load_errors.py (the `test_render_*` tests): a load error's
// diagnostic is shown as a boxed source frame titled `file:line`, the message below. The CLI-run
// tests come with `yian run`.

import { expect, test } from "bun:test";

import { renderDiagnostic } from "../../src/cli/run.ts";
import type { Writer } from "../../src/cli/utils.ts";
import { LoadDiagnostics, type LoadError } from "../../src/compose/helpers/errors.ts";
import { SourceRange } from "../../src/diagnostics.ts";

/** Render `error`'s diagnostic to a non-TTY writer and return the text written. */
async function render(error: LoadError): Promise<string> {
  const chunks: string[] = [];
  const out: Writer = { write: async (text) => void chunks.push(text) };
  await renderDiagnostic(error.diagnostic, out, () => undefined);
  return chunks.join("");
}

test("render located error", async () => {
  const text = "id: f\nname: f\nnodes:\n  b:\n    kind: agent\n";
  const error = new LoadDiagnostics("flow.yaml", text).error("node 'b' is broken", {
    range: new SourceRange(4, 3, 4, 5),
  });
  const out = await render(error);
  expect(out).toContain("flow.yaml:4"); // the box title carries file:line
  expect(out).toContain("b:"); // the source frame echoes the YAML
  expect(out).toMatch(/❱\s+4\s+b:/); // line numbers are shown
  expect(out).toContain("node 'b' is broken"); // the message is printed
});

test("render unlocated error has no frame", async () => {
  const out = await render(new LoadDiagnostics().error("global problem"));
  expect(out).not.toContain("flow.yaml");
  expect(out).toContain("global problem");
});

test("render out-of-range line falls back", async () => {
  // A line past the end must not crash; the frame degrades to its label.
  const error = new LoadDiagnostics("flow.yaml", "id: f\n").error("weird", {
    range: new SourceRange(999, 1, 999, 2),
  });
  const out = await render(error);
  expect(out).toBe("flow.yaml\nweird\n");
});

test("render multi-line error spans and titles all", async () => {
  const text = `${Array.from({ length: 20 }, (_, i) => `line ${i + 1}`).join("\n")}\n`;
  const diagnostics = new LoadDiagnostics("flow.yaml", text);
  const frames = [4, 12].map((line) => diagnostics.frame(new SourceRange(line, 1, line, 7))!);
  const out = await render(diagnostics.error("a cycle", { frames }));
  expect(out).toContain("source locations:");
  expect(out).toContain("flow.yaml:4");
  expect(out).toContain("flow.yaml:12");
  expect(out).toContain("line 4"); // the window covers the first mark
  expect(out).toContain("line 12");
});

test("render notes printed under message", async () => {
  const diagnostics = new LoadDiagnostics("flow.yaml", "id: f\nname: f\n");
  const out = await render(
    diagnostics.error("a cycle", { range: new SourceRange(1, 1, 1, 6), notes: ["a depends on b (a.input.x)"] }),
  );
  expect(out.indexOf("a cycle")).toBeLessThan(out.indexOf("↳ a depends on b (a.input.x)"));
});
