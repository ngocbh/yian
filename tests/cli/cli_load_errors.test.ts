// A load error's diagnostic is shown as a boxed source frame titled `file:line`, the message
// below; through `yian run`, a broken flow is located and exits 1 without an engine stack, which
// `--engine-trace` adds, and a cycle frames both nodes.

import { afterEach, beforeEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PassThrough } from "node:stream";

import { main } from "../../src/cli/index.ts";
import { renderDiagnostic } from "../../src/cli/render.ts";
import type { CliIo, Writer } from "../../src/cli/utils.ts";
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

// ---------- CLI integration: a real broken flow ----------

const BROKEN =
  "id: f\nname: f\nnodes:\n" +
  "  a:\n    kind: agent\n    prompt: hi\n" +
  "  b:\n    kind: agent\n    input:\n      brief: ${frame_typo.output}\n    prompt: use ${brief}\n" +
  "output: ${b.output}\n";

// A cycle a -> b -> a: each node implicates the other, so the error spans both lines.
const CYCLE =
  "id: f\nname: f\nnodes:\n" +
  "  a:\n    kind: agent\n    input:\n      x: ${b.output}\n    prompt: use ${x}\n" +
  "  b:\n    kind: agent\n    input:\n      y: ${a.output}\n    prompt: use ${y}\n" +
  "output: ${a.output}\n";

let tmp: string;

beforeEach(() => {
  tmp = mkdtempSync(join(tmpdir(), "yian-cli-load-"));
});

afterEach(() => {
  rmSync(tmp, { recursive: true, force: true });
});

/** Run `yian run flow.yaml ...args` on `text`; return the exit code and stderr. */
async function runCli(text: string, args: string[] = []): Promise<{ code: number; err: string }> {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, text);
  const chunks: string[] = [];
  const io: CliIo = {
    stdout: { write: async () => {} },
    stderr: { write: async (chunk) => void chunks.push(chunk) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? tmp : undefined),
  };
  const code = await main(["run", flow, ...args], io);
  return { code, err: chunks.join("") };
}

test("cli compile error is located, not a stack", async () => {
  const { code, err } = await runCli(BROKEN);
  expect(code).toBe(1);
  expect(err).toContain("flow.yaml:");
  expect(err).toContain("frame_typo"); // the offending ref is named
  expect(err).not.toContain("    at "); // no engine stack by default
});

test("cli --engine-trace adds the stack", async () => {
  const { code, err } = await runCli(BROKEN, ["--engine-trace"]);
  expect(code).toBe(1);
  expect(err).toContain("flow.yaml:"); // still located
  expect(err).toContain("LoadError"); // plus the engine stack
  expect(err).toContain("    at ");
});

test("cli cycle error highlights both nodes", async () => {
  const { code, err } = await runCli(CYCLE);
  expect(code).toBe(1);
  // Node `a` is on line 4, node `b` on line 9; each is its own ordered source frame.
  expect(err).toContain("source locations:");
  expect(err).toContain("flow.yaml:4");
  expect(err).toContain("flow.yaml:9");
  expect(err).toContain("a:");
  expect(err).toContain("b:");
  expect(err).toContain("cycle");
  // The "why" legend names each dependency edge that closes the loop.
  expect(err).toContain("a depends on b (a.input.x)");
  expect(err).toContain("b depends on a (b.input.y)");
});

/** Run `yian check flow.yaml ...args` on `text`; return the exit code, stdout and stderr. */
async function checkCli(text: string, args: string[] = []): Promise<{ code: number; out: string; err: string }> {
  const flow = join(tmp, "flow.yaml");
  writeFileSync(flow, text);
  const out: string[] = [];
  const err: string[] = [];
  const io: CliIo = {
    stdout: { write: async (chunk) => void out.push(chunk) },
    stderr: { write: async (chunk) => void err.push(chunk) },
    stdin: new PassThrough(),
    stderrStream: new PassThrough(),
    env: (name) => (name === "HOME" ? tmp : undefined),
  };
  const code = await main(["check", flow, ...args], io);
  return { code, out: out.join(""), err: err.join("") };
}

test.each([
  ["an unclosed quote", 'id: t\nname: t\nkind: agent\nprompt: "unclosed\noutput: str\n', 4, [4, 9, 4, 10]],
  ["an unclosed flow list", "id: x\nname: x\noutput: x\nnodes:\n  a: !str [1, 2\n", 5, [5, 11, 5, 12]],
])("cli check boxes %s at its opener", async (_id, text, line, range) => {
  const { code, err } = await checkCli(text);
  expect(code).toBe(1);
  expect(err).toContain(`flow.yaml:${line}`);

  const json = await checkCli(text, ["--json"]);
  expect(JSON.parse(json.out).flows[0].diagnostics[0].frames[0].range).toEqual(range);
});

