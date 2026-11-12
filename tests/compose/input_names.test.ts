/**
 * A flow input is read as `${input.<name>}`, so its name must be an identifier: a flow, a def,
 * a single-node def, a compact flow and an inline body each refuse another name at load.
 */

import { afterAll, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { assemble } from "../../src/compose/build/pipeline.ts";
import { CallableScope } from "../../src/compose/loader.ts";
import { SourceCursor, SourceDocument } from "../../src/compose/source/index.ts";
import type { SourceRange } from "../../src/diagnostics.ts";
import { runFlow } from "../../src/run.ts";
import { loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-input-names-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

const RULE = "ASCII letters, digits, and _, not starting with a digit";

/** The message for a flow input `name` that isn't an identifier, with its suggested spelling. */
const refused = (what: string, name: string, suggestion: string) =>
  `${what} '${name}' must be an identifier (${RULE}) so \${input.${name}} can read it; try '${suggestion}'`;

/** Expect loading `text` to fail with a message ending in `message`, located at `range`. */
async function expectRefused(text: string, message: string, range: SourceRange | null): Promise<void> {
  expect(range).not.toBeNull();
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toEndWith(message);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(range!);
}

const ECHO = "kind: code, code: ./fixtures/compose_codefns.ts:echo, output: str";

test.each([
  ["first-name", "first_name"],
  ["x y", "x_y"],
  ["1st", "_1st"],
  ["café", "caf_"],
])("a flow input named %p is refused at its key", async (name, suggestion) => {
  const text =
    `id: f\nname: f\ninput:\n  "${name}": !str\n` + `nodes:\n  a: {${ECHO}, input: {topic: x}}\noutput: \${a.output}\n`;
  const root = SourceDocument.parse(text).root;
  await expectRefused(text, refused("flow input", name, suggestion), root.child("input").keyRangeOf(name));
});

test("a def's input that isn't an identifier is refused at its key", async () => {
  const text =
    'id: f\nname: f\ndefs:\n  d:\n    input:\n      "my-in": !str\n' +
    `    nodes:\n      b: {${ECHO}, input: {topic: x}}\n    output: \${b.output}\n` +
    "nodes:\n  go: {kind: call, call: d, input: {my-in: x}}\noutput: ${go.output}\n";
  const def = SourceDocument.parse(text).root.child("defs").child("d");
  await expectRefused(text, refused("flow input", "my-in", "my_in"), def.child("input").keyRangeOf("my-in"));
});

test("a single-node def's input that isn't an identifier is refused at its key", async () => {
  const text =
    "id: f\nname: f\ndefs:\n  d:\n    kind: code\n    code: ./fixtures/compose_codefns.ts:echo\n" +
    '    input:\n      "my-in": !str\n    output: str\n' +
    "nodes:\n  go: {kind: call, call: d, input: {my-in: x}}\noutput: ${go.output}\n";
  const def = SourceDocument.parse(text).root.child("defs").child("d");
  await expectRefused(
    text,
    refused("defs entry 'd': input", "my-in", "my_in"),
    def.child("input").keyRangeOf("my-in"),
  );
});

test("a compact flow's input that isn't an identifier is refused at its key", async () => {
  const text =
    "id: f\nname: f\nkind: code\ncode: ./fixtures/compose_codefns.ts:echo\n" +
    'input:\n  "first-name": !str\noutput: str\n';
  const root = SourceDocument.parse(text).root;
  await expectRefused(
    text,
    refused("compact (single-node) flow: input", "first-name", "first_name"),
    root.child("input").keyRangeOf("first-name"),
  );
});

test("an inline body's input that isn't an identifier is refused at the node's binding", async () => {
  const text =
    "id: f\nname: f\nnodes:\n  go:\n    kind: call\n    input:\n      first-name: x\n" +
    `    nodes:\n      b: {${ECHO}, input: {topic: x}}\n    output: \${b.output}\n` +
    "output: ${go.output}\n";
  const binding = SourceDocument.parse(text).root.child("nodes").child("go").child("input").child("first-name");
  await expectRefused(text, refused("inline body input", "first-name", "first_name"), binding.range);
});

test("a source-free flow input that isn't an identifier is refused", () => {
  expect(() =>
    assemble({
      inputsSection: { "first-name": "str" },
      nodesCursor: SourceCursor.virtual({ worker: { kind: "code", code: "return 1", outputs: "int" } }),
      outputsSection: "${worker.output}",
      assertsSection: [],
      registry: new Map(),
      scope: new CallableScope({}, new Map(), null, {}),
      owner: ["root"],
    }),
  ).toThrow(refused("flow input", "first-name", "first_name"));
});

test("an input named like a keyword or starting with _ loads and runs", async () => {
  const text =
    "id: f\nname: f\ninput:\n  in: !str\n  _x: !str\n  x1: !str\n" +
    "nodes:\n  joined: !str ${input.in + input._x + input.x1}\noutput: ${joined.output}\n";
  const env = hostEnv(dir);
  const result = await runFlow(await loadConformanceFlow(text, { env }), { in: "a", _x: "b", x1: "c" }, { env });
  expect(result.failure).toBeNull();
  expect(result.output).toBe("abc");
});
