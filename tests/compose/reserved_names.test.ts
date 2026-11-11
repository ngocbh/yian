/**
 * Names starting with `__` belong to yian, which generates them (wiring keys, synthesized nodes
 * and END params): a flow can't take one as a node id, an input, an output or a variable.
 */

import { expect, test } from "bun:test";

import { SourceDocument } from "../../src/compose/source/index.ts";
import type { SourceRange } from "../../src/diagnostics.ts";
import { loadConformanceFlow, loadErrorAsync } from "../helpers/declarations.ts";

/** The message for an authored `name` in the role `what`. */
const reserved = (what: string, name: string) =>
  `${what} '${name}' starts with '__', which yian keeps for the names it generates; rename it`;

/** Expect loading `text` to fail with `message`, located at `range`. */
async function expectRefused(text: string, message: string, range: SourceRange | null): Promise<void> {
  expect(range).not.toBeNull();
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toBe(message);
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(range!);
}

const ECHO = "kind: code, code: ./fixtures/compose_codefns.ts:echo, output: str";

test("a node id starting with __ is refused at its node", async () => {
  const text = `id: f\nname: f\nnodes:\n  __a: {${ECHO}, input: {topic: x}}\noutput: \${__a.output}\n`;
  const root = SourceDocument.parse(text).root;
  await expectRefused(text, reserved("node id", "__a"), root.child("nodes").child("__a").range);
});

test("a node id starting with __ in a def body is refused", async () => {
  const text =
    "id: f\nname: f\ndefs:\n  d:\n    nodes:\n" +
    `      __b: {${ECHO}, input: {topic: x}}\n    output: \${__b.output}\n` +
    "nodes:\n  go: {kind: call, call: d}\noutput: ${go.output}\n";
  const def = SourceDocument.parse(text).root.child("defs").child("d");
  const error = await loadErrorAsync(loadConformanceFlow(text));
  expect(error.message).toEndWith(reserved("node id", "__b"));
  expect(error.diagnostic.frames.at(-1)!.range).toEqual(def.child("nodes").child("__b").range!);
});

test("a flow input starting with __ is refused at its key", async () => {
  const text =
    "id: f\nname: f\ninput:\n  __x: !str\n" +
    `nodes:\n  a: {${ECHO}, input: {topic: x}}\noutput: \${a.output}\n`;
  const root = SourceDocument.parse(text).root;
  await expectRefused(text, reserved("flow input", "__x"), root.child("input").keyRangeOf("__x"));
});

test("a compact flow input starting with __ is refused at its key", async () => {
  const text =
    "id: f\nname: f\nkind: code\ncode: ./fixtures/compose_codefns.ts:echo\n" +
    "input:\n  __x: !str\noutput: str\n";
  const root = SourceDocument.parse(text).root;
  await expectRefused(
    text,
    `compact (single-node) flow: ${reserved("input", "__x")}`,
    root.child("input").keyRangeOf("__x"),
  );
});

test("a node input starting with __ is refused at its key", async () => {
  const text = `id: f\nname: f\nnodes:\n  a: {${ECHO}, input: {topic: x, __y: z}}\noutput: \${a.output}\n`;
  const input = SourceDocument.parse(text).root.child("nodes").child("a").child("input");
  await expectRefused(text, reserved("node 'a' input", "__y"), input.keyRangeOf("__y"));
});

test("an agent variable starting with __ is refused at its key", async () => {
  const text =
    "id: f\nname: f\nnodes:\n  a:\n    kind: agent\n    prompt: hi\n    output: str\n" +
    "    variables:\n      __v: !str hello\noutput: ${a.output}\n";
  const variables = SourceDocument.parse(text).root.child("nodes").child("a").child("variables");
  await expectRefused(text, reserved("variable", "__v"), variables.keyRangeOf("__v"));
});

test("a flow output named like an assert's generated name is refused at its key", async () => {
  // yian names the END input an assert on node `a` reads `__assert_a`.
  const text =
    `id: f\nname: f\nnodes:\n  a: {${ECHO}, input: {topic: x}}\n` +
    "asserts:\n  - ${a.output == 'x'}\n" +
    "output:\n  __assert_a: ${a.output}\n  b: ${a.output}\n";
  const root = SourceDocument.parse(text).root;
  await expectRefused(text, reserved("flow output", "__assert_a"), root.child("output").keyRangeOf("__assert_a"));
});

test("the names yian generates still load", async () => {
  // A case's `on:` input, an assert's END input, an inline call's node and a memory grant's key.
  const text =
    "id: f\nname: f\ninput:\n  cmd: !str\n  lessons: !FileMemory\nnodes:\n" +
    "  route:\n    kind: case\n    on: ${input.cmd}\n    cases:\n      - when: a\n        then: ask\n" +
    "    else: ask\n" +
    "  ask:\n    kind: agent\n    prompt: hi\n    output: str\n    tools: [lessons]\n" +
    `  echo: {${ECHO}, input: {topic: "\${upper(input.cmd)}"}}\n` +
    "asserts:\n  - ${ask.output != ''}\n" +
    "output:\n  a: ${ask.output}\n  e: ${echo.output}\n";
  await loadConformanceFlow(text);
});
