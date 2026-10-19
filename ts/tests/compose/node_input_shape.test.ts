/**
 * A node's `input:` must be a mapping; any other value is a located load error naming the kind's
 * binding noun and showing a fix: the parse half of `tests/compose/test_node_input_shape.py`
 * (rows 1-4 and 6), through each stand-in's template `parse` on the node's cursor with `input:`
 * renamed to `inputs` as the parse phase does. A text value is spelled by `valueRepr` (`"abc"`
 * and `true` where Python shows `'abc'` and `True`).
 *
 * Dropped: the date row of `test_non_mapping_input_is_a_load_error` (TS reads an unquoted date
 * as text).
 *
 * Later (T3-22): `test_non_mapping_input_in_a_defs_body_is_located` (the `defs entry` prefix)
 * and the load half of `test_null_or_empty_mapping_input_still_loads`. Later (T3-16):
 * `test_listed_carried_fields_load_and_run`. Later (T3-17):
 * `test_inline_loop_whole_reference_keeps_its_message`.
 */

import { expect, test } from "bun:test";

import type { NodeDescriptor } from "../../src/compose/nodes/base.ts";
import type { DescriptorClass } from "../../src/compose/nodes/index.ts";
import { SourceDocument } from "../../src/compose/source/document.ts";
import { loadError } from "./declarations.ts";
import { StandInAgent, StandInCode, standIn } from "./stand_ins.ts";

const STAND_INS: ReadonlyMap<string, DescriptorClass> = new Map([
  ["loop", standIn("loop", { allow: ["times", "call", "inputs"] })],
  ["call", standIn("call", { allow: ["call", "inputs"] })],
  ["map", standIn("map", { allow: ["over", "call", "inputs"] })],
  ["code", StandInCode],
  ["tool", standIn("tool", { allow: ["tool_id", "inputs"] })],
  ["agent", StandInAgent],
  ["human_input", standIn("human_input", { allow: ["prompt", "inputs"] })],
]);

/** Parse node `nodeId` of `text`'s `nodes:` with the stand-in its `kind:` names. */
function parseNode(text: string, nodeId: string): NodeDescriptor {
  const cursor = SourceDocument.parse(text).root.child("nodes").child(nodeId);
  const cls = STAND_INS.get(cursor.get("kind") as string)!;
  return cls.parse(nodeId, cursor.renamed(new Map([["input", "inputs"]])));
}

/** A flow whose LOOP `repeat` has `input: <inputText>`. */
function loopFlow(inputText: string): string {
  return `nodes:\n  repeat:\n    kind: loop\n    times: 2\n    call: step\n    input: ${inputText}\n`;
}

/** The message of the load error parsing `repeat` in `loopFlow(inputText)` throws. */
function loopMessage(inputText: string): string {
  return loadError(() => parseNode(loopFlow(inputText), "repeat")).diagnostic.message;
}

test("a LOOP seeded with a whole record is a located load error", () => {
  const text = loopFlow("${seed.output}");
  const error = loadError(() => parseNode(text, "repeat"));
  expect(error.diagnostic.message).toBe(
    "node 'repeat' (kind=loop): `input:` must be a mapping of carried fields " +
      "(got the reference '${seed.output}'); list each carried field, such as " +
      "`<field>: ${seed.output.<field>}`",
  );
  const line = text.split("\n").indexOf("    input: ${seed.output}") + 1;
  expect(error.diagnostic.frames.at(-1)!.range.startLine).toBe(line);
});

const CHILD = ["the child's arguments", "bind each argument"];
const PARAMETERS = ["parameter names to bindings", "bind each parameter"];

test.each([
  ["call", "call: step", CHILD],
  ["map", "over: [1, 2]\n    call: step", CHILD],
  ["code", "code: return 1", PARAMETERS],
  ["tool", "tool_id: http_get", PARAMETERS],
  ["agent", "prompt: hi", PARAMETERS],
  ["human_input", "prompt: hi", PARAMETERS],
] as const)("a whole-reference input is a load error for %s", (kind, fields, [noun, verb]) => {
  const text = `nodes:\n  under_test:\n    kind: ${kind}\n    ${fields}\n    input: \${seed.output}\n`;
  expect(loadError(() => parseNode(text, "under_test")).diagnostic.message).toBe(
    `node 'under_test' (kind=${kind}): \`input:\` must be a mapping of ${noun} ` +
      `(got the reference '\${seed.output}'); ${verb}, such as ` +
      "`<field>: ${seed.output.<field>}`",
  );
});

const CARRIED = "node 'repeat' (kind=loop): `input:` must be a mapping of carried fields";

test.each([
  [
    "a nested reference",
    "${seed.output.n}",
    "(got the reference '${seed.output.n}')",
    "`<field>: ${seed.output.n.<field>}`",
  ],
  [
    "an expression",
    "${seed.output.n + 1}",
    "(got the expression '${seed.output.n + 1}')",
    "`<field>: ${...}`",
  ],
  ["an embedded span", "x ${seed.output}", '(got "x ${seed.output}")', "`<field>: ${...}`"],
])("%s gets its example", (_label, inputText, got, example) => {
  expect(loopMessage(inputText)).toBe(
    `${CARRIED} ${got}; list each carried field, such as ${example}`,
  );
});

const CARRIED_TAIL = "list each carried field, such as `<field>: ...`";

test.each([
  ["a number", "5", `(got 5); ${CARRIED_TAIL}`],
  ["a list", "[1]", `(got [1]); ${CARRIED_TAIL}`],
  ["text", "abc", `(got "abc"); ${CARRIED_TAIL}`],
  ["a boolean", "true", `(got true); ${CARRIED_TAIL}`],
  ["an empty list", "[]", "(got []); write `input: {}` or remove `input:`"],
  ["empty text", "''", '(got ""); write `input: {}` or remove `input:`'],
  [
    "a tagged reference",
    "!object ${seed.output}",
    "(got the reference '${seed.output}'); list each carried field, such as " +
      "`<field>: ${seed.output.<field>}`",
  ],
])("%s as input is a load error", (_label, inputText, tail) => {
  expect(loopMessage(inputText)).toBe(`${CARRIED} ${tail}`);
});

test.each([
  ["null", ""],
  ["an empty mapping", " {}"],
])("a CODE node with %s input still parses", (_label, inputText) => {
  const text = `nodes:\n  c:\n    kind: code\n    code: return 1\n    input:${inputText}\n`;
  expect(parseNode(text, "c").inputs).toEqual({});
});
