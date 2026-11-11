/**
 * A CASE routes only: a branch names a node or calls a flow, and a CASE has no `output:`.
 *
 * An untagged unquoted word names a node and an unquoted whole-value `call(...)` calls a flow.
 * Every other branch is a value: a tagged value (even a word naming a node), quoted text, `null` or
 * an empty `else:`, a number, a bool, a mapping, a list, or unquoted text holding `${`. A value
 * branch is a located load error that says to put the value in an entry in `nodes:`. An unquoted
 * word that names no node suggests the nearest node. A CASE `output:`, tagged or not, is a load
 * error at the `output:` that says to tag the result where it is read.
 */

import { afterAll, afterEach, expect, test } from "bun:test";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { LoadError } from "../../src/compose/helpers/errors.ts";
import { forget } from "../../src/defaults.ts";
import { runFlow } from "../../src/run.ts";
import { hostEnv } from "../helpers/mcp_helpers.ts";
import { assembled, assembleText, type CaseRow, loadConformanceFlow, loadError, runCase } from "../helpers/declarations.ts";

const dir = mkdtempSync(join(tmpdir(), "yian-case-branches-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const env = hostEnv(dir);

afterEach(() => forget());

const INPUTS = "  cmd: !str\n";
const INPUTS_N = "  cmd: !str\n  n: !int\n";

/** Return the 1-based line number of the first line containing `needle`. */
function lineOf(text: string, needle: string): number {
  return text.split("\n").findIndex((line) => line.includes(needle)) + 1;
}

/** Load `text`, expecting a load error whose whole message is `message`. */
function loadErrorWith(text: string, message: string): LoadError {
  const error = loadError(() => assembleText(text));
  expect(error.message).toBe(message);
  return error;
}

/** Return the start line of the innermost frame of a load error. */
function errorLine(error: LoadError): number {
  return error.diagnostic.frames.at(-1)!.range.startLine;
}

/** Return a flow with a `route` case (indented body) and a `reply` node it reads. */
function flow(
  caseBody: string,
  options: { head?: string; inputs?: string; extraNodes?: string } = {},
): string {
  const { head = "", inputs = INPUTS, extraNodes = "" } = options;
  return (
    "id: f\n" +
    "name: f\n" +
    head +
    "input:\n" +
    inputs +
    "nodes:\n" +
    "  route:\n" +
    "    kind: case\n" +
    caseBody +
    "  reply:\n" +
    "    kind: code\n" +
    "    code: ./fixtures/compose_codefns.ts:echo\n" +
    "    input:\n" +
    "      topic: replied\n" +
    "    output: !str\n" +
    extraNodes +
    "output: ${route.output}\n"
  );
}

/** Return an `on: ${input.cmd}` case body with one `when: a` branch and an `else:`. */
function on(then: string, else_ = "reply", options: { output?: string } = {}): string {
  return (
    "    on: ${input.cmd}\n" +
    (options.output ?? "") +
    "    cases:\n" +
    "      - when: a\n" +
    `        then: ${then}\n` +
    `    else: ${else_}\n`
  );
}

/** Return the message for a value branch of `route`. */
function valueBranch(label: string): string {
  return (
    `case node 'route' ${label}: a branch names a node or calls a flow; put the value in an entry ` +
    "in `nodes:` and name that entry"
  );
}

/** Return the message for an unquoted word of `route` that names no node. */
function namesNoNode(label: string, word: string, nearest: string | null = null): string {
  const suggestion = nearest === null ? "" : ` Did you mean \`${nearest}\`?`;
  return (
    `case node 'route' ${label}: \`${word}\` names no node.${suggestion} For a value, put it in an ` +
    "entry in `nodes:` and name that entry"
  );
}

// ---------- node and call branches ----------

test("case output over node branches reads the taken branch", async () => {
  const text = flow(on("other"), {
    extraNodes: "  other:\n" + "    kind: code\n" + "    code: |\n" + "      return 'other'\n" + "    output: !str\n",
  });
  const loaded = await loadConformanceFlow(text);

  expect((await runFlow(loaded, { cmd: "a" }, { env })).output).toBe("other");
  expect((await runFlow(loaded, { cmd: "b" }, { env })).output).toBe("replied");
});

test("case output over a value entry branch reads the taken branch", async () => {
  const text = flow(on("verdict"), { extraNodes: '  verdict: !date "2026-01-02"\n' });
  const loaded = await loadConformanceFlow(text);

  expect((await runFlow(loaded, { cmd: "a" }, { env })).output).toBe("2026-01-02");
  expect((await runFlow(loaded, { cmd: "b" }, { env })).output).toBe("replied");
});

// ---------- value branches ----------

test.each<[string, string]>([
  ['"approve"', 'then: "approve"'],
  ["'approve'", "then: 'approve'"],
  ['"reply"', 'then: "reply"'],
  ["|\n          two\n          lines", "then: |"],
  [">\n          one\n          line", "then: >"],
  ["3", "then: 3"],
  ["1e3", "then: 1e3"],
  [".inf", "then: .inf"],
  ["true", "then: true"],
  ["null", "then: null"],
  ["\n          reply: ${input.cmd}\n          exited: true", "reply: ${input.cmd}"],
  ["{a: 1, b: [2.5]}", "then: {a: 1"],
  ['{"": x}', 'then: {"": x}'],
  ["{a: call(child, x=1)}", "then: {a: call"],
  ["[1, a]", "then: [1, a]"],
  ["&R [*R]", "then: &R"],
  ["hi ${input.cmd}", "then: hi"],
  ["${input.n}", "then: ${input.n}"],
  ["${route.output}", "then: ${route.output}"],
  ['{a: "${input.cmd"}', "then: {a:"],
  ['${ take(stance="pro") }', "then: ${ take"],
  ["int = ${input.n}", "then: int ="],
  ['"call(take"', 'then: "call(take"'],
  // A tag makes the branch a value, even a word naming a node, as in a value entry.
  ["!int 3", "then: !int 3"],
  ["!str reply", "then: !str reply"],
  ["!list(int)\n          - 1\n          - 2", "then: !list(int)"],
])("a then value is a located branch error: %j", (then, needle) => {
  const text = flow(on(then), { inputs: INPUTS_N });

  const error = loadErrorWith(text, valueBranch("then"));

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test.each<[string, string]>([
  ['"declined"', 'else: "declined"'],
  ["null", "else: null"],
  ["", "else:"],
  ["1.5", "else: 1.5"],
])("an else value is a located branch error: %j", (else_, needle) => {
  const text = flow(on("reply", else_));

  const error = loadErrorWith(text, valueBranch("else"));

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test("a value in a later then is located at its branch", () => {
  const text = flow(
    "    on: ${input.cmd}\n" +
      "    cases:\n" +
      "      - when: a\n" +
      "        then: reply\n" +
      "      - when: b\n" +
      '        then: "bee"\n' +
      "    else: reply\n",
  );

  const error = loadErrorWith(text, valueBranch("then"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "bee"'));
});

test("a value branch is reported before the CASE's output:", () => {
  const text = flow(on("3", "reply", { output: "    output: !str\n" }));

  const error = loadErrorWith(text, valueBranch("then"));

  expect(errorLine(error)).toBe(lineOf(text, "then: 3"));
});

test("a NaN branch is refused by the YAML reader at the scalar", () => {
  const text = flow(on(".nan"));

  const error = loadErrorWith(text, "invalid YAML: NaN is not a yian value; quote it to keep it as text");

  expect(errorLine(error)).toBe(lineOf(text, "then: .nan"));
});

test("a quoted call holding a span is its own error", () => {
  const text = flow(on('"call(take, x=${input.cmd})"'));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a quoted `call(...)` is text, so it can't hold `${...}` " +
      "here; drop the quotes to call the flow, or reword the text",
  );

  expect(errorLine(error)).toBe(lineOf(text, 'then: "call(take'));
});

// ---------- unquoted words that name no node ----------

test("an unknown then word suggests the nearest node", () => {
  const text = flow(on("replyy"));

  const error = loadErrorWith(text, namesNoNode("then", "replyy", "reply"));

  expect(errorLine(error)).toBe(lineOf(text, "then: replyy"));
});

test("an unknown else word suggests the nearest node", () => {
  const text = flow(on("reply", "replyy"));

  const error = loadErrorWith(text, namesNoNode("else", "replyy", "reply"));

  expect(errorLine(error)).toBe(lineOf(text, "else: replyy"));
});

test.each([
  ["The user declined", "then: The user declined"],
  // The unquoted date is text.
  ["2026-01-02", "then: 2026-01-02"],
])("an unknown word with no near node gets no suggestion: %s", (then, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, namesNoNode("then", then));

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test.each([
  [
    "call(child, x=1) | x",
    "reply",
    "case node 'route' then: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| x'; a call(...) must be the whole value",
    "then: call(",
  ],
  [
    "reply",
    "call(child, x=1) | y",
    "case node 'route' else: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| y'; a call(...) must be the whole value",
    "else: call(",
  ],
])("unquoted call that is not one directive is rejected: %s / %s", (then, else_, message, needle) => {
  const text = flow(on(then, else_));

  const error = loadErrorWith(text, message);

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- a CASE `output:` ----------

test.each([
  ["output: str", "output: str"],
  ["output: !str", "output: !str"],
  ["output: Optional(str)", "output:"],
  ["output: Union(int, str)", "output: Union"],
  ["output:", "output:"],
  ["output:\n      a: !int", "output:"],
  ["output: &a [*a]", "output: &a"],
])("a case output is rejected at its line: %j", (output, needle) => {
  const text = flow(on("reply", "reply", { output: `    ${output}\n` }));

  const error = loadErrorWith(
    text,
    "case node 'route': a `case` routes and has no `output:`; tag its result where you read it, " +
      "e.g. `x: !<type> ${route.output}`",
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_block_mapping_value_branch",
    "a_bool_value_branch",
    "a_float_else_value",
    "a_list_value_branch",
    "a_quoted_call_holding_a_span_is_its_own_error",
    "a_quoted_text_then_is_a_value_branch",
    "an_empty_else_is_a_null_value",
    "an_unknown_then_word_suggests_the_nearest_node",
    "an_unquoted_call_with_trailing_text_is_rejected",
  ].map((name) => ({ id: `load-case/compose_case_branches__${name}`, call: assembled })),
)("assemble: $id", runCase);
