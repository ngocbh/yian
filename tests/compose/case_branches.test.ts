/**
 * A CASE routes only: a branch names a node or calls a flow, and a CASE has no `output:`.
 *
 * An untagged unquoted word names a node and an unquoted whole-value `call(...)` calls a flow.
 * Every other branch is a value: a tagged value (even a word naming a node), quoted text, `null` or
 * an empty `else:`, a number, a bool, a mapping, a list, or unquoted text holding `${`. A value
 * branch is a located load error that shows the value entry to write and the branch naming it, and
 * each suggested entry loads when applied. The entry keeps the branch's own tag, else takes the
 * CASE's former `output:` type when a tag spells it and the value fits it (a lone reference fits
 * when its type does), else the type of a lone reference or a tag guessed from the YAML value, else
 * a `!<type>` placeholder. A value no entry can hold (a value containing itself, a written-out
 * record with an empty key, a nested `call(...)`, a flow call or malformed span in `${...}`, a span
 * reading the case) is its own located error. An unquoted word that names no node suggests the
 * nearest node and the value entry for the text, which loads too. A CASE `output:`, tagged or not,
 * is a load error at the `output:` that says to tag the result where it is read.
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
// The value entry and branch a value-branch message suggests.
const REWRITE = /name it: `(?<entry>[^`]+)`(?: \(.*\))? and `(?<label>then|else): (?<id>\w+)`$/;
// The text entry a branch that names no node, or is broken call text, suggests.
const TEXT_ENTRY = /^case node 'route' (?<label>then|else): .* name it: `(?<entry>(?<id>\w+): [^`]+)`$/;

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
    "    output: str\n" +
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

/**
 * Return the `on` flow with the value entry and branch an error suggests applied.
 *
 * The CASE keeps `when: a` and `reply` for its other branch, and has no `output:`. A text entry's
 * branch names the entry. `typeText` fills a `!<type>` placeholder.
 */
function rewritten(
  error: LoadError,
  options: { head?: string; inputs?: string; extraNodes?: string; typeText?: string } = {},
): string {
  const { head = "", inputs = INPUTS, extraNodes = "", typeText } = options;
  const found = REWRITE.exec(error.message) ?? TEXT_ENTRY.exec(error.message);
  expect(found).not.toBeNull();
  const groups = found!.groups!;
  let entry = groups.entry!;
  if (typeText !== undefined) entry = entry.replace("!<type>", `!${typeText}`);
  const caseBody = groups.label === "then" ? on(groups.id!) : on("reply", groups.id!);
  return flow(caseBody, { head, inputs, extraNodes: `${extraNodes}  ${entry}\n` });
}

/** Assemble `text`, which must load. */
function assembles(text: string): void {
  assembleText(text);
}

/** Load and run `text` and return the flow's output, expecting it succeeded; `cmd: a` takes the `then:`. */
async function taken(text: string, cmd = "a", inputs: Record<string, unknown> = {}): Promise<unknown> {
  const result = await runFlow(await loadConformanceFlow(text), { cmd, ...inputs }, { env });
  expect(result.status, result.failure?.message).toBe("succeeded");
  return result.output;
}

/** Return the message for a value branch of `route`. */
function valueBranch(label: string, entry: string, shown: string, note = ""): string {
  return (
    `case node 'route' ${label}: a branch names a node or calls a flow; put the value ` +
    `in its own entry in \`nodes:\` and name it: \`${entry}: ${shown}\`${note} and ` +
    `\`${label}: ${entry}\``
  );
}

// ---------- node and call branches ----------

test("case output over node branches reads the taken branch", async () => {
  const text = flow(on("other"), {
    extraNodes: "  other:\n" + "    kind: code\n" + "    code: |\n" + "      return 'other'\n" + "    output: str\n",
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

test.each(["__value_0", "__case_x"])("former reserved prefixes name authored nodes: %s", async (nodeId) => {
  const text = flow(on(nodeId), {
    extraNodes: `  ${nodeId}:\n` + "    kind: code\n" + "    code: |\n" + "      return 'own'\n" + "    output: str\n",
  });

  assembles(text);
  expect(await taken(text)).toBe("own");
});

// ---------- value branches ----------

test.each<[string, string, string, unknown]>([
  ['"approve"', "!str approve", 'then: "approve"', "approve"],
  ["'approve'", "!str approve", "then: 'approve'", "approve"],
  ['"reply"', "!str reply", 'then: "reply"', "reply"],
  ['"it\'s: here"', '!str "it\'s: here"', "then: \"it's", "it's: here"],
  ['"[x"', '!str "[x"', 'then: "[x"', "[x"],
  ["|\n          two\n          lines", '!str "two\\nlines\\n"', "then: |", "two\nlines\n"],
  [">\n          one\n          line", '!str "one line\\n"', "then: >", "one line\n"],
  ["3", "!int 3", "then: 3", 3n],
  ["1e3", "!float 1e3", "then: 1e3", 1000],
  ["true", "!bool true", "then: true", true],
  ["null", "!Optional(str) null", "then: null", null],
  [
    "\n          reply: ${input.cmd}\n          exited: true",
    '!object {reply: "${input.cmd}", exited: true}',
    "reply: ${input.cmd}",
    { reply: "a", exited: true },
  ],
  ["{a: 1, b: [2.5]}", "!object {a: 1, b: [2.5]}", "then: {a: 1", { a: 1n, b: [2.5] }],
  // The unquoted date is text, shown plain.
  ["{a: 2026-01-02}", "!object {a: 2026-01-02}", "then: {a: 2026", { a: "2026-01-02" }],
  ['[{"": x}]', '!list(object) [{"": x}]', 'then: [{"": x}]', [{ "": "x" }]],
  ['[a, "${input.cmd}"]', '!list(str) [a, "${input.cmd}"]', "then: [a", ["a", "a"]],
  // The unquoted date is text, so the list is a `list(str)`.
  ["[2026-01-02]", "!list(str) [2026-01-02]", "then: [2026", ["2026-01-02"]],
  ["[[1, 2], [3]]", "!list(list(int)) [[1, 2], [3]]", "then: [[1", [[1n, 2n], [3n]]],
  ["[null, 1]", "!list(Optional(int)) [null, 1]", "then: [null", [null, 1n]],
  ["[[1], [null]]", "!list(list(Optional(int))) [[1], [null]]", "then: [[1]", [[1n], [null]]],
  ["[[], [1]]", "!list(list(int)) [[], [1]]", "then: [[]", [[], [1n]]],
  // The whole number stays an int in the float list (decision 38).
  ["[1, 2.5]", "!list(float) [1, 2.5]", "then: [1, 2.5]", [1n, 2.5]],
  ["[{a: 1}, {b: 2}]", "!list(object) [{a: 1}, {b: 2}]", "then: [{a", [{ a: 1n }, { b: 2n }]],
  ["hi ${input.cmd}", "!str hi ${input.cmd}", "then: hi", "hi a"],
  ["${input.cmd}", "!str ${input.cmd}", "then: ${input.cmd}", "a"],
  ['"call(take"', '!str "call(take"', 'then: "call(take"', "call(take"],
])("a then value shows its value entry and the entry loads: %j", async (then, shown, needle, value) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, needle));
  expect(await taken(rewritten(error))).toStrictEqual(value);
});

test("an unquoted date then is a word that names no node, and its text entry loads", async () => {
  // The unquoted date is text.
  const text = flow(on("2026-01-02"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `2026-01-02` names no node. For the text, add a value entry and " +
      "name it: `route_then_0: !str 2026-01-02`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: 2026-01-02"));
  expect(await taken(rewritten(error))).toBe("2026-01-02");
});

test.each<[string, string, string, unknown]>([
  ['"declined"', "!str declined", 'else: "declined"', "declined"],
  ["null", "!Optional(str) null", "else: null", null],
  ["", "!Optional(str) null", "else:", null],
  ["1.5", "!float 1.5", "else: 1.5", 1.5],
])("an else value shows its value entry and the entry loads: %j", async (else_, shown, needle, value) => {
  const text = flow(on("reply", else_));

  const error = loadErrorWith(text, valueBranch("else", "route_else", shown));

  expect(errorLine(error)).toBe(lineOf(text, needle));
  expect(await taken(rewritten(error), "b")).toStrictEqual(value);
});

test("a value in a later then names its index", () => {
  const text = flow(
    "    on: ${input.cmd}\n" +
      "    cases:\n" +
      "      - when: a\n" +
      "        then: reply\n" +
      "      - when: b\n" +
      '        then: "bee"\n' +
      "    else: reply\n",
  );

  const error = loadErrorWith(text, valueBranch("then", "route_then_1", "!str bee"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "bee"'));
});

test("a long mapping is cut and written as a block", () => {
  const text = flow(
    on(
      "\n          reply: ${input.cmd}\n          exited: true\n" +
        "          note: a much longer note that goes on and on",
    ),
  );

  const error = loadErrorWith(
    text,
    valueBranch(
      "then",
      "route_then_0",
      '!object {reply: "${input.cmd}", exited: true, note: a much longer no…',
      " (write it as a block)",
    ),
  );

  expect(errorLine(error)).toBe(lineOf(text, "reply: ${input.cmd}"));
});

test("a cut value keeps an astral character at the cut whole", () => {
  const word = "x".repeat(59) + "😀" + "yz";
  const text = flow(on(`"${word}"`));

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", `!str ${"x".repeat(59)}😀…`, " (write it as a block)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: "));
});

test("a suggested entry id skips ids already taken", async () => {
  const takenIds = "  route_then_0: !str one\n" + "  route_then_0_2: !str two\n";
  const text = flow(on('"approve"'), { extraNodes: takenIds });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0_3", "!str approve"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "approve"'));
  expect(await taken(rewritten(error, { extraNodes: takenIds }))).toBe("approve");
});

test("the old typed text is tagged with its type", async () => {
  const text = flow(on("int = ${input.n}"), { inputs: INPUTS_N });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!int ${input.n}"));

  expect(errorLine(error)).toBe(lineOf(text, "then: int ="));
  expect(await taken(rewritten(error, { inputs: INPUTS_N }), "a", { n: 3n })).toBe(3n);
});

test.each([
  ["${input.n}", "!int ${input.n}", 3n],
  ["${count.output}", "!int ${count.output}", 7n],
])("a lone reference is tagged with its type: %s", async (then, shown, value) => {
  const count = "  count: !int 7\n";
  const text = flow(on(then), { inputs: INPUTS_N, extraNodes: count });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  expect(await taken(rewritten(error, { inputs: INPUTS_N, extraNodes: count }), "a", { n: 3n })).toBe(value);
});

test.each<[string, string, string, unknown]>([
  ["${input.n}", "Optional(int)", "!Optional(int) ${input.n}", 3n],
  ["${input.cmd}", "int", "!str ${input.cmd}", "a"],
  ["${input.n}", "str", "!int ${input.n}", 3n],
  ["${count.output}", "str", "!int ${count.output}", 7n],
])(
  "a lone reference takes the former output type only when it fits: %s under %s",
  async (then, output, shown, value) => {
    const count = "  count: !int 7\n";
    const text = flow(on(then, "reply", { output: `    output: ${output}\n` }), {
      inputs: INPUTS_N,
      extraNodes: count,
    });

    const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

    expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
    expect(await taken(rewritten(error, { inputs: INPUTS_N, extraNodes: count }), "a", { n: 3n })).toBe(value);
  },
);

test("a lone span of unknown type asks for its type", async () => {
  const text = flow(on("${input.n + 1}"), { inputs: INPUTS_N });

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", "!<type> ${input.n + 1}", " (name its type for `<type>`)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: ${input.n + 1}"));
  expect(await taken(rewritten(error, { inputs: INPUTS_N, typeText: "int" }), "a", { n: 3n })).toBe(4n);
});

test.each<[string, string, unknown]>([
  ["[1, a]", "!list(Any) [1, a]", [1n, "a"]],
  ["[[1], [a]]", "!list(Any) [[1], [a]]", [[1n], ["a"]]],
  ["[{a: 1}, 2]", "!list(Any) [{a: 1}, 2]", [{ a: 1n }, 2n]],
  ["[null, 1, a]", "!list(Optional(Any)) [null, 1, a]", [null, 1n, "a"]],
  // Plain YAML reads the date back as the same text, so it is shown unquoted.
  ['[1, "2024-01-01"]', "!list(Any) [1, 2024-01-01]", [1n, "2024-01-01"]],
])("a list of mixed items is tagged list of Any: %s", async (then, shown, value) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  expect(await taken(rewritten(error))).toStrictEqual(value);
});

test.each<[string, unknown]>([
  ["[1, 2024-01-01]", [1n, "2024-01-01"]],
  ["[{a: 2024-01-01}, 2]", [{ a: "2024-01-01" }, 2n]],
])("a list of mixed items with a date is shown with the date as written: %s", async (then, value) => {
  // The unquoted date is text, so it reads back plain.
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", `!list(Any) ${then}`));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  expect(await taken(rewritten(error))).toStrictEqual(value);
});

test.each(["&R [*R]", "&R [1, *R]", "&R {a: *R}"])("a value that contains itself is rejected: %s", (then) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, "case node 'route' then: a branch value can't contain itself");

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
});

test.each([
  [".inf", "!float .inf"],
  ["-.inf", "!float -.inf"],
  ["{a: .inf}", "!object {a: .inf}"],
  ["[1.5, -.inf]", "!list(float) [1.5, -.inf]"],
  ["!float .inf", "!float .inf"],
])("an infinite number is a value branch: %s", (then, shown) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
  assembles(rewritten(error));
});

test("a NaN branch is refused by the YAML reader at the scalar", () => {
  const text = flow(on(".nan"));

  const error = loadErrorWith(text, "invalid YAML: NaN is not a yian value; quote it to keep it as text");

  expect(errorLine(error)).toBe(lineOf(text, "then: .nan"));
});

test.each<[string, string, unknown]>([
  ["!int 3", "!int 3", 3n],
  ["!float 1e3", "!float 1e3", 1000],
  ["!str approve", "!str approve", "approve"],
  // Plain YAML reads `2026-01-02` back as the same text, so it is shown unquoted.
  ['!date "2026-01-02"', "!date 2026-01-02", "2026-01-02"],
  ["!int ${input.n}", "!int ${input.n}", 3n],
  ["!Optional(int) null", "!Optional(int) null", null],
  ["!Buy {side: buy}", "!Buy {side: buy}", { side: "buy" }],
  ["!list(int)\n          - 1\n          - 2", "!list(int) [1, 2]", [1n, 2n]],
  ["!str reply", "!str reply", "reply"],
])("a tagged value keeps its tag: %j", async (then, shown, value) => {
  const head = "typedefs:\n  Buy:\n    side: str\n";
  // The branch's own tag wins over the former `output:` type, and a tagged word that names a node
  // is text, as in a value entry.
  const text = flow(on(then, "reply", { output: "    output: str\n" }), { head, inputs: INPUTS_N });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then.split("\n")[0]}`));
  expect(await taken(rewritten(error, { head, inputs: INPUTS_N }), "a", { n: 3n })).toStrictEqual(value);
});

test("a value branch is tagged with the former output type", async () => {
  const text = flow(on("3", "reply", { output: "    output: str\n" }));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", '!str "3"'));

  // The value branch is reported before the CASE's `output:`.
  expect(errorLine(error)).toBe(lineOf(text, "then: 3"));
  expect(await taken(rewritten(error))).toBe("3");
});

test("a value branch is spelled to fit the former output type", async () => {
  const text = flow(on('"yes"', "reply", { output: "    output: bool\n" }));

  const error = loadErrorWith(
    text,
    valueBranch("then", "route_then_0", "!bool true", " (`yes` is text; write `true`)"),
  );

  expect(errorLine(error)).toBe(lineOf(text, 'then: "yes"'));
  expect(await taken(rewritten(error))).toBe(true);
});

test.each<[string, string, string, unknown]>([
  ['"3"', "int", '!str "3"', "3"],
  ['"v"', "int", "!str v", "v"],
  ["null", "str", "!Optional(str) null", null],
  ["{a: 1}", "Reply", "!object {a: 1}", { a: 1n }],
  ['"approve"', "Strr", "!str approve", "approve"],
])(
  "a value that does not fit the former output type guesses its tag: %s under %s",
  async (then, output, shown, value) => {
    const head = "typedefs:\n  Reply:\n    reply: str\n";
    const text = flow(on(then, "reply", { output: `    output: ${output}\n` }), { head });

    const error = loadErrorWith(text, valueBranch("then", "route_then_0", shown));

    expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
    expect(await taken(rewritten(error, { head }))).toStrictEqual(value);
  },
);

test("a value branch is tagged with the former output typedef", async () => {
  const head = "typedefs:\n  Reply:\n    reply: str\n    exited: bool\n";
  const text = flow(on("\n          reply: ok\n          exited: true", "reply", { output: "    output: Reply\n" }), {
    head,
  });

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!Reply {reply: ok, exited: true}"));

  expect(errorLine(error)).toBe(lineOf(text, "reply: ok"));
  expect(await taken(rewritten(error, { head }))).toStrictEqual({ reply: "ok", exited: true });
});

test("a value branch under an unspellable output guesses its tag", async () => {
  const text = flow(on('"approve"', "reply", { output: "    output: Union(int, str)\n" }));

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", "!str approve"));

  expect(errorLine(error)).toBe(lineOf(text, 'then: "approve"'));
  expect(await taken(rewritten(error))).toBe("approve");
});

test.each([
  ['{"": x}', 'then: {"": x}'],
  ['{a: {"": 1}}', "then: {a:"],
  ['!object {"": x}', "then: !object"],
])("a mapping with an empty key is rejected: %s", (then, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a record key can't be empty; a written-out `!object` " +
      "keeps its keys as record fields",
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
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

test.each([
  ["{a: call(child, x=1)}", "then: {a: call"],
  ['[a, "call(child)"]', "then: [a"],
])("a call nested in a value is rejected: %s", (then, needle) => {
  const text = flow(on(then));

  const error = loadErrorWith(
    text,
    "case node 'route' then: a `call(...)` nested in a value is not supported; write " +
      "the call as the branch (`then: call(...)`) or in a node, or reword the text",
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

test.each(["${route.output}", '{a: "got ${route.output}"}'])("a value reading its own case is rejected: %s", (then) => {
  const text = flow(on(then));

  const error = loadErrorWith(text, "case node 'route' then: the value reads 'route' itself");

  expect(errorLine(error)).toBe(lineOf(text, `then: ${then}`));
});

test("a malformed span in a value is rejected", () => {
  const text = flow(on('{a: "${input.cmd"}'));

  const error = loadErrorWith(text, "case node 'route' then: unbalanced '${' in template '${input.cmd'");

  expect(errorLine(error)).toBe(lineOf(text, "then: {a:"));
});

test.each(
  ["token", "access_token"].flatMap((key) => [
    [key, `then: {${key}: abc}`, "then:"],
    [key, `then:\n          ${key}: abc`, `${key}: abc`],
  ]),
)("a mapping value branch gets the branch error under any key: %s %j", (key, branch, needle) => {
  const text = flow("    on: ${input.cmd}\n" + "    cases:\n" + "      - when: a\n" + `        ${branch}\n` + "    else: reply\n");

  const error = loadErrorWith(text, valueBranch("then", "route_then_0", `!object {${key}: abc}`));

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- unquoted words that name no node ----------

test("an unknown then word suggests the nearest node and its text entry", async () => {
  const text = flow(on("replyy"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `replyy` names no node. Did you mean `reply`? For the " +
      "text, add a value entry and name it: `replyy_text: !str replyy`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: replyy"));
  expect(await taken(rewritten(error))).toBe("replyy");
});

test("an unknown else word suggests the nearest node and its text entry", async () => {
  const text = flow(on("reply", "replyy"));

  const error = loadErrorWith(
    text,
    "case node 'route' else: `replyy` names no node. Did you mean `reply`? For the " +
      "text, add a value entry and name it: `replyy_text: !str replyy`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "else: replyy"));
  expect(await taken(rewritten(error), "b")).toBe("replyy");
});

test("an unknown sentence gets no suggestion and an entry named after the branch", async () => {
  const text = flow(on("The user declined"));

  const error = loadErrorWith(
    text,
    "case node 'route' then: `The user declined` names no node. For the text, add a " +
      "value entry and name it: `route_then_0: !str The user declined`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: The user declined"));
  expect(await taken(rewritten(error))).toBe("The user declined");
});

test.each(["", "    output: bool\n"])("an unknown yes gets the same text entry: %j", async (output) => {
  const text = flow(on("reply", "yes", { output }));

  const error = loadErrorWith(
    text,
    "case node 'route' else: `yes` names no node. For the text, add a value entry " +
      "and name it: `yes_text: !str yes`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "else: yes"));
  expect(await taken(rewritten(error), "b")).toBe("yes");
});

test("an unknown word entry id skips an id already taken", async () => {
  const takenIds = "  nope_text: !str taken\n";
  const text = flow(on("nope"), { extraNodes: takenIds });

  const error = loadErrorWith(
    text,
    "case node 'route' then: `nope` names no node. Did you mean `nope_text`? For the " +
      "text, add a value entry and name it: `nope_text_2: !str nope`",
  );

  expect(errorLine(error)).toBe(lineOf(text, "then: nope"));
  expect(await taken(rewritten(error, { extraNodes: takenIds }))).toBe("nope");
});

test.each([
  [
    "call(child, x=1) | x",
    "reply",
    "case node 'route' then: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| x'; a call(...) must be the whole " +
      "value. For the text, add a value entry and name it: " +
      '`route_then_0: !str "call(child, x=1) | x"`',
    "then: call(",
    "a",
    "call(child, x=1) | x",
  ],
  [
    "reply",
    "call(child, x=1) | y",
    "case node 'route' else: a branch target must be a node id or a single " +
      "call: text follows the closing `)`: '| y'; a call(...) must be the whole " +
      "value. For the text, add a value entry and name it: " +
      '`route_else: !str "call(child, x=1) | y"`',
    "else: call(",
    "b",
    "call(child, x=1) | y",
  ],
])("unquoted call that is not one directive is rejected: %s / %s", async (then, else_, message, needle, cmd, value) => {
  const text = flow(on(then, else_));

  const error = loadErrorWith(text, message);

  expect(errorLine(error)).toBe(lineOf(text, needle));
  expect(await taken(rewritten(error), cmd)).toBe(value);
});

// ---------- a CASE `output:` ----------

test.each([
  ["output: str", ", e.g. `x: !str ${route.output}`", "output: str"],
  ["output: !str", ", e.g. `x: !str ${route.output}`", "output: !str"],
  ["output: !int", ", e.g. `x: !int ${route.output}`", "output: !int"],
  ["output: Optional(str)", ", e.g. `x: !Optional(str) ${route.output}`", "output:"],
  ["output: Union(int, str)", "; name `Union(int, str)` in `typedefs:` and tag with its name", "output: Union"],
  ["output: Strr", ", e.g. `x: !Strr ${route.output}`", "output: Strr"],
  ["output:", ", e.g. `x: !<type> ${route.output}`", "output:"],
  ["output:\n      a: int", "; name `{a: int}` in `typedefs:` and tag with its name", "output:"],
  ["output:\n      a: Strr", "; name `{a: Strr}` in `typedefs:` and tag with its name", "output:"],
  ["output:\n      a: .inf", "; name `{a: .inf}` in `typedefs:` and tag with its name", "output:"],
  ["output: &a [*a]", ", e.g. `x: !<type> ${route.output}`", "output: &a"],
  ["output: &a {x: *a}", ", e.g. `x: !<type> ${route.output}`", "output: &a"],
])("a case output is rejected at its line: %j", (output, example, needle) => {
  const text = flow(on("reply", "reply", { output: `    ${output}\n` }));

  const error = loadErrorWith(
    text,
    "case node 'route': a `case` routes and has no `output:`; tag its result where " + `you read it${example}`,
  );

  expect(errorLine(error)).toBe(lineOf(text, needle));
});

// ---------- harvested conformance cases ----------

test.each<CaseRow>(
  [
    "a_block_mapping_value_branch",
    "a_block_text_value_branch_is_shown_escaped",
    "a_bool_value_branch",
    "a_branch_tag_wins_over_the_former_output_type",
    "a_call_nested_in_a_value_is_rejected",
    "a_case_output_union_asks_for_a_typedef_name",
    "a_float_else_value",
    "a_list_value_branch",
    "a_lone_reference_is_tagged_with_its_type",
    "a_lone_span_of_unknown_type_asks_for_its_type",
    "a_long_mapping_is_cut_and_written_as_a_block",
    "a_malformed_span_in_a_value_is_rejected",
    "a_mapping_with_an_empty_key_is_rejected",
    "a_quoted_call_holding_a_span_is_its_own_error",
    "a_quoted_text_then_is_a_value_branch",
    "a_text_value_branch_keeps_the_quotes_it_needs",
    "a_value_branch_is_spelled_to_fit_the_former_output_type",
    "a_value_branch_is_tagged_with_the_former_output_typedef",
    "a_value_reading_its_own_case_is_rejected",
    "a_value_that_contains_itself_is_rejected",
    "an_empty_else_is_a_null_value",
    "an_unknown_then_word_suggests_the_nearest_node",
    "an_unknown_yes_gets_a_text_entry",
    "an_unquoted_call_with_trailing_text_is_rejected",
    "text_that_reads_as_a_number_is_shown_quoted",
  ].map((name) => ({ id: `load-case/compose_case_branches__${name}`, call: assembled })),
)("assemble: $id", runCase);
